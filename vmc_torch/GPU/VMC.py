import os
import time
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from vmc_torch.GPU.optimizer import (
    OptimizerGPU,
    PreconditionerGPU,
)
from vmc_torch.GPU.sampler import SamplerGPU
# from vmc_torch.GPU.vmc_modules import (
#     distributed_minres_solver_gpu,
#     minSR_solver_gpu,
# )
from vmc_torch.GPU.vmc_utils import (
    compute_grads_gpu,
    evaluate_energy,
)


def _find_free_port():
    """Find a free TCP port on localhost."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_distributed(cuda_rank: Optional[int] = None, cpu: bool = False):
    if "RANK" not in os.environ:
        print("Warning: Not using torchrun. Single device.")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        if cuda_rank is None:
            os.environ["LOCAL_RANK"] = "0"
        else:
            os.environ["LOCAL_RANK"] = str(cuda_rank)

    dist.init_process_group(
        backend="nccl", init_method="env://",
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    if not cpu:
        # torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def print_sampling_settings(
    rank, world_size, batch_size, ns_per_rank,
    grad_batch_size,
):
    if rank == 0:
        total_ns_expected = ns_per_rank * world_size
        n_sweeps = int(np.ceil(ns_per_rank / batch_size))
        print(
            f"B={batch_size}, Ns_per_rank={ns_per_rank}, "
            f"sweeps/rank={n_sweeps}, "
            f"Total_Ns~{total_ns_expected}, "
            f"grad_batch={grad_batch_size}"
        )


@dataclass
class VMCWarmupConfig:
    use_export_compile: bool = False
    grad_batch_size: int = 64
    verbose: bool = True
    run_sampling: bool = True
    run_locE: bool = True
    run_grad: bool = True
    use_log_amp: bool = False
    def __init__(self, **kwargs):
          for f in fields(self):
              setattr(self, f.name, kwargs.pop(f.name, f.default))
          for k, v in kwargs.items():
              setattr(self, k, v)


@dataclass
class VMCLoopConfig:
    vmc_steps: int
    ns_per_rank: int
    grad_batch_size: int
    n_params: int
    nsites: int
    learning_rate: float = 0.1
    diag_shift: float = 1e-4
    burn_in_steps: int = 0
    run_sr: bool = True
    use_export_compile: bool = False
    show_progress: bool = True
    step_offset: int = 0  # global step = local step + step_offset
    debug: bool = False  # print [dbg] diagnostics each step
    use_log_amp: bool = False  # log-amplitude mode
    # log_psi_grad outlier clipping: drop samples whose per-sample
    # |log_psi_grad| norm exceeds clip_factor * median norms.
    # Set to 0 to disable.  Typical value: 5-10.
    outlier_clip_factor: float = 100.0
    # Optional LR scheduler: callable(step) -> float.
    # If set, overrides learning_rate each step.
    lr_scheduler: object = None
    verbose: bool = False
    
    def __init__(self, **kwargs):
          for f in fields(self):
              setattr(self, f.name, kwargs.pop(f.name, f.default))
          for k, v in kwargs.items():
              setattr(self, k, v)

    @classmethod
    def from_vmc_config(cls, cfg, *, n_params, nsites):
        """Build from a run-script VMCConfig dataclass.

        Copies all fields whose names match between cfg and
        VMCLoopConfig.  The two extra fields (n_params, nsites)
        are model/system-dependent so must be passed explicitly.
        ``resume_step`` in cfg maps to ``step_offset`` here.
        ``sr_diag_shift`` in cfg maps to ``diag_shift`` here.
        """
        import dataclasses as _dc
        loop_fields = {f.name for f in _dc.fields(cls)}
        cfg_fields = {f.name for f in _dc.fields(cfg)}
        kwargs = {}
        for f in _dc.fields(cfg):
            kwargs[f.name] = getattr(cfg, f.name)
        # Pass through extra (non-dataclass-field) attributes
        for k, v in vars(cfg).items():
            if k not in cfg_fields and k not in kwargs:
                kwargs[k] = v
        # Map resume_step -> step_offset
        if hasattr(cfg, 'resume_step'):
            kwargs['step_offset'] = cfg.resume_step
        # Map sr_diag_shift -> diag_shift
        if 'sr_diag_shift' in kwargs:
            kwargs['diag_shift'] = kwargs.pop('sr_diag_shift')
        kwargs['n_params'] = n_params
        kwargs['nsites'] = nsites
        return cls(**kwargs)


class VMC_GPU:
    """GPU VMC driver.

    The sampler handles MCMC only (proposing moves,
    accepting/rejecting). This driver orchestrates the
    full sample -> energy -> gradient loop, SR solve,
    and parameter update.
    """

    def __init__(
        self,
        sampler: SamplerGPU,
        preconditioner: Optional[PreconditionerGPU] = None,
        optimizer: Optional[OptimizerGPU] = None,
        evaluate_energy_fn: Callable = evaluate_energy,
        compute_grads_fn: Callable = compute_grads_gpu,
    ):
        self.sampler = sampler
        self.preconditioner = preconditioner
        self.optimizer = optimizer
        self.evaluate_energy_fn = evaluate_energy_fn
        self.compute_grads_fn = compute_grads_fn

    # ----------------------------------------------------------
    # Sampling phase: MCMC + energy + gradients
    # ----------------------------------------------------------

    def _run_sampling_phase(
        self,
        fxs,
        model,
        hamiltonian,
        graph,
        ns_per_rank,
        grad_batch_size,
        burn_in=False,
        burn_in_steps=0,
        use_export_compile=False,
        debug=False,
        outlier_clip_factor=0.0,
        offload_lpg_loc_cpu=False,
        use_log_amp=False,
        verbose=False,
    ):
        """Run MCMC sampling, energy eval, and gradient
        computation for one VMC step.

        The sampler only does MCMC (step / burn_in). This
        method calls evaluate_energy_fn and
        compute_grads_fn directly.

        Args:
            offload_lpg_loc_cpu: if True, move log_psi_grad chunks
                to CPU immediately after GPU computation.
            use_log_amp: if True, work in log-amplitude
                space throughout (sampler, energy, grads).

        Returns:
            (local_energies, local_log_psi_grad): tensors
                of shapes (Ns,) and (Ns, Np).
            fxs: (B, N_sites) updated walker configs.
            phase_times: dict with t_samp, t_locE, t_grad.
        """
        B = fxs.shape[0]
        t_samp, t_locE, t_grad = 0.0, 0.0, 0.0

        # Burn-in
        if burn_in:
            t0 = time.time()
            fxs = self.sampler.burn_in(
                fxs, model, graph, burn_in_steps,
                compile=use_export_compile,
                use_log_amp=use_log_amp,
            )
            print(f'Burn-in: {burn_in_steps} steps, T_b = {time.time()-t0}')

        local_energies_list = []
        local_lpg_list = []
        current_count = 0

        while current_count < ns_per_rank:
            needed = min(B, ns_per_rank - current_count)
            with torch.inference_mode():
                # 1. MCMC sweep
                t0 = time.time()
                fxs, amps_out = self.sampler.step(
                    fxs, model, graph,
                    compile=use_export_compile,
                    use_log_amp=use_log_amp,
                    verbose=verbose,
                )
                t_samp += time.time() - t0

                # 2. Local energy
                t0 = time.time()
                energy_result = self.evaluate_energy_fn(
                    fxs, model, hamiltonian, amps_out,
                    use_log_amp=use_log_amp,
                    verbose=verbose,
                    return_bMPS=True,
                )
                if len(energy_result) == 4:
                    _, local_E, bMPS_x, bMPS_y = energy_result
                elif len(energy_result) == 3:
                    _, local_E, bMPS_x = energy_result
                else:
                    _, local_E = energy_result
                    bMPS_x = None
                t_locE += time.time() - t0

            # Free sampling/energy tensors so allocator
            # can reuse their blocks for grad computation
            del amps_out

            # 3. Gradients -> log_psi_grad
            t0 = time.time()
            with torch.enable_grad():
                grads, grads_aux = self.compute_grads_fn(
                    fxs, model,
                    vectorize=True,
                    batch_size=grad_batch_size,
                    vmap_grad=True,
                    offload_to_cpu=offload_lpg_loc_cpu,
                    use_log_amp=use_log_amp,
                    verbose=verbose,
                    bMPS_params_x=bMPS_x,
                )
            _rank = dist.get_rank() if dist.is_initialized() else 0

            if use_log_amp:
                # grads is already d(log|psi|)/d(params)
                # No division needed.
                if debug and _rank == 0:
                    g_rms = (
                        torch.norm(grads).item()
                        / grads.numel() ** 0.5
                    )
                    print(
                        f"  [dbg] log_psi_grad: "
                        f"rms={g_rms:.4e}, "
                        f"max={grads.abs().max().item():.4e}"
                    )
            else:
                # grads_aux is raw amps — divide to get
                # lpg_loc = d(psi)/d(params) / psi
                amps2 = grads_aux
                if debug and _rank == 0:
                    abs_a = amps2.abs()
                    print(
                        f"  [dbg] amps: "
                        f"min={abs_a.min().item():.4e}"
                        f", median={abs_a.median().item():.4e}"
                        f", mean={abs_a.mean().item():.4e}"
                        f", max={abs_a.max().item():.4e}"
                    )
                    g_rms = (
                        torch.norm(grads).item()
                        / grads.numel() ** 0.5
                    )
                    print(
                        f"  [dbg] raw grads: "
                        f"rms={g_rms:.4e}, "
                        f"max={grads.abs().max().item():.4e}"
                    )
                grads /= amps2.unsqueeze(1)
                if debug and _rank == 0:
                    print(
                        f"  [dbg] log_psi_grad: rms="
                        f"{torch.norm(grads).item() / grads.numel()**0.5:.4e}"
                        f", max={grads.abs().max().item():.4e}"
                    )
            t_grad += time.time() - t0

            local_energies_list.append(
                local_E[:needed].detach(),
            )
            lpg_chunk = grads[:needed].detach()
            local_lpg_list.append(lpg_chunk)
            current_count += needed
            del grads, grads_aux, local_E, lpg_chunk

        local_energies = torch.cat(local_energies_list)
        local_log_psi_grad = torch.cat(local_lpg_list)

        # --- Outlier masking ---
        # Drop samples whose per-sample log_psi_grad norm
        # exceeds clip_factor * median.
        if outlier_clip_factor > 0:
            lpg_norms = local_log_psi_grad.norm(dim=1)
            median_norm = lpg_norms.median()
            threshold = outlier_clip_factor * median_norm
            mask = lpg_norms <= threshold  # True = keep
            n_dropped = int((~mask).sum().item())
            if n_dropped > 0:
                _rank = (
                    dist.get_rank()
                    if dist.is_initialized() else 0
                )
                if _rank == 0:
                    print(
                        f"  [outlier] dropped {n_dropped}"
                        f"/{local_log_psi_grad.shape[0]} "
                        f"samples "
                        f"(|lpg| > {threshold.item():.2e},"
                        f" median={median_norm.item():.2e})"
                    )
                local_log_psi_grad[~mask] = 0.0
                local_energies[~mask] = local_energies[
                    mask
                ].mean()

        phase_times = {
            't_samp': t_samp,
            't_locE': t_locE,
            't_grad': t_grad,
        }
        sample_time = t_samp + t_locE + t_grad
        return (
            (local_energies, local_log_psi_grad),
            fxs,
            sample_time,
            phase_times,
        )

    # ----------------------------------------------------------
    # Parameter sync
    # ----------------------------------------------------------

    @staticmethod
    def _sync_params(model):
        """Broadcast model params from rank 0 to all ranks."""
        if not dist.is_initialized():
            return
        if dist.get_world_size() <= 1:
            return
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    # ----------------------------------------------------------
    # Warmup
    # ----------------------------------------------------------

    def run_warmup(
        self,
        fxs,
        model,
        graph,
        hamiltonian,
        rank,
        config: VMCWarmupConfig,
    ):
        run_sampling = config.run_sampling
        run_locE = config.run_locE
        run_grad = config.run_grad
        use_log_amp = config.use_log_amp

        # Offload gradients to CPU when MINRES SR solver can work with
        # CPU-resident data (scipy MINRES).
        offload_grad_cpu = (                                                                                                                                                                                                                                    
            hasattr(self.preconditioner, 'use_scipy')                                                                                                                                                                                                         
            and self.preconditioner.use_scipy
        ) or getattr(config, 'offload_grad_to_cpu', False)

        self._sync_params(model)

        if rank == 0 and config.verbose:
            print("\n--- Warmup (1 sweep) ---")
        t_warm = time.time()

        with torch.inference_mode():
            if run_sampling:
                fxs, amps_out = self.sampler.step(
                    fxs, model, graph,
                    compile=config.use_export_compile,
                    verbose=config.verbose,
                    use_log_amp=use_log_amp,
                )
                if rank == 0 and config.verbose:
                    print(
                        f"  sample_next:     "
                        f"{time.time() - t_warm:.2f}s"
                    )
                t1 = time.time()
            if run_locE:
                _, evals = self.evaluate_energy_fn(
                    fxs, model, hamiltonian, amps_out,
                    verbose=config.verbose,
                    use_log_amp=use_log_amp,
                )
                if rank == 0 and config.verbose:
                    print(
                        f"  evaluate_energy: "
                        f"{time.time() - t1:.2f}s"
                    )
        # Free inference-phase tensors before grad computation
        try:
            del amps_out, evals
            torch.cuda.empty_cache()
        except Exception:
            pass

        if not run_grad:
            return fxs
        t2 = time.time()
        with torch.enable_grad():
            grads, grads_aux = self.compute_grads_fn(
                fxs, model,
                vectorize=True,
                batch_size=config.grad_batch_size,
                vmap_grad=True,
                offload_to_cpu=offload_grad_cpu,
                verbose=config.verbose,
                use_log_amp=use_log_amp,
            )
        if rank == 0 and config.verbose:
            print(
                f"  compute_grads:   "
                f"{time.time() - t2:.2f}s"
            )
            print(
                f"  Warmup total:    "
                f"{time.time() - t_warm:.2f}s"
            )
            # analyze grad stats
            g_rms = (
                torch.norm(grads).item()                / grads.numel() ** 0.5
            )
            print(
                f"  [dbg] log_psi_grad: "
                f"rms={g_rms:.4e}, "
                f"max={grads.abs().max().item():.4e}"
            )

        del grads, grads_aux
        torch.cuda.empty_cache()
        return fxs

    # ----------------------------------------------------------
    # Global energy statistics
    # ----------------------------------------------------------

    def compute_global_energy_stats(
        self, local_energies, world_size,
    ):
        n_local = local_energies.shape[0]
        total_ns = n_local * world_size

        local_e_sum = local_energies.sum()
        dist.all_reduce(
            local_e_sum, op=dist.ReduceOp.SUM,
        )
        energy_mean = local_e_sum.item() / total_ns

        local_e_sq_sum = (local_energies ** 2).sum()
        dist.all_reduce(
            local_e_sq_sum, op=dist.ReduceOp.SUM,
        )
        energy_var = (
            local_e_sq_sum.item() / total_ns
            - energy_mean ** 2
        )

        return total_ns, energy_mean, energy_var

    # ----------------------------------------------------------
    # SR solver step
    # ----------------------------------------------------------

    def solve_sr_step(
        self,
        local_o,
        local_energies,
        energy_mean,
        total_samples,
        n_params,
        diag_shift,
        device,
        run_sr,
    ):
        if self.preconditioner is not None:
            return self.preconditioner.solve(
                local_o=local_o,
                local_energies=local_energies,
                energy_mean=energy_mean,
                total_samples=total_samples,
                n_params=n_params,
                diag_shift=diag_shift,
                device=device,
                run_sr=run_sr,
            )
        else:
            # No preconditioner — raw energy gradient for SGD.
            # Stays on GPU: no CPU numpy conversion.
            t0 = time.time()

            if not isinstance(local_o, torch.Tensor):
                local_o = torch.tensor(
                    local_o, device=device,
                    dtype=torch.float64,
                )
            if not isinstance(local_energies, torch.Tensor):
                local_energies = torch.tensor(
                    local_energies, device=device,
                    dtype=torch.float64,
                )

            n_local = local_energies.shape[0]
            world_size = (
                dist.get_world_size()
                if dist.is_initialized() else 1
            )

            if n_local > 0:
                local_sum_lpg = local_o.sum(dim=0)
                local_sum_EO = local_energies @ local_o
            else:
                local_sum_lpg = torch.zeros(
                    n_params, device=device,
                    dtype=torch.float64,
                )
                local_sum_EO = torch.zeros(
                    n_params, device=device,
                    dtype=torch.float64,
                )

            if world_size > 1:
                dist.all_reduce(
                    local_sum_lpg,
                    op=dist.ReduceOp.SUM,
                )
                dist.all_reduce(
                    local_sum_EO,
                    op=dist.ReduceOp.SUM,
                )

            mean_lpg = local_sum_lpg / total_samples
            mean_EO = local_sum_EO / total_samples
            energy_grad = (
                mean_EO - energy_mean * mean_lpg
            )
            return energy_grad, time.time() - t0, None

    # ----------------------------------------------------------
    # Parameter update
    # ----------------------------------------------------------

    def apply_parameter_update(
        self, model, dp, learning_rate, device,
    ):
        if self.optimizer is not None:
            self.optimizer.step(
                model=model,
                direction=dp,
                device=device,
                learning_rate=learning_rate,
            )
            return

        with torch.no_grad():
            dp_tensor = torch.as_tensor(
                dp, device=device, dtype=torch.float64,
            )
            current_params_vec = (
                torch.nn.utils.parameters_to_vector(
                    model.parameters(),
                )
            )
            new_params_vec = (
                current_params_vec
                - learning_rate * dp_tensor
            )
            torch.nn.utils.vector_to_parameters(
                new_params_vec, model.parameters(),
            )

    # ----------------------------------------------------------
    # Main VMC loop
    # ----------------------------------------------------------

    def run_vmc_loop(
        self,
        fxs,
        model,
        hamiltonian,
        graph,
        rank,
        world_size,
        config: VMCLoopConfig,
        on_step_end: Optional[
            Callable[[Dict[str, Any]], None]
        ] = None,
    ):
        device = next(model.parameters()).device
        self._sync_params(model)

        # Offload lpg_loc to CPU when MINRES SR solver can work with
        # CPU-resident data (scipy MINRES).
        offload_lpg_loc_cpu = (
            hasattr(self.preconditioner, 'use_scipy')
            and self.preconditioner.use_scipy
        ) or getattr(config, 'offload_grad_to_cpu', False)

        if rank == 0 and config.show_progress:
            print(f"\n--- VMC ({config.vmc_steps} steps) ---")
            vmc_pbar = tqdm(
                total=config.vmc_steps, desc="VMC Steps",
            )
        else:
            vmc_pbar = None

        energy_history = []
        for step in range(config.vmc_steps):
            t0 = time.time()

            (local_energies, local_lpg), fxs, sample_time, phase_times = (
                self._run_sampling_phase(
                    fxs=fxs,
                    model=model,
                    hamiltonian=hamiltonian,
                    graph=graph,
                    ns_per_rank=config.ns_per_rank,
                    grad_batch_size=config.grad_batch_size,
                    burn_in=(step == 0),
                    burn_in_steps=config.burn_in_steps,
                    use_export_compile=config.use_export_compile,
                    debug=config.debug,
                    outlier_clip_factor=config.outlier_clip_factor,
                    offload_lpg_loc_cpu=offload_lpg_loc_cpu,
                    use_log_amp=config.use_log_amp,
                    verbose=config.verbose,
                )
            )

            total_ns, energy_mean, energy_var = (
                self.compute_global_energy_stats(
                    local_energies, world_size,
                )
            )

            # --- Diagnostics on SR inputs ---
            if rank == 0:
                n_nan_E = torch.isnan(
                    local_energies
                ).sum().item()
                n_nan_lpg = torch.isnan(
                    local_lpg
                ).sum().item()
                n_inf_lpg = torch.isinf(
                    local_lpg
                ).sum().item()
                if n_nan_E or n_nan_lpg or n_inf_lpg:
                    print(
                        f"[WARNING] SR inputs: "
                        f"local_E has {n_nan_E} NaN, "
                        f"log_psi_grad has "
                        f"{n_nan_lpg} NaN / "
                        f"{n_inf_lpg} Inf"
                    )
                if config.debug:
                    Np = local_lpg.shape[1]
                    pv0 = torch.nn.utils.parameters_to_vector(
                        model.parameters(),
                    )
                    print(
                        f"  [dbg] params: Np={Np}, "
                        f"rms="
                        f"{torch.norm(pv0).item()/Np**0.5:.4e}, "
                        f"max={pv0.abs().max().item():.4e}"
                    )
                    print(
                        f"  [dbg] log_psi_grad: rms="
                        f"{torch.norm(local_lpg).item()/(local_lpg.numel()**0.5):.4e}, "
                        f"max={local_lpg.abs().max().item():.4e}"
                    )
            torch.cuda.empty_cache()
            # SR solve
            dp, t_sr, info = self.solve_sr_step(
                local_o=local_lpg,
                local_energies=local_energies,
                energy_mean=energy_mean,
                total_samples=total_ns,
                n_params=config.n_params,
                diag_shift=config.diag_shift,
                device=device,
                run_sr=config.run_sr,
            )

            # Free CPU-resident lpg before next step's grad alloc
            del local_lpg, local_energies

            # --- NaN/Inf check on SR gradients ---
            pv0 = torch.nn.utils.parameters_to_vector(
                model.parameters(),
            )
            max_pv0 = pv0.abs().max().item()
            dp_t = torch.as_tensor(dp, device=device)
            max_dp_t = dp_t.abs().max().item()
            dp_nan = torch.isnan(dp_t).sum().item()
            dp_inf = torch.isinf(dp_t).sum().item()
            if (dp_nan > 0 or dp_inf > 0) and rank == 0:
                print(
                    f"[WARNING] SR dp has {dp_nan} NaN, "
                    f"{dp_inf} Inf out of {dp_t.numel()}"
                )
            
            # Apply parameter update
            if config.lr_scheduler is not None:
                global_step = step + config.step_offset
                config.learning_rate = config.lr_scheduler(
                    global_step,
                )

            self.apply_parameter_update(
                model, dp, config.learning_rate, device,
            )

            # End of step diagnostics and logging
            step_time = time.time() - t0

            e_per_site = energy_mean / config.nsites
            err = (
                np.sqrt(max(energy_var, 0.0) / total_ns)
                / config.nsites
            )
            energy_history.append(e_per_site)

            global_step = step + config.step_offset
            if rank == 0 and config.show_progress:
                t_s = phase_times.get('t_samp', 0.0)
                t_e = phase_times.get('t_locE', 0.0)
                t_g = phase_times.get('t_grad', 0.0)
                lr_str = (
                    f" lr={config.learning_rate:.2e}"
                    if config.lr_scheduler is not None else ""
                )
                print(
                    f"Step {global_step:3d} | E/site: "
                    f"{e_per_site:.6f} "
                    f"+/- {err:.6f} | N={total_ns}{lr_str} | "
                    f"T_samp={t_s:.1f}s T_locE={t_e:.1f}s "
                    f"T_grad={t_g:.1f}s T_SR={t_sr:.2f}s "
                    f"T_total={step_time:.1f}s | "
                    f"SR_info={info} "
                    f"max_p={max_pv0:.4e} max_dp={max_dp_t:.4e}"
                )
                vmc_pbar.update(1)

            # Gather walker configs from all ranks
            # for checkpoint saving (Ns, N_sites)
            if world_size > 1:
                _fxs_list = [
                    torch.zeros_like(fxs)
                    for _ in range(world_size)
                ]
                dist.all_gather(
                    _fxs_list, fxs.contiguous(),
                )
                all_fxs = torch.cat(_fxs_list, dim=0)
            else:
                all_fxs = fxs

            if on_step_end is not None:
                on_step_end(
                    {
                        "step": global_step,
                        "energy_mean": energy_mean,
                        "energy_var": energy_var,
                        "energy_per_site": e_per_site,
                        "error_per_site": err,
                        "total_samples": total_ns,
                        "sample_time": sample_time,
                        "phase_times": phase_times,
                        "sr_time": t_sr,
                        "total_time": step_time,
                        "solver_info": info,
                        "fxs": all_fxs,
                    }
                )

        if vmc_pbar is not None:
            vmc_pbar.close()
        return energy_history, fxs


__all__ = [
    "setup_distributed",
    "print_sampling_settings",
    "VMCWarmupConfig",
    "VMCLoopConfig",
    "VMC_GPU",
]
