import os
import time
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
from vmc_torch.GPU.vmc_utils import (
    compute_grads_gpu,
    evaluate_energy,
    evaluate_energy_grad,
    check_NaN_or_inf,
)
from vmc_torch.GPU.swo_utils import (
    accumulate_fidelity_terms,
    accumulate_supervised_sr_terms,
    collect_swo_dataset,
    compute_swo_direction,
    fidelity_from_model_on_configs,
    fidelity_stats_from_log_amps,
    save_swo_checkpoint,
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
        mixed_precision: bool = False,
    ):
        self.sampler = sampler
        self.preconditioner = preconditioner
        self.optimizer = optimizer
        self.evaluate_energy_fn = evaluate_energy_fn
        self.compute_grads_fn = compute_grads_fn
        self.mixed_precision = mixed_precision

    # ==========================================================
    # Distributed utilities
    # ==========================================================

    @staticmethod
    def _sync_params(model):
        """Broadcast model params from rank 0 to all ranks."""
        if not dist.is_initialized():
            return
        if dist.get_world_size() <= 1:
            return
        for p in model.parameters():
            if not p.data.is_contiguous():
                p.data = p.data.contiguous()
            dist.broadcast(p.data, src=0)

    @staticmethod
    def _allreduce_grads(model, world_size):
        """Average param.grad across all ranks."""
        if world_size <= 1:
            return
        for p in model.parameters():
            if p.grad is not None:
                if not p.grad.is_contiguous():
                    p.grad = p.grad.contiguous()
                dist.all_reduce(
                    p.grad, op=dist.ReduceOp.SUM,
                )
                p.grad /= world_size

    @staticmethod
    def _count_nan_inf_tensor(tensor, max_check_elems=10_000_000):
        """Count NaN/Inf without materializing full-size bool masks."""
        if tensor.numel() == 0:
            return 0, 0

        if tensor.dim() >= 2:
            rows_per_check = max(
                1,
                max_check_elems // max(tensor.shape[1], 1),
            )
            n_nan = 0
            n_inf = 0
            for start in range(0, tensor.shape[0], rows_per_check):
                stop = min(start + rows_per_check, tensor.shape[0])
                chunk = tensor[start:stop]
                n_nan += torch.isnan(chunk).sum().item()
                n_inf += torch.isinf(chunk).sum().item()
            return n_nan, n_inf

        n_nan = 0
        n_inf = 0
        for start in range(0, tensor.shape[0], max_check_elems):
            stop = min(start + max_check_elems, tensor.shape[0])
            chunk = tensor[start:stop]
            n_nan += torch.isnan(chunk).sum().item()
            n_inf += torch.isinf(chunk).sum().item()
        return n_nan, n_inf

    @staticmethod
    def _copy_vector_to_model(model, vec) -> None:
        """Copy a flat parameter vector into ``model`` in place."""
        with torch.no_grad():
            offset = 0
            for p in model.parameters():
                n = p.numel()
                p.data.copy_(vec[offset:offset + n].view_as(p.data))
                offset += n

    def _swo_line_search_step(
        self,
        *,
        configs,
        sign_B,
        log_abs_B,
        training_model,
        direction,
        old_log_f,
        grad_batch_size,
        n_total,
        max_backtracks,
        shrink,
        min_lr,
        rank,
    ):
        """Backtracking line search on the fixed SWO sample set."""
        current = torch.nn.utils.parameters_to_vector(
            training_model.parameters(),
        ).detach().clone()
        base_lr = float(self.optimizer.lr)
        best_log_f = old_log_f

        for n_backtrack in range(max_backtracks + 1):
            lr = base_lr * (shrink ** n_backtrack)
            if lr < min_lr:
                break

            self.optimizer.step(
                training_model,
                direction,
                learning_rate=lr,
            )
            _, new_log_f = fidelity_from_model_on_configs(
                configs,
                training_model,
                sign_B,
                log_abs_B,
                n_total,
                grad_batch_size,
            )
            if float(new_log_f) <= float(old_log_f):
                self.optimizer.lr = lr
                return {
                    'accepted': True,
                    'lr': lr,
                    'n_backtrack': n_backtrack,
                    'log_f': new_log_f,
                }

            self._copy_vector_to_model(training_model, current)

        if rank == 0:
            print(
                "[SWO line search] rejected update: no fixed-sample "
                f"decrease from -logF={float(old_log_f):.6e}"
            )
        return {
            'accepted': False,
            'lr': 0.0,
            'n_backtrack': max_backtracks + 1,
            'log_f': best_log_f,
        }

    # ==========================================================
    # Warmup
    # ==========================================================

    def run_warmup(
        self,
        fxs,
        model,
        graph,
        hamiltonian,
        rank,
        config,
    ):
        if self.mixed_precision:
            param_dtype = next(model.parameters()).dtype
            assert param_dtype == torch.float32, (
                f"mixed_precision=True requires a float32 model, "
                f"got {param_dtype}"
            )
        run_sampling = config.run_sampling
        run_locE = config.run_locE
        run_grad = config.run_grad
        use_log_amp = config.use_log_amp

        # Only rank 0 prints — silence verbose=True on other ranks
        # for all downstream calls.
        verbose = config.verbose if rank == 0 else False

        # Offload gradients to CPU when MINRES SR solver can work with
        # CPU-resident data (scipy MINRES).
        offload_grad_cpu = (
            hasattr(self.preconditioner, 'use_scipy')
            and self.preconditioner.use_scipy
        ) or getattr(config, 'offload_grad_to_cpu', False)

        self._sync_params(model)

        if verbose:
            print("\n--- Warmup (1 sweep) ---")
        t_warm = time.time()

        with torch.inference_mode():
            bMPS_x = None
            if run_sampling:
                fxs, amps_out = self.sampler.step(
                    fxs, model, graph,
                    compile=config.use_export_compile,
                    verbose=verbose,
                    use_log_amp=use_log_amp,
                )
                if verbose:
                    print(
                        f"  sample_next:     "
                        f"{time.time() - t_warm:.2f}s"
                    )
                t1 = time.time()
            if run_locE:
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
                if verbose:
                    print(
                        f"  evaluate_energy: "
                        f"{time.time() - t1:.2f}s"
                    )
        # Free inference-phase tensors before grad computation
        try:
            del amps_out, local_E
            torch.cuda.empty_cache()
        except Exception:
            pass

        t2 = time.time()
        if run_grad:
            if self.mixed_precision:
                model.double()
            with torch.enable_grad():
                grads, grads_aux = self.compute_grads_fn(
                    fxs, model,
                    vectorize=True,
                    batch_size=config.grad_batch_size,
                    vmap_grad=True,
                    offload_to_cpu=offload_grad_cpu,
                    verbose=verbose,
                    use_log_amp=use_log_amp,
                    bMPS_params_x=bMPS_x,
                )
            if self.mixed_precision:
                model.float()

        if verbose:
            print(
                f"  compute_grads:   "
                f"{time.time() - t2:.2f}s"
            )
            print(
                f"  Warmup total:    "
                f"{time.time() - t_warm:.2f}s"
            )
            if run_grad:
                print(
                    f"  [dbg] log_psi_grad: "
                    f"max={grads.abs().max().item():.4e}"
                )

        if run_grad:
            del grads, grads_aux
        torch.cuda.empty_cache()
        
        return fxs

    # ==========================================================
    # Core VMC pipeline
    # ==========================================================

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

        local_energies = None
        local_log_psi_grad = None
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
            if self.mixed_precision:
                model.double()  # f32 → f64 for accurate backprop
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

            if self.mixed_precision:
                model.float()  # restore f32 for next sampling sweep

            # bMPS tensors no longer needed after grad computation
            bMPS_x = None
            if 'bMPS_y' in dir():
                bMPS_y = None
            t_grad += time.time() - t0

            if (
                local_log_psi_grad is None
                and current_count == 0
                and needed == ns_per_rank
                and needed == grads.shape[0]
            ):
                local_energies = local_E[:needed].detach()
                local_log_psi_grad = grads[:needed].detach()
                current_count = needed
                del grads, grads_aux, local_E
                continue

            if local_energies is None:
                local_energies = torch.empty(
                    ns_per_rank,
                    dtype=local_E.dtype,
                    device=local_E.device,
                )
            if local_log_psi_grad is None:
                lpg_shape = (ns_per_rank, grads.shape[1])
                if (
                    grads.device.type == 'cpu'
                    and torch.cuda.is_available()
                ):
                    local_log_psi_grad = torch.empty(
                        lpg_shape,
                        dtype=grads.dtype,
                        device=grads.device,
                        pin_memory=True,
                    )
                else:
                    local_log_psi_grad = torch.empty(
                        lpg_shape,
                        dtype=grads.dtype,
                        device=grads.device,
                    )

            count_next = current_count + needed
            local_energies[current_count:count_next].copy_(
                local_E[:needed].detach(),
            )
            local_log_psi_grad[current_count:count_next].copy_(
                grads[:needed].detach(),
            )
            current_count = count_next
            del grads, grads_aux, local_E

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

    def _compute_step_diagnostics(
        self,
        local_energies,
        local_lpg,
        fxs,
        model,
        energy_mean,
        use_log_amp,
        world_size,
        rank,
    ):
        """Collect distributional stats for one VMC step.

        Cheap (one extra forward + a few all-gathers); call only when
        ``config.diagnostics`` is set. Returns a dict (only meaningful
        on rank 0; other ranks get an empty dict).

        Diagnostics:
          - E_loc: min, max, median, std, count of outliers
            (|E_loc - median| > 5 * MAD).
          - log|psi|: min, max, median, std (or, equivalently, the
            spread of amplitudes — large spread signals near-nodal
            walkers and likely E_loc blow-ups).
          - ||g||: 2-norm of the SR right-hand side
            g = <(E_loc - E_mean) * (O_loc - O_mean)>_sample.
            If huge, even a converged MINRES solve produces a huge dp.
        """
        import numpy as np

        # ---- Gather local_energies across ranks ----
        if world_size > 1:
            E_list = [
                torch.empty_like(local_energies)
                for _ in range(world_size)
            ]
            dist.all_gather(E_list, local_energies.contiguous())
            E_all = torch.cat(E_list)
        else:
            E_all = local_energies

        # ---- Compute ||g|| globally
        # g = <(E - <E>) O>_sample = (1/Ns) sum_i (E_i - E_mean) O_i
        # (the O_mean subtraction cancels under SUM-then-divide).
        diff = (local_energies - energy_mean).to(local_lpg.dtype)
        g_local = (
            diff.unsqueeze(1) * local_lpg
        ).sum(dim=0)
        if world_size > 1:
            dist.all_reduce(g_local, op=dist.ReduceOp.SUM)
        Ns_total = local_energies.shape[0] * world_size
        g = g_local / Ns_total
        g_norm = g.norm().item()
        g_max = g.abs().max().item()

        # ---- log|psi| via a forward over local fxs (then gather) ----
        with torch.inference_mode():
            if use_log_amp:
                _, log_abs_local = model.forward_log(fxs)
            else:
                amps = model(fxs)
                log_abs_local = amps.abs().clamp_min(1e-300).log()
        if world_size > 1:
            la_list = [
                torch.empty_like(log_abs_local)
                for _ in range(world_size)
            ]
            dist.all_gather(la_list, log_abs_local.contiguous())
            log_abs_all = torch.cat(la_list)
        else:
            log_abs_all = log_abs_local

        if rank != 0:
            return {}

        E_np = E_all.detach().cpu().numpy().astype(np.float64)
        la_np = log_abs_all.detach().cpu().numpy().astype(np.float64)

        E_median = float(np.median(E_np))
        mad = float(np.median(np.abs(E_np - E_median)))
        n_outlier = int(
            np.sum(np.abs(E_np - E_median) > 5.0 * max(mad, 1e-12))
        )

        return {
            'Ns': int(E_np.size),
            'E_min': float(E_np.min()),
            'E_max': float(E_np.max()),
            'E_median': E_median,
            'E_std': float(E_np.std()),
            'E_MAD': mad,
            'E_outlier_5MAD': n_outlier,
            'logpsi_min': float(la_np.min()),
            'logpsi_max': float(la_np.max()),
            'logpsi_median': float(np.median(la_np)),
            'logpsi_std': float(la_np.std()),
            'logpsi_spread': float(la_np.max() - la_np.min()),
            'g_norm': float(g_norm),
            'g_max': float(g_max),
        }

    def solve_sr_step(
        self,
        O_loc,
        E_loc,
        E_mean,
        Ns,
        Np,
        rshift,
        ashift,
        device,
    ):
        return self.preconditioner.solve(
            O_loc=O_loc,
            E_loc=E_loc,
            E_mean=E_mean,
            Ns=Ns,
            Np=Np,
            rshift=rshift,
            ashift=ashift,
            device=device,
        )

    def apply_parameter_update(
        self, model, dp, learning_rate, device,
    ):
        self.optimizer.step(
            model=model,
            direction=dp,
            device=device,
            learning_rate=learning_rate,
        )
        return


    def run_vmc_loop(
        self,
        fxs,
        model,
        hamiltonian,
        graph,
        rank,
        world_size,
        config,
        n_params: int,
        nsites: int,
        on_step_end: Optional[
            Callable[[Dict[str, Any]], None]
        ] = None,
    ):
        """Drive the VMC sampling/SR-solve/parameter-update loop.

        Args:
            config: a ``VMCConfig``-shaped object (any object with
                the expected attributes works).
            n_params: total trainable parameter count of ``model``.
            nsites: number of lattice sites (for per-site energy
                reporting).
        """
        if self.mixed_precision:
            param_dtype = next(model.parameters()).dtype
            assert param_dtype == torch.float32, (
                f"mixed_precision=True requires a float32 model, "
                f"got {param_dtype}"
            )
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
                n_nan_lpg, n_inf_lpg = (
                    self._count_nan_inf_tensor(local_lpg)
                )
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
                    
            # --- Optional per-step diagnostics (E_loc / log|psi| /
            # SR RHS norm). Independent of `config.debug`. ---
            diag_stats = None
            if config.diagnostics:
                diag_stats = self._compute_step_diagnostics(
                    local_energies=local_energies,
                    local_lpg=local_lpg,
                    fxs=fxs,
                    model=model,
                    energy_mean=energy_mean,
                    use_log_amp=config.use_log_amp,
                    world_size=world_size,
                    rank=rank,
                )

            # SR solve. When the preconditioner supports it, hand the
            # Jacobian over via a 1-element "ownership box" and drop
            # our own reference now, so the (Ns, Np) Jacobian can be
            # freed inside _solve (right after the all_to_all copy)
            # instead of lingering in this frame until step end.
            if getattr(
                self.preconditioner, '_supports_ownership_box', False,
            ):
                _lpg_arg = [local_lpg]
                del local_lpg
            else:
                _lpg_arg = local_lpg

            dp, t_sr, info = self.solve_sr_step(
                O_loc=_lpg_arg,
                E_loc=local_energies,
                E_mean=energy_mean,
                Ns=total_ns,
                Np=n_params,
                rshift=config.sr_rshift,
                ashift=config.sr_ashift,
                device=device,
            )

            # Free lpg/energies before next step's grad alloc
            del _lpg_arg, local_energies
            torch.cuda.empty_cache()

            # NaN/Inf check on SR gradients
            dp_t = torch.as_tensor(dp, device=device)
            if rank == 0:
                check_NaN_or_inf(dp_t, "SR dp")

            # Diagnostic magnitudes for step logging. Reduce per-param
            # (not parameters_to_vector, whose .view(-1) fails on the
            # non-contiguous TN params when train_tn=True).
            max_pv0 = max(
                p.detach().abs().max().item()
                for p in model.parameters()
            )
            max_dp_t = dp_t.abs().max().item()

            # Apply parameter update
            if config.lr_scheduler is not None:
                global_step = step + config.resume_step
                config.learning_rate = config.lr_scheduler(
                    global_step,
                )

            self.apply_parameter_update(
                model, dp, config.learning_rate, device,
            )

            # Paranoid sync
            self._sync_params(model)

            # End of step diagnostics and logging
            step_time = time.time() - t0

            e_per_site = energy_mean / nsites
            err = (
                np.sqrt(max(energy_var, 0.0) / total_ns)
                / nsites
            )
            energy_history.append(e_per_site)

            global_step = step + config.resume_step
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
                
                # Always-printed SR mask report (independent of
                # `diagnostics`): fires whenever rows were masked this
                # step — non-finite (NaN/Inf, always checked) and/or
                # |E|>clip outliers (when local_E_clip is set).
                mask_info = getattr(
                    self.preconditioner, '_last_mask_info', None,
                )
                if rank == 0 and mask_info:
                    n_masked = mask_info['n_masked']
                    n_nf = mask_info.get('n_nonfinite', 0)
                    Ns_total = mask_info['Ns_total']
                    clip = mask_info['clip']
                    clip_str = (
                        f"{clip:.3e}" if clip is not None else "off"
                    )
                    # Non-finite rows signal fp32 forward/backward
                    # instability -> louder WARNING tag.
                    tag = "[SR mask WARNING]" if n_nf else "[SR mask]"
                    print(
                        f"  {tag} n_masked={n_masked}/{Ns_total} "
                        f"({100.0 * n_masked / Ns_total:.2f}%) "
                        f"non-finite={n_nf} clip={clip_str} "
                        f"E_mean_clean={mask_info['E_mean_clean']:+.3e}"
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
                        "diagnostics": diag_stats,
                    }
                )

        if vmc_pbar is not None:
            vmc_pbar.close()
        return energy_history, fxs

    # ==========================================================
    # EXPERIMENTAL: Log-variance optimization (NOT VERIFIED)
    #
    # The pathwise and REINFORCE gradient terms individually
    # pass FD checks, but they nearly perfectly cancel
    # (cos ~ -0.9999), leaving a tiny, noise-dominated
    # residual.  As a result, logvar optimization does NOT
    # converge in practice.  These methods are kept for
    # future investigation but should not be relied upon.
    # See test_scripts/test_logvar_grad.py for details.
    # ==========================================================

    def _run_logvar_sampling_phase(
        self,
        fxs,
        model,
        hamiltonian,
        graph,
        ns_per_rank,
        burn_in=False,
        burn_in_steps=0,
        use_export_compile=False,
        use_log_amp=False,
        verbose=False,
    ):
        """MCMC sampling + energy eval only (no gradient).

        Returns:
            all_fxs: (Ns, N_sites) sampled configs.
            local_energies: (Ns,) local energies.
            fxs: (B, N_sites) updated walker configs.
            phase_times: dict with t_samp, t_locE.
        """
        B = fxs.shape[0]
        t_samp, t_locE = 0.0, 0.0

        if burn_in:
            t0 = time.time()
            fxs = self.sampler.burn_in(
                fxs, model, graph, burn_in_steps,
                compile=use_export_compile,
                use_log_amp=use_log_amp,
            )
            print(
                f'Burn-in: {burn_in_steps} steps, '
                f'T_b = {time.time()-t0}'
            )

        local_energies_list = []
        all_fxs_list = []
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
                    return_bMPS=False,
                )
                if len(energy_result) == 2:
                    _, local_E = energy_result
                else:
                    _, local_E = (
                        energy_result[0],
                        energy_result[1],
                    )
                t_locE += time.time() - t0

            del amps_out

            local_energies_list.append(
                local_E[:needed].detach(),
            )
            # Save configs for surrogate loss forward pass
            all_fxs_list.append(
                fxs[:needed].detach().clone(),
            )
            current_count += needed
            del local_E

        local_energies = torch.cat(local_energies_list)
        all_fxs = torch.cat(all_fxs_list)

        phase_times = {
            't_samp': t_samp,
            't_locE': t_locE,
        }
        return all_fxs, local_energies, fxs, phase_times

    @staticmethod
    def _compute_logvar_loss(
        all_fxs,
        hamiltonian,
        gamma,
        model,
        grad_batch_size,
        use_log_amp=True,
    ):
        """Two-pass log-variance gradient: pathwise + REINFORCE.

        Pass 1 (pathwise): Forward through evaluate_energy_grad
        with grad tracking, compute L = log(Var[E_L] + gamma),
        backprop. This captures dE_L/dtheta contributions.

        Pass 2 (REINFORCE / score function): Using detached E_L
        from pass 1, compute centered surrogate loss that
        estimates Cov(O_k, (E_L - Ebar)^2). The centering
        (-sigma^2 baseline) is essential because our psi is
        unnormalized, so <O_k> != 0.

        Full gradient:
          dL/dtheta = (1/(sigma^2+gamma)) * d(sigma^2)/dtheta
        where:
          d(sigma^2)/dtheta = 2*Cov(O, (E_L-Ebar)^2)  [REINFORCE]
                            + 2*<(E_L-Ebar)*dE_L/dtheta> [pathwise]

        Returns:
            (log_var_loss, energy_mean, energy_var):
                scalars for logging. Grads are in param.grad.
        """
        Ns = all_fxs.shape[0]
        model.zero_grad()

        # === Pass 1: Pathwise (autograd through E_L) ===
        local_E_list = []
        for b_start in range(0, Ns, grad_batch_size):
            b_end = min(b_start + grad_batch_size, Ns)
            fxs_chunk = all_fxs[b_start:b_end]

            if use_log_amp:
                signs, log_abs = model.forward_log(
                    fxs_chunk,
                )
                amps_for_eval = (signs, log_abs)
            else:
                amps_for_eval = model(fxs_chunk)

            _, chunk_E = evaluate_energy_grad(
                fxs_chunk, model, hamiltonian,
                amps_for_eval,
                use_log_amp=use_log_amp,
            )
            local_E_list.append(chunk_E)

        local_E = torch.cat(local_E_list)
        E_mean = local_E.mean()
        E_var = ((local_E - E_mean) ** 2).mean()
        loss = torch.log(E_var + gamma)
        loss.backward()  # pathwise -> param.grad

        # === Pass 2: REINFORCE (score function) ===
        E_mean_val = E_mean.item()
        E_var_val = E_var.item()
        local_E_det = local_E.detach()

        # Centered weights: Cov(O, (E_L-Ebar)^2)
        # The -sigma^2 baseline accounts for <O> != 0
        # from unnormalized psi.
        weights = (
            (local_E_det - E_mean_val) ** 2 - E_var_val
        ) / (E_var_val + gamma)

        for b_start in range(0, Ns, grad_batch_size):
            b_end = min(b_start + grad_batch_size, Ns)
            fxs_chunk = all_fxs[b_start:b_end]
            w = weights[b_start:b_end]

            if use_log_amp:
                _, log_abs = model.forward_log(fxs_chunk)
            else:
                log_abs = torch.log(
                    model(fxs_chunk).abs()
                )

            surr = (2.0 / Ns) * (w * log_abs).sum()
            surr.backward()  # accumulates into param.grad

        return (
            loss.item(),
            E_mean_val,
            E_var_val,
        )

    def run_vmc_loop_logvar(
        self,
        fxs,
        model,
        hamiltonian,
        graph,
        rank,
        world_size,
        config,
        n_params: int,
        nsites: int,
        torch_optimizer,
        on_step_end: Optional[
            Callable[[Dict[str, Any]], None]
        ] = None,
    ):
        """VMC loop using log-variance loss + AdamW.

        Instead of SR, computes a surrogate loss whose
        gradient equals the REINFORCE estimator of
        grad log(Var[E_L] + gamma).  AdamW updates params.
        """
        Warning(
            "Log-variance optimization is experimental and "
            "does not currently converge in practice. Use "
            "with caution."
        )
        self._sync_params(model)

        if rank == 0 and config.show_progress:
            print(
                f"\n--- VMC logvar "
                f"({config.vmc_steps} steps) ---"
            )
            vmc_pbar = tqdm(
                total=config.vmc_steps,
                desc="VMC LogVar",
            )
        else:
            vmc_pbar = None

        gamma = config.logvar_gamma
        energy_history = []

        for step in range(config.vmc_steps):
            t0 = time.time()

            # 1. MCMC sampling (inference mode)
            all_fxs_local, _, fxs, phase_times = (
                self._run_logvar_sampling_phase(
                    fxs=fxs,
                    model=model,
                    hamiltonian=hamiltonian,
                    graph=graph,
                    ns_per_rank=config.ns_per_rank,
                    burn_in=(step == 0),
                    burn_in_steps=config.burn_in_steps,
                    use_export_compile=(
                        config.use_export_compile
                    ),
                    use_log_amp=config.use_log_amp,
                    verbose=config.verbose,
                )
            )

            # 2. Log-variance loss with full autograd
            #    (re-evaluates amps + E_L with grad tracking)
            t_grad_0 = time.time()
            torch_optimizer.zero_grad()
            with torch.enable_grad():
                log_var_loss, energy_mean, energy_var = (
                    self._compute_logvar_loss(
                        all_fxs_local,
                        hamiltonian,
                        gamma,
                        model,
                        config.grad_batch_size,
                        use_log_amp=config.use_log_amp,
                    )
                )
            t_grad = time.time() - t_grad_0

            total_ns = (
                all_fxs_local.shape[0] * world_size
            )

            # 3. Allreduce gradients
            t_comm_0 = time.time()
            self._allreduce_grads(model, world_size)
            t_comm = time.time() - t_comm_0

            # 4. AdamW step
            torch_optimizer.step()

            # 5. Sync params across ranks
            self._sync_params(model)

            del all_fxs_local

            # Logging
            step_time = time.time() - t0
            e_per_site = energy_mean / nsites
            err = (
                np.sqrt(max(energy_var, 0.0) / total_ns)
                / nsites
            )
            energy_history.append(e_per_site)

            # Gradient norm for diagnostics
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5

            global_step = step + config.resume_step
            if rank == 0 and config.show_progress:
                t_s = phase_times.get('t_samp', 0.0)
                t_e = phase_times.get('t_locE', 0.0)
                print(
                    f"Step {global_step:3d} | "
                    f"E/site: {e_per_site:.6f} "
                    f"+/- {err:.6f} | "
                    f"logvar={log_var_loss:.4f} | "
                    f"|g|={grad_norm:.4e} | "
                    f"N={total_ns} | "
                    f"T_samp={t_s:.1f}s "
                    f"T_locE={t_e:.1f}s "
                    f"T_grad={t_grad:.1f}s "
                    f"T_comm={t_comm:.2f}s "
                    f"T_total={step_time:.1f}s"
                )
                vmc_pbar.update(1)

            # Gather walker configs for checkpoint
            if world_size > 1:
                _fxs_list = [
                    torch.zeros_like(fxs)
                    for _ in range(world_size)
                ]
                dist.all_gather(
                    _fxs_list, fxs.contiguous(),
                )
                all_fxs_ckpt = torch.cat(
                    _fxs_list, dim=0,
                )
            else:
                all_fxs_ckpt = fxs

            if on_step_end is not None:
                on_step_end(
                    {
                        "step": global_step,
                        "energy_mean": energy_mean,
                        "energy_var": energy_var,
                        "energy_per_site": e_per_site,
                        "error_per_site": err,
                        "total_samples": total_ns,
                        "sample_time": (
                            phase_times['t_samp']
                            + phase_times['t_locE']
                        ),
                        "phase_times": phase_times,
                        "sr_time": 0.0,
                        "total_time": step_time,
                        "solver_info": None,
                        "fxs": all_fxs_ckpt,
                    }
                )

        if vmc_pbar is not None:
            vmc_pbar.close()
        return energy_history, fxs

    # ==========================================================
    # Supervised Wavefunction Optimization (SWO)
    # ==========================================================

    def run_SWO_state_fitting_gpu(
        self,
        fxs,
        training_model,
        target_model,
        graph,
        ns_per_rank,
        *,
        loss: str = 'fidelity',
        sample_times: int = 10,
        SWO_max_iter: int = int(1e3),
        log_fidelity_tol: float = 1e-4,
        burn_in_steps: int = 0,
        grad_batch_size: Optional[int] = None,
        hamiltonian=None,
        tmpdir: Optional[str] = None,
        save: bool = True,
        save_step_offset: int = 0,
        scheduler=None,
        verbose: bool = False,
        sr_rshift: float = 0.0,
        sr_ashift: float = 1e-4,
        sr_ratio_clip: Optional[float] = None,
        learning_rate: float = 1e-3,
        torch_optimizer_cls=torch.optim.SGD,
        torch_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        ratio: float = 0.0,
        line_search: bool = False,
        line_search_shrink: float = 0.5,
        line_search_max_backtracks: int = 8,
        line_search_min_lr: float = 1e-8,
    ):
        """Run SWO state fitting with a selected fitting loss.

        Args:
            loss: One of ``'fidelity'``, ``'sr'``, or ``'l2'``.
                ``'fidelity'`` uses the CPU-mirrored explicit fidelity
                gradient. ``'sr'`` uses quantax-style supervised SR/MinSR.
                ``'l2'`` uses sample-normalized amplitude L2 with
                autograd.
            ratio: Fraction of each outer-loop dataset sampled from
                target_model B instead of training_model A. This is an
                empirical mixed-sampling knob for tiny-overlap fits; the
                strict A-distribution estimator is recovered at ``0``.
            line_search: If True, accept fidelity/SR updates only when
                they reduce the fixed-sample sampled-fidelity loss.

        The outer loop is shared for all losses: sample from A, cache B,
        report initial fidelity, run the selected inner fit, optionally
        measure energy, and save the checkpoint.
        """
        loss = loss.lower()
        valid_losses = {'fidelity', 'sr', 'l2'}
        if loss not in valid_losses:
            raise ValueError(
                f"Unknown SWO loss {loss!r}; expected one of "
                f"{sorted(valid_losses)}."
            )
        if loss in ('fidelity', 'sr') and self.optimizer is None:
            raise ValueError(
                f"loss={loss!r} requires an OptimizerGPU, e.g. SGDGPU()."
            )
        if loss == 'sr' and self.preconditioner is None:
            raise ValueError(
                "loss='sr' requires a GPU SR preconditioner, "
                "e.g. MinSRGPU()."
            )

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = (
            dist.get_world_size() if dist.is_initialized() else 1
        )
        device = next(training_model.parameters()).device
        Np = sum(p.numel() for p in training_model.parameters())
        B = fxs.shape[0]
        if grad_batch_size is None:
            grad_batch_size = B
        n_total = ns_per_rank * world_size

        trainable_params = None
        if loss == 'l2':
            trainable_params = [
                p for p in training_model.parameters() if p.requires_grad
            ]
            if not trainable_params:
                raise ValueError("training_model has no trainable parameters")

        MC_stats = {
            'sample_size': n_total,
            'target_sample_ratio': ratio,
            '-logf': [],
            'fidelity': [],
            'fidelity_diagnostics': [],
            'energy': [],
        }
        if loss == 'sr':
            MC_stats['sr_time'] = []
            MC_stats['sr_info'] = []
        if loss == 'l2':
            MC_stats['l2_loss'] = []

        self._sync_params(training_model)
        self._sync_params(target_model)

        label = {
            'fidelity': 'SWO',
            'sr': 'SWO-SR',
            'l2': 'SWO-L2',
        }[loss]

        for local_step in range(sample_times):
            t_step = save_step_offset + local_step
            
            with torch.no_grad():
                (
                    fxs, configs, sign_B, log_abs_B,
                    sign_A0, log_abs_A0,
                ) = collect_swo_dataset(
                    self.sampler, fxs,
                    training_model, target_model, graph,
                    ns_per_rank,
                    burn_in=(t_step == 0),
                    burn_in_steps=burn_in_steps,
                    ratio=ratio,
                )
                
                log_abs_B_f64 = log_abs_B.to(torch.float64)
                sign_B_f64 = sign_B.to(torch.float64)
                log_abs_A0_f64 = log_abs_A0.to(torch.float64)
            
                fid_stats = fidelity_stats_from_log_amps(
                    sign_A0, log_abs_A0,
                    sign_B, log_abs_B,
                    n_total,
                )
                fid_init = fid_stats['fidelity']
                logf_init = fid_stats['log_f']

            if rank == 0:
                MC_stats['-logf'].append(float(logf_init))
                MC_stats['fidelity'].append(float(fid_init))
                fid_diag = {
                    key: float(value)
                    for key, value in fid_stats.items()
                    if key not in ('fidelity', 'log_f')
                }
                MC_stats['fidelity_diagnostics'].append(fid_diag)
                print(
                    f"[{label} outer {t_step}] init fidelity="
                    f"{float(fid_init):.6e}, "
                    f"-log f={float(logf_init):.6e}, "
                )

            checkpoint_optimizer = self.optimizer
            if loss == 'fidelity':
                self._run_swo_fidelity_inner(
                    configs=configs,
                    sign_B_f64=sign_B_f64,
                    log_abs_B_f64=log_abs_B_f64,
                    log_abs_A0_f64=log_abs_A0_f64,
                    training_model=training_model,
                    grad_batch_size=grad_batch_size,
                    Np=Np,
                    n_total=n_total,
                    device=device,
                    t_step=t_step,
                    SWO_max_iter=SWO_max_iter,
                    log_fidelity_tol=log_fidelity_tol,
                    scheduler=scheduler,
                    rank=rank,
                    verbose=verbose,
                    line_search=line_search,
                    line_search_shrink=line_search_shrink,
                    line_search_max_backtracks=line_search_max_backtracks,
                    line_search_min_lr=line_search_min_lr,
                )
            elif loss == 'sr':
                sr_stats = self._run_swo_sr_inner(
                    configs=configs,
                    sign_B_f64=sign_B_f64,
                    log_abs_B_f64=log_abs_B_f64,
                    training_model=training_model,
                    grad_batch_size=grad_batch_size,
                    Np=Np,
                    n_total=n_total,
                    device=device,
                    t_step=t_step,
                    SWO_max_iter=SWO_max_iter,
                    log_fidelity_tol=log_fidelity_tol,
                    scheduler=scheduler,
                    sr_rshift=sr_rshift,
                    sr_ashift=sr_ashift,
                    sr_ratio_clip=sr_ratio_clip,
                    rank=rank,
                    verbose=verbose,
                    line_search=line_search,
                    line_search_shrink=line_search_shrink,
                    line_search_max_backtracks=line_search_max_backtracks,
                    line_search_min_lr=line_search_min_lr,
                )
                if rank == 0:
                    MC_stats['sr_time'].append(sr_stats['sr_time'])
                    MC_stats['sr_info'].append(sr_stats['sr_info'])
            else:
                l2_stats = self._run_swo_l2_inner(
                    configs=configs,
                    sign_B=sign_B,
                    log_abs_B=log_abs_B,
                    training_model=training_model,
                    trainable_params=trainable_params,
                    batch_size=grad_batch_size,
                    n_total=n_total,
                    t_step=t_step,
                    SWO_max_iter=SWO_max_iter,
                    learning_rate=learning_rate,
                    torch_optimizer_cls=torch_optimizer_cls,
                    torch_optimizer_kwargs=torch_optimizer_kwargs,
                    scheduler=scheduler,
                    rank=rank,
                    world_size=world_size,
                    verbose=verbose,
                )
                checkpoint_optimizer = l2_stats['optimizer']
                if rank == 0:
                    MC_stats['l2_loss'].append(l2_stats['l2_loss'])

            if hamiltonian is not None:
                local_E_sum = torch.zeros(
                    (), device=device, dtype=torch.float64,
                )
                with torch.inference_mode():
                    for start in range(0, configs.shape[0], grad_batch_size):
                        stop = min(start + grad_batch_size, configs.shape[0])
                        cfg_chunk = configs[start:stop]
                        amps_out = training_model.forward_log(cfg_chunk)
                        _, local_E = self.evaluate_energy_fn(
                            cfg_chunk, training_model, hamiltonian, amps_out,
                            use_log_amp=True,
                        )
                        local_E_sum += local_E.detach().to(
                            torch.float64,
                        ).sum()
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(local_E_sum, op=dist.ReduceOp.SUM)
                energy_mean_per_site = (
                    local_E_sum / n_total / graph.n_nodes
                )
                if rank == 0:
                    MC_stats['energy'].append(float(energy_mean_per_site))
                    print(
                        f"[{label} outer {t_step}] "
                        f"energy={float(energy_mean_per_site):.12e}\n"
                    )

            if rank == 0 and tmpdir is not None and save:
                save_swo_checkpoint(
                    training_model, MC_stats, tmpdir, t_step,
                    optimizer=checkpoint_optimizer,
                )

        return fxs, MC_stats

    def _run_swo_fidelity_inner(
        self,
        *,
        configs,
        sign_B_f64,
        log_abs_B_f64,
        log_abs_A0_f64,
        training_model,
        grad_batch_size,
        Np,
        n_total,
        device,
        t_step,
        SWO_max_iter,
        log_fidelity_tol,
        scheduler,
        rank,
        verbose,
        line_search,
        line_search_shrink,
        line_search_max_backtracks,
        line_search_min_lr,
    ):
        if scheduler is not None:
            self.optimizer.lr = scheduler(t_step)
        self.optimizer.reset()

        pbar = tqdm(range(SWO_max_iter)) if rank == 0 and verbose else None
        for it in range(SWO_max_iter):
            terms = accumulate_fidelity_terms(
                configs,
                log_abs_A0_f64,
                sign_B_f64, log_abs_B_f64,
                training_model,
                grad_batch_size=grad_batch_size,
                Np=Np, device=device,
            )
            direction, log_f = compute_swo_direction(terms, n_total)
            line_search_stats = None
            if line_search:
                line_search_stats = self._swo_line_search_step(
                    configs=configs,
                    sign_B=sign_B_f64,
                    log_abs_B=log_abs_B_f64,
                    training_model=training_model,
                    direction=direction,
                    old_log_f=log_f,
                    grad_batch_size=grad_batch_size,
                    n_total=n_total,
                    max_backtracks=line_search_max_backtracks,
                    shrink=line_search_shrink,
                    min_lr=line_search_min_lr,
                    rank=rank,
                )
            else:
                self.optimizer.step(training_model, direction)

            if rank == 0 and pbar is not None:
                desc = f"SWO {it} -logF={float(log_f):.3e}"
                if line_search_stats is not None:
                    desc += (
                        f" -> {float(line_search_stats['log_f']):.3e} "
                        f"lr={line_search_stats['lr']:.2e} "
                        f"bt={line_search_stats['n_backtrack']}"
                    )
                pbar.set_description(desc)
                pbar.update(1)

            if float(log_f) < log_fidelity_tol:
                break

        if pbar is not None:
            pbar.close()

    def _run_swo_sr_inner(
        self,
        *,
        configs,
        sign_B_f64,
        log_abs_B_f64,
        training_model,
        grad_batch_size,
        Np,
        n_total,
        device,
        t_step,
        SWO_max_iter,
        log_fidelity_tol,
        scheduler,
        sr_rshift,
        sr_ashift,
        sr_ratio_clip,
        rank,
        verbose,
        line_search,
        line_search_shrink,
        line_search_max_backtracks,
        line_search_min_lr,
    ):
        if scheduler is not None:
            self.optimizer.lr = scheduler(t_step)
        self.optimizer.reset()

        pbar = tqdm(range(SWO_max_iter)) if rank == 0 and verbose else None
        total_sr_time = 0.0
        last_info = None
        for it in range(SWO_max_iter):
            terms = accumulate_supervised_sr_terms(
                configs,
                sign_B_f64, log_abs_B_f64,
                training_model,
                grad_batch_size=grad_batch_size,
                device=device,
                n_total=n_total,
                ratio_clip=sr_ratio_clip,
            )
            direction, sr_time, info = self.preconditioner.solve(
                O_loc=terms['local_lpg'],
                E_loc=terms['local_signal'],
                E_mean=float(terms['signal_mean']),
                Ns=n_total,
                Np=Np,
                rshift=sr_rshift,
                ashift=sr_ashift,
                device=device,
            )

            total_sr_time += sr_time
            last_info = info

            log_f = terms['log_f']
            line_search_stats = None
            if line_search:
                line_search_stats = self._swo_line_search_step(
                    configs=configs,
                    sign_B=sign_B_f64,
                    log_abs_B=log_abs_B_f64,
                    training_model=training_model,
                    direction=direction,
                    old_log_f=log_f,
                    grad_batch_size=grad_batch_size,
                    n_total=n_total,
                    max_backtracks=line_search_max_backtracks,
                    shrink=line_search_shrink,
                    min_lr=line_search_min_lr,
                    rank=rank,
                )
            else:
                self.optimizer.step(training_model, direction)

            # DEBUG:
            # compare norm of direction and model parameters
            model_params_vec = torch.nn.utils.parameters_to_vector(
                training_model.parameters(),
            )
            dir_norm = torch.norm(direction).item()
            params_norm = torch.norm(model_params_vec).item()
            print(
                f"  [dbg]: "
                f"dir_norm={dir_norm:.4e} "
                f"params_norm={params_norm:.4e} "
                f"log_f={float(log_f):.3e} "
                f"ESS={float(terms['ess']):.1f}/{n_total} "
            )

            if rank == 0 and pbar is not None:
                desc = (
                    f"SWO-SR {it} -logF={float(log_f):.3e} "
                    f"T_SR={sr_time:.2f}s info={info}"
                )
                if line_search_stats is not None:
                    desc += (
                        f" -> {float(line_search_stats['log_f']):.3e} "
                        f"lr={line_search_stats['lr']:.2e} "
                        f"bt={line_search_stats['n_backtrack']}"
                    )
                pbar.set_description(desc)
                pbar.update(1)

            del terms
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if float(log_f) < log_fidelity_tol:
                break

        if pbar is not None:
            pbar.close()

        return {
            'sr_time': float(total_sr_time),
            'sr_info': None if last_info is None else int(last_info)
            if isinstance(last_info, (int, np.integer)) else str(last_info),
        }

    def _run_swo_l2_inner(
        self,
        *,
        configs,
        sign_B,
        log_abs_B,
        training_model,
        trainable_params,
        batch_size,
        n_total,
        t_step,
        SWO_max_iter,
        learning_rate,
        torch_optimizer_cls,
        torch_optimizer_kwargs,
        scheduler,
        rank,
        world_size,
        verbose,
    ):
        device = next(training_model.parameters()).device
        amp_dtype = next(training_model.parameters()).dtype
        target_amps = (
            sign_B.to(device=device, dtype=amp_dtype)
            * torch.exp(log_abs_B.to(device=device, dtype=amp_dtype))
        ).detach()
        local_b2_sum = (target_amps.to(torch.float64) ** 2).sum()
        global_b2_sum = local_b2_sum.clone()
        if dist.is_initialized() and world_size > 1:
            dist.all_reduce(global_b2_sum, op=dist.ReduceOp.SUM)
        target_norm = torch.sqrt(
            (global_b2_sum / n_total).clamp(min=1e-300)
        ).to(dtype=amp_dtype)

        lr = scheduler(t_step) if scheduler is not None else learning_rate
        opt_kwargs = dict(torch_optimizer_kwargs or {})
        opt_kwargs['lr'] = lr
        torch_optimizer = torch_optimizer_cls(trainable_params, **opt_kwargs)

        pbar = tqdm(range(SWO_max_iter)) if rank == 0 and verbose else None
        last_l2_loss = None
        for it in range(SWO_max_iter):
            torch_optimizer.zero_grad(set_to_none=True)

            current_amps = torch.empty_like(target_amps)
            local_a2_sum = torch.zeros(
                (), device=device, dtype=torch.float64,
            )

            with torch.inference_mode():
                for start in range(0, configs.shape[0], batch_size):
                    stop = min(start + batch_size, configs.shape[0])
                    amp_A = training_model(configs[start:stop])
                    current_amps[start:stop] = amp_A.detach()
                    local_a2_sum += (amp_A.detach().to(
                        torch.float64,
                    ) ** 2).sum()

            global_a2_sum = local_a2_sum.clone()
            if dist.is_initialized() and world_size > 1:
                dist.all_reduce(global_a2_sum, op=dist.ReduceOp.SUM)
            current_norm = torch.sqrt(
                (global_a2_sum / n_total).clamp(min=1e-300)
            ).to(dtype=amp_dtype)

            normed_A = current_amps / current_norm
            normed_B = target_amps / target_norm
            residual = normed_A - normed_B
            local_loss_sum = (residual.to(torch.float64) ** 2).sum()
            total_loss_sum = local_loss_sum.clone()
            if dist.is_initialized() and world_size > 1:
                dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
            last_l2_loss = total_loss_sum / n_total

            local_y_dot_a = (
                residual.to(torch.float64)
                * current_amps.to(torch.float64)
            ).sum()
            global_y_dot_a = local_y_dot_a.clone()
            if dist.is_initialized() and world_size > 1:
                dist.all_reduce(global_y_dot_a, op=dist.ReduceOp.SUM)

            coeffs = (2.0 / n_total) * (
                residual.to(torch.float64) / current_norm.to(torch.float64)
                - (
                    current_amps.to(torch.float64)
                    * global_y_dot_a
                    / (
                        n_total
                        * current_norm.to(torch.float64) ** 3
                    )
                )
            )

            for start in range(0, configs.shape[0], batch_size):
                stop = min(start + batch_size, configs.shape[0])
                amp_A = training_model(configs[start:stop])
                vjp = (
                    amp_A.to(torch.float64)
                    * coeffs[start:stop]
                ).sum()
                vjp.backward()

            if dist.is_initialized() and world_size > 1:
                for p in trainable_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

            grad_rms = None
            if rank == 0 and pbar is not None:
                grad_sq_sum = torch.zeros(
                    (), device=device, dtype=torch.float64,
                )
                grad_numel = 0
                for p in trainable_params:
                    if p.grad is not None:
                        grad_sq_sum += (p.grad.detach().to(
                            torch.float64,
                        ) ** 2).sum()
                        grad_numel += p.grad.numel()
                grad_rms = torch.sqrt(grad_sq_sum / max(grad_numel, 1))

            torch_optimizer.step()

            if rank == 0 and pbar is not None:
                pbar.set_description(
                    f"SWO-L2 {it} "
                    f"L2={float(last_l2_loss):.6e} "
                    f"|g|rms={float(grad_rms):.3e}"
                )
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        return {
            'l2_loss': float(last_l2_loss),
            'optimizer': torch_optimizer,
        }


__all__ = [
    "setup_distributed",
    "print_sampling_settings",
    "VMC_GPU",
]
