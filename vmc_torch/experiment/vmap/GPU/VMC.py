import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from vmc_torch.experiment.vmap.GPU.optimizer import (
    OptimizerGPU,
    PreconditionerGPU,
)
from vmc_torch.experiment.vmap.GPU.sampler import SamplerGPU
from vmc_torch.experiment.vmap.GPU.vmc_modules import (
    distributed_minres_solver_gpu,
    minSR_solver_gpu,
)
from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    compute_grads_gpu,
    evaluate_energy,
)


def setup_distributed():
    if "RANK" not in os.environ:
        print("Warning: Not using torchrun. Single device.")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["LOCAL_RANK"] = "0"

    dist.init_process_group(
        backend="nccl", init_method="env://",
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
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
    use_min_sr: bool = False
    use_export_compile: bool = False
    show_progress: bool = True
    step_offset: int = 0  # global step = local step + step_offset
    debug: bool = False  # print [dbg] diagnostics each step
    # O_loc outlier clipping: drop samples whose per-sample
    # |O_loc| norm exceeds clip_factor * median(|O_loc| norms).
    # Set to 0 to disable.  Typical value: 5-10.
    outlier_clip_factor: float = 100.0
    # Optional LR scheduler: callable(step) -> float.
    # If set, overrides learning_rate each step.
    lr_scheduler: object = None


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
    ):
        """Run MCMC sampling, energy eval, and gradient
        computation for one VMC step.

        The sampler only does MCMC (step / burn_in). This
        method calls evaluate_energy_fn and
        compute_grads_fn directly.

        Returns:
            (local_energies, local_O): GPU tensors,
                shapes (Ns,) and (Ns, Np).
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
            )
            t_samp += time.time() - t0

        local_energies_list = []
        local_O_list = []
        current_count = 0

        while current_count < ns_per_rank:
            needed = min(B, ns_per_rank - current_count)

            # 1. MCMC sweep
            t0 = time.time()
            fxs, amps = self.sampler.step(
                fxs, model, graph,
                compile=use_export_compile,
            )
            t_samp += time.time() - t0

            # 2. Local energy
            t0 = time.time()
            _, local_E = self.evaluate_energy_fn(
                fxs, model, hamiltonian, amps,
            )
            t_locE += time.time() - t0

            # 3. Gradients -> O_loc
            t0 = time.time()
            with torch.enable_grad():
                grads, amps2 = self.compute_grads_fn(
                    fxs, model,
                    vectorize=True,
                    batch_size=grad_batch_size,
                    vmap_grad=True,
                )
            # O_loc = grads / amps
            _rank = dist.get_rank() if dist.is_initialized() else 0
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
                    f"  [dbg] O_loc: rms="
                    f"{torch.norm(grads).item() / grads.numel()**0.5:.4e}"
                    f", max={grads.abs().max().item():.4e}"
                )
            t_grad += time.time() - t0

            local_energies_list.append(
                local_E[:needed].detach(),
            )
            local_O_list.append(
                grads[:needed].detach(),
            )
            current_count += needed
            del grads, amps2, local_E

        local_energies = torch.cat(local_energies_list)
        local_O = torch.cat(local_O_list)

        # --- Outlier masking ---
        # Drop samples whose per-sample O_loc norm exceeds
        # clip_factor * median.  Zeroed samples don't
        # contribute to SR gradient or energy mean.
        if outlier_clip_factor > 0:
            o_norms = local_O.norm(dim=1)  # (Ns,)
            median_norm = o_norms.median()
            threshold = outlier_clip_factor * median_norm
            mask = o_norms <= threshold  # True = keep
            n_dropped = int((~mask).sum().item())
            if n_dropped > 0:
                _rank = (
                    dist.get_rank()
                    if dist.is_initialized() else 0
                )
                if _rank == 0:
                    print(
                        f"  [outlier] dropped {n_dropped}"
                        f"/{local_O.shape[0]} samples "
                        f"(|O_loc| > {threshold.item():.2e},"
                        f" median={median_norm.item():.2e})"
                    )
                local_O[~mask] = 0.0
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
            (local_energies, local_O),
            fxs,
            sample_time,
            phase_times,
        )

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
        if rank == 0 and config.verbose:
            print("\n--- Warmup (1 sweep) ---")
        t_warm = time.time()

        with torch.inference_mode():
            fxs, amps = self.sampler.step(
                fxs, model, graph,
                compile=config.use_export_compile,
                verbose=config.verbose,
            )
            if rank == 0 and config.verbose:
                print(
                    f"  sample_next:     "
                    f"{time.time() - t_warm:.2f}s"
                )
            t1 = time.time()
            _, evals = self.evaluate_energy_fn(
                fxs, model, hamiltonian, amps,
            )
            if rank == 0 and config.verbose:
                print(
                    f"  evaluate_energy: "
                    f"{time.time() - t1:.2f}s"
                )

        t2 = time.time()
        with torch.enable_grad():
            grads, amps2 = self.compute_grads_fn(
                fxs, model,
                vectorize=True,
                batch_size=config.grad_batch_size,
                vmap_grad=True,
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

        del grads, amps2, evals
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
            # No preconditioner — raw energy gradient for SGD
            t0 = time.time()

            if isinstance(local_o, torch.Tensor):
                local_o_np = local_o.cpu().numpy()
            else:
                local_o_np = local_o
            if isinstance(local_energies, torch.Tensor):
                local_e_np = local_energies.cpu().numpy()
            else:
                local_e_np = local_energies

            n_local = local_e_np.shape[0]
            world_size = (
                dist.get_world_size()
                if dist.is_initialized() else 1
            )

            if n_local > 0:
                local_sum_O = np.sum(local_o_np, axis=0)
                local_sum_EO = np.dot(local_e_np, local_o_np)
            else:
                local_sum_O = np.zeros(
                    n_params, dtype=np.float64,
                )
                local_sum_EO = np.zeros(
                    n_params, dtype=np.float64,
                )

            if world_size > 1:
                sum_O_t = torch.tensor(
                    local_sum_O, device=device,
                )
                sum_EO_t = torch.tensor(
                    local_sum_EO, device=device,
                )
                dist.all_reduce(
                    sum_O_t, op=dist.ReduceOp.SUM,
                )
                dist.all_reduce(
                    sum_EO_t, op=dist.ReduceOp.SUM,
                )
                global_sum_O = sum_O_t.cpu().numpy()
                global_sum_EO = sum_EO_t.cpu().numpy()
            else:
                global_sum_O = local_sum_O
                global_sum_EO = local_sum_EO

            mean_O = global_sum_O / total_samples
            mean_EO = global_sum_EO / total_samples
            energy_grad = (
                mean_EO - energy_mean * mean_O
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

            (local_energies, local_o), fxs, sample_time, phase_times = (
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
                n_nan_O = torch.isnan(local_o).sum().item()
                n_inf_O = torch.isinf(local_o).sum().item()
                if n_nan_E or n_nan_O or n_inf_O:
                    print(
                        f"[WARNING] SR inputs: "
                        f"local_E has {n_nan_E} NaN, "
                        f"local_O has {n_nan_O} NaN / "
                        f"{n_inf_O} Inf"
                    )
                if config.debug:
                    Np = local_o.shape[1]
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
                        f"  [dbg] O_loc: rms="
                        f"{torch.norm(local_o).item()/(local_o.numel()**0.5):.4e}, "
                        f"max={local_o.abs().max().item():.4e}"
                    )
                    print(
                        f"  [dbg] local_E: "
                        f"mean={local_energies.mean().item():.4e}, "
                        f"std={local_energies.std().item():.4e}"
                    )

            dp, t_sr, info = self.solve_sr_step(
                local_o=local_o,
                local_energies=local_energies,
                energy_mean=energy_mean,
                total_samples=total_ns,
                n_params=config.n_params,
                diag_shift=config.diag_shift,
                device=device,
                run_sr=config.run_sr,
            )

            # --- NaN/Inf check on SR direction ---
            dp_t = torch.as_tensor(dp, device=device)
            dp_nan = torch.isnan(dp_t).sum().item()
            dp_inf = torch.isinf(dp_t).sum().item()
            if (dp_nan > 0 or dp_inf > 0) and rank == 0:
                print(
                    f"[WARNING] SR dp has {dp_nan} NaN, "
                    f"{dp_inf} Inf out of {dp_t.numel()}"
                )

            if config.lr_scheduler is not None:
                global_step = step + config.step_offset
                config.learning_rate = config.lr_scheduler(
                    global_step,
                )

            self.apply_parameter_update(
                model, dp, config.learning_rate, device,
            )

            # --- NaN/Inf check on updated params ---
            with torch.no_grad():
                pv = torch.nn.utils.parameters_to_vector(
                    model.parameters(),
                )
                pv_nan = torch.isnan(pv).sum().item()
                pv_inf = torch.isinf(pv).sum().item()
                if (pv_nan > 0 or pv_inf > 0) and rank == 0:
                    print(
                        f"[WARNING] After update: params "
                        f"have {pv_nan} NaN, {pv_inf} Inf "
                        f"out of {pv.numel()}"
                    )
                elif config.debug and rank == 0:
                    Np = pv.numel()
                    dp_rms = (
                        torch.norm(dp_t).item() / Np ** 0.5
                    )
                    pv_rms = (
                        torch.norm(pv).item() / Np ** 0.5
                    )
                    print(
                        f"  [dbg] dp rms={dp_rms:.4e}, "
                        f"params rms={pv_rms:.4e}, "
                        f"lr*dp_rms="
                        f"{config.learning_rate * dp_rms:.4e}"
                    )
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
                    f"T_total={step_time:.1f}s"
                )
                vmc_pbar.update(1)

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
