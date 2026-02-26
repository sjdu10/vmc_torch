import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from vmc_torch.experiment.vmap.GPU.optimizer import OptimizerGPU, PreconditionerGPU
from vmc_torch.experiment.vmap.GPU.sampler import SamplerGPU
from vmc_torch.experiment.vmap.GPU.vmc_modules import (
    distributed_minres_solver_gpu,
    minSR_solver_gpu,
    run_sampling_phase_gpu,
)
from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    compute_grads_gpu,
    evaluate_energy,
    sample_next,
)


def setup_distributed():
    if "RANK" not in os.environ:
        print("Warning: Not using torchrun. Single device.")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["LOCAL_RANK"] = "0"

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def print_sampling_settings(rank, world_size, batch_size, ns_per_rank, grad_batch_size):
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


class VMC_GPU:
    """
    GPU VMC driver with class-based component injection.

    Defaults preserve legacy function-based behavior.
    """

    def __init__(
        self,
        sampling_phase_fn: Callable[..., Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, float]] = run_sampling_phase_gpu,
        distributed_sr_solver_fn: Callable[..., Tuple[np.ndarray, float, Any]] = distributed_minres_solver_gpu,
        min_sr_solver_fn: Callable[..., Tuple[np.ndarray, float, Any]] = minSR_solver_gpu,
        sample_next_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = sample_next,
        evaluate_energy_fn: Callable[..., Tuple[Any, torch.Tensor]] = evaluate_energy,
        compute_grads_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = compute_grads_gpu,
        sampling_count_key: str = "Ns",
        sampler: Optional[SamplerGPU] = None,
        preconditioner: Optional[PreconditionerGPU] = None,
        optimizer: Optional[OptimizerGPU] = None,
    ):
        self.sampling_phase_fn = sampling_phase_fn
        self.distributed_sr_solver_fn = distributed_sr_solver_fn
        self.min_sr_solver_fn = min_sr_solver_fn
        self.sample_next_fn = sample_next_fn
        self.evaluate_energy_fn = evaluate_energy_fn
        self.compute_grads_fn = compute_grads_fn
        self.sampling_count_key = sampling_count_key

        self.sampler = sampler
        self.preconditioner = preconditioner
        self.optimizer = optimizer

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
            if self.sampler is not None:
                fxs, amps = self.sampler.warmup_step(
                    fxs=fxs,
                    model=model,
                    graph=graph,
                    use_export_compile=config.use_export_compile,
                    verbose=config.verbose,
                )
            else:
                fxs, amps = self.sample_next_fn(
                    fxs,
                    model,
                    graph,
                    verbose=config.verbose,
                    compile=config.use_export_compile,
                )
            if rank == 0 and config.verbose:
                print(f"  sample_next:     {time.time() - t_warm:.2f}s")
            t1 = time.time()
            _, evals = self.evaluate_energy_fn(fxs, model, hamiltonian, amps)
            if rank == 0 and config.verbose:
                print(f"  evaluate_energy: {time.time() - t1:.2f}s")

        t2 = time.time()
        with torch.enable_grad():
            grads, amps2 = self.compute_grads_fn(
                fxs,
                model,
                vectorize=True,
                batch_size=config.grad_batch_size,
                vmap_grad=True,
            )
        if rank == 0 and config.verbose:
            print(f"  compute_grads:   {time.time() - t2:.2f}s")
            print(f"  Warmup total:    {time.time() - t_warm:.2f}s")

        del grads, amps2, evals
        return fxs

    def compute_global_energy_stats(self, local_energies, world_size):
        n_local = local_energies.shape[0]
        total_ns = n_local * world_size

        local_e_sum = local_energies.sum()
        dist.all_reduce(local_e_sum, op=dist.ReduceOp.SUM)
        energy_mean = local_e_sum.item() / total_ns

        local_e_sq_sum = (local_energies ** 2).sum()
        dist.all_reduce(local_e_sq_sum, op=dist.ReduceOp.SUM)
        energy_var = local_e_sq_sum.item() / total_ns - energy_mean ** 2

        return total_ns, energy_mean, energy_var

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
        use_min_sr,
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

        if use_min_sr:
            return self.min_sr_solver_fn(
                local_O=local_o,
                local_energies=local_energies,
                energy_mean=energy_mean,
                total_samples=total_samples,
                n_params=n_params,
                diag_shift=diag_shift,
                device=device,
                do_SR=run_sr,
            )

        return self.distributed_sr_solver_fn(
            local_O=local_o,
            local_energies=local_energies,
            energy_mean=energy_mean,
            total_samples=total_samples,
            n_params=n_params,
            diag_shift=diag_shift,
            rtol=5e-5,
            run_SR=run_sr,
        )

    def apply_parameter_update(self, model, dp, learning_rate, device):
        if self.optimizer is not None:
            self.optimizer.step(
                model=model,
                direction=dp,
                device=device,
                learning_rate=learning_rate,
            )
            return

        with torch.no_grad():
            dp_tensor = torch.as_tensor(dp, device=device, dtype=torch.float64)
            current_params_vec = torch.nn.utils.parameters_to_vector(model.parameters())
            new_params_vec = current_params_vec - learning_rate * dp_tensor
            torch.nn.utils.vector_to_parameters(new_params_vec, model.parameters())

    def run_vmc_loop(
        self,
        fxs,
        model,
        hamiltonian,
        graph,
        rank,
        world_size,
        config: VMCLoopConfig,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        on_step_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        if sampling_kwargs is None:
            sampling_kwargs = {}

        device = next(model.parameters()).device
        if rank == 0 and config.show_progress:
            print(f"\n--- VMC ({config.vmc_steps} steps) ---")
            vmc_pbar = tqdm(total=config.vmc_steps, desc="VMC Steps")
        else:
            vmc_pbar = None

        energy_history = []
        for step in range(config.vmc_steps):
            t0 = time.time()

            if self.sampler is not None:
                (local_energies, local_o), fxs, sample_time = self.sampler.run_sampling_phase(
                    fxs=fxs,
                    model=model,
                    hamiltonian=hamiltonian,
                    graph=graph,
                    ns_per_rank=config.ns_per_rank,
                    grad_batch_size=config.grad_batch_size,
                    burn_in=(step == 0),
                    burn_in_steps=config.burn_in_steps,
                    use_export_compile=config.use_export_compile,
                    sampling_kwargs=sampling_kwargs,
                )
            else:
                phase_kwargs = dict(
                    fxs=fxs,
                    model=model,
                    hamiltonian=hamiltonian,
                    graph=graph,
                    grad_batch_size=config.grad_batch_size,
                    burn_in=(step == 0),
                    burn_in_steps=config.burn_in_steps,
                    verbose=False,
                    compile=config.use_export_compile,
                )
                phase_kwargs[self.sampling_count_key] = config.ns_per_rank
                phase_kwargs.update(sampling_kwargs)
                (local_energies, local_o), fxs, sample_time = self.sampling_phase_fn(**phase_kwargs)

            total_ns, energy_mean, energy_var = self.compute_global_energy_stats(local_energies, world_size)

            dp, t_sr, info = self.solve_sr_step(
                local_o=local_o,
                local_energies=local_energies,
                energy_mean=energy_mean,
                total_samples=total_ns,
                n_params=config.n_params,
                diag_shift=config.diag_shift,
                device=device,
                run_sr=config.run_sr,
                use_min_sr=config.use_min_sr,
            )

            self.apply_parameter_update(model, dp, config.learning_rate, device)
            step_time = time.time() - t0

            e_per_site = energy_mean / config.nsites
            err = np.sqrt(max(energy_var, 0.0) / total_ns) / config.nsites
            energy_history.append(e_per_site)

            if rank == 0 and config.show_progress:
                print(
                    f"Step {step:3d} | E/site: {e_per_site:.6f} "
                    f"+/- {err:.6f} | N={total_ns} | "
                    f"T_samp={sample_time:.1f}s T_SR={t_sr:.2f}s "
                    f"T_total={step_time:.1f}s"
                )
                vmc_pbar.update(1)

            if on_step_end is not None:
                on_step_end(
                    {
                        "step": step,
                        "energy_mean": energy_mean,
                        "energy_var": energy_var,
                        "energy_per_site": e_per_site,
                        "error_per_site": err,
                        "total_samples": total_ns,
                        "sample_time": sample_time,
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
