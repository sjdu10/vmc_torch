"""GPU VMC for Slater determinant on spinful Fermi-Hubbard.

Mean-field variational ansatz (no correlations).
Useful as a baseline / sanity check.

Run:
    torchrun --nproc_per_node=<N> run_scripts/vmc_run_slater.py
    torchrun --nproc_per_node=1 run_scripts/vmc_run_slater.py
"""
import json
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from vmc_torch.GPU.VMC import (
    VMC_GPU,
    VMCLoopConfig,
    VMCWarmupConfig,
    print_sampling_settings,
    setup_distributed,
)
from vmc_torch.GPU.hamiltonian import (
    spinful_Fermi_Hubbard_square_lattice_torch,
)
from vmc_torch.GPU.models import SlaterDeterminant_GPU
from vmc_torch.GPU.optimizer import (
    DecayScheduler,
    MinSRGPU,
    SGDGPU,
)
from vmc_torch.GPU.sampler import (
    MetropolisExchangeSpinfulSamplerGPU,
)
from vmc_torch.GPU.vmc_setup import (
    initialize_walkers,
)
from vmc_torch.GPU.vmc_utils import random_initial_config

dtype = torch.float64
DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/GPU/data'
)


@dataclass
class VMCConfig:
    """VMC numerical / training settings."""

    batch_size: int = 4096
    ns_per_rank: int = 4096
    grad_batch_size: int = 4096
    vmc_steps: int = 100
    learning_rate: float = 0.1
    diag_shift: float = 1e-4
    burn_in_steps: int = 4
    use_export_compile: bool = True
    use_min_sr: bool = False  # MinSR if Ns<Np, I prefer minres over MinSR
    sr_rtol: float = 1e-4
    sr_maxiter: int = 100
    save_every: int = 10
    debug: bool = False
    outlier_clip_factor: float = 100.0 # drop O_loc outliers > factor * median
    run_sr: bool = True
    lr_scheduler: object = None  # set after construction


def main():
    # No linalg hooks needed — Slater det uses no SVD/eigh
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # ========== System parameters ==========
        Lx, Ly = 4, 4
        N_sites = Lx * Ly
        t = 1.0
        U = 8.0
        N_f = N_sites - 2  # 2 holes
        n_fermions_per_spin = (N_f // 2, N_f // 2)

        # ========== Hamiltonian ==========
        H = spinful_Fermi_Hubbard_square_lattice_torch(
            Lx,
            Ly,
            t,
            U,
            N_f,
            pbc=False,
            n_fermions_per_spin=n_fermions_per_spin,
            no_u1_symmetry=False,
            gpu=True,
        )
        H.precompute_hops_gpu(device)
        graph = H.graph

        # ========== Variational state (Slater determinant) ==========
        n_orbitals = 2 * N_sites  # spin-up + spin-down orbitals
        model = SlaterDeterminant_GPU(
            n_orbitals=n_orbitals,
            n_fermions=N_f,
            dtype=dtype,
        )
        model.to(device)

        N_params = sum(p.numel() for p in model.parameters())
        if rank == 0:
            print(
                f"SlaterDeterminant: {n_orbitals} orbitals, "
                f"{N_f} fermions, {N_params} params"
            )
            print(
                f"{world_size} GPUs | {device}"
            )

        vmc_cfg = VMCConfig()
        vmc_cfg.lr_scheduler = DecayScheduler(
            init_lr=vmc_cfg.learning_rate,
            decay_rate=0.9, patience=50,
        )

        # ========== Output directory ==========
        output_dir = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/"
            f"t={t}_U={U}/N={N_f}/slater/"
        )
        os.makedirs(output_dir, exist_ok=True)
        model_name = model._get_name()
        
        # Export + compile (optional, ~10-40s one-time cost)
        if vmc_cfg.use_export_compile:
            example_x = random_initial_config(
                N_f, N_sites, seed=0,
            ).to(device)
            if rank == 0:
                print("Running torch.export + compile...")
            model.export_and_compile(example_x, mode='default')

        print_sampling_settings(
            rank,
            world_size,
            vmc_cfg.batch_size,
            vmc_cfg.ns_per_rank,
            vmc_cfg.grad_batch_size,
        )

        # ========== Initialize walkers ==========
        fxs = initialize_walkers(
            init_fn=lambda seed: random_initial_config(
                N_f, N_sites, seed=seed,
            ),
            batch_size=vmc_cfg.batch_size,
            seed=42, rank=rank, device=device,
        )

        # ========== Stats tracking ==========
        step_tag = (
            f'_from{vmc_cfg.resume_step}'
            if vmc_cfg.resume_step > 0 else ''
        )
        stats_file = os.path.join(
            output_dir,
            f'stats_{model_name}{step_tag}.json',
        )
        total_ns = vmc_cfg.ns_per_rank * world_size
        stats = {
            'system': (
                f'{Lx}x{Ly} Fermi-Hubbard, t={t}, U={U}, '
                f'N_f={N_f}, Slater det'
            ),
            'Np': N_params,
            'sample size': total_ns,
            'mean': [],
            'error': [],
            'variance': [],
        }

        # ========== VMC driver ==========
        preconditioner = MinSRGPU()
        vmc = VMC_GPU(
            sampler=MetropolisExchangeSpinfulSamplerGPU(),
            preconditioner=preconditioner,
            optimizer=SGDGPU(
                learning_rate=vmc_cfg.learning_rate,
            ),
        )

        fxs = vmc.run_warmup(
            fxs=fxs,
            model=model,
            graph=graph,
            hamiltonian=H,
            rank=rank,
            config=VMCWarmupConfig(
                use_export_compile=vmc_cfg.use_export_compile,
                grad_batch_size=vmc_cfg.grad_batch_size,
            ),
        )

        # ========== Data-saving callback ==========
        def on_step_end(info):
            if rank != 0:
                return
            stats['mean'].append(info['energy_per_site'])
            stats['error'].append(info['error_per_site'])
            stats['variance'].append(info['energy_var'])
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)

            step = info['step']
            if (step + 1) % vmc_cfg.save_every == 0:
                ckpt_path = os.path.join(
                    output_dir,
                    f'checkpoint_{model_name}_{step + 1}.pt',
                )
                torch.save(model.state_dict(), ckpt_path)

        energy_history, _ = vmc.run_vmc_loop(
            fxs=fxs,
            model=model,
            hamiltonian=H,
            graph=graph,
            rank=rank,
            world_size=world_size,
            config=VMCLoopConfig(
                vmc_steps=vmc_cfg.vmc_steps,
                ns_per_rank=vmc_cfg.ns_per_rank,
                grad_batch_size=vmc_cfg.grad_batch_size,
                n_params=N_params,
                nsites=N_sites,
                learning_rate=vmc_cfg.learning_rate,
                diag_shift=vmc_cfg.diag_shift,
                burn_in_steps=vmc_cfg.burn_in_steps,
                run_sr=vmc_cfg.run_sr,
                use_min_sr=vmc_cfg.use_min_sr,
                use_export_compile=vmc_cfg.use_export_compile,
                step_offset=0,
                debug=vmc_cfg.debug,
                outlier_clip_factor=vmc_cfg.outlier_clip_factor,
                lr_scheduler=vmc_cfg.lr_scheduler,
            ),
            on_step_end=on_step_end,
        )

        # ========== Summary ==========
        if rank == 0 and energy_history:
            print(f"\n{'=' * 50}")
            print(
                f"Result: {Lx}x{Ly} Fermi-Hubbard, t={t}, U={U}, "
                f"N_f={N_f}, Slater determinant"
            )
            print(f"{'=' * 50}")
            print(f"First E/site: {energy_history[0]:.6f}")
            print(f"Last  E/site: {energy_history[-1]:.6f}")
            print(f"Min   E/site: {min(energy_history):.6f}")
            print(f"Stats saved to: {stats_file}")
            if energy_history[-1] < energy_history[0]:
                print("\nEnergy decreased.")
            else:
                print("\nWARNING: Energy did NOT decrease.")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
