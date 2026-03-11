"""GPU VMC for Slater determinant on spinful Fermi-Hubbard.

Mean-field variational ansatz (no correlations).
Useful as a baseline / sanity check.

Run:
    torchrun --nproc_per_node=<N> run_scripts/vmc_run_slater.py
    torchrun --nproc_per_node=1 run_scripts/vmc_run_slater.py
"""
import os

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
from vmc_torch.GPU.vmc_setup import initialize_walkers
from vmc_torch.GPU.vmc_utils import random_initial_config
from vmcconfig import (
    VMCConfig,
    load_checkpoint,
    make_on_step_end,
    make_stats,
    make_stats_file,
    print_summary,
)

dtype = torch.float64
DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/GPU/data'
)

vmc_cfg = VMCConfig(
    batch_size=2048,
    ns_per_rank=2048,
    grad_batch_size=1024,
    vmc_steps=1000,
    burn_in_steps=1,
    learning_rate=0.1,
    sr_diag_shift=5e-4,
    use_distributed_sr_minres=True,
    sr_rtol=1e-4,
    offload_grad_to_cpu=True,
    use_log_amp=True,
    use_export_compile=True,
    save_every=10,
    resume_step=0,
    verbose=False,
)
vmc_cfg.lr_scheduler = DecayScheduler(
    init_lr=vmc_cfg.learning_rate,
    decay_rate=0.9, patience=50,
)


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
        t, U = 1.0, 8.0
        N_f = N_sites - 2
        n_fermions_per_spin = (N_f // 2, N_f // 2)

        # ========== Hamiltonian ==========
        H = spinful_Fermi_Hubbard_square_lattice_torch(
            Lx, Ly, t, U, N_f,
            pbc=False,
            n_fermions_per_spin=n_fermions_per_spin,
            no_u1_symmetry=False,
            gpu=True,
        )
        H.precompute_hops_gpu(device)

        # ========== Model ==========
        n_orbitals = 2 * N_sites
        model = SlaterDeterminant_GPU(
            n_orbitals=n_orbitals,
            n_fermions=N_f,
            dtype=dtype,
        )
        model.to(device)

        # ========== Setup ==========
        output_dir = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/"
            f"t={t}_U={U}/N={N_f}/slater/"
        )
        os.makedirs(output_dir, exist_ok=True)
        model_name = model._get_name()
        N_params = sum(
            p.numel() for p in model.parameters()
        )

        if rank == 0:
            print(
                f"SlaterDeterminant: {n_orbitals} orbitals,"
                f" {N_f} fermions, {N_params} params"
            )
            print(f"{world_size} GPUs | {device}")

        # ========== Export + compile ==========
        if vmc_cfg.use_export_compile:
            example_x = random_initial_config(
                N_f, N_sites, seed=0,
            ).to(device)
            if rank == 0:
                print("Running torch.export + compile...")
            model.export_and_compile(
                example_x, mode='default',
                use_log_amp=vmc_cfg.use_log_amp,
            )

        load_checkpoint(
            model, output_dir, model_name,
            vmc_cfg.resume_step, device, rank,
        )

        print_sampling_settings(
            rank, world_size, vmc_cfg.batch_size,
            vmc_cfg.ns_per_rank, vmc_cfg.grad_batch_size,
        )

        # ========== Initialize walkers ==========
        fxs = initialize_walkers(
            init_fn=lambda seed: random_initial_config(
                N_f, N_sites, seed=seed,
            ),
            batch_size=vmc_cfg.batch_size,
            seed=42, rank=rank, device=device,
        )

        # ========== Stats + callback ==========
        system_str = (
            f'{Lx}x{Ly} Fermi-Hubbard, t={t}, U={U}, '
            f'N_f={N_f}, Slater det'
        )
        stats_file = make_stats_file(
            output_dir, model_name,
            vmc_cfg.resume_step,
        )
        stats = make_stats(
            system_str, N_params,
            vmc_cfg.ns_per_rank, world_size,
        )
        on_step_end = make_on_step_end(
            rank, stats, stats_file, output_dir,
            model_name, model, vmc_cfg.save_every,
        )

        # ========== VMC driver ==========
        # Slater det: small Np, use MinSR directly
        vmc = VMC_GPU(
            sampler=MetropolisExchangeSpinfulSamplerGPU(),
            preconditioner=MinSRGPU(),
            optimizer=SGDGPU(
                learning_rate=vmc_cfg.learning_rate,
            ),
        )

        fxs = vmc.run_warmup(
            fxs=fxs, model=model, graph=H.graph,
            hamiltonian=H, rank=rank,
            config=VMCWarmupConfig(
                use_export_compile=vmc_cfg.use_export_compile,
                grad_batch_size=vmc_cfg.grad_batch_size,
                use_log_amp=vmc_cfg.use_log_amp,
            ),
        )
        energy_history, _ = vmc.run_vmc_loop(
            fxs=fxs, model=model, hamiltonian=H,
            graph=H.graph, rank=rank,
            world_size=world_size,
            config=VMCLoopConfig.from_vmc_config(
                vmc_cfg,
                n_params=N_params,
                nsites=N_sites,
            ),
            on_step_end=on_step_end,
        )

        print_summary(
            rank, energy_history,
            system_str, stats_file,
        )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
