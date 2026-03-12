"""GPU VMC for spin-1/2 Heisenberg model on a square lattice.

Run:
    torchrun --nproc_per_node=<N> vmc_run_spin.py
    torchrun --nproc_per_node=1 vmc_run_spin.py   # single GPU
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
from vmc_torch.hamiltonian_torch import (
    spin_Heisenberg_square_lattice_torch,
)
from vmc_torch.GPU.models import PEPS_Model_GPU
from vmc_torch.GPU.optimizer import DecayScheduler, SGDGPU
from vmc_torch.GPU.sampler import (
    MetropolisExchangeSpinSamplerGPU,
)
from vmc_torch.GPU.vmc_setup import (
    generate_random_spin_peps,
    initialize_walkers,
    random_spin_config_sz0,
    setup_linalg_hooks,
)
from vmcconfig import (
    VMCConfig,
    make_on_step_end,
    make_preconditioner,
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
    vmc_steps=1,
    burn_in_steps=1,
    learning_rate=0.1,
    sr_diag_shift=5e-4,
    use_distributed_sr_minres=True,
    sr_rtol=1e-4,
    offload_grad_to_cpu=True,
    use_log_amp=True,
    use_export_compile=False,
    save_every=10,
    resume_step=0,
    verbose=False,
)
vmc_cfg.lr_scheduler = DecayScheduler(
    init_lr=vmc_cfg.learning_rate,
    decay_rate=0.9, patience=50,
)

warmup_cfg = VMCWarmupConfig(
    use_export_compile=vmc_cfg.use_export_compile,
    grad_batch_size=vmc_cfg.grad_batch_size,
    use_log_amp=vmc_cfg.use_log_amp,
    offload_grad_to_cpu=vmc_cfg.offload_grad_to_cpu,
    run_sampling=True,
    run_locE=False,
    run_grad=False,
)


def main():
    setup_linalg_hooks(
        jitter=1e-8, qr_via_eigh=True,
        cholesky_qr=False, cholesky_qr_adaptive_jitter=False,
    )
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # ========== System parameters ==========
        Lx, Ly = 8, 8
        N_sites = Lx * Ly
        J = 1.0
        D = 6
        chi = 10

        # ========== Hamiltonian ==========
        H = spin_Heisenberg_square_lattice_torch(
            Lx, Ly, J=J, total_sz=0,
        )
        H.precompute_hops_gpu(device)

        # ED reference for small systems
        if rank == 0 and N_sites <= 16:
            H_dense = H.to_dense()
            import scipy.sparse.linalg as la
            gs_e = la.eigsh(
                H_dense, k=1, which='SA', tol=1e-8,
            )[0][0]
            print(
                f"ED ground state E/site: "
                f"{gs_e / N_sites:.8f}"
            )

        # ========== Model ==========
        peps = generate_random_spin_peps(
            Lx, Ly, D, seed=42, dtype=dtype,
        )
        model = PEPS_Model_GPU(
            tn=peps,
            max_bond=chi,
            dtype=dtype,
            contract_boundary_opts={
                'mode': 'mps',
                'equalize_norms': 1.0,
                'canonize': True,
                # 'compress_opts': {'seed': 42},
            },
        )
        model.to(device)

        # ========== Setup ==========
        output_dir = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/"
            f"J={J}/D={D}/chi={chi}/"
        )
        os.makedirs(output_dir, exist_ok=True)
        model_name = model._get_name()
        N_params = sum(
            p.numel() for p in model.parameters()
        )

        if rank == 0:
            print(
                f"System: {Lx}x{Ly} Heisenberg, "
                f"J={J}, Sz=0"
            )
            print(
                f"Model: PEPS D={D}, chi={chi}, "
                f"{N_params} params | "
                f"{world_size} GPUs | {device}"
            )

        # ========== Export + compile ==========
        if vmc_cfg.use_export_compile:
            example_x = random_spin_config_sz0(
                N_sites, seed=0,
            ).to(device)
            if rank == 0:
                print("Running torch.export + compile...")
            model.export_and_compile(
                example_x, mode='default',
                use_log_amp=vmc_cfg.use_log_amp,
            )

        print_sampling_settings(
            rank, world_size, vmc_cfg.batch_size,
            vmc_cfg.ns_per_rank, vmc_cfg.grad_batch_size,
        )

        # ========== Initialize walkers ==========
        fxs = initialize_walkers(
            init_fn=lambda seed: random_spin_config_sz0(
                N_sites, seed=seed,
            ),
            batch_size=vmc_cfg.batch_size,
            seed=42, rank=rank, device=device,
        )

        # ========== Stats + callback ==========
        system_str = (
            f'{Lx}x{Ly} Heisenberg, J={J}, '
            f'D={D}, chi={chi}'
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
        vmc = VMC_GPU(
            sampler=MetropolisExchangeSpinSamplerGPU(),
            preconditioner=make_preconditioner(vmc_cfg),
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
        energy_history, fxs = vmc.run_vmc_loop(
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
