"""GPU VMC for spin-1/2 Heisenberg model on a square lattice.

Run:
    torchrun --nproc_per_node=<N> vmc_run_spin.py
    torchrun --nproc_per_node=1 vmc_run_spin.py   # single GPU
"""
import numpy as np
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
from vmc_torch.GPU.models import fPEPS_Model_GPU
from vmc_torch.GPU.optimizer import (
    DistributedSRMinresGPU,
    SGDGPU,
)
from vmc_torch.GPU.sampler import (
    MetropolisExchangeSpinSamplerGPU,
)
from vmc_torch.GPU.vmc_setup import (
    generate_random_spin_peps,
    initialize_walkers,
    random_spin_config_sz0,
    setup_linalg_hooks,
)
from vmcconfig import VMCConfig

dtype = torch.float64

vmc_cfg = vmc_cfg = VMCConfig(
    batch_size=2048,
    ns_per_rank=2048,
    grad_batch_size=1024,
    vmc_steps=1000,
    burn_in_steps=1,
    learning_rate=0.1,
    sr_diag_shift=5e-4,
    use_distributed_sr_minres=True,
    sr_tol=1e-4,
    offload_grad_to_cpu=True,
    use_log_amp=True,
    use_export_compile=True,
    save_every=10,
    resume_step=0,
    verbose=False,
)


def main():
    setup_linalg_hooks(jitter=1e-16)
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # ========== System parameters ==========
        Lx, Ly = 8, 8
        N_sites = Lx * Ly
        J = 1.0
        D = 4    # PEPS bond dimension
        chi = 16  # boundary bond dim

        # ========== Hamiltonian ==========
        H = spin_Heisenberg_square_lattice_torch(
            Lx, Ly, J=J, total_sz=0,
        )
        H.precompute_hops_gpu(device)
        graph = H.graph

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

        # ========== Model (PEPS) ==========
        peps = generate_random_spin_peps(
            Lx, Ly, D, seed=42, dtype=dtype,
        )
        model = fPEPS_Model_GPU(
            tn=peps,
            max_bond=chi,
            dtype=dtype,
            contract_boundary_opts={
                'mode': 'mps',
                'equalize_norms': 1.0,
                'canonize': True,
            },
        )
        model.to(device)
        N_params = sum(p.numel() for p in model.parameters())
        if rank == 0:
            print(
                f"System: {Lx}x{Ly} Heisenberg, J={J}, "
                f"Sz=0"
            )
            print(
                f"Model: PEPS D={D}, chi={chi}, "
                f"{N_params} params | "
                f"{world_size} GPUs | {device}"
            )

        # Export + compile (optional)
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
            rank,
            world_size,
            vmc_cfg.batch_size,
            vmc_cfg.ns_per_rank,
            vmc_cfg.grad_batch_size,
        )

        # ========== Initialize walkers ==========
        fxs = initialize_walkers(
            init_fn=lambda seed: random_spin_config_sz0(
                N_sites, seed=seed,
            ),
            batch_size=vmc_cfg.batch_size,
            seed=42, rank=rank, device=device,
        )

        # ========== VMC driver ==========
        vmc = VMC_GPU(
            sampler=MetropolisExchangeSpinSamplerGPU(),
            preconditioner=DistributedSRMinresGPU(
                rtol=vmc_cfg.sr_rtol,
                maxiter=vmc_cfg.sr_maxiter,
                use_scipy=vmc_cfg.minres_sr_use_scipy,
            ),
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
                use_log_amp=vmc_cfg.use_log_amp,
            ),
        )

        energy_history, fxs = vmc.run_vmc_loop(
            fxs=fxs,
            model=model,
            hamiltonian=H,
            graph=graph,
            rank=rank,
            world_size=world_size,
            config=VMCLoopConfig.from_vmc_config(
                vmc_cfg,
                n_params=N_params,
                nsites=N_sites,
            ),
        )

        # ========== Summary ==========
        if rank == 0 and energy_history:
            print(f"\n{'=' * 50}")
            print(
                f"Result: {Lx}x{Ly} Heisenberg, J={J}, "
                f"D={D}, chi={chi}"
            )
            print(f"{'=' * 50}")
            print(f"First E/site: {energy_history[0]:.6f}")
            print(f"Last  E/site: {energy_history[-1]:.6f}")
            print(f"Min   E/site: {min(energy_history):.6f}")
            if energy_history[-1] < energy_history[0]:
                print("\nEnergy decreased.")
            else:
                print("\nWARNING: Energy did NOT decrease.")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
