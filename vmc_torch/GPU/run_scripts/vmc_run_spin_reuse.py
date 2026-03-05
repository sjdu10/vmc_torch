"""GPU VMC for spin-1/2 Heisenberg with bMPS reuse.

Uses PEPS_Model_reuse_GPU with cached boundary MPS environments
for incremental updates during sampling.

Run:
    torchrun --nproc_per_node=<N> vmc_run_spin_reuse.py
    torchrun --nproc_per_node=1 vmc_run_spin_reuse.py
"""
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
from vmc_torch.GPU.vmc_utils import (
    evaluate_energy_reuse,
)
from vmc_torch.hamiltonian_torch import (
    spin_Heisenberg_square_lattice_torch,
)
from vmc_torch.GPU.models import (
    PEPS_Model_reuse_GPU,
)
from vmc_torch.GPU.optimizer import (
    DistributedSRMinresGPU,
    SGDGPU,
)
from vmc_torch.GPU.sampler import (
    MetropolisExchangeSpinSamplerReuse_GPU,
)
from vmc_torch.GPU.vmc_setup import (
    generate_random_spin_peps,
    initialize_walkers,
    random_spin_config_sz0,
    setup_linalg_hooks,
)

dtype = torch.float64


@dataclass
class VMCConfig:
    """VMC numerical / training settings."""

    batch_size: int = 128
    ns_per_rank: int = 128
    grad_batch_size: int = 64
    vmc_steps: int = 50
    learning_rate: float = 0.1
    diag_shift: float = 1e-4
    burn_in_steps: int = 0
    use_export_compile: bool = False
    sr_rtol: float = 1e-4
    sr_maxiter: int = 100
    use_scipy: bool = False


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
        D = 4  # PEPS bond dimension
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

        # ========== Model (PEPS with reuse) ==========
        peps = generate_random_spin_peps(
            Lx, Ly, D, seed=42, dtype=dtype,
        )
        model = PEPS_Model_reuse_GPU(
            tn=peps,
            max_bond=chi,
            dtype=dtype,
            contract_boundary_opts={
                'mode': 'mps',
                # 'equalize_norms': 1.0,
                'canonize': True,
            },
        )
        model.to(device)
        N_params = sum(p.numel() for p in model.parameters())

        # bMPS skeleton init (one-time)
        example_x = random_spin_config_sz0(
            N_sites, seed=0,
        ).to(device)
        model.cache_bMPS_skeleton(example_x)

        if rank == 0:
            print(
                f"System: {Lx}x{Ly} Heisenberg, J={J}, "
                f"Sz=0"
            )
            print(
                f"Model: PEPS_Model_reuse_GPU D={D}, "
                f"chi={chi}, {N_params} params | "
                f"{world_size} GPUs | {device}"
            )

        # ========== VMC settings ==========
        vmc_cfg = VMCConfig()

        # Export + compile (optional, full contraction only)
        if vmc_cfg.use_export_compile:
            if rank == 0:
                print("Running torch.export + compile...")
            model.export_and_compile(
                example_x, mode='default',
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
            sampler=MetropolisExchangeSpinSamplerReuse_GPU(),
            preconditioner=DistributedSRMinresGPU(
                rtol=vmc_cfg.sr_rtol,
                maxiter=vmc_cfg.sr_maxiter,
                use_scipy=vmc_cfg.use_scipy,
            ),
            optimizer=SGDGPU(
                learning_rate=vmc_cfg.learning_rate,
            ),
            evaluate_energy_fn=evaluate_energy_reuse,
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

        energy_history, fxs = vmc.run_vmc_loop(
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
                run_sr=True,
                use_export_compile=vmc_cfg.use_export_compile,
            ),
        )

        # ========== Summary ==========
        if rank == 0 and energy_history:
            print(f"\n{'=' * 50}")
            print(
                f"Result: {Lx}x{Ly} Heisenberg, J={J}, "
                f"D={D}, chi={chi} (reuse)"
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
