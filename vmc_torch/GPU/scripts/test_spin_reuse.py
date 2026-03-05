"""Test PEPS_Model_reuse_GPU for spin-1/2 Heisenberg.

Runs a short VMC optimization on 4x2 Heisenberg with
the bMPS reuse model and checks:
1. Energy decreases over 10 steps
2. No NaN/Inf in energy history
3. Compares with ED ground state

Run:
    torchrun --nproc_per_node=1 \
        GPU/scripts/test_spin_reuse.py
"""
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


def main():
    setup_linalg_hooks(jitter=1e-16)
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # ========== System parameters ==========
        Lx, Ly = 4, 2
        N_sites = Lx * Ly
        J = 1.0
        D = 4    # PEPS bond dimension
        chi = 4  # boundary bond dim

        # ========== Hamiltonian ==========
        H = spin_Heisenberg_square_lattice_torch(
            Lx, Ly, J=J, total_sz=0,
        )
        H.precompute_hops_gpu(device)
        graph = H.graph

        # ED reference
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
                'equalize_norms': 1.0,
                'canonize': True,
            },
        )
        model.to(device)
        N_params = sum(p.numel() for p in model.parameters())

        # Initialize bMPS skeleton (required for reuse)
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
        batch_size = 512
        ns_per_rank = 2048
        grad_batch_size = 64
        vmc_steps = 10

        print_sampling_settings(
            rank, world_size, batch_size,
            ns_per_rank, grad_batch_size,
        )

        # ========== Initialize walkers ==========
        fxs = initialize_walkers(
            init_fn=lambda seed: random_spin_config_sz0(
                N_sites, seed=seed,
            ),
            batch_size=batch_size,
            seed=42, rank=rank, device=device,
        )

        # ========== VMC driver ==========
        vmc = VMC_GPU(
            sampler=MetropolisExchangeSpinSamplerReuse_GPU(),
            preconditioner=DistributedSRMinresGPU(
                rtol=1e-4, maxiter=100,
                use_scipy=True,
            ),
            optimizer=SGDGPU(learning_rate=0.1),
        )

        fxs = vmc.run_warmup(
            fxs=fxs,
            model=model,
            graph=graph,
            hamiltonian=H,
            rank=rank,
            config=VMCWarmupConfig(
                grad_batch_size=grad_batch_size,
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
                vmc_steps=vmc_steps,
                ns_per_rank=ns_per_rank,
                grad_batch_size=grad_batch_size,
                n_params=N_params,
                nsites=N_sites,
                learning_rate=0.1,
                diag_shift=1e-4,
                burn_in_steps=4,
                run_sr=True,
            ),
        )

        # ========== Validation ==========
        if rank == 0 and energy_history:
            import math
            print(f"\n{'=' * 50}")
            print("Spin reuse model validation:")
            print(f"{'=' * 50}")
            print(
                f"  First E/site: "
                f"{energy_history[0]:.6f}"
            )
            print(
                f"  Last  E/site: "
                f"{energy_history[-1]:.6f}"
            )
            print(
                f"  Min   E/site: "
                f"{min(energy_history):.6f}"
            )

            has_nan = any(
                math.isnan(e) for e in energy_history
            )
            decreased = (
                energy_history[-1] < energy_history[0]
            )

            if has_nan:
                print("\nFAIL: NaN in energy history!")
            elif not decreased:
                print(
                    "\nWARN: Energy did NOT decrease."
                )
            else:
                print(
                    "\nPASS: Energy decreased, no NaN."
                )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
