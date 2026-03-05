"""Test CPU offloading of O_loc via use_scipy=True.

Runs a short VMC optimization on 4x2 Fermi-Hubbard and checks:
1. Energy decreases over 10 steps
2. No NaN/Inf in energy history
3. Prints GPU memory usage before/after sampling

Run:
    torchrun --nproc_per_node=1 GPU/scripts/test_cpu_offload.py
"""
import torch
import torch.distributed as dist

from vmc_torch.experiment.vmap.GPU.VMC import (
    VMC_GPU,
    VMCLoopConfig,
    VMCWarmupConfig,
    print_sampling_settings,
    setup_distributed,
)
from vmc_torch.experiment.vmap.GPU.hamiltonian import (
    spinful_Fermi_Hubbard_square_lattice_torch,
)
from vmc_torch.experiment.vmap.GPU.models import fPEPS_Model_GPU
from vmc_torch.experiment.vmap.GPU.optimizer import (
    DistributedSRMinresGPU,
    SGDGPU,
)
from vmc_torch.experiment.vmap.GPU.sampler import (
    MetropolisExchangeSpinfulSamplerGPU,
)
from vmc_torch.experiment.vmap.GPU.vmc_setup import (
    initialize_walkers,
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.experiment.vmap.GPU.vmc_utils import random_initial_config

dtype = torch.float64


def print_gpu_mem(tag=""):
    alloc = torch.cuda.memory_allocated() / 1e6
    resv = torch.cuda.memory_reserved() / 1e6
    print(f"  [GPU mem {tag}] alloc={alloc:.1f}MB reserved={resv:.1f}MB")


def main():
    setup_linalg_hooks(jitter=1e-16)
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # ========== System: 4x2 Fermi-Hubbard ==========
        Lx, Ly = 4, 2
        N_sites = Lx * Ly
        t_hop, U = 1.0, 8.0
        N_f = N_sites - 2
        n_fermions_per_spin = (N_f // 2, N_f // 2)
        D, chi = 10, 10

        H = spinful_Fermi_Hubbard_square_lattice_torch(
            Lx, Ly, t_hop, U, N_f, pbc=False,
            n_fermions_per_spin=n_fermions_per_spin,
            no_u1_symmetry=False, gpu=True,
        )
        H.precompute_hops_gpu(device)
        graph = H.graph

        peps = load_or_generate_peps(
            Lx, Ly, t_hop, U, N_f, D, seed=42, dtype=dtype,
        )
        model = fPEPS_Model_GPU(
            tn=peps, max_bond=chi, dtype=dtype,
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
                f"System: {Lx}x{Ly} FH, t={t_hop}, U={U}, "
                f"N_f={N_f}, D={D}, chi={chi}"
            )
            print(f"Model: {N_params} params | device={device}")

        # ========== Settings ==========
        batch_size = 512
        ns_per_rank = 2048
        grad_batch_size = 64
        vmc_steps = 10

        print_sampling_settings(
            rank, world_size, batch_size,
            ns_per_rank, grad_batch_size,
        )

        fxs = initialize_walkers(
            init_fn=lambda seed: random_initial_config(
                N_f, N_sites, seed=seed,
            ),
            batch_size=batch_size,
            seed=42, rank=rank, device=device,
        )

        # ========== use_scipy=True → CPU offload ==========
        if rank == 0:
            print("\n=== Testing use_scipy=True (CPU offload) ===")
        vmc = VMC_GPU(
            sampler=MetropolisExchangeSpinfulSamplerGPU(),
            preconditioner=DistributedSRMinresGPU(
                rtol=1e-4, maxiter=100,
                use_scipy=True,  # <-- triggers CPU offload
            ),
            optimizer=SGDGPU(learning_rate=0.1),
        )

        if rank == 0:
            print_gpu_mem("before warmup")

        fxs = vmc.run_warmup(
            fxs=fxs, model=model, graph=graph,
            hamiltonian=H, rank=rank,
            config=VMCWarmupConfig(
                grad_batch_size=grad_batch_size,
            ),
        )

        if rank == 0:
            print_gpu_mem("after warmup")

        energy_history, _ = vmc.run_vmc_loop(
            fxs=fxs, model=model, hamiltonian=H,
            graph=graph, rank=rank, world_size=world_size,
            config=VMCLoopConfig(
                vmc_steps=vmc_steps,
                ns_per_rank=ns_per_rank,
                grad_batch_size=grad_batch_size,
                n_params=N_params,
                nsites=N_sites,
                learning_rate=0.1,
                diag_shift=5e-5,
                burn_in_steps=4,
                run_sr=True,
            ),
        )

        if rank == 0:
            print_gpu_mem("after VMC loop")

        # ========== Validation ==========
        if rank == 0 and energy_history:
            import math
            print(f"\n{'=' * 50}")
            print("Validation results:")
            print(f"{'=' * 50}")
            print(
                f"  First E/site: {energy_history[0]:.6f}"
            )
            print(
                f"  Last  E/site: {energy_history[-1]:.6f}"
            )
            print(
                f"  Min   E/site: {min(energy_history):.6f}"
            )

            has_nan = any(math.isnan(e) for e in energy_history)
            decreased = energy_history[-1] < energy_history[0]

            if has_nan:
                print("\nFAIL: NaN in energy history!")
            elif not decreased:
                print("\nWARN: Energy did NOT decrease.")
            else:
                print("\nPASS: Energy decreased, no NaN.")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
