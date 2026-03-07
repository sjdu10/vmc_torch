"""Integration test: run a few VMC steps with cheap grad on 4x4.

System: 4x4 spinful Fermi-Hubbard, t=1, U=8, N_f=14, D=4, chi=16.
Compares cheap gradient vs standard gradient for correctness,
then runs 5 VMC steps with cheap grad to verify energy decreases.

Usage:
    torchrun --nproc_per_node=1 GPU/scripts/test_cheap_grad_vmc.py
"""
import json
import os
import time

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
from vmc_torch.GPU.models import fPEPS_Model_reuse_GPU
from vmc_torch.GPU.optimizer import (
    DistributedSRMinresGPU,
    SGDGPU,
)
from vmc_torch.GPU.sampler import (
    MetropolisExchangeSpinfulSamplerGPU,
)
from vmc_torch.GPU.vmc_setup import (
    initialize_walkers,
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.GPU.vmc_utils import (
    compute_grads_cheap_gpu,
    compute_grads_gpu,
    evaluate_energy_reuse,
    random_initial_config,
)

dtype = torch.float64


def main():
    setup_linalg_hooks(
        jitter=1e-8, qr_via_eigh=False,
        cholesky_qr=True,
    )
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # ========== System parameters ==========
        Lx, Ly = 4, 4
        N_sites = Lx * Ly
        t_hop = 1.0
        U = 8.0
        N_f = N_sites - 2
        n_fermions_per_spin = (N_f // 2, N_f // 2)
        D = 4
        chi = 16

        if rank == 0:
            print(f"System: {Lx}x{Ly} Fermi-Hubbard, "
                  f"t={t_hop}, U={U}, N_f={N_f}, "
                  f"D={D}, chi={chi}")

        # ========== Hamiltonian ==========
        H = spinful_Fermi_Hubbard_square_lattice_torch(
            Lx, Ly, t_hop, U, N_f,
            pbc=False,
            n_fermions_per_spin=n_fermions_per_spin,
            no_u1_symmetry=False,
            gpu=True,
        )
        H.precompute_hops_gpu(device)
        graph = H.graph

        # ========== Model ==========
        peps = load_or_generate_peps(
            Lx, Ly, t_hop, U, N_f, D,
            seed=42, dtype=dtype,
            scale_factor=4,
        )
        model = fPEPS_Model_reuse_GPU(
            tn=peps,
            max_bond=chi,
            dtype=dtype,
            contract_boundary_opts={
                'mode': 'mps',
                'canonize': True,
            },
        )
        model.to(device)

        N_params = sum(p.numel() for p in model.parameters())
        if rank == 0:
            print(f"N_params: {N_params}")

        # ========== bMPS skeleton init ==========
        example_x = random_initial_config(
            N_f, N_sites, seed=0,
        ).to(device)
        if rank == 0:
            print("Initializing bMPS skeleton...")
        model.cache_bMPS_skeleton(example_x)

        # ========== Gradient comparison ==========
        B = 4
        fxs = torch.stack([
            random_initial_config(N_f, N_sites, seed=s).to(device)
            for s in range(B)
        ])

        if rank == 0:
            print(f"\n--- Gradient comparison (B={B}) ---")

        # Standard grad
        t0 = time.time()
        with torch.enable_grad():
            grads_std, (signs_std, log_abs_std) = compute_grads_gpu(
                fxs, model,
                vectorize=True, vmap_grad=True,
                use_log_amp=True,
            )
        t_std = time.time() - t0

        # Cheap grad
        t0 = time.time()
        with torch.enable_grad():
            grads_cheap, (signs_cheap, log_abs_cheap) = (
                compute_grads_cheap_gpu(
                    fxs, model, use_log_amp=True,
                )
            )
        t_cheap = time.time() - t0

        per_sample_rel = (
            (grads_std - grads_cheap).norm(dim=1)
            / (grads_std.norm(dim=1) + 1e-30)
        )
        if rank == 0:
            print(f"  Standard grad time: {t_std:.2f}s")
            print(f"  Cheap grad time:    {t_cheap:.2f}s")
            print(f"  Per-sample L2 rel:  "
                  f"mean={per_sample_rel.mean().item():.4e}, "
                  f"max={per_sample_rel.max().item():.4e}")
            print(f"  Speedup: {t_std / t_cheap:.2f}x")
        del grads_std, grads_cheap

        # ========== VMC run with cheap grad ==========
        B_vmc = 16
        ns_per_rank = 16
        grad_batch_size = 16
        vmc_steps = 5

        if rank == 0:
            print(f"\n--- VMC run ({vmc_steps} steps, "
                  f"B={B_vmc}, Ns={ns_per_rank}) ---")

        fxs = initialize_walkers(
            init_fn=lambda seed: random_initial_config(
                N_f, N_sites, seed=seed,
            ),
            batch_size=B_vmc,
            seed=42, rank=rank, device=device,
        )

        preconditioner = DistributedSRMinresGPU(
            rtol=1e-4, maxiter=100,
        )
        vmc = VMC_GPU(
            sampler=MetropolisExchangeSpinfulSamplerGPU(),
            preconditioner=preconditioner,
            optimizer=SGDGPU(learning_rate=0.1),
            evaluate_energy_fn=evaluate_energy_reuse,
            compute_grads_fn=compute_grads_cheap_gpu,
        )

        # Warmup
        fxs = vmc.run_warmup(
            fxs=fxs,
            model=model,
            graph=graph,
            hamiltonian=H,
            rank=rank,
            config=VMCWarmupConfig(
                grad_batch_size=grad_batch_size,
                use_log_amp=True,
            ),
        )

        # VMC loop
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
                burn_in_steps=2,
                run_sr=True,
                use_log_amp=True,
                show_progress=True,
                debug=True,
            ),
        )

        if rank == 0 and energy_history:
            print(f"\nFirst E/site: {energy_history[0]:.6f}")
            print(f"Last  E/site: {energy_history[-1]:.6f}")
            if energy_history[-1] < energy_history[0]:
                print("Energy DECREASED — OK")
            else:
                print("WARNING: Energy did NOT decrease")

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
