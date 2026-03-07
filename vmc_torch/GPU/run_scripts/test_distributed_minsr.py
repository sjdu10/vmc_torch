"""Correctness test: distributed_minSR_solver_gpu vs minSR_solver_gpu.

Tests both GPU-resident and CPU-resident (offloaded) local_O paths.

Run:
    torchrun --nproc_per_node=1 run_scripts/test_distributed_minsr.py
    torchrun --nproc_per_node=2 run_scripts/test_distributed_minsr.py
"""
import torch
import torch.distributed as dist
import numpy as np

from vmc_torch.GPU.VMC import setup_distributed
from vmc_torch.GPU.hamiltonian import (
    spinful_Fermi_Hubbard_square_lattice_torch,
)
from vmc_torch.GPU.models import (
    Conv2D_Geometric_fPEPS_GPU,
)
from vmc_torch.GPU.vmc_modules import (
    run_sampling_phase_gpu,
    minSR_solver_gpu,
    distributed_minSR_solver_gpu,
)
from vmc_torch.GPU.vmc_setup import (
    initialize_walkers,
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.GPU.vmc_utils import (
    random_initial_config,
    sample_next,
)

dtype = torch.float64
CPU_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/experiment/vmap/data'
)


def compare_dp(dp_a, dp_b, device):
    """Convert both to GPU tensors and return rel diff."""
    if isinstance(dp_a, np.ndarray):
        dp_a = torch.tensor(dp_a, device=device)
    else:
        dp_a = dp_a.to(device)
    if isinstance(dp_b, np.ndarray):
        dp_b = torch.tensor(dp_b, device=device)
    else:
        dp_b = dp_b.to(device)
    norm_a = torch.norm(dp_a).item()
    diff = torch.norm(dp_a - dp_b).item()
    rel = diff / (norm_a + 1e-30)
    return norm_a, torch.norm(dp_b).item(), diff, rel


def main():
    setup_linalg_hooks(jitter=1e-12, qr_via_eigh=True)
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # ========== System (same as vmc_run_nnfpeps_4x4) ==========
        Lx, Ly = 4, 4
        N_sites = Lx * Ly
        t, U = 1.0, 8.0
        N_f = N_sites - 2
        n_fermions_per_spin = (N_f // 2, N_f // 2)
        D, chi = 4, -1

        H = spinful_Fermi_Hubbard_square_lattice_torch(
            Lx, Ly, t, U, N_f,
            pbc=False,
            n_fermions_per_spin=n_fermions_per_spin,
            no_u1_symmetry=False,
            gpu=True,
        )
        H.precompute_hops_gpu(device)
        graph = H.graph

        fpeps_base = (
            f"{CPU_DATA_ROOT}/{Lx}x{Ly}/t={t}_U={U}"
            f"/N={N_f}/Z2/D={D}/"
        )
        peps = load_or_generate_peps(
            Lx, Ly, t, U, N_f, D,
            seed=42, dtype=dtype,
            file_path=fpeps_base,
            scale_factor=4,
        )
        model = Conv2D_Geometric_fPEPS_GPU(
            tn=peps, max_bond=chi,
            nn_eta=1.0, embed_dim=16,
            hidden_dim=N_sites, kernel_size=3,
            layers=1, init_scale=1e-5,
            dtype=dtype, backbone_dtype=torch.float64,
            contract_boundary_opts={
                'mode': 'mps',
                'equalize_norms': 1.0,
                'canonize': True,
            },
        )
        model.to(device)
        N_params = sum(p.numel() for p in model.parameters())

        if rank == 0:
            print(f"System: {Lx}x{Ly} FH, t={t}, U={U}, "
                  f"N_f={N_f}, D={D}, chi={chi}")
            print(f"Model: {model._get_name()} | {N_params} params")
            print(f"GPUs: {world_size}")

        # ========== Sampling ==========
        Ns_per_rank = 64
        B = 64
        fxs = initialize_walkers(
            init_fn=lambda seed: random_initial_config(
                N_f, N_sites, seed=seed,
            ),
            batch_size=B,
            seed=42, rank=rank, device=device,
        )

        if rank == 0:
            print("Burn-in...")
        for _ in range(4):
            fxs, _ = sample_next(fxs, model, graph)

        if rank == 0:
            print(f"Sampling {Ns_per_rank} configs per rank...")
        (local_energies, local_O), fxs, _, _ = run_sampling_phase_gpu(
            fxs, model, H, graph,
            Ns=Ns_per_rank,
            grad_batch_size=16,
            verbose=(rank == 0),
        )

        total_ns = Ns_per_rank * world_size
        if rank == 0:
            print(f"local_O: {local_O.shape} on {local_O.device}")
            print(f"Total samples: {total_ns}")

        # ========== Compute energy_mean ==========
        local_e_sum = local_energies.sum()
        if world_size > 1:
            dist.all_reduce(local_e_sum, op=dist.ReduceOp.SUM)
        energy_mean = local_e_sum.item() / total_ns

        if rank == 0:
            print(f"Energy mean: {energy_mean:.6f}")

        diag_shift = 1e-4

        # ========== Reference: old minSR (GPU gather) ==========
        dp_ref, t_ref, info_ref = minSR_solver_gpu(
            local_lpg=local_O.clone(),
            local_energies=local_energies.clone(),
            energy_mean=energy_mean,
            total_samples=total_ns,
            n_params=N_params,
            diag_shift=diag_shift,
            device=device,
            do_SR=True,
        )

        # ========== Test 1: GPU-resident local_O ==========
        if rank == 0:
            print("\n=== Test 1: GPU-resident local_O ===")

        for chunk_size in [32, 128, 1024]:
            dp_new, t_new, info_new = (
                distributed_minSR_solver_gpu(
                    local_lpg=local_O.clone(),
                    local_energies=local_energies.clone(),
                    energy_mean=energy_mean,
                    total_samples=total_ns,
                    n_params=N_params,
                    diag_shift=diag_shift,
                    param_chunk_size=chunk_size,
                    device=device,
                    do_SR=True,
                )
            )
            na, nb, diff, rel = compare_dp(
                dp_ref, dp_new, device,
            )
            if rank == 0:
                status = "PASS" if rel < 1e-8 else "FAIL"
                print(
                    f"  C={chunk_size}: rel_diff={rel:.2e} "
                    f"t={t_new:.4f}s  {status}"
                )

        # ========== Test 2: CPU-resident local_O ==========
        if rank == 0:
            print("\n=== Test 2: CPU-offloaded local_O ===")

        local_O_cpu = local_O.cpu()
        local_E_cpu = local_energies.cpu()

        for chunk_size in [32, 128, 1024]:
            dp_cpu, t_cpu, info_cpu = (
                distributed_minSR_solver_gpu(
                    local_lpg=local_O_cpu.clone(),
                    local_energies=local_E_cpu.clone(),
                    energy_mean=energy_mean,
                    total_samples=total_ns,
                    n_params=N_params,
                    diag_shift=diag_shift,
                    param_chunk_size=chunk_size,
                    device=device,
                    do_SR=True,
                )
            )
            na, nb, diff, rel = compare_dp(
                dp_ref, dp_cpu, device,
            )
            if rank == 0:
                status = "PASS" if rel < 1e-8 else "FAIL"
                print(
                    f"  C={chunk_size}: rel_diff={rel:.2e} "
                    f"t={t_cpu:.4f}s  {status}"
                )

        # ========== Test 3: do_SR=False, CPU path ==========
        if rank == 0:
            print("\n=== Test 3: gradient-only, CPU vs GPU ===")

        dp_grad_gpu, _, _ = distributed_minSR_solver_gpu(
            local_lpg=local_O.clone(),
            local_energies=local_energies.clone(),
            energy_mean=energy_mean,
            total_samples=total_ns,
            n_params=N_params,
            device=device,
            do_SR=False,
        )
        dp_grad_cpu, _, _ = distributed_minSR_solver_gpu(
            local_lpg=local_O_cpu.clone(),
            local_energies=local_E_cpu.clone(),
            energy_mean=energy_mean,
            total_samples=total_ns,
            n_params=N_params,
            device=device,
            do_SR=False,
        )
        na, nb, diff, rel = compare_dp(
            dp_grad_gpu, dp_grad_cpu, device,
        )
        if rank == 0:
            status = "PASS" if rel < 1e-10 else "FAIL"
            print(f"  rel_diff={rel:.2e}  {status}")
            print("\n=== All tests done. ===")

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
