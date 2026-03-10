"""Test randomized SVD vs standard SVD amplitude consistency.

Compares forward_log output using svd_truncated (standard)
vs svd_rand_truncated (randomized) for fPEPS_Model_GPU.

Usage:
    torchrun --nproc_per_node=1 test_rand_svd_correctness.py
"""
import time

import autoray as ar
import torch
import torch.distributed as dist

from symmray.linalg import svd_rand_truncated, svd_truncated
from vmc_torch.GPU.VMC import setup_distributed
from vmc_torch.GPU.models import fPEPS_Model_GPU
from vmc_torch.GPU.vmc_setup import (
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.GPU.vmc_utils import random_initial_config

dtype = torch.float64
DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/GPU/data'
)


def main():
    setup_linalg_hooks(
        jitter=1e-8, qr_via_eigh=False,
        cholesky_qr=True, cholesky_qr_adaptive_jitter=False,
        nonuniform_diag=True,
    )
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42)

        # System params (same as vmc_run_fpeps.py)
        Lx, Ly = 4, 4
        N_sites = Lx * Ly
        t, U = 1.0, 8.0
        N_f = N_sites - 2
        D = 4
        chi = 8
        B = 32

        print(f"System: {Lx}x{Ly} FH, D={D}, chi={chi}, B={B}")

        # Load PEPS
        fpeps_base = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/t={t}_U={U}"
            f"/N={N_f}/Z2/D={D}/"
        )
        peps = load_or_generate_peps(
            Lx, Ly, t, U, N_f, D,
            seed=42, dtype=dtype,
            file_path=fpeps_base,
            scale_factor=4,
        )

        # Build batch
        batch_x = torch.stack([
            random_initial_config(N_f, N_sites, seed=s)
            for s in range(B)
        ]).to(device)

        # ===== Standard SVD =====
        print("\n--- Standard SVD (svd_truncated) ---")
        ar.register_function(
            "symmray", "svd_truncated", svd_truncated,
        )
        model_std = fPEPS_Model_GPU(
            tn=peps,
            max_bond=chi,
            dtype=dtype,
            contract_boundary_opts={
                'mode': 'mps',
                'equalize_norms': 1.0,
                'canonize': True,
            },
        )
        model_std.to(device)
        with torch.no_grad():
            t0 = time.time()
            signs_std, logabs_std = model_std.forward_log(
                batch_x,
            )
            torch.cuda.synchronize()
            t_std = time.time() - t0
        print(f"Time: {t_std:.4f}s")
        print(f"signs:  {signs_std[:5]}")
        print(f"logabs: {logabs_std[:5]}")
        print(
            f"logabs range: [{logabs_std.min():.4f}, "
            f"{logabs_std.max():.4f}]"
        )

        # ===== Randomized SVD =====
        print("\n--- Randomized SVD (svd_rand_truncated) ---")
        ar.register_function(
            "symmray", "svd_truncated", svd_rand_truncated,
        )
        model_rand = fPEPS_Model_GPU(
            tn=peps,
            max_bond=chi,
            dtype=dtype,
            contract_boundary_opts={
                'mode': 'mps',
                'equalize_norms': 1.0,
                'canonize': True,
                'compress_opts': {'seed': 42},
            },
        )
        model_rand.to(device)
        with torch.no_grad():
            t0 = time.time()
            signs_rand, logabs_rand = model_rand.forward_log(
                batch_x,
            )
            torch.cuda.synchronize()
            t_rand = time.time() - t0
        print(f"Time: {t_rand:.4f}s")
        print(f"signs:  {signs_rand[:5]}")
        print(f"logabs: {logabs_rand[:5]}")
        print(
            f"logabs range: [{logabs_rand.min():.4f}, "
            f"{logabs_rand.max():.4f}]"
        )

        # ===== Randomized SVD (different seed) =====
        print("\n--- Randomized SVD (seed=123) ---")
        model_rand2 = fPEPS_Model_GPU(
            tn=peps,
            max_bond=chi,
            dtype=dtype,
            contract_boundary_opts={
                'mode': 'mps',
                'equalize_norms': 1.0,
                'canonize': True,
                'compress_opts': {'seed': 123},
            },
        )
        model_rand2.to(device)
        with torch.no_grad():
            t0 = time.time()
            signs_rand2, logabs_rand2 = (
                model_rand2.forward_log(batch_x)
            )
            torch.cuda.synchronize()
            t_rand2 = time.time() - t0
        print(f"Time: {t_rand2:.4f}s")
        print(f"signs:  {signs_rand2[:5]}")
        print(f"logabs: {logabs_rand2[:5]}")

        # ===== Compare =====
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")

        # Standard vs Randomized (seeded)
        sign_match = (signs_std == signs_rand).all()
        logabs_diff = (logabs_std - logabs_rand).abs()
        logabs_reldiff = logabs_diff / (
            logabs_std.abs().clamp(min=1e-45)
        )
        print("\nStandard vs Randomized (seed=42):")
        print(f"  Signs match: {sign_match}")
        print(
            f"  logabs abs diff: "
            f"max={logabs_diff.max():.2e}, "
            f"mean={logabs_diff.mean():.2e}"
        )
        print(
            f"  logabs rel diff: "
            f"max={logabs_reldiff.max():.2e}, "
            f"mean={logabs_reldiff.mean():.2e}"
        )

        # Standard vs Randomized (no seed)
        sign_match2 = (signs_std == signs_rand2).all()
        logabs_diff2 = (logabs_std - logabs_rand2).abs()
        logabs_reldiff2 = logabs_diff2 / (
            logabs_std.abs().clamp(min=1e-45)
        )
        print("\nStandard vs Randomized (seed=123):")
        print(f"  Signs match: {sign_match2}")
        print(
            f"  logabs abs diff: "
            f"max={logabs_diff2.max():.2e}, "
            f"mean={logabs_diff2.mean():.2e}"
        )
        print(
            f"  logabs rel diff: "
            f"max={logabs_reldiff2.max():.2e}, "
            f"mean={logabs_reldiff2.mean():.2e}"
        )

        # Randomized self-consistency
        sign_match3 = (signs_rand == signs_rand2).all()
        logabs_diff3 = (logabs_rand - logabs_rand2).abs()
        print("\nRandomized (seed=42) vs Randomized (seed=123):")
        print(f"  Signs match: {sign_match3}")
        print(
            f"  logabs abs diff: "
            f"max={logabs_diff3.max():.2e}, "
            f"mean={logabs_diff3.mean():.2e}"
        )

        # NaN/Inf check
        print(f"\nNaN/Inf:")
        print(
            f"  Std:  NaN={logabs_std.isnan().any()}, "
            f"Inf={logabs_std.isinf().any()}"
        )
        print(
            f"  Rand: NaN={logabs_rand.isnan().any()}, "
            f"Inf={logabs_rand.isinf().any()}"
        )

        # Timing
        print(f"\nTiming:")
        print(f"  Standard SVD:    {t_std:.4f}s")
        print(f"  Randomized SVD:  {t_rand:.4f}s")
        print(f"  Speedup:         {t_std/t_rand:.2f}x")

        # Per-sample breakdown if large differences
        if logabs_reldiff.max() > 1e-2:
            print("\n--- Large diffs (std vs rand) ---")
            bad_mask = logabs_reldiff > 1e-2
            bad_idx = bad_mask.nonzero().squeeze()
            if bad_idx.dim() == 0:
                bad_idx = bad_idx.unsqueeze(0)
            for idx in bad_idx[:10]:
                i = idx.item()
                print(
                    f"  [{i}]: std={logabs_std[i]:.8f}"
                    f"  rand={logabs_rand[i]:.8f}"
                    f"  diff={logabs_diff[i]:.2e}"
                )

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
