"""Test inductor correctness: compiled vs eager forward_log.

Confirmed correct: export-only, aot_eager (on 8x8 DGX).
This script tests inductor — the only layer that may
produce wrong results.

Run:
    torchrun --nproc_per_node=1 test_compile_correctness.py
"""
import time

import torch
import torch.distributed as dist

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


def compare(name, signs, log_abs, ref_signs, ref_log_abs,
            tol=1e-3):
    """Print comparison and return whether it matches."""
    sign_diff = (signs - ref_signs).abs()
    log_diff = (log_abs - ref_log_abs).abs()
    ok = log_diff.max().item() < tol
    status = "OK" if ok else "*** WRONG ***"
    print(f"\n=== {name} === {status}")
    print(f"  signs:   {signs[:4]}")
    print(f"  log_abs: {log_abs[:4]}")
    print(
        f"  vs eager: sign_diff max={sign_diff.max():.2e} "
        f"log_diff max={log_diff.max():.2e}"
    )
    if not ok:
        n_sign_flip = (sign_diff > 0.5).sum().item()
        print(f"  sign flips: {n_sign_flip}/{len(signs)}")
    return ok


def main():
    setup_linalg_hooks(
        jitter=1e-8, qr_via_eigh=True,
        cholesky_qr=False,
        cholesky_qr_adaptive_jitter=False,
        nonuniform_diag=False,
    )
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        Lx, Ly = 6, 4
        N_sites = Lx * Ly
        t, U = 1.0, 8.0
        N_f = N_sites
        D, chi = 14, -1
        B = 4

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

        fxs = torch.stack([
            random_initial_config(N_f, N_sites, seed=s)
            .to(device)
            for s in range(B)
        ])
        example_x = random_initial_config(
            N_f, N_sites, seed=0,
        ).to(device)
        print(f"System: {Lx}x{Ly}, D={D}, chi={chi}, B={B}")

        # ==========================================================
        # 1. Eager baseline
        # ==========================================================
        with torch.inference_mode():
            t0 = time.time()
            eager_signs, eager_log_abs = model.forward_log(
                fxs,
            )
            torch.cuda.synchronize()
            t_eager = time.time() - t0
        print(f"\n=== Eager === {t_eager:.2f}s")
        print(f"  signs:   {eager_signs[:4]}")
        print(f"  log_abs: {eager_log_abs[:4]}")

        # ==========================================================
        # 2. Export + compile (inductor)
        # ==========================================================
        print("\nExporting...")
        t0 = time.time()
        model.export_and_compile(
            example_x, use_log_amp=True,
        )
        t_export_compile = time.time() - t0
        print(f"  export_and_compile: {t_export_compile:.1f}s")

        # Warmup (triggers Triton compilation)
        print("Warmup (first compiled forward)...")
        with torch.inference_mode():
            t0 = time.time()
            _ = model.forward_log(fxs)
            torch.cuda.synchronize()
            t_warmup = time.time() - t0
        print(f"  warmup: {t_warmup:.1f}s")

        # Timed compiled forward
        with torch.inference_mode():
            t0 = time.time()
            ind_signs, ind_log_abs = model.forward_log(fxs)
            torch.cuda.synchronize()
            t_compiled = time.time() - t0

        ind_ok = compare(
            "inductor", ind_signs, ind_log_abs,
            eager_signs, eager_log_abs,
        )
        print(f"  compiled forward: {t_compiled:.4f}s")

        # ==========================================================
        # 3. Per-sample detail if wrong
        # ==========================================================
        if not ind_ok:
            print("\n=== Per-sample: inductor vs eager ===")
            for i in range(min(B, 16)):
                ld = abs(
                    ind_log_abs[i].item()
                    - eager_log_abs[i].item()
                )
                print(
                    f"  [{i:2d}] eager=({eager_signs[i]:+.0f},"
                    f" {eager_log_abs[i]:+.4f})  "
                    f"ind=({ind_signs[i]:+.0f},"
                    f" {ind_log_abs[i]:+.4f})  "
                    f"Δlog={ld:.2e}"
                )

        # ==========================================================
        # Summary
        # ==========================================================
        print(f"\n{'=' * 50}")
        print(
            f"{'PASS' if ind_ok else 'FAIL'}: "
            f"{Lx}x{Ly} D={D} chi={chi} B={B}"
        )
        print(f"  eager:    {t_eager:.2f}s")
        print(f"  compiled: {t_compiled:.4f}s "
              f"({t_eager/t_compiled:.1f}x speedup)")
        print(f"{'=' * 50}")

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
