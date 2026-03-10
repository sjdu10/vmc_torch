"""Test inductor correctness: compiled vs eager forward
(direct amplitude, NO log/strip_exponent).

Isolates whether the inductor bug is in the log-amplitude
path (strip_exponent) or in the core TN contraction.

Run:
    python test_compile_correctness_nolog.py
"""
import time

import torch

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
        jitter=1e-8, qr_via_eigh=True,
        cholesky_qr=False,
        cholesky_qr_adaptive_jitter=False,
        nonuniform_diag=False,
    )
    torch.set_default_dtype(dtype)
    device = torch.device('cuda:0')
    torch.set_default_device(device)
    torch.manual_seed(42)

    Lx, Ly = 6, 4
    N_sites = Lx * Ly
    t, U = 1.0, 8.0
    N_f = N_sites - 2
    D, chi = 10, 10
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
    # 1. Eager baseline (direct amplitude, no log)
    # ==========================================================
    with torch.inference_mode():
        t0 = time.time()
        eager_amps = model.forward(fxs)  # (B,) scalar amps
        torch.cuda.synchronize()
        t_eager = time.time() - t0
    print(f"\n=== Eager (direct amp) === {t_eager:.2f}s")
    print(f"  amps: {eager_amps}")

    # ==========================================================
    # 2. Export + compile (inductor, use_log_amp=False)
    # ==========================================================
    print("\nExporting (use_log_amp=False)...")
    t0 = time.time()
    model.export_and_compile(
        example_x, use_log_amp=False,
    )
    t_export = time.time() - t0
    print(f"  export_and_compile: {t_export:.1f}s")

    # Warmup
    print("Warmup...")
    with torch.inference_mode():
        t0 = time.time()
        _ = model.forward(fxs)
        torch.cuda.synchronize()
        t_warmup = time.time() - t0
    print(f"  warmup: {t_warmup:.1f}s")

    # Timed compiled forward
    with torch.inference_mode():
        t0 = time.time()
        ind_amps = model.forward(fxs)
        torch.cuda.synchronize()
        t_compiled = time.time() - t0

    # Compare
    diff = (ind_amps - eager_amps).abs()
    rel_diff = diff / (eager_amps.abs() + 1e-30)
    ok = diff.max().item() < 1e-3
    print(f"\n=== Inductor (direct amp) === "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"  amps:     {ind_amps}")
    print(f"  abs diff: {diff}")
    print(f"  rel diff: {rel_diff}")
    print(f"  max abs diff: {diff.max():.2e}")
    print(f"  max rel diff: {rel_diff.max():.2e}")
    print(f"  time: {t_compiled:.4f}s")

    # ==========================================================
    # Summary
    # ==========================================================
    print(f"\n{'=' * 50}")
    print(
        f"Direct amplitude (no log): "
        f"{'PASS' if ok else 'FAIL'} "
        f"(max diff={diff.max():.2e})"
    )
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
