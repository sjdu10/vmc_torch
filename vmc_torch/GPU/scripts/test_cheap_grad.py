"""Test cheap gradient vs standard gradient for fPEPS_Model_reuse_GPU.

Compares compute_grads_cheap_gpu (per-row hole contraction) against
compute_grads_gpu (standard vmap(grad) through full TN contraction).

Usage:
    python GPU/scripts/test_cheap_grad.py
"""
import time

import torch
import quimb.tensor as qtn

from vmc_torch.GPU.models import fPEPS_Model_reuse_GPU
from vmc_torch.GPU.vmc_utils import (
    compute_grads_cheap_gpu,
    compute_grads_gpu,
    random_initial_config,
)
from vmc_torch.GPU.vmc_setup import (
    load_or_generate_peps,
    setup_linalg_hooks,
)


def run_test(
    Lx, Ly, D, chi, B, N_f=None, dtype=torch.float64,
    device=None, use_log_amp=False,
):
    """Compare cheap grad vs standard grad on a small system."""
    if device is None:
        device = (
            torch.device('cuda')
            if torch.cuda.is_available()
            else torch.device('cpu')
        )
    N_sites = Lx * Ly
    if N_f is None:
        N_f = N_sites - 2  # slight doping
    t_hop, U = 1.0, 4.0

    print(f"\n{'='*60}")
    print(
        f"Test: {Lx}x{Ly}, D={D}, chi={chi}, B={B}, "
        f"N_f={N_f}, use_log_amp={use_log_amp}"
    )
    print(f"Device: {device}, dtype: {dtype}")
    print(f"{'='*60}")

    # Generate PEPS
    peps = load_or_generate_peps(
        Lx, Ly, t_hop, U, N_f, D,
        seed=42, dtype=dtype,
        scale_factor=4,
    )

    # Create reuse model
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

    # Initialize bMPS skeletons
    example_x = random_initial_config(
        N_f, N_sites, seed=0,
    ).to(device)
    model.cache_bMPS_skeleton(example_x)

    N_params = sum(p.numel() for p in model.parameters())
    print(f"N_params: {N_params}")

    # Generate random configs
    fxs = torch.stack([
        random_initial_config(N_f, N_sites, seed=s).to(device)
        for s in range(B)
    ])

    # --- Standard gradient ---
    print("\n--- Standard gradient (vmap(grad)) ---")
    t0 = time.time()
    with torch.enable_grad():
        if use_log_amp:
            grads_std, (signs_std, log_abs_std) = (
                compute_grads_gpu(
                    fxs, model,
                    vectorize=True, vmap_grad=True,
                    use_log_amp=True,
                )
            )
        else:
            grads_std, amps_std = compute_grads_gpu(
                fxs, model,
                vectorize=True, vmap_grad=True,
                use_log_amp=False,
            )
    t_std = time.time() - t0
    print(f"  Time: {t_std:.4f}s")
    if use_log_amp:
        print(
            f"  log_abs: {log_abs_std[:3].tolist()}"
        )
        print(
            f"  grad rms: "
            f"{torch.norm(grads_std).item() / grads_std.numel()**0.5:.4e}"
        )
    else:
        print(f"  amps: {amps_std[:3].tolist()}")
        print(
            f"  grad rms: "
            f"{torch.norm(grads_std).item() / grads_std.numel()**0.5:.4e}"
        )

    # --- Cheap gradient ---
    print("\n--- Cheap gradient (hole contraction) ---")
    t0 = time.time()
    with torch.enable_grad():
        if use_log_amp:
            grads_cheap, (signs_cheap, log_abs_cheap) = (
                compute_grads_cheap_gpu(
                    fxs, model,
                    use_log_amp=True,
                )
            )
        else:
            grads_cheap, amps_cheap = (
                compute_grads_cheap_gpu(
                    fxs, model,
                    use_log_amp=False,
                )
            )
    t_cheap = time.time() - t0
    print(f"  Time: {t_cheap:.4f}s")
    if use_log_amp:
        print(
            f"  log_abs: {log_abs_cheap[:3].tolist()}"
        )
        print(
            f"  grad rms: "
            f"{torch.norm(grads_cheap).item() / grads_cheap.numel()**0.5:.4e}"
        )
    else:
        print(f"  amps: {amps_cheap[:3].tolist()}")
        print(
            f"  grad rms: "
            f"{torch.norm(grads_cheap).item() / grads_cheap.numel()**0.5:.4e}"
        )

    # --- Compare ---
    print("\n--- Comparison ---")

    if use_log_amp:
        amp_diff = (
            (log_abs_std - log_abs_cheap).abs().max().item()
        )
        amp_rel = (
            (log_abs_std - log_abs_cheap).abs()
            / (log_abs_std.abs() + 1e-30)
        ).max().item()
        print(
            f"  log_abs max abs diff: {amp_diff:.4e}"
        )
        print(f"  log_abs max rel diff: {amp_rel:.4e}")
    else:
        amp_diff = (
            (amps_std - amps_cheap).abs().max().item()
        )
        amp_rel = (
            (amps_std - amps_cheap).abs()
            / (amps_std.abs() + 1e-30)
        ).max().item()
        print(f"  amp max abs diff: {amp_diff:.4e}")
        print(f"  amp max rel diff: {amp_rel:.4e}")

    grad_diff = (grads_std - grads_cheap).abs()
    grad_abs_max = grad_diff.max().item()
    grad_rel = (
        grad_diff
        / (grads_std.abs() + 1e-30)
    )
    grad_rel_max = grad_rel.max().item()
    grad_rel_mean = grad_rel.mean().item()
    # Per-sample relative error (L2 norm)
    per_sample_rel = (
        (grads_std - grads_cheap).norm(dim=1)
        / (grads_std.norm(dim=1) + 1e-30)
    )
    print(f"  grad max abs diff: {grad_abs_max:.4e}")
    print(f"  grad max rel diff: {grad_rel_max:.4e}")
    print(f"  grad mean rel diff: {grad_rel_mean:.4e}")
    print(
        f"  per-sample L2 rel: "
        f"mean={per_sample_rel.mean().item():.4e}, "
        f"max={per_sample_rel.max().item():.4e}"
    )

    print(f"\n  Speedup: {t_std / t_cheap:.2f}x")

    # Pass/fail check
    tol = 1e-4
    passed = per_sample_rel.max().item() < tol
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] tol={tol}")
    return passed


if __name__ == "__main__":
    setup_linalg_hooks(
        jitter=1e-8, qr_via_eigh=False,
        cholesky_qr=True,
    )
    torch.set_default_dtype(torch.float64)
    device = (
        torch.device('cuda')
        if torch.cuda.is_available()
        else torch.device('cpu')
    )
    torch.set_default_device(device)

    all_passed = True

    # Test 1: exact contraction (chi=-1), small system
    all_passed &= run_test(
        Lx=3, Ly=2, D=2, chi=-1, B=4,
        device=device,
    )

    # Test 2: boundary contraction with chi >= 4*D
    all_passed &= run_test(
        Lx=3, Ly=2, D=2, chi=8, B=4,
        device=device,
    )

    # Test 3: log-amplitude mode
    all_passed &= run_test(
        Lx=3, Ly=2, D=2, chi=-1, B=4,
        device=device, use_log_amp=True,
    )

    # Test 4: slightly larger system
    all_passed &= run_test(
        Lx=4, Ly=2, D=2, chi=-1, B=4,
        device=device,
    )

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"{'='*60}")
