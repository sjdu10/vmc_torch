"""
test_get_conn_batch.py — Correctness test for get_conn_batch_gpu.

Compares the new GPU-batched Hamiltonian connectivity
(get_conn_batch_gpu) against the original per-sample CPU
get_conn as the ground-truth baseline.

Tests:
  1. Single config: connected configs and coefficients match exactly.
  2. Batch of random configs: all match.
  3. Edge cases: all-empty, all-doubly-occupied, half-filled, doped.
  4. Larger lattice (3x3) with more hop terms.

Run:
    python GPU/scripts/test_get_conn_batch.py
"""
import sys
import os
import numpy as np
import torch

# --- Make sure the package is importable ---
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        '../../../../..',    # VMC_code/vmc_torch
    ),
)

from vmc_torch.hamiltonian_torch import (
    spinful_Fermi_Hubbard_square_lattice_torch,
)
from vmc_torch.experiment.vmap.GPU.vmc_utils import random_initial_config


# ============================================================
# Helpers
# ============================================================

def get_conn_reference(H, fxs_cpu):
    """
    Run the original per-sample get_conn on a CPU tensor batch.
    Returns dict: config_index -> {config_tuple: coeff}.
    """
    results = []
    for fx in fxs_cpu:
        etas, coeffs = H.get_conn(fx)
        d = {}
        for eta, c in zip(etas, coeffs):
            key = tuple(int(x) for x in eta)
            # accumulate if same config appears more than once
            d[key] = d.get(key, 0.0) + float(c)
        results.append(d)
    return results


def get_conn_batch_result(H, fxs_gpu):
    """
    Run get_conn_batch_gpu and reorganise into the same format
    as get_conn_reference: list of {config_tuple: coeff}.
    """
    B = fxs_gpu.shape[0]
    conn_etas, conn_coeffs, batch_ids = H.get_conn_batch_gpu(fxs_gpu)

    # Move to CPU for comparison
    conn_etas_cpu = conn_etas.cpu().numpy()
    conn_coeffs_cpu = conn_coeffs.cpu().numpy()
    batch_ids_cpu = batch_ids.cpu().numpy()

    results = [{} for _ in range(B)]
    for eta, c, bid in zip(conn_etas_cpu, conn_coeffs_cpu, batch_ids_cpu):
        key = tuple(int(x) for x in eta)
        results[bid][key] = results[bid].get(key, 0.0) + float(c)
    return results


def compare_results(ref, new, label="", tol=1e-10):
    """
    Compare two lists of {config_tuple: coeff} dicts.
    Returns (n_pass, n_fail, error_messages).
    """
    n_pass = 0
    n_fail = 0
    errors = []

    assert len(ref) == len(new), "Length mismatch"

    for i, (r, n) in enumerate(zip(ref, new)):
        # Check same set of connected configs
        r_keys = set(r.keys())
        n_keys = set(n.keys())

        missing = r_keys - n_keys
        extra = n_keys - r_keys

        if missing or extra:
            n_fail += 1
            errors.append(
                f"  [{label}] sample {i}: "
                f"missing configs={missing}, extra configs={extra}"
            )
            continue

        # Check coefficients match
        coeff_ok = True
        for key in r_keys:
            diff = abs(r[key] - n[key])
            if diff > tol:
                coeff_ok = False
                errors.append(
                    f"  [{label}] sample {i}, config {key}: "
                    f"ref={r[key]:.8f} new={n[key]:.8f} diff={diff:.2e}"
                )

        if coeff_ok:
            n_pass += 1
        else:
            n_fail += 1

    return n_pass, n_fail, errors


def run_test(name, H, fxs_cpu, device, verbose=False):
    """Run one test case and print result."""
    fxs_gpu = fxs_cpu.to(device)

    ref = get_conn_reference(H, fxs_cpu)
    new = get_conn_batch_result(H, fxs_gpu)

    n_pass, n_fail, errors = compare_results(ref, new, label=name)

    status = "PASS" if n_fail == 0 else "FAIL"
    print(f"  [{status}] {name}: {n_pass}/{n_pass+n_fail} samples OK")
    if verbose or n_fail > 0:
        for e in errors[:10]:  # cap output
            print(e)
    return n_fail == 0


# ============================================================
# Main
# ============================================================

def main():
    # Use CUDA if available, else CPU (for CI without GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("No CUDA available, using CPU (results still valid)")

    all_pass = True

    # ===========================================================
    # Test suite 1: 2x2 Hubbard, half-filling N_f=4
    # ===========================================================
    print("\n=== 2x2 Hubbard, t=1 U=8, N_f=4 (half-filling) ===")
    Lx, Ly = 2, 2
    N_f = 4
    H22 = spinful_Fermi_Hubbard_square_lattice_torch(
        Lx, Ly, 1.0, 8.0, N_f,
        pbc=False,
        n_fermions_per_spin=(N_f // 2, N_f // 2),
        no_u1_symmetry=False,
        gpu=True,
    )
    H22.precompute_hops_gpu(device)

    # Test 1a: single config, fixed seed
    cfg = random_initial_config(N_f, Lx * Ly, seed=0)  # (N,)
    fxs_single = cfg.unsqueeze(0)  # (1, N)
    ok = run_test("2x2 single config (seed=0)", H22, fxs_single, device)
    all_pass = all_pass and ok

    # Test 1b: batch of 64 random configs
    batch = torch.stack([
        random_initial_config(N_f, Lx * Ly, seed=s)
        for s in range(64)
    ])
    ok = run_test("2x2 batch B=64 (seeds 0-63)", H22, batch, device)
    all_pass = all_pass and ok

    # Test 1c: all possible 2x2 half-filled configs (exhaustive)
    from itertools import combinations
    nsites = Lx * Ly
    all_cfgs = []
    # Quimb encoding: 0=empty, 1=down, 2=up, 3=both
    # For half-filling with 2 up + 2 down, enumerate all valid configs
    for up_pos in combinations(range(nsites), N_f // 2):
        for dn_pos in combinations(range(nsites), N_f // 2):
            cfg = torch.zeros(nsites, dtype=torch.long)
            for p in up_pos:
                cfg[p] += 2  # add spin-up
            for p in dn_pos:
                cfg[p] += 1  # add spin-down
            all_cfgs.append(cfg)
    fxs_all = torch.stack(all_cfgs)
    ok = run_test(
        f"2x2 exhaustive ({len(all_cfgs)} configs)",
        H22, fxs_all, device
    )
    all_pass = all_pass and ok

    # ===========================================================
    # Test suite 2: 2x2 Hubbard, doped (N_f=2)
    # ===========================================================
    print("\n=== 2x2 Hubbard, t=1 U=8, N_f=2 (doped) ===")
    N_f2 = 2
    H22d = spinful_Fermi_Hubbard_square_lattice_torch(
        Lx, Ly, 1.0, 8.0, N_f2,
        pbc=False,
        n_fermions_per_spin=(N_f2 // 2, N_f2 // 2),
        no_u1_symmetry=False,
        gpu=True,
    )
    H22d.precompute_hops_gpu(device)

    batch_d = torch.stack([
        random_initial_config(N_f2, Lx * Ly, seed=s)
        for s in range(32)
    ])
    ok = run_test("2x2 doped B=32 (seeds 0-31)", H22d, batch_d, device)
    all_pass = all_pass and ok

    # ===========================================================
    # Test suite 3: 2x3 Hubbard, half-filling N_f=6 (3+3)
    # ===========================================================
    print("\n=== 2x3 Hubbard, t=1 U=4, N_f=6 (3+3, half-filling) ===")
    Lx3, Ly3 = 2, 3
    N_f3 = 6
    n_up3, n_dn3 = N_f3 // 2, N_f3 // 2
    H23 = spinful_Fermi_Hubbard_square_lattice_torch(
        Lx3, Ly3, 1.0, 4.0, N_f3,
        pbc=False,
        n_fermions_per_spin=(n_up3, n_dn3),
        no_u1_symmetry=False,
        gpu=True,
    )
    H23.precompute_hops_gpu(device)

    batch23 = torch.stack([
        random_initial_config(N_f3, Lx3 * Ly3, seed=s)
        for s in range(64)
    ])
    ok = run_test("2x3 half-filling B=64 (seeds 0-63)", H23, batch23, device)
    all_pass = all_pass and ok

    # ===========================================================
    # Test suite 4: 4x4 Hubbard, half-filling N_f=16
    # ===========================================================
    print("\n=== 4x4 Hubbard, t=1 U=8, N_f=16 (half-filling) ===")
    Lx4, Ly4 = 4, 4
    N_f4 = 16
    H44 = spinful_Fermi_Hubbard_square_lattice_torch(
        Lx4, Ly4, 1.0, 8.0, N_f4,
        pbc=False,
        n_fermions_per_spin=(N_f4 // 2, N_f4 // 2),
        no_u1_symmetry=False,
        gpu=True,
    )
    H44.precompute_hops_gpu(device)

    batch44 = torch.stack([
        random_initial_config(N_f4, Lx4 * Ly4, seed=s)
        for s in range(128)
    ])
    ok = run_test("4x4 batch B=128 (seeds 0-127)", H44, batch44, device)
    all_pass = all_pass and ok

    # ===========================================================
    # Test suite 5: Coefficient sign stress test
    # Use configs with many doubly-occupied sites to stress JW phases
    # ===========================================================
    print("\n=== Sign/Phase stress test (2x2, fixed configs) ===")
    H22_s = H22  # reuse 2x2 half-filling Hamiltonian

    # Hand-crafted configs covering all quimb site values
    stress_cfgs = torch.tensor([
        [3, 0, 0, 1],  # doubly-occ, empty, empty, down
        [0, 3, 1, 0],  # empty, doubly-occ, down, empty
        [2, 1, 2, 1],  # up, down, up, down
        [1, 2, 1, 2],  # down, up, down, up
        [3, 3, 0, 0],  # two doubly-occ (invalid N_f but still test H)
        [0, 0, 3, 3],  # (same)
        [2, 2, 1, 1],  # two up, two down
        [1, 1, 2, 2],  # same, permuted
    ], dtype=torch.long)

    ok = run_test(
        "2x2 sign stress (hand-crafted)", H22_s, stress_cfgs, device,
        verbose=True,
    )
    all_pass = all_pass and ok

    # ===========================================================
    # Test suite 6: Large batch performance check
    # (not a correctness test, just makes sure it runs)
    # ===========================================================
    print("\n=== Large batch B=512 (2x2, no correctness detail) ===")
    batch_large = torch.stack([
        random_initial_config(N_f, Lx * Ly, seed=s)
        for s in range(512)
    ])
    import time
    t0 = time.time()
    ref_large = get_conn_reference(H22, batch_large)
    t_ref = time.time() - t0

    t0 = time.time()
    new_large = get_conn_batch_result(H22, batch_large.to(device))
    t_new = time.time() - t0

    n_pass, n_fail, errors = compare_results(
        ref_large, new_large, label="B=512"
    )
    status = "PASS" if n_fail == 0 else "FAIL"
    print(
        f"  [{status}] B=512: {n_pass}/{n_pass+n_fail} OK "
        f"| ref={t_ref:.3f}s new={t_new:.3f}s "
        f"(speedup={t_ref/max(t_new,1e-9):.1f}x)"
    )
    all_pass = all_pass and (n_fail == 0)
    for e in errors[:5]:
        print(e)

    # ===========================================================
    # Summary
    # ===========================================================
    print(f"\n{'='*50}")
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — see errors above")
        sys.exit(1)
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
