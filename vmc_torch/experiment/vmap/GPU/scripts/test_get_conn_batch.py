"""
test_get_conn_batch.py — Correctness test for get_conn_batch_gpu.

Compares the new GPU-batched Hamiltonian connectivity
(get_conn_batch_gpu) against the original per-sample CPU
get_conn as the ground-truth baseline.

Covers all Hamiltonian types:
  - Spinful Fermi-Hubbard (square lattice + chain)
  - Spinless Fermi-Hubbard (chain + square lattice)
  - Heisenberg (chain + square lattice)
  - Transverse-field Ising (chain + square lattice)

Run:
    python GPU/scripts/test_get_conn_batch.py
"""
import sys
import os
import time
import numpy as np
import torch
from itertools import combinations

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
    spinful_Fermi_Hubbard_chain_torch,
    spinless_Fermi_Hubbard_chain_torch,
    spinless_Fermi_Hubbard_square_lattice_torch,
    spin_Heisenberg_chain_torch,
    spin_Heisenberg_square_lattice_torch,
    spin_transverse_Ising_chain_torch,
    spin_transverse_Ising_square_lattice_torch,
)
from vmc_torch.experiment.vmap.GPU.vmc_utils import random_initial_config


# ============================================================
# Config generators
# ============================================================

def random_spinless_config(N_f, N_sites, seed=None):
    """Random binary config with exactly N_f occupied sites."""
    rng = np.random.default_rng(seed)
    positions = rng.choice(N_sites, size=N_f, replace=False)
    cfg = torch.zeros(N_sites, dtype=torch.long)
    cfg[positions] = 1
    return cfg


def random_spin_config(N_sites, total_sz=None, seed=None):
    """Random binary {0,1} spin config.

    If total_sz is given, constrain to n_up = N/2 + total_sz spins up.
    """
    rng = np.random.default_rng(seed)
    if total_sz is not None:
        n_up = N_sites // 2 + total_sz
        positions = rng.choice(N_sites, size=n_up, replace=False)
        cfg = torch.zeros(N_sites, dtype=torch.long)
        cfg[positions] = 1
    else:
        cfg = torch.tensor(
            rng.integers(0, 2, size=N_sites), dtype=torch.long
        )
    return cfg


def all_spinless_configs(N_sites, N_f):
    """Enumerate all C(N_sites, N_f) binary configs."""
    cfgs = []
    for pos in combinations(range(N_sites), N_f):
        cfg = torch.zeros(N_sites, dtype=torch.long)
        for p in pos:
            cfg[p] = 1
        cfgs.append(cfg)
    return torch.stack(cfgs)


def all_spin_configs(N_sites, total_sz=None):
    """Enumerate all spin configs (optionally fixed total_sz)."""
    if total_sz is not None:
        n_up = N_sites // 2 + total_sz
        return all_spinless_configs(N_sites, n_up)
    else:
        cfgs = []
        for i in range(2**N_sites):
            bits = [(i >> b) & 1 for b in range(N_sites)]
            cfgs.append(torch.tensor(bits, dtype=torch.long))
        return torch.stack(cfgs)


def all_spinful_configs(N_sites, n_up, n_dn):
    """Enumerate all spinful fermion configs in quimb encoding."""
    cfgs = []
    for up_pos in combinations(range(N_sites), n_up):
        for dn_pos in combinations(range(N_sites), n_dn):
            cfg = torch.zeros(N_sites, dtype=torch.long)
            for p in up_pos:
                cfg[p] += 2
            for p in dn_pos:
                cfg[p] += 1
            cfgs.append(cfg)
    return torch.stack(cfgs)


# ============================================================
# Helpers
# ============================================================

def get_conn_reference(H, fxs_cpu):
    """Run the original per-sample get_conn on a CPU tensor batch."""
    results = []
    for fx in fxs_cpu:
        etas, coeffs = H.get_conn(fx)
        d = {}
        for eta, c in zip(etas, coeffs):
            key = tuple(int(x) for x in eta)
            d[key] = d.get(key, 0.0) + float(c)
        results.append(d)
    return results


def get_conn_batch_result(H, fxs_gpu):
    """Run get_conn_batch_gpu and reorganise into same format."""
    B = fxs_gpu.shape[0]
    conn_etas, conn_coeffs, batch_ids = H.get_conn_batch_gpu(fxs_gpu)

    conn_etas_cpu = conn_etas.cpu().numpy()
    conn_coeffs_cpu = conn_coeffs.cpu().numpy()
    batch_ids_cpu = batch_ids.cpu().numpy()

    results = [{} for _ in range(B)]
    for eta, c, bid in zip(conn_etas_cpu, conn_coeffs_cpu, batch_ids_cpu):
        key = tuple(int(x) for x in eta)
        results[bid][key] = results[bid].get(key, 0.0) + float(c)
    return results


def compare_results(ref, new, label="", tol=1e-10):
    """Compare two lists of {config_tuple: coeff} dicts."""
    n_pass = 0
    n_fail = 0
    errors = []

    assert len(ref) == len(new), "Length mismatch"

    for i, (r, n) in enumerate(zip(ref, new)):
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
        for e in errors[:10]:
            print(e)
    return n_fail == 0


# ============================================================
# Test suites
# ============================================================

def test_spinful_square(device):
    """Spinful Fermi-Hubbard square lattice tests."""
    all_pass = True

    # --- 2x2, half-filling N_f=4 ---
    print("\n=== Spinful Hubbard Square 2x2, t=1 U=8, N_f=4 ===")
    Lx, Ly, N_f = 2, 2, 4
    H = spinful_Fermi_Hubbard_square_lattice_torch(
        Lx, Ly, 1.0, 8.0, N_f, pbc=False,
        n_fermions_per_spin=(2, 2), no_u1_symmetry=False, gpu=True,
    )
    H.precompute_hops_gpu(device)

    cfg = random_initial_config(N_f, Lx * Ly, seed=0).unsqueeze(0)
    all_pass &= run_test("single config", H, cfg, device)

    batch = torch.stack([
        random_initial_config(N_f, Lx * Ly, seed=s) for s in range(64)
    ])
    all_pass &= run_test("batch B=64", H, batch, device)

    fxs_all = all_spinful_configs(Lx * Ly, 2, 2)
    all_pass &= run_test(
        f"exhaustive ({len(fxs_all)} configs)", H, fxs_all, device
    )

    # --- 2x2, doped N_f=2 ---
    print("\n=== Spinful Hubbard Square 2x2, t=1 U=8, N_f=2 (doped) ===")
    H2 = spinful_Fermi_Hubbard_square_lattice_torch(
        2, 2, 1.0, 8.0, 2, pbc=False,
        n_fermions_per_spin=(1, 1), no_u1_symmetry=False, gpu=True,
    )
    H2.precompute_hops_gpu(device)
    batch_d = torch.stack([
        random_initial_config(2, 4, seed=s) for s in range(32)
    ])
    all_pass &= run_test("doped B=32", H2, batch_d, device)

    return all_pass


def test_spinful_chain(device):
    """Spinful Fermi-Hubbard chain tests."""
    all_pass = True

    print("\n=== Spinful Hubbard Chain L=4, t=1 U=4, N_f=4 ===")
    L, N_f = 4, 4
    H = spinful_Fermi_Hubbard_chain_torch(
        L, 1.0, 4.0, N_f, pbc=False,
        n_fermions_per_spin=(2, 2), no_u1_symmetry=False,
    )
    H.precompute_hops_gpu(device)

    batch = torch.stack([
        random_initial_config(N_f, L, seed=s) for s in range(64)
    ])
    all_pass &= run_test("batch B=64", H, batch, device)

    fxs_all = all_spinful_configs(L, 2, 2)
    all_pass &= run_test(
        f"exhaustive ({len(fxs_all)} configs)", H, fxs_all, device
    )

    return all_pass


def test_spinless_chain(device):
    """Spinless Fermi-Hubbard chain tests."""
    all_pass = True

    print("\n=== Spinless Hubbard Chain L=6, t=1 V=0.5, N_f=3 ===")
    L, N_f = 6, 3
    H = spinless_Fermi_Hubbard_chain_torch(L, 1.0, 0.5, N_f, pbc=False)
    H.precompute_hops_gpu(device)

    batch = torch.stack([
        random_spinless_config(N_f, L, seed=s) for s in range(64)
    ])
    all_pass &= run_test("batch B=64", H, batch, device)

    fxs_all = all_spinless_configs(L, N_f)
    all_pass &= run_test(
        f"exhaustive ({len(fxs_all)} configs)", H, fxs_all, device
    )

    return all_pass


def test_spinless_square(device):
    """Spinless Fermi-Hubbard square lattice tests."""
    all_pass = True

    print("\n=== Spinless Hubbard Square 2x3, t=1 V=0.5 mu=0.2, N_f=3 ===")
    Lx, Ly, N_f = 2, 3, 3
    H = spinless_Fermi_Hubbard_square_lattice_torch(
        Lx, Ly, t=1.0, V=0.5, mu=0.2, N_f=N_f, pbc=False,
    )
    H.precompute_hops_gpu(device)

    batch = torch.stack([
        random_spinless_config(N_f, Lx * Ly, seed=s) for s in range(64)
    ])
    all_pass &= run_test("batch B=64", H, batch, device)

    fxs_all = all_spinless_configs(Lx * Ly, N_f)
    all_pass &= run_test(
        f"exhaustive ({len(fxs_all)} configs)", H, fxs_all, device
    )

    return all_pass


def test_heisenberg_chain(device):
    """Heisenberg chain tests."""
    all_pass = True

    print("\n=== Heisenberg Chain L=6, J=1, total_sz=0 ===")
    L = 6
    H = spin_Heisenberg_chain_torch(L, J=1.0, pbc=False, total_sz=0)
    H.precompute_hops_gpu(device)

    batch = torch.stack([
        random_spin_config(L, total_sz=0, seed=s) for s in range(64)
    ])
    all_pass &= run_test("batch B=64", H, batch, device)

    fxs_all = all_spin_configs(L, total_sz=0)
    all_pass &= run_test(
        f"exhaustive ({len(fxs_all)} configs)", H, fxs_all, device
    )

    # Also test L=4 exhaustive without total_sz constraint
    print("\n=== Heisenberg Chain L=4, J=1, no Sz constraint ===")
    H4 = spin_Heisenberg_chain_torch(4, J=1.0, pbc=False)
    H4.precompute_hops_gpu(device)
    fxs_all4 = all_spin_configs(4)
    all_pass &= run_test(
        f"exhaustive ({len(fxs_all4)} configs)", H4, fxs_all4, device
    )

    return all_pass


def test_heisenberg_square(device):
    """Heisenberg square lattice tests."""
    all_pass = True

    print("\n=== Heisenberg Square 3x3, J=1, total_sz=0 ===")
    Lx, Ly = 3, 3
    N = Lx * Ly
    H = spin_Heisenberg_square_lattice_torch(
        Lx, Ly, J=1.0, pbc=False, total_sz=0,
    )
    H.precompute_hops_gpu(device)

    batch = torch.stack([
        random_spin_config(N, total_sz=0, seed=s) for s in range(64)
    ])
    all_pass &= run_test("batch B=64", H, batch, device)

    # Exhaustive for 2x2
    print("\n=== Heisenberg Square 2x2, J=1, total_sz=0 ===")
    H22 = spin_Heisenberg_square_lattice_torch(
        2, 2, J=1.0, pbc=False, total_sz=0,
    )
    H22.precompute_hops_gpu(device)
    fxs_all = all_spin_configs(4, total_sz=0)
    all_pass &= run_test(
        f"exhaustive ({len(fxs_all)} configs)", H22, fxs_all, device
    )

    return all_pass


def test_ising_chain(device):
    """Transverse-field Ising chain tests."""
    all_pass = True

    print("\n=== Transverse Ising Chain L=6, J=1 h=0.5 ===")
    L = 6
    H = spin_transverse_Ising_chain_torch(
        L, J=1.0, h=0.5, pbc=False,
    )
    H.precompute_hops_gpu(device)

    batch = torch.stack([
        random_spin_config(L, seed=s) for s in range(64)
    ])
    all_pass &= run_test("batch B=64", H, batch, device)

    # Exhaustive L=4
    print("\n=== Transverse Ising Chain L=4, J=1 h=0.5 (exhaustive) ===")
    H4 = spin_transverse_Ising_chain_torch(4, J=1.0, h=0.5, pbc=False)
    H4.precompute_hops_gpu(device)
    fxs_all = all_spin_configs(4)
    all_pass &= run_test(
        f"exhaustive ({len(fxs_all)} configs)", H4, fxs_all, device
    )

    return all_pass


def test_ising_square(device):
    """Transverse-field Ising square lattice tests."""
    all_pass = True

    print("\n=== Transverse Ising Square 3x3, J=1 h=0.5 ===")
    Lx, Ly = 3, 3
    N = Lx * Ly
    H = spin_transverse_Ising_square_lattice_torch(
        Lx, Ly, J=1.0, h=0.5, pbc=False,
    )
    H.precompute_hops_gpu(device)

    batch = torch.stack([
        random_spin_config(N, seed=s) for s in range(64)
    ])
    all_pass &= run_test("batch B=64", H, batch, device)

    # Exhaustive 2x2
    print("\n=== Transverse Ising Square 2x2, J=1 h=0.5 (exhaustive) ===")
    H22 = spin_transverse_Ising_square_lattice_torch(
        2, 2, J=1.0, h=0.5, pbc=False,
    )
    H22.precompute_hops_gpu(device)
    fxs_all = all_spin_configs(4)
    all_pass &= run_test(
        f"exhaustive ({len(fxs_all)} configs)", H22, fxs_all, device
    )

    return all_pass


# ============================================================
# Main
# ============================================================

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("No CUDA available, using CPU (results still valid)")

    all_pass = True

    # Spinful fermion tests
    all_pass &= test_spinful_square(device)
    all_pass &= test_spinful_chain(device)

    # Spinless fermion tests
    all_pass &= test_spinless_chain(device)
    all_pass &= test_spinless_square(device)

    # Heisenberg spin tests
    all_pass &= test_heisenberg_chain(device)
    all_pass &= test_heisenberg_square(device)

    # Transverse-field Ising tests
    all_pass &= test_ising_chain(device)
    all_pass &= test_ising_square(device)

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
