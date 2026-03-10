"""Minimal reproducer: vmap+inductor bug for D=10 fPEPS.

Demonstrates:
  1. Single-sample through inductor: CORRECT
  2. Same graph vmap'd through inductor: WRONG
  3. D=8 vmap'd through inductor: CORRECT (control)

System: 6x4 lattice, chi=-1 (exact contraction, no SVD/QR).

Run:
    python reproduce_vmap_inductor_bug.py
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
DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/GPU/data'
)

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
B = 4


def test_D(D):
    """Test single-sample vs vmap inductor for given D."""
    print(f"\n{'=' * 60}")
    print(f"D={D}, chi=-1 (exact), {Lx}x{Ly}, B={B}")
    print(f"{'=' * 60}")

    fpeps_base = (
        f"{DATA_ROOT}/{Lx}x{Ly}/t={t}_U={U}"
        f"/N={N_f}/Z2/D={D}/"
    )
    peps = load_or_generate_peps(
        Lx, Ly, t, U, N_f, D,
        seed=42, dtype=dtype,
        file_path=fpeps_base,
        scale_factor=4,
    )
    model = fPEPS_Model_GPU(
        tn=peps, max_bond=-1, dtype=dtype,
        contract_boundary_opts={
            'mode': 'mps',
            'equalize_norms': 1.0,
            'canonize': True,
        },
    )
    model.to(device)

    example_x = random_initial_config(
        N_f, N_sites, seed=0,
    ).to(device)
    fxs = torch.stack([
        random_initial_config(N_f, N_sites, seed=s)
        .to(device)
        for s in range(B)
    ])

    # --- Export ---
    model.export_only(example_x, use_log_amp=True)
    params_list = list(model.params)
    gm = model._exported_module

    # --- Test 1: Single-sample inductor ---
    with torch.inference_mode():
        eager_single = gm(example_x, *params_list)
    torch._dynamo.reset()
    compiled_single = torch.compile(gm, backend='inductor')
    with torch.inference_mode():
        ind_single = compiled_single(example_x, *params_list)

    if isinstance(eager_single, tuple):
        diffs = []
        for e, c in zip(eager_single, ind_single):
            if isinstance(e, torch.Tensor):
                diffs.append((e - c).abs().max().item())
        diff1 = max(diffs)
    else:
        diff1 = (eager_single - ind_single).abs().max().item()

    ok1 = diff1 < 1e-6
    print(f"\n  [1] Single-sample inductor: "
          f"diff={diff1:.2e} {'PASS' if ok1 else 'FAIL'}")
    if isinstance(eager_single, tuple):
        for i, (e, c) in enumerate(
            zip(eager_single, ind_single)
        ):
            if isinstance(e, torch.Tensor):
                print(f"      out[{i}]: eager={e.item():.6e}"
                      f"  ind={c.item():.6e}")

    # --- Test 2: Vmap + inductor ---
    vmapped = model._vmapped_exported
    with torch.inference_mode():
        eager_vmap = vmapped(fxs, *params_list)

    torch._dynamo.reset()
    compiled_vmap = torch.compile(vmapped, backend='inductor')
    with torch.inference_mode():
        t0 = time.time()
        ind_vmap = compiled_vmap(fxs, *params_list)
        torch.cuda.synchronize()
        dt = time.time() - t0

    if isinstance(eager_vmap, tuple):
        diffs = []
        for i, (e, c) in enumerate(
            zip(eager_vmap, ind_vmap)
        ):
            if isinstance(e, torch.Tensor):
                d = (e - c).abs().max().item()
                diffs.append(d)
                print(f"      out[{i}]: max_diff={d:.2e}")
                print(f"        eager: {e}")
                print(f"        ind:   {c}")
        diff2 = max(diffs)
    else:
        diff2 = (eager_vmap - ind_vmap).abs().max().item()

    ok2 = diff2 < 1e-3
    print(f"\n  [2] Vmap+inductor:          "
          f"diff={diff2:.2e} {'PASS' if ok2 else 'FAIL'}"
          f"  ({dt:.1f}s)")

    # --- Test 3: Vmap eager (sanity) ---
    with torch.inference_mode():
        eager_check = vmapped(fxs, *params_list)
    if isinstance(eager_check, tuple):
        d = max(
            (e - c).abs().max().item()
            for e, c in zip(eager_vmap, eager_check)
            if isinstance(e, torch.Tensor)
        )
    else:
        d = (eager_vmap - eager_check).abs().max().item()
    print(f"  [3] Vmap eager consistency: "
          f"diff={d:.2e} (sanity check)")

    return ok1, ok2


results = {}
D_list = [4, 6, 8, 10, 12, 14, 16, 18, 20]
for D in D_list:
    results[D] = test_D(D)

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"  {'D':>4}  {'Z2 block dim':>12}  "
      f"{'single':>8}  {'vmap':>8}")
print(f"  {'-'*4}  {'-'*12}  {'-'*8}  {'-'*8}")
for D in D_list:
    ok1, ok2 = results[D]
    bdim = D // 2
    print(f"  {D:>4}  {bdim:>12}  "
          f"{'PASS' if ok1 else 'FAIL':>8}  "
          f"{'PASS' if ok2 else 'FAIL':>8}")
print(f"{'=' * 60}")
