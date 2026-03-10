"""Ablate inductor configs to find root cause.

Tests inductor with various safety configs disabled
to find which one fixes the accuracy bug.

Run:
    python debug_inductor_configs.py
"""
import gc
import time

import torch
import torch._inductor.config as inductor_config

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

example_x = random_initial_config(
    N_f, N_sites, seed=0,
).to(device)

# B=1 is faster for testing
fxs = example_x.unsqueeze(0)  # (1, N_sites)

print(f"System: {Lx}x{Ly}, D={D}, chi={chi}, B=1")

# Eager baseline
with torch.inference_mode():
    eager_signs, eager_log = model.forward_log(fxs)
print(f"Eager: sign={eager_signs}, log={eager_log}")

# Export once (reused across all config tests)
model.export_only(example_x, use_log_amp=True)
params_list = list(model.params)

# Configs to test: (name, config_dict)
# Each config_dict is applied before compile, then reset
configs = [
    ("default", {}),
    ("allow_buffer_reuse=False", {
        "allow_buffer_reuse": False,
    }),
    ("inplace_buffers=False", {
        "inplace_buffers": False,
    }),
    ("pattern_matcher=False", {
        "pattern_matcher": False,
    }),
    ("constant_folding=False", {
        "constant_folding": False,
    }),
    ("split_reductions=False", {
        "split_reductions": False,
    }),
    ("combo: no buffer reuse + no inplace", {
        "allow_buffer_reuse": False,
        "inplace_buffers": False,
    }),
    ("combo: no pattern + no const fold", {
        "pattern_matcher": False,
        "constant_folding": False,
    }),
    ("all safety: no buf reuse/inplace/pattern/fold", {
        "allow_buffer_reuse": False,
        "inplace_buffers": False,
        "pattern_matcher": False,
        "constant_folding": False,
        "split_reductions": False,
    }),
]

results = []
for name, cfg in configs:
    # Save original values
    originals = {}
    for k, v in cfg.items():
        try:
            originals[k] = getattr(inductor_config, k)
            setattr(inductor_config, k, v)
        except AttributeError:
            print(f"  WARNING: config '{k}' not found")

    # Clear compile caches
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()

    compiled = torch.compile(
        model._vmapped_exported,
        backend='inductor',
    )

    try:
        with torch.inference_mode():
            t0 = time.time()
            ind_signs, ind_log = compiled(
                fxs, *params_list,
            )
            torch.cuda.synchronize()
            dt = time.time() - t0

        diff = (ind_log - eager_log).abs().max().item()
        sign_ok = (ind_signs == eager_signs).all().item()
        ok = diff < 1e-3 and sign_ok
        status = "PASS" if ok else "FAIL"
        results.append((name, status, diff, dt))
        print(
            f"\n[{status}] {name}: "
            f"diff={diff:.2e}, sign_ok={sign_ok}, "
            f"time={dt:.1f}s"
        )
    except Exception as e:
        results.append((name, "ERROR", str(e), 0))
        print(f"\n[ERROR] {name}: {e}")

    # Restore original values
    for k, v in originals.items():
        setattr(inductor_config, k, v)

# Summary
print(f"\n{'=' * 60}")
print("Summary:")
for name, status, diff, dt in results:
    if status == "ERROR":
        print(f"  [{status}] {name}: {diff}")
    else:
        print(f"  [{status}] {name}: diff={diff:.2e}")
print(f"{'=' * 60}")

passing = [
    name for name, status, _, _ in results
    if status == "PASS"
]
if passing:
    print(f"\nFIXED BY: {passing}")
else:
    print(
        "\nNone of the configs fixed the bug. "
        "Need deeper graph bisection."
    )
