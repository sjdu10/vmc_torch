"""Bisect inductor accuracy bug: B=1 vs B>1, graph op analysis.

Diagnostics:
1. B=1 (single-sample — vmap is no-op): does inductor still fail?
   If PASS → bug is in batched op lowering (vmap interaction)
   If FAIL → bug is in single-sample op lowering
2. Print FX graph op breakdown (before vmap) to identify suspects.

Run:
    python debug_inductor_bisect.py
"""
import time
from collections import Counter

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
print(f"System: {Lx}x{Ly}, D={D}, chi={chi}")

# ============================================================
# 1. Export and inspect graph
# ============================================================
print("\n=== Exporting (log_amp) ===")
model.export_only(example_x, use_log_amp=True)
params_list = list(model.params)

# Inspect the exported FX graph (single-sample, before vmap)
graph = model._exported_module.graph
op_counts = Counter()
call_fn_targets = Counter()
for node in graph.nodes:
    op_counts[node.op] += 1
    if node.op == 'call_function':
        target = str(node.target)
        # Shorten torch.ops.aten.xxx to aten.xxx
        target = target.replace(
            'torch.ops.aten.', 'aten.',
        )
        call_fn_targets[target] += 1

print(f"\nFX graph: {len(list(graph.nodes))} total nodes")
print(f"  Node types: {dict(op_counts)}")
print(f"\n  Top 20 aten ops:")
for op, count in call_fn_targets.most_common(20):
    print(f"    {op}: {count}")

# Check for linalg ops (likely suspects)
linalg_ops = {
    k: v for k, v in call_fn_targets.items()
    if 'linalg' in k or 'eigh' in k or 'svd' in k
    or 'cholesky' in k or 'solve' in k
}
if linalg_ops:
    print(f"\n  Linalg ops: {linalg_ops}")
else:
    print("\n  No linalg ops found in graph")

# Check for indexing ops
index_ops = {
    k: v for k, v in call_fn_targets.items()
    if 'index' in k or 'gather' in k or 'scatter' in k
    or 'select' in k
}
if index_ops:
    print(f"  Indexing ops: {index_ops}")

# ============================================================
# 2. Test B=1 (single sample — vmap is trivially batched)
# ============================================================
print("\n=== Test B=1 (single sample) ===")
fxs_1 = random_initial_config(
    N_f, N_sites, seed=0,
).to(device).unsqueeze(0)  # (1, N_sites)

with torch.inference_mode():
    eager_signs_1, eager_log_1 = model.forward_log(fxs_1)
print(f"Eager B=1:  sign={eager_signs_1}, log={eager_log_1}")

# Reset export for compile
model._exported = False
model._compiled = False
model.export_only(example_x, use_log_amp=True)

compiled_fn = torch.compile(
    model._vmapped_exported,
    backend='inductor',
)

with torch.inference_mode():
    t0 = time.time()
    ind_signs_1, ind_log_1 = compiled_fn(
        fxs_1, *params_list,
    )
    torch.cuda.synchronize()
    t1 = time.time() - t0

diff_1 = (ind_log_1 - eager_log_1).abs()
ok_1 = diff_1.max().item() < 1e-3
print(
    f"Inductor B=1: sign={ind_signs_1}, "
    f"log={ind_log_1}"
)
print(
    f"diff={diff_1.max():.2e}, "
    f"{'PASS' if ok_1 else 'FAIL'}, time={t1:.1f}s"
)

# ============================================================
# 3. Test B=4 for comparison
# ============================================================
print("\n=== Test B=4 ===")
fxs_4 = torch.stack([
    random_initial_config(N_f, N_sites, seed=s).to(device)
    for s in range(4)
])

with torch.inference_mode():
    eager_signs_4, eager_log_4 = model.forward_log(fxs_4)
print(f"Eager B=4:  signs={eager_signs_4}")
print(f"            log={eager_log_4}")

# Need to recompile for new batch size
compiled_fn_4 = torch.compile(
    model._vmapped_exported,
    backend='inductor',
)
with torch.inference_mode():
    t0 = time.time()
    ind_signs_4, ind_log_4 = compiled_fn_4(
        fxs_4, *params_list,
    )
    torch.cuda.synchronize()
    t4 = time.time() - t0

diff_4 = (ind_log_4 - eager_log_4).abs()
ok_4 = diff_4.max().item() < 1e-3
print(f"Inductor B=4: signs={ind_signs_4}")
print(f"              log={ind_log_4}")
print(
    f"max diff={diff_4.max():.2e}, "
    f"{'PASS' if ok_4 else 'FAIL'}, time={t4:.1f}s"
)

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 50}")
print(f"B=1: {'PASS' if ok_1 else 'FAIL'} "
      f"(diff={diff_1.max():.2e})")
print(f"B=4: {'PASS' if ok_4 else 'FAIL'} "
      f"(diff={diff_4.max():.2e})")
if ok_1 and not ok_4:
    print("=> BUG IS IN BATCHED OP LOWERING (vmap)")
elif not ok_1 and not ok_4:
    print("=> BUG IN SINGLE-SAMPLE OP LOWERING")
elif ok_1 and ok_4:
    print("=> Both pass (unexpected)")
print(f"{'=' * 50}")
