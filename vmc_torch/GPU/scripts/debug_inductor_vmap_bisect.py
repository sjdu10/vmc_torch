"""Bisect vmap+inductor interaction bug.

Single-sample inductor is correct. Bug is in how inductor
handles vmap-batched ops for D=10.

Strategy: export the vmapped function with torch.export to
get a clean FX graph, then bisect via constant injection.

Run:
    python debug_inductor_vmap_bisect.py
"""
import copy
import gc
import time
from collections import Counter

import torch
import torch.fx
import torch.nn as nn
from torch.export import export

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
D, chi = 10, -1
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

example_x = random_initial_config(
    N_f, N_sites, seed=0,
).to(device)

fxs = torch.stack([
    random_initial_config(N_f, N_sites, seed=s).to(device)
    for s in range(B)
])

print(f"System: {Lx}x{Ly}, D={D}, chi={chi}, B={B}")

# ============================================================
# Step 1: Export the vmapped function
# ============================================================
print("\n=== Step 1: Export vmapped function ===")

# First export single-sample to get the FX graph
model.export_only(example_x, use_log_amp=True)
params_list = list(model.params)


# Now export the vmapped version
class _VmapModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, x, *flat_params):
        return self._fn(x, *flat_params)


with torch.no_grad():
    exported_vmap = export(
        _VmapModule(model._vmapped_exported),
        (fxs, *params_list),
    )
vmap_gm = exported_vmap.module()

# Move CPU constants to GPU
for node in vmap_gm.graph.nodes:
    if node.op != 'get_attr':
        continue
    parts = node.target.split('.')
    parent = vmap_gm
    for p in parts[:-1]:
        parent = getattr(parent, p)
    leaf = parts[-1]
    obj = getattr(parent, leaf)
    if (isinstance(obj, torch.Tensor)
            and obj.device.type == 'cpu'):
        setattr(parent, leaf, obj.to(device))

# Print graph stats
op_counts = Counter()
for node in vmap_gm.graph.nodes:
    if node.op == 'call_function':
        target = str(node.target).replace(
            'torch.ops.aten.', 'aten.',
        )
        op_counts[target] += 1
n_total = sum(1 for _ in vmap_gm.graph.nodes)
print(f"Vmap'd graph: {n_total} nodes, "
      f"{sum(op_counts.values())} call_function")
print("Top 15 ops:")
for op, count in op_counts.most_common(15):
    print(f"  {op}: {count}")

# ============================================================
# Step 2: Verify eager vs inductor
# ============================================================
print("\n=== Step 2: Verify ===")

inputs = (fxs, *params_list)
with torch.inference_mode():
    eager_out = vmap_gm(*inputs)
print(f"Eager: {eager_out}")

torch._dynamo.reset()
compiled = torch.compile(vmap_gm, backend='inductor')
with torch.inference_mode():
    t0 = time.time()
    ind_out = compiled(*inputs)
    torch.cuda.synchronize()
    dt = time.time() - t0

if isinstance(eager_out, tuple):
    diffs = []
    for i, (e, c) in enumerate(zip(eager_out, ind_out)):
        if isinstance(e, torch.Tensor):
            d = (e - c).abs().max().item()
            diffs.append(d)
            print(f"  output[{i}]: diff={d:.2e}")
    max_diff = max(diffs) if diffs else 0.0
else:
    max_diff = (eager_out - ind_out).abs().max().item()

print(f"Max diff: {max_diff:.2e} "
      f"({'FAIL' if max_diff > 1e-3 else 'PASS'}) "
      f"time={dt:.1f}s")

if max_diff < 1e-3:
    print("Inductor passes — unexpected! Exiting.")
    exit(0)

# Save eager reference for later comparison
if isinstance(eager_out, tuple):
    eager_ref = [
        v.detach().clone() if isinstance(v, torch.Tensor)
        else v
        for v in eager_out
    ]
else:
    eager_ref = eager_out.detach().clone()

# ============================================================
# Step 3: Record eager intermediates
# ============================================================
print("\n=== Step 3: Record intermediates ===")


class EagerRecorder(torch.fx.Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.recorded = {}

    def run_node(self, n):
        result = super().run_node(n)
        self.recorded[n.name] = result
        return result


recorder = EagerRecorder(vmap_gm)
with torch.inference_mode():
    recorder.run(*inputs)
eager_values = recorder.recorded

# Build checkpoint list (tensor-valued call_function nodes)
checkpoints = []
for node in vmap_gm.graph.nodes:
    if node.op != 'call_function':
        continue
    val = eager_values.get(node.name)
    if isinstance(val, torch.Tensor) and val.numel() > 0:
        checkpoints.append(node.name)

print(f"Total checkpoints: {len(checkpoints)}")

# ============================================================
# Step 4: Binary search via constant injection
# ============================================================
print("\n=== Step 4: Binary search ===")


def compare_output(result, ref):
    if isinstance(result, tuple) and isinstance(ref, list):
        diffs = []
        for e, c in zip(ref, result):
            if isinstance(e, torch.Tensor) and isinstance(
                c, torch.Tensor
            ):
                diffs.append((c - e).abs().max().item())
        return max(diffs) if diffs else 0.0
    elif isinstance(result, torch.Tensor):
        return (result - ref).abs().max().item()
    return 0.0


def inject_and_test(gm_orig, split_idx, checkpoints,
                    eager_values, eager_ref, inputs):
    """Replace first-half nodes with constants, compile rest
    with inductor, compare output."""
    new_gm = copy.deepcopy(gm_orig)
    g = new_gm.graph
    first_half = set(checkpoints[:split_idx])

    n_replaced = 0
    for node in list(g.nodes):
        if node.name not in first_half:
            continue
        val = eager_values.get(node.name)
        if not isinstance(val, torch.Tensor):
            continue
        has_outside_user = any(
            u.name not in first_half for u in node.users
        )
        if not has_outside_user:
            continue
        attr = f"_c{n_replaced}"
        new_gm.register_buffer(attr, val.detach().clone())
        with g.inserting_before(node):
            const = g.get_attr(attr)
        node.replace_all_uses_with(const)
        g.erase_node(node)
        n_replaced += 1

    g.eliminate_dead_code()
    g.lint()
    new_gm.recompile()

    n_rem = sum(
        1 for n in g.nodes if n.op == 'call_function'
    )

    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()

    compiled = torch.compile(new_gm, backend='inductor')
    with torch.inference_mode():
        t0 = time.time()
        result = compiled(*inputs)
        torch.cuda.synchronize()
        dt = time.time() - t0

    diff = compare_output(result, eager_ref)
    return diff, n_replaced, n_rem, dt


lo, hi = 0, len(checkpoints) - 1
print(f"Binary search over {len(checkpoints)} "
      f"checkpoints...")
print(
    f"{'idx':>6} {'split':>40} {'diff':>12} "
    f"{'status':>10} {'repl':>5} {'rem':>6} "
    f"{'time':>6}"
)
print("-" * 90)

while lo < hi:
    mid = (lo + hi) // 2
    name = checkpoints[mid]

    try:
        diff, n_rep, n_rem, dt = inject_and_test(
            vmap_gm, mid, checkpoints,
            eager_values, eager_ref, inputs,
        )
    except Exception as e:
        print(f"{mid:6d} {name:>40} ERROR: {e}")
        hi = mid
        continue

    ok = diff < 1e-3
    status = "2nd OK" if ok else "2nd BAD"
    print(
        f"{mid:6d} {name:>40} {diff:12.2e} "
        f"{status:>10} {n_rep:5d} {n_rem:6d} "
        f"{dt:6.1f}s"
    )

    if ok:
        # Second half correct → error in first half
        hi = mid
    else:
        # Second half wrong → error in second half
        lo = mid + 1

# ============================================================
# Step 5: Report
# ============================================================
first_bad = checkpoints[lo]
print(f"\n{'=' * 60}")
print(f"First divergence at checkpoint {lo}:")
print(f"  node: {first_bad}")

for node in vmap_gm.graph.nodes:
    if node.name == first_bad:
        print(f"  op: {node.target}")
        all_nodes = list(vmap_gm.graph.nodes)
        idx = all_nodes.index(node)
        print(f"\n  Context ({idx-10}..{idx+10}):")
        for i in range(max(0, idx - 10),
                       min(len(all_nodes), idx + 11)):
            n = all_nodes[i]
            m = " >>>" if n.name == first_bad else "    "
            if n.op == 'call_function':
                t = str(n.target).replace(
                    'torch.ops.aten.', 'aten.',
                )
                val = eager_values.get(n.name)
                s = ""
                if isinstance(val, torch.Tensor):
                    s = f" {list(val.shape)}"
                print(f"{m} [{i}] {n.name}: {t}{s}")
            elif n.op in ('get_attr', 'placeholder'):
                print(f"{m} [{i}] {n.name}: "
                      f"{n.op}({n.target})")
        break

print(f"{'=' * 60}")
