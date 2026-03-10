"""Find the exact FX graph node where inductor diverges.

Phase 1: Test single-sample module (no vmap) through inductor.
Phase 2: Binary search by injecting correct constants for the
         first half of the graph, testing if the second half
         produces correct output.

Strategy: instead of truncating the graph (which breaks export
pytree handling), we REPLACE first-half nodes with pre-recorded
eager constants. The output node stays unchanged, so pytree
spec works. If the output matches eager → second half is OK →
error is in first half. Recurse.

Run:
    python debug_inductor_graph_bisect.py
"""
import copy
import gc
import time

import torch
import torch.fx
import torch.nn as nn

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
D, chi = 10, -1  # chi=-1: exact contraction, no linalg

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

# Export
model.export_only(example_x, use_log_amp=True)
params_list = list(model.params)
gm = model._exported_module

# Print graph op breakdown
from collections import Counter
op_counts = Counter()
for node in gm.graph.nodes:
    if node.op == 'call_function':
        target = str(node.target).replace(
            'torch.ops.aten.', 'aten.',
        )
        op_counts[target] += 1
print(f"\nFX graph: "
      f"{sum(1 for _ in gm.graph.nodes)} nodes, "
      f"{sum(op_counts.values())} call_function")
print("Top ops:")
for op, count in op_counts.most_common(15):
    print(f"  {op}: {count}")

# ============================================================
# Phase 1: Single-sample module (no vmap) through inductor
# ============================================================
print("\n=== Phase 1: Single-sample (no vmap) ===")

with torch.inference_mode():
    eager_out = gm(example_x, *params_list)

# Get reference output values
if isinstance(eager_out, tuple):
    eager_ref = [
        v.detach().clone() if isinstance(v, torch.Tensor)
        else v
        for v in eager_out
    ]
else:
    eager_ref = eager_out.detach().clone()

print(f"Eager single: {eager_out}")

torch._dynamo.reset()
compiled_single = torch.compile(gm, backend='inductor')
with torch.inference_mode():
    t0 = time.time()
    ind_out = compiled_single(example_x, *params_list)
    torch.cuda.synchronize()
    dt = time.time() - t0

if isinstance(eager_out, tuple):
    diffs = []
    for i, (e, c) in enumerate(zip(eager_out, ind_out)):
        if isinstance(e, torch.Tensor):
            d = (e - c).abs().max().item()
            diffs.append(d)
            print(f"  output[{i}]: eager={e}, ind={c}, "
                  f"diff={d:.2e}")
    max_diff = max(diffs)
else:
    max_diff = (eager_out - ind_out).abs().max().item()
    print(f"  diff={max_diff:.2e}")

phase1_ok = max_diff < 1e-6
print(
    f"Phase 1: {'PASS' if phase1_ok else 'FAIL'} "
    f"(diff={max_diff:.2e}, time={dt:.1f}s)"
)

if phase1_ok:
    print(
        "\nSingle-sample inductor is CORRECT. "
        "Bug is in vmap+inductor interaction."
    )
    print("Need different bisection strategy.")
    exit(0)

# ============================================================
# Phase 2: FX graph bisection via constant injection
# ============================================================
print("\n=== Phase 2: FX graph bisection ===")
print("Recording eager intermediate values...")


class EagerRecorder(torch.fx.Interpreter):
    """Run graph eagerly, record all outputs."""

    def __init__(self, module):
        super().__init__(module)
        self.recorded = {}

    def run_node(self, n):
        result = super().run_node(n)
        self.recorded[n.name] = result
        return result


recorder = EagerRecorder(gm)
with torch.inference_mode():
    recorder.run(example_x, *params_list)
eager_values = recorder.recorded

# Build ordered list of call_function nodes with tensor
# outputs (these are our bisection checkpoints)
checkpoints = []
for node in gm.graph.nodes:
    if node.op != 'call_function':
        continue
    val = eager_values.get(node.name)
    if isinstance(val, torch.Tensor) and val.numel() > 0:
        checkpoints.append(node.name)

print(f"Total checkpoints: {len(checkpoints)}")


def compare_output(result, eager_ref):
    """Compare compiled output vs eager reference."""
    if isinstance(result, tuple) and isinstance(
        eager_ref, list
    ):
        diffs = []
        for e, c in zip(eager_ref, result):
            if isinstance(e, torch.Tensor) and isinstance(
                c, torch.Tensor
            ):
                diffs.append(
                    (c - e).abs().max().item()
                )
        return max(diffs) if diffs else 0.0
    elif isinstance(result, torch.Tensor):
        return (result - eager_ref).abs().max().item()
    return 0.0


def inject_constants_and_test(
    gm_orig, split_idx, checkpoints, eager_values,
    eager_ref, inputs, threshold=1e-6,
):
    """Replace first-half checkpoint nodes with constants.

    Nodes before split_idx are replaced with their eager
    values. Output node is UNCHANGED. Compile the modified
    graph with inductor and compare output.

    Returns: (output_diff, n_replaced, n_remaining, time)
    """
    new_gm = copy.deepcopy(gm_orig)
    g = new_gm.graph

    # Names of nodes in the first half
    first_half_names = set(checkpoints[:split_idx])

    # Find which first-half nodes have users outside
    # the first half (boundary nodes). Only these need
    # to be replaced — the rest become dead code.
    n_replaced = 0
    nodes_by_name = {n.name: n for n in g.nodes}

    for node in list(g.nodes):
        if node.name not in first_half_names:
            continue
        val = eager_values.get(node.name)
        if not isinstance(val, torch.Tensor):
            continue

        # Check if any user is outside the first half
        has_outside_user = False
        for user in node.users:
            if (user.name not in first_half_names
                    and user.op != 'output'):
                has_outside_user = True
                break
            if user.op == 'output':
                has_outside_user = True
                break

        if not has_outside_user:
            continue

        # Replace this node with a constant
        attr_name = f"_const_{n_replaced}"
        new_gm.register_buffer(
            attr_name, val.detach().clone(),
        )
        with g.inserting_before(node):
            const_node = g.get_attr(attr_name)
        node.replace_all_uses_with(const_node)
        g.erase_node(node)
        n_replaced += 1

    g.eliminate_dead_code()
    g.lint()
    new_gm.recompile()

    n_remaining = sum(
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
    return diff, n_replaced, n_remaining, dt


# Binary search
inputs = (example_x, *params_list)
lo, hi = 0, len(checkpoints) - 1

print(
    f"\nBinary search over {len(checkpoints)} "
    f"checkpoints..."
)
print(
    f"{'idx':>6} {'split_node':>40} "
    f"{'out_diff':>12} {'status':>10} "
    f"{'replaced':>8} {'remaining':>9} {'time':>6}"
)
print("-" * 100)

while lo < hi:
    mid = (lo + hi) // 2
    split_name = checkpoints[mid]

    try:
        diff, n_rep, n_rem, dt = (
            inject_constants_and_test(
                gm, mid, checkpoints, eager_values,
                eager_ref, inputs,
            )
        )
    except Exception as e:
        print(f"{mid:6d} {split_name:>40} "
              f"{'ERROR':>12} {str(e)[:40]}")
        # On error, assume second half has issues
        hi = mid
        continue

    ok = diff < 1e-3
    status = "2nd OK" if ok else "2nd BAD"
    print(
        f"{mid:6d} {split_name:>40} "
        f"{diff:12.2e} {status:>10} "
        f"{n_rep:8d} {n_rem:9d} {dt:6.1f}s"
    )

    if ok:
        # Second half is correct when given correct inputs
        # → error is in first half
        hi = mid
    else:
        # Second half produces wrong output even with
        # correct inputs → error is in second half
        lo = mid + 1

# Found the first diverging checkpoint
first_bad = checkpoints[lo]
print(f"\n{'=' * 60}")
print(f"First divergence at checkpoint index {lo}:")
print(f"  node: {first_bad}")

# Get the node's op and its neighbors
for node in gm.graph.nodes:
    if node.name == first_bad:
        print(f"  op: {node.target}")
        # Print neighbors (10 before, 10 after)
        all_nodes = list(gm.graph.nodes)
        idx = all_nodes.index(node)
        print(f"\n  Context (nodes {idx-10} to {idx+10}):")
        for i in range(max(0, idx - 10),
                       min(len(all_nodes), idx + 11)):
            n = all_nodes[i]
            marker = " >>>" if n.name == first_bad \
                else "    "
            if n.op == 'call_function':
                target = str(n.target).replace(
                    'torch.ops.aten.', 'aten.',
                )
                val = eager_values.get(n.name)
                shape = ""
                if isinstance(val, torch.Tensor):
                    shape = f" {list(val.shape)}"
                print(
                    f"{marker} [{i}] {n.name}: "
                    f"{target}{shape}"
                )
            elif n.op == 'get_attr':
                print(
                    f"{marker} [{i}] {n.name}: "
                    f"get_attr({n.target})"
                )
        break

print(f"{'=' * 60}")
