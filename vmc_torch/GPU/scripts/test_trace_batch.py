"""
Test ways to batch the JIT-traced single-sample amplitude.
Since torch.jit.trace gives 5ms/sample (4.6x faster than eager 23ms),
can we batch it efficiently?

Run with: torchrun --nproc_per_node=1 GPU_test_trace_batch.py
"""
import os
import time
import pickle
import torch
import torch.distributed as dist
import autoray as ar
import quimb as qu
import quimb.tensor as qtn

from vmc_torch.GPU.vmc_utils import (
    random_initial_config,
)
from vmc_torch.GPU.torch_utils import (
    robust_svd_err_catcher_wrapper,
)

ar.register_function(
    'torch', 'linalg.svd',
    lambda x: robust_svd_err_catcher_wrapper(
        x, jitter=1e-16, driver=None
    ),
)

# Setup
if "RANK" not in os.environ:
    os.environ.update({
        "RANK": "0", "WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost", "MASTER_PORT": "12355",
        "LOCAL_RANK": "0",
    })
dist.init_process_group(backend="nccl", init_method="env://")
device = torch.device("cuda:0")
torch.cuda.set_device(0)
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)
torch.manual_seed(42)

# Load PEPS
Lx, Ly = 4, 2
nsites = Lx * Ly
N_f = nsites - 2
D = 4
pwd = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
    'vmc_torch/experiment/vmap/data'
)
params_pkl = pickle.load(open(
    f'{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/'
    'peps_su_params_U1SU.pkl', 'rb'
))
skeleton = pickle.load(open(
    f'{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/'
    'peps_skeleton_U1SU.pkl', 'rb'
))
peps = qtn.unpack(params_pkl, skeleton)
for ts in peps.tensors:
    ts.modify(data=ts.data.to_flat() * 4)
for site in peps.sites:
    peps[site].data._label = site
    peps[site].data.indices[-1]._linearmap = (
        (0, 0), (1, 0), (1, 1), (0, 1)
    )

params, skel = qtn.pack(peps)
params_flat, params_pytree = qu.utils.tree_flatten(params, get_ref=True)
params_tensors = [
    torch.as_tensor(x, dtype=torch.float64, device=device)
    for x in params_flat
]

B = 500
fxs = torch.stack([
    random_initial_config(N_f, nsites, seed=42 + i)
    for i in range(B)
]).to(device)


def amplitude_single(x, *flat_params):
    p = qu.utils.tree_unflatten(list(flat_params), params_pytree)
    tn = qtn.unpack(p, skel)
    tnx = tn.isel({
        tn.site_ind(site): x[i]
        for i, site in enumerate(tn.sites)
    })
    return tnx.contract()


# ============================================================
# Baseline: vmap(eager)
# ============================================================
print("=" * 60)
print("BASELINE: vmap(eager) on B=500")
print("=" * 60)

vf_bare = torch.vmap(
    amplitude_single,
    in_dims=(0, *([None] * len(params_tensors))),
)

# Warmup
with torch.inference_mode():
    out_ref = vf_bare(fxs, *params_tensors)
torch.cuda.synchronize()

for trial in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        out = vf_bare(fxs, *params_tensors)
    torch.cuda.synchronize()
    print(f"  vmap(eager) B={B}: {time.time()-t0:.4f}s")


# ============================================================
# Approach 1: Sequential loop with traced function
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 1: Sequential traced loop")
print("=" * 60)

with torch.inference_mode():
    traced_fn = torch.jit.trace(
        amplitude_single,
        (fxs[0], *params_tensors),
        check_trace=False,
    )

# Warmup traced
for _ in range(3):
    with torch.inference_mode():
        _ = traced_fn(fxs[0], *params_tensors)
torch.cuda.synchronize()

# Sequential loop
for trial in range(2):
    torch.cuda.synchronize()
    t0 = time.time()
    results = []
    with torch.inference_mode():
        for i in range(B):
            results.append(traced_fn(fxs[i], *params_tensors))
        out_seq = torch.stack(results)
    torch.cuda.synchronize()
    print(f"  Sequential traced B={B}: {time.time()-t0:.4f}s")
    print(f"    Matches: {torch.allclose(out_seq, out_ref)}")


# ============================================================
# Approach 2: torch.export + vmap
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 2: torch.export then vmap")
print("=" * 60)

try:
    from torch.export import export

    # Export the single-sample function
    # Need to wrap as a module for export
    class AmpModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, *flat_params):
            return amplitude_single(x, *flat_params)

    amp_mod = AmpModule()
    with torch.inference_mode():
        exported = export(
            amp_mod,
            (fxs[0], *params_tensors),
        )
    print(f"  Export succeeded!")
    print(f"  Graph: {exported.graph}")

    # Try to vmap the exported module
    try:
        vf_exported = torch.vmap(
            exported.module(),
            in_dims=(0, *([None] * len(params_tensors))),
        )
        with torch.inference_mode():
            out_exp = vf_exported(fxs, *params_tensors)
        print(f"  vmap(exported) output matches: "
              f"{torch.allclose(out_exp, out_ref)}")
    except Exception as e:
        print(f"  vmap(exported) FAILED: {type(e).__name__}: {e}")

except Exception as e:
    print(f"  torch.export FAILED: {type(e).__name__}: {e}")


# ============================================================
# Approach 3: Parallel traced calls via CUDA streams
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 3: Parallel traced via CUDA streams")
print("=" * 60)

n_streams = 4
streams = [torch.cuda.Stream() for _ in range(n_streams)]

for trial in range(2):
    torch.cuda.synchronize()
    t0 = time.time()
    results = [None] * B
    with torch.inference_mode():
        for i in range(B):
            s = streams[i % n_streams]
            with torch.cuda.stream(s):
                results[i] = traced_fn(fxs[i], *params_tensors)
    torch.cuda.synchronize()
    out_par = torch.stack(results)
    print(f"  Parallel traced ({n_streams} streams) B={B}: "
          f"{time.time()-t0:.4f}s")
    print(f"    Matches: {torch.allclose(out_par, out_ref)}")


# ============================================================
# Approach 4: Pre-index then batch contract
# Skip quimb for the contraction, do it in pure PyTorch.
# Pre-extract the contraction structure from one eager call.
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 4: Extract traced graph ops and count")
print("=" * 60)

# Print the traced graph to see what ops are in it
graph = traced_fn.graph
all_nodes = list(graph.nodes())
op_counts = {}
for node in all_nodes:
    kind = node.kind()
    op_counts[kind] = op_counts.get(kind, 0) + 1

print(f"  Total graph nodes: {len(all_nodes)}")
print(f"  Unique op types: {len(op_counts)}")
print("  Top 10 ops by count:")
for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"    {op}: {count}")

# Check if the graph uses dynamic shapes
print(f"\n  Graph inputs: {len(list(graph.inputs()))}")
print(f"  Graph outputs: {len(list(graph.outputs()))}")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"vmap(eager) B={B}: ~{0.29:.3f}s  (Python overhead per op)")
print(f"Traced single:      ~0.005s/sample")
print(f"Sequential traced:  ~{B*0.005:.2f}s  (no parallelism)")
print(f"Parallel streams:   TBD")
print()
print("The JIT trace captures the full computation graph")
print("as a fixed sequence of ~N tensor ops, eliminating")
print("quimb/symmray Python overhead.")
print()
print("Next: if vmap(traced) doesn't work, consider:")
print("  1. Batching the traced ops manually")
print("  2. Using torch._C._jit_pass_inline to get ops,")
print("     then building a batched version")

dist.destroy_process_group()
