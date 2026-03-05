"""
Test torch.jit.trace on single-sample amplitude (no vmap).
Isolates whether the issue is vmap or quimb/symmray.

Run with: torchrun --nproc_per_node=1 GPU_test_trace_single.py
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

B = 50
fxs = torch.stack([
    random_initial_config(N_f, nsites, seed=42 + i)
    for i in range(B)
]).to(device)

# Pack params for functional-style call
params, skel = qtn.pack(peps)
params_flat, params_pytree = qu.utils.tree_flatten(params, get_ref=True)
params_tensors = [
    torch.as_tensor(x, dtype=torch.float64, device=device)
    for x in params_flat
]


# ============================================================
# Test 1: Single-sample amplitude (no vmap, no compile)
# ============================================================
print("=" * 60)
print("TEST 1: Single-sample amplitude function")
print("=" * 60)


def amplitude_single(x, *flat_params):
    """Single-sample amplitude: pure function of (x, params)."""
    p = qu.utils.tree_unflatten(list(flat_params), params_pytree)
    tn = qtn.unpack(p, skel)
    tnx = tn.isel({
        tn.site_ind(site): x[i]
        for i, site in enumerate(tn.sites)
    })
    return tnx.contract()


# Eager call
with torch.inference_mode():
    ref = amplitude_single(fxs[0], *params_tensors)
    print(f"  Eager result: {ref.item():.6e}")

# Benchmark eager single-sample
print("\n--- Eager single-sample ---")
for trial in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        _ = amplitude_single(fxs[0], *params_tensors)
    torch.cuda.synchronize()
    print(f"  Call {trial}: {time.time()-t0:.6f}s")


# ============================================================
# Test 2: torch.jit.trace on single sample (NO vmap)
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: torch.jit.trace on single-sample amplitude")
print("=" * 60)

try:
    with torch.inference_mode():
        traced_fn = torch.jit.trace(
            amplitude_single,
            (fxs[0], *params_tensors),
            check_trace=False,
        )
    print("  Trace succeeded!")

    # Test output
    with torch.inference_mode():
        out_traced = traced_fn(fxs[0], *params_tensors)
    print(f"  Traced result: {out_traced.item():.6e}")
    print(f"  Matches ref: {torch.allclose(out_traced, ref)}")

    # Test different input (same charge pattern)
    with torch.inference_mode():
        out_new = traced_fn(fxs[1], *params_tensors)
        ref_new = amplitude_single(fxs[1], *params_tensors)
    print(f"  Different input - traced: {out_new.item():.6e}")
    print(f"  Different input - eager:  {ref_new.item():.6e}")
    print(f"  Different input matches: "
          f"{torch.allclose(out_new, ref_new)}")

    # Benchmark
    print("\n--- Benchmark traced single-sample ---")
    for trial in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            _ = traced_fn(fxs[0], *params_tensors)
        torch.cuda.synchronize()
        print(f"  Traced call {trial}: {time.time()-t0:.6f}s")

except Exception as e:
    print(f"  torch.jit.trace FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()


# ============================================================
# Test 3: torch.compile on single sample (no vmap)
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: torch.compile on single-sample amplitude")
print("=" * 60)

import torch._dynamo as dynamo
dynamo.reset()

compiled_single = torch.compile(
    amplitude_single, fullgraph=False, mode='default'
)

t_comp0 = time.time()
with torch.inference_mode():
    try:
        out_c = compiled_single(fxs[0], *params_tensors)
        torch.cuda.synchronize()
        print(f"  Compile + first call: {time.time()-t_comp0:.2f}s")
        print(f"  Compiled result: {out_c.item():.6e}")
        print(f"  Matches ref: {torch.allclose(out_c, ref)}")

        # Benchmark
        for trial in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                _ = compiled_single(fxs[0], *params_tensors)
            torch.cuda.synchronize()
            print(f"  Call {trial}: {time.time()-t0:.6f}s")
    except Exception as e:
        print(f"  torch.compile FAILED: {type(e).__name__}: {e}")

# Dynamo explain
print("\n--- Dynamo explain on single sample ---")
dynamo.reset()
try:
    explanation = dynamo.explain(amplitude_single)(
        fxs[0], *params_tensors
    )
    print(f"  Graph breaks: {explanation.graph_break_count}")
    print(f"  Compiled regions: {explanation.graph_count}")
    if hasattr(explanation, 'break_reasons') and explanation.break_reasons:
        print("  Break reasons (first 10):")
        for i, reason in enumerate(explanation.break_reasons[:10]):
            print(f"    {i}: {reason}")
except Exception as e:
    print(f"  explain() failed: {type(e).__name__}: {e}")


# ============================================================
# Test 4: vmap on traced/compiled single-sample
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: vmap wrapping traced/compiled single-sample")
print("=" * 60)

# If trace succeeded, try vmap(traced)
if 'traced_fn' in dir():
    print("\n--- vmap(traced_fn) ---")
    try:
        vf_traced = torch.vmap(
            traced_fn,
            in_dims=(0, *([None] * len(params_tensors))),
        )
        with torch.inference_mode():
            out_vt = vf_traced(fxs, *params_tensors)
        print(f"  vmap(traced) succeeded!")
        print(f"  Output[:3]: {out_vt[:3].tolist()}")
    except Exception as e:
        print(f"  vmap(traced) FAILED: {type(e).__name__}: {e}")

# Try vmap on the bare function (should work — this is what model does)
print("\n--- vmap(amplitude_single) [baseline] ---")
vf_bare = torch.vmap(
    amplitude_single,
    in_dims=(0, *([None] * len(params_tensors))),
)
with torch.inference_mode():
    out_vb = vf_bare(fxs, *params_tensors)
    print(f"  vmap(bare) output[:3]: {out_vb[:3].tolist()}")

# Benchmark vmap(bare) for reference
for trial in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        _ = vf_bare(fxs, *params_tensors)
    torch.cuda.synchronize()
    print(f"  Call {trial}: {time.time()-t0:.4f}s")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("The question: JAX JIT traces at array-op level and works.")
print("torch.jit.trace also traces tensor ops. Why doesn't it work?")
print()
print("Hypothesis: the failure was torch.jit.trace + vmap.")
print("Without vmap, trace may work on single samples.")
print("If so, we need: trace(single_sample) then vmap(traced).")

dist.destroy_process_group()
