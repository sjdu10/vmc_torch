"""
Test torch.compile and torch.jit.trace on flat-backend fPEPS amplitude.
Follows the pattern from symmray/examples/batch gpu fermionic amplitudes torch.ipynb.

Run with: torchrun --nproc_per_node=1 GPU_test_compile_flat.py
"""
import os
import time
import pickle
import torch
import torch.nn as nn
import torch.distributed as dist
import autoray as ar
import quimb as qu
import quimb.tensor as qtn

from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    random_initial_config,
)
from vmc_torch.experiment.vmap.GPU.models import fPEPS_Model_GPU
from vmc_torch.experiment.vmap.vmap_torch_utils import (
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
params = pickle.load(open(
    f'{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/'
    'peps_su_params_U1SU.pkl', 'rb'
))
skeleton = pickle.load(open(
    f'{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/'
    'peps_skeleton_U1SU.pkl', 'rb'
))
peps = qtn.unpack(params, skeleton)
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


# ============================================================
# Test 1: Baseline (no compile)
# ============================================================
print("=" * 60)
print("TEST 1: Baseline fPEPS_Model_GPU (chi=-1, no compile)")
print("=" * 60)

model = fPEPS_Model_GPU(
    tn=peps, max_bond=-1, dtype=torch.float64,
    contract_boundary_opts={
        'mode': 'mps', 'equalize_norms': 1.0, 'canonize': True,
    }
)
model.to(device)

with torch.inference_mode():
    out_ref = model(fxs)
torch.cuda.synchronize()

for trial in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        out = model(fxs)
    torch.cuda.synchronize()
    print(f"  Call {trial}: {time.time()-t0:.4f}s")


# ============================================================
# Test 2: torch.compile on the vmapped forward
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: torch.compile on model.forward")
print("=" * 60)

model2 = fPEPS_Model_GPU(
    tn=peps, max_bond=-1, dtype=torch.float64,
    contract_boundary_opts={
        'mode': 'mps', 'equalize_norms': 1.0, 'canonize': True,
    }
)
model2.to(device)

# Warmup eager
with torch.inference_mode():
    _ = model2(fxs)
torch.cuda.synchronize()

# Compile the whole module
compiled_model2 = torch.compile(model2, fullgraph=False, mode='default')
t_comp0 = time.time()
with torch.inference_mode():
    try:
        out2 = compiled_model2(fxs)
        torch.cuda.synchronize()
        t_comp = time.time() - t_comp0
        print(f"  Compile + first call: {t_comp:.2f}s")
        print(f"  Output matches: {torch.allclose(out2, out_ref)}")

        for trial in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                out2 = compiled_model2(fxs)
            torch.cuda.synchronize()
            print(f"  Call {trial}: {time.time()-t0:.4f}s")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")


# ============================================================
# Test 3: torch.jit.trace — traces tensor ops, not bytecode
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: torch.jit.trace on model forward")
print("=" * 60)

model3 = fPEPS_Model_GPU(
    tn=peps, max_bond=-1, dtype=torch.float64,
    contract_boundary_opts={
        'mode': 'mps', 'equalize_norms': 1.0, 'canonize': True,
    }
)
model3.to(device)

with torch.inference_mode():
    _ = model3(fxs)  # warmup
torch.cuda.synchronize()

print("\n--- torch.jit.trace ---")
try:
    with torch.inference_mode():
        traced_model = torch.jit.trace(model3, fxs, check_trace=False)

    # Warmup traced
    with torch.inference_mode():
        out3 = traced_model(fxs)
    torch.cuda.synchronize()
    print(f"  Output matches: {torch.allclose(out3, out_ref)}")

    # Benchmark
    for trial in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out3 = traced_model(fxs)
        torch.cuda.synchronize()
        print(f"  Traced call {trial}: {time.time()-t0:.4f}s")

    # Test with different input (does it generalize?)
    fxs2 = torch.stack([
        random_initial_config(N_f, nsites, seed=100 + i)
        for i in range(B)
    ]).to(device)
    with torch.inference_mode():
        out3_new = traced_model(fxs2)
        out_ref_new = model(fxs2)
    print(f"  Different input matches: "
          f"{torch.allclose(out3_new, out_ref_new)}")

except Exception as e:
    print(f"  torch.jit.trace FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()


# ============================================================
# Test 4: Dynamo graph break analysis
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Dynamo graph break analysis")
print("=" * 60)

import torch._dynamo as dynamo
dynamo.reset()

model4 = fPEPS_Model_GPU(
    tn=peps, max_bond=-1, dtype=torch.float64,
    contract_boundary_opts={
        'mode': 'mps', 'equalize_norms': 1.0, 'canonize': True,
    }
)
model4.to(device)

with torch.inference_mode():
    _ = model4(fxs)  # warmup
torch.cuda.synchronize()

print("\n--- dynamo.explain on model(fxs[:1]) ---")
try:
    dynamo.reset()
    explanation = dynamo.explain(model4)(fxs[:1])
    print(f"  Graph breaks: {explanation.graph_break_count}")
    print(f"  Compiled regions: {explanation.graph_count}")
    if hasattr(explanation, 'break_reasons') and explanation.break_reasons:
        print("  Break reasons:")
        for i, reason in enumerate(explanation.break_reasons[:15]):
            print(f"    {i}: {reason}")
except Exception as e:
    print(f"  explain() failed: {type(e).__name__}: {e}")


# ============================================================
# Test 5: compile(vmap(fn)) vs vmap(compile(fn))
# order matters for torch.compile + vmap interaction
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: Compile + vmap ordering")
print("=" * 60)

params5, skeleton5 = qtn.pack(peps)

def amp_fn(x, params):
    tn = qtn.unpack(params, skeleton5)
    tnx = tn.isel({
        tn.site_ind(site): x[i]
        for i, site in enumerate(tn.sites)
    })
    return tnx.contract()

# A) vmap first, then compile the vmapped function
print("\n--- A) compile(vmap(amp_fn)) ---")
vf_a = torch.vmap(amp_fn, in_dims=(0, None))
vf_a_compiled = torch.compile(vf_a, fullgraph=False, mode='default')
t0 = time.time()
with torch.inference_mode():
    try:
        out_a = vf_a_compiled(fxs, params5)
        torch.cuda.synchronize()
        print(f"  Compile + first call: {time.time()-t0:.2f}s")
        print(f"  Output matches: {torch.allclose(out_a, out_ref)}")
        for trial in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            out_a = vf_a_compiled(fxs, params5)
            torch.cuda.synchronize()
            print(f"  Call {trial}: {time.time()-t0:.4f}s")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

# B) compile first, then vmap
print("\n--- B) vmap(compile(amp_fn)) ---")
dynamo.reset()
amp_fn_compiled = torch.compile(amp_fn, fullgraph=False, mode='default')
vf_b = torch.vmap(amp_fn_compiled, in_dims=(0, None))
t0 = time.time()
with torch.inference_mode():
    try:
        out_b = vf_b(fxs, params5)
        torch.cuda.synchronize()
        print(f"  Compile + first call: {time.time()-t0:.2f}s")
        print(f"  Output matches: {torch.allclose(out_b, out_ref)}")
        for trial in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            out_b = vf_b(fxs, params5)
            torch.cuda.synchronize()
            print(f"  Call {trial}: {time.time()-t0:.4f}s")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Baseline (no compile):  ~{0.029:.4f}s per call")
print("Reference output[:5]:", out_ref[:5].tolist())
print()
print("Key: JAX JIT traces at XLA (array op) level and captures")
print("the full computation graph. torch.compile/dynamo traces")
print("Python bytecode, which breaks on quimb/symmray Python ops.")
print("torch.jit.trace captures tensor ops by running the function.")

dist.destroy_process_group()
