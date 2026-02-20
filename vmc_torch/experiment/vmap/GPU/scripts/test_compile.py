"""
Test torch.compile on fPEPS model forward pass.
Run with: torchrun --nproc_per_node=1 GPU_test_compile.py
"""
import os
import time
import pickle
import torch
import torch.distributed as dist
import autoray as ar
import quimb.tensor as qtn
import logging

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

B = 50  # small batch for compile test
fxs = torch.stack([
    random_initial_config(N_f, nsites, seed=42 + i)
    for i in range(B)
]).to(device)


def benchmark_model(chi_val, use_compile, compile_mode='default'):
    """Test model forward with given chi and compile settings."""
    model = fPEPS_Model_GPU(
        tn=peps, max_bond=chi_val, dtype=torch.float64,
        contract_boundary_opts={
            'mode': 'mps', 'equalize_norms': 1.0,
            'canonize': True,
        }
    )
    model.to(device)

    label = f"chi={chi_val}"
    if use_compile:
        label += f", compile({compile_mode})"
    else:
        label += ", no compile"

    print(f"\n--- {label} ---")

    # Warmup (no compile)
    with torch.inference_mode():
        _ = model(fxs)
    torch.cuda.synchronize()

    if use_compile:
        print("  Compiling...")
        t_comp0 = time.time()
        model.compile_model(mode=compile_mode)
        # First call triggers actual compilation
        with torch.inference_mode():
            try:
                _ = model(fxs)
                torch.cuda.synchronize()
                t_comp = time.time() - t_comp0
                print(f"  Compile + first call: {t_comp:.2f}s")
            except Exception as e:
                print(f"  COMPILE FAILED: {e}")
                return

    # Benchmark 3 calls
    for trial in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model(fxs)
        torch.cuda.synchronize()
        print(f"  Call {trial}: {time.time()-t0:.4f}s")


# ============================================================
# Test 1: chi=-1 (exact contraction, no boundary compression)
# ============================================================
print("=" * 60)
print("TEST: chi=-1 (exact contraction)")
print("=" * 60)
benchmark_model(chi_val=-1, use_compile=False)
benchmark_model(chi_val=-1, use_compile=True, compile_mode='default')

# ============================================================
# Test 2: chi=4 (boundary contraction with SVD)
# ============================================================
print("\n" + "=" * 60)
print("TEST: chi=4 (boundary contraction)")
print("=" * 60)
benchmark_model(chi_val=4, use_compile=False)
benchmark_model(chi_val=4, use_compile=True, compile_mode='default')

# ============================================================
# Test 3: Investigate graph breaks with TORCH_LOGS
# ============================================================
print("\n" + "=" * 60)
print("TEST: Graph break analysis (chi=-1, B=1)")
print("=" * 60)

model_debug = fPEPS_Model_GPU(
    tn=peps, max_bond=-1, dtype=torch.float64,
    contract_boundary_opts={
        'mode': 'mps', 'equalize_norms': 1.0,
        'canonize': True,
    }
)
model_debug.to(device)

# Count graph breaks using torch._dynamo
import torch._dynamo as dynamo
dynamo.reset()

# Use explain mode to count graph breaks
explanation = dynamo.explain(model_debug)(fxs[:1])
print(f"  Graph breaks: {explanation.graph_break_count}")
print(f"  Compiled regions: {explanation.graph_count}")
if explanation.break_reasons:
    print("  Break reasons (first 5):")
    for i, reason in enumerate(explanation.break_reasons[:5]):
        print(f"    {i}: {reason}")

dist.destroy_process_group()
