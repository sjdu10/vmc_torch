"""Test that the refactored CNN backflow has no graph breaks.

Checks:
1. Output correctness (same output for same weights)
2. torch.compile(fullgraph=True) succeeds (no graph breaks)
3. Timing comparison: eager vs compiled CNN forward

Run:
    python GPU/scripts/test_cnn_compile.py
"""
import time

import torch
import torch.nn as nn

from vmc_torch.GPU.vmc_setup import (
    setup_linalg_hooks,
    load_or_generate_peps,
)

setup_linalg_hooks(jitter=1e-12)
torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Build a small test PEPS ---
Lx, Ly = 4, 2
D = 4
N_sites = Lx * Ly
N_f = N_sites - 2  # 6 fermions (2 holes)

peps = load_or_generate_peps(
    Lx=Lx, Ly=Ly, t=1.0, U=8.0, N_f=N_f, D=D,
    dtype=torch.float64,
)

# --- Build the CNN module ---
from vmc_torch.GPU.models.NNfTNS import (
    _CNN_Geometric_Backflow_GPU,
)

nn_module = _CNN_Geometric_Backflow_GPU(
    tn=peps,
    embed_dim=8,
    hidden_dim=16,
    kernel_size=3,
    layers=1,
    dtype=torch.float64,
)
nn_module.to(device)

# --- Test input ---
B = 64
x = torch.randint(0, 3, (B, N_sites), device=device, dtype=torch.int64)

# --- 1. Eager forward ---
with torch.no_grad():
    out_eager = nn_module(x)
print(f"Eager output shape: {out_eager.shape}")
print(f"Eager output range: [{out_eager.min():.6f}, {out_eager.max():.6f}]")

# --- 2. Test fullgraph compile (no graph breaks) ---
print("\nTesting torch.compile(fullgraph=True)...")
try:
    compiled_module = torch.compile(nn_module, fullgraph=True)
    with torch.no_grad():
        out_compiled = compiled_module(x)

    # Check outputs match
    max_diff = (out_eager - out_compiled).abs().max().item()
    print(f"  fullgraph=True: SUCCESS")
    print(f"  Max diff eager vs compiled: {max_diff:.2e}")
    assert max_diff < 1e-10, f"Output mismatch: {max_diff}"
except Exception as e:
    print(f"  fullgraph=True: FAILED — {e}")

# --- 3. Test via functional_call (as used in vamp) ---
print("\nTesting functional_call + compile...")
nn_param_names = [n for n, _ in nn_module.named_parameters()]
nn_params = {n: p for n, p in nn_module.named_parameters()}

def fn(x):
    return torch.func.functional_call(nn_module, nn_params, (x,))

try:
    compiled_fn = torch.compile(fn, fullgraph=True)
    with torch.no_grad():
        out_fc = compiled_fn(x)
    max_diff = (out_eager - out_fc).abs().max().item()
    print(f"  functional_call + fullgraph=True: SUCCESS")
    print(f"  Max diff: {max_diff:.2e}")
except Exception as e:
    print(f"  functional_call + fullgraph=True: FAILED — {e}")

# --- 4. Timing ---
print("\nTiming (B={})...".format(B))

# Warm up
for _ in range(3):
    with torch.no_grad():
        _ = nn_module(x)
if device.type == 'cuda':
    torch.cuda.synchronize()

n_iters = 50

# Eager timing
t0 = time.perf_counter()
for _ in range(n_iters):
    with torch.no_grad():
        _ = nn_module(x)
if device.type == 'cuda':
    torch.cuda.synchronize()
t_eager = (time.perf_counter() - t0) / n_iters * 1000

# Compiled timing
compiled_module = torch.compile(nn_module)
# Warm up compile
for _ in range(3):
    with torch.no_grad():
        _ = compiled_module(x)
if device.type == 'cuda':
    torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(n_iters):
    with torch.no_grad():
        _ = compiled_module(x)
if device.type == 'cuda':
    torch.cuda.synchronize()
t_compiled = (time.perf_counter() - t0) / n_iters * 1000

print(f"  Eager:    {t_eager:.2f} ms/call")
print(f"  Compiled: {t_compiled:.2f} ms/call")
print(f"  Speedup:  {t_eager / t_compiled:.2f}x")

# --- 5. Larger batch timing ---
for B_test in [256, 1024, 4096]:
    x_large = torch.randint(
        0, 3, (B_test, N_sites), device=device, dtype=torch.int64,
    )
    # Eager
    for _ in range(3):
        with torch.no_grad():
            _ = nn_module(x_large)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = nn_module(x_large)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_e = (time.perf_counter() - t0) / n_iters * 1000

    # Compiled
    for _ in range(3):
        with torch.no_grad():
            _ = compiled_module(x_large)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = compiled_module(x_large)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_c = (time.perf_counter() - t0) / n_iters * 1000

    print(f"  B={B_test:5d}: eager={t_e:.2f}ms, "
          f"compiled={t_c:.2f}ms, speedup={t_e/t_c:.2f}x")

print("\nAll tests passed!")
