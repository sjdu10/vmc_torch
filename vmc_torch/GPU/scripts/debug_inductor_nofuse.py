"""Test if disabling inductor fusion fixes the accuracy bug.

Run:
    python debug_inductor_nofuse.py
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
B = 2

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

fxs = torch.stack([
    random_initial_config(N_f, N_sites, seed=s).to(device)
    for s in range(B)
])
example_x = random_initial_config(
    N_f, N_sites, seed=0,
).to(device)

print(f"System: {Lx}x{Ly}, D={D}, chi={chi}, B={B}")

# Eager baseline
with torch.inference_mode():
    eager_signs, eager_log_abs = model.forward_log(fxs)
print(f"\nEager:   signs={eager_signs}")
print(f"         log_abs={eager_log_abs}")

# Export + vmap
model.export_only(example_x, use_log_amp=True)
params_list = list(model.params)

# Default inductor (fused)
print("\n--- Inductor (default fusion) ---")
compiled_fused = torch.compile(
    model._vmapped_exported,
    backend='inductor',
)
with torch.inference_mode():
    t0 = time.time()
    fused_signs, fused_log_abs = compiled_fused(
        fxs, *params_list,
    )
    torch.cuda.synchronize()
    t_fused = time.time() - t0
fused_diff = (fused_log_abs - eager_log_abs).abs()
print(f"signs=   {fused_signs}")
print(f"log_abs= {fused_log_abs}")
print(f"max diff={fused_diff.max():.2e}, time={t_fused:.2f}s")

# Inductor with no fusion
print("\n--- Inductor (max_fusion_size=1) ---")
compiled_nofuse = torch.compile(
    model._vmapped_exported,
    backend='inductor',
    options={"max_fusion_size": 1},
)
with torch.inference_mode():
    t0 = time.time()
    nf_signs, nf_log_abs = compiled_nofuse(
        fxs, *params_list,
    )
    torch.cuda.synchronize()
    t_nofuse = time.time() - t0
nf_diff = (nf_log_abs - eager_log_abs).abs()
print(f"signs=   {nf_signs}")
print(f"log_abs= {nf_log_abs}")
print(f"max diff={nf_diff.max():.2e}, time={t_nofuse:.2f}s")

# Summary
print(f"\n{'='*50}")
fused_ok = fused_diff.max().item() < 1e-3
nofuse_ok = nf_diff.max().item() < 1e-3
print(f"Default fusion: {'PASS' if fused_ok else 'FAIL'} "
      f"(max diff={fused_diff.max():.2e})")
print(f"No fusion:      {'PASS' if nofuse_ok else 'FAIL'} "
      f"(max diff={nf_diff.max():.2e})")
if not fused_ok and nofuse_ok:
    print("=> FUSION IS THE CULPRIT.")
elif fused_ok and nofuse_ok:
    print("=> Both pass.")
elif not fused_ok and not nofuse_ok:
    print("=> Both fail — not a fusion issue.")
print(f"{'='*50}")
