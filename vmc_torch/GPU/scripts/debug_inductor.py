"""Bisect inductor accuracy bug on 6x4 D=10 chi=10.

Uses TORCHDYNAMO_REPRO_LEVEL=4 to automatically find the
minimal subgraph where inductor diverges from eager.
Generates a repro.py file in the current directory.

Run:
    TORCHDYNAMO_REPRO_LEVEL=4 python debug_inductor.py
"""
import os
os.environ['TORCHDYNAMO_REPRO_LEVEL'] = '4'
os.environ['TORCHDYNAMO_REPRO_AFTER'] = 'aot'
import torch
import time

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
B = 2  # small batch for faster bisection

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

# Export + vmap (no compile yet)
print("Exporting...")
model.export_only(example_x, use_log_amp=True)

# Compile with inductor — TORCHDYNAMO_REPRO_LEVEL=4
# will intercept this and bisect for accuracy issues
print("Compiling (REPRO_LEVEL=4 will bisect)...")
t0 = time.time()
params_list = list(model.params)
compiled_fn = torch.compile(
    model._vmapped_exported,
    backend='inductor',
)
print(f"Compilation took {time.time() - t0:.2f} seconds.")

with torch.inference_mode():
    signs_eager, log_abs_eager = model.forward_log(fxs)
    print(f'Eager - signs {signs_eager}, log_abs {log_abs_eager}')

with torch.inference_mode():
    signs, log_abs = compiled_fn(fxs, *params_list)
    

print(f'Inductor - signs {signs}, log_abs {log_abs}')
print("Done. Check for repro.py in current directory.")
