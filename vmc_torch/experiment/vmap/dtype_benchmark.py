"""
Float32 vs Float64 benchmark for pure fPEPS reuse model.

Usage:
    python dtype_benchmark.py --threads 1 --dtype float64
    python dtype_benchmark.py --threads 1 --dtype float32
    python dtype_benchmark.py --threads 4 --dtype float32
    python dtype_benchmark.py --threads 10 --dtype float32

Results saved to:
    debug/D={D}_dtype_benchmark_{num_threads}_cores_{dtype}.json
"""
import argparse
import sys

# ── Parse args FIRST so we can set env vars before any numpy/torch import ──
parser = argparse.ArgumentParser()
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--dtype', type=str, default='float64',
                    choices=['float32', 'float64'])
args = parser.parse_args()

num_threads = args.threads
dtype_str   = args.dtype

import os
os.environ["MKL_NUM_THREADS"]        = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"]    = str(num_threads)
os.environ["OMP_NUM_THREADS"]        = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"]   = str(num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)

# ── Now safe to import everything ──────────────────────────────────────────
import time
import json
import warnings
import numpy as np
import torch
import autoray as ar
import symmray as sr

from vmc_torch.experiment.vmap.vmap_utils import (
    random_initial_config,
    sample_next_reuse,
    evaluate_energy_reuse,
)
from vmc_torch.experiment.vmap.vmap_models import fPEPS_Model_reuse
from vmc_torch.hamiltonian_torch import (
    spinful_Fermi_Hubbard_square_lattice_torch,
)
from vmc_torch.experiment.vmap.vmap_torch_utils import (
    robust_svd_err_catcher_wrapper,
)

warnings.filterwarnings("ignore")
torch.set_num_threads(num_threads)

dtype = torch.float32 if dtype_str == 'float32' else torch.float64
torch.set_default_dtype(dtype)
torch.set_default_device("cpu")
torch.random.manual_seed(42)

ar.register_function(
    'torch', 'linalg.svd',
    lambda x: robust_svd_err_catcher_wrapper(x, jitter=1e-16, driver=None),
)

# ── Physics & model setup (mirrors sampling_time.ipynb) ───────────────────
Lx, Ly = 8, 8
N_f    = Lx * Ly
D, chi = 10, 40
t, U   = 1.0, 8.0

print(f"\n[dtype_benchmark] threads={num_threads}, dtype={dtype_str}, "
      f"D={D}, chi={chi}, {Lx}x{Ly}")

peps = sr.networks.PEPS_fermionic_rand(
    "Z2", Lx, Ly, D,
    phys_dim=[(0,0),(1,1),(1,0),(0,1)],
    subsizes="equal",
    flat=True,
    seed=42,
    dtype=dtype_str,
)

fpeps_model = fPEPS_Model_reuse(
    tn=peps,
    dtype=dtype,
    max_bond=chi,
    contract_boundary_opts={
        'mode': 'mps',
        'canonize': True,
    },
)

H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, t, U, N_f,
    pbc=False,
    n_fermions_per_spin=(N_f // 2, N_f // 2),
    no_u1_symmetry=False,
)

# ── Benchmark loop ────────────────────────────────────────────────────────
batch_sizes    = [1, 2, 4, 6, 8, 10]
sampling_times = []

for batchsize in batch_sizes:
    fxs0 = torch.stack([
        random_initial_config(N_f, peps.nsites) for _ in range(batchsize)
    ]).to(torch.long)

    fpeps_model.cache_bMPS_skeleton(fxs0[0])

    with torch.no_grad():
        t1 = time.time()
        fxs2, amps2 = sample_next_reuse(
            fxs0.clone(), fpeps_model, H.graph, show_pbar=True,
        )
        t2 = time.time()
        _, loc_Es = evaluate_energy_reuse(
            fxs2, fpeps_model, H, amps2, show_pbar=True,
        )
        t3 = time.time()

    samp_t = t2 - t1
    ene_t  = t3 - t2
    sampling_times.append((samp_t, ene_t))
    print(f"  B={batchsize:2d}  samp={samp_t:.2f}s  ene={ene_t:.2f}s  "
          f"total={samp_t+ene_t:.2f}s")

# ── Save ──────────────────────────────────────────────────────────────────
out_dir  = os.path.join(os.path.dirname(__file__), 'debug')
out_path = os.path.join(
    out_dir,
    f"D={D}_dtype_benchmark_{num_threads}_cores_{dtype_str}.json",
)
results = {
    'num_threads': num_threads,
    'dtype':       dtype_str,
    'batch_sizes': batch_sizes,
    'sampling_times': sampling_times,   # list of (samp_time, ene_time)
}
with open(out_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\nSaved: {out_path}")
