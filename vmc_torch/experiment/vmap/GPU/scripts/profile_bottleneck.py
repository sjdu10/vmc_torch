"""
Micro-benchmark to identify GPU VMC bottlenecks.
Run with: torchrun --nproc_per_node=1 GPU_profile_bottleneck.py
"""
import os
import time
import pickle
import torch
import torch.distributed as dist
import numpy as np
import autoray as ar
import quimb.tensor as qtn

from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    sample_next, evaluate_energy, compute_grads_gpu,
    random_initial_config, propose_exchange_or_hopping_vec,
)
from vmc_torch.experiment.vmap.GPU.models import fPEPS_Model_GPU
from vmc_torch.hamiltonian_torch import (
    spinful_Fermi_Hubbard_square_lattice_torch
)
from vmc_torch.experiment.vmap.GPU.torch_utils import (
    robust_svd_err_catcher_wrapper
)

ar.register_function(
    'torch', 'linalg.svd',
    lambda x: robust_svd_err_catcher_wrapper(
        x, jitter=1e-16, driver=None
    ),
)

# --- Setup ---
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

# --- Load model ---
Lx, Ly = 4, 2
nsites = Lx * Ly
N_f = nsites - 2
D, chi = 4, 4
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

model = fPEPS_Model_GPU(
    tn=peps, max_bond=chi, dtype=torch.float64,
    contract_boundary_opts={
        'mode': 'mps', 'equalize_norms': 1.0, 'canonize': True,
    }
)
model.to(device)
n_params = sum(p.numel() for p in model.parameters())

H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, 1.0, 8.0, N_f,
    pbc=False,
    n_fermions_per_spin=(N_f // 2, N_f // 2),
    no_u1_symmetry=False,
    gpu=True,
)
graph = H.graph

B = 500
fxs = torch.stack([
    random_initial_config(N_f, nsites, seed=42 + i)
    for i in range(B)
]).to(device)

print(f"Lattice: {Lx}x{Ly}, N_f={N_f}, D={D}, chi={chi}")
print(f"n_params={n_params}, B={B}")

# --- Warmup ---
with torch.inference_mode():
    fxs, amps = sample_next(fxs, model, graph)

# ============================================================
# Benchmark 1: model forward pass alone
# ============================================================
print("\n=== Benchmark: model forward (B configs) ===")
torch.cuda.synchronize()
for trial in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        out = model(fxs)
    torch.cuda.synchronize()
    print(f"  model(fxs) [{B} samples]: {time.time()-t0:.4f}s")

# Single sample
print("\n=== Benchmark: model forward (1 config) ===")
for trial in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        out = model(fxs[:1])
    torch.cuda.synchronize()
    print(f"  model(fxs[:1]): {time.time()-t0:.4f}s")

# Small batch
print("\n=== Benchmark: model forward (50 configs) ===")
for trial in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        out = model(fxs[:50])
    torch.cuda.synchronize()
    print(f"  model(fxs[:50]): {time.time()-t0:.4f}s")

# ============================================================
# Benchmark 2: sample_next breakdown
# ============================================================
print("\n=== Benchmark: sample_next breakdown ===")

all_edges = []
for edges in graph.row_edges.values():
    all_edges.extend(edges)
for edges in graph.col_edges.values():
    all_edges.extend(edges)
print(f"  Total edges: {len(all_edges)}")

# Time each phase inside one sweep
with torch.inference_mode():
    current_amps = model(fxs)

    t_propose_total = 0
    t_model_total = 0
    t_accept_total = 0
    n_model_calls = 0
    n_changed_total = 0

    for edge in all_edges:
        i, j = edge

        torch.cuda.synchronize()
        tp0 = time.time()
        proposed_fxs, new_flags = propose_exchange_or_hopping_vec(
            i, j, fxs, 0.25
        )
        torch.cuda.synchronize()
        t_propose_total += time.time() - tp0

        if not new_flags.any():
            continue

        n_changed = new_flags.sum().item()
        n_changed_total += n_changed

        torch.cuda.synchronize()
        tm0 = time.time()
        new_proposed_fxs = proposed_fxs[new_flags]
        new_proposed_amps = model(new_proposed_fxs)
        torch.cuda.synchronize()
        t_model_total += time.time() - tm0
        n_model_calls += 1

        torch.cuda.synchronize()
        ta0 = time.time()
        proposed_amps = current_amps.clone()
        proposed_amps[new_flags] = new_proposed_amps
        ratio = (proposed_amps.abs()**2) / (
            current_amps.abs()**2 + 1e-18
        )
        probs = torch.rand(B, device=device)
        accept_mask = new_flags & (probs < ratio)
        if accept_mask.any():
            fxs[accept_mask] = proposed_fxs[accept_mask]
            current_amps[accept_mask] = proposed_amps[accept_mask]
        torch.cuda.synchronize()
        t_accept_total += time.time() - ta0

    print(f"  Propose:  {t_propose_total:.4f}s")
    print(f"  Model:    {t_model_total:.4f}s "
          f"({n_model_calls} calls, "
          f"avg {n_changed_total/max(n_model_calls,1):.0f} "
          f"changed/call)")
    print(f"  Accept:   {t_accept_total:.4f}s")
    print(f"  TOTAL:    "
          f"{t_propose_total+t_model_total+t_accept_total:.4f}s")

# ============================================================
# Benchmark 3: evaluate_energy breakdown
# ============================================================
print("\n=== Benchmark: evaluate_energy breakdown ===")

with torch.inference_mode():
    current_amps = model(fxs)

    # Phase 1: H.get_conn loop (CPU)
    torch.cuda.synchronize()
    t_conn0 = time.time()
    conn_eta_num = []
    conn_etas = []
    conn_eta_coeffs = []
    for fx in fxs:
        eta, coeffs = H.get_conn(fx)
        conn_eta_num.append(len(eta))
        conn_etas.append(torch.as_tensor(eta, device=device))
        conn_eta_coeffs.append(
            torch.as_tensor(coeffs, device=device)
        )
    conn_etas = torch.cat(conn_etas, dim=0)
    conn_eta_coeffs = torch.cat(conn_eta_coeffs, dim=0)
    conn_eta_num_t = torch.tensor(conn_eta_num, device=device)
    torch.cuda.synchronize()
    t_conn = time.time() - t_conn0

    total_conn = conn_etas.shape[0]
    print(f"  get_conn loop ({B} samples): {t_conn:.4f}s")
    print(f"  Total connected configs: {total_conn} "
          f"(avg {total_conn/B:.1f}/sample)")

    # Phase 2: model on connected configs
    torch.cuda.synchronize()
    t_model0 = time.time()
    conn_amps = model(conn_etas)
    torch.cuda.synchronize()
    t_model = time.time() - t_model0
    print(f"  model({total_conn} conn configs): {t_model:.4f}s")

    # Phase 3: local energy assembly
    torch.cuda.synchronize()
    t_assemble0 = time.time()
    batch_ids = torch.repeat_interleave(
        torch.arange(B, device=device), conn_eta_num_t
    )
    current_amps_expanded = current_amps[batch_ids]
    terms = conn_eta_coeffs * (conn_amps / current_amps_expanded)
    local_energies = torch.zeros(
        B, device=device, dtype=terms.dtype
    )
    local_energies.index_add_(0, batch_ids, terms)
    torch.cuda.synchronize()
    t_assemble = time.time() - t_assemble0
    print(f"  Assembly: {t_assemble:.4f}s")
    print(f"  TOTAL:    {t_conn+t_model+t_assemble:.4f}s")

# ============================================================
# Benchmark 4: compute_grads
# ============================================================
print("\n=== Benchmark: compute_grads ===")
for bs in [B, B // 2, B // 4]:
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.enable_grad():
        g, a = compute_grads_gpu(
            fxs, model, vectorize=True,
            batch_size=bs, vmap_grad=True,
        )
    torch.cuda.synchronize()
    print(f"  compute_grads(B={B}, batch={bs}): "
          f"{time.time()-t0:.4f}s")
    del g, a

# ============================================================
# Summary
# ============================================================
print("\n=== Summary ===")
print("The bottleneck is:")
print("  1. model forward pass (called many times per sweep)")
print("  2. H.get_conn Python loop (CPU, serial)")
print("torch.compile cannot help because quimb TN ops")
print("use Python dicts, dynamic shapes, and custom classes.")

dist.destroy_process_group()
