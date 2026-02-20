"""
profile_vmc.py — CUDA kernel profiler for the GPU-VMC pipeline.

Mirrors the exact setup of vmc_run.py (4x2, N_f=6, D=4, chi=-1,
export_and_compile) and profiles one complete VMC step with
torch.profiler.

Traces can be viewed two ways:
  A) TensorBoard (recommended):
       pip install tensorboard torch-tb-profiler
       tensorboard --logdir GPU/profiles/tb
       → open http://localhost:6006  (PyTorch Profiler tab)

  B) Perfetto (no install needed):
       → open https://ui.perfetto.dev
       → drag-and-drop GPU/profiles/chrome_trace.json

Run:
    python GPU/scripts/profile_vmc.py
"""
import os, sys, time, pickle
import numpy as np
import torch
import torch.distributed as dist
import torch.profiler as tprof

import autoray as ar
import quimb.tensor as qtn

from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    random_initial_config, sample_next, evaluate_energy, compute_grads_gpu,
)
from vmc_torch.experiment.vmap.GPU.models import fPEPS_Model_GPU
from vmc_torch.hamiltonian_torch import (
    spinful_Fermi_Hubbard_square_lattice_torch,
)
from vmc_torch.experiment.vmap.vmap_torch_utils import (
    robust_svd_err_catcher_wrapper,
)

# ============================================================
# 0.  Distributed init (single-GPU standalone)
# ============================================================
if "RANK" not in os.environ:
    os.environ.update({"RANK": "0", "WORLD_SIZE": "1",
                       "MASTER_ADDR": "localhost", "MASTER_PORT": "12356",
                       "LOCAL_RANK": "0"})
dist.init_process_group(backend="nccl", init_method="env://")
RANK       = dist.get_rank()
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(LOCAL_RANK)
device = torch.device(f"cuda:{LOCAL_RANK}")
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

ar.register_function(
    'torch', 'linalg.svd',
    lambda x: robust_svd_err_catcher_wrapper(x, jitter=1e-16, driver=None),
)

# ============================================================
# 1.  Match vmc_run.py physics exactly
# ============================================================
Lx, Ly   = 4, 2
nsites    = Lx * Ly
N_f       = nsites - 2       # doped: 6 fermions on 8 sites
D, chi    = 4, -1
t_hop, U  = 1.0, 8.0
n_fps     = (N_f // 2, N_f // 2)
B         = 2048             # walker batch (reduced for profiling speed)
grad_bs   = B // 2

pwd = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
    'vmc_torch/experiment/vmap/data'
)
appendix = '_U1SU'
params_path = (
    f"{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/"
    f"peps_su_params{appendix}.pkl"
)
skeleton_path = (
    f"{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/"
    f"peps_skeleton{appendix}.pkl"
)

with open(params_path, 'rb') as f:   params_pkl = pickle.load(f)
with open(skeleton_path, 'rb') as f: skeleton   = pickle.load(f)
peps = qtn.unpack(params_pkl, skeleton)
for ts in peps.tensors:
    ts.modify(data=ts.data.to_flat() * 4)
for site in peps.sites:
    peps[site].data._label = site
    peps[site].data.indices[-1]._linearmap = (
        (0, 0), (1, 0), (1, 1), (0, 1)
    )

# ============================================================
# 2.  Model, Hamiltonian
# ============================================================
model = fPEPS_Model_GPU(
    tn=peps, max_bond=chi, dtype=torch.float64,
    contract_boundary_opts={'mode': 'mps', 'equalize_norms': 1.0,
                            'canonize': True},
)
model.to(device)

print("[profile] export_and_compile …", flush=True)
example_x = random_initial_config(N_f, nsites, seed=0).to(device)
model.export_and_compile(example_x, mode='default')
print("[profile] done", flush=True)

H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, t_hop, U, N_f,
    pbc=False, n_fermions_per_spin=n_fps,
    no_u1_symmetry=False, gpu=True,
)
graph = H.graph
H.precompute_hops_gpu(device)

# ============================================================
# 3.  Initial walkers + warmup (fill compile caches)
# ============================================================
fxs = torch.stack([
    random_initial_config(N_f, nsites, seed=42 + i) for i in range(B)
]).to(device)

print("[profile] warmup …", flush=True)
with torch.inference_mode():
    fxs, amps = sample_next(fxs, model, graph)
    _, evals  = evaluate_energy(fxs, model, H, amps)
with torch.enable_grad():
    grads, amps2 = compute_grads_gpu(
        fxs, model, vectorize=True, batch_size=grad_bs, vmap_grad=True,
    )
del grads, amps2, evals
torch.cuda.synchronize()
print("[profile] warmup done", flush=True)

# ============================================================
# 4.  Output directories
# ============================================================
script_dir   = os.path.dirname(os.path.abspath(__file__))
profile_dir  = os.path.join(script_dir, '..', 'profiles')
tb_dir = os.path.join(profile_dir, 'tb')
os.makedirs(tb_dir, exist_ok=True)
chrome_path = os.path.join(profile_dir, 'chrome_trace.json')

# ============================================================
# 5.  Profile schedule:
#     wait=1  (skip first step — still JIT-ing)
#     warmup=1 (record but discard)
#     active=3 (record 3 full steps)
#     repeat=1 (one cycle)
# ============================================================
schedule = tprof.schedule(wait=1, warmup=1, active=3, repeat=1)

# TensorBoard handler writes one .pt.trace.json per step
tb_handler = tprof.tensorboard_trace_handler(tb_dir)

activities = [
    tprof.ProfilerActivity.CPU,
    tprof.ProfilerActivity.CUDA,
]

print("[profile] profiling 5 steps (wait=1, warmup=1, active=3) …",
      flush=True)

with tprof.profile(
    activities=activities,
    schedule=schedule,
    on_trace_ready=tb_handler,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
) as prof:

    for step in range(5):   # wait + warmup + active = 1+1+3 = 5
        # ---- annotate each sub-step so they appear as named regions ----
        with tprof.record_function("sample_next"):
            with torch.inference_mode():
                fxs, amps = sample_next(fxs, model, graph)

        with tprof.record_function("evaluate_energy"):
            with torch.inference_mode():
                _, local_E = evaluate_energy(fxs, model, H, amps)

        with tprof.record_function("compute_grads"):
            with torch.enable_grad():
                local_grads, local_amps = compute_grads_gpu(
                    fxs, model, vectorize=True,
                    batch_size=grad_bs, vmap_grad=True,
                )
            local_grads /= local_amps.unsqueeze(1)   # → O_loc

        torch.cuda.synchronize()
        prof.step()

        del local_grads, local_amps, local_E

# ============================================================
# 6.  Print top kernel table to terminal
# ============================================================
print("\n" + "="*70)
print("Top 20 CUDA kernels by self-CUDA time (active steps only):")
print("="*70)
print(
    prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=20,
    )
)

print("\n" + "="*70)
print("Top 20 ops by CPU time:")
print("="*70)
print(
    prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=20,
    )
)

# ============================================================
# 8.  Instructions
# ============================================================
print(f"""
{'='*70}
Trace files written:
  TensorBoard traces : {os.path.abspath(tb_dir)}/
  Chrome JSON        : {os.path.abspath(chrome_path)}

To view in browser:

  Option A — TensorBoard (full kernel breakdown, memory timeline):
    pip install tensorboard torch-tb-profiler   # if not already installed
    tensorboard --logdir {os.path.abspath(tb_dir)}
    → open http://localhost:6006  (click "PyTorch Profiler" tab)

  Option B — Perfetto (no install needed):
    → open https://ui.perfetto.dev
    → drag-and-drop  {os.path.abspath(chrome_path)}
{'='*70}
""")

dist.destroy_process_group()
