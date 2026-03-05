"""Compare: sequential per-sample grads vs vmap grads.

If sequential grads are normal but vmap grads are huge,
the issue is vmap interaction with the backward pass.

Run: torchrun --nproc_per_node=1 scripts/debug_grad_vmap_vs_seq.py
"""
import os
import time
import torch
import torch.distributed as dist

from vmc_torch.GPU.vmc_setup import (
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.GPU.models import fPEPS_Model_GPU
from vmc_torch.GPU.vmc_utils import random_initial_config

dtype = torch.float64
torch.set_default_dtype(dtype)

if "RANK" not in os.environ:
    os.environ.update({
        "RANK": "0", "WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost", "MASTER_PORT": "12355",
        "LOCAL_RANK": "0",
    })
dist.init_process_group(backend="nccl", init_method="env://")
device = torch.device("cuda:0")
torch.cuda.set_device(0)
torch.set_default_device(device)
torch.manual_seed(42)

setup_linalg_hooks(jitter=1e-12)

Lx, Ly = 4, 2
N_sites = Lx * Ly
N_f = N_sites - 2
D = 10
chi = 10
B = 16  # small batch for quick comparison

DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
    'vmc_torch/GPU/data'
)
fpeps_base = (
    f"{DATA_ROOT}/{Lx}x{Ly}/t=1.0_U=8.0"
    f"/N={N_f}/Z2/D={D}/"
)

fxs = torch.stack([
    random_initial_config(N_f, N_sites, seed=42 + i)
    for i in range(B)
]).to(device)

peps = load_or_generate_peps(
    Lx, Ly, 1.0, 8.0, N_f, D,
    seed=42, dtype=dtype, file_path=fpeps_base,
    scale_factor=4,
)
model = fPEPS_Model_GPU(
    tn=peps, max_bond=chi, dtype=dtype,
    contract_boundary_opts={
        'mode': 'mps', 'equalize_norms': 1.0,
        'canonize': True,
    },
)
model.to(device)

Np = sum(p.numel() for p in model.parameters())

# ============================================================
# Method 1: Sequential per-sample gradients
# ============================================================
print("=" * 60)
print("Sequential per-sample gradients")
print("=" * 60)

seq_grads = []
seq_amps = []

for i in range(B):
    model.zero_grad()
    x_single = fxs[i]
    amp = model._amplitude_for_export(
        x_single, *list(model.params)
    )
    amp.backward()
    g = torch.cat([
        p.grad.flatten() for p in model.params
    ])
    seq_grads.append(g.detach().clone())
    seq_amps.append(amp.item())

seq_grads_t = torch.stack(seq_grads)  # (B, Np)
seq_amps_t = torch.tensor(seq_amps)

print(f"  amps: min={abs(min(seq_amps)):.4e}, "
      f"max={abs(max(seq_amps)):.4e}")
print(f"  grads: rms="
      f"{seq_grads_t.norm() / seq_grads_t.numel()**0.5:.4e}"
      f", max={seq_grads_t.abs().max().item():.4e}")

seq_O = seq_grads_t / seq_amps_t.unsqueeze(1)
print(f"  O_loc: rms="
      f"{seq_O.norm() / seq_O.numel()**0.5:.4e}"
      f", max={seq_O.abs().max().item():.4e}")

# ============================================================
# Method 2: vmap(grad) — same as compute_grads_gpu
# ============================================================
print(f"\n{'=' * 60}")
print("vmap(grad) gradients")
print("=" * 60)

from vmc_torch.GPU.vmc_utils import (
    compute_grads_gpu,
)

model.zero_grad()
with torch.enable_grad():
    vmap_grads, vmap_amps = compute_grads_gpu(
        fxs, model,
        vectorize=True, batch_size=B, vmap_grad=True,
    )

print(f"  amps: min={vmap_amps.abs().min().item():.4e}, "
      f"max={vmap_amps.abs().max().item():.4e}")
print(f"  grads: rms="
      f"{vmap_grads.norm() / vmap_grads.numel()**0.5:.4e}"
      f", max={vmap_grads.abs().max().item():.4e}")

vmap_O = vmap_grads / vmap_amps.unsqueeze(1)
print(f"  O_loc: rms="
      f"{vmap_O.norm() / vmap_O.numel()**0.5:.4e}"
      f", max={vmap_O.abs().max().item():.4e}")

# ============================================================
# Per-sample comparison
# ============================================================
print(f"\n{'=' * 60}")
print("Per-sample comparison (seq vs vmap)")
print("=" * 60)

for i in range(min(B, 8)):
    s_norm = seq_grads[i].norm().item()
    v_norm = vmap_grads[i].norm().item()
    ratio = v_norm / max(s_norm, 1e-30)
    diff = (vmap_grads[i] - seq_grads[i]).norm().item()
    print(
        f"  sample {i}: seq_norm={s_norm:.4e}, "
        f"vmap_norm={v_norm:.4e}, "
        f"ratio={ratio:.4e}, "
        f"diff={diff:.4e}"
    )

dist.destroy_process_group()
