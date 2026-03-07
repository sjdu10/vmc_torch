"""
test_log_amp_debug.py — Debug log_amplitude on single samples (no vmap).

Calls model.log_amplitude(x, params) one sample at a time so that
print statements inside log_amplitude are visible.

Run:
    python scripts/test_log_amp_debug.py
"""
import os
import torch
import torch.distributed as dist

from vmc_torch.GPU.models.pureTNS import fPEPS_Model_GPU
from vmc_torch.GPU.vmc_setup import (
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.GPU.vmc_utils import random_initial_config


# ==========================================
# Setup
# ==========================================
def setup_distributed():
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        os.environ["LOCAL_RANK"] = "0"
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


device = setup_distributed()
setup_linalg_hooks(jitter=1e-12)
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

# ==========================================
# System
# ==========================================
Lx, Ly = 4, 2
N_sites = Lx * Ly
t, U = 1.0, 8.0
N_f = N_sites - 2
D = 4
chi = -1  # exact contraction

# ==========================================
# Model
# ==========================================
peps = load_or_generate_peps(
    Lx, Ly, t, U, N_f, D, seed=42, dtype=torch.float64,
)
model = fPEPS_Model_GPU(
    tn=peps, max_bond=chi, dtype=torch.float64,
    contract_boundary_opts={
        'mode': 'mps',
        'equalize_norms': 1.0,
        'canonize': True,
    },
)
model.to(device)

N_params = sum(p.numel() for p in model.parameters())
print(f"\n{'='*50}")
print(f"fPEPS log_amplitude debug: {Lx}x{Ly}, D={D}, chi={chi}")
print(f"N_params={N_params}")
print(f"{'='*50}\n")

# ==========================================
# Generate a few configs
# ==========================================
n_samples = 5
fxs = torch.stack([
    random_initial_config(N_f, N_sites, seed=100 + i)
    for i in range(n_samples)
]).to(device)

# ==========================================
# Call log_amplitude on single samples (prints visible)
# ==========================================
import quimb as qu

params_pytree = qu.utils.tree_unflatten(
    list(model.params), model.params_pytree,
)

print("--- Single-sample log_amplitude (outside vmap) ---")
for i in range(n_samples):
    x_i = fxs[i]
    sign, log_abs = model.log_amplitude(x_i, params_pytree)
    reconstructed = sign * torch.exp(log_abs)
    print(f"  [{i}] sign={sign.item():.0f}, log_abs={log_abs.item():.4f}, "
          f"amp_recon={reconstructed.item():.6e}")
    print()

# ==========================================
# Compare with batched forward_log
# ==========================================
print("--- Batched forward_log (vmap) ---")
with torch.no_grad():
    signs_batch, log_abs_batch = model.forward_log(fxs)

for i in range(n_samples):
    recon = signs_batch[i] * torch.exp(log_abs_batch[i])
    print(f"  [{i}] sign={signs_batch[i].item():.0f}, "
          f"log_abs={log_abs_batch[i].item():.4f}, "
          f"amp_recon={recon.item():.6e}")

dist.destroy_process_group()
