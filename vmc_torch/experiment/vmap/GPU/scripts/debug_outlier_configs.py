"""Scan configs to identify which ones are outliers and print them.

Run: torchrun --nproc_per_node=1 scripts/debug_outlier_configs.py
"""
import os
import torch
import torch.distributed as dist

from vmc_torch.experiment.vmap.GPU.vmc_setup import (
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.experiment.vmap.GPU.models import fPEPS_Model_GPU
from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    random_initial_config,
)

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

setup_linalg_hooks(jitter=1e-12)

Lx, Ly = 4, 2
N_sites = Lx * Ly
N_f = N_sites - 2
D, chi = 10, 10
DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
    'vmc_torch/experiment/vmap/GPU/data'
)
fpeps_base = (
    f"{DATA_ROOT}/{Lx}x{Ly}/t=1.0_U=8.0"
    f"/N={N_f}/Z2/D={D}/"
)

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

label = {0: '·', 1: '↓', 2: '↑', 3: '↑↓'}

header = (
    f"{'seed':>4} | {'config':>16} | "
    f"{'amp':>12} | {'grad_norm':>12} | "
    f"{'|O_loc|':>12}"
)
print(header)
print('-' * len(header))

results = []
for seed in range(42, 92):
    x = random_initial_config(
        N_f, N_sites, seed=seed
    ).to(device)
    model.zero_grad()
    amp = model._amplitude_for_export(
        x, *list(model.params)
    )
    amp.backward()
    g_norm = torch.cat([
        p.grad.flatten() for p in model.params
    ]).norm().item()
    o_norm = g_norm / max(abs(amp.item()), 1e-30)
    cfg = ''.join(label[v.item()] for v in x.cpu())
    tag = ' ***' if o_norm > 1000 else ''
    print(
        f"{seed:4d} | {cfg:>16} | "
        f"{amp.item():12.4e} | {g_norm:12.4e} | "
        f"{o_norm:12.4e}{tag}"
    )
    results.append((seed, cfg, amp.item(), g_norm, o_norm))

# Summary
o_norms = [r[4] for r in results]
import statistics
print(f"\n--- Summary over {len(results)} configs ---")
print(f"  median |O_loc|: {statistics.median(o_norms):.4e}")
print(f"  max |O_loc|:    {max(o_norms):.4e}")
n_outlier = sum(1 for o in o_norms if o > 1000)
print(f"  #(|O_loc|>1000): {n_outlier}/{len(results)}")

# Print outliers with 2D grid
print("\n--- Outlier configs (|O_loc|>1000) ---")
for seed, cfg, amp, g_norm, o_norm in results:
    if o_norm <= 1000:
        continue
    x = random_initial_config(
        N_f, N_sites, seed=seed
    )
    print(f"\nseed={seed}, amp={amp:.4e}, "
          f"|O_loc|={o_norm:.4e}")
    for row in range(Ly):
        cells = []
        for col in range(Lx):
            site = row * Lx + col
            cells.append(f'{label[x[site].item()]:>2}')
        print(f"  row {row}: {'  '.join(cells)}")

dist.destroy_process_group()
