"""Test gradient scale: chi=-1 (exact, no SVD) vs chi=10 (boundary SVD).

If chi=-1 gives normal grads but chi=10 gives huge grads,
the SVD/eigh backward is the culprit.

Run: torchrun --nproc_per_node=1 scripts/debug_grad_chi.py
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
    compute_grads_gpu,
    random_initial_config,
)

setup_linalg_hooks(jitter=1e-12)
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

Lx, Ly = 4, 2
N_sites = Lx * Ly
N_f = N_sites - 2
D = 10
B = 256

fxs = torch.stack([
    random_initial_config(N_f, N_sites, seed=42 + i)
    for i in range(B)
]).to(device)

DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
    'vmc_torch/experiment/vmap/GPU/data'
)
fpeps_base = (
    f"{DATA_ROOT}/{Lx}x{Ly}/t=1.0_U=8.0"
    f"/N={N_f}/Z2/D={D}/"
)


def diagnose(label, model):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    Np = sum(p.numel() for p in model.parameters())
    pv = torch.nn.utils.parameters_to_vector(
        model.parameters()
    )
    print(
        f"  params: Np={Np}, "
        f"rms={torch.norm(pv).item()/Np**0.5:.4e}, "
        f"max={pv.abs().max().item():.4e}"
    )

    with torch.inference_mode():
        amps = model(fxs)
    abs_a = amps.abs()
    print(
        f"  amps: min={abs_a.min().item():.4e}, "
        f"median={abs_a.median().item():.4e}, "
        f"mean={abs_a.mean().item():.4e}, "
        f"max={abs_a.max().item():.4e}"
    )

    with torch.enable_grad():
        grads, amps2 = compute_grads_gpu(
            fxs, model,
            vectorize=True, batch_size=64, vmap_grad=True,
        )
    g_rms = torch.norm(grads).item() / grads.numel() ** 0.5
    g_max = grads.abs().max().item()
    print(f"  raw grads: rms={g_rms:.4e}, max={g_max:.4e}")

    O = grads / amps2.unsqueeze(1)
    o_rms = torch.norm(O).item() / O.numel() ** 0.5
    o_max = O.abs().max().item()
    print(f"  O_loc: rms={o_rms:.4e}, max={o_max:.4e}")

    # Per-parameter gradient breakdown
    print("  Per-param grad stats (top 5 by max):")
    offset = 0
    param_stats = []
    for i, p in enumerate(model.params):
        sz = p.numel()
        g_slice = grads[:, offset:offset+sz]
        param_stats.append((
            i, p.shape, sz,
            g_slice.abs().max().item(),
            (torch.norm(g_slice).item()
             / g_slice.numel() ** 0.5),
        ))
        offset += sz
    param_stats.sort(key=lambda x: -x[3])
    for i, shape, sz, mx, rms in param_stats[:5]:
        print(
            f"    params[{i}] {list(shape)} "
            f"(numel={sz}): "
            f"grad_max={mx:.4e}, grad_rms={rms:.4e}"
        )


# chi=-1: exact contraction, no SVD in boundary
peps1 = load_or_generate_peps(
    Lx, Ly, 1.0, 8.0, N_f, D,
    seed=42, dtype=dtype, file_path=fpeps_base,
    scale_factor=4,
)
model_exact = fPEPS_Model_GPU(
    tn=peps1, max_bond=-1, dtype=dtype,
    contract_boundary_opts={},
)
model_exact.to(device)
diagnose("SU scale=4, chi=-1 (exact, no SVD)", model_exact)
del model_exact

# chi=10: boundary contraction with SVD
peps2 = load_or_generate_peps(
    Lx, Ly, 1.0, 8.0, N_f, D,
    seed=42, dtype=dtype, file_path=fpeps_base,
    scale_factor=4,
)
model_chi10 = fPEPS_Model_GPU(
    tn=peps2, max_bond=10, dtype=dtype,
    contract_boundary_opts={
        'mode': 'mps',
        'equalize_norms': 1.0,
        'canonize': True,
    },
)
model_chi10.to(device)
diagnose("SU scale=4, chi=10 (boundary SVD)", model_chi10)
del model_chi10

# chi=10, no equalize_norms
peps3 = load_or_generate_peps(
    Lx, Ly, 1.0, 8.0, N_f, D,
    seed=42, dtype=dtype, file_path=fpeps_base,
    scale_factor=4,
)
model_noeq = fPEPS_Model_GPU(
    tn=peps3, max_bond=10, dtype=dtype,
    contract_boundary_opts={
        'mode': 'mps',
        'equalize_norms': False,
        'canonize': True,
    },
)
model_noeq.to(device)
diagnose("SU scale=4, chi=10, no equalize_norms", model_noeq)
del model_noeq

dist.destroy_process_group()
