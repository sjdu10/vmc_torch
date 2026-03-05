"""Compare gradient scales: SU-loaded vs random PEPS.

Run: torchrun --nproc_per_node=1 scripts/debug_grad_scale.py
"""
import os
import pickle
import torch
import torch.distributed as dist

from vmc_torch.GPU.vmc_setup import (
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.GPU.models import fPEPS_Model_GPU
from vmc_torch.GPU.vmc_utils import (
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
chi = 10
B = 512  # smaller batch for quick test

fxs = torch.stack([
    random_initial_config(N_f, N_sites, seed=42 + i)
    for i in range(B)
]).to(device)

contract_opts = {
    'mode': 'mps', 'equalize_norms': 1.0, 'canonize': True,
}


def diagnose(label, model):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    Np = sum(p.numel() for p in model.parameters())
    pv = torch.nn.utils.parameters_to_vector(
        model.parameters()
    )
    print(
        f"  params: Np={Np}, rms={torch.norm(pv).item()/Np**0.5:.4e}"
        f", max={pv.abs().max().item():.4e}"
    )

    # Amplitudes
    with torch.inference_mode():
        amps = model(fxs)
    abs_a = amps.abs()
    print(
        f"  amps: min={abs_a.min().item():.4e}"
        f", median={abs_a.median().item():.4e}"
        f", mean={abs_a.mean().item():.4e}"
        f", max={abs_a.max().item():.4e}"
    )

    # Gradients
    with torch.enable_grad():
        grads, amps2 = compute_grads_gpu(
            fxs, model,
            vectorize=True, batch_size=128, vmap_grad=True,
        )
    g_rms = torch.norm(grads).item() / grads.numel() ** 0.5
    g_max = grads.abs().max().item()
    print(f"  raw grads: rms={g_rms:.4e}, max={g_max:.4e}")

    # O_loc
    O = grads / amps2.unsqueeze(1)
    o_rms = torch.norm(O).item() / O.numel() ** 0.5
    o_max = O.abs().max().item()
    print(f"  O_loc: rms={o_rms:.4e}, max={o_max:.4e}")

    # grad/param ratio
    print(
        f"  |grad|_rms / |param|_rms = "
        f"{g_rms / (torch.norm(pv).item()/Np**0.5):.4e}"
    )


# ============================================================
# Case 1: SU-loaded PEPS (scale_factor=4)
# ============================================================
DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
    'vmc_torch/GPU/data'
)
fpeps_base = f"{DATA_ROOT}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/"
peps_su = load_or_generate_peps(
    Lx, Ly, 1.0, 8.0, N_f, D,
    seed=42, dtype=dtype, file_path=fpeps_base, scale_factor=4,
)
model_su = fPEPS_Model_GPU(
    tn=peps_su, max_bond=chi, dtype=dtype,
    contract_boundary_opts=contract_opts,
)
model_su.to(device)
diagnose("SU-loaded, scale_factor=4", model_su)
del model_su

# ============================================================
# Case 2: SU-loaded PEPS (scale_factor=1, no rescale)
# ============================================================
peps_su1 = load_or_generate_peps(
    Lx, Ly, 1.0, 8.0, N_f, D,
    seed=42, dtype=dtype, file_path=fpeps_base, scale_factor=1,
)
model_su1 = fPEPS_Model_GPU(
    tn=peps_su1, max_bond=chi, dtype=dtype,
    contract_boundary_opts=contract_opts,
)
model_su1.to(device)
diagnose("SU-loaded, scale_factor=1", model_su1)
del model_su1

# ============================================================
# Case 3: Random PEPS (symmray default)
# ============================================================
import symmray as sr
peps_rand = sr.networks.PEPS_fermionic_rand(
    "Z2", Lx, Ly, D,
    phys_dim=[(0, 0), (1, 1), (1, 0), (0, 1)],
    subsizes="equal", flat=True, seed=42,
    dtype="float64",
)
model_rand = fPEPS_Model_GPU(
    tn=peps_rand, max_bond=chi, dtype=dtype,
    contract_boundary_opts=contract_opts,
)
model_rand.to(device)
diagnose("Random PEPS (symmray default)", model_rand)
del model_rand

dist.destroy_process_group()
