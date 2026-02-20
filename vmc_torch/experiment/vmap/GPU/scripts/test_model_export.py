"""
Test fPEPS_Model_GPU with export_and_compile.

Run with: torchrun --nproc_per_node=1 GPU_test_model_export.py
"""
import os
import time
import pickle
import torch
import torch.distributed as dist
import autoray as ar
import quimb.tensor as qtn

from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    random_initial_config,
)
from vmc_torch.experiment.vmap.GPU.models import fPEPS_Model_GPU
from vmc_torch.experiment.vmap.vmap_torch_utils import (
    robust_svd_err_catcher_wrapper,
)

ar.register_function(
    'torch', 'linalg.svd',
    lambda x: robust_svd_err_catcher_wrapper(
        x, jitter=1e-16, driver=None
    ),
)

# Setup
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

# Load PEPS
Lx, Ly = 4, 2
nsites = Lx * Ly
N_f = nsites - 2
D = 4
pwd = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
    'vmc_torch/experiment/vmap/data'
)
params_pkl = pickle.load(open(
    f'{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/'
    'peps_su_params_U1SU.pkl', 'rb'
))
skeleton_pkl = pickle.load(open(
    f'{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/'
    'peps_skeleton_U1SU.pkl', 'rb'
))
peps = qtn.unpack(params_pkl, skeleton_pkl)
for ts in peps.tensors:
    ts.modify(data=ts.data.to_flat() * 4)
for site in peps.sites:
    peps[site].data._label = site
    peps[site].data.indices[-1]._linearmap = (
        (0, 0), (1, 0), (1, 1), (0, 1)
    )

B = 500
fxs = torch.stack([
    random_initial_config(N_f, nsites, seed=42 + i)
    for i in range(B)
]).to(device)


for chi_val in [-1, 4]:
    print("=" * 60)
    print(f"chi={chi_val}")
    print("=" * 60)

    model = fPEPS_Model_GPU(
        tn=peps, max_bond=chi_val, dtype=torch.float64,
        contract_boundary_opts={
            'mode': 'mps', 'equalize_norms': 1.0,
            'canonize': True,
        }
    )
    model.to(device)

    # --- Eager baseline ---
    with torch.inference_mode():
        out_ref = model(fxs)
    torch.cuda.synchronize()

    print("\n--- Eager (no export) ---")
    for trial in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model(fxs)
        torch.cuda.synchronize()
        print(f"  Call {trial}: {time.time()-t0:.4f}s")

    # --- Export + compile ---
    print("\n--- export_and_compile ---")
    t_setup = time.time()
    model.export_and_compile(fxs[0], mode='default')
    print(f"  Export + compile setup: {time.time()-t_setup:.2f}s")

    # First compiled call (triggers actual compilation)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        out_comp = model(fxs)
    torch.cuda.synchronize()
    print(f"  First compiled call: {time.time()-t0:.2f}s")
    print(f"  Output matches: {torch.allclose(out_comp, out_ref)}")

    # Steady-state benchmark
    for trial in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out_comp = model(fxs)
        torch.cuda.synchronize()
        print(f"  Call {trial}: {time.time()-t0:.4f}s")

    # --- Test with different configs ---
    fxs2 = torch.stack([
        random_initial_config(N_f, nsites, seed=1000 + i)
        for i in range(B)
    ]).to(device)

    # Get eager reference with new configs
    model_eager = fPEPS_Model_GPU(
        tn=peps, max_bond=chi_val, dtype=torch.float64,
        contract_boundary_opts={
            'mode': 'mps', 'equalize_norms': 1.0,
            'canonize': True,
        }
    )
    model_eager.to(device)
    # Copy the same params
    with torch.no_grad():
        for p_new, p_old in zip(
            model_eager.parameters(), model.parameters()
        ):
            p_new.copy_(p_old)

    with torch.inference_mode():
        ref_new = model_eager(fxs2)
        comp_new = model(fxs2)
    print(f"\n  Different configs match: "
          f"{torch.allclose(comp_new, ref_new)}")

    # --- Test grad compatibility ---
    print("\n--- Gradient test ---")
    # exported module doesn't support .train(), but gradients
    # flow through self.params which are nn.Parameter
    out_grad = model(fxs[:10])
    loss = out_grad.abs().sum()
    try:
        loss.backward()
        grad_ok = all(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        print(f"  Gradients flow: {grad_ok}")
        model.zero_grad()
    except Exception as e:
        print(f"  Gradient FAILED: {type(e).__name__}: {e}")

    print()

dist.destroy_process_group()
