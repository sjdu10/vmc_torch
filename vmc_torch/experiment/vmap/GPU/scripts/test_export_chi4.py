"""
Test torch.export + vmap + compile with chi=4 (boundary contraction).

Run with: torchrun --nproc_per_node=1 GPU_test_export_chi4.py
"""
import os
import time
import pickle
import torch
import torch.distributed as dist
import autoray as ar
import quimb as qu
import quimb.tensor as qtn

from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    random_initial_config,
)
from vmc_torch.experiment.vmap.GPU.torch_utils import (
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
chi = 4
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

params, skel = qtn.pack(peps)
params_flat, params_pytree = qu.utils.tree_flatten(params, get_ref=True)
params_tensors = [
    torch.as_tensor(x, dtype=torch.float64, device=device)
    for x in params_flat
]

B = 500
fxs = torch.stack([
    random_initial_config(N_f, nsites, seed=42 + i)
    for i in range(B)
]).to(device)


# ============================================================
# chi=-1: exact contraction (no boundary compression)
# ============================================================
print("=" * 60)
print("chi=-1 (exact contraction)")
print("=" * 60)


def amplitude_exact(x, *flat_params):
    p = qu.utils.tree_unflatten(list(flat_params), params_pytree)
    tn = qtn.unpack(p, skel)
    tnx = tn.isel({
        tn.site_ind(site): x[i]
        for i, site in enumerate(tn.sites)
    })
    return tnx.contract()


vf_exact = torch.vmap(
    amplitude_exact, in_dims=(0, *([None] * len(params_tensors)))
)
with torch.inference_mode():
    ref_exact = vf_exact(fxs, *params_tensors)
torch.cuda.synchronize()

print("vmap(eager):")
for trial in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        _ = vf_exact(fxs, *params_tensors)
    torch.cuda.synchronize()
    print(f"  {time.time()-t0:.4f}s")


# ============================================================
# chi=4: boundary contraction with SVD
# ============================================================
print("\n" + "=" * 60)
print(f"chi={chi} (boundary contraction with SVD)")
print("=" * 60)

contract_opts = {
    'mode': 'mps', 'equalize_norms': 1.0, 'canonize': True,
}


def amplitude_chi(x, *flat_params):
    p = qu.utils.tree_unflatten(list(flat_params), params_pytree)
    tn = qtn.unpack(p, skel)
    tnx = tn.isel({
        tn.site_ind(site): x[i]
        for i, site in enumerate(tn.sites)
    })
    tnx.contract_boundary_from_xmin_(
        max_bond=chi, cutoff=0.0,
        xrange=[0, tnx.Lx // 2 - 1],
        **contract_opts,
    )
    tnx.contract_boundary_from_xmax_(
        max_bond=chi, cutoff=0.0,
        xrange=[tnx.Lx // 2, tnx.Lx - 1],
        **contract_opts,
    )
    return tnx.contract()


vf_chi = torch.vmap(
    amplitude_chi, in_dims=(0, *([None] * len(params_tensors)))
)
with torch.inference_mode():
    ref_chi = vf_chi(fxs, *params_tensors)
torch.cuda.synchronize()

print("vmap(eager):")
for trial in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        _ = vf_chi(fxs, *params_tensors)
    torch.cuda.synchronize()
    print(f"  {time.time()-t0:.4f}s")


# ============================================================
# Export chi=-1
# ============================================================
print("\n" + "=" * 60)
print("torch.export + vmap + compile: chi=-1")
print("=" * 60)

from torch.export import export


class AmpExact(torch.nn.Module):
    def forward(self, x, *flat_params):
        return amplitude_exact(x, *flat_params)


t_exp = time.time()
with torch.inference_mode():
    exported_exact = export(AmpExact(), (fxs[0], *params_tensors))
print(f"Export time: {time.time()-t_exp:.2f}s")

vf_exp_exact = torch.vmap(
    exported_exact.module(),
    in_dims=(0, *([None] * len(params_tensors))),
)
vf_compiled_exact = torch.compile(vf_exp_exact, mode='default')

t_comp = time.time()
with torch.inference_mode():
    out = vf_compiled_exact(fxs, *params_tensors)
torch.cuda.synchronize()
print(f"Compile + first call: {time.time()-t_comp:.2f}s")
print(f"Output matches: {torch.allclose(out, ref_exact)}")

for trial in range(5):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        out = vf_compiled_exact(fxs, *params_tensors)
    torch.cuda.synchronize()
    print(f"  Call {trial}: {time.time()-t0:.4f}s")


# ============================================================
# Export chi=4
# ============================================================
print("\n" + "=" * 60)
print(f"torch.export + vmap + compile: chi={chi}")
print("=" * 60)


class AmpChi(torch.nn.Module):
    def forward(self, x, *flat_params):
        return amplitude_chi(x, *flat_params)


t_exp = time.time()
with torch.inference_mode():
    try:
        exported_chi = export(AmpChi(), (fxs[0], *params_tensors))
        print(f"Export time: {time.time()-t_exp:.2f}s")

        vf_exp_chi = torch.vmap(
            exported_chi.module(),
            in_dims=(0, *([None] * len(params_tensors))),
        )
        vf_compiled_chi = torch.compile(vf_exp_chi, mode='default')

        t_comp = time.time()
        with torch.inference_mode():
            out_chi = vf_compiled_chi(fxs, *params_tensors)
        torch.cuda.synchronize()
        print(f"Compile + first call: {time.time()-t_comp:.2f}s")
        print(f"Output matches: {torch.allclose(out_chi, ref_chi)}")

        for trial in range(5):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                out_chi = vf_compiled_chi(fxs, *params_tensors)
            torch.cuda.synchronize()
            print(f"  Call {trial}: {time.time()-t0:.4f}s")

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

dist.destroy_process_group()
