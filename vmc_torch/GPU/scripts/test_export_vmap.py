"""
Benchmark torch.export + vmap for fPEPS amplitude.
torch.export captures the computation graph as pure aten ops,
then vmap can batch it, and compile can optimize it.

Run with: torchrun --nproc_per_node=1 GPU_test_export_vmap.py
"""
import os
import time
import pickle
import torch
import torch.distributed as dist
import autoray as ar
import quimb as qu
import quimb.tensor as qtn

from vmc_torch.GPU.vmc_utils import (
    random_initial_config,
)
from vmc_torch.GPU.torch_utils import (
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

params, skel = qtn.pack(peps)
params_flat, params_pytree = qu.utils.tree_flatten(params, get_ref=True)
params_tensors = [
    torch.as_tensor(x, dtype=torch.float64, device=device)
    for x in params_flat
]

for B in [50, 500]:
    fxs = torch.stack([
        random_initial_config(N_f, nsites, seed=42 + i)
        for i in range(B)
    ]).to(device)

    def amplitude_single(x, *flat_params):
        p = qu.utils.tree_unflatten(list(flat_params), params_pytree)
        tn = qtn.unpack(p, skel)
        tnx = tn.isel({
            tn.site_ind(site): x[i]
            for i, site in enumerate(tn.sites)
        })
        return tnx.contract()

    # ============================================================
    # Baseline: vmap(eager)
    # ============================================================
    print("=" * 60)
    print(f"B={B}")
    print("=" * 60)

    vf_eager = torch.vmap(
        amplitude_single,
        in_dims=(0, *([None] * len(params_tensors))),
    )
    with torch.inference_mode():
        out_ref = vf_eager(fxs, *params_tensors)
    torch.cuda.synchronize()

    print("\n--- vmap(eager) ---")
    for trial in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = vf_eager(fxs, *params_tensors)
        torch.cuda.synchronize()
        print(f"  Call {trial}: {time.time()-t0:.4f}s")

    # ============================================================
    # torch.export + vmap
    # ============================================================
    print("\n--- torch.export + vmap ---")
    from torch.export import export

    class AmpModule(torch.nn.Module):
        def forward(self, x, *flat_params):
            return amplitude_single(x, *flat_params)

    amp_mod = AmpModule()

    t_export0 = time.time()
    with torch.inference_mode():
        exported = export(amp_mod, (fxs[0], *params_tensors))
    t_export = time.time() - t_export0
    print(f"  Export time: {t_export:.2f}s")

    exported_mod = exported.module()
    vf_exported = torch.vmap(
        exported_mod,
        in_dims=(0, *([None] * len(params_tensors))),
    )

    # Warmup
    with torch.inference_mode():
        out_exp = vf_exported(fxs, *params_tensors)
    torch.cuda.synchronize()
    print(f"  Output matches: {torch.allclose(out_exp, out_ref)}")

    for trial in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out_exp = vf_exported(fxs, *params_tensors)
        torch.cuda.synchronize()
        print(f"  Call {trial}: {time.time()-t0:.4f}s")

    # ============================================================
    # torch.export + vmap + compile
    # ============================================================
    print("\n--- torch.export + vmap + compile ---")

    vf_exported2 = torch.vmap(
        exported.module(),
        in_dims=(0, *([None] * len(params_tensors))),
    )
    vf_compiled = torch.compile(vf_exported2, mode='default')

    t_comp0 = time.time()
    with torch.inference_mode():
        try:
            out_comp = vf_compiled(fxs, *params_tensors)
            torch.cuda.synchronize()
            t_comp = time.time() - t_comp0
            print(f"  Compile + first call: {t_comp:.2f}s")
            print(f"  Output matches: "
                  f"{torch.allclose(out_comp, out_ref)}")

            for trial in range(5):
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.inference_mode():
                    out_comp = vf_compiled(fxs, *params_tensors)
                torch.cuda.synchronize()
                print(f"  Call {trial}: {time.time()-t0:.4f}s")
        except Exception as e:
            print(f"  compile FAILED: {type(e).__name__}: {e}")

    # ============================================================
    # torch.export + compile (no vmap, single sample baseline)
    # ============================================================
    print("\n--- torch.export + compile (single sample) ---")
    exported_mod_c = torch.compile(exported.module(), mode='default')
    t_comp0 = time.time()
    with torch.inference_mode():
        try:
            out_sc = exported_mod_c(fxs[0], *params_tensors)
            torch.cuda.synchronize()
            print(f"  Compile + first call: {time.time()-t_comp0:.2f}s")

            for trial in range(5):
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.inference_mode():
                    _ = exported_mod_c(fxs[0], *params_tensors)
                torch.cuda.synchronize()
                print(f"  Single call {trial}: {time.time()-t0:.6f}s")
        except Exception as e:
            print(f"  compile FAILED: {type(e).__name__}: {e}")

    print()

dist.destroy_process_group()
