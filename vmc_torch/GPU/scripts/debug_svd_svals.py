"""Log SVD S values from RobustSVD_EIG during boundary contraction.

The boundary contraction path is:
  autoray linalg.qr -> size_aware_qr(via_eigh=True) -> qr_via_eigh
    -> RobustSVD_EIG.apply(x, jitter, None)

No linalg.svd call is made. We hook RobustSVD_EIG.forward directly.

Run: torchrun --nproc_per_node=1 scripts/debug_svd_svals.py
"""
import os
import torch
import torch.distributed as dist

from vmc_torch.GPU.vmc_setup import (
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.GPU.models import fPEPS_Model_GPU
from vmc_torch.GPU.vmc_utils import random_initial_config
import vmc_torch.GPU.torch_utils as tu

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

DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
    'vmc_torch/GPU/data'
)
fpeps_base = (
    f"{DATA_ROOT}/{Lx}x{Ly}/t=1.0_U=8.0"
    f"/N={N_f}/Z2/D={D}/"
)

# ============================================================
# Monkey-patch qr_via_eigh to log S from RobustSVD_EIG
# ============================================================
svd_log = []
_orig_qr_via_eigh = tu.qr_via_eigh


def _logging_qr_via_eigh(x, jitter=1e-12):
    """Wrap qr_via_eigh: call RobustSVD_EIG.apply, log S."""
    U, S, Vh = tu.RobustSVD_EIG.apply(x, jitter, None)
    svd_log.append({
        'shape': tuple(x.shape),
        'S': S.detach().cpu().clone(),
    })
    R = S.unsqueeze(-1) * Vh
    return U, R


# Patch the module-level function so size_aware_qr calls it
tu.qr_via_eigh = _logging_qr_via_eigh

# Now register autoray hooks — size_aware_qr(via_eigh=True)
# calls tu.qr_via_eigh which is our patched version
import autoray as ar
ar.register_function(
    'torch', 'linalg.qr',
    lambda x: tu.size_aware_qr(x, via_eigh=True, jitter=1e-12),
)
ar.register_function(
    'torch', 'linalg.svd',
    lambda x: tu.size_aware_svd(x, jitter=1e-12),
)

x = random_initial_config(N_f, N_sites, seed=42).to(device)


def analyze(label):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  RobustSVD_EIG calls: {len(svd_log)}")
    print(f"{'=' * 60}")

    all_min_gaps = []
    all_conds = []

    for i, info in enumerate(svd_log):
        S = info['S']
        shape = info['shape']
        if S.dim() == 1:
            Ss = [S]
        else:
            Ss = [S[j] for j in range(S.shape[0])]

        print(f"\n  SVD #{i}: A shape={shape}")
        for j, s in enumerate(Ss):
            s_sorted, _ = s.sort(descending=True)
            if s_sorted.numel() >= 2:
                diffs = (s_sorted[:-1] - s_sorted[1:]).abs()
                min_gap = diffs.min().item()
                s_min = s_sorted[-1].item()
                cond = (
                    s_sorted[0].item()
                    / max(abs(s_min), 1e-30)
                )
                all_min_gaps.append(min_gap)
                all_conds.append(cond)
            else:
                min_gap = float('inf')
                cond = 1.0
            print(
                f"    [{j}] S = "
                f"{[f'{v:.6e}' for v in s_sorted.tolist()]}"
            )
            print(
                f"        cond={cond:.4e}, "
                f"min_gap(si-sj)={min_gap:.4e}"
            )

    if all_min_gaps:
        gt = torch.tensor(all_min_gaps)
        ct = torch.tensor(all_conds)
        print(f"\n  Summary over {len(all_min_gaps)} matrices:")
        print(
            f"  Condition: min={ct.min():.4e}, "
            f"median={ct.median():.4e}, "
            f"max={ct.max():.4e}"
        )
        print(
            f"  Min s_gap: min={gt.min():.4e}, "
            f"median={gt.median():.4e}, "
            f"max={gt.max():.4e}"
        )
        print(
            f"  #(gap<1e-10)={int((gt < 1e-10).sum())}, "
            f"#(gap<1e-6)={int((gt < 1e-6).sum())}, "
            f"#(gap<1e-3)={int((gt < 1e-3).sum())}"
        )


# ============================================================
# SU-loaded: single "normal" sample (sample 0)
# ============================================================
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

svd_log.clear()
amp0 = model._amplitude_for_export(
    x, *list(model.params)
)
print(f"Sample 0 amp: {amp0.item():.6e}")
amp0.backward()
g0_norm = torch.cat([
    p.grad.flatten() for p in model.params
]).norm().item()
print(f"Sample 0 grad norm: {g0_norm:.4e}")
model.zero_grad()
analyze("SU sample 0 (normal)")

# ============================================================
# SU-loaded: "outlier" sample (sample 3 was bad in B=16 test)
# ============================================================
x3 = random_initial_config(
    N_f, N_sites, seed=45
).to(device)

svd_log.clear()
amp3 = model._amplitude_for_export(
    x3, *list(model.params)
)
print(f"\nSample 3 amp: {amp3.item():.6e}")
amp3.backward()
g3_norm = torch.cat([
    p.grad.flatten() for p in model.params
]).norm().item()
print(f"Sample 3 grad norm: {g3_norm:.4e}")
model.zero_grad()
analyze("SU sample 3 (outlier)")

dist.destroy_process_group()
