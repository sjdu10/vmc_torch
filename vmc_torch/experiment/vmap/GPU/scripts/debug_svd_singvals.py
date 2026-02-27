"""Inspect singular values during boundary contraction.

Monkey-patches torch.linalg.svd to log all singular values
encountered during forward+backward. Checks for near-degenerate
pairs that blow up the SVD backward (1/(s_i^2 - s_j^2) terms).

Run: torchrun --nproc_per_node=1 scripts/debug_svd_singvals.py
"""
import os
import torch
import torch.distributed as dist

from vmc_torch.experiment.vmap.GPU.vmc_setup import (
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.experiment.vmap.GPU.models import fPEPS_Model_GPU
from vmc_torch.experiment.vmap.GPU.vmc_utils import random_initial_config

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
    'vmc_torch/experiment/vmap/GPU/data'
)
fpeps_base = (
    f"{DATA_ROOT}/{Lx}x{Ly}/t=1.0_U=8.0"
    f"/N={N_f}/Z2/D={D}/"
)

# ============================================================
# SVD logging via monkey-patching torch.linalg.svd
# ============================================================
svd_log = []
_real_svd = torch.linalg.svd


def _logging_svd(A, full_matrices=True, *, driver=None):
    U, S, Vh = _real_svd(A, full_matrices=full_matrices,
                         driver=driver)
    svd_log.append((A.shape, S.detach().cpu().clone()))
    return U, S, Vh


# Also patch eigh since RobustSVD_EIG uses eigh
eigh_log = []
_real_eigh = torch.linalg.eigh


def _logging_eigh(A, UPLO='L'):
    w, V = _real_eigh(A, UPLO=UPLO)
    eigh_log.append((A.shape, w.detach().cpu().clone()))
    return w, V


def enable_logging():
    torch.linalg.svd = _logging_svd
    torch.linalg.eigh = _logging_eigh


def disable_logging():
    torch.linalg.svd = _real_svd
    torch.linalg.eigh = _real_eigh


def analyze_log(label):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  SVD calls: {len(svd_log)}, "
          f"eigh calls: {len(eigh_log)}")
    print(f"{'=' * 60}")

    # Analyze SVD singular values
    if svd_log:
        print("\n  --- SVD singular values ---")
        min_gaps = []
        cond_numbers = []
        for i, (shape, S) in enumerate(svd_log):
            S_sorted = S.sort(descending=True).values
            if S_sorted.numel() < 2:
                continue
            cond = (
                S_sorted[0] / S_sorted[-1]
            ).item() if S_sorted[-1] > 0 else float('inf')
            cond_numbers.append(cond)
            gaps = (
                S_sorted[:-1] ** 2 - S_sorted[1:] ** 2
            ).abs()
            min_gaps.append(gaps.min().item())

        if cond_numbers:
            ct = torch.tensor(cond_numbers)
            print(
                f"  Condition numbers: "
                f"min={ct.min().item():.4e}, "
                f"median={ct.median().item():.4e}, "
                f"max={ct.max().item():.4e}"
            )
        if min_gaps:
            gt = torch.tensor(min_gaps)
            print(
                f"  Min s^2-gap: "
                f"min={gt.min().item():.4e}, "
                f"median={gt.median().item():.4e}, "
                f"max={gt.max().item():.4e}"
            )
            print(
                f"  #(gap<1e-10)={int((gt < 1e-10).sum())}, "
                f"#(gap<1e-6)={int((gt < 1e-6).sum())}, "
                f"#(gap<1e-3)={int((gt < 1e-3).sum())}"
            )

        # Show worst case
        if min_gaps:
            worst_i = min_gaps.index(min(min_gaps))
            shape, S = svd_log[worst_i]
            S_sorted = S.sort(descending=True).values
            print(
                f"\n  Worst SVD (call #{worst_i}):"
            )
            print(f"    shape={shape}")
            print(f"    S={S_sorted.tolist()}")

    # Analyze eigh eigenvalues
    if eigh_log:
        print("\n  --- eigh eigenvalues ---")
        all_min_gaps = []
        all_conds = []
        for i, (shape, w) in enumerate(eigh_log):
            # w may be 1D or multi-dim (batched eigh)
            if w.dim() == 1:
                ws = [w]
            else:
                ws = [w[j] for j in range(w.shape[0])]

            for j, wj in enumerate(ws):
                wj_s, _ = wj.sort(descending=True)
                if wj_s.numel() < 2:
                    continue
                wa = wj_s.abs()
                if wa[-1] > 0:
                    all_conds.append(
                        (wa[0] / wa[-1]).item()
                    )
                # eigh backward has 1/(w_i - w_j) terms
                gaps = (wj_s[:-1] - wj_s[1:]).abs()
                mg = gaps.min().item()
                all_min_gaps.append(mg)

            print(
                f"  call #{i}: input {shape}, "
                f"eigenval shape {w.shape}"
            )
            # Print a few representative eigenvalue vectors
            for j, wj in enumerate(ws[:3]):
                wj_s, _ = wj.sort(descending=True)
                print(
                    f"    [{j}] {wj_s.tolist()}"
                )
            if len(ws) > 3:
                print(f"    ... ({len(ws)} total)")

        if all_conds:
            ct = torch.tensor(all_conds)
            print(
                f"\n  Condition numbers ({len(all_conds)} "
                f"matrices): "
                f"min={ct.min().item():.4e}, "
                f"median={ct.median().item():.4e}, "
                f"max={ct.max().item():.4e}"
            )
        if all_min_gaps:
            gt = torch.tensor(all_min_gaps)
            print(
                f"  Min eigenvalue gap "
                f"({len(all_min_gaps)} matrices): "
                f"min={gt.min().item():.4e}, "
                f"median={gt.median().item():.4e}, "
                f"max={gt.max().item():.4e}"
            )
            print(
                f"  #(gap<1e-10)={int((gt < 1e-10).sum())}"
                f", #(gap<1e-6)={int((gt < 1e-6).sum())}"
                f", #(gap<1e-3)={int((gt < 1e-3).sum())}"
            )


# ============================================================
# Setup linalg hooks first, then enable logging
# ============================================================
setup_linalg_hooks(jitter=1e-12)

fxs = torch.stack([
    random_initial_config(N_f, N_sites, seed=42 + i)
    for i in range(1)
]).to(device)

# ============================================================
# Test 1: SU-loaded PEPS, forward + backward
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

enable_logging()
svd_log.clear()
eigh_log.clear()

# Use single-sample path (no vmap) so eigenvalues
# are plain tensors, not batched.
x_single = fxs[0]
flat_params = [p for p in model.params]
amp = model._amplitude_for_export(x_single, *flat_params)
print(f"SU amp: {amp.item():.6e}")
amp.backward()
model.zero_grad()

analyze_log("SU-loaded, chi=10: forward+backward")
disable_logging()

# ============================================================
# Test 2: Random PEPS, forward + backward
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
    contract_boundary_opts={
        'mode': 'mps', 'equalize_norms': 1.0,
        'canonize': True,
    },
)
model_rand.to(device)

enable_logging()
svd_log.clear()
eigh_log.clear()

flat_params_r = [p for p in model_rand.params]
amp_r = model_rand._amplitude_for_export(
    x_single, *flat_params_r
)
print(f"\nRandom amp: {amp_r.item():.6e}")
amp_r.backward()

analyze_log("Random, chi=10: forward+backward")
disable_logging()

dist.destroy_process_group()
