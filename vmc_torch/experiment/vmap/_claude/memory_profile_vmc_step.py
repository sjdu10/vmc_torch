"""
Memory profiling script for a single VMC step.

Compares OLD flow (separate grads + amps -> O_loc copy) vs
OPTIMIZED flow (inline O_loc, no extra copy).

Run with:
    /home/sijingdu/TNVMC/VMC_code/clean_symmray/bin/python \
        memory_profile_vmc_step.py

No MPI required — runs single-process with a local SR solver.
"""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gc
import time
import tracemalloc

import numpy as np
import psutil
import torch

# ── Memory snapshot utilities ────────────────────────────────
PROCESS = psutil.Process(os.getpid())
CHECKPOINTS = []


def _torch_tensor_bytes():
    """Total bytes of all live torch tensors (approximate)."""
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                total += obj.nelement() * obj.element_size()
        except Exception:
            pass
    return total


def snap(label, detail=""):
    """Record a memory snapshot."""
    gc.collect()
    rss = PROCESS.memory_info().rss
    heap_curr, heap_peak = tracemalloc.get_traced_memory()
    torch_bytes = _torch_tensor_bytes()
    entry = {
        "label": label,
        "rss_mb": rss / 1e6,
        "heap_mb": heap_curr / 1e6,
        "heap_peak_mb": heap_peak / 1e6,
        "torch_mb": torch_bytes / 1e6,
        "detail": detail,
    }
    CHECKPOINTS.append(entry)
    print(
        f"  [{label:45s}] "
        f"RSS={entry['rss_mb']:8.1f}MB  "
        f"heap={entry['heap_mb']:8.1f}MB  "
        f"torch={entry['torch_mb']:8.1f}MB"
        f"  {detail}"
    )
    return entry


def print_summary(title, checkpoints):
    """Print a table + delta analysis."""
    print("\n" + "=" * 95)
    print(f"  {title}")
    print("=" * 95)
    print(
        f"{'Checkpoint':<47s} {'RSS(MB)':>8s} "
        f"{'Heap(MB)':>9s} {'Torch(MB)':>10s} "
        f"{'dRSS(MB)':>9s}"
    )
    print("-" * 95)
    prev_rss = checkpoints[0]["rss_mb"]
    for c in checkpoints:
        delta = c["rss_mb"] - prev_rss
        print(
            f"{c['label']:<47s} {c['rss_mb']:8.1f} "
            f"{c['heap_mb']:9.1f} {c['torch_mb']:10.1f} "
            f"{delta:+9.1f}"
        )
        prev_rss = c["rss_mb"]

    peak = max(checkpoints, key=lambda c: c["rss_mb"])
    base = checkpoints[0]
    print("-" * 95)
    print(
        f"Peak RSS: {peak['rss_mb']:.1f} MB "
        f"at [{peak['label'].strip()}]  "
        f"(+{peak['rss_mb'] - base['rss_mb']:.1f} MB "
        f"over baseline)"
    )
    print(
        f"Peak heap (tracemalloc): "
        f"{max(c['heap_peak_mb'] for c in checkpoints):.1f}"
        f" MB"
    )
    print("=" * 95)


# ── Start tracemalloc ────────────────────────────────────────
tracemalloc.start()

# ── Imports (heavy) ──────────────────────────────────────────
import pickle  # noqa: E402
from functools import partial  # noqa: E402

import autoray as ar  # noqa: E402
import quimb.tensor as qtn  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

from vmc_torch.experiment.vmap.vmap_utils import (  # noqa: E402
    compute_grads,
    evaluate_energy,
    random_initial_config,
    sample_next,
)
from vmc_torch.experiment.vmap.models import (  # noqa: E402
    Conv2D_Geometric_fPEPS_Model_Cluster,
)
from vmc_torch.experiment.vmap.vmap_torch_utils import (  # noqa: E402
    robust_svd_err_catcher_wrapper,
)
from vmc_torch.hamiltonian_torch import (  # noqa: E402
    spinful_Fermi_Hubbard_square_lattice_torch,
)

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

# ── SVD registration ─────────────────────────────────────────
SVD_JITTER = 1e-12
ar.register_function(
    "torch",
    "linalg.svd",
    lambda x: robust_svd_err_catcher_wrapper(
        x, jitter=SVD_JITTER, driver=None
    ),
)

# ── Torch config ─────────────────────────────────────────────
torch.set_default_device("cpu")
torch.random.manual_seed(42)
torch.set_num_threads(1)

snap("0. After imports")

# ==============================================================
# 1. Model & Hamiltonian initialization
# ==============================================================
Lx, Ly = 4, 4
N_f = 14
D, chi = 4, -2
t_hop, U = 1.0, 8.0

pwd = (
    "/home/sijingdu/TNVMC/VMC_code/vmc_torch/"
    "vmc_torch/experiment/vmap/data"
)
params_path = (
    f"{pwd}/{Lx}x{Ly}/t={t_hop}_U={U}/N={N_f}/Z2/D={D}/"
)
params = pickle.load(
    open(params_path + "peps_su_params_U1SU.pkl", "rb")
)
skeleton = pickle.load(
    open(params_path + "peps_skeleton_U1SU.pkl", "rb")
)
peps = qtn.unpack(params, skeleton)
for ts in peps.tensors:
    ts.modify(data=ts.data.to_flat() * 4)
for site in peps.sites:
    peps[site].data._label = site
    peps[site].data.indices[-1]._linearmap = (
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
    )

model_config = {
    "max_bond": chi,
    "embed_dim": 16,
    "attn_depth": 1,
    "attn_heads": 4,
    "nn_hidden_dim": peps.nsites,
    "init_perturbation_scale": 1e-3,
    "nn_eta": 1,
    "dtype_str": "float64",
    "uniform_kernel": 0,
}
model_dtype = torch.float64
init_kwargs = model_config.copy()
init_kwargs.pop("dtype_str")

fpeps_model = Conv2D_Geometric_fPEPS_Model_Cluster(
    tn=peps,
    dtype=model_dtype,
    layers=1,
    kernel_size=5,
    contract_boundary_opts={
        "mode": "mps",
        "equalize_norms": 1.0,
        "canonize": True,
    },
    **init_kwargs,
)

n_params = sum(p.numel() for p in fpeps_model.parameters())

H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx,
    Ly,
    t_hop,
    U,
    N_f,
    pbc=False,
    n_fermions_per_spin=(N_f // 2, N_f // 2),
    no_u1_symmetry=False,
)

B = 500
B_grad = 250

print(f"\nLattice: {Lx}x{Ly}, N_f={N_f}, D={D}, chi={chi}")
print(f"Model params: {n_params}")
print(f"B={B}, B_grad={B_grad}")

snap("1. After model+H init (baseline)")


# ==============================================================
# Helper: run one full VMC step and profile it
# ==============================================================
def run_vmc_step_old(tag_prefix):
    """OLD flow: grads + amps stored separately, O_loc copy."""
    fxs = torch.stack(
        [random_initial_config(N_f, peps.nsites) for _ in range(B)]
    ).to(torch.long)
    snap(f"{tag_prefix} 2. initial configs", f"B={B}")

    # --- sample ---
    t0 = time.perf_counter()
    fxs, current_amps = sample_next(
        fxs, fpeps_model, H.graph, hopping_rate=0.25
    )
    t_sample = time.perf_counter() - t0
    snap(f"{tag_prefix} 3. after sample_next")

    # --- energy ---
    t0 = time.perf_counter()
    energy, local_energies = evaluate_energy(
        fxs, fpeps_model, H, current_amps
    )
    t_energy = time.perf_counter() - t0
    snap(f"{tag_prefix} 4. after evaluate_energy")

    # --- grads ---
    get_grads = partial(
        compute_grads,
        vectorize=True,
        vmap_grad=True,
        batch_size=B_grad,
        verbose=False,
    )
    t0 = time.perf_counter()
    local_grads_torch, amps_torch = get_grads(fxs, fpeps_model)
    t_grad = time.perf_counter() - t0
    snap(
        f"{tag_prefix} 5. after compute_grads",
        f"grads={local_grads_torch.shape}",
    )

    # --- numpy offload (copy) ---
    local_energies_np = (
        local_energies.detach().cpu().numpy().copy()
    )
    local_grads_np = (
        local_grads_torch.detach().cpu().numpy().copy()
    )
    local_amps_np = (
        amps_torch.detach().cpu().numpy().flatten().copy()
    )
    del local_grads_torch, amps_torch
    del local_energies, current_amps, energy
    gc.collect()
    snap(f"{tag_prefix} 6. after numpy offload + del torch")

    # --- O_loc = grad / amp (NEW ALLOCATION) ---
    amps_reshaped = local_amps_np.reshape(-1, 1)
    local_O = local_grads_np / amps_reshaped  # copy!
    snap(
        f"{tag_prefix} 7. after O_loc = grad/amp (copy)",
        f"~{local_O.nbytes / 1e6:.1f}MB new",
    )

    # --- MINRES ---
    total_samples = B
    energy_mean = np.mean(local_energies_np)
    diag_shift = 1e-5
    mean_O = np.mean(local_O, axis=0)
    mean_EO = (
        np.dot(local_energies_np, local_O) / total_samples
    )
    energy_grad = mean_EO - energy_mean * mean_O

    def matvec(x):
        inner = local_O.dot(x)
        Sx = local_O.T.dot(inner) / total_samples
        Sx -= np.dot(mean_O, x) * mean_O
        return Sx + diag_shift * x

    A = spla.LinearOperator(
        (n_params, n_params), matvec=matvec, dtype=np.float64
    )
    t0 = time.perf_counter()
    dp, info = spla.minres(
        A, energy_grad, rtol=5e-5, maxiter=100
    )
    t_sr = time.perf_counter() - t0
    snap(f"{tag_prefix} 8. after MINRES solve")

    # --- param update ---
    with torch.no_grad():
        dp_tensor = torch.tensor(
            dp, device="cpu", dtype=model_dtype
        )
        curr_params = torch.nn.utils.parameters_to_vector(
            fpeps_model.parameters()
        )
        new_params = curr_params - 0.1 * dp_tensor
        torch.nn.utils.vector_to_parameters(
            new_params, fpeps_model.parameters()
        )
    del dp_tensor, curr_params, new_params, dp
    snap(f"{tag_prefix} 9. after param update")

    # --- cleanup ---
    del local_O, local_grads_np, local_amps_np
    del local_energies_np
    gc.collect()
    snap(f"{tag_prefix} 10. after cleanup")

    return t_sample, t_energy, t_grad, t_sr


def run_vmc_step_optimized(tag_prefix):
    """OPTIMIZED: inline O_loc, in-place division, no extra copy."""
    fxs = torch.stack(
        [random_initial_config(N_f, peps.nsites) for _ in range(B)]
    ).to(torch.long)
    snap(f"{tag_prefix} 2. initial configs", f"B={B}")

    # --- sample ---
    t0 = time.perf_counter()
    fxs, current_amps = sample_next(
        fxs, fpeps_model, H.graph, hopping_rate=0.25
    )
    t_sample = time.perf_counter() - t0
    snap(f"{tag_prefix} 3. after sample_next")

    # --- energy ---
    t0 = time.perf_counter()
    energy, local_energies = evaluate_energy(
        fxs, fpeps_model, H, current_amps
    )
    t_energy = time.perf_counter() - t0
    snap(f"{tag_prefix} 4. after evaluate_energy")

    # --- grads ---
    get_grads = partial(
        compute_grads,
        vectorize=True,
        vmap_grad=True,
        batch_size=B_grad,
        verbose=False,
    )
    t0 = time.perf_counter()
    local_grads_torch, amps_torch = get_grads(fxs, fpeps_model)
    t_grad = time.perf_counter() - t0
    snap(
        f"{tag_prefix} 5. after compute_grads",
        f"grads={local_grads_torch.shape}",
    )

    # --- TRICK #2: compute O_loc inline, no separate grads/amps
    local_energies_np = (
        local_energies.detach().cpu().numpy().copy()
    )
    # numpy view of torch tensor (no copy)
    g_np = local_grads_torch.detach().cpu().numpy()
    a_np = amps_torch.detach().cpu().numpy().ravel()
    # in-place division: g_np becomes O_loc
    g_np = g_np.copy()  # need owned memory for in-place
    g_np /= a_np.reshape(-1, 1)
    local_O = g_np  # this IS O_loc, no extra allocation

    # free torch tensors
    del local_grads_torch, amps_torch
    del local_energies, current_amps, energy
    gc.collect()
    snap(
        f"{tag_prefix} 6. after inline O_loc + del torch",
        "(no separate grads/amps numpy)",
    )

    # --- TRICK #1: no extra O_loc allocation in SR ---
    # local_O already computed above, skip the division step
    snap(
        f"{tag_prefix} 7. O_loc ready (no extra alloc)",
        f"~{local_O.nbytes / 1e6:.1f}MB total",
    )

    # --- MINRES ---
    total_samples = B
    energy_mean = np.mean(local_energies_np)
    diag_shift = 1e-5
    mean_O = np.mean(local_O, axis=0)
    mean_EO = (
        np.dot(local_energies_np, local_O) / total_samples
    )
    energy_grad = mean_EO - energy_mean * mean_O

    def matvec(x):
        inner = local_O.dot(x)
        Sx = local_O.T.dot(inner) / total_samples
        Sx -= np.dot(mean_O, x) * mean_O
        return Sx + diag_shift * x

    A = spla.LinearOperator(
        (n_params, n_params), matvec=matvec, dtype=np.float64
    )
    t0 = time.perf_counter()
    dp, info = spla.minres(
        A, energy_grad, rtol=5e-5, maxiter=100
    )
    t_sr = time.perf_counter() - t0
    snap(f"{tag_prefix} 8. after MINRES solve")

    # --- param update ---
    with torch.no_grad():
        dp_tensor = torch.tensor(
            dp, device="cpu", dtype=model_dtype
        )
        curr_params = torch.nn.utils.parameters_to_vector(
            fpeps_model.parameters()
        )
        new_params = curr_params - 0.1 * dp_tensor
        torch.nn.utils.vector_to_parameters(
            new_params, fpeps_model.parameters()
        )
    del dp_tensor, curr_params, new_params, dp
    snap(f"{tag_prefix} 9. after param update")

    # --- cleanup ---
    del local_O, local_energies_np
    gc.collect()
    snap(f"{tag_prefix} 10. after cleanup")

    return t_sample, t_energy, t_grad, t_sr


# ==============================================================
# Run OLD flow
# ==============================================================
print("\n" + "#" * 60)
print("# OLD FLOW (separate grads + amps, O_loc copy)")
print("#" * 60)
old_start_idx = len(CHECKPOINTS)
t_s, t_e, t_g, t_sr = run_vmc_step_old("OLD")
old_checkpoints = CHECKPOINTS[old_start_idx:]
old_peak = max(c["rss_mb"] for c in old_checkpoints)

# ==============================================================
# Run OPTIMIZED flow
# ==============================================================
print("\n" + "#" * 60)
print("# OPTIMIZED FLOW (inline O_loc, in-place division)")
print("#" * 60)
opt_start_idx = len(CHECKPOINTS)
t_s2, t_e2, t_g2, t_sr2 = run_vmc_step_optimized("OPT")
opt_checkpoints = CHECKPOINTS[opt_start_idx:]
opt_peak = max(c["rss_mb"] for c in opt_checkpoints)

# ==============================================================
# Summaries
# ==============================================================
print_summary("OLD FLOW", old_checkpoints)
print_summary("OPTIMIZED FLOW", opt_checkpoints)

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"OLD peak RSS:       {old_peak:.1f} MB")
print(f"OPTIMIZED peak RSS: {opt_peak:.1f} MB")
print(f"Savings:            {old_peak - opt_peak:.1f} MB "
      f"({(old_peak - opt_peak) / old_peak * 100:.1f}%)")
print(f"\nOLD timing:  sample={t_s:.1f}s  energy={t_e:.1f}s  "
      f"grad={t_g:.1f}s  SR={t_sr:.1f}s")
print(f"OPT timing:  sample={t_s2:.1f}s  energy={t_e2:.1f}s  "
      f"grad={t_g2:.1f}s  SR={t_sr2:.1f}s")
