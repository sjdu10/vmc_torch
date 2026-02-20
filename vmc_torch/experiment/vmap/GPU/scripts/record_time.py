"""
Timing benchmark: default SVD+QR vs size-aware dispatch for 8x8 D=chi=10 fPEPS.

For chi=10, boundary contraction matrices are ~10x10 (n<=32).
Main expected difference: QR-via-SVD (fused Jacobi kernel) vs default QR
(~3k sub-kernel launches per call).

Usage (no torchrun needed):
    cd VMC_code/vmc_torch/vmc_torch/experiment/vmap
    python GPU/scripts/record_time.py
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import autoray as ar
import numpy as np
import symmray as sr
import torch

from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    random_initial_config,
)
from vmc_torch.experiment.vmap.models.pureTNS import fPEPS_Model
from vmc_torch.experiment.vmap.vmap_torch_utils import (
    robust_svd_err_catcher_wrapper,
    size_aware_qr,
    size_aware_svd,
)

# ==========================================
# 1. Parameters
# ==========================================
dtype = torch.float64
torch.set_default_dtype(dtype)

Lx, Ly = 8, 8
N_f = Lx * Ly
nsites = Lx * Ly
D = 10
chi = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
has_gpu = device.type == "cuda"

batch_sizes = [1, 2, 4, 8, 10]
N_WARMUP = 1
N_REPS = 1

SVD_JITTER = 1e-16
driver = None

# ==========================================
# 2. Build random PEPS
# ==========================================
peps = sr.networks.PEPS_fermionic_rand(
    "Z2",
    Lx,
    Ly,
    D,
    phys_dim=[
        (0, 0),
        (1, 1),
        (1, 0),
        (0, 1),
    ],
    subsizes="equal",
    flat=True,
    seed=42,
    dtype=str(dtype).split(".")[-1],
)


def build_models():
    """Build GPU and CPU fPEPS_Model from the same PEPS skeleton."""
    model_gpu = None
    if has_gpu:
        model_gpu = fPEPS_Model(tn=peps, max_bond=chi, dtype=dtype)
        model_gpu.to(device)
    model_cpu = fPEPS_Model(tn=peps, max_bond=chi, dtype=dtype)
    model_cpu.to("cpu")
    return model_gpu, model_cpu


def make_configs(batch_size):
    """Generate a batch of random half-filled configs."""
    fxs_list = [
        random_initial_config(N_f, nsites, seed=42 + i)
        for i in range(batch_size)
    ]
    return torch.stack(fxs_list)


# ==========================================
# 3. Timing helpers
# ==========================================
def time_gpu(model, fxs, n_warmup=N_WARMUP, n_reps=N_REPS):
    """Time GPU forward with CUDA events. Returns (mean_s, std_s)."""
    fxs_dev = fxs.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(fxs_dev)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        model(fxs_dev)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0  # ms -> s


def time_cpu(model, fxs, n_warmup=N_WARMUP, n_reps=N_REPS):
    """Time CPU forward with perf_counter. Returns (mean_s, std_s)."""
    fxs_cpu = fxs.cpu()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(fxs_cpu)

    t0 = time.perf_counter()
    with torch.no_grad():
        model(fxs_cpu)
    t1 = time.perf_counter()
    return t1 - t0


def run_benchmark(model_gpu, model_cpu, label):
    """Run timing for all batch sizes. Returns dict of results."""
    gpu_times, cpu_times = [], []

    for bs in batch_sizes:
        print(f"  batch_size={bs}")
        fxs = make_configs(bs)

        if has_gpu and model_gpu is not None:
            gt = time_gpu(model_gpu, fxs)
            gpu_times.append(gt)
            print(f"    GPU: {gt:.4f} s")
        else:
            gpu_times.append(-1.0)

        ct = time_cpu(model_cpu, fxs)
        cpu_times.append(ct)
        print(f"    CPU: {ct:.4f} s")

    return {
        "label": label,
        "batch_sizes": batch_sizes,
        "gpu_times": gpu_times,
        "cpu_times": cpu_times,
    }


# ==========================================
# 4. MODE A — Default SVD+QR (baseline)
# ==========================================
print("=" * 60)
print("MODE A: Default SVD + QR (baseline)")
print("  SVD -> robust_svd_err_catcher_wrapper")
print("  QR  -> torch.linalg.qr (default)")
print("=" * 60)

# Register robust SVD for numerical stability (no custom QR)
ar.register_function(
    "torch",
    "linalg.svd",
    lambda x: robust_svd_err_catcher_wrapper(
        x, jitter=SVD_JITTER, driver=driver
    ),
)

model_gpu_a, model_cpu_a = build_models()
results_a = run_benchmark(model_gpu_a, model_cpu_a, "default")

# ==========================================
# 5. MODE B — Size-aware dispatch
# ==========================================
print()
print("=" * 60)
print("MODE B: Size-aware SVD + QR dispatch")
print("  SVD -> size_aware_svd (Jacobi n<=32, EIG+MAGMA n>32)")
print("  QR  -> size_aware_qr (QR-via-SVD on GPU for n<=32)")
print("=" * 60)

# Overwrite autoray registrations
ar.register_function(
    "torch",
    "linalg.svd",
    lambda x: size_aware_svd(x, jitter=SVD_JITTER, driver=driver),
)
ar.register_function("torch", "linalg.qr", size_aware_qr)

# Rebuild models so vmap traces through new dispatch
model_gpu_b, model_cpu_b = build_models()
results_b = run_benchmark(model_gpu_b, model_cpu_b, "size_aware")

# ==========================================
# 6. Comparison table and save
# ==========================================
print()
print("=" * 60)
print(
    f"COMPARISON: 8x8 D={D} chi={chi}  "
    f"({N_WARMUP} warmup + {N_REPS} reps)"
)
print("=" * 60)

header = (
    f"{'BS':>4}  "
    f"{'GPU default':>12}  {'GPU size-aware':>14}  {'GPU speedup':>11}  "
    f"{'CPU default':>12}  {'CPU size-aware':>14}  {'CPU speedup':>11}"
)
print(header)
print("-" * len(header))

for i, bs in enumerate(batch_sizes):
    gt_a = results_a["gpu_times"][i]
    gt_b = results_b["gpu_times"][i]
    ct_a = results_a["cpu_times"][i]
    ct_b = results_b["cpu_times"][i]

    gpu_speedup = gt_a / gt_b if gt_a > 0 and gt_b > 0 else float("nan")
    cpu_speedup = ct_a / ct_b if ct_a > 0 and ct_b > 0 else float("nan")

    def fmt(t):
        return "N/A".rjust(12) if t < 0 else f"{t:.4f}s"

    print(
        f"{bs:>4}  "
        f"{fmt(gt_a):>12}  {fmt(gt_b):>14}  {gpu_speedup:>10.2f}x  "
        f"{fmt(ct_a):>12}  {fmt(ct_b):>14}  {cpu_speedup:>10.2f}x"
    )

# Save results
out_file = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    f"{Lx}x{Ly}_D={D}_chi={chi}_dispatch_comparison_{str(dtype).split('.')[-1]}.npy",
)
os.makedirs(os.path.dirname(out_file), exist_ok=True)
np.save(
    out_file,
    {"default": results_a, "size_aware": results_b},
)
print(f"\nSaved to {out_file}")
