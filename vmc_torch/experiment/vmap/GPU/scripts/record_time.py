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

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
N_WARMUP = 1
N_REPS = 2

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
        for _ in range(n_reps):
            model(fxs_dev)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0 / n_reps  # ms -> s


def time_cpu(model, fxs, n_warmup=N_WARMUP, n_reps=N_REPS):
    """Time CPU forward with perf_counter. Returns (mean_s, std_s)."""
    fxs_cpu = fxs.cpu()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(fxs_cpu)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_reps):
            model(fxs_cpu)
    t1 = time.perf_counter()
    return (t1 - t0) / n_reps


def run_benchmark(model_gpu, model_cpu, label, record_gpu=True, record_cpu=True):
    """Run timing for all batch sizes. Returns dict of results."""
    gpu_times, cpu_times = [], []

    for bs in batch_sizes:
        print(f"  batch_size={bs}")
        fxs = make_configs(bs)

        if has_gpu and model_gpu is not None and record_gpu:
            gt = time_gpu(model_gpu, fxs)
            gpu_times.append(gt)
            print(f"    GPU: {gt:.4f} s")
        else:
            gpu_times.append(-1.0)
        if model_cpu is not None and record_cpu:
            ct = time_cpu(model_cpu, fxs)
            cpu_times.append(ct)
            print(f"    CPU: {ct:.4f} s")
        else:            
            cpu_times.append(-1.0)

    return {
        "label": label,
        "batch_sizes": batch_sizes,
        "gpu_times": gpu_times,
        "cpu_times": cpu_times,
    }


# # ==========================================
# # 4. MODE A — Default SVD+QR (baseline)
# # ==========================================
# print("=" * 60)
# print("MODE A: Default SVD + QR (baseline)")
# print("  SVD -> robust_svd_err_catcher_wrapper")
# print("  QR  -> torch.linalg.qr (default)")
# print("=" * 60)

# # Register robust SVD for numerical stability (no custom QR)
# ar.register_function(
#     "torch",
#     "linalg.svd",
#     lambda x: robust_svd_err_catcher_wrapper(
#         x, jitter=SVD_JITTER, driver=driver
#     ),
# )

# model_gpu_a, model_cpu_a = build_models()
# results_a = run_benchmark(model_gpu_a, model_cpu_a, "default")
# # ==========================================
# # Print Mode A results
# # ==========================================
# print()
# print("=" * 60)
# print(
#     f"MODE A RESULTS: {Lx}x{Ly} D={D} chi={chi}  "
#     f"({N_WARMUP} warmup + {N_REPS} reps)"
# )
# print("=" * 60)

# print(f"{'BS':>4}  {'GPU (s)':>10}  {'CPU (s)':>10}")
# print("-" * 30)
# for i, bs in enumerate(batch_sizes):
#     gt = results_a["gpu_times"][i]
#     ct = results_a["cpu_times"][i]
#     gpu_str = f"{gt:.4f}" if gt > 0 else "N/A"
#     cpu_str = f"{ct:.4f}" if ct > 0 else "N/A"
#     print(f"{bs:>4}  {gpu_str:>10}  {cpu_str:>10}")

# # Save Mode A results
# out_file = os.path.join(
#     os.path.dirname(__file__),
#     "..",
#     "data",
#     f"{Lx}x{Ly}_D={D}_chi={chi}_default_{str(dtype).split('.')[-1]}.npy",
# )
# os.makedirs(os.path.dirname(out_file), exist_ok=True)
# np.save(out_file, {"default": results_a})
# print(f"\nSaved to {out_file}")



# # ==========================================
# # 5. MODE B — Size-aware dispatch
# # ==========================================
# print()
# print("=" * 60)
# print("MODE B: Size-aware SVD + QR dispatch")
# print("  SVD -> size_aware_svd (Jacobi n<=32, EIG+MAGMA n>32)")
# print("  QR  -> size_aware_qr (QR-via-SVD on GPU for n<=32)")
# print("=" * 60)

# # Overwrite autoray registrations
# ar.register_function(
#     "torch",
#     "linalg.svd",
#     lambda x: size_aware_svd(x, jitter=SVD_JITTER, driver=driver),
# )
# ar.register_function("torch", "linalg.qr", size_aware_qr)

# # Rebuild models so vmap traces through new dispatch
# model_gpu_b, model_cpu_b = build_models()
# results_b = run_benchmark(model_gpu_b, model_cpu_b, "size_aware")
# # ==========================================
# # Print Mode B results
# # ==========================================
# print()
# print("=" * 60)
# print(
#     f"MODE B RESULTS: {Lx}x{Ly} D={D} chi={chi}  "
#     f"({N_WARMUP} warmup + {N_REPS} reps)"
# )
# print("=" * 60)

# print(f"{'BS':>4}  {'GPU (s)':>10}  {'CPU (s)':>10}")
# print("-" * 30)
# for i, bs in enumerate(batch_sizes):
#     gt = results_b["gpu_times"][i]
#     ct = results_b["cpu_times"][i]
#     gpu_str = f"{gt:.4f}" if gt > 0 else "N/A"
#     cpu_str = f"{ct:.4f}" if ct > 0 else "N/A"
#     print(f"{bs:>4}  {gpu_str:>10}  {cpu_str:>10}")

# # Save Mode B results
# out_file = os.path.join(
#     os.path.dirname(__file__),
#     "..",
#     "data",
#     f"{Lx}x{Ly}_D={D}_chi={chi}_MAGMA_eigh_{str(dtype).split('.')[-1]}.npy",
# )
# os.makedirs(os.path.dirname(out_file), exist_ok=True)
# np.save(out_file, {"size_aware+MAGMA_eigh": results_b})
# print(f"\nSaved to {out_file}")


# ==========================================
# 6. MODE C — Size-aware + XsyevBatched eigh (via C++ extension)
# ==========================================
# Patched torch removes the `n <= 32` gate in eigh dispatch, so
# stock torch.linalg.eigh now uses XsyevBatched for all batched
# matrices. No C++ extension or eigh monkey-patching needed.
print()
qr_via_eigh = False
print("=" * 60)
print("MODE C: Size-aware QR + Size-aware SVD with XsyevBatched eigh (via C++ extension)")
print("  SVD  -> size_aware_svd (Jacobi n<=32, EIG+XsyevBatched n>32)")
print(f"  QR   -> size_aware_qr (QR-via-SVD{'-via-EIGH' if qr_via_eigh else ''} on GPU)")
print("  eigh -> stock torch.linalg.eigh (patched: XsyevBatched)")
print("=" * 60)

ar.register_function(
    "torch",
    "linalg.svd",
    lambda x: size_aware_svd(x, jitter=SVD_JITTER, driver=driver, backend='cuSOLVER'),
)

ar.register_function("torch", "linalg.qr", lambda x: size_aware_qr(x, via_eigh=qr_via_eigh, jitter=0.0))

model_gpu_c, model_cpu_c = build_models()
results_c = run_benchmark(model_gpu_c, model_cpu_c, "size_aware+syevBatched_eigh", record_gpu=True, record_cpu=True)
# ==========================================
# Print Mode C results
# ==========================================
print()
print("=" * 60)
print(
    f"MODE C RESULTS: {Lx}x{Ly} D={D} chi={chi}  "
    f"({N_WARMUP} warmup + {N_REPS} reps)"
)
print("=" * 60)

print(f"{'BS':>4}  {'GPU (s)':>10}  {'CPU (s)':>10}")
print("-" * 30)
for i, bs in enumerate(batch_sizes):
    gt = results_c["gpu_times"][i]
    ct = results_c["cpu_times"][i]
    gpu_str = f"{gt:.4f}" if gt > 0 else "N/A"
    cpu_str = f"{ct:.4f}" if ct > 0 else "N/A"
    print(f"{bs:>4}  {gpu_str:>10}  {cpu_str:>10}")

# Save Mode C results
out_file = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    f"{Lx}x{Ly}_D={D}_chi={chi}_syevBatched_eigh{'_qr_eigh' if qr_via_eigh else ''}_{str(dtype).split('.')[-1]}.npy",
)
os.makedirs(os.path.dirname(out_file), exist_ok=True)
np.save(out_file, {"size_aware+syevBatched_eigh"+f'{'_qr_eigh' if qr_via_eigh else ''}': results_c})
print(f"\nSaved to {out_file}")



