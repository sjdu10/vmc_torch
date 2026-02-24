"""
Benchmark torch.linalg.{eigh, svd, qr} vs batch size on GPU.

Measures how linalg op time scales with batch size B for two matrix
sizes: n=32 (within Jacobi batched kernel threshold) and n=64
(above threshold). Uses the default cuSOLVER backend.

Usage:
    python GPU/scripts/linalg_batch_timing.py
"""

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch

# ==========================================
# Config
# ==========================================
N_WARMUP = 3
N_ITERS = 5
n_sizes = [32, 64]
B_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise RuntimeError("This benchmark requires a CUDA GPU.")

torch.backends.cuda.preferred_linalg_library("default")


# ==========================================
# Timing helper
# ==========================================
def time_op(fn, a, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Time a linalg op on GPU using CUDA events. Returns time in ms."""
    for _ in range(n_warmup):
        fn(a)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn(a)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


# ==========================================
# Run benchmarks
# ==========================================
results = {}

for n in n_sizes:
    eigh_times, svd_times, qr_times = [], [], []

    print(f"\n{'='*60}")
    print(f"n={n}, backend=default (cuSOLVER)")
    print(f"{'='*60}")

    w = 10
    print(
        f"  {'B':>{w}} | {'eigh (ms)':>{w}} | "
        f"{'svd (ms)':>{w}} | {'qr (ms)':>{w}}"
    )
    print("-" * (4 * w + 9))

    for B in B_list:
        # Symmetric matrix for eigh
        x = torch.randn(B, n, n, device=device)
        a_sym = (x + x.transpose(-1, -2)) / 2.0
        a_gen = torch.randn(B, n, n, device=device)

        t_eigh = time_op(torch.linalg.eigh, a_sym)
        t_svd = time_op(torch.linalg.svd, a_gen)
        t_qr = time_op(torch.linalg.qr, a_gen)

        eigh_times.append(t_eigh)
        svd_times.append(t_svd)
        qr_times.append(t_qr)

        print(
            f"  {B:{w}d} | {t_eigh:{w}.3f} | "
            f"{t_svd:{w}.3f} | {t_qr:{w}.3f}"
        )

    results[f"n={n}"] = {
        "eigh_ms": eigh_times,
        "svd_ms": svd_times,
        "qr_ms": qr_times,
    }

# ==========================================
# Save data
# ==========================================
data = {
    "B_list": B_list,
    "n_sizes": n_sizes,
    "results": results,
}
npy_path = os.path.join(DATA_DIR, "linalg_batch_timing.npy")
np.save(npy_path, data)
print(f"\nData saved to {npy_path}")
