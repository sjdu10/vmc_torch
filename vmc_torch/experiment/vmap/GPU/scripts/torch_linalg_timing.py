"""
Benchmark torch.linalg.{eigh, svd, qr} for batched small matrices on GPU.

Compares cuSOLVER (default) vs MAGMA backend across the n=32 Jacobi
threshold. For n<=32, cuSOLVER uses fused batched Jacobi kernels for
eigh/svd (~single kernel launch). QR has no such fused kernel at any n.

Usage:
    python GPU/scripts/torch_linalg_timing.py
"""

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time

import numpy as np
import torch

# ==========================================
# Config
# ==========================================
B = 64  # batch size
N_WARMUP = 3
N_ITERS = 5

# Dense sampling around n=32 threshold, sparser for larger n
n_list = list(range(4, 36)) + [40, 48, 56, 64, 80, 96, 128, 256]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise RuntimeError("This benchmark requires a CUDA GPU.")

# Detect custom-built PyTorch (dev version from source) vs pre-built.
# Custom builds have version strings like "2.12.0a0+git67c428c";
# pre-built releases are clean like "2.5.1" or "2.5.1+cu128".
_is_custom_torch = "git" in torch.__version__


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
def run_benchmark(backend_name):
    """Benchmark eigh/svd/qr for all n values. Returns dict of lists."""
    eigh_times, svd_times, qr_times = [], [], []

    w = 8
    print(
        f"  {'n':>{w}} | {'eigh (ms)':>{w}} | "
        f"{'svd (ms)':>{w}} | {'qr (ms)':>{w}}"
    )
    print("-" * (4 * w + 9))

    for n in n_list:
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

        marker = " <-- n=32 threshold" if n == 32 else ""
        print(
            f"  {n:{w}d} | {t_eigh:{w}.3f} | "
            f"{t_svd:{w}.3f} | {t_qr:{w}.3f}{marker}"
        )

    return {
        "eigh_ms": eigh_times,
        "svd_ms": svd_times,
        "qr_ms": qr_times,
    }


# # --- Default backend (cuSOLVER) ---
# default_lib = torch.backends.cuda.preferred_linalg_library()
# print(f"\n{'='*60}")
# print(f"Backend: default ({default_lib}), B={B}")
# print(f"{'='*60}")
# res_default = run_benchmark("default")

# --- MAGMA backend ---
if _is_custom_torch:
    # Custom-built PyTorch (dev branch) deprecated MAGMA SVD/eigh but
    # didn't update svd_uses_cusolver(), causing stride mismatches for
    # n>32. Skip MAGMA backend to avoid CUSOLVER_STATUS_INVALID_VALUE.
    print(f"\n{'='*60}")
    print(f"Skipping MAGMA backend: custom-built torch {torch.__version__}")
    print("MAGMA SVD is deprecated in this build (routes to cuSOLVER "
          "with wrong strides for n>32).")
    print(f"{'='*60}")
    res_magma = None
else:
    torch.backends.cuda.preferred_linalg_library("magma")
    print(f"\n{'='*60}")
    print(f"Backend: MAGMA, B={B}")
    print(f"{'='*60}")
    res_magma = run_benchmark("magma")

# # Reset
# torch.backends.cuda.preferred_linalg_library(default_lib)

# # ==========================================
# # Comparison tables
# # ==========================================
# n_arr = np.array(n_list)
# ops = ["eigh", "svd", "qr"]

# print(f"\n{'='*70}")
# print("COMPARISON: cuSOLVER (default) vs MAGMA")
# print(f"B={B}, n_list={n_list[0]}..{n_list[-1]}")
# print(f"{'='*70}")

# for op in ops:
#     key = f"{op}_ms"
#     print(f"\n--- {op.upper()} ---")
#     w = 10
#     print(
#         f"  {'n':>4} | {'default':>{w}} | "
#         f"{'MAGMA':>{w}} | {'def/MAGMA':>{w}}"
#     )
#     print("  " + "-" * (4 + 3 * w + 9))
#     for i, n in enumerate(n_list):
#         t_d = res_default[key][i]
#         t_m = res_magma[key][i]
#         ratio = t_d / t_m if t_m > 0 else float("nan")
#         tag = ""
#         if ratio > 1.5:
#             tag = "  MAGMA wins"
#         elif ratio < 0.67:
#             tag = "  default wins"
#         print(
#             f"  {n:4d} | {t_d:{w}.3f} | "
#             f"{t_m:{w}.3f} | {ratio:{w}.2f}x{tag}"
#         )

# # ==========================================
# # Summary: best backend per operation per regime
# # ==========================================
# print(f"\n{'='*70}")
# print("SUMMARY: Best backend per regime")
# print(f"{'='*70}")

# for op in ops:
#     key = f"{op}_ms"
#     # n<=32 regime
#     idx_le32 = [i for i, n in enumerate(n_list) if n <= 32]
#     idx_gt32 = [i for i, n in enumerate(n_list) if n > 32]

#     for label, idx in [("n<=32", idx_le32), ("n>32", idx_gt32)]:
#         if not idx:
#             continue
#         avg_speedup = np.mean(
#             [res_default[key][i] / res_magma[key][i] for i in idx if res_magma[key][i] > 0]
#         )
#         winner = "default" if avg_speedup > 1 else "MAGMA"
#         ratio = avg_speedup if avg_speedup > 1 else 1 / avg_speedup
#         print(
#             f"  {op.upper():>4} {label:>5}: "
#             f"avg_speedup={avg_speedup:.2f}x → "
#             f"{winner} wins ({ratio:.1f}x)"
#         )

# # ==========================================
# # Save data
# # ==========================================
# data = {
#     "n_list": n_list,
#     "B": B,
#     "default": res_default,
#     "magma": res_magma,
# }
# npy_path = os.path.join(DATA_DIR, f"linalg_timing_B={B}.npy")
# np.save(npy_path, data)
# print(f"\nData saved to {npy_path}")
# print("Plot in GPU/notebooks/linalg_timing.ipynb")
