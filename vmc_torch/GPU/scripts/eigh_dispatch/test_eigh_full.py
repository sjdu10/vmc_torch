"""
Benchmark torch.linalg.{eigh, svd, qr} for batched small matrices on GPU.

Compares cuSOLVER (default) vs MAGMA backend across the n=32 Jacobi
threshold. For n<=32, cuSOLVER uses fused batched Jacobi kernels for
eigh/svd (~single kernel launch). QR has no such fused kernel at any n.

Usage:
    python GPU/scripts/torch_linalg_timing.py
"""

import argparse
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time

import numpy as np
import torch

parser = argparse.ArgumentParser(
    description="Benchmark torch.linalg.eigh for batched matrices"
)
parser.add_argument("--B", type=int, default=32, help="Batch size")
parser.add_argument(
    "--dtype",
    type=str,
    default="float32",
    choices=["float32", "float64"],
    help="Data type",
)
args = parser.parse_args()

dtype_map = {"float32": torch.float32, "float64": torch.float64}
torch.set_default_dtype(dtype_map[args.dtype])
# ==========================================
# Config
# ==========================================
B = args.B
N_WARMUP = 3
N_ITERS = 5

# Dense sampling around n=32 threshold, sparser for larger n
n_list = [8,16,24,28,30,31,32,33,34,40, 48, 56, 64, 80, 96, 128, 256, 400, 512,513,600,1024]

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
    """Benchmark eigh for all n values. Returns dict of lists."""
    eigh_times = [] 

    w = 8
    print(
        f"  {'n':>{w}} | {'eigh (ms)':>{w}} | "
    )
    print("-" * (2 * w + 9))

    for n in n_list:
        # Symmetric matrix for eigh
        x = torch.randn(B, n, n, device=device)
        a_sym = (x + x.transpose(-1, -2)) / 2.0

        t_eigh = time_op(torch.linalg.eigh, a_sym)

        eigh_times.append(t_eigh)

        marker = " <-- n=32 threshold" if n == 32 else ""
        print(
            f"  {n:{w}d} | {t_eigh:{w}.3f} | {marker}"
        )

    return {
        "eigh_ms": eigh_times,
    }


# --- Default backend (cuSOLVER) ---
print(f"\n{'='*60}")
if _is_custom_torch:
    print(f"With the fix, B={B}, dtype={torch.get_default_dtype()}")
else:
    print(f"Without the fix, B={B}, dtype={torch.get_default_dtype()}")
print(f"{'='*60}")
res_default = run_benchmark("default")
