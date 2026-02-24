"""
Benchmark torch.linalg.eigh vs custom_batched_eigh (my_eigh.cpp).

The C++ extension calls cusolverDnXsyevBatched directly for all
matrix sizes, bypassing PyTorch's n<=32 dispatch gate.

Usage:
    CUDA_HOME=/usr/local/cuda-12.6 python GPU/scripts/eigh_dispatch/test_eigh_cpp.py
    CUDA_HOME=/usr/local/cuda-12.6 python GPU/scripts/eigh_dispatch/test_eigh_cpp.py --save
    python GPU/scripts/eigh_dispatch/test_eigh_cpp.py --stock-only
"""

import argparse
import os
import torch
import numpy as np

DEVICE = "cuda"
DTYPE = torch.float64
N_LIST = [8, 16, 24, 28, 32, 33, 34, 40, 48, 64, 96, 128, 512, 520, 1024]
B_LIST = [64, 256]
N_WARMUP = 1
N_REPS = 1


def load_ext():
    """Load my_eigh.cpp via torch.utils.cpp_extension.load."""
    from torch.utils.cpp_extension import load
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-12.6")
    ext = load(
        name="my_eigh",
        sources=[os.path.join(script_dir, "my_eigh.cpp")],
        extra_cflags=["-O3"],
        extra_include_paths=[f"{cuda_home}/include"],
        extra_ldflags=[
            f"-L{cuda_home}/lib64",
            "-lcusolver", "-lcudart",
        ],
        verbose=True,
    )
    return ext


def bench(fn, a, n_warmup=N_WARMUP, n_reps=N_REPS):
    """Time fn(a) using CUDA events."""
    for _ in range(n_warmup):
        fn(a)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_reps):
        fn(a)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_reps


def make_sym(B, n, dtype=DTYPE, device=DEVICE):
    x = torch.randn(B, n, n, device=device, dtype=dtype)
    return (x + x.mT) / 2


def verify_correctness(ext, n=48, B=16):
    """Check custom eigh gives same results as torch.linalg.eigh."""
    a = make_sym(B, n)

    vals_ref, vecs_ref = torch.linalg.eigh(a)
    vals_ext, vecs_ext = ext.eigh(a, False)

    # Eigenvalues should match closely
    val_err = (vals_ref - vals_ext).abs().max().item()

    # Eigenvectors can differ by sign per column.
    # Check via residual: A @ v - lambda * v should be ~0.
    res_ref = (a @ vecs_ref - vecs_ref * vals_ref.unsqueeze(-2)).norm()
    res_ext = (a @ vecs_ext - vecs_ext * vals_ext.unsqueeze(-2)).norm()

    print(f"Correctness check (B={B}, n={n}):")
    print(f"  eigenvalue max |diff|: {val_err:.2e}")
    print(f"  residual ||Av - lv|| (stock):  {res_ref:.2e}")
    print(f"  residual ||Av - lv|| (custom): {res_ext:.2e}")
    ok = val_err < 1e-10 and res_ext < 1e-10
    print(f"  {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--stock-only", action="store_true",
                        help="Only benchmark stock torch.linalg.eigh")
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"dtype: {DTYPE}, warmup: {N_WARMUP}, reps: {N_REPS}")
    print(f"linalg library: "
          f"{torch.backends.cuda.preferred_linalg_library()}")
    print()

    ext = None
    if not args.stock_only:
        print("Compiling my_eigh.cpp extension...")
        try:
            ext = load_ext()
            print("Extension loaded.\n")
            verify_correctness(ext)
        except Exception as e:
            print(f"Extension failed to compile: {e}")
            print("Falling back to stock-only mode.\n")
            ext = None

    results = {}

    for B in B_LIST:
        print("=" * 60)
        print(f"  B = {B}")
        print("=" * 60)

        if ext:
            print(f"{'n':>6}  {'stock (ms)':>12}  "
                  f"{'custom (ms)':>12}  {'speedup':>8}  ")
            print(f"{'-'*6}  {'-'*12}  "
                  f"{'-'*12}  {'-'*8}")
        else:
            print(f"{'n':>6}  {'eigh (ms)':>12} ")
            print(f"{'-'*6}  {'-'*12}")

        for n in N_LIST:
            a = make_sym(B, n)

            t_stock = bench(torch.linalg.eigh, a)
            results[("stock", B, n)] = t_stock


            if ext:
                def fn_ext(x):
                    return ext.eigh(x, False)
                t_ext = bench(fn_ext, a)
                results[("custom", B, n)] = t_ext
                speedup = t_stock / t_ext
                print(f"{n:>6}  {t_stock:>12.2f}  "
                      f"{t_ext:>12.2f}  {speedup:>7.1f}x  "
                      )
            else:
                print(f"{n:>6}  {t_stock:>12.2f}  ")
        print()

    if args.save:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(
            script_dir, "..", "..", "data", "eigh_dispatch"
        )
        os.makedirs(data_dir, exist_ok=True)
        out = os.path.join(data_dir, "eigh_dispatch_timing.npy")
        np.save(out, results)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
