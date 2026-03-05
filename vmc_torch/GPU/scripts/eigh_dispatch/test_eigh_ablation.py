"""
Ablation test for eigh dispatch branching.

Compares stock torch.linalg.eigh vs cusolverDnXsyevBatched (my_eigh.cpp)
across all dispatch branches: B=1 (non-batched fallback), B=64/1024
(batched), float32/float64, and matrix sizes spanning all boundary
crossings (n=32 syevjBatched threshold, n=512 syevj/syevd boundary).

Key question: For B=1, is XsyevBatched slower than single-matrix solvers?
If yes, the proposed PyTorch patch's non-batched fallback is justified.

Usage:
    cd experiment/vmap
    CUDA_HOME=/usr/local/cuda-12.6 python \
        GPU/scripts/eigh_dispatch/test_eigh_ablation.py
    CUDA_HOME=/usr/local/cuda-12.6 python \
        GPU/scripts/eigh_dispatch/test_eigh_ablation.py --save
"""

import argparse
import os
import torch
import numpy as np

DEVICE = "cuda"
N_LIST = [8, 16, 32, 33, 48, 64, 128, 256, 512, 513]
B_LIST = [1, 64, 1024]
DTYPE_LIST = [torch.float32, torch.float64]
N_WARMUP = 5
N_REPS = 10

# Cap batch size for large n to avoid GPU OOM.
# B=1024, n=512, float64 = 2.1 GB per matrix batch — too much when
# stock eigh allocates per-matrix workspace on top of that.
MAX_B_FOR_N = {256: 64, 512: 64, 513: 64}


def effective_B_list(n):
    """Return B values that fit in GPU memory for matrix size n."""
    max_b = MAX_B_FOR_N.get(n, max(B_LIST))
    return [B for B in B_LIST if B <= max_b]


def load_ext():
    """Load my_eigh.cpp (cusolverDnXsyevBatched) extension."""
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
            "-lcusolver",
            "-lcudart",
        ],
        verbose=True,
    )
    return ext


def bench(fn, a, n_warmup=N_WARMUP, n_reps=N_REPS):
    """Time fn(a) using CUDA events. Returns (mean_ms, std_ms)."""
    for _ in range(n_warmup):
        fn(a)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(a)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def make_sym(B, n, dtype, device=DEVICE):
    """Create batch of random symmetric matrices."""
    x = torch.randn(B, n, n, device=device, dtype=dtype)
    return (x + x.mT) / 2


# ── Correctness ──────────────────────────────────────────────────────


def run_correctness(ext):
    """Verify eigenvalue agreement and Av=lv residual for all combos."""
    print("=" * 70)
    print("  CORRECTNESS SWEEP")
    print("=" * 70)
    all_ok = True
    for dtype in DTYPE_LIST:
        # Eigenvalue disagreement between stock (syevj/syevd) and
        # XsyevBatched grows with n for float32 — this is expected
        # since they use different algorithms.  Residual ||Av-lv||
        # is the true accuracy check; the val_err tolerance is loose.
        tol_val = 1e-2 if dtype == torch.float32 else 1e-8
        tol_res = 1e-4 if dtype == torch.float32 else 1e-8
        dname = "f32" if dtype == torch.float32 else "f64"
        for n in N_LIST:
            for B in effective_B_list(n):
                a = make_sym(B, n, dtype)
                vals_s, vecs_s = torch.linalg.eigh(a)
                vals_x, vecs_x = ext.eigh(a, False)

                val_err = (vals_s - vals_x).abs().max().item()
                # Av = lv residual (normalized per element)
                res_s = (
                    a @ vecs_s - vecs_s * vals_s.unsqueeze(-2)
                ).norm() / (B * n)
                res_x = (
                    a @ vecs_x - vecs_x * vals_x.unsqueeze(-2)
                ).norm() / (B * n)

                ok = val_err < tol_val and res_x < tol_res
                if not ok:
                    print(
                        f"  FAIL {dname} B={B:>4} n={n:>3}: "
                        f"val_err={val_err:.2e} "
                        f"res_stock={res_s:.2e} "
                        f"res_xsyev={res_x:.2e}"
                    )
                    all_ok = False

                del a, vals_s, vecs_s, vals_x, vecs_x
                torch.cuda.empty_cache()

    if all_ok:
        print("  All correctness checks PASSED")
    print()
    return all_ok


# ── Timing ───────────────────────────────────────────────────────────


def run_timing(ext):
    """Time stock vs XsyevBatched for all (B, n, dtype) combos."""
    print("=" * 70)
    print("  TIMING SWEEP")
    print("=" * 70)

    results = {}

    for dtype in DTYPE_LIST:
        dname = "f32" if dtype == torch.float32 else "f64"
        print(f"\n--- dtype = {dname} ---\n")

        for B in B_LIST:
            n_list_B = [n for n in N_LIST if B <= MAX_B_FOR_N.get(n, max(B_LIST))]
            if not n_list_B:
                continue

            print(f"  B = {B}")
            print(
                f"  {'n':>5}  {'stock (ms)':>14}  "
                f"{'xsyev (ms)':>14}  {'speedup':>8}  "
                f"{'winner':>8}"
            )
            print(
                f"  {'-'*5}  {'-'*14}  "
                f"{'-'*14}  {'-'*8}  {'-'*8}"
            )

            for n in n_list_B:
                a = make_sym(B, n, dtype)

                t_s, s_s = bench(torch.linalg.eigh, a)

                def fn_x(x):
                    return ext.eigh(x, False)

                t_x, s_x = bench(fn_x, a)

                spd = t_s / t_x if t_x > 0 else float("inf")
                winner = "xsyev" if spd > 1.0 else "stock"

                print(
                    f"  {n:>5}  "
                    f"{t_s:>8.2f}+/-{s_s:>4.2f}  "
                    f"{t_x:>8.2f}+/-{s_x:>4.2f}  "
                    f"{spd:>7.1f}x  "
                    f"{winner:>8}"
                )

                results[(dname, B, n, "stock")] = (t_s, s_s)
                results[(dname, B, n, "xsyev")] = (t_x, s_x)

                del a
                torch.cuda.empty_cache()

            print()

    return results


# ── Analysis ─────────────────────────────────────────────────────────


def run_analysis(results):
    """Print focused summaries answering the key questions."""
    print("=" * 70)
    print("  ANALYSIS")
    print("=" * 70)

    # 1. B=1 analysis: non-batched fallback
    print("\n--- B=1: Is XsyevBatched slower for single matrices? ---\n")
    print(
        f"  {'dtype':>5}  {'n':>5}  {'stock':>10}  "
        f"{'xsyev':>10}  {'speedup':>8}  {'verdict':>10}"
    )
    for dname in ["f32", "f64"]:
        for n in N_LIST:
            key_s = (dname, 1, n, "stock")
            key_x = (dname, 1, n, "xsyev")
            if key_s not in results:
                continue
            t_s = results[key_s][0]
            t_x = results[key_x][0]
            spd = t_s / t_x
            # "fallback justified" if stock wins (spd < 1)
            verdict = (
                "stock wins"
                if spd < 0.95
                else ("~tie" if spd < 1.05 else "xsyev wins")
            )
            print(
                f"  {dname:>5}  {n:>5}  {t_s:>8.2f}ms  "
                f"{t_x:>8.2f}ms  {spd:>7.2f}x  {verdict:>10}"
            )

    # 2. Boundary analysis: n=32/33 cliff
    print("\n--- n=32 vs n=33 cliff (stock eigh) ---\n")
    for dname in ["f32", "f64"]:
        for B in B_LIST:
            key_32 = (dname, B, 32, "stock")
            key_33 = (dname, B, 33, "stock")
            if key_32 not in results or key_33 not in results:
                continue
            t32 = results[key_32][0]
            t33 = results[key_33][0]
            ratio = t33 / t32
            print(
                f"  {dname} B={B:>4}: "
                f"n=32 {t32:.2f}ms, n=33 {t33:.2f}ms, "
                f"ratio={ratio:.1f}x"
            )

    # 3. n=512/513 boundary (syevj range for f32)
    print("\n--- n=512 vs n=513 boundary ---\n")
    for dname in ["f32", "f64"]:
        for B in B_LIST:
            key_512 = (dname, B, 512, "stock")
            key_513 = (dname, B, 513, "stock")
            if key_512 not in results or key_513 not in results:
                continue
            t512 = results[key_512][0]
            t513 = results[key_513][0]
            ratio = t513 / t512
            print(
                f"  {dname} B={B:>4}: "
                f"n=512 {t512:.2f}ms, n=513 {t513:.2f}ms, "
                f"ratio={ratio:.1f}x"
            )

    # 4. Float32 vs float64 comparison
    print(
        "\n--- float32 vs float64: XsyevBatched speedup comparison ---\n"
    )
    print(
        f"  {'B':>5}  {'n':>5}  "
        f"{'f32 speedup':>12}  {'f64 speedup':>12}"
    )
    for B in [64, 1024]:
        for n in N_LIST:
            key_f32_s = ("f32", B, n, "stock")
            key_f32_x = ("f32", B, n, "xsyev")
            key_f64_s = ("f64", B, n, "stock")
            key_f64_x = ("f64", B, n, "xsyev")
            if key_f32_s not in results:
                continue
            spd32 = results[key_f32_s][0] / results[key_f32_x][0]
            spd64 = results[key_f64_s][0] / results[key_f64_x][0]
            print(
                f"  {B:>5}  {n:>5}  "
                f"{spd32:>10.1f}x  {spd64:>10.1f}x"
            )

    print()


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Ablation test for eigh dispatch branching"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to GPU/data/eigh_dispatch/",
    )
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"linalg library: "
        f"{torch.backends.cuda.preferred_linalg_library()}"
    )
    print(f"B_LIST: {B_LIST}")
    print(f"N_LIST: {N_LIST}")
    print(f"DTYPE_LIST: {[str(d) for d in DTYPE_LIST]}")
    print(f"warmup: {N_WARMUP}, reps: {N_REPS}")
    print()

    print("Compiling my_eigh.cpp extension...")
    ext = load_ext()
    print("Extension loaded.\n")

    run_correctness(ext)
    results = run_timing(ext)
    run_analysis(results)

    if args.save:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(
            script_dir, "..", "..", "data", "eigh_dispatch"
        )
        os.makedirs(data_dir, exist_ok=True)
        out = os.path.join(data_dir, "eigh_ablation.npy")
        np.save(out, results)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
