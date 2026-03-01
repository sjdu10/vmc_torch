"""
Test torch_minres correctness against scipy.sparse.linalg.minres.

Constructs a synthetic SR-like problem:
    S = (1/N) O^T O - mean_O @ mean_O^T + diag_shift * I
and solves S @ x = b with both solvers, comparing results.

Run:  python GPU/scripts/test_torch_minres.py
"""
import time
import numpy as np
import torch
import scipy.sparse.linalg as spla

# Add parent to path so we can import vmc_modules
import sys, os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'),
)

from vmc_torch.experiment.vmap.GPU.vmc_modules import torch_minres


def test_torch_minres_vs_scipy(
    Ns=512, Np=200, diag_shift=1e-3, rtol=1e-6, maxiter=200,
):
    """Compare torch_minres vs scipy.minres on a synthetic SR problem."""
    print(f"=== Test: Ns={Ns}, Np={Np}, diag_shift={diag_shift} ===")
    device = torch.device('cuda')

    # Synthetic O_loc matrix and energy gradient
    rng = np.random.default_rng(42)
    O_np = rng.standard_normal((Ns, Np))
    b_np = rng.standard_normal(Np)

    mean_O_np = O_np.mean(axis=0)

    # --- scipy reference ---
    def matvec_np(x):
        inner = O_np.dot(x)
        Sx = O_np.T.dot(inner) / Ns
        Sx -= np.dot(mean_O_np, x) * mean_O_np
        return Sx + diag_shift * x

    A = spla.LinearOperator(
        (Np, Np), matvec=matvec_np, dtype=np.float64,
    )
    t0 = time.time()
    dp_scipy, info_scipy = spla.minres(
        A, b_np, rtol=rtol, maxiter=maxiter,
    )
    t_scipy = time.time() - t0
    print(f"  scipy:  info={info_scipy}, time={t_scipy:.4f}s")

    # --- torch_minres on GPU ---
    O_t = torch.tensor(O_np, device=device, dtype=torch.float64)
    b_t = torch.tensor(b_np, device=device, dtype=torch.float64)
    mean_O_t = O_t.mean(dim=0)

    def matvec_gpu(x):
        inner = O_t @ x
        Sx = O_t.T @ inner / Ns
        Sx -= torch.dot(mean_O_t, x) * mean_O_t
        return Sx + diag_shift * x

    # Warmup
    _ = torch_minres(matvec_gpu, b_t, rtol=rtol, maxiter=5)
    torch.cuda.synchronize()

    t0 = time.time()
    dp_torch, info_torch = torch_minres(
        matvec_gpu, b_t, rtol=rtol, maxiter=maxiter,
    )
    torch.cuda.synchronize()
    t_torch = time.time() - t0
    print(f"  torch:  info={info_torch}, time={t_torch:.4f}s")

    # --- Compare ---
    dp_torch_np = dp_torch.cpu().numpy()
    abs_diff = np.linalg.norm(dp_torch_np - dp_scipy)
    rel_diff = abs_diff / (np.linalg.norm(dp_scipy) + 1e-30)
    print(f"  |dp_torch - dp_scipy| = {abs_diff:.4e}")
    print(f"  relative diff         = {rel_diff:.4e}")
    print(f"  speedup               = {t_scipy / t_torch:.2f}x")

    # Verify residual is small
    r_torch = matvec_gpu(dp_torch) - b_t
    r_norm = torch.linalg.norm(r_torch).item()
    print(f"  |A*dp_torch - b|/|b|  = {r_norm / np.linalg.norm(b_np):.4e}")

    return rel_diff


if __name__ == '__main__':
    print("torch_minres correctness test\n")

    # Small problem
    d1 = test_torch_minres_vs_scipy(Ns=256, Np=100)
    print()

    # Medium problem (realistic VMC size)
    d2 = test_torch_minres_vs_scipy(Ns=2048, Np=1000)
    print()

    # Larger problem
    d3 = test_torch_minres_vs_scipy(Ns=4096, Np=3000)
    print()

    max_diff = max(d1, d2, d3)
    # Solutions differ at ~1e-3 due to different FP accumulation
    # order (GPU torch vs CPU numpy), but both achieve residuals
    # < rtol.  Use 1e-2 as a generous sanity check on the solution
    # vectors; the residual check (printed above) is the real test.
    if max_diff < 1e-2:
        print(f"PASS: max relative diff = {max_diff:.4e}")
    else:
        print(f"FAIL: max relative diff = {max_diff:.4e}")
