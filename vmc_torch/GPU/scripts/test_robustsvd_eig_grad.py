"""Gradcheck tests for RobustSVD_EIG backward correctness.

Compares analytical gradients (from the custom backward) against
numerical finite differences via torch.autograd.gradcheck.

Tests multiple matrix shapes, conditioning regimes, and loss functions
to isolate which part of the backward formula might be wrong.

Usage:
    python scripts/test_robustsvd_eig_grad.py
"""
import torch
import sys
import os
# Allow running from GPU/scripts/, GPU/, or vmc_torch/ root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from vmc_torch.GPU.torch_utils import (
    RobustSVD_EIG,
    RobustSVD,
    svd_via_eigh,
    safe_inverse_random,
)


def make_well_conditioned(M, N, dtype=torch.float64, seed=None):
    """Random matrix with singular values ~O(1)."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(M, N, dtype=dtype)


def make_ill_conditioned(M, N, log_kappa=6, dtype=torch.float64, seed=42):
    """Matrix with condition number ~10^log_kappa."""
    torch.manual_seed(seed)
    K = min(M, N)
    U, _ = torch.linalg.qr(torch.randn(M, M, dtype=dtype))
    V, _ = torch.linalg.qr(torch.randn(N, N, dtype=dtype))
    S = torch.logspace(0, -log_kappa, steps=K, dtype=dtype)
    A = U[:, :K] @ torch.diag(S) @ V[:K, :]
    return A


def make_degenerate(M, N, dtype=torch.float64, seed=42):
    """Matrix with repeated singular values."""
    torch.manual_seed(seed)
    K = min(M, N)
    U, _ = torch.linalg.qr(torch.randn(M, M, dtype=dtype))
    V, _ = torch.linalg.qr(torch.randn(N, N, dtype=dtype))
    # Pairs of equal singular values
    S = torch.zeros(K, dtype=dtype)
    for i in range(K):
        S[i] = float(K // 2 - i // 2)  # e.g. 4,4,3,3,2,2,1,1
    S = S.clamp(min=0.1)
    A = U[:, :K] @ torch.diag(S) @ V[:K, :]
    return A


# ================================================================
#  Test 1: gradcheck on individual SVD outputs
# ================================================================

def test_gradcheck_S_only(shapes, jitter=1e-12, atol=1e-4, eps=1e-6):
    """Test grad through S only (simplest path in backward)."""
    print("\n=== Test: grad through S only (loss = S.sum()) ===")
    for M, N, label in shapes:
        A = make_well_conditioned(M, N, seed=42)
        A_gc = A.clone().requires_grad_(True)

        def fn(A):
            _, S, _ = RobustSVD_EIG.apply(A, jitter, None)
            return S.sum()

        try:
            ok = torch.autograd.gradcheck(fn, (A_gc,), eps=eps, atol=atol)
            print(f"  {label:<20} PASS")
        except Exception as e:
            print(f"  {label:<20} FAIL: {e}")


def test_gradcheck_U_only(shapes, jitter=1e-12, atol=1e-4, eps=1e-6):
    """Test grad through U only."""
    print("\n=== Test: grad through U only (loss = U.sum()) ===")
    for M, N, label in shapes:
        A = make_well_conditioned(M, N, seed=42)
        A_gc = A.clone().requires_grad_(True)

        def fn(A):
            U, _, _ = RobustSVD_EIG.apply(A, jitter, None)
            return U.sum()

        try:
            ok = torch.autograd.gradcheck(fn, (A_gc,), eps=eps, atol=atol)
            print(f"  {label:<20} PASS")
        except Exception as e:
            print(f"  {label:<20} FAIL: {e}")


def test_gradcheck_Vh_only(shapes, jitter=1e-12, atol=1e-4, eps=1e-6):
    """Test grad through Vh only."""
    print("\n=== Test: grad through Vh only (loss = Vh.sum()) ===")
    for M, N, label in shapes:
        A = make_well_conditioned(M, N, seed=42)
        A_gc = A.clone().requires_grad_(True)

        def fn(A):
            _, _, Vh = RobustSVD_EIG.apply(A, jitter, None)
            return Vh.sum()

        try:
            ok = torch.autograd.gradcheck(fn, (A_gc,), eps=eps, atol=atol)
            print(f"  {label:<20} PASS")
        except Exception as e:
            print(f"  {label:<20} FAIL: {e}")


def test_gradcheck_reconstruction(shapes, jitter=1e-12, atol=1e-4, eps=1e-6):
    """Test grad through full reconstruction U @ diag(S) @ Vh.

    This is the most relevant test: in TN contraction, SVD is used
    for truncation and the result feeds into further contractions.
    """
    print("\n=== Test: grad through reconstruction "
          "(loss = (U @ diag(S) @ Vh).sum()) ===")
    for M, N, label in shapes:
        A = make_well_conditioned(M, N, seed=42)
        A_gc = A.clone().requires_grad_(True)

        def fn(A):
            U, S, Vh = RobustSVD_EIG.apply(A, jitter, None)
            return (U @ torch.diag_embed(S) @ Vh).sum()

        try:
            ok = torch.autograd.gradcheck(fn, (A_gc,), eps=eps, atol=atol)
            print(f"  {label:<20} PASS")
        except Exception as e:
            print(f"  {label:<20} FAIL: {e}")


def test_gradcheck_truncated(shapes, jitter=1e-12, atol=1e-4, eps=1e-6):
    """Test grad through truncated SVD (keep top-k singular values).

    This mimics boundary contraction where max_bond truncates.
    """
    print("\n=== Test: grad through truncated SVD "
          "(loss = (U[:,:k] @ diag(S[:k]) @ Vh[:k,:]).sum()) ===")
    for M, N, label in shapes:
        A = make_well_conditioned(M, N, seed=42)
        A_gc = A.clone().requires_grad_(True)
        K = min(M, N)
        k = max(1, K // 2)

        def fn(A, k=k):
            U, S, Vh = RobustSVD_EIG.apply(A, jitter, None)
            return (U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]).sum()

        try:
            ok = torch.autograd.gradcheck(fn, (A_gc,), eps=eps, atol=atol)
            print(f"  {label:<20} (k={k}) PASS")
        except Exception as e:
            print(f"  {label:<20} (k={k}) FAIL: {e}")


# ================================================================
#  Test 2: Compare against RobustSVD (standard SVD) backward
# ================================================================

def test_grad_vs_standard_svd(shapes, jitter=1e-12):
    """Compare analytical grads of RobustSVD_EIG vs RobustSVD.

    Both should produce the same dA for the same loss, if
    forward outputs match and backward formula is correct.
    """
    print("\n=== Test: EIG grad vs standard SVD grad "
          "(loss = S.sum()) ===")
    for M, N, label in shapes:
        A = make_well_conditioned(M, N, seed=42)

        # --- Standard SVD grad ---
        A1 = A.clone().requires_grad_(True)
        U1, S1, Vh1 = RobustSVD.apply(A1, jitter, None)
        loss1 = S1.sum()
        loss1.backward()
        grad_std = A1.grad.clone()

        # --- EIG SVD grad ---
        A2 = A.clone().requires_grad_(True)
        U2, S2, Vh2 = RobustSVD_EIG.apply(A2, jitter, None)
        loss2 = S2.sum()
        loss2.backward()
        grad_eig = A2.grad.clone()

        # Compare
        diff = torch.norm(grad_eig - grad_std).item()
        rel = diff / (torch.norm(grad_std).item() + 1e-30)
        s_diff = torch.norm(S1 - S2).item()

        status = "PASS" if rel < 1e-4 else "FAIL"
        print(f"  {label:<20} |dA diff|={diff:.2e}  "
              f"rel={rel:.2e}  |S diff|={s_diff:.2e}  {status}")


def test_grad_vs_standard_reconstruction(shapes, jitter=1e-12):
    """Compare grads through reconstruction loss."""
    print("\n=== Test: EIG grad vs standard SVD grad "
          "(loss = (U@diag(S)@Vh).sum()) ===")
    for M, N, label in shapes:
        A = make_well_conditioned(M, N, seed=42)

        A1 = A.clone().requires_grad_(True)
        U1, S1, Vh1 = RobustSVD.apply(A1, jitter, None)
        loss1 = (U1 @ torch.diag_embed(S1) @ Vh1).sum()
        loss1.backward()
        grad_std = A1.grad.clone()

        A2 = A.clone().requires_grad_(True)
        U2, S2, Vh2 = RobustSVD_EIG.apply(A2, jitter, None)
        loss2 = (U2 @ torch.diag_embed(S2) @ Vh2).sum()
        loss2.backward()
        grad_eig = A2.grad.clone()

        diff = torch.norm(grad_eig - grad_std).item()
        rel = diff / (torch.norm(grad_std).item() + 1e-30)
        status = "PASS" if rel < 1e-4 else "FAIL"
        print(f"  {label:<20} |dA diff|={diff:.2e}  "
              f"rel={rel:.2e}  {status}")


# ================================================================
#  Test 3: Conditioning sensitivity
# ================================================================

def test_conditioning(jitter=1e-12, atol=1e-4, eps=1e-6):
    """Test gradcheck across conditioning regimes."""
    print("\n=== Test: gradcheck across conditioning "
          "(8x4, loss=S.sum()) ===")
    M, N = 8, 4

    for log_kappa, label in [
        (0, "kappa=1"),
        (3, "kappa=1e3"),
        (6, "kappa=1e6"),
        (9, "kappa=1e9"),
        (12, "kappa=1e12"),
    ]:
        A = make_ill_conditioned(M, N, log_kappa=log_kappa)
        A_gc = A.clone().requires_grad_(True)

        def fn(A):
            _, S, _ = RobustSVD_EIG.apply(A, jitter, None)
            return S.sum()

        try:
            ok = torch.autograd.gradcheck(fn, (A_gc,), eps=eps, atol=atol)
            print(f"  {label:<20} PASS")
        except Exception as e:
            print(f"  {label:<20} FAIL: {e}")

    print("\n--- Same test with loss = reconstruction ---")
    for log_kappa, label in [
        (0, "kappa=1"),
        (3, "kappa=1e3"),
        (6, "kappa=1e6"),
        (9, "kappa=1e9"),
    ]:
        A = make_ill_conditioned(M, N, log_kappa=log_kappa)
        A_gc = A.clone().requires_grad_(True)

        def fn(A):
            U, S, Vh = RobustSVD_EIG.apply(A, jitter, None)
            return (U @ torch.diag_embed(S) @ Vh).sum()

        try:
            ok = torch.autograd.gradcheck(fn, (A_gc,), eps=eps, atol=atol)
            print(f"  {label:<20} PASS")
        except Exception as e:
            print(f"  {label:<20} FAIL: {e}")


def test_degenerate_singular_values(jitter=1e-12, atol=1e-4, eps=1e-6):
    """Test with repeated singular values (stress test for F matrix)."""
    print("\n=== Test: degenerate singular values (8x4) ===")
    M, N = 8, 4
    A = make_degenerate(M, N)

    for loss_name, fn_factory in [
        ("S.sum()", lambda: lambda A: RobustSVD_EIG.apply(
            A, jitter, None)[1].sum()),
        ("recon.sum()", lambda: lambda A: (
            lambda r: r[0] @ torch.diag_embed(r[1]) @ r[2])(
            RobustSVD_EIG.apply(A, jitter, None)).sum()),
    ]:
        A_gc = A.clone().requires_grad_(True)
        fn = fn_factory()
        try:
            ok = torch.autograd.gradcheck(fn, (A_gc,), eps=eps, atol=atol)
            print(f"  {loss_name:<30} PASS")
        except Exception as e:
            print(f"  {loss_name:<30} FAIL: {e}")


# ================================================================
#  Test 4: Forward correctness of svd_via_eigh
# ================================================================

def test_forward_correctness(shapes):
    """Verify svd_via_eigh forward matches torch.linalg.svd."""
    print("\n=== Test: svd_via_eigh forward vs torch.linalg.svd ===")
    for M, N, label in shapes:
        A = make_well_conditioned(M, N, seed=42)

        U_ref, S_ref, Vh_ref = torch.linalg.svd(A, full_matrices=False)
        U_eig, S_eig, Vh_eig = svd_via_eigh(A)

        # S should match
        s_diff = torch.norm(S_eig - S_ref).item()

        # Reconstruction should match
        recon_ref = torch.norm(
            A - U_ref @ torch.diag(S_ref) @ Vh_ref
        ).item()
        recon_eig = torch.norm(
            A - U_eig @ torch.diag(S_eig) @ Vh_eig
        ).item()

        # Orthogonality of U and Vh
        K = min(M, N)
        eye_K = torch.eye(K, dtype=torch.float64)
        orth_U = torch.norm(U_eig.mT @ U_eig - eye_K).item()
        orth_Vh = torch.norm(Vh_eig @ Vh_eig.mT - eye_K).item()

        print(f"  {label:<20} |S diff|={s_diff:.2e}  "
              f"recon_eig={recon_eig:.2e}  recon_ref={recon_ref:.2e}  "
              f"orth_U={orth_U:.2e}  orth_Vh={orth_Vh:.2e}")


# ================================================================
#  Test 5: Gradient magnitude comparison
# ================================================================

def test_grad_magnitude(jitter=1e-12):
    """Check if EIG grads are abnormally large vs standard SVD grads.

    This directly tests the grad explosion hypothesis.
    """
    print("\n=== Test: gradient magnitude comparison ===")
    shapes = [
        (8, 4, "8x4 well-cond"),
        (8, 8, "8x8 well-cond"),
        (4, 8, "4x8 well-cond"),
        (16, 8, "16x8 well-cond"),
    ]

    for M, N, label in shapes:
        A = make_well_conditioned(M, N, seed=42)

        # Test multiple loss functions
        for loss_name, loss_fn in [
            ("S.sum", lambda U, S, Vh: S.sum()),
            ("U.sum", lambda U, S, Vh: U.sum()),
            ("Vh.sum", lambda U, S, Vh: Vh.sum()),
            ("recon", lambda U, S, Vh: (
                U @ torch.diag_embed(S) @ Vh).sum()),
        ]:
            # Standard SVD
            A1 = A.clone().requires_grad_(True)
            U1, S1, Vh1 = RobustSVD.apply(A1, jitter, None)
            loss1 = loss_fn(U1, S1, Vh1)
            loss1.backward()
            g_std = A1.grad.norm().item()

            # EIG SVD
            A2 = A.clone().requires_grad_(True)
            U2, S2, Vh2 = RobustSVD_EIG.apply(A2, jitter, None)
            loss2 = loss_fn(U2, S2, Vh2)
            loss2.backward()
            g_eig = A2.grad.norm().item()

            ratio = g_eig / (g_std + 1e-30)
            flag = " <<<" if ratio > 10 else ""
            print(f"  {label:<20} {loss_name:<8}  "
                  f"|grad_std|={g_std:.4e}  |grad_eig|={g_eig:.4e}  "
                  f"ratio={ratio:.2f}{flag}")


# ================================================================
#  Test 6: nonuniform_diag=True tests
# ================================================================

def test_nonuniform_diag_gradcheck(shapes, jitter=1e-6, atol=1e-4, eps=1e-6):
    """Gradcheck with nonuniform_diag=True on well-conditioned matrices."""
    print("\n=== Test: nonuniform_diag=True gradcheck "
          "(well-conditioned) ===")
    for M, N, label in shapes:
        A = make_well_conditioned(M, N, seed=42)
        for loss_name, fn in [
            ("S.sum", lambda A: RobustSVD_EIG.apply(
                A, jitter, None, True)[1].sum()),
            ("recon", lambda A: (
                lambda r: r[0] @ torch.diag_embed(r[1]) @ r[2])(
                RobustSVD_EIG.apply(A, jitter, None, True)).sum()),
        ]:
            A_gc = A.clone().requires_grad_(True)
            try:
                ok = torch.autograd.gradcheck(
                    fn, (A_gc,), eps=eps, atol=atol,
                    nondet_tol=1e-3,  # random jitter is non-deterministic
                )
                print(f"  {label:<20} {loss_name:<8} PASS")
            except Exception as e:
                err_msg = str(e).split('\n')[0]
                print(f"  {label:<20} {loss_name:<8} FAIL: {err_msg}")


def test_nonuniform_diag_degenerate(jitter=1e-6, atol=1e-4, eps=1e-6):
    """Test nonuniform_diag=True on degenerate matrices.

    This is the key test: random diagonal jitter should lift
    degeneracies and fix the backward.
    """
    print("\n=== Test: nonuniform_diag=True on DEGENERATE matrices ===")
    M, N = 8, 4
    A = make_degenerate(M, N)

    for loss_name, fn in [
        ("S.sum", lambda A: RobustSVD_EIG.apply(
            A, jitter, None, True)[1].sum()),
        ("recon", lambda A: (
            lambda r: r[0] @ torch.diag_embed(r[1]) @ r[2])(
            RobustSVD_EIG.apply(A, jitter, None, True)).sum()),
        ("U.sum", lambda A: RobustSVD_EIG.apply(
            A, jitter, None, True)[0].sum()),
        ("Vh.sum", lambda A: RobustSVD_EIG.apply(
            A, jitter, None, True)[2].sum()),
    ]:
        A_gc = A.clone().requires_grad_(True)
        try:
            ok = torch.autograd.gradcheck(
                fn, (A_gc,), eps=eps, atol=atol,
                nondet_tol=1e-3,
            )
            print(f"  {loss_name:<30} PASS")
        except Exception as e:
            err_msg = str(e).split('\n')[0]
            print(f"  {loss_name:<30} FAIL: {err_msg}")


def test_nonuniform_diag_ill_conditioned(jitter=1e-6, atol=1e-4, eps=1e-6):
    """Test nonuniform_diag=True across conditioning regimes."""
    print("\n=== Test: nonuniform_diag=True across conditioning "
          "(8x4, loss=S.sum()) ===")
    M, N = 8, 4

    for log_kappa, label in [
        (0, "kappa=1"),
        (3, "kappa=1e3"),
        (6, "kappa=1e6"),
        (9, "kappa=1e9"),
    ]:
        A = make_ill_conditioned(M, N, log_kappa=log_kappa)
        A_gc = A.clone().requires_grad_(True)

        def fn(A):
            _, S, _ = RobustSVD_EIG.apply(A, jitter, None, True)
            return S.sum()

        try:
            ok = torch.autograd.gradcheck(
                fn, (A_gc,), eps=eps, atol=atol,
                nondet_tol=1e-3,
            )
            print(f"  {label:<20} PASS")
        except Exception as e:
            err_msg = str(e).split('\n')[0]
            print(f"  {label:<20} FAIL: {err_msg}")


def test_nonuniform_diag_grad_magnitude(jitter=1e-6):
    """Compare grad magnitudes: nonuniform_diag=True vs False vs std SVD."""
    print("\n=== Test: grad magnitude with nonuniform_diag ===")
    shapes = [
        (8, 4, "8x4 well-cond"),
        (8, 8, "8x8 well-cond"),
    ]

    for M, N, label in shapes:
        A = make_well_conditioned(M, N, seed=42)

        for loss_name, loss_fn in [
            ("S.sum", lambda U, S, Vh: S.sum()),
            ("recon", lambda U, S, Vh: (
                U @ torch.diag_embed(S) @ Vh).sum()),
        ]:
            # Standard SVD
            A1 = A.clone().requires_grad_(True)
            U1, S1, Vh1 = RobustSVD.apply(A1, jitter, None)
            loss_fn(U1, S1, Vh1).backward()
            g_std = A1.grad.norm().item()

            # EIG identity jitter
            A2 = A.clone().requires_grad_(True)
            U2, S2, Vh2 = RobustSVD_EIG.apply(A2, jitter, None, False)
            loss_fn(U2, S2, Vh2).backward()
            g_id = A2.grad.norm().item()

            # EIG random diagonal jitter
            A3 = A.clone().requires_grad_(True)
            U3, S3, Vh3 = RobustSVD_EIG.apply(A3, jitter, None, True)
            loss_fn(U3, S3, Vh3).backward()
            g_rand = A3.grad.norm().item()

            print(f"  {label:<16} {loss_name:<8}  "
                  f"std={g_std:.4e}  eig_id={g_id:.4e}  "
                  f"eig_rand={g_rand:.4e}")

    # Also test degenerate case
    print("  --- degenerate 8x4 ---")
    A = make_degenerate(8, 4)
    for loss_name, loss_fn in [
        ("S.sum", lambda U, S, Vh: S.sum()),
        ("recon", lambda U, S, Vh: (
            U @ torch.diag_embed(S) @ Vh).sum()),
    ]:
        A1 = A.clone().requires_grad_(True)
        U1, S1, Vh1 = RobustSVD.apply(A1, jitter, None)
        loss_fn(U1, S1, Vh1).backward()
        g_std = A1.grad.norm().item()

        A2 = A.clone().requires_grad_(True)
        U2, S2, Vh2 = RobustSVD_EIG.apply(A2, jitter, None, False)
        loss_fn(U2, S2, Vh2).backward()
        g_id = A2.grad.norm().item()

        A3 = A.clone().requires_grad_(True)
        U3, S3, Vh3 = RobustSVD_EIG.apply(A3, jitter, None, True)
        loss_fn(U3, S3, Vh3).backward()
        g_rand = A3.grad.norm().item()

        print(f"  {'degen 8x4':<16} {loss_name:<8}  "
              f"std={g_std:.4e}  eig_id={g_id:.4e}  "
              f"eig_rand={g_rand:.4e}")


# ================================================================
#  Main
# ================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    shapes = [
        (6, 4, "tall 6x4"),
        (4, 4, "square 4x4"),
        (4, 6, "wide 4x6"),
        (8, 4, "tall 8x4"),
        (8, 8, "square 8x8"),
        (4, 8, "wide 4x8"),
    ]

    # Forward correctness
    test_forward_correctness(shapes)

    # Gradcheck: individual outputs (identity jitter)
    test_gradcheck_S_only(shapes)
    test_gradcheck_U_only(shapes)
    test_gradcheck_Vh_only(shapes)

    # Gradcheck: composite losses (identity jitter)
    test_gradcheck_reconstruction(shapes)
    test_gradcheck_truncated(shapes)

    # Compare against standard SVD backward
    test_grad_vs_standard_svd(shapes)
    test_grad_vs_standard_reconstruction(shapes)

    # Gradient magnitude check
    test_grad_magnitude()

    # Conditioning sensitivity (identity jitter)
    test_conditioning()
    test_degenerate_singular_values()

    # ---- nonuniform_diag=True tests ----
    print("\n" + "=" * 70)
    print("  RANDOM DIAGONAL JITTER TESTS")
    print("=" * 70)

    test_nonuniform_diag_gradcheck(shapes)
    test_nonuniform_diag_degenerate()
    test_nonuniform_diag_ill_conditioned()
    test_nonuniform_diag_grad_magnitude()
