"""Numerical linear-algebra utilities for tensor-network VMC in PyTorch.

Three generations of differentiable matrix decompositions live here:

1. Legacy jittered SVD (``RobustSVD``, ``RobustSVD_random`` and their
   ``*_wrapper`` functions): cuSOLVER/LAPACK SVD of a jittered copy of
   A with a hand-written, Lorentzian-broadened backward.
2. Gram-matrix decompositions (``svd_via_eigh``, ``RobustSVD_EIG``,
   ``CholeskyQR``, ``qr_via_eigh``, ``size_aware_*``): SVD/QR obtained
   from eigh / cholesky of A^T A with the jitter added to the Gram
   matrix, so eigenvector orthogonality is preserved.
3. CUDA-graph-capturable path (``jacobi_eigh`` torch.library op,
   ``jacobi_eigh_diff``, ``svd_via_jacobi``, ``qr_via_jacobi``,
   ``install_capture_safe_linalg``): a fixed-sweep batched Jacobi
   eigensolver with no host syncs, differentiable by graph
   composition so it survives torch.export / vmap / cudagraphs.

Also: a pure-PyTorch MINRES (``torch_minres``), ``GraphedGradFn``
(manual CUDA-graph capture of a gradient callable) and two benchmark
drivers.
"""
from __future__ import annotations

import torch
import contextlib
import math
from typing import Any, Callable, Iterator, Optional, Sequence

# === Global control ===
_ENABLE_JITTER = False

@contextlib.contextmanager
def use_jitter_svd() -> Iterator[None]:
    """Set the module flag ``_ENABLE_JITTER`` to True inside a block.

    The flag is reset to False on exit (also on exceptions). Nothing
    inside this module reads the flag; it is only a switch for
    callers that import ``_ENABLE_JITTER`` themselves.
    """
    global _ENABLE_JITTER
    _ENABLE_JITTER = True
    try:
        yield
    finally:
        _ENABLE_JITTER = False

# === SVD Patch ===

def safe_inverse(x: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Lorentzian-broadened reciprocal x / (x^2 + epsilon).

    Equals 1/x for |x| >> sqrt(epsilon) and goes smoothly to 0 at
    x = 0 instead of diverging, which damps gradient flow through
    (near-)degenerate or vanishing singular values in the SVD/QR
    backwards of this module.

    Args:
        x: real tensor of any shape.
        epsilon: absolute broadening (not scaled by x); the default
            1e-12 is tuned for f64 singular values of O(1).

    Returns:
        Tensor of the same shape and dtype as ``x``.
    """
    return x / (x.pow(2) + epsilon)

class RobustSVD(torch.autograd.Function):
    """Reduced SVD of an identity-jittered matrix with a stable backward.

    Forward: A' = A + jitter_strength * ||A||_F * I (jitter on A itself,
    a relative shift of the leading diagonal), then cuSOLVER/LAPACK
    ``torch.linalg.svd`` of A' / max|A'| (rescaled back afterwards),
    then a canonical sign per singular vector pair (the max-abs entry
    of each column of U is made positive). Backward: the analytic
    SVD adjoint with all 1/(s_i +- s_j) and 1/s terms replaced by
    ``safe_inverse``. Uses the PyTorch 2 ``setup_context`` protocol
    and ``generate_vmap_rule`` so it composes with torch.func / vmap.
    Real inputs only (``.mT`` transposes, no conjugation).

    Call via ``RobustSVD.apply(A, jitter_strength, driver)`` or the
    ``robust_svd_wrapper`` helper.
    """

    # automatically generate vmap rules for forward func that contains only pure pytorch operations
    generate_vmap_rule = True

    @staticmethod
    def forward(
        A: torch.Tensor, jitter_strength: float, driver: Optional[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Jittered, rescaled and sign-fixed reduced SVD (pure aten ops).

        Args:
            A: (..., M, N) real matrix (batched).
            jitter_strength: relative diagonal shift; the actual shift
                is jitter_strength * ||A||_F per matrix.
            driver: cuSOLVER driver name for ``torch.linalg.svd``
                ('gesvd', 'gesvdj', 'gesvda') or None for the default;
                must be None on CPU.

        Returns:
            (U, S, Vh) with U (..., M, K), S (..., K) descending and
            Vh (..., K, N), K = min(M, N).
        """
        # --- 1. Jitter Logic (Same as before) ---
        # A: (Batch, M, N) or (M, N)
        scale = A.norm(dim=(-2,-1), keepdim=True)
        
        # Jitter
        relative_eps = jitter_strength
        
        M, N = A.shape[-2:]

        # A = A + \delta * I
        eye = torch.eye(M, N, device=A.device, dtype=A.dtype)
        effective_jitter = scale * relative_eps
        jitter_matrix = eye * effective_jitter
        A_new = A + jitter_matrix

        # # A = A + random_matrix * \delta
        # R = torch.randn_like(A)
        # R = R / torch.norm(R)
        # A_new = A + scale * relative_eps * R
        
        # --- 2. SVD Calculation ---
        scale_new = torch.amax(torch.abs(A_new), dim=(-2, -1), keepdim=True)
        scale_new = torch.where(scale_new < 1e-16, torch.ones_like(scale_new), scale_new)
        A_new_normalized = A_new / scale_new
        if driver is not None:
            U, S_norm, Vh = torch.linalg.svd(A_new_normalized, full_matrices=False, driver=driver)
        else:
            U, S_norm, Vh = torch.linalg.svd(A_new_normalized, full_matrices=False)
        S = S_norm * scale_new.squeeze(-1)
        
        # --- 3. Sign Fixing (Vectorized) ---
        max_abs_cols = torch.argmax(torch.abs(U), dim=-2, keepdim=True)
        gathered = torch.gather(U, -2, max_abs_cols)
        signs = torch.sign(gathered)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        
        U = U * signs          
        Vh = Vh * signs.mT
        return U, S, Vh

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[Any, ...], output: tuple[Any, ...],
    ) -> None:
        """Save (U, S, Vh) for the backward pass.

        Args:
            ctx: autograd context.
            inputs: forward inputs (A, jitter_strength, driver).
            output: forward outputs (U, S, Vh).
        """
        U, S, Vh = output
        ctx.save_for_backward(U, S, Vh)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dU: torch.Tensor, dS: torch.Tensor, dVh: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        """Broadened SVD adjoint (Wan & Zhang, arXiv:1903.09650).

        F[i,j] = 1/(s_j - s_i) and G[i,j] = 1/(s_j + s_i) (zero on the
        diagonal) and 1/s are all evaluated with ``safe_inverse`` at
        an absolute epsilon of 1e-12, so degenerate or zero singular
        values give finite (damped) gradients. Extra terms are added
        for the non-square cases M > K (from dU) and N > K (from dVh).

        Args:
            ctx: autograd context holding (U, S, Vh).
            dU: (..., M, K) cotangent of U.
            dS: (..., K) cotangent of S.
            dVh: (..., K, N) cotangent of Vh.

        Returns:
            (dA, None, None): dA is (..., M, N); no gradient for
            ``jitter_strength`` or ``driver``.
        """
        U, S, Vh = ctx.saved_tensors
        
        M = U.size(-2)
        N = Vh.size(-1)
        K = S.size(-1)
        eye_K = torch.eye(K, dtype=U.dtype, device=U.device)

        # Epsilon for safe inverse in backward
        epsilon = 1e-12

        F = S.unsqueeze(-2) - S.unsqueeze(-1)
        F = safe_inverse(F, epsilon=epsilon)
        F = F * (1 - eye_K) 

        G = S.unsqueeze(-2) + S.unsqueeze(-1)
        G = safe_inverse(G, epsilon=epsilon)
        G = G * (1 - eye_K)

        UdU = U.mT @ dU
        VdV = Vh @ dVh.mT

        Su = (F + G) * (UdU - UdU.mT) / 2
        Sv = (F - G) * (VdV - VdV.mT) / 2
        
        # # NaN Guard
        # Su = torch.nan_to_num(Su, nan=0.0, posinf=0.0, neginf=0.0)
        # Sv = torch.nan_to_num(Sv, nan=0.0, posinf=0.0, neginf=0.0)

        dA = U @ (Su + Sv + torch.diag_embed(dS)) @ Vh
        
        S_inv = safe_inverse(S, epsilon=epsilon)
        
        if M > K:
            term1 = (dU * S_inv.unsqueeze(-2)) @ Vh
            term2 = U @ (U.mT @ term1)
            delta = term1 - term2
            # delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            dA = dA + delta
            
        if N > K:
            term1 = (U * S_inv.unsqueeze(-2)) @ dVh
            term2 = term1 @ (Vh.mT @ Vh)
            delta = term1 - term2
            # delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            dA = dA + delta

        return dA, None, None


# ========================================== SVD with random jitter ==========================================
def safe_inverse_random(
    x: torch.Tensor, epsilon: float = 1e-12,
) -> torch.Tensor:
    """Lorentzian-broadened reciprocal x / (x^2 + epsilon).

    Identical to ``safe_inverse``; kept as a separate name because
    the random and eigh SVD backwards reference it.

    Args:
        x: real tensor of any shape.
        epsilon: absolute broadening.

    Returns:
        Tensor of the same shape and dtype as ``x``.
    """
    return x / (x**2 + epsilon)

class RobustSVD_random(torch.autograd.Function):
    """Reduced SVD of a randomly jittered matrix with a stable backward.

    Same as ``RobustSVD`` except that the perturbation is a random
    Gaussian matrix of unit Frobenius norm scaled by
    jitter_strength * ||A||_F, which breaks exact singular-value
    degeneracies that make cuSOLVER's gesvdj/gesvd fail to converge.
    The noise is drawn with ``torch.randn_like`` on every call, so
    results are not deterministic and the function cannot run under
    vmap with the default ``randomness='error'``. Real inputs only.
    """

    # Automatically generate vmap rules for pure PyTorch operations in forward
    generate_vmap_rule = True

    @staticmethod
    def forward(
        A: torch.Tensor, jitter_strength: float, driver: Optional[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly jittered, rescaled and sign-fixed reduced SVD.

        Args:
            A: (..., M, N) real matrix (batched).
            jitter_strength: relative noise amplitude; the added noise
                has Frobenius norm jitter_strength * ||A||_F.
            driver: cuSOLVER driver name for ``torch.linalg.svd`` or
                None for the default (must be None on CPU).

        Returns:
            (U, S, Vh) with U (..., M, K), S (..., K) descending and
            Vh (..., K, N), K = min(M, N).
        """
        # Calculate scale based on norm
        scale = torch.linalg.norm(A, dim=(-2, -1), keepdim=True)
        
        # --- Random Jitter Logic ---
        noise = torch.randn_like(A)
        
        # Normalize noise to unit norm
        noise_norm = torch.linalg.norm(noise, dim=(-2, -1), keepdim=True)
        noise = noise / (noise_norm + 1e-16)
        
        # Apply Random Jitter: A_new = A + jitter * Noise * Scale
        A_new = A + noise * (scale * jitter_strength)
        
        # --- SVD Calculation ---
        scale_new = torch.amax(torch.abs(A_new), dim=(-2, -1), keepdim=True)
        scale_new = torch.where(scale_new < 1e-16, torch.ones_like(scale_new), scale_new)
        A_new_normalized = A_new / scale_new
        if driver is not None:
            U, S_norm, Vh = torch.linalg.svd(A_new_normalized, full_matrices=False, driver=driver)
        else:
            U, S_norm, Vh = torch.linalg.svd(A_new_normalized, full_matrices=False)
        S = S_norm * scale_new.squeeze(-1)
            
        # --- Sign Fixing ---
        max_abs_cols = torch.argmax(torch.abs(U), dim=-2, keepdim=True)
        gathered = torch.gather(U, -2, max_abs_cols)
        signs = torch.sign(gathered)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        
        U = U * signs          
        Vh = Vh * signs.mT

        return U, S, Vh

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[Any, ...], output: tuple[Any, ...],
    ) -> None:
        """Save (U, S, Vh) for the backward pass.

        Args:
            ctx: autograd context.
            inputs: forward inputs (A, jitter_strength, driver).
            output: forward outputs (U, S, Vh).
        """
        U, S, Vh = output
        ctx.save_for_backward(U, S, Vh)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dU: torch.Tensor, dS: torch.Tensor, dVh: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        """Broadened SVD adjoint, as ``RobustSVD.backward``.

        Differs only in an additional ``nan_to_num`` guard on the
        skew-symmetric Su / Sv terms.

        Args:
            ctx: autograd context holding (U, S, Vh).
            dU: (..., M, K) cotangent of U.
            dS: (..., K) cotangent of S.
            dVh: (..., K, N) cotangent of Vh.

        Returns:
            (dA, None, None) with dA (..., M, N).
        """
        U, S, Vh = ctx.saved_tensors

        M = U.size(-2)
        N = Vh.size(-1)
        K = S.size(-1)
        eye_K = torch.eye(K, dtype=U.dtype, device=U.device)
        epsilon = 1e-12

        # Use safe_inverse_random
        F = S.unsqueeze(-2) - S.unsqueeze(-1)
        F = safe_inverse_random(F, epsilon=epsilon)
        F = F * (1 - eye_K) 

        G = S.unsqueeze(-2) + S.unsqueeze(-1)
        G = safe_inverse_random(G, epsilon=epsilon)
        G = G * (1 - eye_K)

        UdU = U.mT @ dU
        VdV = Vh @ dVh.mT

        Su = (F + G) * (UdU - UdU.mT) / 2
        Sv = (F - G) * (VdV - VdV.mT) / 2
        
        # NaN Guard
        Su = torch.nan_to_num(Su, nan=0.0, posinf=0.0, neginf=0.0)
        Sv = torch.nan_to_num(Sv, nan=0.0, posinf=0.0, neginf=0.0)

        dA = U @ (Su + Sv + torch.diag_embed(dS)) @ Vh
        
        # Handle non-square contributions
        S_inv = safe_inverse_random(S, epsilon=epsilon)
        
        if M > K:
            term1 = (dU * S_inv.unsqueeze(-2)) @ Vh
            term2 = U @ (U.mT @ term1)
            delta = term1 - term2
            dA = dA + delta
            
        if N > K:
            term1 = (U * S_inv.unsqueeze(-2)) @ dVh
            term2 = term1 @ (Vh.mT @ Vh)
            delta = term1 - term2
            dA = dA + delta

        return dA, None, None


def robust_svd_wrapper(
    A: torch.Tensor, jitter: float = 1e-12, driver: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable identity-jittered SVD (``RobustSVD.apply``).

    Args:
        A: (..., M, N) real matrix.
        jitter: relative diagonal shift jitter * ||A||_F added to A.
        driver: cuSOLVER SVD driver or None.

    Returns:
        (U, S, Vh) with U (..., M, K), S (..., K), Vh (..., K, N).
    """
    return RobustSVD.apply(A, jitter, driver)


# QR via SVD wrappers

def qr_svd_wrapper(
    A: torch.Tensor, jitter: float = 1e-12, driver: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD through a preliminary QR: A = Q R, R = U' S Vh, U = Q U'.

    The robust SVD only sees the small K x N triangular factor R,
    which improves conditioning for tall A. The QR itself is plain
    ``torch.linalg.qr`` (standard autograd through it).

    Args:
        A: (..., M, N) real matrix.
        jitter: relative diagonal shift for ``RobustSVD`` on R.
        driver: cuSOLVER SVD driver or None.

    Returns:
        (U, S, Vh) with U (..., M, K), S (..., K), Vh (..., K, N).
    """
    # 1. QR Decomposition
    Q, R = torch.linalg.qr(A, mode='reduced')
    
    # 2. 对 R 做 Robust SVD
    U_prime, S, Vh = RobustSVD.apply(R, jitter, driver)
    
    # 3. 还原 U
    # 将正交基 Q 作用在 U' 上，得到原始矩阵 A 的左奇异向量
    U = Q @ U_prime
    
    return U, S, Vh


def robust_svd_wrapper_random(
    A: torch.Tensor, jitter: float = 1e-12, driver: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable randomly jittered SVD (``RobustSVD_random``).

    Args:
        A: (..., M, N) real matrix.
        jitter: relative noise amplitude jitter * ||A||_F.
        driver: cuSOLVER SVD driver or None.

    Returns:
        (U, S, Vh) with U (..., M, K), S (..., K), Vh (..., K, N).
    """
    return RobustSVD_random.apply(A, jitter, driver)

def qr_svd_wrapper_random(
    A: torch.Tensor, jitter: float = 1e-12, driver: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QR-then-SVD as ``qr_svd_wrapper`` but with random jitter on R.

    Args:
        A: (..., M, N) real matrix.
        jitter: relative noise amplitude for ``RobustSVD_random``.
        driver: cuSOLVER SVD driver or None.

    Returns:
        (U, S, Vh) with U (..., M, K), S (..., K), Vh (..., K, N).
    """
    # 1. QR Decomposition
    Q, R = torch.linalg.qr(A, mode='reduced')
    
    # 2. Robust SVD on R (Calling the _random version class)
    U_prime, S, Vh = RobustSVD_random.apply(R, jitter, driver)
    
    # 3. Reconstruct U
    U = Q @ U_prime
    
    return U, S, Vh


# ========================================== SVD via Eigen Decomposition ==========================================
def svd_via_eigh(
    A: torch.Tensor, epsilon: float = 1e-16,
    jitter: float | str = 0.0, nonuniform_diag: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduced SVD from eigh of the Gram matrix A^T A (or A A^T).

    Tall/square A (M >= N): P = A^T A (N x N), eigh gives
    P = V L V^T, S = sqrt(L) flipped to descending order, Vh = V^T and
    U = A V diag(1/S). Wide A (M < N): P = A A^T (M x M), U = its
    eigenvectors, Vh = diag(1/S) U^T A. The jitter is added to P, not
    to A, so the eigenvectors returned by eigh stay exactly
    orthogonal. Singular values below S_max * epsilon are set to 0
    together with their reciprocal, so U (tall) or Vh (wide) has
    zero columns (rows) in the numerical null space instead of inf.
    Batched eigh runs through cuSOLVER (or MAGMA if preferred) and
    is far faster than batched SVD for n > 32 on GPU.

    Plain aten ops, no custom backward: autograd differentiates
    through ``torch.linalg.eigh``, whose adjoint has 1/(l_i - l_j)
    terms that are inf/nan for exactly degenerate eigenvalues (use
    ``RobustSVD_EIG`` for a broadened backward, or ``svd_via_jacobi``
    for a capture-safe one). The Gram matrix squares the condition
    number, so small singular values are resolved only to about
    sqrt(eps_dtype) * S_max (1e-8 in f64, 3e-4 in f32). Real inputs
    only (``.mT``, no conjugation). No sign fixing of the vectors.

    Args:
        A: (..., M, N) real matrix (batched).
        epsilon: relative tail threshold; S < S_max * epsilon is
            zeroed, and 1/S is evaluated on S clamped to epsilon.
        jitter: numeric -> relative regularisation, the shift added to
            the diagonal of P is jitter * trace(P) / K (jitter in
            units of the mean eigenvalue of P, i.e. ||A||_F^2 / K);
            'auto' -> dtype-aware resolution-based shift, see
            ``_gram_diag_shift``.
        nonuniform_diag: if True add diag_shift * diag(1, 2, ..., K)/K
            (lifts degeneracies deterministically); else
            diag_shift * I.

    Returns:
        (U, S, Vh) with U (..., M, K), S (..., K) descending,
        Vh (..., K, N), K = min(M, N).
    """
    M, N = A.shape[-2:]

    # --- Case 1: Tall Matrix (M >= N) ---
    if M >= N:
        P = A.mT @ A

        # Relative jitter on P = A^T A (not on A); numeric jitter =
        # mean-eigenvalue units, 'auto' = dtype-aware resolution-
        # based shift (see _gram_diag_shift).
        K = N
        diag_shift = _gram_diag_shift(A, P, K, jitter)  # (..., 1, 1)
        if nonuniform_diag:
            diag_vals = torch.arange(
                1, K + 1, device=A.device, dtype=A.dtype,
            ) / K
            P = P + diag_shift * torch.diag_embed(
                diag_vals.expand(P.shape[:-1]),
            )
        else:
            P = P + diag_shift * torch.eye(
                K, device=A.device, dtype=A.dtype,
            )

        L, V = torch.linalg.eigh(P)
        L = torch.clamp(L, min=0.0)
        S = torch.sqrt(L)

        # eigh returns ascending order; SVD needs descending
        S = torch.flip(S, dims=[-1])
        V = torch.flip(V, dims=[-1])
        Vh = V.mT

        # U = A @ V @ diag(1/S), with 1/S clamped for stability
        threshold = S.max(dim=-1, keepdim=True).values * epsilon
        mask = S > threshold
        S = torch.where(mask, S, torch.zeros_like(S))
        inv_s = torch.where(
            mask, 1.0 / S.clamp(min=epsilon),
            torch.zeros_like(S),
        )
        U = A @ V @ torch.diag_embed(inv_s)

    # --- Case 2: Wide Matrix (M < N) ---
    else:
        P = A @ A.mT

        K = M
        diag_shift = _gram_diag_shift(A, P, K, jitter)  # (..., 1, 1)
        if nonuniform_diag:
            diag_vals = torch.arange(
                1, K + 1, device=A.device, dtype=A.dtype,
            ) / K
            P = P + diag_shift * torch.diag_embed(
                diag_vals.expand(P.shape[:-1]),
            )
        else:
            P = P + diag_shift * torch.eye(
                K, device=A.device, dtype=A.dtype,
            )

        L, U_eig = torch.linalg.eigh(P)
        L = torch.clamp(L, min=0.0)
        S = torch.sqrt(L)

        S = torch.flip(S, dims=[-1])
        U = torch.flip(U_eig, dims=[-1])

        # Vh = diag(1/S) @ U^T @ A
        threshold = S.max(dim=-1, keepdim=True).values * epsilon
        mask = S > threshold
        S = torch.where(mask, S, torch.zeros_like(S))
        inv_s = torch.where(
            mask, 1.0 / S.clamp(min=epsilon),
            torch.zeros_like(S),
        )
        Vh = torch.diag_embed(inv_s) @ U.mT @ A

    return U, S, Vh

class RobustSVD_EIG(torch.autograd.Function):
    """SVD via eigh of the Gram matrix with a broadened SVD backward.

    Forward is ``svd_via_eigh`` (jitter on A^T A, optional non-uniform
    diagonal). Backward is the same Lorentzian-broadened SVD adjoint
    as ``RobustSVD_random`` (absolute epsilon 1e-12 in every
    reciprocal, ``nan_to_num`` guard), replacing the nan-prone eigh
    adjoint autograd would otherwise use. With nonuniform_diag=True
    the shift is jitter * diag(1, 2, ..., K)/K instead of jitter * I,
    lifting singular-value degeneracies so the 1/(s_i - s_j) terms
    stay finite. Deterministic (no randomness) with
    ``generate_vmap_rule = True``, so it works under nested vmap. The
    custom backward is NOT preserved by torch.export (export
    dissolves autograd.Function); use ``svd_via_jacobi`` there.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(
        A: torch.Tensor, jitter: float | str, driver: Optional[str],
        nonuniform_diag: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduced SVD through ``svd_via_eigh``.

        Args:
            A: (..., M, N) real matrix (batched).
            jitter: relative Gram-matrix shift (or 'auto'), see
                ``svd_via_eigh``.
            driver: accepted for signature compatibility with
                ``RobustSVD``; unused (eigh has no driver argument).
            nonuniform_diag: use diag(1..K)/K instead of I for the
                shift.

        Returns:
            (U, S, Vh) with U (..., M, K), S (..., K) descending,
            Vh (..., K, N).
        """
        # Jitter is added to A^T A (not A) inside svd_via_eigh,
        # so eigenvectors from eigh stay orthogonal.
        U, S, Vh = svd_via_eigh(
            A, jitter=jitter, nonuniform_diag=nonuniform_diag,
        )
        return U, S, Vh

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[Any, ...], output: tuple[Any, ...],
    ) -> None:
        """Save (U, S, Vh) for the backward pass.

        Args:
            ctx: autograd context.
            inputs: forward inputs (A, jitter, driver, nonuniform_diag).
            output: forward outputs (U, S, Vh).
        """
        U, S, Vh = output
        ctx.save_for_backward(U, S, Vh)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dU: torch.Tensor, dS: torch.Tensor, dVh: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        """Broadened SVD adjoint (arXiv:1903.09650), see ``RobustSVD``.

        Args:
            ctx: autograd context holding (U, S, Vh).
            dU: (..., M, K) cotangent of U.
            dS: (..., K) cotangent of S.
            dVh: (..., K, N) cotangent of Vh.

        Returns:
            (dA, None, None, None) with dA (..., M, N); no gradient
            for jitter, driver or nonuniform_diag.
        """
        U, S, Vh = ctx.saved_tensors

        M = U.size(-2)
        N = Vh.size(-1)
        K = S.size(-1)
        eye_K = torch.eye(K, dtype=U.dtype, device=U.device)
        epsilon = 1e-12

        UdU = U.mT @ dU
        VdV = Vh @ dVh.mT

        # SVD backward formula (arXiv:1903.09650):
        #   F[i,j] = 1/(s_i - s_j),  G[i,j] = 1/(s_i + s_j)
        #   Su = (F+G) * skew(U^T dU) / 2
        #   Sv = (F-G) * skew(V^T dV) / 2
        # Use safe_inverse_random for numerical stability
        F = S.unsqueeze(-2) - S.unsqueeze(-1)
        F = safe_inverse_random(F, epsilon=epsilon)
        F = F * (1 - eye_K)

        G = S.unsqueeze(-2) + S.unsqueeze(-1)
        G = safe_inverse_random(G, epsilon=epsilon)
        G = G * (1 - eye_K)

        Su = (F + G) * (UdU - UdU.mT) / 2
        Sv = (F - G) * (VdV - VdV.mT) / 2

        # NaN Guard
        Su = torch.nan_to_num(Su, nan=0.0, posinf=0.0, neginf=0.0)
        Sv = torch.nan_to_num(Sv, nan=0.0, posinf=0.0, neginf=0.0)

        dA = U @ (Su + Sv + torch.diag_embed(dS)) @ Vh

        # Handle non-square contributions
        S_inv = safe_inverse_random(S, epsilon=epsilon)


        if M > K:
            term1 = (dU * S_inv.unsqueeze(-2)) @ Vh
            term2 = U @ (U.mT @ term1)
            delta = term1 - term2
            dA = dA + delta

        if N > K:
            term1 = (U * S_inv.unsqueeze(-2)) @ dVh
            term2 = term1 @ (Vh.mT @ Vh)
            delta = term1 - term2
            dA = dA + delta

        return dA, None, None, None

def robust_svd_eig_wrapper(
    A: torch.Tensor, jitter: float | str = 1e-12,
    driver: Optional[str] = None, nonuniform_diag: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable eigh-based SVD (``RobustSVD_EIG.apply``).

    Args:
        A: (..., M, N) real matrix.
        jitter: relative Gram-matrix shift or 'auto'.
        driver: unused, kept for signature compatibility.
        nonuniform_diag: lift degeneracies with a diag(1..K)/K shift.

    Returns:
        (U, S, Vh) with U (..., M, K), S (..., K), Vh (..., K, N).
    """
    return RobustSVD_EIG.apply(A, jitter, driver, nonuniform_diag)

def robust_svd_err_catcher_wrapper(
    A: torch.Tensor, jitter: float = 1e-12, driver: Optional[str] = None,
    nonuniform_diag: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``RobustSVD`` with fallback to ``RobustSVD_EIG`` on failure.

    The cuSOLVER/LAPACK SVD raises ``torch.linalg.LinAlgError`` (a
    RuntimeError subclass) when it fails to converge on degenerate or
    ill-conditioned matrices; in that case the SVD is recomputed from
    eigh of the Gram matrix. Note that ``jitter`` means a shift on A
    in the first path and a shift on A^T A in the fallback. Only
    errors raised synchronously by the forward are caught.

    Args:
        A: (..., M, N) real matrix.
        jitter: relative jitter strength for both paths.
        driver: cuSOLVER SVD driver for the first path or None.
        nonuniform_diag: if True the eigh fallback uses the
            non-uniform diagonal shift to lift degeneracies.

    Returns:
        (U, S, Vh) with U (..., M, K), S (..., K), Vh (..., K, N).
    """
    try:
        return RobustSVD.apply(A, jitter, driver)
    except RuntimeError:
        return RobustSVD_EIG.apply(A, jitter, driver, nonuniform_diag)

# ========== Cholesky QR dispatch ================


def _cholesky_qr_forward(
    A: torch.Tensor, rel_jitter: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduced QR via Cholesky of the shifted Gram matrix.

    Tall/square (M >= N): G = A^T A + jitter * I, L = cholesky(G),
    R = L^T and Q = A R^{-1} (via ``solve_triangular``). Wide (M < N):
    the same on the leading square block A[..., :M] gives Q, then
    R = Q^T A (upper trapezoidal). The shift is relative,
    jitter = rel_jitter * s with s = max over the batch of ||A||_F^2,
    one scalar shared by all matrices in the batch (per matrix only
    under vmap), detached so no gradient flows through it. Only
    cholesky + triangular solves, hence much faster than Householder
    QR for batched small matrices, but the Gram matrix squares the
    condition number and ``cholesky`` raises on a numerically
    indefinite G (rank-deficient A with too small a shift; for wide
    A the leading block alone must have full rank). Autograd can
    flow through it, but the raw cholesky/solve adjoint is unstable;
    ``CholeskyQR`` supplies the standard QR backward instead.

    Args:
        A: (..., M, N) real matrix (batched).
        rel_jitter: relative Gram shift (in units of ||A||_F^2).

    Returns:
        (Q, R) with Q (..., M, K) and R (..., K, N), K = min(M, N).
    """
    # detach: the shift is a constant regularizer, no grad to A
    scale = (A * A).sum(dim=(-2, -1)).detach()
    jitter = rel_jitter * scale.amax()  # scalar
    M, N = A.shape[-2:]
    if M >= N:
        G = A.mT @ A
        G = G + jitter * torch.eye(
            N, device=G.device, dtype=G.dtype,
        )
        L = torch.linalg.cholesky(G)
        R = L.mT
        QT = torch.linalg.solve_triangular(
            L, A.mT, upper=False,
        )
        Q = QT.mT
    else:
        A1 = A[..., :M]
        G = A1.mT @ A1
        G = G + jitter * torch.eye(
            M, device=G.device, dtype=G.dtype,
        )
        L = torch.linalg.cholesky(G)
        QT = torch.linalg.solve_triangular(
            L, A1.mT, upper=False,
        )
        Q = QT.mT
        R = Q.mT @ A
    return Q, R


def _solve_R_inv_T(
    R: torch.Tensor, A: torch.Tensor, rel_jitter: float = 0.0,
) -> torch.Tensor:
    """Compute A @ R^{-T} through a broadened pseudo-inverse of R.

    R = U_R diag(S_R) Vh_R (via ``svd_via_eigh``, avoiding the
    batched cuSOLVER SVD performance cliff) gives
    R^{-T} = U_R diag(1/S_R) Vh_R, with 1/S_R evaluated as
    ``safe_inverse`` (absolute epsilon 1e-12) so near-zero singular
    values of R are damped rather than inverted. ``rel_jitter`` is
    the relative Gram-matrix shift handed to ``svd_via_eigh`` (scaled
    by trace(R^T R)/K internally); it regularises the eigh and is not
    a singular-value threshold. Used only inside the QR backward
    formulas.

    Args:
        R: (..., K, K) upper triangular factor.
        A: (..., M, K) matrix to multiply.
        rel_jitter: relative jitter for ``svd_via_eigh``.

    Returns:
        A @ R^{-T}, shape (..., M, K).
    """
    # R = U_R @ diag(S_R) @ Vh_R
    # R^{-T} = U_R @ diag(1/S_R) @ Vh_R
    U_R, S_R, Vh_R = svd_via_eigh(R, jitter=rel_jitter)
    S_inv = safe_inverse(S_R)
    return (A @ U_R) * S_inv.unsqueeze(-2) @ Vh_R


def _qr_backward_tall(
    Q: torch.Tensor, R: torch.Tensor, dQ: torch.Tensor,
    dR: torch.Tensor, rel_jitter: float = 0.0,
) -> torch.Tensor:
    """Standard reduced-QR adjoint for tall/square A (M >= N).

    Seeger et al. (2017) / Liao et al. (2019):
    M = copyltu(R dR^T - dQ^T Q), dA = (dQ + Q M) R^{-T}, where
    copyltu mirrors the lower triangle onto the upper one. R^{-T} is
    applied with ``_solve_R_inv_T``.

    Args:
        Q: (..., M, N) orthonormal factor.
        R: (..., N, N) upper triangular factor.
        dQ: (..., M, N) cotangent of Q.
        dR: (..., N, N) cotangent of R.
        rel_jitter: relative jitter for the inverse of R.

    Returns:
        dA of shape (..., M, N).
    """
    M_mat = R @ dR.mT - dQ.mT @ Q
    M_mat = M_mat.tril(0) + M_mat.tril(-1).mT
    tmp = dQ + Q @ M_mat
    return _solve_R_inv_T(R, tmp, rel_jitter)


def _qr_backward_wide(
    A: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
    dQ: torch.Tensor, dR: torch.Tensor, rel_jitter: float = 0.0,
) -> torch.Tensor:
    """Standard reduced-QR adjoint for wide A (M < N).

    Split A = [X, Y] and R = [U, V] with X, U square (M x M). Then
    (Q, U) is the QR of X and V = Q^T Y, so the tall formula is
    applied to X with the cotangent of Q augmented by Y dV^T, and the
    gradient with respect to Y is Q dV.

    Args:
        A: (..., M, N) input matrix.
        Q: (..., M, M) orthonormal factor.
        R: (..., M, N) upper trapezoidal factor.
        dQ: (..., M, M) cotangent of Q.
        dR: (..., M, N) cotangent of R.
        rel_jitter: relative jitter for the inverse of U.

    Returns:
        dA of shape (..., M, N).
    """
    M, N = A.shape[-2:]
    U, V = R.split((M, N - M), dim=-1)
    dU, dV = dR.split((M, N - M), dim=-1)

    tmp = dQ + A[..., M:] @ dV.mT
    M_mat = U @ dU.mT - tmp.mT @ Q
    M_mat = M_mat.tril(0) + M_mat.tril(-1).mT
    tmp = tmp + Q @ M_mat
    dX = _solve_R_inv_T(U, tmp, rel_jitter)
    return torch.cat((dX, Q @ dV), dim=-1)


class CholeskyQR(torch.autograd.Function):
    """Cholesky QR forward + standard QR backward.

    Forward uses the fast Cholesky path (cholesky + solve_triangular).
    Backward uses the standard QR gradient formula instead of
    autograd through cholesky/solve, which is numerically unstable
    for rank-deficient boundary MPS matrices.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(
        A: torch.Tensor, rel_jitter: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cholesky QR forward (see ``_cholesky_qr_forward``).

        Args:
            A: (..., M, N) real matrix (batched).
            rel_jitter: relative Gram shift in units of ||A||_F^2.

        Returns:
            (Q, R) with Q (..., M, K) and R (..., K, N).
        """
        Q, R = _cholesky_qr_forward(A, rel_jitter)
        return Q, R

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[Any, ...], output: tuple[Any, ...],
    ) -> None:
        """Save what the backward needs: (Q, R), plus A if wide.

        Args:
            ctx: autograd context; also receives ``tall`` (bool) and
                ``rel_jitter``.
            inputs: forward inputs (A, rel_jitter).
            output: forward outputs (Q, R).
        """
        A, rel_jitter = inputs
        Q, R = output
        M, N = A.shape[-2:]
        if M < N:
            ctx.save_for_backward(A, Q, R)
        else:
            ctx.save_for_backward(Q, R)
        ctx.tall = (M >= N)
        ctx.rel_jitter = rel_jitter

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dQ: torch.Tensor, dR: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Standard QR adjoint (``_qr_backward_tall`` / ``_wide``).

        Args:
            ctx: autograd context.
            dQ: (..., M, K) cotangent of Q.
            dR: (..., K, N) cotangent of R.

        Returns:
            (dA, None) with dA (..., M, N); no gradient for
            ``rel_jitter``.
        """
        nt = ctx.rel_jitter
        if ctx.tall:
            Q, R = ctx.saved_tensors
            dA = _qr_backward_tall(Q, R, dQ, dR, nt)
        else:
            A, Q, R = ctx.saved_tensors
            dA = _qr_backward_wide(A, Q, R, dQ, dR, nt)
        return dA, None


def qr_via_cholesky(
    x: torch.Tensor, jitter: float = 1e-16, adaptive_jitter: bool = False,
    forward_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduced QR via Cholesky forward and standard QR backward.

    Dispatches to ``CholeskyQR.apply`` (stable custom backward) or,
    with ``forward_only=True``, directly to ``_cholesky_qr_forward``
    (autograd then goes through cholesky/solve_triangular, which is
    unstable for rank-deficient inputs).

    Args:
        x: (..., M, N) real matrix (batched).
        jitter: relative regularisation strength;
            ``_cholesky_qr_forward`` scales it by (the batch max of)
            ||x||_F^2 before adding it to the Gram diagonal, so the
            shift tracks the scale of x automatically.
        adaptive_jitter: ignored. The jitter is always relative; the
            argument is kept so existing call sites do not break.
        forward_only: if True skip the custom autograd.Function.

    Returns:
        (Q, R) with Q (..., M, K) and R (..., K, N), K = min(M, N).
    """
    del adaptive_jitter  # always relative; arg kept for compatibility
    if forward_only:
        Q, R = _cholesky_qr_forward(x, jitter)
        return Q, R
    else:
        return CholeskyQR.apply(x, jitter)


# ========== Size/device-aware dispatch ==========

def qr_via_svd(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """"QR" from the robust SVD: A = U S Vh -> Q = U, R = diag(S) Vh.

    Uses ``robust_svd_err_catcher_wrapper`` with its defaults (jitter
    1e-12 on A). R is not upper triangular, which does not matter
    for TN canonicalisation where only Q^T Q = I and Q R = A are
    needed. Intended for small GPU matrices (n <= 32) where
    cuSOLVER's batched Jacobi SVD is much faster than QR.

    Args:
        x: (..., M, N) real matrix.

    Returns:
        (Q, R) with Q (..., M, K) and R (..., K, N), K = min(M, N).
    """
    U, S, Vh = robust_svd_err_catcher_wrapper(x)
    R = S.unsqueeze(-1) * Vh
    return U, R


def qr_via_eigh(
    x: torch.Tensor, jitter: float | str = 1e-16,
    nonuniform_diag: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """"QR" from the eigh-based SVD: Q = U, R = diag(S) Vh.

    Uses ``RobustSVD_EIG`` (SVD through eigh of A^T A or A A^T, with
    the broadened SVD backward). With cuSOLVER's batched eigh this
    is much faster than native SVD or QR for batched n > 32. R is
    not upper triangular; irrelevant for TN canonicalisation. This
    is the production QR hook installed by ``setup_linalg_hooks``.

    Args:
        x: (..., M, N) real matrix (batched).
        jitter: relative Gram-matrix shift (or 'auto'), see
            ``svd_via_eigh``.
        nonuniform_diag: if True use the non-uniform diagonal shift
            to lift singular-value degeneracies (stabilises the
            backward).

    Returns:
        (Q, R) with Q (..., M, K) and R (..., K, N), K = min(M, N).
    """
    U, S, Vh = RobustSVD_EIG.apply(x, jitter, None, nonuniform_diag)
    R = S.unsqueeze(-1) * Vh
    return U, R


def size_aware_qr(
    x: torch.Tensor, via_eigh: bool = False, jitter: float | str = 0.0,
    nonuniform_diag: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Size/device-aware reduced QR.

    With ``via_eigh=True`` always returns ``qr_via_eigh`` (Q = U,
    R = diag(S) Vh, not upper triangular). Otherwise: on CUDA with
    n = min(M, N) <= 32 use ``qr_via_svd`` (cuSOLVER's batched
    Jacobi SVD is ~10-15x faster than QR, which has no fused batched
    kernel and spawns ~3k sub-kernel launches per call); on CPU or
    for n > 32 use Householder ``torch.linalg.qr`` (upper triangular
    R).

    Args:
        x: (..., M, N) real matrix (batched).
        via_eigh: force the eigh-based path.
        jitter: relative Gram shift, used by the eigh path only.
        nonuniform_diag: non-uniform diagonal shift, eigh path only.

    Returns:
        (Q, R) with Q (..., M, K) and R (..., K, N), K = min(M, N).
    """
    if via_eigh:
        return qr_via_eigh(x, jitter, nonuniform_diag=nonuniform_diag)
    n = min(x.shape[-2], x.shape[-1])
    if x.is_cuda and n <= 32:
        return qr_via_svd(x)
    return torch.linalg.qr(x)


def size_aware_svd(
    x: torch.Tensor, jitter: float | str = 1e-16,
    driver: Optional[str] = None, backend: str = 'cuSOLVER',
    nonuniform_diag: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backend-selected differentiable reduced SVD (production hook).

    backend='cuSOLVER' (default): always ``RobustSVD_EIG`` (SVD via
    cuSOLVER/LAPACK eigh of the Gram matrix) for every size and
    device; the n <= 32 special case is disabled.

    backend='auto': on CUDA with 32 < n <= 256 (n = min(M, N)) run
    ``RobustSVD_EIG`` with the preferred CUDA linalg library
    temporarily switched to MAGMA (cuSOLVER SVD jumps from ~0.4 ms
    for its fused n <= 32 Jacobi kernel to ~70 ms above it, while
    MAGMA eigh stays ~4 ms); otherwise (CPU, n <= 32 or n > 256) fall
    through to ``robust_svd_err_catcher_wrapper`` (cuSOLVER/LAPACK
    SVD of the jittered x with eigh fallback).

    Args:
        x: (..., M, N) real matrix (batched).
        jitter: relative jitter; Gram shift in the eigh paths, shift
            on x itself in the ``RobustSVD`` path.
        driver: cuSOLVER SVD driver for the ``RobustSVD`` path; not
            used by the eigh paths.
        backend: 'cuSOLVER' or 'auto'; anything else raises
            ValueError.
        nonuniform_diag: if True use the non-uniform diagonal shift
            in the eigh paths to lift singular-value degeneracies.

    Returns:
        (U, S, Vh) with U (..., M, K), S (..., K) descending,
        Vh (..., K, N), K = min(M, N).
    """
    n = min(x.shape[-2], x.shape[-1])
    if backend == 'auto':
        if x.is_cuda and n > 32 and n <= 256:
            prev = torch.backends.cuda.preferred_linalg_library()
            torch.backends.cuda.preferred_linalg_library('magma')
            try:
                return RobustSVD_EIG.apply(
                    x, jitter, driver, nonuniform_diag,
                )
            finally:
                torch.backends.cuda.preferred_linalg_library(prev)
    elif backend == 'cuSOLVER':
        # if x.is_cuda and n > 32:
        return RobustSVD_EIG.apply(
            x, jitter, driver, nonuniform_diag,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return robust_svd_err_catcher_wrapper(
        x, jitter=jitter, driver=driver, nonuniform_diag=nonuniform_diag,
    )


# ===========================================================================
# Pure-PyTorch MINRES (Paige & Saunders 1975)
# ===========================================================================
def torch_minres(
    matvec: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor,
    rtol: float = 1e-5, maxiter: int = 100,
) -> tuple[torch.Tensor, int]:
    """MINRES solver in pure PyTorch — runs entirely on GPU.

    Solves A x = b where A is symmetric (accessed via matvec).
    Implements Paige & Saunders (1975), mirroring scipy's
    implementation exactly.

    One matvec call per iteration; all other ops are O(Np) vector
    arithmetic.  Scalar extractions (.item()) are negligible
    versus the matvec cost.

    Not CUDA-graph capturable: the per-iteration ``.item()`` calls
    are host syncs and the loop exit is data dependent. Not meant to
    be differentiated through.

    Args:
        matvec: callable, x -> A @ x for symmetric A; (Np,) real
            tensor in and out, on the same device as ``b``.
        b: (Np,) real right-hand side.
        rtol: relative tolerance. Converges when
              ||r|| < rtol * ||A|| * ||x|| (matching scipy).
        maxiter: maximum Lanczos iterations.

    Returns:
        (x, info): x is the (Np,) solution, info is 0 if converged
        else ``maxiter``.
    """
    b_norm = torch.linalg.norm(b).item()
    if b_norm == 0:
        return torch.zeros_like(b), 0

    # Lanczos init: r1 = b, beta1 = ||b||
    x = torch.zeros_like(b)
    r1 = b.clone()
    r2 = b.clone()
    beta1 = b_norm
    beta = beta1

    # Givens rotation state
    cs = -1.0
    sn = 0.0
    oldb = 0.0
    dbar = 0.0
    epsln = 0.0
    phibar = beta1

    # Estimates of ||A|| and ||x|| (following scipy MINRES)
    Anorm2 = 0.0

    # w vectors for solution update
    w = torch.zeros_like(b)
    w2 = torch.zeros_like(b)

    info = maxiter
    for itn in range(1, maxiter + 1):
        # Lanczos step
        s = 1.0 / beta
        v = s * r2                          # v_k

        y = matvec(v)

        if itn >= 2:
            y = y - (beta / oldb) * r1

        alfa = torch.dot(v, y).item()
        y = y - (alfa / beta) * r2

        r1 = r2
        r2 = y
        oldb = beta
        beta = torch.linalg.norm(r2).item()

        # Apply previous rotation Q_{k-1}
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta

        # Compute new rotation Q_k
        gamma = math.sqrt(gbar ** 2 + beta ** 2)
        gamma = max(gamma, 1e-300)
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar

        # Update x
        denom = 1.0 / gamma
        w1 = w2
        w2 = w
        w = (v - oldeps * w1 - delta * w2) * denom
        x = x + phi * w

        # Update ||A|| estimate from Lanczos coefficients
        Anorm2 += alfa ** 2 + oldb ** 2 + beta ** 2
        Anorm = math.sqrt(Anorm2)

        # Update ||x|| estimate
        ynorm = torch.linalg.norm(x).item()

        # Convergence: ||r|| < rtol * ||A|| * ||x|| (scipy criterion)
        rnorm = abs(phibar)
        if rnorm < rtol * Anorm * ynorm:
            info = 0
            break

        if beta == 0.0:
            info = 0
            break

    return x, info

# ===========================================================================
# CUDA-graph-capturable truncation linalg (fixed-sweep Jacobi eigh)
# ===========================================================================
# torch.linalg.eigh/svd/cholesky host-sync during CUDA-graph capture
# and invalidate it, so chi > 0 contractions (whose SVD/QR hooks all
# funnel into cuSOLVER eigh) could never run under captured/replayed
# graphs. The fixed-sweep batched cyclic Jacobi below is built from
# plain capture-safe aten ops. It is registered as a torch.library
# op so exported graphs keep ONE node per call site, and made
# differentiable by graph composition (straight-through + analytic
# first-order reattachment, `jacobi_eigh_diff`) — the gradient is
# exactly the Lorentzian-broadened eigh adjoint and survives
# torch.export / torch.func.grad / vmap / cudagraphs. Derivation and
# validation: projects/nnfpeps_larger_D/torch/
# truncation_backward_theory.md.
#
# Extra benefit: the raw aten eigh backward is nan-prone on the
# rank-deficient Gram matrices at truncation points (exact
# degenerate zero eigenvalues -> 1/(w_i - w_j) = inf) once export
# has dissolved RobustSVD_EIG's custom backward; the broadened
# adjoint is finite by construction.


def _round_robin_schedule(n: int) -> list[list[tuple[int, int]]]:
    """Round-robin pairing: n-1 rounds of floor(n/2) disjoint pairs.

    Standard circle method (index n-1 fixed, others rotate). For odd
    n a dummy index pairs with one real index each round and that
    pair is dropped (giving n rounds of (n-1)/2 pairs). Pure Python —
    runs at trace time only.

    Args:
        n: matrix dimension (n < 2 gives an empty schedule).

    Returns:
        List of rounds; each round is a list of (p, q) index pairs
        with p < q, all pairs within a round disjoint. Over one full
        cycle every off-diagonal (p, q) appears exactly once.
    """
    if n < 2:
        return []
    m = n + (n % 2)  # pad to even
    idx = list(range(m))
    rounds = []
    for _ in range(m - 1):
        pairs = [
            (idx[i], idx[m - 1 - i])
            for i in range(m // 2)
            if idx[i] < n and idx[m - 1 - i] < n  # drop dummy pairs
        ]
        rounds.append([(min(p, q), max(p, q)) for p, q in pairs])
        idx = [idx[0]] + [idx[-1]] + idx[1:-1]
    return rounds


_JACOBI_SCHEDULE_CACHE = {}


def _jacobi_schedule_tensors(
    n: int, device: torch.device | str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Round-robin schedule as device LongTensors, cached per device.

    Building the index tensors is an H2D copy, which is illegal
    DURING CUDA-graph capture; the module-level cache keyed by
    (n, str(device)) guarantees it happens at warmup time instead
    (cudagraphs always warm up first), so capture only sees already
    resident device tensors.

    Args:
        n: matrix dimension.
        device: device the index tensors live on.

    Returns:
        List over rounds of (p_idx, q_idx), each an int64 tensor of
        shape (n_pairs,) with p_idx < q_idx elementwise.
    """
    key = (n, str(device))
    if key not in _JACOBI_SCHEDULE_CACHE:
        _JACOBI_SCHEDULE_CACHE[key] = [
            (
                torch.tensor([p for p, _ in pairs], device=device),
                torch.tensor([q for _, q in pairs], device=device),
            )
            for pairs in _round_robin_schedule(n)
        ]
    return _JACOBI_SCHEDULE_CACHE[key]


def _jacobi_eigh_impl(
    P: torch.Tensor, sweeps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched cyclic Jacobi for real symmetric ``P`` (..., n, n).

    Returns (w, V) with eigenvalues ASCENDING and eigenvectors in
    columns, matching ``torch.linalg.eigh``. Fixed ``sweeps`` full
    cyclic sweeps — no convergence test, no host sync, no
    data-dependent control flow, so every launched kernel is
    CUDA-graph capturable.

    Each round applies floor(n/2) disjoint Givens rotations at once
    as a single orthogonal matrix G: A <- G^T A G, V <- V G. The
    angle theta = 0.5*atan2(2*A[p,q], A[q,q]-A[p,p]) annihilates
    A[p,q] and is NaN-free for A[p,q] = 0. This is the
    CompositeExplicitAutograd kernel of ``vmc_torch::jacobi_eigh``;
    call it through ``jacobi_eigh`` / ``jacobi_eigh_diff``.

    Args:
        P: (..., n, n) real symmetric matrix (batched); complex is
            unsupported (atan2 of the entries).
        sweeps: number of full cyclic sweeps (Python int, trace-time
            constant). ~10 reaches f64 machine precision for n <= 32.

    Returns:
        (w, V): w (..., n) ascending eigenvalues, V (..., n, n)
        orthogonal eigenvectors in columns; both are new tensors that
        do not alias ``P``.
    """
    n = P.shape[-1]
    batch_shape = P.shape[:-2]
    if n == 1:
        # clone: op outputs must not alias the input
        w = P[..., 0, 0].unsqueeze(-1).clone()
        V = torch.ones_like(P)
        return w, V

    A = P.clone().reshape(-1, n, n)  # (B, n, n)
    B = A.shape[0]
    eye = torch.eye(n, dtype=P.dtype, device=P.device)
    V = eye.expand(B, n, n).clone()
    barange = torch.arange(B, device=P.device)

    schedule = _jacobi_schedule_tensors(n, P.device)
    for _ in range(sweeps):
        for p_idx, q_idx in schedule:
            app = A[:, p_idx, p_idx]  # (B, n_pairs)
            aqq = A[:, q_idx, q_idx]
            apq = A[:, p_idx, q_idx]
            # tan(2 theta) = 2 apq / (aqq - app); atan2 handles
            # apq == 0 and app == aqq without NaN/Inf
            theta = 0.5 * torch.atan2(2.0 * apq, aqq - app)
            c = torch.cos(theta)
            s = torch.sin(theta)
            # assemble all disjoint rotations into one G
            G = eye.expand(B, n, n).clone()
            bidx = barange[:, None].expand(B, p_idx.numel())
            G[bidx, p_idx, p_idx] = c
            G[bidx, q_idx, q_idx] = c
            G[bidx, p_idx, q_idx] = s
            G[bidx, q_idx, p_idx] = -s
            A = G.mT @ A @ G
            V = V @ G

    w = torch.diagonal(A, dim1=-2, dim2=-1)
    # eigh convention: ascending eigenvalues
    w, order = torch.sort(w, dim=-1)
    V = torch.gather(V, -1, order.unsqueeze(-2).expand_as(V))
    return (
        w.reshape(*batch_shape, n),
        V.reshape(*batch_shape, n, n),
    )


# Registered through the low-level Library API (not the custom_op
# decorator): the decorator's generated autograd wrapper is not
# functorch-compatible in current nightlies, and differentiability
# is provided by jacobi_eigh_diff (graph composition) anyway.
_VMC_LIB = torch.library.Library("vmc_torch", "DEF")
_VMC_LIB.define("jacobi_eigh(Tensor P, int sweeps) -> (Tensor, Tensor)")
_VMC_LIB.impl(
    "jacobi_eigh", _jacobi_eigh_impl, "CompositeExplicitAutograd",
)


def jacobi_eigh(
    P: torch.Tensor, sweeps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture-safe eigh ORACLE via the ``vmc_torch::jacobi_eigh`` op.

    Thin wrapper around ``torch.ops.vmc_torch.jacobi_eigh`` so that
    exported graphs keep ONE node per call site. No backward: if
    ``P.requires_grad`` under grad mode the Autograd kernel raises
    RuntimeError instead of silently detaching — differentiate via
    ``jacobi_eigh_diff``. Works under vmap (batch rule registered)
    and with fake tensors / torch.export (fake kernel registered).

    Args:
        P: (..., n, n) real symmetric matrix (batched).
        sweeps: fixed number of cyclic Jacobi sweeps.

    Returns:
        (w, V): w (..., n) ascending eigenvalues, V (..., n, n)
        eigenvectors in columns.
    """
    return torch.ops.vmc_torch.jacobi_eigh(P, sweeps)


@torch.library.register_fake("vmc_torch::jacobi_eigh")
def _jacobi_eigh_fake(
    P: torch.Tensor, sweeps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake (meta) kernel: output shapes/dtypes for tracing/export.

    Args:
        P: (..., n, n) fake tensor.
        sweeps: unused.

    Returns:
        Empty tensors of shapes (..., n) and (..., n, n) with the
        dtype and device of ``P``.
    """
    return P.new_empty(P.shape[:-1]), P.new_empty(P.shape)


def _jacobi_eigh_autograd(
    P: torch.Tensor, sweeps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autograd-key kernel: raise instead of silently detaching.

    Registered under the "Autograd" dispatch key. If grad mode is on
    and ``P.requires_grad``, raises RuntimeError pointing the caller
    to ``jacobi_eigh_diff``; otherwise redispatches below autograd
    to the CompositeExplicitAutograd kernel.

    Args:
        P: (..., n, n) real symmetric matrix.
        sweeps: fixed number of cyclic Jacobi sweeps.

    Returns:
        (w, V) as ``_jacobi_eigh_impl``.
    """
    if torch.is_grad_enabled() and P.requires_grad:
        raise RuntimeError(
            "jacobi_eigh has no backward; call jacobi_eigh_diff "
            "for a differentiable eigh"
        )
    with torch._C._AutoDispatchBelowAutograd():
        return torch.ops.vmc_torch.jacobi_eigh(P, sweeps)


_VMC_LIB.impl("jacobi_eigh", _jacobi_eigh_autograd, "Autograd")


def _jacobi_eigh_vmap(
    info: Any, in_dims: tuple[Optional[int], Optional[int]],
    P: torch.Tensor, sweeps: int,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor], tuple[Optional[int], Optional[int]]
]:
    """vmap batch rule: fold the mapped dim of P into the batch dims.

    The impl is batch-agnostic, so the mapped dimension is moved to
    the front and both outputs report batch dim 0. If P is not
    mapped (``pdim is None``) the op is called as is.

    Args:
        info: torch vmap info (unused).
        in_dims: (pdim, None) mapped dim of P and of ``sweeps``.
        P: (..., n, n) possibly with an extra mapped dim.
        sweeps: fixed number of cyclic Jacobi sweeps.

    Returns:
        ((w, V), (out_dim_w, out_dim_V)) with out dims 0 or None.
    """
    # impl is batch-agnostic: move the vmapped dim into the batch
    (pdim, _) = in_dims
    if pdim is None:
        w, V = jacobi_eigh(P, sweeps)
        return (w, V), (None, None)
    P = P.movedim(pdim, 0)
    w, V = jacobi_eigh(P, sweeps)
    return (w, V), (0, 0)


torch.library.register_vmap(
    "vmc_torch::jacobi_eigh", _jacobi_eigh_vmap,
)


def jacobi_eigh_diff(
    P: torch.Tensor, sweeps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable eigh: Jacobi oracle + analytic reattachment.

    The oracle runs on ``P.detach()`` (values only); differentiability
    is then restored through plain aten ops via first-order
    perturbation theory around the oracle's (w0, V0):

        M  = V0^T P V0                  (value ~ diag(w0))
        w  = diag(M)                    (dw_i = (V0^T dP V0)_ii)
        C  = F o M,  F_ij = 1/(w0_j - w0_i) broadened, 0 on diag
        V  = V0 + V0 @ (C - C.detach()) (value = V0 exactly,
                                         dV = V0 (F o V0^T dP V0))

    Gradients are therefore EXACTLY the (Lorentzian-broadened) eigh
    adjoint expressed as graph composition: no autograd.Function, no
    op backward, so it composes with torch.func.grad / vmap /
    export / cudagraphs. Full derivation:
    projects/nnfpeps_larger_D/torch/truncation_backward_theory.md.

    Broadening: eps = max(1e-12, 1e-6 * max_k |w0_k|), i.e. relative
    to the spectral radius (~10 ulp in f32); degenerate eigenvalue
    pairs get a finite, damped rotation gradient. Capture-safe like
    the oracle (plain aten ops, no host syncs).

    Args:
        P: (..., n, n) real symmetric matrix (batched); may require
            grad.
        sweeps: fixed number of cyclic Jacobi sweeps.

    Returns:
        (w, V): w (..., n) ascending Rayleigh quotients (equal to the
        oracle eigenvalues up to the Jacobi residual), V (..., n, n)
        eigenvectors in columns (value bitwise equal to the oracle's).
    """
    w0, V0 = jacobi_eigh(P.detach(), sweeps)
    n = w0.shape[-1]

    M = V0.mT @ P @ V0
    w = torch.diagonal(M, dim1=-2, dim2=-1)  # Rayleigh quotients

    d = w0.unsqueeze(-2) - w0.unsqueeze(-1)  # d[i, j] = w0_j - w0_i
    eps = (w0.abs().amax(dim=-1, keepdim=True) * 1e-6).clamp(
        min=1e-12,
    ).unsqueeze(-1)
    F = d / (d * d + eps * eps)
    F = F * (1.0 - torch.eye(n, dtype=P.dtype, device=P.device))

    C = F * M
    V = V0 + V0 @ (C - C.detach())
    return w, V


def _jacobi_auto_sweeps(n: int, base: int) -> int:
    """Size-adaptive sweep count for the Jacobi eigensolver.

    Measured f64 convergence to ~1e-14 rel: n <= 32 -> base,
    <= 48 -> base + 2, <= 64 -> base + 4, <= 96 -> base + 8, i.e.
    base + 2 * ceil((n - 32) / 16) above 32. Trace-time Python
    arithmetic only.

    Args:
        n: matrix dimension.
        base: sweep count for n <= 32.

    Returns:
        Number of sweeps.
    """
    if n <= 32:
        return base
    return base + 2 * math.ceil((n - 32) / 16)


# safety factor for jitter='auto': covers the eigensolver's backward
# -error polynomial c_n, the Frobenius overestimate of lambda_max
# (<= sqrt(K)) and the null-space projection prefactor of the
# nonuniform-diagonal splitting (O(1/sqrt(K))..O(1/K)). Validated
# 8/8 across K in [8, 64], rank ratios, scales 1e-4..1e6 and
# flat/decaying spectra (safety=10 fails 4/8).
_AUTO_JITTER_SAFETY = 100.0


def _gram_diag_shift(
    A: torch.Tensor, P: torch.Tensor, K: int, jitter: float | str,
) -> torch.Tensor:
    """Diagonal shift for the Gram matrix P (..., K, K), detached.

    jitter='auto': resolution-based, dtype-aware. A symmetric
    eigensolver resolves eigenvalues only to
    delta ~ eps(dtype) * lambda_max(P) (backward stability + Weyl),
    an ABSOLUTE scale — so a useful degeneracy-lifting shift must
    give slot spacings above delta:

        spacing = diag_shift / K = safety * eps * ||P||_F
                  (||P||_F >= lambda_max, cheap one-reduction bound)

    Numeric jitter: legacy relative scheme, shift = jitter *
    trace(P)/K (jitter in mean-eigenvalue units). NOTE: in f32 the
    production value 1e-8 sits BELOW the eigensolver resolution —
    the splitting it induces is rounding noise, not deterministic.

    Args:
        A: (..., M, N) matrix whose Gram matrix is P; only its dtype
            is used (``finfo`` in 'auto' mode).
        P: (..., K, K) Gram matrix A^T A or A A^T.
        K: size of P.
        jitter: float (relative, mean-eigenvalue units) or 'auto';
            any other string raises ValueError.

    Returns:
        (..., 1, 1) detached real tensor: the total shift to add to
        the diagonal of P (times I or times diag(1..K)/K).
    """
    if isinstance(jitter, str):
        if jitter != 'auto':
            raise ValueError(f"unknown jitter mode {jitter!r}")
        eps_dt = torch.finfo(A.dtype).eps
        normF = torch.linalg.matrix_norm(P).detach()
        return (_AUTO_JITTER_SAFETY * eps_dt * normF * K)[
            ..., None, None,
        ]
    trace_P = torch.diagonal(P, dim1=-2, dim2=-1).sum(-1)
    if trace_P.is_complex():
        trace_P = trace_P.real
    return (jitter * trace_P.detach() / K)[..., None, None]


def svd_via_jacobi(
    A: torch.Tensor, epsilon: float = 1e-16, jitter: float | str = 0.0,
    nonuniform_diag: bool = False, sweeps: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD via jacobi_eigh_diff(A^T A) or (A A^T).

    Line-for-line the same math as ``svd_via_eigh`` (relative Gram
    jitter, optional non-uniform diagonal, descending flip, tail
    threshold + safe inverse); only the eigh core is the
    capture-safe fixed-sweep Jacobi op. Differentiable through
    ``jacobi_eigh_diff`` (broadened eigh adjoint by graph
    composition), so gradients survive torch.export / vmap /
    CUDA-graph capture, unlike ``RobustSVD_EIG``. No host syncs.
    Real inputs only.

    Args:
        A: (..., M, N) real matrix (batched).
        epsilon: relative tail threshold; S < S_max * epsilon is
            zeroed and its reciprocal set to 0.
        jitter: numeric (legacy, relative to the mean eigenvalue of
            the Gram matrix) or 'auto' (dtype-aware resolution-based
            shift, see ``_gram_diag_shift`` — recommended with
            nonuniform_diag=True for deterministic degeneracy
            tie-breaking in BOTH f32 and f64).
        nonuniform_diag: if True the shift is diag_shift *
            diag(1, 2, ..., K)/K instead of diag_shift * I.
        sweeps: base Jacobi sweep count; increased for K > 32 by
            ``_jacobi_auto_sweeps``.

    Returns:
        (U, S, Vh) with U (..., M, K), S (..., K) descending,
        Vh (..., K, N), K = min(M, N) — shapes as
        ``torch.linalg.svd(full_matrices=False)``.
    """
    M, N = A.shape[-2:]

    if M >= N:  # tall: eigh of A^T A
        P = A.mT @ A
        K = N
    else:       # wide: eigh of A A^T
        P = A @ A.mT
        K = M

    # constant regularizer -> detached
    diag_shift = _gram_diag_shift(A, P, K, jitter)
    if nonuniform_diag:
        diag_vals = torch.arange(
            1, K + 1, device=A.device, dtype=A.dtype,
        ) / K
        P = P + diag_shift * torch.diag_embed(
            diag_vals.expand(P.shape[:-1]),
        )
    else:
        P = P + diag_shift * torch.eye(
            K, device=A.device, dtype=A.dtype,
        )

    L, W = jacobi_eigh_diff(P, _jacobi_auto_sweeps(K, sweeps))
    L = torch.clamp(L, min=0.0)
    S = torch.sqrt(L)

    # ascending -> descending
    S = torch.flip(S, dims=[-1])
    W = torch.flip(W, dims=[-1])

    # zero the tail below S_max * epsilon; safe reciprocal
    threshold = S.max(dim=-1, keepdim=True).values * epsilon
    mask = S > threshold
    S = torch.where(mask, S, torch.zeros_like(S))
    inv_s = torch.where(
        mask, 1.0 / S.clamp(min=epsilon), torch.zeros_like(S),
    )

    if M >= N:
        Vh = W.mT
        U = A @ W @ torch.diag_embed(inv_s)
    else:
        U = W
        Vh = torch.diag_embed(inv_s) @ W.mT @ A

    return U, S, Vh


def qr_via_jacobi(
    A: torch.Tensor, jitter: float | str = 0.0,
    nonuniform_diag: bool = False, sweeps: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """QR via the Jacobi SVD: Q = U, R = diag(S) @ Vh.

    R is not upper triangular — irrelevant for TN canonicalization
    (same convention as ``qr_via_eigh``). Capture-safe and
    differentiable like ``svd_via_jacobi``.

    Args:
        A: (..., M, N) real matrix (batched).
        jitter: relative Gram shift or 'auto' (see
            ``svd_via_jacobi``).
        nonuniform_diag: non-uniform diagonal shift.
        sweeps: base Jacobi sweep count.

    Returns:
        (Q, R) with Q (..., M, K) and R (..., K, N), K = min(M, N).
    """
    U, S, Vh = svd_via_jacobi(
        A, jitter=jitter, nonuniform_diag=nonuniform_diag,
        sweeps=sweeps,
    )
    return U, S.unsqueeze(-1) * Vh


_CAPTURE_SAFE_LINALG_INSTALLED = False


def install_capture_safe_linalg(
    jitter: float | str = 1e-8, nonuniform_diag: bool = True,
    sweeps: int = 10,
) -> None:
    """Route autoray 'linalg.svd'/'linalg.qr' (torch backend) through
    the capture-safe Jacobi implementations.

    Call AFTER ``setup_linalg_hooks`` — autoray registrations are
    global and last-write-wins. Affects every symmray/quimb
    decomposition (bMPS compress, HOTRG compress_between, canonize).
    Must run BEFORE any model export (torch.export bakes whichever
    implementation is registered at trace time into the graph).

    Args:
        jitter: relative Gram-matrix regularization (same meaning
            and default as the run scripts' setup_linalg_hooks), or
            'auto' for the dtype-aware resolution-based shift
            (recommended; see ``_gram_diag_shift`` — a fixed 1e-8 is
            below the f32 eigensolver resolution and does not
            deterministically lift degeneracies there).
        nonuniform_diag: lift singular-value degeneracies
            (stabilizes backward), as in production.
        sweeps: base Jacobi sweep count (auto-increased for n > 32
            blocks). 10 gives ~machine-precision f64 / ~1e-6-rel f32
            agreement with cuSOLVER for n <~ 50.

    Returns:
        None. Sets the module flag ``_CAPTURE_SAFE_LINALG_INSTALLED``
        (read by ``ensure_capture_safe_linalg``). The registered
        callables take a single positional matrix argument, like the
        ones installed by ``setup_linalg_hooks``.
    """
    import autoray as ar
    global _CAPTURE_SAFE_LINALG_INSTALLED

    ar.register_function(
        'torch', 'linalg.svd',
        lambda x: svd_via_jacobi(
            x, jitter=jitter, nonuniform_diag=nonuniform_diag,
            sweeps=sweeps,
        ),
    )
    ar.register_function(
        'torch', 'linalg.qr',
        lambda x: qr_via_jacobi(
            x, jitter=jitter, nonuniform_diag=nonuniform_diag,
            sweeps=sweeps,
        ),
    )
    # quimb's split drivers resolve backend fns through autoray
    # NAMESPACES (get_namespace(x).linalg.svd), which memoize the
    # first resolution — a registration made after any decomposition
    # already ran would silently not take effect. Clearing the
    # namespace cache forces re-resolution against the registry.
    try:
        from autoray.autoray import _NAMESPACE_CACHE
        _NAMESPACE_CACHE.clear()
    except ImportError:
        pass
    _CAPTURE_SAFE_LINALG_INSTALLED = True


def ensure_capture_safe_linalg() -> None:
    """Install the capture-safe linalg hooks with defaults, once.

    No-op if ``install_capture_safe_linalg`` was already called (a
    manual install with custom settings wins). Called automatically
    by ``export_and_compile`` so exported graphs are capture-safe and
    nan-free by default.
    """
    if not _CAPTURE_SAFE_LINALG_INSTALLED:
        install_capture_safe_linalg()


# ===========================================================================
# CUDA-graph capture of the per-sample gradient function
# ===========================================================================
class GraphedGradFn:
    """CUDA-graph-replayed wrapper around an eager grad callable.

    Wraps ``fn(x, *params) -> pytree of tensors`` (the vmap(grad)
    callable built by ``export_grad``) and replays it as a recorded
    CUDA graph. This removes the per-kernel launch overhead of the
    eager forward+backward execution (measured ~13x on the chunked
    ``compute_grads_gpu`` path) WITHOUT torch.compile — compiling
    the joint forward+backward graph with inductor is prohibitively
    slow, while manual capture costs a few warmup runs plus one
    recording.

    Requirements:
      - static shapes: calls with a different input shape fall back
        to the eager ``fn`` (``compute_grads_gpu`` pads every chunk
        to ``B_grad``, so replay is used for all chunks);
      - capture-safe kernels: graphs containing cuSOLVER
        eigh/svd/cholesky (e.g. chi > 0 contractions with the
        default linalg hooks) CANNOT be captured — construction
        raises, callers should fall back to the unwrapped fn;
      - parameter VALUES are re-read on every call via ``copy_``
        into owned static buffers, so optimizer updates propagate
        regardless of whether they are in-place. Calls whose param
        dtypes/shapes mismatch the captured buffers (e.g. after
        ``model.double()`` under mixed precision) fall back to the
        eager ``fn`` instead of silently casting.

    Args:
        fn: the eager callable to capture.
        example_x: example input batch (defines the captured shape).
        example_params: parameter tensors (values only used for
            warmup; live values are copied in per call).
        warmup: eager warmup runs on a side stream before capture
            (CUDA-graphs requirement; also primes library caches).
        clone_outputs: if True (default), return fresh clones each
            call. If False, return the static output tensors
            directly — cheaper, but the caller MUST finish consuming
            them before the next call (replay overwrites in place).
            ``compute_grads_gpu`` consumes each chunk before the
            next call, so it uses False.
    """

    def __init__(
        self, fn: Callable[..., Any], example_x: torch.Tensor,
        example_params: Sequence[torch.Tensor],
        warmup: int = 3, clone_outputs: bool = True,
    ) -> None:
        """Warm up ``fn`` on a side stream and record the CUDA graph.

        See the class docstring for the argument semantics. Raises
        whatever the capture raises (e.g. for cuSOLVER kernels in
        the graph); callers should catch and fall back to ``fn``.
        ``warmup`` must be >= 1 (the warmup output is deleted after
        the loop). The side stream and the capture live on
        ``example_x.device`` (made the current CUDA device for the
        duration of ``__init__``), so multi-GPU ranks that never
        called ``torch.cuda.set_device`` capture on the right device.

        Args:
            fn: eager callable ``fn(x, *params) -> pytree``.
            example_x: (B, ...) example input, defines the static
                shape.
            example_params: parameter tensors (cloned, detached).
            warmup: eager warmup runs before capture.
            clone_outputs: clone the static outputs on every call.
        """
        self._fn = fn
        self._clone_outputs = clone_outputs
        self._x_shape = tuple(example_x.shape)

        self._static_x = example_x.clone()
        self._static_params = [
            p.detach().clone() for p in example_params
        ]

        # Streams and graph capture are bound to the CURRENT CUDA
        # device, which torch.set_default_device() does NOT change.
        # Pin it to the data's device: otherwise on a multi-GPU rank
        # with tensors on cuda:k the capture runs on cuda:0's stream
        # and dies with cudaErrorStreamCaptureInvalidated.
        dev = self._static_x.device
        with torch.cuda.device(dev):
            # warmup on a side stream (CUDA-graphs requirement)
            side = torch.cuda.Stream(device=dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    out = fn(self._static_x, *self._static_params)
            torch.cuda.current_stream(dev).wait_stream(side)
            del out

            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._static_out = fn(
                    self._static_x, *self._static_params,
                )

    def _compatible(
        self, x: torch.Tensor, params: Sequence[torch.Tensor],
    ) -> bool:
        """Whether (x, params) match the captured static buffers.

        Checks the shape of ``x`` and the count, dtype and shape of
        every parameter (device and the dtype of ``x`` are not
        checked).

        Args:
            x: candidate input batch.
            params: candidate parameter tensors.

        Returns:
            True if the recorded graph can be replayed on them.
        """
        if tuple(x.shape) != self._x_shape:
            return False
        if len(params) != len(self._static_params):
            return False
        return all(
            p.dtype == sp.dtype and p.shape == sp.shape
            for p, sp in zip(params, self._static_params)
        )

    def __call__(self, x: torch.Tensor, *params: torch.Tensor) -> Any:
        """Evaluate ``fn(x, *params)`` by graph replay when possible.

        Copies ``x`` and the parameter values into the static buffers,
        replays the graph and returns the outputs (clones if
        ``clone_outputs``, else the static buffers themselves, valid
        only until the next call). Incompatible inputs (see
        ``_compatible``) are evaluated eagerly with ``fn``.

        Args:
            x: (B, ...) input batch.
            *params: parameter tensors, same order as at capture.

        Returns:
            Pytree of tensors with the structure returned by ``fn``.
        """
        if not self._compatible(x, params):
            # ragged chunk / dtype switch (mixed precision): eager
            return self._fn(x, *params)
        self._static_x.copy_(x)
        for sp, p in zip(self._static_params, params):
            sp.copy_(p)
        self._graph.replay()
        if self._clone_outputs:
            from torch.utils import _pytree as pytree
            return pytree.tree_map(
                lambda t: t.clone()
                if isinstance(t, torch.Tensor) else t,
                self._static_out,
            )
        return self._static_out


# ========================================== Benchmarking Suite ==========================================

def benchmark_svd_full(
    M: int, N: int, batch_size: int = 10, num_batches: int = 10,
    jitter: float = 1e-12, driver: Optional[str] = None,
    device: str | torch.device = 'cpu',
    dtype: torch.dtype = torch.float64,
    condition_mode: str = 'normal', seed: int = 42,
) -> None:
    """Benchmark the legacy SVD variants against ``torch.linalg.svd``.

    Compares standard SVD, RobustSVD (identity jitter), QR+RobustSVD,
    RobustSVD_random, QR+RobustSVD_random and RobustSVD_EIG on
    reconstruction error and on the deviation of S and (sign-aligned)
    U from the baseline, averaged over ``num_batches`` random
    batches, and prints a table.

    Args:
        M: number of rows.
        N: number of columns.
        batch_size: matrices per batch.
        num_batches: number of random batches to average over.
        jitter: jitter strength passed to every robust variant.
        driver: cuSOLVER SVD driver or None.
        device: torch device for the test matrices.
        dtype: floating dtype of the test matrices.
        condition_mode: 'normal' (Gaussian), 'decay' (singular values
            1 .. 1e-15, ill-conditioned) or 'degenerate' (blocks of
            repeated singular values that trigger cuSOLVER
            convergence failures). Any other value raises
            UnboundLocalError.
        seed: unused; call ``torch.manual_seed`` yourself.

    Returns:
        None (prints results).
    """
    print(f"\n{'='*100}")
    print(f"BENCHMARK: Shape=({batch_size}, {M}, {N}) | Batches={num_batches}")
    print(f"Jitter={jitter:.1e} | Mode={condition_mode} | Device={device}")
    print(f"{'='*100}")

    # Initialize metrics dictionary for 5 methods
    methods = ["std", "robust_id", "qr_id", "robust_rand", "qr_rand", "eigh_ref"]
    metrics = {m: {"diff_U": 0., "diff_S": 0., "recon": 0.} for m in methods}
    
    # Helper to align signs
    def align_signs(
        U_target: torch.Tensor, Vh_target: torch.Tensor,
        U_pred: torch.Tensor, Vh_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flip column signs of U_pred / rows of Vh_pred to match
        U_target, using the sign of the first-row entry per column.

        Args:
            U_target: (..., M, K) reference left singular vectors.
            Vh_target: unused.
            U_pred: (..., M, K) left singular vectors to align.
            Vh_pred: (..., K, N) right singular vectors to align.

        Returns:
            (U_pred, Vh_pred) with consistent signs.
        """
        # Align sign of the first column of U
        sign_flip = torch.sign(U_pred[..., 0:1, :] * U_target[..., 0:1, :])
        sign_flip = torch.where(sign_flip == 0, torch.ones_like(sign_flip), sign_flip)
        return U_pred * sign_flip, Vh_pred * sign_flip.mT

    for i in range(num_batches):
        # --- 1. Data Generation ---
        K = min(M, N)
        
        if condition_mode == 'normal':
            A = torch.randn(batch_size, M, N, device=device, dtype=dtype)
            
        elif condition_mode == 'decay':
            # Ill-conditioned: 1.0 -> 1e-15
            U_gen, _, _ = torch.linalg.svd(torch.randn(batch_size, M, M, device=device, dtype=dtype))
            Vh_gen, _, _ = torch.linalg.svd(torch.randn(batch_size, N, N, device=device, dtype=dtype))
            S_gen = torch.logspace(0, -15, steps=K, device=device, dtype=dtype)
            S_gen = S_gen.unsqueeze(0).expand(batch_size, -1)
            S_mat = torch.zeros(batch_size, M, N, device=device, dtype=dtype)
            S_mat[:, :K, :K] = torch.diag_embed(S_gen)
            A = U_gen @ S_mat @ Vh_gen
            
        elif condition_mode == 'degenerate':
            # Degenerate: Blocks of repeated values (e.g., 1.0, 1.0, 1.0...)
            U_gen, _, _ = torch.linalg.svd(torch.randn(batch_size, M, M, device=device, dtype=dtype))
            Vh_gen, _, _ = torch.linalg.svd(torch.randn(batch_size, N, N, device=device, dtype=dtype))
            
            # Create stairs: [1, 1, 1, 0.1, 0.1, 0.1, ...]
            num_blocks = 8
            block_size = K // num_blocks
            S_vals = []
            for b in range(num_blocks):
                val = 10.0 ** (-b * 2) # 1, 1e-2, 1e-4...
                S_vals.append(torch.full((block_size,), val, device=device, dtype=dtype))
            
            # Fill remaining
            rem = K - len(S_vals)*block_size
            if rem > 0: S_vals.append(torch.full((rem,), 1e-8, device=device, dtype=dtype))
            
            S_gen = torch.cat(S_vals).unsqueeze(0).expand(batch_size, -1)
            S_mat = torch.zeros(batch_size, M, N, device=device, dtype=dtype)
            S_mat[:, :K, :K] = torch.diag_embed(S_gen)
            A = U_gen @ S_mat @ Vh_gen

        # --- 2. Run Methods ---
        
        # A. Standard SVD (Baseline)
        try:
            U_std, S_std, Vh_std = torch.linalg.svd(A, full_matrices=False)
            A_std = U_std @ torch.diag_embed(S_std) @ Vh_std
            metrics["std"]["recon"] += torch.norm(A - A_std).item()
        except RuntimeError:
            print(f"Batch {i}: Standard SVD failed (Convergence Error). Skipping baseline comparison.")
            # If baseline fails, we can't compute diff_U/S, but we can still check recon error of others
            # Set baseline to dummy for this loop to avoid crash
            U_std, S_std, Vh_std = torch.zeros_like(A), torch.zeros(batch_size, K), torch.zeros_like(A.mT)

        # Helper to run and record
        def run_and_record(
            name: str, func: Callable[..., Any], **kwargs: Any,
        ) -> None:
            """Run ``func(A, jitter, driver, **kwargs)`` and accumulate
            the metrics for ``name``; print and skip on RuntimeError.
            """
            try:
                U_res, S_res, Vh_res = func(A, jitter, driver, **kwargs)
                A_res = U_res @ torch.diag_embed(S_res) @ Vh_res
                
                # Align for comparison
                U_res_a, _ = align_signs(U_std, Vh_std, U_res, Vh_res)
                
                metrics[name]["diff_U"] += torch.norm(U_res_a - U_std).item()
                metrics[name]["diff_S"] += torch.norm(S_res - S_std).item()
                metrics[name]["recon"] += torch.norm(A - A_res).item()
            except RuntimeError as e:
                print(f"Method {name} failed: {e}")

        # B. Robust SVD (Identity Jitter) - The old class
        # Note: robust_svd_wrapper needs to be defined as per previous context (wrapping RobustSVD.apply)
        # Here assuming RobustSVD.apply signature is (A, jitter, driver)
        run_and_record("robust_id", robust_svd_wrapper)

        # C. QR-SVD (Identity Jitter)
        run_and_record("qr_id", qr_svd_wrapper)

        # D. Robust SVD (Random Jitter) - The new champion
        # Passing seed ensures reproducibility
        run_and_record("robust_rand", lambda a, j, d: robust_svd_wrapper_random(a, j, d))

        # E. QR-SVD (Random Jitter) - The new champion
        # Passing seed ensures reproducibility
        run_and_record("qr_rand", lambda a, j, d: qr_svd_wrapper_random(a, j, d))

        # F. Eigh-based SVD (for reference, not compared)
        run_and_record("eigh_ref", robust_svd_eig_wrapper)

    # --- 3. Reporting ---
    def avg(val: float) -> float:
        """Mean over the ``num_batches`` batches."""
        return val / num_batches

    print(f"\n{'Metric':<25} | {'Std SVD':<12} | {'Robust(Id)':<12} | {'QR(Id)':<12} | {'Robust(Rand)':<12} | {'QR(Rand)':<12} | {'Eigh(Ref)':<12}")
    print("-" * 120)
    
    # Recon Error
    print(f"{'Recon Error':<25} | {avg(metrics['std']['recon']):<12.2e} | "
          f"{avg(metrics['robust_id']['recon']):<12.2e} | "
          f"{avg(metrics['qr_id']['recon']):<12.2e} | "
          f"{avg(metrics['robust_rand']['recon']):<12.2e} | "
          f"{avg(metrics['qr_rand']['recon']):<12.2e} | "
          f"{avg(metrics['eigh_ref']['recon']):<12.2e}")
    
    # Diff S
    print(f"{'Diff S (vs Std)':<25} | {'-':<12} | "
          f"{avg(metrics['robust_id']['diff_S']):<12.2e} | "
          f"{avg(metrics['qr_id']['diff_S']):<12.2e} | "
          f"{avg(metrics['robust_rand']['diff_S']):<12.2e} | "
          f"{avg(metrics['qr_rand']['diff_S']):<12.2e} | "
          f"{avg(metrics['eigh_ref']['diff_S']):<12.2e}")

    # Diff U (Ignore if large, as discussed)
    print(f"{'Diff U (vs Std)':<25} | {'-':<12} | "
          f"{avg(metrics['robust_id']['diff_U']):<12.2e} | "
          f"{avg(metrics['qr_id']['diff_U']):<12.2e} | "
          f"{avg(metrics['robust_rand']['diff_U']):<12.2e} | "
          f"{avg(metrics['qr_rand']['diff_U']):<12.2e} | "
          f"{avg(metrics['eigh_ref']['diff_U']):<12.2e}")
    print("-" * 120)

    # Analysis
    best_recon = min(avg(metrics['qr_id']['recon']), avg(metrics['qr_rand']['recon']))
    print("\n[Analysis]")
    if condition_mode == 'degenerate':
        print("For degenerate matrices, Random Jitter is expected to be more robust against 'error code 1'.")
        print("If QR(Rand) matches QR(Id) in accuracy but survives where others fail, it's the winner.")


def benchmark_qr_cholesky(
    device: str | torch.device = 'cpu',
    dtype: torch.dtype = torch.float64,
) -> None:
    """Benchmark ``qr_via_cholesky`` against ``torch.linalg.qr``.

    Prints reconstruction error, orthogonality and column-span
    agreement across tall/square/wide/batched shapes, a
    condition-number sweep (kappa up to 1e15) and ``gradcheck``
    results (always in f64).

    Args:
        device: torch device for the test matrices.
        dtype: floating dtype for the shape and condition sweeps.

    Returns:
        None (prints results).
    """
    print(f"\n{'='*80}")
    print("BENCHMARK: qr_via_cholesky vs torch.linalg.qr")
    print(f"Device={device} | Dtype={dtype}")
    print(f"{'='*80}")

    # --- 1. Shape sweep (tall, square, wide, batched) ---
    shapes = [
        # (shape,         label)
        ((8, 4),          "tall 8x4"),
        ((16, 16),        "square 16x16"),
        ((64, 32),        "tall 64x32"),
        ((128, 64),       "tall 128x64"),
        ((4, 8),          "wide 4x8"),
        ((16, 64),        "wide 16x64"),
        ((32, 128),       "wide 32x128"),
        ((10, 64, 32),    "batch tall"),
        ((5, 128, 64),    "batch tall lg"),
        ((10, 16, 64),    "batch wide"),
        ((8, 32, 128),    "batch wide lg"),
        ((8, 16, 16),     "batch square"),
    ]

    print(f"\n{'label':<16} {'shape':<16} | "
          f"{'||A-QR||':<12} {'||QtQ-I||':<12} "
          f"{'span diff':<12} {'ref ||A-QR||':<12} "
          f"{'Q shape':<14} {'R shape':<14}")
    print("-" * 120)

    for shape, label in shapes:
        A = torch.randn(*shape, device=device, dtype=dtype)
        Q_c, R_c = qr_via_cholesky(A)
        Q_ref, R_ref = torch.linalg.qr(A, mode='reduced')

        K = min(shape[-2], shape[-1])
        eye_K = torch.eye(K, device=device, dtype=dtype)

        recon = torch.norm(A - Q_c @ R_c).item()
        ortho = torch.norm(Q_c.mT @ Q_c - eye_K).item()

        # Column-span agreement: Q_c @ Q_c^T @ Q_ref ≈ Q_ref
        proj = Q_c @ (Q_c.mT @ Q_ref)
        span_diff = torch.norm(proj - Q_ref).item()

        ref_recon = torch.norm(A - Q_ref @ R_ref).item()

        q_shape = str(tuple(Q_c.shape))
        r_shape = str(tuple(R_c.shape))

        print(f"{label:<16} {str(shape):<16} | "
              f"{recon:<12.2e} {ortho:<12.2e} "
              f"{span_diff:<12.2e} {ref_recon:<12.2e} "
              f"{q_shape:<14} {r_shape:<14}")

    # --- 2. Condition number sweep (tall + wide) ---
    for (M_s, N_s), tag in [((64, 32), "tall"), ((16, 64), "wide")]:
        K_s = min(M_s, N_s)
        print(f"\n--- Condition number sweep ({M_s}x{N_s}, {tag}) ---")
        print(f"{'kappa':<12} | {'||A-QR||':<12} {'||QtQ-I||':<12} "
              f"{'ref ||A-QR||':<12}")
        print("-" * 60)

        for log_kappa in [0, 4, 8, 12, 15]:
            U_gen, _ = torch.linalg.qr(
                torch.randn(M_s, M_s, device=device, dtype=dtype)
            )
            V_gen, _ = torch.linalg.qr(
                torch.randn(N_s, N_s, device=device, dtype=dtype)
            )
            S_gen = torch.logspace(
                0, -log_kappa, steps=K_s,
                device=device, dtype=dtype
            )
            A = U_gen[:, :K_s] @ torch.diag(S_gen) @ V_gen[:K_s, :]

            Q_c, R_c = qr_via_cholesky(A)
            Q_ref, R_ref = torch.linalg.qr(A, mode='reduced')

            eye_K = torch.eye(
                K_s, device=device, dtype=dtype
            )
            recon = torch.norm(A - Q_c @ R_c).item()
            ortho = torch.norm(Q_c.mT @ Q_c - eye_K).item()
            ref_recon = torch.norm(A - Q_ref @ R_ref).item()

            print(f"1e{log_kappa:<9} | {recon:<12.2e} "
                  f"{ortho:<12.2e} {ref_recon:<12.2e}")

    # --- 3. Autograd gradcheck (tall, square, wide, batched) ---
    print("\n--- Autograd gradcheck ---")
    grad_shapes = [
        ((8, 4),       "tall"),
        ((6, 6),       "square"),
        ((4, 8),       "wide"),
        ((16, 8),      "tall lg"),
        ((8, 16),      "wide lg"),
        ((5, 12, 6),   "batch tall"),
        ((3, 6, 12),   "batch wide"),
        ((4, 8, 8),    "batch square"),
    ]
    for shape, tag in grad_shapes:
        A_check = torch.randn(
            *shape, device=device, dtype=torch.float64,
            requires_grad=True
        )
        try:
            passed = torch.autograd.gradcheck(
                qr_via_cholesky, (A_check,),
                eps=1e-6, atol=1e-4
            )
            status = 'PASS' if passed else 'FAIL'
        except Exception:
            status = 'FAIL (ill-conditioned Cholesky backward)'
        print(f"  {tag:<14} {str(shape):<16} gradcheck: {status}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    torch.manual_seed(42)

    # # 1. Test Degenerate Case (The Killer Case)
    # # Using a slightly larger matrix to increase chance of collision
    # benchmark_svd_full(M=64, N=32, batch_size=20, num_batches=10,
    #                    jitter=1e-12, condition_mode='degenerate')

    # # 2. Test Ill-Conditioned Case
    # benchmark_svd_full(M=64, N=32, batch_size=20, num_batches=10,
    #                    jitter=1e-12, condition_mode='decay')

    # 3. Cholesky QR benchmark
    benchmark_qr_cholesky()