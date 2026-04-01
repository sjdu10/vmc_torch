import torch
import contextlib
import math

# === Global control ===
_ENABLE_JITTER = False

@contextlib.contextmanager
def use_jitter_svd():
    global _ENABLE_JITTER
    _ENABLE_JITTER = True
    try:
        yield
    finally:
        _ENABLE_JITTER = False

# === SVD Patch ===

def safe_inverse(x, epsilon=1e-12):
    """ Lorentzian broadening of the inverse to avoid division by zero. """
    return x / (x.pow(2) + epsilon)

class RobustSVD(torch.autograd.Function):
    """
    Robust SVD with Relative Jitter.
    Updated for PyTorch 2.0+ (torch.func / vmap compatibility).
    """
    
    # automatically generate vmap rules for forward func that contains only pure pytorch operations
    generate_vmap_rule = True

    @staticmethod
    def forward(A, jitter_strength, driver):
        """
        forward must be a pure function of pytorch operations.
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
    def setup_context(ctx, inputs, output):
        """
        Staticmethod for PyTorch 2.0 
        save data for backward use.
        inputs: forward input: tuple (A,)
        output: forward output: tuple (U, S, Vh)
        """
        U, S, Vh = output
        ctx.save_for_backward(U, S, Vh)

    @staticmethod
    def backward(ctx, dU, dS, dVh):
        """
        Backward logic remains unchanged.
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
def safe_inverse_random(x, epsilon=1e-12):
    """
    Helper function to compute safe inverse of x.
    Renamed to avoid conflict with existing utils.
    """
    return x / (x**2 + epsilon)

class RobustSVD_random(torch.autograd.Function):
    """
    Renamed RobustSVD class with Random Jitter support.
    Suffix '_random' added to prevent namespace collision.
    """
    
    # Automatically generate vmap rules for pure PyTorch operations in forward
    generate_vmap_rule = True

    @staticmethod
    def forward(A, jitter_strength, driver):
        """
        Args:
            A: Input matrix (..., M, N)
            jitter_strength: Magnitude of the jitter
            driver: LAPACK/cuSOLVER driver
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
    def setup_context(ctx, inputs, output):
        U, S, Vh = output
        ctx.save_for_backward(U, S, Vh)

    @staticmethod
    def backward(ctx, dU, dS, dVh):
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


def robust_svd_wrapper(A, jitter=1e-12, driver=None):
    """
    Wrapper for Robust SVD with Identity Jitter.
    """
    return RobustSVD.apply(A, jitter, driver)


# QR via SVD wrappers

def qr_svd_wrapper(A, jitter=1e-12, driver=None):
    """
    QR-based SVD wrapper for stability on CPU/GPU.
    Solves A = U S Vh by doing:
    1. A = Q R  (QR decomposition)
    2. R = U' S Vh (Robust SVD on R)
    3. U = Q U'
    """
    # 1. QR Decomposition
    Q, R = torch.linalg.qr(A, mode='reduced')
    
    # 2. 对 R 做 Robust SVD
    U_prime, S, Vh = RobustSVD.apply(R, jitter, driver)
    
    # 3. 还原 U
    # 将正交基 Q 作用在 U' 上，得到原始矩阵 A 的左奇异向量
    U = Q @ U_prime
    
    return U, S, Vh


def robust_svd_wrapper_random(A, jitter=1e-12, driver=None):
    """
    Renamed wrapper for Robust SVD with Random Jitter.
    """
    return RobustSVD_random.apply(A, jitter, driver)

def qr_svd_wrapper_random(A, jitter=1e-12, driver=None):
    """
    Renamed wrapper for QR-SVD using RobustSVD_random.
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
    A, epsilon=1e-16,
    jitter=0.0, nonuniform_diag=False,
):
    """Compute SVD via eigh(A^T A) or eigh(A A^T).

    Args:
        A: (..., M, N)
        epsilon: threshold for reciprocal of S
        jitter: regularization added to A^T A (not to A itself),
            preserving eigenvector orthogonality from eigh
        nonuniform_diag: if True, jitter * diag(1,2,...,K)/K
            to lift degeneracies; else jitter * I
    Returns:
        U, S, Vh
    """
    M, N = A.shape[-2:]

    # --- Case 1: Tall Matrix (M >= N) ---
    if M >= N:
        P = A.mT @ A

        # Add jitter to P = A^T A, not to A
        K = N
        if nonuniform_diag:
            diag_vals = torch.arange(
                1, K + 1, device=A.device, dtype=A.dtype,
            ) / K
            P = P + jitter * torch.diag_embed(
                diag_vals.expand(P.shape[:-1]),
            )
        else:
            P = P + jitter * torch.eye(
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
        if nonuniform_diag:
            diag_vals = torch.arange(
                1, K + 1, device=A.device, dtype=A.dtype,
            ) / K
            P = P + jitter * torch.diag_embed(
                diag_vals.expand(P.shape[:-1]),
            )
        else:
            P = P + jitter * torch.eye(
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
    """SVD via eigendecomposition with optional non-uniform diagonal jitter.

    When nonuniform_diag=True, adds jitter * diag(1, 2, ..., K)/K
    instead of jitter * I.  The non-uniform diagonal lifts degeneracies
    in singular values so the backward's 1/(s_i - s_j) terms stay
    finite, avoiding gradient explosion from degenerate eigenvalues.
    Deterministic (no torch.randn), so it works under nested vmap.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(A, jitter, driver, nonuniform_diag=False):
        # Jitter is added to A^T A (not A) inside svd_via_eigh,
        # so eigenvectors from eigh stay orthogonal.
        U, S, Vh = svd_via_eigh(
            A, jitter=jitter, nonuniform_diag=nonuniform_diag,
        )
        return U, S, Vh

    @staticmethod
    def setup_context(ctx, inputs, output):
        U, S, Vh = output
        ctx.save_for_backward(U, S, Vh)

    @staticmethod
    def backward(ctx, dU, dS, dVh):
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

def robust_svd_eig_wrapper(A, jitter=1e-12, driver=None, nonuniform_diag=False):
    return RobustSVD_EIG.apply(A, jitter, driver, nonuniform_diag)

def robust_svd_err_catcher_wrapper(
    A, jitter=1e-12, driver=None, nonuniform_diag=False,
):
    """Wrapper that tries standard Robust SVD first,
    falls back to EIG-based SVD on failure.

    Args:
        nonuniform_diag: if True, use non-uniform diagonal jitter in the
            EIG fallback to lift singular value degeneracies.
    """
    try:
        return RobustSVD.apply(A, jitter, driver)
    except RuntimeError:
        return RobustSVD_EIG.apply(A, jitter, driver, nonuniform_diag)

# ========== Cholesky QR dispatch ================


def _cholesky_qr_forward(A, rel_jitter):
    """Cholesky QR forward: fast but no autograd.

    Tall/square (M >= N):
      G = A^T A + jitter*I, cholesky(G)=L, R=L^T, Q=A R^{-1}
    Wide (M < N):
      Cholesky QR on A[:,:M] → Q, then R = Q^T A
    """
    scale = (A * A).sum(dim=(-2, -1))
    scale.detach()
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


def _solve_R_inv_T(R, A, rel_jitter=0.0) -> torch.Tensor:
    """Compute A @ R^{-T} via thresholded pseudoinverse.

    Uses svd_via_eigh (fast eigh-based SVD) instead of
    torch.linalg.svd to avoid the batched SVD performance cliff.
    Singular values of R below rel_jitter get 1/S → 0.

    The jitter passed to svd_via_eigh is scaled by tr(R) so
    it stays meaningful relative to R's scale,
    even when R is ill-conditioned.

    Args:
        R: (..., K, K) upper triangular
        A: (..., M, K)
        rel_jitter: used as relative jitter scale for svd_via_eigh.
    Returns:
        A @ R^{-T}: (..., M, K)
    """
    # R = U_R @ diag(S_R) @ Vh_R
    # R^{-T} = U_R @ diag(1/S_R) @ Vh_R
    # Scale jitter by tr(R) so it's meaningful relative to R's scale.
    tr_R = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1).max().detach()
    adaptive_jitter = rel_jitter * tr_R
    U_R, S_R, Vh_R = svd_via_eigh(R, jitter=adaptive_jitter)
    S_inv = safe_inverse(S_R)
    return (A @ U_R) * S_inv.unsqueeze(-2) @ Vh_R


def _qr_backward_tall(Q, R, dQ, dR, rel_jitter=0.0):
    """Standard QR backward for tall/square (M >= N).

    From Seeger et al. (2017) / Liao et al. (2019):
      M = copyltu(R dR^T - dQ^T Q)
      dA = (dQ + Q M) R^{-T}
    """
    M_mat = R @ dR.mT - dQ.mT @ Q
    M_mat = M_mat.tril(0) + M_mat.tril(-1).mT
    tmp = dQ + Q @ M_mat
    return _solve_R_inv_T(R, tmp, rel_jitter)


def _qr_backward_wide(A, Q, R, dQ, dR, rel_jitter=0.0):
    """Standard QR backward for wide (M < N)."""
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
    def forward(A, rel_jitter):
        Q, R = _cholesky_qr_forward(A, rel_jitter)
        return Q, R

    @staticmethod
    def setup_context(ctx, inputs, output):
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
    def backward(ctx, dQ, dR):
        nt = ctx.rel_jitter
        if ctx.tall:
            Q, R = ctx.saved_tensors
            dA = _qr_backward_tall(Q, R, dQ, dR, nt)
        else:
            A, Q, R = ctx.saved_tensors
            dA = _qr_backward_wide(A, Q, R, dQ, dR, nt)
        return dA, None


def qr_via_cholesky(x, jitter=1e-16, adaptive_jitter=False, forward_only=False):
    """QR via Cholesky forward + standard QR backward.

    Uses CholeskyQR custom autograd.Function for stable gradients.

    Args:
        x: (..., M, N) tensor.
        jitter: scalar added to Gram diagonal for regularization.
        adaptive_jitter: if True, scale jitter by ||A||_F^2.
        forward_only: if True, only compute the forward pass (no backward).
    """
    if adaptive_jitter:
        scale = (x * x).sum(dim=(-2, -1))
        scale.detach()
        jitter = jitter * scale.amax()  # scalar
    if forward_only:
        Q, R = _cholesky_qr_forward(x, jitter)
        return Q, R
    else:
        return CholeskyQR.apply(x, jitter)


# ========== Size/device-aware dispatch ==========

def qr_via_svd(x):
    """QR via SVD for GPU small matrices. A = USVh -> Q=U, R=diag(S)@Vh. 
    
    Note here R is not upper triangular, but in our case of canonizing TN tensors it doesn't matter.
    """
    U, S, Vh = robust_svd_err_catcher_wrapper(x)
    R = S.unsqueeze(-1) * Vh
    return U, R


def qr_via_eigh(x, jitter=1e-16, nonuniform_diag=False):
    """QR via eigh-based SVD. A = USVh -> Q=U, R=diag(S)@Vh.

    Uses RobustSVD_EIG which computes SVD through eigendecomposition
    of A^T A (or A A^T). With the cuSOLVER batched eigh fix,
    this is much faster than native SVD or QR for batched n>32.

    Note: R is not upper triangular, but for TN canonicalization
    this doesn't matter.

    Args:
        nonuniform_diag: if True, use non-uniform diagonal jitter to lift
            singular value degeneracies (stabilizes backward).
    """
    U, S, Vh = RobustSVD_EIG.apply(x, jitter, None, nonuniform_diag)
    R = S.unsqueeze(-1) * Vh
    return U, R


def size_aware_qr(x, via_eigh=False, jitter=0.0, nonuniform_diag=False):
    """Size/device-aware QR. Uses SVD-based QR on GPU for n<=32.

    cuSOLVER's batched Jacobi kernel (n<=32) makes SVD ~10-15x faster
    than QR on GPU for small matrices. QR has no fused batched kernel
    and spawns ~3k sub-kernel launches per call.
    On CPU or for n>32, standard QR (Householder) is used.
    """
    if via_eigh:
        return qr_via_eigh(x, jitter, nonuniform_diag=nonuniform_diag)
    n = min(x.shape[-2], x.shape[-1])
    if x.is_cuda and n <= 32:
        return qr_via_svd(x)
    return torch.linalg.qr(x)


def size_aware_svd(
    x, jitter=1e-16, driver=None, backend='cuSOLVER',
    nonuniform_diag=False,
):
    """Size/device-aware SVD. Uses EIG-based+MAGMA on GPU for n>32.

    When backend=='cuSOLVER':
        For n<=32, cuSOLVER's fused Jacobi kernel is fast (~0.4ms).
        For n>32, use SVD-VIA-EIG with cuSOLVER eigh backend.

    When backend=='auto':
        For n<=32, cuSOLVER's fused Jacobi kernel is fast (~0.4ms).
        For n>32, Jacobi is unavailable and cuSOLVER SVD jumps to ~70ms.
        MAGMA eigh avoids this cliff (~4ms for all n), so we route
        through RobustSVD_EIG with MAGMA backend for large GPU matrices.
        On CPU, always uses default RobustSVD.

    Args:
        nonuniform_diag: if True, use non-uniform diagonal jitter in EIG path
            to lift singular value degeneracies (stabilizes backward).
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
def torch_minres(matvec, b, rtol=1e-5, maxiter=100):
    """MINRES solver in pure PyTorch — runs entirely on GPU.

    Solves A x = b where A is symmetric (accessed via matvec).
    Implements Paige & Saunders (1975), mirroring scipy's
    implementation exactly.

    One matvec call per iteration; all other ops are O(Np) vector
    arithmetic.  Scalar extractions (.item()) are negligible
    versus the matvec cost.

    Args:
        matvec: callable, x -> A @ x (GPU tensor in/out).
        b: (Np,) right-hand-side GPU tensor.
        rtol: relative tolerance. Converges when
              ||r|| < rtol * ||A|| * ||x|| (matching scipy).
        maxiter: maximum Lanczos iterations.

    Returns:
        x: (Np,) solution GPU tensor.
        info: 0 if converged, else maxiter.
    """
    b_norm = torch.linalg.norm(b).item()
    if b_norm == 0:
        return torch.zeros_like(b), 0

    # Lanczos init: r1 = b, beta1 = ||b||
    n = b.shape[0]
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
    ynorm2 = 0.0

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

# ========================================== Benchmarking Suite ==========================================

def benchmark_svd_full(M, N, batch_size=10, num_batches=10, jitter=1e-12, 
                       driver=None, device='cpu', dtype=torch.float64, 
                       condition_mode='normal', seed=42):
    """
    Comprehensive Benchmark: Standard vs Robust(Id) vs QR(Id) vs QR(Random).
    
    Args:
        condition_mode: 
            'normal': Random Gaussian matrices.
            'decay': Rapidly decaying singular values (Ill-conditioned).
            'degenerate': Blocks of repeated singular values (The "Error Code 1" Killer).
    """
    print(f"\n{'='*100}")
    print(f"BENCHMARK: Shape=({batch_size}, {M}, {N}) | Batches={num_batches}")
    print(f"Jitter={jitter:.1e} | Mode={condition_mode} | Device={device}")
    print(f"{'='*100}")

    # Initialize metrics dictionary for 5 methods
    methods = ["std", "robust_id", "qr_id", "robust_rand", "qr_rand", "eigh_ref"]
    metrics = {m: {"diff_U": 0., "diff_S": 0., "recon": 0.} for m in methods}
    
    # Helper to align signs
    def align_signs(U_target, Vh_target, U_pred, Vh_pred):
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
        def run_and_record(name, func, **kwargs):
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
    def avg(val): return val / num_batches

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


def benchmark_qr_cholesky(device='cpu', dtype=torch.float64):
    """Benchmark qr_via_cholesky vs torch.linalg.qr.

    Tests reconstruction, orthogonality, column-span agreement,
    and autograd across tall, square, wide, and batched shapes.
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