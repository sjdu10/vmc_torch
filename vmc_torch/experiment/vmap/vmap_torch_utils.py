import torch

def safe_inverse(x, epsilon=1e-12):
    """ Lorentzian broadening of the inverse to avoid division by zero. """
    return x / (x.pow(2) + epsilon)

class RobustSVD(torch.autograd.Function):
    """
    A robust SVD implementation with custom backward pass for stability.
    Fully compatible with vmap (pure PyTorch, no data-dependent control flow).
    """
    @staticmethod
    def forward(ctx, A):
        # Inside vmap, A is a generic Tensor (batch dim is hidden)
        # 1. Standard PyTorch SVD (removed SciPy fallback)
        # Note: If A is very ill-conditioned, we rely on the robustness of the backend (cuSOLVER/LAPACK).
        # Adding a tiny jitter to A before calling this function is recommended if crashes persist.
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        
        # 2. Vectorized Sign Fixing (Deterministic SVD)
        # Original loop replaced with tensor operations to satisfy vmap
        # Logic: Flip sign of U[:, i] and Vh[i, :] so that the element with max abs value in U[:, i] is positive.
        
        # Find index of max magnitude element in each column of U
        # U shape: (M, K), Vh shape: (K, N)
        max_abs_cols = torch.argmax(torch.abs(U), dim=-2) # (K,)
        
        # Gather the actual values at those indices
        # We need to construct indices to gather from U
        # This part handles the "sign consistency" without python loops
        gathered = torch.gather(U, -2, max_abs_cols.unsqueeze(-2)).squeeze(-2) # (K,)
        
        # Get signs (1.0 or -1.0)
        signs = torch.sign(gathered)
        # If sign is 0 (column of zeros), keep as 1.0
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        
        # Apply signs (Broadcasting handles dimensions)
        U = U * signs.unsqueeze(-2)       # (M, K) * (1, K)
        Vh = Vh * signs.unsqueeze(-1)     # (K, N) * (K, 1)

        ctx.save_for_backward(U, S, Vh)
        return U, S, Vh

    @staticmethod
    def backward(ctx, dU, dS, dVh):
        U, S, Vh = ctx.saved_tensors
        
        # Matrix dimensions
        M = U.size(-2)
        N = Vh.size(-1)
        K = S.size(-1)
        
        # Helper identity matrix for masking diagonal
        # Using shape-agnostic creation for vmap compatibility
        eye_K = torch.eye(K, dtype=U.dtype, device=U.device)

        # F matrix (Lorentzian broadening for 1/(s_i - s_j))
        # S[:, None] broadcasting works for single matrix; 
        # vmap handles batch dims automatically via generic tracing.
        F = S - S.unsqueeze(-1)
        F = safe_inverse(F)
        # Remove in-place fill_(0) which is bad for autograd
        F = F * (1 - eye_K) 

        # G matrix (Lorentzian broadening for 1/(s_i + s_j))
        G = S + S.unsqueeze(-1)
        G = safe_inverse(G)
        G = G * (1 - eye_K)

        # Contraction terms
        # Note: Transpose logic (.mT) is safer than .t() for batches
        UdU = U.mT @ dU
        VdV = Vh @ dVh.mT

        Su = (F + G) * (UdU - UdU.mT) / 2
        Sv = (F - G) * (VdV - VdV.mT) / 2
        
        # Reconstruct dA
        dA = U @ (Su + Sv + torch.diag_embed(dS)) @ Vh
        
        # Projector terms for non-square matrices
        # S_inv for projection
        S_inv = safe_inverse(S)
        
        # (M > K) term: Projecting out the orthogonal complement of U
        if M > K:
            # Replaced manual loops/checks with direct projection formula
            # dA += (I - U U^T) dU S^{-1} V
            term1 = (dU * S_inv.unsqueeze(-2)) @ Vh
            term2 = U @ (U.mT @ term1)
            dA = dA + (term1 - term2)
            
        # (N > K) term: Projecting out the orthogonal complement of V
        if N > K:
            # dA += U S^{-1} dV^T (I - V^T V)
            term1 = (U * S_inv.unsqueeze(-2)) @ dVh
            term2 = term1 @ (Vh.mT @ Vh)
            dA = dA + (term1 - term2)

        return dA

def svd_robust(A):
    """
    Functional wrapper for RobustSVD.
    Use this function in your model.
    """
    return RobustSVD.apply(A)