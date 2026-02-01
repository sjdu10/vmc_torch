import torch
import contextlib

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

# # === SVD Patch ===
# if not hasattr(torch.linalg, 'svd_orig'):
#     torch.linalg.svd_orig = torch.linalg.svd

# def vmap_friendly_svd(A, full_matrices=True, *, driver=None, **kwargs):
#     if _ENABLE_JITTER:
#         # Inside vmap, A is a BatchedTensor.
#         # We add a very small diagonal perturbation.
#         # For double (float64), 1e-12 is safe.
        
#         # 1. Construct perturbation matrix (Identity)
#         # Note: Here we need to handle the case where A is not square, only add on the diagonal
#         M, N = A.shape[-2], A.shape[-1]
        
#         # Create diagonal noise (using PyTorch broadcasting to be compatible with vmap batch dimensions)
#         # jitter_scale = A.norm(dim=(-2,-1), keepdim=True) * 1e-12 # Relative jitter is safer
#         jitter_val = 1e-12 * A.norm(dim=(-2,-1), keepdim=True)
        
#         # Construct a diagonal matrix with the same device and type as A
#         # torch.eye is compatible with vmap's automatic broadcasting
#         eye = torch.eye(M, N, device=A.device, dtype=A.dtype) * jitter_val
        
#         # 2. Perform addition (vmap compatible)
#         A_new = A + eye
#         return torch.linalg.svd_orig(A_new, full_matrices=full_matrices, driver=driver, **kwargs)
#     else:
#         return torch.linalg.svd_orig(A, full_matrices=full_matrices, driver=driver, **kwargs)

# # Apply Patch （Must be done before any SVD operation in vmap）
# torch.linalg.svd = vmap_friendly_svd


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
        # scale = torch.amax(torch.abs(A), dim=(-2, -1), keepdim=True)
        scale = A.norm(dim=(-2,-1), keepdim=True)
        
        # Jitter
        relative_eps = jitter_strength
        
        M, N = A.shape[-2:]
        eye = torch.eye(M, N, device=A.device, dtype=A.dtype)
        
        effective_jitter = scale * relative_eps
        jitter_matrix = eye * effective_jitter
        A_new = A + jitter_matrix
        
        # --- 2. SVD Calculation ---
        U, S, Vh = torch.linalg.svd(A_new, full_matrices=False, driver=driver)
        
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