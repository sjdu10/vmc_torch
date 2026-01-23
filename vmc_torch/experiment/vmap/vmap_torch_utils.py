import torch
def safe_inverse(x, epsilon=1e-12):
    """ Lorentzian broadening of the inverse to avoid division by zero. """
    return x / (x.pow(2) + epsilon)

class RobustSVD(torch.autograd.Function):
    """
    A robust SVD implementation with custom backward pass for stability.
    Includes Relative Jitter to improve convergence of gesdd.
    """
    
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, A):
        # A shape: (..., M, N)
        # 1. 计算 Relative Jitter Scale
        # 我们取矩阵中绝对值最大的元素作为参考尺度
        # dim=(-2, -1) 确保无论 A 是单个矩阵还是 Batched 矩阵，都只在最后两个维度归约
        # keepdim=True 是为了让 scale 能正确广播回 A
        scale = torch.amax(torch.abs(A), dim=(-2, -1), keepdim=True)
        
        # 避免 scale 为 0 (如果 A 全是 0，我们设 scale 为 1.0 以允许微扰生效)
        scale = torch.where(scale > 0, scale, torch.tensor(1.0, dtype=A.dtype, device=A.device))
        
        # 2. 构造微扰矩阵
        # 相对微扰系数，float64 下 1e-12 比较合适，float32 下建议 1e-6
        relative_eps = 1e-12
        M, N = A.shape[-2:]
        
        # 构造单位阵 (或者对角阵)
        # device=A.device 确保兼容
        eye = torch.eye(M, N, device=A.device, dtype=A.dtype)
        
        # 3. 应用微扰: A_safe = A + (epsilon * scale) * I
        # 利用广播机制: (..., 1, 1) * (M, N) -> (..., M, N)
        # 这确保了 batch 中每个矩阵加上的是相对于它自己数值范围的 jitter
        jitter_matrix = eye * scale * relative_eps
        A_new = A + jitter_matrix
        
        # 4. Standard SVD call on perturbed matrix
        U, S, Vh = torch.linalg.svd(A_new, full_matrices=False)
        
        # 5. Vectorized Sign Fixing (Deterministic SVD)
        # Find index of max magnitude element in each column of U
        max_abs_cols = torch.argmax(torch.abs(U), dim=-2, keepdim=True) # (..., 1, K)
        
        # Gather the actual values
        gathered = torch.gather(U, -2, max_abs_cols) # (..., 1, K)
        
        # Get signs
        signs = torch.sign(gathered) # (..., 1, K)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        
        # Apply signs
        U = U * signs          
        Vh = Vh * signs.mT     

        ctx.save_for_backward(U, S, Vh)
        return U, S, Vh

    @staticmethod
    def backward(ctx, dU, dS, dVh):
        U, S, Vh = ctx.saved_tensors
        
        M = U.size(-2)
        N = Vh.size(-1)
        K = S.size(-1)
        eye_K = torch.eye(K, dtype=U.dtype, device=U.device)

        F = S - S.unsqueeze(-1)
        F = safe_inverse(F)
        F = F * (1 - eye_K) 

        G = S + S.unsqueeze(-1)
        G = safe_inverse(G)
        G = G * (1 - eye_K)

        UdU = U.mT @ dU
        VdV = Vh @ dVh.mT

        Su = (F + G) * (UdU - UdU.mT) / 2
        Sv = (F - G) * (VdV - VdV.mT) / 2
        
        dA = U @ (Su + Sv + torch.diag_embed(dS)) @ Vh
        
        S_inv = safe_inverse(S)
        
        if M > K:
            term1 = (dU * S_inv.unsqueeze(-2)) @ Vh
            term2 = U @ (U.mT @ term1)
            dA = dA + (term1 - term2)
            
        if N > K:
            term1 = (U * S_inv.unsqueeze(-2)) @ dVh
            term2 = term1 @ (Vh.mT @ Vh)
            dA = dA + (term1 - term2)

        return dA

def svd_robust(A):
    return RobustSVD.apply(A)