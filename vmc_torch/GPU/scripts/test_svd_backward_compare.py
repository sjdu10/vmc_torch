"""Direct comparison of SVD backward: PyTorch vs RobustSVD_EIG.

Takes actual boundary MPS matrices from a forward pass, then compares
||dA|| from:
  (A) torch.linalg.svd backward (C++ native)
  (B) RobustSVD_EIG backward (custom Python with Lorentzian eps=1e-12)

Uses a realistic loss: loss = sum(U_trunc @ diag(S_trunc) @ Vh_trunc)
which gives nonzero gU, gS, gVh for the top-chi components.

This isolates the backward implementation as the sole variable.

Run:
    torchrun --nproc_per_node=1 scripts/test_svd_backward_compare.py
"""
import autoray as ar
import torch
import torch.distributed as dist

from vmc_torch.GPU.models import fPEPS_Model_GPU
from vmc_torch.GPU.torch_utils import (
    RobustSVD_EIG,
    safe_inverse_random,
    size_aware_svd,
)
from vmc_torch.GPU.vmc_setup import (
    load_or_generate_peps,
    initialize_walkers,
)
from vmc_torch.GPU.vmc_utils import random_initial_config

_captured = []


def _capturing_svd(x):
    _captured.append(x.detach().cpu().clone())
    return torch.linalg.svd(x, full_matrices=False)


def test_backward_on_matrix(A, chi=10, label=""):
    """Compare backward dA from PyTorch SVD vs RobustSVD_EIG."""
    A_torch = A.clone().requires_grad_(True)
    A_eig = A.clone().requires_grad_(True)

    K = min(A.shape[-2], A.shape[-1])
    # Truncation to chi
    k = min(chi, K)

    # ========== (A) PyTorch native SVD backward ==========
    U_t, S_t, Vh_t = torch.linalg.svd(
        A_torch, full_matrices=False,
    )
    # Truncate and compute a realistic loss
    loss_t = (
        U_t[..., :, :k]
        @ torch.diag_embed(S_t[..., :k])
        @ Vh_t[..., :k, :]
    ).sum()
    loss_t.backward()
    dA_torch = A_torch.grad.clone()

    # ========== (B) RobustSVD_EIG backward ==========
    U_e, S_e, Vh_e = RobustSVD_EIG.apply(
        A_eig, 1e-8, None, True,  # nonuniform_diag=True
    )
    loss_e = (
        U_e[..., :, :k]
        @ torch.diag_embed(S_e[..., :k])
        @ Vh_e[..., :k, :]
    ).sum()
    loss_e.backward()
    dA_eig = A_eig.grad.clone()

    # ========== Print comparison ==========
    print(f"\n--- {label} shape={tuple(A.shape)}, "
          f"K={K}, trunc_k={k} ---")
    print(f"  Forward spectra (top 5):")
    S_t_d = S_t.detach()
    S_e_d = S_e.detach()
    if S_t_d.dim() == 1:
        print(f"    torch: {S_t_d[:5].tolist()}")
        print(f"    eig:   {S_e_d[:5].tolist()}")
    else:
        print(f"    torch[0]: "
              f"{S_t_d[0, :5].tolist()}")
        print(f"    eig[0]:   "
              f"{S_e_d[0, :5].tolist()}")

    print(f"  loss_torch = {loss_t.item():.6e}, "
          f"loss_eig = {loss_e.item():.6e}")
    print(f"  ||dA_torch||  = {dA_torch.norm().item():.6e}")
    print(f"  ||dA_eig||    = {dA_eig.norm().item():.6e}")
    print(f"  ratio         = "
          f"{dA_eig.norm().item() / max(dA_torch.norm().item(), 1e-30):.6e}")
    print(f"  max|dA_torch| = "
          f"{dA_torch.abs().max().item():.6e}")
    print(f"  max|dA_eig|   = "
          f"{dA_eig.abs().max().item():.6e}")

    # Check for NaN/Inf
    for name, dA in [("torch", dA_torch), ("eig", dA_eig)]:
        nans = torch.isnan(dA).sum().item()
        infs = torch.isinf(dA).sum().item()
        if nans or infs:
            print(f"  WARNING {name}: "
                  f"{nans} NaN, {infs} Inf")

    return dA_torch, dA_eig


def analyze_backward_internals(A, chi=10, label=""):
    """Manually compute the backward internals to see
    where the explosion happens."""
    K = min(A.shape[-2], A.shape[-1])
    k = min(chi, K)

    # Forward (PyTorch SVD)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # Simulate truncation gradient:
    # loss = sum(U[:,:k] @ diag(S[:k]) @ Vh[:k,:])
    # gU[:, :k] = diag(S[:k]) @ Vh[:k, :] summed
    # (simplified: we just need any realistic gU, gS, gVh)
    gU = torch.zeros_like(U)
    gS = torch.zeros_like(S)
    gVh = torch.zeros_like(Vh)

    # Gradient of sum(U_trunc @ diag(S_trunc) @ Vh_trunc)
    # w.r.t. U: (diag(S[:k]) @ Vh[:k,:])^T = Vh[:k,:]^T @ diag(S[:k])
    gU[..., :, :k] = (
        Vh[..., :k, :].mT
        @ torch.diag_embed(S[..., :k])
    )
    gS[..., :k] = (
        U[..., :, :k].mT
        @ Vh[..., :k, :].mT
    ).diagonal(dim1=-2, dim2=-1)
    gVh[..., :k, :] = (
        torch.diag_embed(S[..., :k])
        @ U[..., :, :k].mT
    )

    # ========== PyTorch backward formula ==========
    UhgU = U.mT @ gU - (U.mT @ gU).mT  # skew
    VhgV = Vh @ gVh.mT - (Vh @ gVh.mT).mT  # skew

    S2 = S * S
    eye_K = torch.eye(
        K, device=A.device, dtype=A.dtype,
    )
    E_pt = S2.unsqueeze(-2) - S2.unsqueeze(-1)
    E_pt = E_pt + eye_K  # fill diagonal with 1

    numer_pt = UhgU * S.unsqueeze(-2) + S.unsqueeze(-1) * VhgV
    ret_pt = numer_pt / E_pt

    # ========== RobustSVD_EIG backward formula ==========
    eps = 1e-12
    diff = S.unsqueeze(-2) - S.unsqueeze(-1)
    F = safe_inverse_random(diff, epsilon=eps)
    F = F * (1 - eye_K)

    summ = S.unsqueeze(-2) + S.unsqueeze(-1)
    G = safe_inverse_random(summ, epsilon=eps)
    G = G * (1 - eye_K)

    Su = (F + G) * (UhgU - UhgU.mT) / 2
    Sv = (F - G) * (VhgV - VhgV.mT) / 2
    ret_eig = Su + Sv

    print(f"\n=== Backward internals: {label} ===")
    print(f"  Shape: {tuple(A.shape)}, K={K}, trunc={k}")

    # Handle batch dimension
    if ret_pt.dim() == 3:
        for bi in range(ret_pt.shape[0]):
            _print_internals(
                bi, K, k, S[bi], UhgU[bi], VhgV[bi],
                E_pt[bi], numer_pt[bi], ret_pt[bi],
                F[bi], G[bi], Su[bi], Sv[bi], ret_eig[bi],
                eye_K,
            )
    else:
        _print_internals(
            0, K, k, S, UhgU, VhgV,
            E_pt, numer_pt, ret_pt,
            F, G, Su, Sv, ret_eig, eye_K,
        )


def _print_internals(
    bi, K, k, S, UhgU, VhgV,
    E_pt, numer_pt, ret_pt,
    F, G, Su, Sv, ret_eig, eye_K,
):
    off_diag = (1 - eye_K).bool()

    # Break down by block
    top_top = torch.zeros(K, K, dtype=torch.bool,
                          device=S.device)
    top_top[:k, :k] = True
    top_top = top_top & off_diag

    top_tail = torch.zeros(K, K, dtype=torch.bool,
                           device=S.device)
    top_tail[:k, k:] = True
    top_tail[k:, :k] = True
    top_tail = top_tail & off_diag

    tail_tail = torch.zeros(K, K, dtype=torch.bool,
                            device=S.device)
    tail_tail[k:, k:] = True
    tail_tail = tail_tail & off_diag

    def block_stats(tensor, mask, name):
        vals = tensor[mask]
        if vals.numel() == 0:
            return
        mx = vals.abs().max().item()
        nrm = vals.norm().item()
        nz = (vals.abs() > 1e-30).sum().item()
        print(f"    {name:>25}: max={mx:.4e}, "
              f"||.||={nrm:.4e}, "
              f"#nonzero={nz}/{vals.numel()}")

    print(f"\n  [batch {bi}]")
    print(f"  S[:5] = "
          f"{[f'{v:.4e}' for v in S[:5].tolist()]}")
    print(f"  S[{k-2}:{k+2}] = "
          f"{[f'{v:.4e}' for v in S[max(0,k-2):k+2].tolist()]}")
    print(f"  S[-3:] = "
          f"{[f'{v:.4e}' for v in S[-3:].tolist()]}")

    print(f"\n  --- skew(U^T gU) and skew(V^T gV) ---")
    block_stats(UhgU, top_top, "UhgU [top,top]")
    block_stats(UhgU, top_tail, "UhgU [top,tail]")
    block_stats(UhgU, tail_tail, "UhgU [tail,tail]")
    block_stats(VhgV, top_top, "VhgV [top,top]")
    block_stats(VhgV, top_tail, "VhgV [top,tail]")
    block_stats(VhgV, tail_tail, "VhgV [tail,tail]")

    print(f"\n  --- PyTorch: E, numerator, result ---")
    block_stats(E_pt, top_top, "E [top,top]")
    block_stats(E_pt, top_tail, "E [top,tail]")
    block_stats(E_pt, tail_tail, "E [tail,tail]")
    block_stats(numer_pt, top_top, "numer [top,top]")
    block_stats(numer_pt, top_tail, "numer [top,tail]")
    block_stats(numer_pt, tail_tail, "numer [tail,tail]")
    block_stats(ret_pt, top_top, "ret [top,top]")
    block_stats(ret_pt, top_tail, "ret [top,tail]")
    block_stats(ret_pt, tail_tail, "ret [tail,tail]")

    print(f"\n  --- RobustSVD_EIG: F, G, Su+Sv ---")
    block_stats(F, top_top, "F [top,top]")
    block_stats(F, top_tail, "F [top,tail]")
    block_stats(F, tail_tail, "F [tail,tail]")
    block_stats(G, top_top, "G [top,top]")
    block_stats(G, top_tail, "G [top,tail]")
    block_stats(G, tail_tail, "G [tail,tail]")
    block_stats(Su + Sv, top_top, "Su+Sv [top,top]")
    block_stats(Su + Sv, top_tail, "Su+Sv [top,tail]")
    block_stats(Su + Sv, tail_tail, "Su+Sv [tail,tail]")

    print(f"\n  --- Final comparison ---")
    print(f"    ||ret_pytorch|| = {ret_pt.norm().item():.4e}")
    print(f"    ||Su+Sv_eig||   = "
          f"{(Su + Sv).norm().item():.4e}")
    print(f"    ratio = "
          f"{(Su + Sv).norm().item() / max(ret_pt.norm().item(), 1e-30):.4e}")


def main():
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.set_default_device(device)
    torch.manual_seed(42)

    Lx, Ly = 8, 8
    N_sites = Lx * Ly
    t, U_ = 1.0, 8.0
    N_f = N_sites - 8
    D, chi = 10, 10

    DEFAULT_DATA_ROOT = (
        '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
        'vmc_torch/GPU/data'
    )
    fpeps_base = (
        f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/t={t}_U={U_}"
        f"/N={N_f}/Z2/D={D}/"
    )
    peps = load_or_generate_peps(
        Lx, Ly, t, U_, N_f, D,
        seed=42, dtype=dtype,
        file_path=fpeps_base,
        scale_factor=4,
    )
    model = fPEPS_Model_GPU(
        tn=peps, max_bond=chi, dtype=dtype,
        contract_boundary_opts={
            'mode': 'mps',
            'equalize_norms': 1.0,
            'canonize': True,
        },
    )
    model.to(device)

    fxs = initialize_walkers(
        init_fn=lambda seed: random_initial_config(
            N_f, N_sites, seed=seed,
        ),
        batch_size=1, seed=42, rank=0, device=device,
    )

    print(
        f"System: {Lx}x{Ly} Fermi-Hubbard, "
        f"t={t}, U={U_}, D={D}, chi={chi}"
    )

    # Capture matrices
    _captured.clear()
    ar.register_function(
        'torch', 'linalg.svd', _capturing_svd,
    )
    import quimb as qu
    params = qu.utils.tree_unflatten(
        list(model.params), model.params_pytree,
    )
    with torch.no_grad():
        model.log_amplitude(fxs[0], params)

    ar.register_function(
        'torch', 'linalg.svd',
        lambda x: torch.linalg.svd(x, full_matrices=False),
    )

    print(f"Captured {len(_captured)} matrices")

    # Test on first few matrices
    for i in [0, 5, 15, 25]:
        if i >= len(_captured):
            break
        A = _captured[i].to(device)
        test_backward_on_matrix(
            A, chi=chi, label=f"Matrix {i}",
        )
        analyze_backward_internals(
            A.cpu(), chi=chi, label=f"Matrix {i}",
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
