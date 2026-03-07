"""Compare U, Vh from torch.linalg.svd vs svd_via_eigh.

The backward computes dA = U @ M @ Vh where M = Su+Sv+diag(dS).
If M is the same (which we confirmed) but U/Vh differ, that's
the source of the explosion.

Run:
    torchrun --nproc_per_node=1 scripts/test_svd_vectors_compare.py
"""
import autoray as ar
import torch
import torch.distributed as dist

from vmc_torch.GPU.models import fPEPS_Model_GPU
from vmc_torch.GPU.torch_utils import svd_via_eigh
from vmc_torch.GPU.vmc_setup import (
    load_or_generate_peps,
    initialize_walkers,
)
from vmc_torch.GPU.vmc_utils import random_initial_config

_captured = []


def _capturing_svd(x):
    _captured.append(x.detach().cpu().clone())
    return torch.linalg.svd(x, full_matrices=False)


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

    print(f"Captured {len(_captured)} matrices\n")

    # Analyze first few
    for idx in [0, 5, 15]:
        if idx >= len(_captured):
            break
        A = _captured[idx]
        # Handle batch dimension
        if A.dim() == 3:
            A = A[0]  # take first batch element

        print(f"{'='*80}")
        print(f"Matrix {idx}: shape={tuple(A.shape)}")

        U_svd, S_svd, Vh_svd = torch.linalg.svd(
            A, full_matrices=False,
        )
        U_eig, S_eig, Vh_eig = svd_via_eigh(A)

        K = S_svd.shape[0]
        k = min(chi, K)

        print(f"\n--- Singular values ---")
        print(f"  S_svd[:5] = "
              f"{[f'{v:.6e}' for v in S_svd[:5].tolist()]}")
        print(f"  S_eig[:5] = "
              f"{[f'{v:.6e}' for v in S_eig[:5].tolist()]}")
        print(f"  max|S_svd - S_eig| = "
              f"{(S_svd - S_eig).abs().max().item():.4e}")

        # Compare U and Vh
        print(f"\n--- Singular vectors (full K={K}) ---")
        print(f"  ||U_svd - U_eig||   = "
              f"{(U_svd - U_eig).norm().item():.4e}")
        print(f"  ||Vh_svd - Vh_eig|| = "
              f"{(Vh_svd - Vh_eig).norm().item():.4e}")

        # Orthogonality check
        eye_K = torch.eye(K, dtype=A.dtype, device='cpu')
        print(f"  ||U_svd^T U_svd - I|| = "
              f"{(U_svd.mT @ U_svd - eye_K).norm().item():.4e}")
        print(f"  ||U_eig^T U_eig - I|| = "
              f"{(U_eig.mT @ U_eig - eye_K).norm().item():.4e}")

        # Reconstruction
        print(f"\n--- Reconstruction ---")
        A_svd = U_svd @ torch.diag(S_svd) @ Vh_svd
        A_eig = U_eig @ torch.diag(S_eig) @ Vh_eig
        print(f"  ||A - U_svd S Vh_svd|| = "
              f"{(A - A_svd).norm().item():.4e}")
        print(f"  ||A - U_eig S Vh_eig|| = "
              f"{(A - A_eig).norm().item():.4e}")

        # Column norms of U (should all be 1)
        u_svd_norms = U_svd.norm(dim=0)
        u_eig_norms = U_eig.norm(dim=0)
        print(f"\n--- Column norms of U ---")
        print(f"  U_svd norms (tail): "
              f"{[f'{v:.4e}' for v in u_svd_norms[-5:].tolist()]}")
        print(f"  U_eig norms (tail): "
              f"{[f'{v:.4e}' for v in u_eig_norms[-5:].tolist()]}")
        print(f"  max|U_svd_col_norm - 1| = "
              f"{(u_svd_norms - 1).abs().max().item():.4e}")
        print(f"  max|U_eig_col_norm - 1| = "
              f"{(u_eig_norms - 1).abs().max().item():.4e}")

        # Row norms of Vh
        vh_svd_norms = Vh_svd.norm(dim=1)
        vh_eig_norms = Vh_eig.norm(dim=1)
        print(f"\n--- Row norms of Vh ---")
        print(f"  max|Vh_svd_row_norm - 1| = "
              f"{(vh_svd_norms - 1).abs().max().item():.4e}")
        print(f"  max|Vh_eig_row_norm - 1| = "
              f"{(vh_eig_norms - 1).abs().max().item():.4e}")

        # The key test: if M is the same, does U @ M @ Vh
        # give different results?
        M = torch.randn(K, K, dtype=A.dtype, device='cpu')
        dA_svd = U_svd @ M @ Vh_svd
        dA_eig = U_eig @ M @ Vh_eig
        print(f"\n--- U @ M @ Vh (random M) ---")
        print(f"  ||U_svd M Vh_svd|| = "
              f"{dA_svd.norm().item():.4e}")
        print(f"  ||U_eig M Vh_eig|| = "
              f"{dA_eig.norm().item():.4e}")
        print(f"  ratio = "
              f"{dA_eig.norm().item() / max(dA_svd.norm().item(), 1e-30):.4e}")

        # If U, Vh are both orthogonal, then
        # ||U M Vh|| = ||M|| regardless.
        # So the ratio should be ~1.
        # If it's not, U_eig or Vh_eig is not orthogonal!
        print(f"  ||M|| = {M.norm().item():.4e}")
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
