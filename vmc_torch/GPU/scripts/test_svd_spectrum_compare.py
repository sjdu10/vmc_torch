"""Compare singular value spectra: torch.linalg.svd vs svd_via_eigh.

Captures the actual boundary MPS matrices from a forward pass,
then compares their SVD spectra computed by different methods.

Run:
    torchrun --nproc_per_node=1 scripts/test_svd_spectrum_compare.py
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

# Capture actual matrices fed to SVD during forward pass
_captured_matrices = []


def _capturing_svd(x):
    """SVD hook that saves input matrices, then uses default SVD."""
    _captured_matrices.append(x.detach().cpu().clone())
    return torch.linalg.svd(x, full_matrices=False)


def main():
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.set_default_device(device)
    torch.manual_seed(42)

    # System setup
    Lx, Ly = 8, 8
    N_sites = Lx * Ly
    t, U = 1.0, 8.0
    N_f = N_sites - 8
    D, chi = 10, 10

    DEFAULT_DATA_ROOT = (
        '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
        'vmc_torch/GPU/data'
    )
    fpeps_base = (
        f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/t={t}_U={U}"
        f"/N={N_f}/Z2/D={D}/"
    )
    peps = load_or_generate_peps(
        Lx, Ly, t, U, N_f, D,
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
        f"t={t}, U={U}, N_f={N_f}, D={D}, chi={chi}"
    )

    # Capture matrices during forward pass
    _captured_matrices.clear()
    ar.register_function(
        'torch', 'linalg.svd', _capturing_svd,
    )

    import quimb as qu
    params = qu.utils.tree_unflatten(
        list(model.params), model.params_pytree,
    )
    with torch.no_grad():
        sign, log_abs = model.log_amplitude(fxs[0], params)
    print(f"Captured {len(_captured_matrices)} SVD input "
          f"matrices")

    # Reset hook
    ar.register_function(
        'torch', 'linalg.svd',
        lambda x: torch.linalg.svd(x, full_matrices=False),
    )

    # Compare spectra
    print(f"\n{'='*120}")
    print("Singular value spectrum comparison: "
          "torch.linalg.svd vs svd_via_eigh")
    print(f"{'='*120}")
    print(
        f"{'#':>3} {'shape':>14} "
        f"| {'SVD s_min':>12} {'EIG s_min':>12} "
        f"| {'SVD #zeros':>10} {'EIG #zeros':>10} "
        f"| {'max|S_svd - S_eig|':>18} "
        f"| {'rel_diff_tail':>14}"
    )
    print("-" * 120)

    for i, A in enumerate(_captured_matrices):
        shape = tuple(A.shape)

        # Method 1: torch.linalg.svd (LAPACK/cuSOLVER)
        _, S_svd, _ = torch.linalg.svd(
            A, full_matrices=False,
        )

        # Method 2: svd_via_eigh (A^T A -> eigh -> sqrt)
        _, S_eig, _ = svd_via_eigh(A)

        # Flatten batch for comparison
        S_svd_flat = S_svd.reshape(-1, S_svd.shape[-1])
        S_eig_flat = S_eig.reshape(-1, S_eig.shape[-1])

        for bi in range(S_svd_flat.shape[0]):
            s1 = S_svd_flat[bi]
            s2 = S_eig_flat[bi]
            K = s1.shape[0]

            abs_diff = (s1 - s2).abs()
            max_diff = abs_diff.max().item()

            n_zero_svd = (s1 < 1e-15).sum().item()
            n_zero_eig = (s2 < 1e-15).sum().item()

            # Relative diff for tail (last 10 svals)
            tail_start = max(0, K - 10)
            s1_tail = s1[tail_start:]
            s2_tail = s2[tail_start:]
            scale = s1_tail.abs().clamp(min=1e-30)
            rel_diff_tail = (
                (s1_tail - s2_tail).abs() / scale
            ).max().item()

            batch_str = (
                f"[{bi}]" if S_svd_flat.shape[0] > 1
                else ""
            )
            shape_str = 'x'.join(str(d) for d in shape)

            print(
                f"{i:>3}{batch_str:<4} "
                f"{shape_str:>14} "
                f"| {s1.min().item():>12.4e} "
                f"{s2.min().item():>12.4e} "
                f"| {n_zero_svd:>10} {n_zero_eig:>10} "
                f"| {max_diff:>18.4e} "
                f"| {rel_diff_tail:>14.4e}"
            )

        # Detailed comparison for first 3 matrices
        if i < 3:
            for bi in range(
                min(1, S_svd_flat.shape[0])
            ):
                s1 = S_svd_flat[bi]
                s2 = S_eig_flat[bi]
                K = s1.shape[0]
                print(
                    f"    --- Matrix {i}[{bi}] "
                    f"detailed tail comparison ---"
                )
                print(
                    f"    {'idx':>4} {'S_svd':>14} "
                    f"{'S_eig':>14} {'|diff|':>14} "
                    f"{'rel_diff':>14}"
                )
                start = max(0, K - 15)
                for j in range(start, K):
                    sv = s1[j].item()
                    se = s2[j].item()
                    d = abs(sv - se)
                    rd = d / max(abs(sv), 1e-30)
                    print(
                        f"    {j:>4} {sv:>14.6e} "
                        f"{se:>14.6e} {d:>14.6e} "
                        f"{rd:>14.6e}"
                    )

    # Summary: min gap comparison
    print(f"\n{'='*80}")
    print("Min gap comparison (excluding zero-zero pairs)")
    print(f"{'='*80}")
    print(
        f"{'#':>3} {'batch':>5} "
        f"| {'SVD min_gap':>14} {'EIG min_gap':>14} "
        f"| {'SVD max|F|':>14} {'EIG max|F|':>14}"
    )
    print("-" * 80)

    eps = 1e-12
    from vmc_torch.GPU.torch_utils import safe_inverse_random

    for i, A in enumerate(_captured_matrices):
        _, S_svd, _ = torch.linalg.svd(
            A, full_matrices=False,
        )
        _, S_eig, _ = svd_via_eigh(A)

        S_svd_flat = S_svd.reshape(-1, S_svd.shape[-1])
        S_eig_flat = S_eig.reshape(-1, S_eig.shape[-1])

        for bi in range(S_svd_flat.shape[0]):
            s1 = S_svd_flat[bi]
            s2 = S_eig_flat[bi]
            K = s1.shape[0]

            eye_K = torch.eye(K, dtype=s1.dtype, device='cpu')

            # SVD gaps
            d1 = s1.unsqueeze(-1) - s1.unsqueeze(-2)
            g1 = (d1.abs() + eye_K * 1e30).min().item()
            F1 = safe_inverse_random(d1, eps) * (1 - eye_K)
            mF1 = F1.abs().max().item()

            # EIG gaps
            d2 = s2.unsqueeze(-1) - s2.unsqueeze(-2)
            g2 = (d2.abs() + eye_K * 1e30).min().item()
            F2 = safe_inverse_random(d2, eps) * (1 - eye_K)
            mF2 = F2.abs().max().item()

            batch_str = (
                f"[{bi}]" if S_svd_flat.shape[0] > 1
                else ""
            )
            print(
                f"{i:>3}{batch_str:<5} "
                f"| {g1:>14.4e} {g2:>14.4e} "
                f"| {mF1:>14.4e} {mF2:>14.4e}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
