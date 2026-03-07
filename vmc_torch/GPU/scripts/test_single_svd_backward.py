"""Compare per-SVD backward: PyTorch native vs RobustSVD_EIG.

Uses real boundary MPS matrices. Tests whether single SVD backward
is well-behaved (ratio ~1) to isolate per-SVD vs chain compounding.

Run:
    torchrun --nproc_per_node=1 scripts/test_single_svd_backward.py
"""
import autoray as ar
import torch
import torch.distributed as dist

from vmc_torch.GPU.models import fPEPS_Model_GPU
from vmc_torch.GPU.torch_utils import RobustSVD_EIG
from vmc_torch.GPU.vmc_setup import (
    load_or_generate_peps,
    initialize_walkers,
)
from vmc_torch.GPU.vmc_utils import random_initial_config


def main():
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    dist.init_process_group("nccl")
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.manual_seed(42)

    Lx, Ly = 8, 8
    N_sites = Lx * Ly
    t, U_ = 1.0, 8.0
    N_f = N_sites - 8
    D, chi = 10, 10

    fpeps_base = (
        '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
        f'vmc_torch/GPU/data/{Lx}x{Ly}/t={t}_U={U_}'
        f'/N={N_f}/Z2/D={D}/'
    )
    peps = load_or_generate_peps(
        Lx, Ly, t, U_, N_f, D,
        seed=42, dtype=dtype,
        file_path=fpeps_base, scale_factor=4,
    )
    model = fPEPS_Model_GPU(
        tn=peps, max_bond=chi, dtype=dtype,
        contract_boundary_opts={
            'mode': 'mps', 'equalize_norms': 1.0,
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
    _cap = []

    def _capture(x):
        _cap.append(x.detach().cpu().clone())
        return torch.linalg.svd(x, full_matrices=False)

    ar.register_function('torch', 'linalg.svd', _capture)
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
    print(f"Captured {len(_cap)} matrices\n")

    k = chi
    print(
        f"{'idx':>3} {'shape':>14} "
        f"| {'||dA_pt||':>12} {'NaN?':>5} "
        f"| {'||dA_eig||':>12} {'NaN?':>5} "
        f"| {'ratio':>10}"
    )
    print("-" * 80)

    for idx in range(len(_cap)):
        A = _cap[idx]  # on CPU

        # (A) PyTorch native SVD backward
        A_pt = A.clone().requires_grad_(True)
        U, S, Vh = torch.linalg.svd(
            A_pt, full_matrices=False,
        )
        loss = (
            U[..., :, :k]
            @ torch.diag_embed(S[..., :k])
            @ Vh[..., :k, :]
        ).sum()
        loss.backward()
        dA_pt = A_pt.grad

        # (B) RobustSVD_EIG (jitter=1e-8, nonuniform_diag)
        A_eig = A.clone().requires_grad_(True)
        U2, S2, Vh2 = RobustSVD_EIG.apply(
            A_eig, 1e-8, None, True,
        )
        loss2 = (
            U2[..., :, :k]
            @ torch.diag_embed(S2[..., :k])
            @ Vh2[..., :k, :]
        ).sum()
        loss2.backward()
        dA_eig = A_eig.grad

        nan_pt = torch.isnan(dA_pt).any().item()
        nan_eig = torch.isnan(dA_eig).any().item()
        norm_pt = dA_pt.norm().item()
        norm_eig = dA_eig.norm().item()

        if nan_pt:
            ratio_str = "N/A"
        else:
            ratio_str = f"{norm_eig / max(norm_pt, 1e-30):.4e}"

        shape_str = 'x'.join(str(d) for d in A.shape)
        print(
            f"{idx:>3} {shape_str:>14} "
            f"| {norm_pt:>12.4e} "
            f"{'Yes' if nan_pt else 'No':>5} "
            f"| {norm_eig:>12.4e} "
            f"{'Yes' if nan_eig else 'No':>5} "
            f"| {ratio_str:>10}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
