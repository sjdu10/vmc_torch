"""Diagnose which linalg hook causes gradient explosion.

Tests 4 configurations:
  (A) No hooks      — default torch.linalg.svd + torch.linalg.qr
  (B) SVD hook only — custom size_aware_svd, default QR
  (C) QR hook only  — default SVD, custom qr_via_cholesky
  (D) Both hooks    — full setup_linalg_hooks(...)

For each: compute vmap(grad(log_amplitude)) on B=2 configs,
report grad_norm, grad_max, grad_min, NaN/Inf.

Run:
    torchrun --nproc_per_node=1 scripts/test_linalg_hooks_grad.py
"""
import autoray as ar
import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten

from vmc_torch.GPU.hamiltonian import (
    spinful_Fermi_Hubbard_square_lattice_torch,
)
from vmc_torch.GPU.models import fPEPS_Model_GPU
from vmc_torch.GPU.torch_utils import (
    qr_via_cholesky,
    qr_via_eigh,
    size_aware_svd,
)
from vmc_torch.GPU.vmc_setup import (
    initialize_walkers,
    load_or_generate_peps,
)
from vmc_torch.GPU.vmc_utils import random_initial_config


# ========== Hook registration helpers ==========

def set_svd_hook(jitter=1e-8):
    ar.register_function(
        'torch', 'linalg.svd',
        lambda x: size_aware_svd(
            x, jitter=jitter, nonuniform_diag=True,
        ),
    )


def set_qr_cholesky_hook(jitter=1e-8):
    ar.register_function(
        'torch', 'linalg.qr',
        lambda x: qr_via_cholesky(x, jitter=jitter),
    )


def set_qr_eigh_hook(jitter=1e-8):
    ar.register_function(
        'torch', 'linalg.qr',
        lambda x: qr_via_eigh(x, jitter, nonuniform_diag=True),
    )


def reset_svd_hook():
    ar.register_function(
        'torch', 'linalg.svd',
        lambda x: torch.linalg.svd(x, full_matrices=False),
    )


def reset_qr_hook():
    ar.register_function(
        'torch', 'linalg.qr',
        lambda x: torch.linalg.qr(x),
    )


def reset_all_hooks():
    reset_svd_hook()
    reset_qr_hook()


# ========== Gradient computation ==========

def compute_test_grads(model, fxs):
    """Compute vmap(grad(log_amplitude)) — mirrors VMC warmup."""
    params_pytree = list(model.params)

    def single_sample_log_amp_func(x_i, p):
        sign, log_abs = model.vamp_log(
            x_i.unsqueeze(0), p,
        )
        sign = sign.squeeze(0)
        log_abs = log_abs.squeeze(0)
        return log_abs, (sign, log_abs)

    grad_vmap_fn = torch.vmap(
        torch.func.grad(
            single_sample_log_amp_func,
            argnums=1, has_aux=True,
        ),
        in_dims=(0, None),
    )

    grads_pytree, (signs, log_abs) = grad_vmap_fn(
        fxs, params_pytree,
    )

    # Flatten to (B, Np)
    leaves, _ = tree_flatten(grads_pytree)
    flat_grads = torch.cat(
        [leaf.flatten(start_dim=1) for leaf in leaves],
        dim=1,
    ).detach()

    return flat_grads, signs.detach(), log_abs.detach()


# ========== Main ==========

def main():
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.set_default_device(device)
    torch.manual_seed(42)

    # ========== System setup (mirrors vmc_run_fpeps.py) ==========
    Lx, Ly = 8, 8
    N_sites = Lx * Ly
    t, U = 1.0, 8.0
    N_f = N_sites - 8
    n_fermions_per_spin = (N_f // 2, N_f // 2)
    D, chi = 10, 10

    H = spinful_Fermi_Hubbard_square_lattice_torch(
        Lx, Ly, t, U, N_f, pbc=False,
        n_fermions_per_spin=n_fermions_per_spin,
        no_u1_symmetry=False, gpu=True,
    )

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
        batch_size=2, seed=42, rank=0, device=device,
    )

    N_params = sum(p.numel() for p in model.parameters())
    print(
        f"System: {Lx}x{Ly} Fermi-Hubbard, "
        f"t={t}, U={U}, N_f={N_f}, D={D}, chi={chi}"
    )
    print(f"Model: {N_params} params | B=2 walkers")
    print(f"boundary opts: mode=mps, equalize_norms=1.0, "
          f"canonize=True")
    print()

    # ========== Run 4 configs ==========
    configs = [
        ("No hooks (baseline)", lambda: reset_all_hooks()),
        ("SVD only",            lambda: (reset_all_hooks(),
                                         set_svd_hook())),
        ("QR cholesky only",    lambda: (reset_all_hooks(),
                                         set_qr_cholesky_hook())),
        ("QR eigh only",        lambda: (reset_all_hooks(),
                                         set_qr_eigh_hook())),
        ("SVD + QR cholesky",   lambda: (reset_all_hooks(),
                                         set_svd_hook(),
                                         set_qr_cholesky_hook())),
        ("SVD + QR eigh",       lambda: (reset_all_hooks(),
                                         set_svd_hook(),
                                         set_qr_eigh_hook())),
    ]

    results = []
    for name, setup_fn in configs:
        setup_fn()
        print(f"--- {name} ---")
        try:
            grads, signs, log_abs = compute_test_grads(
                model, fxs,
            )
            grad_norm = torch.linalg.norm(grads).item()
            grad_max = grads.max().item()
            grad_min = grads.min().item()
            has_nan = torch.isnan(grads).any().item()
            has_inf = torch.isinf(grads).any().item()
            print(f"  grad_norm={grad_norm:.4e}, "
                  f"max={grad_max:.4e}, min={grad_min:.4e}")
            print(f"  signs={signs.tolist()}, "
                  f"log_abs={log_abs.tolist()}")
            if has_nan or has_inf:
                nan_count = torch.isnan(grads).sum().item()
                inf_count = torch.isinf(grads).sum().item()
                print(f"  WARNING: {nan_count} NaN, "
                      f"{inf_count} Inf in grads")
            results.append({
                'name': name,
                'grad_norm': grad_norm,
                'grad_max': grad_max,
                'grad_min': grad_min,
                'nan': has_nan,
                'inf': has_inf,
                'sign': signs.tolist(),
                'log_abs': log_abs.tolist(),
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'name': name,
                'grad_norm': float('nan'),
                'grad_max': float('nan'),
                'grad_min': float('nan'),
                'nan': True,
                'inf': False,
                'sign': None,
                'log_abs': None,
                'error': str(e),
            })
        model.zero_grad()
        print()

    # ========== Summary table ==========
    print("=" * 90)
    header = (
        f"{'Config':<22} | {'grad_norm':>12} | {'grad_max':>12} "
        f"| {'grad_min':>12} | {'NaN?':>5} | {'Inf?':>5}"
    )
    print(header)
    print("-" * 90)
    for r in results:
        nan_str = "Yes" if r['nan'] else "No"
        inf_str = "Yes" if r['inf'] else "No"
        print(
            f"{r['name']:<22} | {r['grad_norm']:>12.4e} "
            f"| {r['grad_max']:>12.4e} "
            f"| {r['grad_min']:>12.4e} "
            f"| {nan_str:>5} | {inf_str:>5}"
        )
    print("=" * 90)

    # ========== Interpretation ==========
    if len(results) >= 4:
        baseline_norm = results[0]['grad_norm']
        for r in results[1:]:
            ratio = r['grad_norm'] / baseline_norm if baseline_norm > 0 else float('inf')
            if ratio > 10:
                print(
                    f">> {r['name']}: grad_norm is "
                    f"{ratio:.1f}x baseline — SUSPECT"
                )
            elif r['nan'] or r['inf']:
                print(
                    f">> {r['name']}: has NaN/Inf — SUSPECT"
                )

    reset_all_hooks()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
