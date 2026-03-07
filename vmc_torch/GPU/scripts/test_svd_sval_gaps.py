"""Diagnose SVD backward: are near-degenerate singular values
causing gradient explosion via large F = 1/(s_i - s_j)?

Two-phase approach:
  Phase 1: Forward-only pass (no vmap/grad) to collect all SVD
           singular values and analyze gap structure.
  Phase 2: Forward+backward with instrumented RobustSVD_EIG that
           logs ||dA|| from each backward call.

Run:
    torchrun --nproc_per_node=1 scripts/test_svd_sval_gaps.py
"""
import autoray as ar
import torch
import torch.distributed as dist

from vmc_torch.GPU.hamiltonian import (
    spinful_Fermi_Hubbard_square_lattice_torch,
)
from vmc_torch.GPU.models import fPEPS_Model_GPU
from vmc_torch.GPU.torch_utils import (
    safe_inverse_random,
    size_aware_svd,
)
from vmc_torch.GPU.vmc_setup import (
    initialize_walkers,
    load_or_generate_peps,
)
from vmc_torch.GPU.vmc_utils import random_initial_config

# Global list to capture SVD info during forward-only pass
_svd_log = []


def _logging_svd_no_vmap(x, jitter=1e-8, nonuniform_diag=True):
    """SVD hook that logs singular values.

    Only safe to use OUTSIDE vmap/func.grad context.
    """
    U, S, Vh = size_aware_svd(
        x, jitter=jitter, nonuniform_diag=nonuniform_diag,
    )
    _svd_log.append({
        'shape': tuple(x.shape),
        'S': S.detach().cpu().clone(),
        'n': min(x.shape[-2], x.shape[-1]),
        'path': (
            'EIG' if (x.is_cuda
                      and min(x.shape[-2], x.shape[-1]) > 32)
            else 'SVD'
        ),
    })
    return U, S, Vh


def analyze_sval_gaps(svd_log):
    """Analyze singular value gaps and F/G matrix norms."""
    epsilon = 1e-12  # same as in backward

    print(f"\n{'='*110}")
    print(f"SVD Singular Value Gap Analysis "
          f"({len(svd_log)} SVD calls)")
    print(f"{'='*110}")
    print(
        f"{'#':>4} {'shape':>16} {'path':>5} {'K':>4} "
        f"| {'min_gap':>12} {'max|F|':>12} {'max|G|':>12} "
        f"| {'||F||':>12} {'||G||':>12} "
        f"| {'s_max':>10} {'s_min':>10}"
    )
    print("-" * 110)

    worst_entries = []

    for i, entry in enumerate(svd_log):
        S_raw = entry['S']
        if S_raw is None:
            continue
        # Flatten batch dims: analyze each sval vector separately
        # S could be (K,) or (B, K) or (B1, B2, K)
        S_flat = S_raw.reshape(-1, S_raw.shape[-1])

        for bi in range(S_flat.shape[0]):
            S = S_flat[bi]
            K = S.shape[0]

            diff = S.unsqueeze(-1) - S.unsqueeze(-2)
            eye_K = torch.eye(
                K, dtype=S.dtype, device='cpu',
            )
            off_diag = diff * (1 - eye_K)

            gaps = off_diag.abs()
            # Replace zeros on diagonal with inf
            gaps = gaps + eye_K * 1e30
            min_gap = gaps.min().item()

            F = safe_inverse_random(diff, epsilon=epsilon)
            F = F * (1 - eye_K)
            summ = S.unsqueeze(-1) + S.unsqueeze(-2)
            G = safe_inverse_random(summ, epsilon=epsilon)
            G = G * (1 - eye_K)

            max_F = F.abs().max().item()
            max_G = G.abs().max().item()
            norm_F = torch.linalg.norm(F).item()
            norm_G = torch.linalg.norm(G).item()

            s_max = S.max().item()
            s_min = S.min().item()

            shape_str = 'x'.join(
                str(d) for d in entry['shape']
            )
            batch_str = (
                f"[{bi}]" if S_flat.shape[0] > 1 else ""
            )

            flag = ""
            if max_F > 1e6:
                flag = " <<<< DANGER"
            elif max_F > 1e4:
                flag = " << WARNING"

            print(
                f"{i:>4}{batch_str:<4} "
                f"{shape_str:>16} {entry['path']:>5} "
                f"{K:>4} "
                f"| {min_gap:>12.4e} {max_F:>12.4e} "
                f"{max_G:>12.4e} "
                f"| {norm_F:>12.4e} {norm_G:>12.4e} "
                f"| {s_max:>10.4e} {s_min:>10.4e}"
                f"{flag}"
            )

            worst_entries.append({
                'idx': i,
                'batch': bi,
                'shape': entry['shape'],
                'path': entry['path'],
                'K': K,
                'min_gap': min_gap,
                'max_F': max_F,
                'norm_F': norm_F,
                'max_G': max_G,
                'norm_G': norm_G,
                's_min': s_min,
                's_max': s_max,
                'S': S,
            })

    # Top 5 worst by max|F|
    worst_entries.sort(key=lambda e: e['max_F'], reverse=True)
    print(f"\n--- Top 5 worst SVD calls by max|F| ---")
    for e in worst_entries[:5]:
        print(
            f"  SVD #{e['idx']}: shape={e['shape']}, "
            f"path={e['path']}, K={e['K']}, "
            f"min_gap={e['min_gap']:.4e}, "
            f"max|F|={e['max_F']:.4e}, "
            f"||F||={e['norm_F']:.4e}"
        )
        S = e['S']
        s_vals = S.tolist()
        if len(s_vals) <= 20:
            print(f"    S = {[f'{v:.6e}' for v in s_vals]}")
        else:
            print(f"    S (first 10) = "
                  f"{[f'{v:.6e}' for v in s_vals[:10]]}")
            print(f"    S (last  10) = "
                  f"{[f'{v:.6e}' for v in s_vals[-10:]]}")

        # Show closest pairs
        K = len(s_vals)
        pairs = []
        for a in range(K):
            for b in range(a + 1, K):
                gap = abs(s_vals[a] - s_vals[b])
                pairs.append((a, b, gap))
        pairs.sort(key=lambda p: p[2])
        print(f"    Closest 5 pairs (i, j, |s_i-s_j|, F):")
        for a, b, gap in pairs[:5]:
            f_val = safe_inverse_random(
                torch.tensor(s_vals[a] - s_vals[b]),
                epsilon=1e-12,
            ).item()
            print(
                f"      ({a:>2}, {b:>2}): gap={gap:.4e}, "
                f"s[{a}]={s_vals[a]:.6e}, "
                f"s[{b}]={s_vals[b]:.6e}, "
                f"F={f_val:.4e}"
            )

    # Overall summary
    all_max_F = [e['max_F'] for e in worst_entries]
    n_danger = sum(1 for f in all_max_F if f > 1e6)
    n_warn = sum(1 for f in all_max_F if 1e4 < f <= 1e6)
    print(f"\n--- Summary ---")
    print(f"  Total SVD calls: {len(svd_log)}")
    print(f"  DANGER (max|F| > 1e6): {n_danger}")
    print(f"  WARNING (max|F| > 1e4): {n_warn}")
    print(
        f"  Safe (max|F| <= 1e4): "
        f"{len(svd_log) - n_danger - n_warn}"
    )
    print(
        f"  Theoretical max of safe_inverse(eps=1e-12): "
        f"1/(2*sqrt(1e-12)) = {1/(2*1e-6):.2e}"
    )


def main():
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.set_default_device(device)
    torch.manual_seed(42)

    # ========== System setup ==========
    Lx, Ly = 8, 8
    N_sites = Lx * Ly
    t, U = 1.0, 8.0
    N_f = N_sites - 8
    n_fermions_per_spin = (N_f // 2, N_f // 2)
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

    # =====================================================
    # Phase 1: Forward-only pass to collect SVD svals
    # =====================================================
    print("\n=== Phase 1: Forward-only SVD logging ===")
    _svd_log.clear()
    ar.register_function(
        'torch', 'linalg.svd',
        lambda x: _logging_svd_no_vmap(x),
    )

    # Single forward pass outside vmap/grad context
    import quimb as qu
    params_list = list(model.params)
    params = qu.utils.tree_unflatten(
        params_list, model.params_pytree,
    )
    with torch.no_grad():
        sign, log_abs = model.log_amplitude(fxs[0], params)
    print(f"Forward result: sign={sign.item():.1f}, "
          f"log_abs={log_abs.item():.4f}")

    analyze_sval_gaps(_svd_log)

    # =====================================================
    # Phase 2: Compare gradient with different epsilon
    # =====================================================
    print("\n\n=== Phase 2: Gradient sensitivity to epsilon ===")
    print("Testing: how does changing the backward epsilon "
          "in safe_inverse affect gradients?")

    from torch.utils._pytree import tree_flatten

    # Reset to default SVD (no hooks)
    ar.register_function(
        'torch', 'linalg.svd',
        lambda x: torch.linalg.svd(x, full_matrices=False),
    )

    # Compute baseline gradient (no hooks)
    params_pytree = list(model.params)

    def single_fwd(x_i, p):
        sign, log_abs = model.vamp_log(
            x_i.unsqueeze(0), p,
        )
        return log_abs.squeeze(0), sign.squeeze(0)

    grad_fn = torch.vmap(
        torch.func.grad(
            single_fwd, argnums=1, has_aux=True,
        ),
        in_dims=(0, None),
    )

    grads_base, _ = grad_fn(fxs, params_pytree)
    leaves, _ = tree_flatten(grads_base)
    flat_base = torch.cat(
        [l.flatten(start_dim=1) for l in leaves], dim=1,
    ).detach()
    base_norm = torch.linalg.norm(flat_base).item()
    print(f"\nBaseline (no hooks): grad_norm = {base_norm:.4e}")

    # Now test SVD hook with default epsilon=1e-12
    ar.register_function(
        'torch', 'linalg.svd',
        lambda x: size_aware_svd(
            x, jitter=1e-8, nonuniform_diag=True,
        ),
    )
    grads_hook, _ = grad_fn(fxs, params_pytree)
    leaves, _ = tree_flatten(grads_hook)
    flat_hook = torch.cat(
        [l.flatten(start_dim=1) for l in leaves], dim=1,
    ).detach()
    hook_norm = torch.linalg.norm(flat_hook).item()
    print(f"SVD hook (eps=1e-12): grad_norm = {hook_norm:.4e} "
          f"(ratio: {hook_norm/base_norm:.2e})")

    # Reset hooks
    ar.register_function(
        'torch', 'linalg.svd',
        lambda x: torch.linalg.svd(x, full_matrices=False),
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
