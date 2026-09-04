"""Exact (full Hilbert space) state-fitting helpers for SWO debugging.

These are the small-system reference implementations that every sampled
SWO estimator must reproduce. They enumerate the entire Hilbert space,
so they are only usable for toy lattices (e.g. 2x2 spinful Hubbard at
half filling -> 36 basis states), but within that regime they carry no
Monte Carlo error at all. That makes them the ground truth for the
sampled estimators in `swo_utils.py`.

Mirrors quantax/optimizer/supervised.py:Supervised_exact:
    epsilon[s] = psi[s] - psi_target[s] / <psi|psi_target>
    Omat[s,k]  = (d log psi(s)/d theta_k) * psi(s)    # = d psi/d theta
    Omean[k]   = sum_s psi(s) * Omat[s,k]             # = <psi|d psi/d theta>
    Obar[s,k]  = Omat[s,k] - psi[s] * Omean[k]
    step       = pinv(Obar) @ epsilon                 # MinSR Cholesky
    theta     <- theta - lr * step
`step` matches MinSRGPU.solve's convention: feed it to a GPU optimizer
that does ``theta <- theta - lr * step``.

Public API
----------
enumerate_hilbert_states(...)    -- enumerate every basis state
dense_amplitudes(...)            -- psi(s) on a given config tensor
                                    (= full state vector iff configs
                                    enumerate the Hilbert basis)
compute_full_log_jacobian(...)   -- d log|psi|/d theta on a given
                                    config tensor
exact_fidelity(...)              -- |<A|B>|^2 / (||A||^2 ||B||^2)
exact_supervised_sr_step(...)    -- MinSR step on the exact (Ns x Np)
                                    Obar = (J - <J>_p) * psi[:,None],
                                    epsilon = psi - psi_t/<psi|psi_t>
apply_step_inplace(...)          -- theta <- theta - lr * step
build_dense_hamiltonian(...)     -- Ns x Ns matrix from get_conn_batch_gpu
exact_energy(...)                -- <psi|H|psi> / <psi|psi>
"""
from __future__ import annotations

from typing import Tuple

import torch

from vmc_torch.GPU.vmc_utils import compute_grads_gpu


def enumerate_hilbert_states(
    hilbert,
    device: torch.device,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Enumerate every basis state in `hilbert` as quimb-format configs.

    Returns ``(Ns, N_sites)`` int tensor on `device`.
    """
    import numpy as np

    states_np = hilbert.all_states()
    if not isinstance(states_np, np.ndarray):
        states_np = np.asarray(states_np)
    return torch.as_tensor(states_np, device=device, dtype=dtype)


def dense_amplitudes(
    model,
    configs: torch.Tensor,
    batch_size: int,
    use_inference_mode: bool = True,
) -> torch.Tensor:
    """Evaluate ``psi(s) = sign(s) * exp(log|psi|(s))`` on each config.

    Returns ``(Ns,)`` float64 tensor on the same device as `configs`,
    where ``Ns = configs.shape[0]``. **Caveat**: this function does NOT
    enforce that `configs` enumerates the full Hilbert basis — it just
    evaluates the model on whatever you pass. The output equals the
    full state vector iff `configs` is the output of
    ``enumerate_hilbert_states(...)`` (i.e. ``Ns`` = Hilbert dimension).
    """
    Ns = configs.shape[0]
    device = configs.device
    psi = torch.empty(Ns, device=device, dtype=torch.float64)
    ctx = torch.inference_mode() if use_inference_mode else torch.no_grad()
    with ctx:
        for start in range(0, Ns, batch_size):
            end = min(start + batch_size, Ns)
            sign, log_abs = model.forward_log(configs[start:end])
            psi[start:end] = (
                sign.to(torch.float64) * torch.exp(log_abs.to(torch.float64))
            )
    return psi


def compute_full_log_jacobian(
    model,
    configs: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(J, sign, log_abs)`` for every state.

    ``J[s,k] = d log|psi(s)| / d theta_k`` in float64. For real psi this
    equals ``d log psi / d theta`` away from sign changes.
    """
    Ns = configs.shape[0]
    device = configs.device
    Np = sum(p.numel() for p in model.parameters())

    J = torch.empty(Ns, Np, device=device, dtype=torch.float64)
    sign = torch.empty(Ns, device=device, dtype=torch.float64)
    log_abs = torch.empty(Ns, device=device, dtype=torch.float64)

    for start in range(0, Ns, batch_size):
        end = min(start + batch_size, Ns)
        chunk = configs[start:end]
        lpg, (s_chunk, l_chunk) = compute_grads_gpu(
            chunk, model, batch_size=chunk.shape[0], use_log_amp=True,
        )
        J[start:end] = lpg.to(torch.float64).detach()
        sign[start:end] = s_chunk.to(torch.float64).detach()
        log_abs[start:end] = l_chunk.to(torch.float64).detach()

    return J, sign, log_abs


def exact_fidelity(
    psi_A: torch.Tensor, psi_B: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``F = |<A|B>|^2 / (||A||^2 ||B||^2)`` on full state vectors.

    Returns (fidelity, -log F, overlap_normalized).
    """
    nA = torch.linalg.vector_norm(psi_A)
    nB = torch.linalg.vector_norm(psi_B)
    ovlp = torch.dot(psi_A, psi_B) / (nA * nB).clamp(min=1e-300)
    fid = ovlp * ovlp
    log_f = -torch.log(fid.clamp(min=1e-300))
    return fid, log_f, ovlp


def exact_supervised_sr_step(
    psi_A: torch.Tensor,
    psi_B: torch.Tensor,
    J: torch.Tensor,
    diag_shift: float = 1e-4,
    return_diagnostics: bool = False,
):
    """Quantax-style exact MinSR step for state fitting.

    Args:
        psi_A: ``(Ns,)`` current model amplitudes (float64, unnormalized).
        psi_B: ``(Ns,)`` target amplitudes (float64, unnormalized).
        J:     ``(Ns, Np)`` ``d log|psi_A|/d theta``.
        diag_shift: Tikhonov shift on the (Ns x Ns) Gram matrix.
        return_diagnostics: also return intermediate quantities for debugging.

    Returns:
        ``(Np,)`` float64 step (Tensor) or a dict of diagnostics.
        Optimizer should apply ``theta -= lr * step``.
    """
    device = psi_A.device
    f64 = torch.float64
    psi_A = psi_A.to(f64)
    psi_B = psi_B.to(f64)
    J = J.to(f64)

    nA = torch.linalg.vector_norm(psi_A).clamp(min=1e-300)
    nB = torch.linalg.vector_norm(psi_B).clamp(min=1e-300)
    psi = psi_A / nA
    psi_t = psi_B / nB
    ovlp = torch.dot(psi, psi_t)
    ovlp_safe = torch.where(
        torch.abs(ovlp) < 1e-300,
        torch.where(
            ovlp >= 0,
            torch.ones((), device=device, dtype=f64),
            -torch.ones((), device=device, dtype=f64),
        ) * 1e-300,
        ovlp,
    )

    epsilon = psi - psi_t / ovlp_safe                            # (Ns,)
    Omat = J * psi.unsqueeze(-1)                                 # (Ns, Np)
    Omean = torch.einsum('s,sk->k', psi, Omat)                   # (Np,)
    Obar = Omat - psi.unsqueeze(-1) * Omean.unsqueeze(0)         # (Ns, Np)

    T = Obar @ Obar.T                                            # (Ns, Ns)
    T.diagonal().add_(diag_shift)
    L, info = torch.linalg.cholesky_ex(T)
    if (info == 0).all():
        alpha = torch.cholesky_solve(epsilon.unsqueeze(-1), L).squeeze(-1)
    else:
        alpha = torch.linalg.lstsq(T, epsilon).solution

    step = Obar.T @ alpha                                        # (Np,)

    if not return_diagnostics:
        return step

    fid, log_f, _ = exact_fidelity(psi_A, psi_B)
    return {
        'step': step,
        'epsilon': epsilon,
        'Obar': Obar,
        'Omean': Omean,
        'ovlp': ovlp,
        'psi_A_hat': psi,
        'psi_B_hat': psi_t,
        'fidelity': fid,
        'log_f': log_f,
    }


def apply_step_inplace(
    model,
    step: torch.Tensor,
    learning_rate: float,
) -> None:
    """In-place ``theta <- theta - lr * step`` (matches OptimizerGPU.step)."""
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            n = p.numel()
            update = step[offset:offset + n].to(p.dtype).view_as(p.data)
            p.data.sub_(learning_rate * update)
            offset += n


def build_dense_hamiltonian(
    hamiltonian,
    configs: torch.Tensor,
) -> torch.Tensor:
    """Build the (Ns, Ns) dense Hamiltonian matrix on the enumerated basis.

    Uses ``hamiltonian.get_conn_batch_gpu(configs)`` and a config-to-index
    map. Returns float64 tensor on the same device as `configs`.
    """
    device = configs.device
    Ns = configs.shape[0]

    conn_etas, conn_coeffs, batch_ids = (
        hamiltonian.get_conn_batch_gpu(configs)
    )

    cfg_cpu = configs.detach().cpu().numpy()
    cfg_to_idx = {tuple(int(x) for x in cfg_cpu[i]): i for i in range(Ns)}
    eta_cpu = conn_etas.detach().cpu().numpy()
    eta_idx = torch.tensor(
        [cfg_to_idx[tuple(int(x) for x in eta_cpu[i])]
         for i in range(eta_cpu.shape[0])],
        device=device, dtype=torch.long,
    )

    H_dense = torch.zeros(Ns, Ns, device=device, dtype=torch.float64)
    H_dense.index_put_(
        (eta_idx, batch_ids.long()),
        conn_coeffs.to(torch.float64),
        accumulate=True,
    )
    return H_dense


def exact_energy(
    H_dense: torch.Tensor,
    psi: torch.Tensor,
) -> torch.Tensor:
    """``E = <psi|H|psi> / <psi|psi>`` on the enumerated basis (real psi)."""
    psi64 = psi.to(torch.float64)
    Hpsi = H_dense @ psi64
    norm2 = torch.dot(psi64, psi64).clamp(min=1e-300)
    return torch.dot(psi64, Hpsi) / norm2
