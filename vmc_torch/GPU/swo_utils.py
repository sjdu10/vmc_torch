"""SWO (Supervised Wavefunction Optimization) helpers for the GPU
pipeline.

Mirrors the CPU `run_SWO_state_fitting` (vmc_torch/VMC.py:506).
Given a trainable model A and a frozen target model B, sample configs
from A, then fit A to B by minimizing -log|<A|B>|^2 / (||A||^2 ||B||^2)
on the fixed sample set. All math is done in log-amp space for
numerical stability on large tensor networks.

Public API
----------
collect_swo_dataset(...)         -- sample configs + cache amps of A0/B
fidelity_from_log_amps(...)      -- one-shot fidelity estimator from
                                    cached log-amps (multi-rank aware)
accumulate_fidelity_terms(...)   -- chunked loss/grad accumulation,
                                    fully reduced across ranks
compute_swo_direction(...)       -- combine reduced terms into a flat
                                    (Np,) update direction + -log F
accumulate_supervised_sr_terms(...)
                                  -- quantax-style ratio residuals and
                                     log-derivative matrix for SR/MinSR
save_swo_checkpoint(...)         -- write model + MC stats to disk

Exact (full Hilbert) state fitting helpers — for small-system debugging,
mirrors quantax/optimizer/supervised.py:Supervised_exact:

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

Math (log-amp form)
-------------------
Samples c are drawn from |psi_A_init|^2 (the training model at the
start of each outer step). Let
    s_X(c) = sign_X(c),  l_X(c) = log|psi_X(c)|,
    X in {A_init, A_cur, B}.

The CPU loss is L_emp(theta) = log h_emp - 2 log g_emp with
    g_emp = sum_c psi_A_cur(c) * psi_B(c) / |psi_A_init(c)|^2
    h_emp = sum_c |psi_A_cur(c)|^2          / |psi_A_init(c)|^2
The 1/|psi_A_init|^2 factor reweights the MC samples back to a
"flat" distribution so g_emp ~ <A_cur|B> and h_emp ~ ||A_cur||^2;
hence L_emp ~ -log F + const, and grad descent on L_emp moves A
toward B. (See vmc_torch/VMC.py:506-699 for the CPU reference.)

Per-sample weights (with lpg = d log|psi_A|/d theta):
    u(c) = s_A_cur s_B exp(l_A_cur + l_B - 2 l_A_init)   (= weight 1)
    v(c) = exp(2 (l_A_cur - l_A_init))                    (= weight 2)

Sums:
    L1n = -sum_c [u(c) * lpg(c)]
    L1d =  sum_c [u(c)]
    L2n =  sum_c [v(c) * lpg(c)]
    L2d =  sum_c [v(c)]
    direction = 2 * Re[ L1n / L1d + L2n / L2d ]
              = grad_theta L_emp

The diagnostic fidelity reported each iter mirrors CPU: it forms
ratios psi_B/psi_A_cur (NOT importance-reweighted) and reports
f1^2/f2 from those. This is biased once theta drifts from theta_init,
and matches what CPU prints; resample to refresh.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

from vmc_torch.GPU.vmc_utils import compute_grads_gpu


# ---------------------------------------------------------------------
# Dataset collection
# ---------------------------------------------------------------------

def collect_swo_dataset(
    sampler,
    fxs: torch.Tensor,
    model_A,
    model_B,
    graph,
    ns_per_rank: int,
    burn_in: bool = False,
    burn_in_steps: int = 0,
    use_export_compile: bool = False,
    ratio: float = 0.0,
) -> Tuple[
    torch.Tensor,        # fxs (last walker state)
    torch.Tensor,        # configs (Ns_local, N_sites)
    torch.Tensor,        # sign_B (Ns_local,)
    torch.Tensor,        # log_abs_B (Ns_local,)
    torch.Tensor,        # sign_A_init (Ns_local,)
    torch.Tensor,        # log_abs_A_init (Ns_local,)
]:
    """Run MCMC and cache (config, sign_A, log_abs_A, sign_B, log_abs_B)
    tuples until ns_per_rank samples are collected.

    By default all samples are drawn from model_A. If ``ratio > 0``,
    approximately ``ratio * ns_per_rank`` samples are drawn from
    model_B instead. This mixed dataset is useful when A has tiny
    overlap with B and A-only sampling misses important target
    configurations.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")

    B = fxs.shape[0]
    n_from_B = int(round(ns_per_rank * ratio))
    n_from_B = min(max(n_from_B, 0), ns_per_rank)
    n_from_A = ns_per_rank - n_from_B

    fxs_A = fxs
    fxs_B = fxs.clone()
    if burn_in and burn_in_steps > 0:
        if n_from_A > 0:
            fxs_A = sampler.burn_in(
                fxs_A, model_A, graph, burn_in_steps,
                compile=use_export_compile, use_log_amp=True,
            )
        if n_from_B > 0:
            fxs_B = sampler.burn_in(
                fxs_B, model_B, graph, burn_in_steps,
                compile=use_export_compile, use_log_amp=True,
            )

    cfgs_chunks = []
    sA_chunks, lA_chunks = [], []
    sB_chunks, lB_chunks = [], []

    def append_samples(fxs_chain, sample_model, eval_other_model, n_samples):
        count = 0
        while count < n_samples:
            fxs_chain, (sign_sample, log_abs_sample) = sampler.step(
                fxs_chain, sample_model, graph,
                compile=use_export_compile, use_log_amp=True,
            )
            with torch.inference_mode():
                sign_other, log_abs_other = eval_other_model.forward_log(
                    fxs_chain,
                )

            take = min(B, n_samples - count)
            cfgs_chunks.append(fxs_chain[:take].clone())
            if sample_model is model_A:
                sA_chunks.append(sign_sample[:take].detach().clone())
                lA_chunks.append(log_abs_sample[:take].detach().clone())
                sB_chunks.append(sign_other[:take].detach().clone())
                lB_chunks.append(log_abs_other[:take].detach().clone())
            else:
                sA_chunks.append(sign_other[:take].detach().clone())
                lA_chunks.append(log_abs_other[:take].detach().clone())
                sB_chunks.append(sign_sample[:take].detach().clone())
                lB_chunks.append(log_abs_sample[:take].detach().clone())
            count += take
        return fxs_chain

    if n_from_A > 0:
        fxs_A = append_samples(fxs_A, model_A, model_B, n_from_A)
    if n_from_B > 0:
        fxs_B = append_samples(fxs_B, model_B, model_A, n_from_B)

    return (
        fxs_A,
        torch.cat(cfgs_chunks, dim=0),
        torch.cat(sB_chunks, dim=0),
        torch.cat(lB_chunks, dim=0),
        torch.cat(sA_chunks, dim=0),
        torch.cat(lA_chunks, dim=0),
    )


# ---------------------------------------------------------------------
# Fidelity helpers
# ---------------------------------------------------------------------

def _chunk_slices(n: int, chunk: int):
    for start in range(0, n, chunk):
        yield slice(start, min(start + chunk, n))


def _maybe_all_reduce(tensor: torch.Tensor) -> None:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def fidelity_from_log_amps(
    sign_A: torch.Tensor, log_abs_A: torch.Tensor,
    sign_B: torch.Tensor, log_abs_B: torch.Tensor,
    n_total: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-shot fidelity estimator on cached log-amplitudes.

    Computes f1, f2 (locally), all_reduces them, and returns
    (fidelity, -log fidelity) as float64 scalars on `sign_A.device`.
    """
    f64 = torch.float64
    r = (log_abs_A - log_abs_B).to(f64)
    s = (sign_A * sign_B).to(f64)
    ba = s * torch.exp(-r)
    # Pack f1 and f2 into a 2-tensor so we hit one collective.
    pair = torch.stack([ba.sum(), (ba * ba).sum()])
    _maybe_all_reduce(pair)
    f1 = pair[0] / n_total
    f2 = pair[1] / n_total
    fidelity = (f1 * f1) / f2
    log_f = -torch.log(fidelity.clamp(min=1e-300))
    return fidelity, log_f


def fidelity_stats_from_log_amps(
    sign_A: torch.Tensor,
    log_abs_A: torch.Tensor,
    sign_B: torch.Tensor,
    log_abs_B: torch.Tensor,
    n_total: int,
) -> Dict[str, torch.Tensor]:
    """Fidelity estimator from cached log-amps.

    Samples are drawn from |A|^2. With R = psi_B / psi_A, the sampled
    fidelity is F = <R>^2 / <R^2>.
    """
    f64 = torch.float64
    ratio = (sign_A * sign_B).to(f64) * torch.exp(
        (log_abs_B - log_abs_A).to(f64),
    )
    pair = torch.stack([ratio.sum(), (ratio * ratio).sum()])
    _maybe_all_reduce(pair)
    mean_r = pair[0] / n_total
    mean_r2 = pair[1] / n_total
    fidelity = (mean_r * mean_r) / mean_r2.clamp(min=1e-300)
    log_f = -torch.log(fidelity.clamp(min=1e-300))
    return {'fidelity': fidelity, 'log_f': log_f}


def fidelity_from_model_on_configs(
    configs: torch.Tensor,
    training_model,
    sign_B: torch.Tensor,
    log_abs_B: torch.Tensor,
    n_total: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the sampled fidelity estimator for the current model.

    This is intended for fixed-sample diagnostics/line-search after a
    parameter update. It recomputes A's current log-amplitudes on the
    provided configs and then applies ``fidelity_from_log_amps``.
    """
    sign_A_chunks = []
    log_abs_A_chunks = []
    with torch.inference_mode():
        for sl in _chunk_slices(configs.shape[0], batch_size):
            sign_A, log_abs_A = training_model.forward_log(configs[sl])
            sign_A_chunks.append(sign_A.detach())
            log_abs_A_chunks.append(log_abs_A.detach())

    return fidelity_from_log_amps(
        torch.cat(sign_A_chunks, dim=0),
        torch.cat(log_abs_A_chunks, dim=0),
        sign_B,
        log_abs_B,
        n_total,
    )


def accumulate_fidelity_terms(
    configs: torch.Tensor,
    log_abs_A_init_f64: torch.Tensor,
    sign_B_f64: torch.Tensor,
    log_abs_B_f64: torch.Tensor,
    training_model,
    grad_batch_size: int,
    Np: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Walk the cached dataset in grad_batch_size chunks; on each
    chunk compute (lpg_A, sign_A_cur, log_abs_A_cur) via
    compute_grads_gpu(use_log_amp=True), then accumulate the four
    fidelity-gradient sums (L1n, L1d, L2n, L2d) and the two diagnostic
    fidelity sums (F1, F2).

    All accumulators are float64. After accumulation, all 6 quantities
    are coalesced into a single flat buffer and all-reduced in ONE
    NCCL call (saves 5 syncs per inner iter on multi-GPU).

    `log_abs_A_init_f64` is the importance-reweighting denominator —
    samples were drawn from |psi_A_init|^2 so we divide weights by
    |psi_A_init|^2 to estimate <A_cur|B> and ||A_cur||^2. Sign of
    psi_A_init does NOT enter (only |psi_A_init|^2).

    `sign_B_f64`, `log_abs_B_f64`, `log_abs_A_init_f64` MUST already
    be float64 (cast once outside — see run_SWO_state_fitting_gpu).

    Returns dict with rank-reduced sums (across all ranks):
        {'l1n','l1d','l2n','l2d','f1','f2'}
    """
    Ns = configs.shape[0]
    f64 = torch.float64

    l1n = torch.zeros(Np, device=device, dtype=f64)
    l2n = torch.zeros(Np, device=device, dtype=f64)
    scalars = torch.zeros(4, device=device, dtype=f64)  # [l1d, l2d, f1, f2]

    for sl in _chunk_slices(Ns, grad_batch_size):
        cfg_chunk = configs[sl]
        chunk_size = cfg_chunk.shape[0]
        lpg, (s_A, l_A) = compute_grads_gpu(
            cfg_chunk, training_model,
            batch_size=chunk_size, use_log_amp=True,
        )
        lpg = lpg.to(f64)
        l_A = l_A.to(f64)
        s_A = s_A.to(f64)

        l_A_init_sl = log_abs_A_init_f64[sl]
        log_b_sl = log_abs_B_f64[sl]
        s_b_sl = sign_B_f64[sl]

        # u(c) = sign_A_cur * sign_B * exp(l_A_cur + l_B - 2 l_A_init)
        u = (s_A * s_b_sl) * torch.exp(
            l_A + log_b_sl - 2.0 * l_A_init_sl,
        )
        # v(c) = exp(2 (l_A_cur - l_A_init))
        v = torch.exp(2.0 * (l_A - l_A_init_sl))
        # Diagnostic fidelity (mirrors CPU inner-loop fidelity):
        # ba = psi_B / psi_A_cur with no importance reweighting.
        ba = (s_A * s_b_sl) * torch.exp(log_b_sl - l_A)

        l1n -= (u.unsqueeze(-1) * lpg).sum(dim=0)
        l2n += (v.unsqueeze(-1) * lpg).sum(dim=0)
        scalars[0] += u.sum()
        scalars[1] += v.sum()
        scalars[2] += ba.sum()
        scalars[3] += (ba * ba).sum()

    # Coalesce all reductions: pack [l1n, l2n, scalars] into one buffer,
    # all_reduce once, then unpack. Saves 5 NCCL launches per inner iter.
    if dist.is_initialized() and dist.get_world_size() > 1:
        flat = torch.empty(
            2 * Np + 4, device=device, dtype=f64,
        )
        flat[:Np] = l1n
        flat[Np:2 * Np] = l2n
        flat[2 * Np:] = scalars
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        l1n = flat[:Np].clone()
        l2n = flat[Np:2 * Np].clone()
        scalars = flat[2 * Np:].clone()

    return {
        'l1n': l1n, 'l2n': l2n,
        'l1d': scalars[0], 'l2d': scalars[1],
        'f1': scalars[2], 'f2': scalars[3],
    }


def compute_swo_direction(
    terms: Dict[str, torch.Tensor],
    n_total: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine reduced fidelity terms into the flat (Np,) update
    direction and the scalar -log F.

    Args:
        terms: dict from accumulate_fidelity_terms (already reduced).
        n_total: total number of samples across all ranks.

    Returns:
        direction: (Np,) float64 tensor, the supervised loss gradient.
        log_f: scalar float64, -log(F).
    """
    f1 = terms['f1'] / n_total
    f2 = terms['f2'] / n_total
    fidelity = (f1 * f1) / f2
    log_f = -torch.log(fidelity.clamp(min=1e-300))

    direction = 2.0 * (
        terms['l1n'] / terms['l1d']
        + terms['l2n'] / terms['l2d']
    )
    return direction, log_f


# ---------------------------------------------------------------------
# Quantax-style supervised SR helpers
# ---------------------------------------------------------------------

def accumulate_supervised_sr_terms(
    configs: torch.Tensor,
    sign_B_f64: torch.Tensor,
    log_abs_B_f64: torch.Tensor,
    training_model,
    grad_batch_size: int,
    device: torch.device,
    n_total: int,
    ratio_clip: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Build the sampled linear system for quantax-style state fitting.

    Samples are assumed to be drawn from the current training state A.
    For each sampled configuration, compute

        ratio(s) = psi_B(s) / psi_A(s)
        signal(s) = -ratio(s) / <ratio>

    The MinSR solver later centers ``signal`` internally, producing

        -(ratio / <ratio> - 1) / sqrt(N)

    as quantax's supervised ``Ebar`` residual. If ``ratio_clip`` is
    set, clip the centered residual ``ratio / <ratio> - 1`` before
    sending it to the SR solve.
    """
    f64 = torch.float64
    eps = 1e-300

    lpg_chunks = []
    ratio_chunks = []
    pair = torch.zeros(2, device=device, dtype=f64)  # [sum ratio, sum ratio^2]

    for sl in _chunk_slices(configs.shape[0], grad_batch_size):
        cfg_chunk = configs[sl]
        lpg, (s_A, l_A) = compute_grads_gpu(
            cfg_chunk, training_model,
            batch_size=cfg_chunk.shape[0], use_log_amp=True,
        )
        lpg = lpg.to(f64).detach()
        ratio = ((s_A.to(f64) * sign_B_f64[sl]) * torch.exp(
            log_abs_B_f64[sl] - l_A.to(f64),
        )).detach()

        lpg_chunks.append(lpg)
        ratio_chunks.append(ratio)
        pair[0] += ratio.sum()
        pair[1] += (ratio * ratio).sum()

    _maybe_all_reduce(pair)
    ratio_mean = pair[0] / n_total
    ratio2_mean = pair[1] / n_total

    # Avoid div-by-zero when <ratio> underflows; preserve sign so the
    # downstream `signal = -ratio / <ratio>` keeps the right direction.
    sign = torch.where(
        ratio_mean >= 0,
        ratio_mean.new_tensor(1.0),
        ratio_mean.new_tensor(-1.0),
    )
    ratio_mean_safe = torch.where(torch.abs(ratio_mean) < eps, sign * eps, ratio_mean)

    local_lpg = torch.cat(lpg_chunks, dim=0)
    local_ratio = torch.cat(ratio_chunks, dim=0)
    
    print(f'Before reduction: ratio_mean = {local_ratio.mean():.3e}, ratio_std = {local_ratio.std():.3e}, ratio_max = {local_ratio.abs().max():.3e}')
    
    if ratio_clip is None:
        local_signal = -local_ratio / ratio_mean_safe
        signal_mean = -ratio_mean / ratio_mean_safe
    else:
        local_ratio = (local_ratio / ratio_mean_safe - 1.0)
        print(f"Before clipping: ratio_std = {local_ratio.std():.3e}, ratio_max = {local_ratio.abs().max():.3e}")
        local_ratio = torch.clip(local_ratio, min=-float(ratio_clip), max=float(ratio_clip))
        local_signal = -local_ratio
        signal_mean = torch.zeros((), device=device, dtype=f64)

    # Sampled fidelity F ≈ <ratio>^2 / <ratio^2>. ESS = n_total * F
    # is the effective sample size of the <ratio> estimator; when
    # ESS ~ O(1), the SR signal is dominated by 1-2 outlier samples
    # and the resulting direction is noise-driven.
    fidelity = (ratio_mean * ratio_mean) / ratio2_mean.clip(min=eps)
    log_f = -torch.log(fidelity.clip(min=eps))
    ess = n_total * fidelity

    return {
        'local_lpg': local_lpg,
        'local_signal': local_signal,
        'signal_mean': signal_mean,
        'log_f': log_f,
        'ess': ess,
    }


# ---------------------------------------------------------------------
# Checkpointing (mirrors CPU run_SWO_state_fitting save block)
# ---------------------------------------------------------------------

def save_swo_checkpoint(
    model,
    mc_stats: Dict,
    tmpdir: str,
    t_step: int,
    optimizer=None,
) -> None:
    """Write model state + MC stats for outer-step `t_step`.

    Files:
        {tmpdir}/model_params_step{t_step}.pth
        {tmpdir}/swo_state_fitting.json     (overwritten each step)
    """
    os.makedirs(tmpdir, exist_ok=True)

    combined: Dict = {
        'model_state_dict': model.state_dict(),
        'MC_stats': mc_stats,
    }
    # Capture optimizer state if exposed (mirrors CPU SWO behavior:
    # peek at known momentum/Adam attrs). OptimizerGPU does not yet
    # implement state_dict(), so we sniff attributes instead.
    if optimizer is not None:
        opt_state = {}
        if hasattr(optimizer, 'state_dict'):
            opt_state['state_dict'] = optimizer.state_dict()
        for attr in ('m', 'v', 't', 'velocity'):
            if hasattr(optimizer, attr):
                opt_state[attr] = getattr(optimizer, attr)
        if opt_state:
            opt_state['optimizer_class'] = type(optimizer).__name__
            combined['optimizer_state'] = opt_state

    torch.save(
        combined,
        os.path.join(tmpdir, f'model_params_step{t_step}.pth'),
    )
    with open(
        os.path.join(tmpdir, 'swo_state_fitting.json'), 'w',
    ) as f:
        json.dump(mc_stats, f, indent=4)


# ---------------------------------------------------------------------
# Exact (full Hilbert) state fitting — for small-system debugging.
# Mirrors quantax/optimizer/supervised.py:Supervised_exact:
#     epsilon[s] = psi[s] - psi_target[s] / <psi|psi_target>
#     Omat[s,k]  = (d log psi(s)/d theta_k) * psi(s)    # = d psi/d theta
#     Omean[k]   = sum_s psi(s) * Omat[s,k]             # = <psi|d psi/d theta>
#     Obar[s,k]  = Omat[s,k] - psi[s] * Omean[k]
#     step       = pinv(Obar) @ epsilon                 # MinSR Cholesky
#     theta     <- theta - lr * step
# `step` matches MinSRGPU.solve's convention: feed it to a GPU optimizer
# that does ``theta <- theta - lr * step``.
# ---------------------------------------------------------------------

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
