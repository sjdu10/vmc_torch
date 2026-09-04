"""SWO (Supervised Wavefunction Optimization) helpers for the GPU
pipeline.

Given a trainable model A and a frozen target model B, sample configs
from A, freeze that dataset, then fit A to B by minimizing
``-log F`` with ``F = |<A|B>|^2 / (||A||^2 ||B||^2)``.

The whole module is organized around ONE idea. The inner loop reuses a
frozen sample set while theta moves, so the samples no longer come from
the current Born distribution. Every physical quantity is defined
w.r.t. that distribution, so every estimator is a mean reweighted by
the self-normalized importance weight ``r`` (see the long comment
below). Record ``r``, do a weighted sum -- that is the entire API.

Public API
----------
SWOBatch                  -- frozen dataset + current-theta weights
collect_swo_dataset(...)  -- MCMC + cache B's amps -> (fxs, SWOBatch)
swo_fidelity(...)         -- F, -log F, ESS, coherence from cached amps
swo_weights_p(...)        -- overlap-measure weight p
swo_fidelity_gradient(...)-- 2 ( <O>_r - <O>_p )  ==  grad(-log F)
swo_sr_terms(...)         -- (O, r, eps) for the supervised SR system
weighted_minsr_step(...)  -- distributed MinSR with per-sample weights
swo_energy(...)           -- <E_loc>_r
save_swo_checkpoint(...)  -- write model + MC stats to disk

The exact (full Hilbert space) reference implementations that these
estimators reproduce on small systems live in `vmc_torch.GPU.swo_exact`.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

from vmc_torch.GPU.vmc_utils import compute_grads_gpu

_F64 = torch.float64


# =====================================================================
# Importance weights: the single core concept
# =====================================================================
#
# Samples come from a proposal distribution q; every physical quantity
# is defined w.r.t. the Born measure p = |psi_theta|^2 / Z_theta of the
# CURRENT state. The self-normalized importance weight
#
#     r_i = (p_i / q_i) / mean_j(p_j / q_j)        (1/N) sum_i r_i = 1
#
# makes the unknown normalizations Z_theta, Z_q cancel exactly, and
# turns every estimator into a plain weighted mean
#
#     <f>_r = (1/N) sum_i r_i f_i
#
# For SWO the proposal is a FROZEN state psi_q (= psi at the start of
# the inner loop), so q ∝ |psi_q|^2 and
#
#     r_i = v_i / mean_j(v_j),   v_i = |psi_theta(c_i) / psi_q(c_i)|^2
#
# Two facts make this safe and simple:
#   * r_i >= 0 and sum_i r_i = N, hence r_i in [0, N] -- r can never
#     overflow, only underflow to 0.
#   * On-policy (theta == theta_q) gives r_i == 1 exactly, so every
#     weighted mean degenerates to the usual plain mean.
#
# `-log F = log||A||^2 - 2 log<A|B>` involves TWO measures, so there are
# two weights: r (norm measure, ∝ |psi_A|^2) and p (overlap measure,
# ∝ psi_A psi_B). With R_i = psi_B(c_i)/psi_A(c_i),
#
#     p_i = N r_i R_i / sum_j r_j R_j              (1/N) sum_i p_i = 1
#
# and then EVERY quantity is a weighted sum with no ratios left:
#
#     F            = <R>_r^2 / <R^2>_r
#     grad(-log F) = 2 ( <O>_r - <O>_p )
#     E            = <E_loc>_r
#
# `sqrt(r_i/N)` -- the form in which the weights enter the MinSR linear
# system -- appears ONLY inside the SR solver, never in an estimator
# interface.
# =====================================================================


def _chunk_slices(n: int, chunk: int):
    """Yield ``slice`` objects tiling ``range(n)`` in steps of ``chunk``."""
    for start in range(0, n, chunk):
        yield slice(start, min(start + chunk, n))


def _all_reduce(t: torch.Tensor, op) -> torch.Tensor:
    """In-place global reduction; no-op on a single rank."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=op)
    return t


def _gsum(x: torch.Tensor) -> torch.Tensor:
    """Globally reduced sum over all samples on all ranks."""
    return _all_reduce(x.sum().clone(), dist.ReduceOp.SUM)


def _gmax(x: torch.Tensor) -> torch.Tensor:
    """Globally reduced max; -inf when every rank is empty/zero."""
    local = (
        x.max() if x.numel()
        else torch.tensor(-math.inf, device=x.device, dtype=x.dtype)
    )
    return _all_reduce(local.clone(), dist.ReduceOp.MAX)


def _glse(log_x: torch.Tensor) -> torch.Tensor:
    """Global max-shifted ``log(sum_i exp(log_x_i))``.

    Two collectives (MAX then SUM). Returns -inf if every entry is -inf,
    which is the correct answer for a sum of zeros.
    """
    m = _gmax(log_x)
    if not torch.isfinite(m):          # global, so all ranks agree
        return m
    return m + torch.log(_gsum(torch.exp(log_x - m)))


def _glse_signed(
    log_x: torch.Tensor, sign: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Signed global logsumexp of ``s_i * exp(log_x_i)``.

    Returns ``(log|S|, sign(S), log(sum_i exp(log_x_i)))`` where
    ``S = sum_i s_i exp(log_x_i)``. The third value is the sum of
    magnitudes; ``exp(log|S| - log_sum_abs)`` is the coherence, i.e. how
    much cancellation the signed sum suffered.
    """
    m = _gmax(log_x)
    if not torch.isfinite(m):
        neg_inf = m
        return neg_inf, torch.ones_like(m), neg_inf
    shifted = torch.exp(log_x - m)
    pair = torch.stack([(sign * shifted).sum(), shifted.sum()])
    _all_reduce(pair, dist.ReduceOp.SUM)
    signed_sum, abs_sum = pair[0], pair[1]
    sgn = torch.where(
        signed_sum >= 0,
        torch.ones_like(signed_sum),
        -torch.ones_like(signed_sum),
    )
    log_abs_sum = m + torch.log(signed_sum.abs().clamp(min=1e-300))
    return log_abs_sum, sgn, m + torch.log(abs_sum.clamp(min=1e-300))


@dataclass
class SWOBatch:
    """A frozen SWO training set plus its current-theta reweighting state.

    The first block is fixed at collection time and never mutated. The
    second block is refreshed once per inner iteration, because ``r``
    depends on theta.

    ``log_abs_q`` is ``log|psi_q|`` for the state the MCMC actually ran
    on; the proposal density is ``q(c) ∝ |psi_q(c)|^2``. In SWO that is
    the training model frozen at the start of the inner loop. Naming it
    after the *role* (proposal) rather than the instance makes
    ``log_abs_q = 0`` -- "the proposal state has amplitude 1 everywhere",
    i.e. a uniform proposal -- a legal and meaningful use of the API.
    That is exactly how the exact full-Hilbert tests drive this class.
    """

    # --- frozen at collection ---
    configs: torch.Tensor        # (n_loc, N_sites) int64
    log_abs_q: torch.Tensor      # (n_loc,) f64
    sign_B: torch.Tensor         # (n_loc,) f64
    log_abs_B: torch.Tensor      # (n_loc,) f64
    n_total: int                 # samples summed over all ranks

    # --- refreshed per inner iteration ---
    sign_A: Optional[torch.Tensor] = None
    log_abs_A: Optional[torch.Tensor] = None
    log_r: Optional[torch.Tensor] = None
    r: Optional[torch.Tensor] = None
    log_norm: Optional[torch.Tensor] = None   # log mean_i(v_i)

    @property
    def n_local(self) -> int:
        return self.configs.shape[0]

    @property
    def device(self) -> torch.device:
        return self.configs.device

    def refresh(
        self, sign_A: torch.Tensor, log_abs_A: torch.Tensor,
    ) -> "SWOBatch":
        """Install current-theta amplitudes and recompute ``r``.

        One global logsumexp. ``log_v = 2(log|psi_A| - log|psi_q|)`` may
        span hundreds of nats for large tensor networks, so the
        normalization is done entirely in log space.
        """
        self.sign_A = sign_A.to(_F64)
        self.log_abs_A = log_abs_A.to(_F64)
        log_v = 2.0 * (self.log_abs_A - self.log_abs_q)
        self.log_norm = _glse(log_v) - math.log(self.n_total)
        self.log_r = log_v - self.log_norm
        self.r = torch.exp(self.log_r)
        return self

    def refresh_from_model(self, model, batch_size: int) -> "SWOBatch":
        """Chunked forward-only pass, then :meth:`refresh`.

        Used by the fidelity path, the line search and the energy
        measurement. The SR path gets the amplitudes for free out of its
        gradient pass and calls :meth:`refresh` directly.
        """
        signs, logs = [], []
        with torch.inference_mode():
            for sl in _chunk_slices(self.n_local, batch_size):
                s, l = model.forward_log(self.configs[sl])
                signs.append(s.detach().clone())
                logs.append(l.detach().clone())
        return self.refresh(torch.cat(signs), torch.cat(logs))

    def _require_refreshed(self) -> None:
        if self.r is None:
            raise RuntimeError(
                "SWOBatch not refreshed: call refresh(...) or "
                "refresh_from_model(...) before using any estimator."
            )

    def wmean(
        self, f: torch.Tensor, w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``(1/n_total) * sum_i w_i f_i``, reduced over all ranks.

        ``w`` defaults to ``self.r``. ``f`` may be ``(n_loc,)`` -> scalar
        or ``(n_loc, Np)`` -> ``(Np,)``. Intended for well-behaved
        integrands (log-derivatives, local energies); quantities built
        out of ``R = psi_B/psi_A`` go through the log-space helpers
        instead, because ``R`` can overflow while ``r`` underflows.
        """
        self._require_refreshed()
        if w is None:
            w = self.r
        w = w.to(_F64)
        f = f.to(_F64)
        acc = (w @ f) if f.dim() == 2 else (w * f).sum()
        return _all_reduce(acc.clone(), dist.ReduceOp.SUM) / self.n_total

    def ess_frac(self) -> torch.Tensor:
        """``ESS/N = 1 / mean_i(r_i^2) = exp(-D_2(p_theta || q))``.

        1 when on-policy; -> 0 as theta drifts away from the proposal.
        This is the reweighting-degeneracy diagnostic, i.e. the direct
        analogue of PPO's early-stopping-on-KL criterion.
        """
        self._require_refreshed()
        return self.n_total / _gsum(self.r * self.r).clamp(min=1e-300)


def swo_fidelity(batch: SWOBatch) -> Dict[str, torch.Tensor]:
    """Reweighted fidelity and the scalar diagnostics, from cached amps.

    With ``R_i = psi_B(c_i)/psi_A(c_i)``::

        F = <R>_r^2 / <R^2>_r

    Both moments are formed in log space: ``R`` alone can overflow on
    configs where ``psi_A`` is tiny, but ``r_i R_i`` is bounded because
    ``r_i`` carries a compensating ``|psi_A|^2``.

    Returns ``fidelity``, ``neg_log_f`` (the objective the SWO gradient
    actually minimizes, up to an additive constant), ``mean_R``,
    ``coherence``, ``ess_frac`` and ``ess_R``.

    On-policy this reduces exactly to the plain ``<R>^2/<R^2>``.
    """
    batch._require_refreshed()
    log_n = math.log(batch.n_total)
    d_log = batch.log_abs_B - batch.log_abs_A

    # <R>_r : signed, needs cancellation bookkeeping.
    log_rR = batch.log_r + d_log
    sgn_rR = batch.sign_A * batch.sign_B
    log_abs_sum, sgn_sum, log_sum_abs = _glse_signed(log_rR, sgn_rR)
    log_mean_R = log_abs_sum - log_n

    # <R^2>_r : strictly positive.
    log_mean_R2 = _glse(batch.log_r + 2.0 * d_log) - log_n

    neg_log_f = log_mean_R2 - 2.0 * log_mean_R
    fidelity = torch.exp(-neg_log_f)
    # |sum r R| / sum r|R| in (0, 1]. Its reciprocal is the factor by
    # which sign cancellation in <A|B> amplifies gradient noise -- the
    # principled replacement for clamping a near-zero denominator.
    coherence = torch.exp(log_abs_sum - log_sum_abs)

    return {
        'fidelity': fidelity,
        'neg_log_f': neg_log_f,
        'mean_R': sgn_sum * torch.exp(log_mean_R),
        'log_abs_mean_R': log_mean_R,
        'sign_mean_R': sgn_sum,
        'coherence': coherence,
        'ess_frac': batch.ess_frac(),
        'ess_R': fidelity,          # ESS of the overlap estimator / N
    }


def swo_weights_p(batch: SWOBatch) -> torch.Tensor:
    """Overlap-measure weight ``p_i = N r_i R_i / sum_j r_j R_j``.

    Signed, and self-normalized the same way as ``r``:
    ``(1/N) sum_i p_i = 1``. Built in log space so that neither an
    overflowing ``R`` nor an underflowing ``r`` can produce ``0 * inf``.
    """
    batch._require_refreshed()
    d_log = batch.log_abs_B - batch.log_abs_A
    log_rR = batch.log_r + d_log
    sgn_rR = batch.sign_A * batch.sign_B
    log_abs_sum, sgn_sum, _ = _glse_signed(log_rR, sgn_rR)
    log_p = log_rR - log_abs_sum + math.log(batch.n_total)
    return (sgn_rR * sgn_sum) * torch.exp(log_p)


def swo_fidelity_gradient(
    batch: SWOBatch,
    model,
    *,
    grad_batch_size: int,
    Np: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """``grad_theta(-log F) = 2 ( <O>_r - <O>_p )``.

    Two weighted means of the same log-derivative matrix, one under the
    norm measure and one under the overlap measure. No ratio of Monte
    Carlo estimators appears, so there is no near-zero denominator to
    guard -- the ill-conditioning that used to live in ``l1n / l1d``
    shows up instead as a small ``coherence`` in the returned stats.

    ``O`` is streamed in ``grad_batch_size`` chunks and never stored;
    the two accumulators are packed into a single all_reduce. The
    amplitudes returned by ``compute_grads_gpu`` are deliberately
    ignored -- ``batch`` already holds the amplitudes that ``r`` and
    ``p`` were built from, so weights and derivatives are guaranteed to
    come from the same forward pass.

    Returns ``(direction, stats)`` where ``stats`` is
    :func:`swo_fidelity` on the same batch, i.e. the value of the very
    objective this direction descends.
    """
    batch._require_refreshed()
    p = swo_weights_p(batch)

    acc = torch.zeros(2, Np, device=device, dtype=_F64)
    for sl in _chunk_slices(batch.n_local, grad_batch_size):
        chunk = batch.configs[sl]
        lpg, _ = compute_grads_gpu(
            chunk, model, batch_size=chunk.shape[0], use_log_amp=True,
        )
        lpg = lpg.to(_F64).detach()
        acc[0] += batch.r[sl] @ lpg
        acc[1] += p[sl] @ lpg
        del lpg

    _all_reduce(acc, dist.ReduceOp.SUM)
    direction = 2.0 * (acc[0] - acc[1]) / batch.n_total
    return direction, swo_fidelity(batch)


# ---------------------------------------------------------------------
# Supervised SR / MinSR with importance weights
# ---------------------------------------------------------------------

# Largest exponent whose exp() is finite in float64 (exp(709.78) = 1.8e308).
_EXP_MAX = 700.0


def swo_sr_terms(
    batch: SWOBatch,
    model,
    *,
    grad_batch_size: int,
    ratio_clip: Optional[float] = None,
) -> Dict:
    """Build the weighted supervised-SR system on the cached dataset.

    The residual is the Monte Carlo image of quantax's
    ``epsilon = psi_hat - phi_hat / <psi_hat|phi_hat>``::

        eps_i = 1 - R_i / <R>_r

    It needs no extra centering: ``sum_i r_i eps_i = 0`` holds exactly by
    construction, which is precisely the statement that the residual is
    orthogonal to the current state (so no step is wasted changing the
    norm). The ``sqrt(r_i/N)`` scaling that turns ``(O, eps)`` into
    ``(Obar, epsbar)`` lives in :func:`weighted_minsr_step`.

    One gradient pass, no extra forward: the amplitudes that come back
    from ``compute_grads_gpu`` are the ones used to refresh ``batch``,
    so ``r`` and ``O`` provably come from the same theta.

    Overflow: ``R_i/<R>_r`` can exceed float64 range on configs where
    ``psi_A`` is tiny. Such rows are *dropped* (``r_i = 0``), which
    removes them from the Gram exactly, and counted in ``n_over``.
    ``ratio_clip`` additionally bounds ``|eps|``, i.e. the opt-in
    outlier knob, applied on top of the principled drop.
    """
    lpg_chunks, sign_chunks, log_chunks = [], [], []
    for sl in _chunk_slices(batch.n_local, grad_batch_size):
        chunk = batch.configs[sl]
        lpg, (s_A, l_A) = compute_grads_gpu(
            chunk, model, batch_size=chunk.shape[0], use_log_amp=True,
        )
        lpg_chunks.append(lpg.to(_F64).detach())
        sign_chunks.append(s_A.to(_F64).detach())
        log_chunks.append(l_A.to(_F64).detach())

    O_loc = torch.cat(lpg_chunks, dim=0)
    del lpg_chunks
    batch.refresh(torch.cat(sign_chunks), torch.cat(log_chunks))
    stats = swo_fidelity(batch)

    # R_i / <R>_r, entirely in log space.
    log_q = (
        (batch.log_abs_B - batch.log_abs_A) - stats['log_abs_mean_R']
    )
    sgn_q = (batch.sign_A * batch.sign_B) * stats['sign_mean_R']

    # Branch-free on purpose: `over` is a per-rank mask, so gating on
    # `over.any()` would make ranks take different paths. Everything
    # here is local elementwise work, so just always do it.
    over = log_q > _EXP_MAX
    n_over = _gsum(over.to(_F64))
    r = torch.where(over, torch.zeros_like(batch.r), batch.r)
    log_q = torch.where(over, torch.zeros_like(log_q), log_q)

    eps = 1.0 - sgn_q * torch.exp(log_q)
    eps = torch.where(over, torch.zeros_like(eps), eps)
    if ratio_clip is not None:
        eps = eps.clamp(-float(ratio_clip), float(ratio_clip))

    stats = dict(stats)
    stats['n_over'] = n_over
    return {'O_loc': O_loc, 'r': r, 'eps': eps, 'stats': stats}


def weighted_minsr_step(
    O_loc: torch.Tensor,
    eps: torch.Tensor,
    r: Optional[torch.Tensor],
    n_total: int,
    Np: int,
    *,
    rshift: float,
    ashift: float,
    device: torch.device,
) -> Tuple[torch.Tensor, float, int]:
    r"""Distributed MinSR with per-sample importance weights.

    .. math::
        \bar O_i = \sqrt{r_i/N}\,(O_i - \langle O\rangle_r), \qquad
        \bar\epsilon_i = \sqrt{r_i/N}\,\epsilon_i

    and ``step = Obar^T (Obar Obar^T + shift I)^{-1} epsbar``, the
    ``N_s``-form of Rende et al. 2024 Eq. 17.

    The weights cannot be folded in by the caller: centering is
    idempotent, so ``O - mean(O)`` can never be turned back into
    ``O - <O>_r`` from outside. Hence this mirrors the distributed
    machinery of ``MinSRGPU._solve`` (all_to_all column redistribution,
    Gram all_reduce, Cholesky) with weighted centering instead.

    ``r=None`` selects uniform weights, reproducing plain on-policy
    MinSR; that path is asserted equal to ``MinSRGPU`` in the tests.

    Rows with ``r_i = 0`` drop out of the Gram *exactly*, which is the
    correct way to mask a sample (zeroing ``O_i`` before a plain
    centering would instead leave ``-mean(O)/sqrt(N)`` behind).

    Returns ``(dp, elapsed_seconds, cholesky_info)`` with ``dp`` of
    shape ``(Np,)``, matching ``PreconditionerGPU.solve``'s contract so
    a GPU optimizer can apply ``theta <- theta - lr * dp``.
    """
    import time

    from vmc_torch.GPU.optimizer import _two_term_shift

    t0 = time.time()
    world_size = (
        dist.get_world_size()
        if dist.is_initialized() else 1
    )
    O_loc = O_loc.to(device=device, dtype=_F64).contiguous()
    eps = eps.to(device=device, dtype=_F64)
    n_local = eps.shape[0]

    if r is None:
        w = torch.ones(n_local, device=device, dtype=_F64)
    else:
        w = r.to(device=device, dtype=_F64)

    # <O>_r, then the sqrt(r/N) row scaling.
    O_mean = _all_reduce((w @ O_loc).clone(), dist.ReduceOp.SUM) / n_total
    scale = torch.sqrt(w / n_total)
    O_loc.sub_(O_mean.unsqueeze(0)).mul_(scale.unsqueeze(-1))
    eps_scaled = eps * scale

    if world_size > 1:
        eps_bar = torch.empty(n_total, device=device, dtype=_F64)
        dist.all_gather_into_tensor(eps_bar, eps_scaled)

        np_per_rank = (Np + world_size - 1) // world_size
        np_pad = world_size * np_per_rank
        if np_pad != Np:
            O_padded = torch.zeros(
                (n_local, np_pad), device=device, dtype=_F64,
            )
            O_padded[:, :Np].copy_(O_loc)
            del O_loc
        else:
            O_padded = O_loc
        send_buf = (
            O_padded
            .view(n_local, world_size, np_per_rank)
            .permute(1, 0, 2)
            .contiguous()
        )
        del O_padded
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf)
        del send_buf
        O_bar = recv_buf.reshape(world_size * n_local, np_per_rank)
        del recv_buf
        T = O_bar @ O_bar.T
        dist.all_reduce(T, op=dist.ReduceOp.SUM)
    else:
        np_pad = np_per_rank = Np
        eps_bar = eps_scaled
        O_bar = O_loc
        T = O_bar @ O_bar.T

    shift = _two_term_shift(T.trace().item(), T.shape[0], rshift, ashift)
    T.diagonal().add_(shift)
    L, info = torch.linalg.cholesky_ex(T)
    if bool((info == 0).all()):
        alpha = torch.cholesky_solve(eps_bar.unsqueeze(-1), L).squeeze(-1)
    else:
        alpha = torch.linalg.lstsq(T, eps_bar).solution
    del T

    dp_local = O_bar.T @ alpha
    if world_size > 1:
        del O_bar
        dp_padded = torch.empty(np_pad, device=device, dtype=_F64)
        dist.all_gather_into_tensor(dp_padded, dp_local)
        dp = dp_padded[:Np].contiguous()
    else:
        dp = dp_local

    return dp, time.time() - t0, int(info.max())


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
    *,
    burn_in: bool = True,
    burn_in_steps: int = 0,
    use_export_compile: bool = False,
) -> Tuple[torch.Tensor, SWOBatch]:
    """Run MCMC on model_A and cache everything theta-independent.

    Returns ``(fxs, batch)`` with the walker state and a
    :class:`SWOBatch` already refreshed at the sampling theta, so
    ``batch.r == 1`` exactly on return.

    ``burn_in`` defaults to True and should stay that way: after an
    inner loop of K updates the walkers are still equilibrated to the
    *old* Born distribution, and the collection loop only performs
    ``ceil(ns_per_rank / batch_size)`` sweeps -- often just one. Without
    re-equilibration the proposal is not ``|psi_A|^2`` at all, and no
    importance weight can repair that: ``r`` corrects for a known
    proposal, not for an unconverged chain.
    """
    B = fxs.shape[0]
    if burn_in and burn_in_steps > 0:
        fxs = sampler.burn_in(
            fxs, model_A, graph, burn_in_steps,
            compile=use_export_compile, use_log_amp=True,
        )

    cfgs, sA, lA, sB, lB = [], [], [], [], []
    count = 0
    while count < ns_per_rank:
        fxs, (sign_A, log_abs_A) = sampler.step(
            fxs, model_A, graph,
            compile=use_export_compile, use_log_amp=True,
        )
        with torch.inference_mode():
            sign_B, log_abs_B = model_B.forward_log(fxs)

        take = min(B, ns_per_rank - count)
        cfgs.append(fxs[:take].clone())
        sA.append(sign_A[:take].detach().clone())
        lA.append(log_abs_A[:take].detach().clone())
        sB.append(sign_B[:take].detach().clone())
        lB.append(log_abs_B[:take].detach().clone())
        count += take

    world_size = (
        dist.get_world_size() if dist.is_initialized() else 1
    )
    sign_A_all = torch.cat(sA).to(_F64)
    log_abs_A_all = torch.cat(lA).to(_F64)
    batch = SWOBatch(
        configs=torch.cat(cfgs, dim=0),
        log_abs_q=log_abs_A_all.clone(),
        sign_B=torch.cat(sB).to(_F64),
        log_abs_B=torch.cat(lB).to(_F64),
        n_total=ns_per_rank * world_size,
    ).refresh(sign_A_all, log_abs_A_all)
    return fxs, batch


def swo_energy(
    batch: SWOBatch,
    model,
    hamiltonian,
    evaluate_energy_fn,
    *,
    batch_size: int,
) -> torch.Tensor:
    """``E = <E_loc>_r`` on the cached configs.

    ``batch`` must be refreshed at the theta you want the energy of --
    typically *after* the inner loop has moved it. Taking a plain mean
    here (as the pre-refactor code did) measures the energy in the stale
    sampling measure, which is a pure measurement bias.
    """
    batch._require_refreshed()
    e_chunks = []
    with torch.inference_mode():
        for sl in _chunk_slices(batch.n_local, batch_size):
            cfg = batch.configs[sl]
            amps_out = model.forward_log(cfg)
            _, local_E = evaluate_energy_fn(
                cfg, model, hamiltonian, amps_out, use_log_amp=True,
            )
            e_chunks.append(local_E.detach().to(_F64))
    return batch.wmean(torch.cat(e_chunks))


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
