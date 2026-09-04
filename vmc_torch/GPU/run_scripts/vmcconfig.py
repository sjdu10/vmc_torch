"""Shared VMC configuration and helpers for GPU run scripts.

Base VMCConfig contains all common hyperparameters with sensible
defaults.  Run scripts import it and override fields as needed:

    # Simple override:
    cfg = VMCConfig(batch_size=256, use_log_amp=True)

    # Extend with script-specific fields:
    @dataclass
    class ReuseCfg(VMCConfig):
        use_export_compile_reuse: bool = False
        use_cheap_grad: bool = True

Helper functions (make_preconditioner, load_checkpoint, etc.)
reduce boilerplate in run scripts while keeping full flexibility
— every helper is optional and can be replaced with inline code.
"""
import json
import os
from dataclasses import dataclass, fields
from typing import Optional

# Persist compiled kernel cache to home dir so it survives WSL2/tmp wipes.
# setdefault: user can still override by setting the env var before launch.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.expanduser("~/.cache/torchinductor"),
)

import torch
from vmc_torch.GPU.optimizer import (
    AdamSRIterGPU,
    AdamSRMinSRGPU,
    IterSRGPU,
    MARCHIterGPU,
    MARCHMinSRGPU,
    MinSRGPU,
    SPRINGIterGPU,
    SPRINGMinSRGPU,
    TrivialPreconditionerGPU,
)

# Valid values for ``VMCConfig.preconditioner``.
_VALID_PRECONDITIONERS = ('sr', 'spring', 'march', 'adamsr')


@dataclass
class VMCWarmupConfig:
    """One-shot warmup config: which phase(s) to dry-run before
    the main VMC loop (compile / cache / smoke-test).
    """

    use_export_compile: bool = False
    grad_batch_size: int = 64
    verbose: bool = True
    run_sampling: bool = True
    run_locE: bool = True
    run_grad: bool = True
    use_log_amp: bool = False

    def __init__(self, **kwargs):
        for f in fields(self):
            setattr(self, f.name, kwargs.pop(f.name, f.default))
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class VMCConfig:
    """VMC numerical / training settings.

    Groups:
        Sampling:    batch_size, ns_per_rank, grad_batch_size,
                     burn_in_steps
        Training:    vmc_steps, learning_rate, lr_scheduler
        Preconditioner: preconditioner, sr_iter_solver,
                     sr_rshift, sr_ashift, sr_iter_rtol,
                     sr_maxiter, minres_sr_use_scipy, spring_mu,
                     march_mu/beta, adamsr_mu/beta/norm_clip
        Compile:     use_export_compile, use_log_amp
        Gradient:    offload_grad_to_cpu
        Checkpoint:  save_every, resume_step
        Debug:       debug, verbose
        Log-variance: logvar_gamma, logvar_lr, logvar_weight_decay
    """

    # ----- Sampling -----
    batch_size: int = 1024
    ns_per_rank: int = 1024
    grad_batch_size: int = 1024
    burn_in_steps: int = 0

    # ----- Training -----
    vmc_steps: int = 100
    learning_rate: float = 0.1
    lr_scheduler: object = None

    # ----- Preconditioner -----
    # ``preconditioner`` selects the gradient-update scheme:
    #   None      -> pure SGD on the energy gradient (no SR solve).
    #   'sr'      -> plain Stochastic Reconfiguration.
    #   'spring'  -> SPRING (arXiv:2401.10190), SR + Kaczmarz
    #                one-iteration momentum.
    #   'march'   -> MARCH (arXiv:2507.02644), SR + Adam-like
    #                first/second-moment momentum.
    #   'adamsr'  -> AdamSR, Adam-style moments on top of SR.
    # ``sr_iter_solver`` picks the linear-system solver shape:
    #   True  -> iterative Np-form MINRES on the SR matrix.
    #   False -> direct Ns-form MinSR (Cholesky/eigh on Ns x Ns).
    # All four (sr/spring/march/adamsr) support both shapes.
    preconditioner: Optional[str] = None
    sr_iter_solver: bool = True
    # Two-term Tikhonov regularization of the SR Gram matrix:
    #   shift = sr_rshift * trace(T)/sqrt(n) + sr_ashift
    # sr_rshift scales with the Gram magnitude (relative term);
    # sr_ashift is an absolute floor. rshift=0 recovers the old
    # pure-absolute behavior (sr_ashift = old sr_diag_shift).
    sr_rshift: float = 0.0
    sr_ashift: float = 1e-4
    # If True, the Ns-form preconditioners (spring/march/adamsr at
    # sr_iter_solver=False) compute the Ns x Ns Gram matmul in fp32
    # and cast back to fp64. Big speedup on consumer GPUs where
    # fp64 is slow; on A100/H100 leave False to keep full fp64.
    # No effect when sr_iter_solver=True (Np-form doesn't form
    # a Gram).
    sr_mixed_precision: bool = False

    # ----- Iterative (Np-form MINRES) options -----
    sr_iter_rtol: float = 1e-4
    sr_maxiter: int = 100
    minres_sr_use_scipy: bool = False

    # ----- SPRING -----
    # Setting mu=0 recovers plain MINRES / MinSR exactly.
    # ``norm_constraint`` enforces the paper's Eq. 37 Euclidean
    # bound ||d_theta|| <= sqrt(C) via the SGD optimizer; leave
    # None to disable.
    spring_mu: float = 0.99
    norm_constraint: Optional[float] = None

    # ----- MARCH (arXiv:2507.02644) -----
    march_mu: float = 0.95
    march_beta: float = 0.995

    # ----- AdamSR -----
    adamsr_mu: float = 0.95
    adamsr_beta: float = 0.995
    adamsr_norm_clip: Optional[float] = None
    # Gram solver for the dense MinSR form (regularization comes
    # from the global sr_rshift / sr_ashift above):
    #   'direct': solve of T + shift*I.
    #   'pinv_eig': eigh pinv with smooth relative cutoff
    #     rtol = sr_rshift (sr_ashift ignored).
    adamsr_solver: str = 'direct'

    # ----- Compile -----
    use_export_compile: bool = False

    # ----- Log-amp -----
    use_log_amp: bool = True

    # ----- Gradient -----
    offload_grad_to_cpu: bool = False

    # ----- Checkpoint -----
    save_every: int = 50
    resume_step: int = 0

    # ----- Debug / UI -----
    debug: bool = False
    verbose: bool = False
    # Per-step diagnostic logging: prints distribution statistics of
    # local energies, log|psi|, and the SR gradient (RHS of S*dp=g).
    # Useful for catching outlier-driven blowups in early VMC steps.
    # Independent of `debug` (which is noisier and per-iteration);
    # `diagnostics` adds one extra forward per step (cheap).
    diagnostics: bool = False
    show_progress: bool = True

    # Outlier mask on raw local energies, applied ONLY inside the
    # SR solve (PreconditionerGPU._mask_outlier_samples): samples
    # with ``|E_loc| > local_E_clip`` (absolute threshold) have
    # their E_loc entry and O_loc row zeroed before the solve, and
    # the clean mean / masked count are recorded in
    # ``_last_mask_info`` for the ``[SR clip]`` report. The
    # reported step energy_mean / energy_var are NOT affected.
    # Typical value: ~1e3 (units of t). Default None disables.
    local_E_clip: Optional[float] = None

    # ----- Log-variance (arXiv:2603.15853) -----
    logvar_gamma: float = 1e-3
    logvar_lr: float = 1e-3
    logvar_weight_decay: float = 0.0

    def __init__(self, **kwargs):
        # Set every declared dataclass field from kwargs or its default,
        # then attach any extra kwargs as ad-hoc attributes. This lets
        # callers do e.g. VMCConfig(batch_size=256, my_extra_flag=True)
        # without subclassing.
        for f in fields(self):
            setattr(self, f.name, kwargs.pop(f.name, f.default))
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__post_init__()

    def __repr__(self):
        # Include declared dataclass fields *and* any extra attributes
        # attached via __init__ kwargs, so dynamic attributes show up
        # when the config is printed.
        declared = [f.name for f in fields(self)]
        extras = [k for k in self.__dict__ if k not in declared]
        parts = [f"{k}={getattr(self, k)!r}" for k in declared + extras]
        return f"{type(self).__name__}({', '.join(parts)})"

    def __post_init__(self):
        if (self.preconditioner is not None
                and self.preconditioner not in _VALID_PRECONDITIONERS):
            raise ValueError(f"preconditioner must be one of "
                             f"{_VALID_PRECONDITIONERS} or None, got "
                             f"{self.preconditioner!r}")


# ============================================================
# Helper functions for run scripts
# ============================================================


def make_torch_optimizer(model, cfg):
    """Create torch.optim.AdamW for log-variance path."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.logvar_lr,
        weight_decay=cfg.logvar_weight_decay,
    )


def make_preconditioner(cfg):
    """Dispatch a preconditioner from ``cfg.preconditioner`` and
    ``cfg.sr_iter_solver``.

    Returns ``None`` for ``preconditioner=None`` (pure SGD on the
    energy gradient).  ``sr_iter_solver`` flips between the
    iterative Np-form MINRES (``True``) and the direct Ns-form
    MinSR (``False``).  All four preconditioners support both.

    Scripts that need a non-standard preconditioner can skip this
    helper and instantiate one directly.

    Sets ``local_E_clip`` on the returned preconditioner from
    ``cfg.local_E_clip`` (None disables; otherwise samples with
    ``|E_loc| > local_E_clip`` are entirely masked from the SR
    direction — both E_loc and the corresponding O_loc row are
    zeroed, the clean E_mean / Ns_clean are recomputed). The outer
    VMC reported energy is NOT affected.
    """
    def _attach_clip(pre):
        pre.local_E_clip = getattr(cfg, 'local_E_clip', None)
        return pre

    name = getattr(cfg, 'preconditioner', None)
    if name is None:
        return _attach_clip(TrivialPreconditionerGPU())

    iter_solver = bool(getattr(cfg, 'sr_iter_solver', False))

    if name == 'sr':
        if iter_solver:
            return _attach_clip(IterSRGPU(
                rtol=cfg.sr_iter_rtol,
                maxiter=cfg.sr_maxiter,
                use_scipy=cfg.minres_sr_use_scipy,
            ))
        return _attach_clip(MinSRGPU())

    mp = bool(getattr(cfg, 'sr_mixed_precision', False))

    if name == 'spring':
        if iter_solver:
            return _attach_clip(SPRINGIterGPU(
                mu=cfg.spring_mu,
                rtol=cfg.sr_iter_rtol,
                maxiter=cfg.sr_maxiter,
            ))
        return _attach_clip(SPRINGMinSRGPU(mu=cfg.spring_mu, mixed_precision=mp))

    if name == 'march':
        if iter_solver:
            return _attach_clip(MARCHIterGPU(
                mu=cfg.march_mu,
                beta=cfg.march_beta,
                rtol=cfg.sr_iter_rtol,
                maxiter=cfg.sr_maxiter,
            ))
        return _attach_clip(MARCHMinSRGPU(
            mu=cfg.march_mu,
            beta=cfg.march_beta,
            mixed_precision=mp,
        ))

    if name == 'adamsr':
        if iter_solver:
            return _attach_clip(AdamSRIterGPU(
                mu=cfg.adamsr_mu,
                beta=cfg.adamsr_beta,
                norm_clip=cfg.adamsr_norm_clip,
                rtol=cfg.sr_iter_rtol,
                maxiter=cfg.sr_maxiter,
            ))
        return _attach_clip(AdamSRMinSRGPU(
            mu=cfg.adamsr_mu,
            beta=cfg.adamsr_beta,
            norm_clip=cfg.adamsr_norm_clip,
            mixed_precision=mp,
            solver=getattr(cfg, 'adamsr_solver', 'direct'),
        ))

    raise ValueError(
        f"Unknown preconditioner {name!r}; expected one of "
        f"{_VALID_PRECONDITIONERS} or None"
    )


def load_checkpoint(
    model, output_dir, model_name,
    resume_step, device, rank,
    world_size=1, batch_size=None,
    optimizer=None, preconditioner=None,
):
    """Load model checkpoint if resume_step > 0.

    Handles both old format (bare state_dict) and new
    format (dict with 'model_state_dict' + 'fxs' keys).

    Walker restore policy (only when ``batch_size`` is given):
      * The saved ``fxs`` field is the all-gather'd walker
        tensor of shape ``(world_size_saved * batch_size_saved,
        N_sites)`` taken at the end of the saved step.
      * If saved global walker count == ``world_size *
        batch_size``, slice it by rank and return that rank's
        chunk. This gives a perfectly continuous resume: the
        first post-resume energy estimate uses the same walker
        distribution that produced the pre-resume one.
      * If sizes mismatch (different batch size or rank count
        from the saved run), print a warning and return None.
        Caller should fall back to random walker init plus a
        longer burn-in.
      * If the checkpoint is the old bare-state_dict format and
        thus has no walkers, return None.

    Returns:
        local_fxs: (batch_size, N_sites) int64 tensor on
            ``device`` when walkers were successfully restored,
            else None.
    """
    if resume_step <= 0:
        return None
    ckpt_path = os.path.join(
        output_dir,
        f'checkpoint_{model_name}_{resume_step}.pt',
    )
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=True,
    )
    if (
        isinstance(ckpt, dict)
        and 'model_state_dict' in ckpt
    ):
        model.load_state_dict(ckpt['model_state_dict'])
        saved_fxs = ckpt.get('fxs', None)
    else:
        model.load_state_dict(ckpt)
        saved_fxs = None
    if rank == 0:
        print(f"Loaded checkpoint: {ckpt_path}")

    # Auto-restore optimizer / preconditioner state if present.
    # Stateless objects declare ``_STATE_ATTRS = ()`` so they never
    # produce a checkpoint key in the first place; this branch is a
    # no-op for them. Run scripts can pass these unconditionally.
    for label, obj in (
        ('optimizer_state', optimizer),
        ('preconditioner_state', preconditioner),
    ):
        if (
            obj is not None
            and isinstance(ckpt, dict)
            and label in ckpt
            and hasattr(obj, 'load_state_dict')
        ):
            obj.load_state_dict(ckpt[label])
            if rank == 0:
                print(
                    f"Restored {label} from checkpoint "
                    f"({type(obj).__name__})."
                )

    if saved_fxs is None or batch_size is None:
        return None

    n_new_total = world_size * batch_size
    n_saved_total = saved_fxs.shape[0]
    if n_new_total != n_saved_total:
        if rank == 0:
            print(
                f"[load_checkpoint] WARNING: saved walker "
                f"count={n_saved_total} but requested "
                f"world_size*batch_size={n_new_total}. "
                f"Walker state cannot be restored exactly; "
                f"falling back to random walker init. "
                f"Recommend increasing burn_in_steps for this "
                f"run."
            )
        return None

    start = rank * batch_size
    local_fxs = saved_fxs[start:start + batch_size].to(device)
    if rank == 0:
        print(
            f"Restored walker state from checkpoint "
            f"(walker count = {n_saved_total})."
        )
    return local_fxs


def load_phase1_nn_into_phase2(
    phase2_model, phase1_ckpt_path, device='cpu',
):
    """Transfer NN params from a ``train_tn=False`` phase-1 checkpoint
    into a freshly-built ``train_tn=True`` phase-2 model.

    Use case: multi-phase training of an NN-fTNS model where phase 1
    trains the NN backflow alone (TN frozen at zero, absent from the
    SR/QGT system) and phase 2 unfreezes TN to jointly optimize both.
    Because the two phases use different ``self.params`` layouts
    (phase 1: NN only; phase 2: ``[TN..., NN...]`` with TN in front),
    a plain ``load_state_dict`` fails — the key indices don't match.
    This helper re-indexes the NN keys and leaves TN at zero.

    Key remap::

        phase 1 'params.i'    ->  phase 2 'params.{n_ftn + i}'   (NN)
        phase 1 '_ftn_buf_i'  ->  (discarded; phase 2 TN starts at 0)

    Phase-2 model construction reminder::

        model = <Same NNfTNS class as phase 1>(
            tn=tn, ...,                # all structural args identical
            train_tn=True,             # the only change
            set_tn_to_zeros=True,      # TN init at 0 (matches phase 1 end)
        )

    After this helper returns:
      * Caller MUST build a fresh optimizer over ``model.params`` —
        the optimizer's param shape / count is different from phase 1
        so phase-1 optimizer state cannot be reused.
      * If using the compiled forward path, call
        ``model.export_and_compile(...)`` with a NEW ``cache_dir``
        (the phase-1 cached graph was built for the NN-only param
        count and would silently break).

    Args:
        phase2_model: freshly-constructed phase-2 model with
            ``train_tn=True``.
        phase1_ckpt_path: filesystem path to a phase-1 checkpoint
            (either bare ``state_dict`` or the new dict format with
            a ``'model_state_dict'`` key).
        device: where to map the loaded tensors. Pass the device
            ``phase2_model`` lives on.

    Raises:
        AssertionError: if ``phase2_model.train_tn`` is False, or if
            the NN param count in the checkpoint doesn't match the
            phase-2 model (signals an architecture mismatch).
    """
    assert phase2_model.train_tn, (
        "phase 2 model must be constructed with train_tn=True"
    )

    ckpt = torch.load(
        phase1_ckpt_path, map_location=device, weights_only=True,
    )
    sd_p1 = (
        ckpt['model_state_dict']
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt
        else ckpt
    )

    n_ftn = phase2_model.n_ftn
    sd_p2 = phase2_model.state_dict()  # target (TN slots already at 0)

    # Sort phase-1 'params.i' keys by integer index so the remap is
    # deterministic regardless of dict iteration order.
    nn_keys_p1 = sorted(
        (k for k in sd_p1 if k.startswith('params.')),
        key=lambda k: int(k.split('.')[1]),
    )
    expected_nn = len(
        [k for k in sd_p2 if k.startswith('params.')]
    ) - n_ftn
    assert len(nn_keys_p1) == expected_nn, (
        f"NN param count mismatch: phase-1 checkpoint has "
        f"{len(nn_keys_p1)} NN params, phase-2 model expects "
        f"{expected_nn}. Architectures differ?"
    )

    for i, k in enumerate(nn_keys_p1):
        sd_p2[f'params.{n_ftn + i}'] = sd_p1[k]

    phase2_model.load_state_dict(sd_p2, strict=True)
    print(
        f"Loaded {len(nn_keys_p1)} NN params from {phase1_ckpt_path}; "
        f"TN params left at zero."
    )


def make_stats(system_str, n_params, ns_per_rank, world_size):
    """Create stats tracking dict for energy history."""
    return {
        'system': system_str,
        'Np': n_params,
        'sample size': ns_per_rank * world_size,
        'mean': [],
        'error': [],
        'variance': [],
    }


def make_stats_file(
    output_dir, model_name, resume_step, suffix='',
):
    """Build stats JSON file path.

    Args:
        suffix: optional suffix before step tag,
            e.g. '_reuse'.
    """
    step_tag = (
        f'_from_{resume_step}'
    )
    return os.path.join(
        output_dir,
        f'stats_{model_name}{suffix}{step_tag}.json',
    )


def make_on_step_end(
    rank, stats, stats_file, output_dir,
    model_name, model, save_every,
    optimizer=None, preconditioner=None,
):
    """Create on_step_end callback for the VMC loop.

    The returned callback:
    - Appends energy stats and writes JSON (rank 0 only)
    - Saves checkpoint with model state_dict + fxs
      every ``save_every`` steps (rank 0 only)
    - If ``optimizer`` / ``preconditioner`` is supplied AND its
      ``state_dict()`` is non-empty, that snapshot is also written
      under ``optimizer_state`` / ``preconditioner_state``. Stateless
      objects (e.g. plain SGD) return an empty dict and are skipped
      automatically — the run script never needs to know whether
      state exists.
    """
    def on_step_end(info):
        if rank != 0:
            return
        stats['mean'].append(info['energy_per_site'])
        stats['error'].append(info['error_per_site'])
        stats['variance'].append(info['energy_var'])
        # Per-step diagnostics (E_loc / log|psi| / SR rhs spread).
        # Populated only when VMCConfig.diagnostics is True; the
        # corresponding history lists grow on the same schedule as
        # the energy lists so they stay step-aligned.
        diag = info.get('diagnostics')
        if diag:
            stats.setdefault('diagnostics', []).append(diag)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)

        step = info['step']
        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(
                output_dir,
                f'checkpoint_{model_name}_{step + 1}.pt',
            )
            ckpt = {
                'model_state_dict': model.state_dict(),
                'fxs': info['fxs'].cpu(),
            }
            for label, obj in (
                ('optimizer_state', optimizer),
                ('preconditioner_state', preconditioner),
            ):
                if obj is None or not hasattr(obj, 'state_dict'):
                    continue
                sd = obj.state_dict()
                if sd:
                    ckpt[label] = sd
            torch.save(ckpt, ckpt_path)

    return on_step_end


def print_summary(
    rank, energy_history, system_str, stats_file=None,
):
    """Print VMC run summary (rank 0 only)."""
    if rank != 0 or not energy_history:
        return
    print(f"\n{'=' * 50}")
    print(f"Result: {system_str}")
    print(f"{'=' * 50}")
    print(f"First E/site: {energy_history[0]:.6f}")
    print(f"Last  E/site: {energy_history[-1]:.6f}")
    print(f"Min   E/site: {min(energy_history):.6f}")
    if stats_file:
        print(f"Stats saved to: {stats_file}")
    if energy_history[-1] < energy_history[0]:
        print("\nEnergy decreased.")
    else:
        print("\nWARNING: Energy did NOT decrease.")
