from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, fields
from typing import (  # noqa: UP035
    TYPE_CHECKING, Any, Callable, Dict, Optional,
)

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from vmc_torch.GPU.optimizer import (
    OptimizerGPU,
    PreconditionerGPU,
    TrivialPreconditionerGPU,
)
from vmc_torch.GPU.sampler import SamplerGPU
from vmc_torch.GPU.swo_utils import (
    SWOBatch,
    collect_swo_dataset,
    save_swo_checkpoint,
    swo_energy,
    swo_fidelity,
    swo_fidelity_gradient,
    swo_sr_terms,
    weighted_minsr_step,
)
from vmc_torch.GPU.vmc_utils import (
    check_NaN_or_inf,
    compute_grads_gpu,
    evaluate_energy,
    evaluate_energy_grad,
)

if TYPE_CHECKING:
    # Annotation-only imports: keep the heavy models package and the
    # quimb-backed Hamiltonian module out of the runtime import graph.
    from vmc_torch.GPU.models._base import WavefunctionModel_GPU
    from vmc_torch.hamiltonian_torch import Graph, Hamiltonian


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_distributed(
    cuda_rank: Optional[int] = None, cpu: bool = False,
) -> tuple[int, int, torch.device]:
    """Initialise ``torch.distributed`` and pick this rank's device.

    Under ``torchrun`` the RANK / WORLD_SIZE / LOCAL_RANK / MASTER_*
    environment variables are already set and used as-is. Without
    torchrun a single-process group is created on a free local port.

    Args:
        cuda_rank: GPU index to use in the single-process (no torchrun)
            case; None -> ``cuda:0``. Ignored under torchrun, where
            LOCAL_RANK decides.
        cpu: if True, use the gloo backend and return a CPU device
            (NCCL has no CPU support).

    Side effects:
        Initialises the default process group and, on GPU, calls
        ``torch.cuda.set_device`` for this rank's device.

    Returns:
        (rank, world_size, device)
    """
    if "RANK" not in os.environ:
        print("Warning: Not using torchrun. Single device.")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        if cuda_rank is None:
            os.environ["LOCAL_RANK"] = "0"
        else:
            os.environ["LOCAL_RANK"] = str(cuda_rank)

    # NCCL has no CPU support -> use gloo for CPU runs.
    dist.init_process_group(
        backend="gloo" if cpu else "nccl",
        init_method="env://",
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    if not cpu:
        device = torch.device(f"cuda:{local_rank}")
        # Make this rank's GPU the CURRENT CUDA device. Run scripts
        # typically only call torch.set_default_device(device), which
        # steers tensor factories but not streams / CUDA-graph
        # capture / NCCL, all of which follow the current device.
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def print_sampling_settings(
    rank: int, world_size: int, batch_size: int, ns_per_rank: int,
    grad_batch_size: int,
) -> None:
    """Print the per-rank sampling plan (rank 0 only).

    Args:
        rank, world_size: distributed info from ``setup_distributed``.
        batch_size: number of parallel Markov chains (walkers) per rank.
        ns_per_rank: samples collected per rank per VMC step; the
            sampling phase runs ceil(ns_per_rank / batch_size) sweeps.
        grad_batch_size: chunk size of the per-sample gradient
            computation (memory knob only, no effect on results).
    """
    if rank == 0:
        total_ns_expected = ns_per_rank * world_size
        n_sweeps = int(np.ceil(ns_per_rank / batch_size))
        print(
            f"B={batch_size}, Ns_per_rank={ns_per_rank}, "
            f"sweeps/rank={n_sweeps}, "
            f"Total_Ns~{total_ns_expected}, "
            f"grad_batch={grad_batch_size}"
        )

_VALID_PRECONDITIONERS = ('sr', 'spring', 'march', 'adamsr')
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
        Warmup:      run_sampling, run_locE, run_grad
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

    # ----- Warmup -----
    # Which phases ``run_warmup`` exercises: the sampling sweep, the
    # local energies, the gradients. ``run_locE`` needs ``run_sampling``
    # (it consumes the sampler's amplitudes).
    run_sampling: bool = True
    run_locE: bool = True
    run_grad: bool = True

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

    def __init__(self, **kwargs: Any) -> None:
        """Set declared fields from kwargs; attach extras as attributes.

        Every dataclass field takes its value from ``kwargs`` or its
        default; any remaining kwargs become ad-hoc attributes, so
        callers can do ``VMCConfig(batch_size=256, my_extra_flag=True)``
        without subclassing.
        """
        for f in fields(self):
            setattr(self, f.name, kwargs.pop(f.name, f.default))
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__post_init__()

    def __repr__(self) -> str:
        """Repr including declared fields AND ad-hoc extra attributes."""
        declared = [f.name for f in fields(self)]
        extras = [k for k in self.__dict__ if k not in declared]
        parts = [f"{k}={getattr(self, k)!r}" for k in declared + extras]
        return f"{type(self).__name__}({', '.join(parts)})"

    def __post_init__(self) -> None:
        """Validate ``preconditioner`` against ``_VALID_PRECONDITIONERS``."""
        if (self.preconditioner is not None
                and self.preconditioner not in _VALID_PRECONDITIONERS):
            raise ValueError(f"preconditioner must be one of "
                             f"{_VALID_PRECONDITIONERS} or None, got "
                             f"{self.preconditioner!r}")


class VMC_GPU:
    """GPU VMC driver.

    The sampler handles MCMC only (proposing moves,
    accepting/rejecting). This driver orchestrates the
    full sample -> energy -> gradient loop, SR solve,
    and parameter update.

    Args:
        sampler: MCMC sampler providing ``step`` (one sweep over all
            walkers, returns updated configs + amplitudes) and
            ``burn_in``.
        preconditioner: SR / MinSR solver mapping (O_loc, E_loc) to a
            flat update direction ``dp``. None ->
            ``TrivialPreconditionerGPU`` (plain SGD: ``dp`` is the raw
            energy gradient). Ignored by the logvar / SWO drivers.
        optimizer: ``OptimizerGPU`` applying
            ``theta <- theta - lr * dp``. Required by ``run_vmc_loop``
            and the SWO driver.
        evaluate_energy_fn: callable with the ``evaluate_energy``
            signature ``(fxs, model, hamiltonian, amps, ...)`` returning
            ``(amps, local_E)`` or, for the boundary-MPS reuse variants,
            ``(amps, local_E, bMPS_x[, bMPS_y])``; the extra
            environments are forwarded to ``compute_grads_fn``.
        compute_grads_fn: per-sample gradient function with the
            ``compute_grads_gpu`` signature, returning
            ``(grads (B, Np), aux)`` where ``aux`` is the raw amplitudes
            (``use_log_amp=False``, grads are then divided by them) or
            ``(sign, log_abs)`` (``use_log_amp=True``, grads are already
            d log|psi| / d theta).
        mixed_precision: if True the model must be float32; it is
            switched to float64 around every gradient computation and
            back to float32 for sampling / energies. Cuts sampling cost
            on GPUs with slow fp64 while keeping O_loc for SR in fp64.
    """

    def __init__(
        self,
        sampler: SamplerGPU,
        preconditioner: Optional[PreconditionerGPU] = None,
        optimizer: Optional[OptimizerGPU] = None,
        evaluate_energy_fn: Callable = evaluate_energy,
        compute_grads_fn: Callable = compute_grads_gpu,
        mixed_precision: bool = False,
    ) -> None:
        """Store the components; see the class docstring for each."""
        self.sampler = sampler
        if preconditioner is None:
            # Plain SGD: the direction is the raw energy gradient.
            preconditioner = TrivialPreconditionerGPU()
        self.preconditioner = preconditioner
        self.optimizer = optimizer
        self.evaluate_energy_fn = evaluate_energy_fn
        self.compute_grads_fn = compute_grads_fn
        self.mixed_precision = mixed_precision

    # ==========================================================
    # Distributed utilities
    # ==========================================================

    @staticmethod
    def _sync_params(model: WavefunctionModel_GPU) -> None:
        """Broadcast model params from rank 0 to all ranks."""
        if not dist.is_initialized():
            return
        if dist.get_world_size() <= 1:
            return
        for p in model.parameters():
            if not p.data.is_contiguous():
                p.data = p.data.contiguous()
            dist.broadcast(p.data, src=0)

    @staticmethod
    def _allreduce_grads(
        model: WavefunctionModel_GPU, world_size: int,
    ) -> None:
        """Average param.grad across all ranks."""
        if world_size <= 1:
            return
        for p in model.parameters():
            if p.grad is not None:
                if not p.grad.is_contiguous():
                    p.grad = p.grad.contiguous()
                dist.all_reduce(
                    p.grad, op=dist.ReduceOp.SUM,
                )
                p.grad /= world_size

    @staticmethod
    def _count_nan_inf_tensor(
        tensor: torch.Tensor, max_check_elems: int = 10_000_000,
    ) -> tuple[int, int]:
        """Count NaN/Inf without materializing full-size bool masks.

        Args:
            tensor: any tensor; 2-D tensors are scanned row-block-wise,
                others in flat chunks along dim 0.
            max_check_elems: max elements per scanned chunk (bounds the
                temporary bool mask size).

        Returns:
            (n_nan, n_inf) as Python ints.
        """
        if tensor.numel() == 0:
            return 0, 0

        if tensor.dim() >= 2:
            rows_per_check = max(
                1,
                max_check_elems // max(tensor.shape[1], 1),
            )
            n_nan = 0
            n_inf = 0
            for start in range(0, tensor.shape[0], rows_per_check):
                stop = min(start + rows_per_check, tensor.shape[0])
                chunk = tensor[start:stop]
                n_nan += torch.isnan(chunk).sum().item()
                n_inf += torch.isinf(chunk).sum().item()
            return n_nan, n_inf

        n_nan = 0
        n_inf = 0
        for start in range(0, tensor.shape[0], max_check_elems):
            stop = min(start + max_check_elems, tensor.shape[0])
            chunk = tensor[start:stop]
            n_nan += torch.isnan(chunk).sum().item()
            n_inf += torch.isinf(chunk).sum().item()
        return n_nan, n_inf

    @staticmethod
    def _copy_vector_to_model(
        model: WavefunctionModel_GPU, vec: torch.Tensor,
    ) -> None:
        """Copy a flat parameter vector into ``model`` in place."""
        with torch.no_grad():
            offset = 0
            for p in model.parameters():
                n = p.numel()
                p.data.copy_(vec[offset:offset + n].view_as(p.data))
                offset += n

    def _swo_line_search_step(
        self,
        *,
        batch: SWOBatch,
        training_model: WavefunctionModel_GPU,
        direction: torch.Tensor,
        grad_batch_size: int,
        max_backtracks: int,
        shrink: float,
        min_lr: float,
        rank: int,
    ) -> dict[str, Any]:
        """Backtracking line search on the frozen SWO sample set.

        Tries ``theta - lr * direction`` with
        ``lr = optimizer.lr * shrink**k``, k = 0..max_backtracks, and
        accepts the first trial that does not increase the reweighted
        ``-log F`` on ``batch`` -- the very objective ``direction``
        descends. Both sides of the comparison are measured here on the
        same code path (the reference value is not passed in). On
        rejection the parameters AND the optimizer state are restored
        (``optimizer.step`` advances Adam / SPRING momenta, which would
        otherwise be corrupted by rejected trials); ``optimizer.lr``
        itself is never written, so a shrunk trial lr does not ratchet
        the base lr down.

        On return ``batch`` is refreshed at whatever theta ends up
        installed, so the caller can use it immediately.

        Args:
            batch: frozen ``SWOBatch`` (importance-reweighted dataset).
            training_model: model being fitted (updated in place).
            direction: flat (Np,) update direction ``dp``.
            grad_batch_size: chunk size for ``batch.refresh_from_model``.
            max_backtracks: maximum number of lr shrinks to try.
            shrink: multiplicative lr factor per backtrack.
            min_lr: give up once the trial ``lr < min_lr``.
            rank: distributed rank (rank 0 prints the rejection notice).

        Returns:
            dict with ``accepted`` (bool), ``lr`` (accepted lr, 0.0 on
            rejection), ``n_backtrack`` (shrinks used; max_backtracks+1
            on rejection) and ``neg_log_f`` (objective at the installed
            theta).
        """
        current = torch.cat(
            [p.reshape(-1) for p in training_model.parameters()]
        ).detach().clone()
        opt_state = self.optimizer.state_dict()
        old_log_f = float(swo_fidelity(batch)['neg_log_f'])
        base_lr = float(self.optimizer.lr)

        for n_backtrack in range(max_backtracks + 1):
            lr = base_lr * (shrink ** n_backtrack)
            if lr < min_lr:
                break

            self.optimizer.step(
                training_model, direction, learning_rate=lr,
            )
            batch.refresh_from_model(training_model, grad_batch_size)
            new_log_f = float(swo_fidelity(batch)['neg_log_f'])
            if new_log_f <= old_log_f:
                return {
                    'accepted': True,
                    'lr': lr,
                    'n_backtrack': n_backtrack,
                    'neg_log_f': new_log_f,
                }

            self._copy_vector_to_model(training_model, current)
            self.optimizer.load_state_dict(opt_state)

        self._copy_vector_to_model(training_model, current)
        self.optimizer.load_state_dict(opt_state)
        batch.refresh_from_model(training_model, grad_batch_size)
        if rank == 0:
            print(
                "[SWO line search] rejected update: no fixed-sample "
                f"decrease from -logF={old_log_f:.6e}"
            )
        return {
            'accepted': False,
            'lr': 0.0,
            'n_backtrack': max_backtracks + 1,
            'neg_log_f': old_log_f,
        }

    # ==========================================================
    # Warmup
    # ==========================================================

    def run_warmup(
        self,
        fxs: torch.Tensor,
        model: WavefunctionModel_GPU,
        graph: Graph,
        hamiltonian: Hamiltonian,
        rank: int,
        config: VMCConfig,
    ) -> torch.Tensor:
        """One sampling / energy / gradient sweep before the VMC loop.

        Triggers all lazy one-off costs (torch.compile codegen, CUDA
        graph capture of the gradient fn, allocator growth) and prints
        per-phase timings, so the first real VMC step is representative.
        No parameters are updated; energies and gradients are discarded.

        Args:
            fxs: (B, N_sites) int64 initial walker configurations.
            model: wavefunction ``nn.Module`` (float32 if
                ``self.mixed_precision``).
            graph: lattice graph consumed by the sampler's proposals.
            hamiltonian: Hamiltonian providing connected configurations
                and matrix elements for ``evaluate_energy_fn``.
            rank: distributed rank; only rank 0 prints.
            config: ``VMCConfig``-like object; reads ``run_sampling`` /
                ``run_locE`` / ``run_grad`` (which phases to exercise;
                ``run_locE`` requires ``run_sampling`` since it needs
                the sampler's amplitudes), ``use_export_compile``,
                ``use_log_amp``, ``grad_batch_size``, ``verbose`` and
                ``offload_grad_to_cpu``.

        Returns:
            fxs: (B, N_sites) walker configs after the sweep (unchanged
                if ``run_sampling`` is False).
        """
        if self.mixed_precision:
            param_dtype = next(model.parameters()).dtype
            assert param_dtype == torch.float32, (
                f"mixed_precision=True requires a float32 model, "
                f"got {param_dtype}"
            )
        run_sampling = config.run_sampling
        run_locE = config.run_locE
        run_grad = config.run_grad
        use_log_amp = config.use_log_amp

        # Only rank 0 prints — silence verbose=True on other ranks
        # for all downstream calls.
        verbose = config.verbose if rank == 0 else False

        # Offload gradients to CPU when MINRES SR solver can work with
        # CPU-resident data (scipy MINRES).
        offload_grad_cpu = (
            hasattr(self.preconditioner, 'use_scipy')
            and self.preconditioner.use_scipy
        ) or getattr(config, 'offload_grad_to_cpu', False)

        self._sync_params(model)

        if verbose:
            print("\n--- Warmup (1 sweep) ---")
        t_warm = time.time()

        with torch.inference_mode():
            bMPS_x = None
            if run_sampling:
                fxs, amps_out = self.sampler.step(
                    fxs, model, graph,
                    compile=config.use_export_compile,
                    verbose=verbose,
                    use_log_amp=use_log_amp,
                )
                if verbose:
                    print(
                        f"  sample_next:     "
                        f"{time.time() - t_warm:.2f}s"
                    )
                t1 = time.time()
            if run_locE:
                energy_result = self.evaluate_energy_fn(
                    fxs, model, hamiltonian, amps_out,
                    use_log_amp=use_log_amp,
                    verbose=verbose,
                    return_bMPS=True,
                )
                if len(energy_result) == 4:
                    _, local_E, bMPS_x, bMPS_y = energy_result
                elif len(energy_result) == 3:
                    _, local_E, bMPS_x = energy_result
                else:
                    _, local_E = energy_result
                if verbose:
                    print(
                        f"  evaluate_energy: "
                        f"{time.time() - t1:.2f}s"
                    )
        # Free inference-phase tensors before grad computation
        try:
            del amps_out, local_E
            torch.cuda.empty_cache()
        except Exception:
            pass

        t2 = time.time()
        if run_grad:
            if self.mixed_precision:
                model.double()
            with torch.enable_grad():
                grads, grads_aux = self.compute_grads_fn(
                    fxs, model,
                    vectorize=True,
                    batch_size=config.grad_batch_size,
                    vmap_grad=True,
                    offload_to_cpu=offload_grad_cpu,
                    verbose=verbose,
                    use_log_amp=use_log_amp,
                    bMPS_params_x=bMPS_x,
                )
            if self.mixed_precision:
                model.float()

        if verbose:
            print(
                f"  compute_grads:   "
                f"{time.time() - t2:.2f}s"
            )
            print(
                f"  Warmup total:    "
                f"{time.time() - t_warm:.2f}s"
            )
            if run_grad:
                print(
                    f"  [dbg] log_psi_grad: "
                    f"max={grads.abs().max().item():.4e}"
                )

        if run_grad:
            del grads, grads_aux
        torch.cuda.empty_cache()
        
        return fxs

    # ==========================================================
    # Core VMC pipeline
    # ==========================================================

    def _run_sampling_phase(
        self,
        fxs: torch.Tensor,
        model: WavefunctionModel_GPU,
        hamiltonian: Hamiltonian,
        graph: Graph,
        ns_per_rank: int,
        grad_batch_size: int,
        burn_in: bool = False,
        burn_in_steps: int = 0,
        use_export_compile: bool = False,
        debug: bool = False,
        offload_lpg_loc_cpu: bool = False,
        use_log_amp: bool = False,
        verbose: bool = False,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor], torch.Tensor, float,
        dict[str, float],
    ]:
        """Run MCMC sampling, energy eval, and gradient
        computation for one VMC step.

        The sampler only does MCMC (step / burn_in). This
        method calls evaluate_energy_fn and
        compute_grads_fn directly.

        Args:
            fxs: (B, N_sites) int64 walker configurations; B is the
                number of parallel Markov chains.
            model: wavefunction model.
            hamiltonian: Hamiltonian for the local energies.
            graph: lattice graph consumed by the sampler's proposals.
            ns_per_rank: samples to collect on this rank (Ns). The
                loop runs ceil(Ns / B) sweeps and keeps only the first
                ``Ns - collected`` walkers of the last one.
            grad_batch_size: chunk size for ``compute_grads_fn``
                (memory knob only).
            burn_in: if True, run ``burn_in_steps`` sampler sweeps
                first (the caller does this on VMC step 0 only).
            burn_in_steps: number of burn-in sweeps.
            use_export_compile: forwarded to the sampler as
                ``compile`` -- use the exported + compiled forward.
            debug: rank-0 prints of amplitude / gradient statistics
                for every chunk.
            offload_lpg_loc_cpu: if True, move log_psi_grad chunks
                to CPU immediately after GPU computation.
            use_log_amp: if True, work in log-amplitude
                space throughout (sampler, energy, grads).
            verbose: forwarded to the sampler / energy / gradient
                functions for their own timing prints.

        Returns:
            (local_energies, local_log_psi_grad): (Ns,) local
                energies and (Ns, Np) log-derivatives
                O_loc = d log|psi| / d theta, Ns = ns_per_rank.
            fxs: (B, N_sites) updated walker configs.
            sample_time: t_samp + t_locE + t_grad in seconds.
            phase_times: dict with t_samp, t_locE, t_grad.
        """
        B = fxs.shape[0]
        t_samp, t_locE, t_grad = 0.0, 0.0, 0.0

        # Burn-in
        if burn_in:
            t0 = time.time()
            fxs = self.sampler.burn_in(
                fxs, model, graph, burn_in_steps,
                compile=use_export_compile,
                use_log_amp=use_log_amp,
            )
            print(f'Burn-in: {burn_in_steps} steps, T_b = {time.time()-t0}')

        local_energies = None
        local_log_psi_grad = None
        current_count = 0

        while current_count < ns_per_rank:
            needed = min(B, ns_per_rank - current_count)
            with torch.inference_mode():
                # 1. MCMC sweep
                t0 = time.time()
                fxs, amps_out = self.sampler.step(
                    fxs, model, graph,
                    compile=use_export_compile,
                    use_log_amp=use_log_amp,
                    verbose=verbose,
                )
                t_samp += time.time() - t0

                # 2. Local energy
                t0 = time.time()
                energy_result = self.evaluate_energy_fn(
                    fxs, model, hamiltonian, amps_out,
                    use_log_amp=use_log_amp,
                    verbose=verbose,
                    return_bMPS=True,
                )
                if len(energy_result) == 4:
                    _, local_E, bMPS_x, bMPS_y = energy_result
                elif len(energy_result) == 3:
                    _, local_E, bMPS_x = energy_result
                else:
                    _, local_E = energy_result
                    bMPS_x = None
                t_locE += time.time() - t0

            # Free sampling/energy tensors so allocator
            # can reuse their blocks for grad computation
            del amps_out

            # 3. Gradients -> log_psi_grad
            t0 = time.time()
            if self.mixed_precision:
                model.double()  # f32 → f64 for accurate backprop
            with torch.enable_grad():
                grads, grads_aux = self.compute_grads_fn(
                    fxs, model,
                    vectorize=True,
                    batch_size=grad_batch_size,
                    vmap_grad=True,
                    offload_to_cpu=offload_lpg_loc_cpu,
                    use_log_amp=use_log_amp,
                    verbose=verbose,
                    bMPS_params_x=bMPS_x,
                )
            _rank = dist.get_rank() if dist.is_initialized() else 0

            if use_log_amp:
                # grads is already d(log|psi|)/d(params)
                # No division needed.
                if debug and _rank == 0:
                    g_rms = (
                        torch.norm(grads).item()
                        / grads.numel() ** 0.5
                    )
                    print(
                        f"  [dbg] log_psi_grad: "
                        f"rms={g_rms:.4e}, "
                        f"max={grads.abs().max().item():.4e}"
                    )
            else:
                # grads_aux is raw amps — divide to get
                # lpg_loc = d(psi)/d(params) / psi
                amps2 = grads_aux
                if debug and _rank == 0:
                    abs_a = amps2.abs()
                    print(
                        f"  [dbg] amps: "
                        f"min={abs_a.min().item():.4e}"
                        f", median={abs_a.median().item():.4e}"
                        f", mean={abs_a.mean().item():.4e}"
                        f", max={abs_a.max().item():.4e}"
                    )
                    g_rms = (
                        torch.norm(grads).item()
                        / grads.numel() ** 0.5
                    )
                    print(
                        f"  [dbg] raw grads: "
                        f"rms={g_rms:.4e}, "
                        f"max={grads.abs().max().item():.4e}"
                    )
                grads /= amps2.unsqueeze(1)
                if debug and _rank == 0:
                    print(
                        f"  [dbg] log_psi_grad: rms="
                        f"{torch.norm(grads).item() / grads.numel()**0.5:.4e}"
                        f", max={grads.abs().max().item():.4e}"
                    )

            if self.mixed_precision:
                model.float()  # restore f32 for next sampling sweep

            # bMPS tensors no longer needed after grad computation
            bMPS_x = None
            if 'bMPS_y' in dir():
                bMPS_y = None
            t_grad += time.time() - t0

            if (
                local_log_psi_grad is None
                and current_count == 0
                and needed == ns_per_rank
                and needed == grads.shape[0]
            ):
                local_energies = local_E[:needed].detach()
                local_log_psi_grad = grads[:needed].detach()
                current_count = needed
                del grads, grads_aux, local_E
                continue

            if local_energies is None:
                local_energies = torch.empty(
                    ns_per_rank,
                    dtype=local_E.dtype,
                    device=local_E.device,
                )
            if local_log_psi_grad is None:
                lpg_shape = (ns_per_rank, grads.shape[1])
                if (
                    grads.device.type == 'cpu'
                    and torch.cuda.is_available()
                ):
                    local_log_psi_grad = torch.empty(
                        lpg_shape,
                        dtype=grads.dtype,
                        device=grads.device,
                        pin_memory=True,
                    )
                else:
                    local_log_psi_grad = torch.empty(
                        lpg_shape,
                        dtype=grads.dtype,
                        device=grads.device,
                    )

            count_next = current_count + needed
            local_energies[current_count:count_next].copy_(
                local_E[:needed].detach(),
            )
            local_log_psi_grad[current_count:count_next].copy_(
                grads[:needed].detach(),
            )
            current_count = count_next
            del grads, grads_aux, local_E

        phase_times = {
            't_samp': t_samp,
            't_locE': t_locE,
            't_grad': t_grad,
        }
        sample_time = t_samp + t_locE + t_grad
        return (
            (local_energies, local_log_psi_grad),
            fxs,
            sample_time,
            phase_times,
        )

    def compute_global_energy_stats(
        self, local_energies: torch.Tensor, world_size: int,
    ) -> tuple[int, float, float]:
        """All-reduce local energies into the global mean and variance.

        Requires an initialised process group (``setup_distributed``).

        Args:
            local_energies: (Ns_local,) local energies on this rank.
            world_size: number of ranks; all are assumed to hold the
                same Ns_local.

        Returns:
            total_ns: Ns_local * world_size.
            energy_mean: global <E_loc>.
            energy_var: global population variance
                <E_loc^2> - <E_loc>^2 (NOT the variance of the mean;
                the statistical error of the mean is
                sqrt(energy_var / total_ns)).
        """
        n_local = local_energies.shape[0]
        total_ns = n_local * world_size

        local_e_sum = local_energies.sum()
        dist.all_reduce(
            local_e_sum, op=dist.ReduceOp.SUM,
        )
        energy_mean = local_e_sum.item() / total_ns

        local_e_sq_sum = (local_energies ** 2).sum()
        dist.all_reduce(
            local_e_sq_sum, op=dist.ReduceOp.SUM,
        )
        energy_var = (
            local_e_sq_sum.item() / total_ns
            - energy_mean ** 2
        )

        return total_ns, energy_mean, energy_var

    def _compute_step_diagnostics(
        self,
        local_energies: torch.Tensor,
        local_lpg: torch.Tensor,
        fxs: torch.Tensor,
        model: WavefunctionModel_GPU,
        energy_mean: float,
        use_log_amp: bool,
        world_size: int,
        rank: int,
    ) -> dict[str, Any]:
        """Collect distributional stats for one VMC step.

        Cheap (one extra forward + a few all-gathers); call only when
        ``config.diagnostics`` is set. Returns a dict (only meaningful
        on rank 0; other ranks get an empty dict).

        Args:
            local_energies: (Ns_local,) local energies on this rank.
            local_lpg: (Ns_local, Np) log-derivatives on this rank.
            fxs: (B, N_sites) current walkers (for the log|psi|
                forward).
            model: wavefunction model.
            energy_mean: global <E_loc> from
                ``compute_global_energy_stats``.
            use_log_amp: use ``model.forward_log`` rather than
                ``log|model(fxs)|``.
            world_size, rank: distributed info.

        Diagnostics:
          - E_loc: min, max, median, std, count of outliers
            (|E_loc - median| > 5 * MAD).
          - log|psi|: min, max, median, std (or, equivalently, the
            spread of amplitudes — large spread signals near-nodal
            walkers and likely E_loc blow-ups).
          - ||g||: 2-norm of the SR right-hand side
            g = <(E_loc - E_mean) * (O_loc - O_mean)>_sample.
            If huge, even a converged MINRES solve produces a huge dp.
        """
        import numpy as np

        # ---- Gather local_energies across ranks ----
        if world_size > 1:
            E_list = [
                torch.empty_like(local_energies)
                for _ in range(world_size)
            ]
            dist.all_gather(E_list, local_energies.contiguous())
            E_all = torch.cat(E_list)
        else:
            E_all = local_energies

        # ---- Compute ||g|| globally
        # g = <(E - <E>) O>_sample = (1/Ns) sum_i (E_i - E_mean) O_i
        # (the O_mean subtraction cancels under SUM-then-divide).
        diff = (local_energies - energy_mean).to(local_lpg.dtype)
        g_local = (
            diff.unsqueeze(1) * local_lpg
        ).sum(dim=0)
        if world_size > 1:
            dist.all_reduce(g_local, op=dist.ReduceOp.SUM)
        Ns_total = local_energies.shape[0] * world_size
        g = g_local / Ns_total
        g_norm = g.norm().item()
        g_max = g.abs().max().item()

        # ---- log|psi| via a forward over local fxs (then gather) ----
        with torch.inference_mode():
            if use_log_amp:
                _, log_abs_local = model.forward_log(fxs)
            else:
                amps = model(fxs)
                log_abs_local = amps.abs().clamp_min(1e-300).log()
        if world_size > 1:
            la_list = [
                torch.empty_like(log_abs_local)
                for _ in range(world_size)
            ]
            dist.all_gather(la_list, log_abs_local.contiguous())
            log_abs_all = torch.cat(la_list)
        else:
            log_abs_all = log_abs_local

        if rank != 0:
            return {}

        E_np = E_all.detach().cpu().numpy().astype(np.float64)
        la_np = log_abs_all.detach().cpu().numpy().astype(np.float64)

        E_median = float(np.median(E_np))
        mad = float(np.median(np.abs(E_np - E_median)))
        n_outlier = int(
            np.sum(np.abs(E_np - E_median) > 5.0 * max(mad, 1e-12))
        )

        return {
            'Ns': int(E_np.size),
            'E_min': float(E_np.min()),
            'E_max': float(E_np.max()),
            'E_median': E_median,
            'E_std': float(E_np.std()),
            'E_MAD': mad,
            'E_outlier_5MAD': n_outlier,
            'logpsi_min': float(la_np.min()),
            'logpsi_max': float(la_np.max()),
            'logpsi_median': float(np.median(la_np)),
            'logpsi_std': float(la_np.std()),
            'logpsi_spread': float(la_np.max() - la_np.min()),
            'g_norm': float(g_norm),
            'g_max': float(g_max),
        }

    def solve_sr_step(
        self,
        O_loc: torch.Tensor | list[torch.Tensor],
        E_loc: torch.Tensor,
        E_mean: float,
        Ns: int,
        Np: int,
        rshift: float,
        ashift: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, float, Any]:
        """Solve the SR / MinSR system for the update direction.

        Thin wrapper around ``self.preconditioner.solve``.

        Args:
            O_loc: (Ns_local, Np) log-derivatives, or a 1-element list
                wrapping them ("ownership box") when the preconditioner
                supports freeing the Jacobian early.
            E_loc: (Ns_local,) local energies.
            E_mean: global energy mean.
            Ns: GLOBAL sample count (all ranks).
            Np: number of parameters.
            rshift, ashift: Tikhonov regularisation of the Gram matrix,
                shift = rshift * trace(T)/sqrt(n) + ashift
                (see ``VMCConfig``).
            device: device on which the solve runs.

        Returns:
            dp: (Np,) update direction (S + shift)^-1 F.
            t_sr: wall time of the solve in seconds.
            info: solver-specific convergence info (e.g. iteration
                count); printed in the step log.
        """
        return self.preconditioner.solve(
            O_loc=O_loc,
            E_loc=E_loc,
            E_mean=E_mean,
            Ns=Ns,
            Np=Np,
            rshift=rshift,
            ashift=ashift,
            device=device,
        )

    def apply_parameter_update(
        self,
        model: WavefunctionModel_GPU,
        dp: torch.Tensor,
        learning_rate: float,
        device: torch.device,
    ) -> None:
        """Apply ``theta <- theta - learning_rate * dp`` in place.

        Args:
            model: model whose parameters are updated.
            dp: (Np,) flat update direction from ``solve_sr_step``.
            learning_rate: step size.
            device: device on which the update is computed.
        """
        self.optimizer.step(
            model=model,
            direction=dp,
            device=device,
            learning_rate=learning_rate,
        )
        return


    def run_vmc_loop(
        self,
        fxs: torch.Tensor,
        model: WavefunctionModel_GPU,
        hamiltonian: Hamiltonian,
        graph: Graph,
        rank: int,
        world_size: int,
        config: VMCConfig,
        n_params: int,
        nsites: int,
        on_step_end: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[float], torch.Tensor]:
        """Drive the VMC sampling/SR-solve/parameter-update loop.

        Args:
            fxs: (B, N_sites) int64 initial walker configurations.
            model: wavefunction model (float32 if
                ``self.mixed_precision``).
            hamiltonian: Hamiltonian for the local energies.
            graph: lattice graph consumed by the sampler's proposals.
            rank, world_size: from ``setup_distributed``.
            config: a ``VMCConfig``-shaped object (any object with
                the expected attributes works). ``config.learning_rate``
                is overwritten in place every step when
                ``config.lr_scheduler`` is set.
            n_params: total trainable parameter count of ``model``.
            nsites: number of lattice sites (for per-site energy
                reporting).
            on_step_end: optional callback, invoked on EVERY rank after
                each step with a dict: ``step`` (global step, includes
                ``config.resume_step``), ``energy_mean``, ``energy_var``,
                ``energy_per_site``, ``error_per_site``,
                ``total_samples``, ``sample_time``, ``phase_times``,
                ``sr_time``, ``total_time``, ``solver_info``, ``fxs``
                ((Ns, N_sites) walkers gathered from all ranks) and
                ``diagnostics`` (dict from ``_compute_step_diagnostics``
                or None). Intended for checkpointing / logging.

        Returns:
            energy_history: list of per-site energies, one per step.
            fxs: (B, N_sites) final walker configs on this rank.
        """
        if self.mixed_precision:
            param_dtype = next(model.parameters()).dtype
            assert param_dtype == torch.float32, (
                f"mixed_precision=True requires a float32 model, "
                f"got {param_dtype}"
            )
        device = next(model.parameters()).device
        self._sync_params(model)

        # Offload lpg_loc to CPU when MINRES SR solver can work with
        # CPU-resident data (scipy MINRES).
        offload_lpg_loc_cpu = (
            hasattr(self.preconditioner, 'use_scipy')
            and self.preconditioner.use_scipy
        ) or getattr(config, 'offload_grad_to_cpu', False)

        if rank == 0 and config.show_progress:
            print(f"\n--- VMC ({config.vmc_steps} steps) ---")
            vmc_pbar = tqdm(
                total=config.vmc_steps, desc="VMC Steps",
            )
        else:
            vmc_pbar = None

        energy_history = []
        for step in range(config.vmc_steps):
            t0 = time.time()

            (local_energies, local_lpg), fxs, sample_time, phase_times = (
                self._run_sampling_phase(
                    fxs=fxs,
                    model=model,
                    hamiltonian=hamiltonian,
                    graph=graph,
                    ns_per_rank=config.ns_per_rank,
                    grad_batch_size=config.grad_batch_size,
                    burn_in=(step == 0),
                    burn_in_steps=config.burn_in_steps,
                    use_export_compile=config.use_export_compile,
                    debug=config.debug,
                    offload_lpg_loc_cpu=offload_lpg_loc_cpu,
                    use_log_amp=config.use_log_amp,
                    verbose=config.verbose,
                )
            )

            total_ns, energy_mean, energy_var = (
                self.compute_global_energy_stats(
                    local_energies, world_size,
                )
            )

            # --- Diagnostics on SR inputs ---
            if rank == 0:
                n_nan_E = torch.isnan(
                    local_energies
                ).sum().item()
                n_nan_lpg, n_inf_lpg = (
                    self._count_nan_inf_tensor(local_lpg)
                )
                if n_nan_E or n_nan_lpg or n_inf_lpg:
                    print(
                        f"[WARNING] SR inputs: "
                        f"local_E has {n_nan_E} NaN, "
                        f"log_psi_grad has "
                        f"{n_nan_lpg} NaN / "
                        f"{n_inf_lpg} Inf"
                    )
                if config.debug:
                    Np = local_lpg.shape[1]
                    pv0 = torch.nn.utils.parameters_to_vector(
                        model.parameters(),
                    )
                    print(
                        f"  [dbg] params: Np={Np}, "
                        f"rms="
                        f"{torch.norm(pv0).item()/Np**0.5:.4e}, "
                        f"max={pv0.abs().max().item():.4e}"
                    )
                    print(
                        f"  [dbg] log_psi_grad: rms="
                        f"{torch.norm(local_lpg).item()/(local_lpg.numel()**0.5):.4e}, "
                        f"max={local_lpg.abs().max().item():.4e}"
                    )
                    
            # --- Optional per-step diagnostics (E_loc / log|psi| /
            # SR RHS norm). Independent of `config.debug`. ---
            diag_stats = None
            if config.diagnostics:
                diag_stats = self._compute_step_diagnostics(
                    local_energies=local_energies,
                    local_lpg=local_lpg,
                    fxs=fxs,
                    model=model,
                    energy_mean=energy_mean,
                    use_log_amp=config.use_log_amp,
                    world_size=world_size,
                    rank=rank,
                )

            # SR solve. When the preconditioner supports it, hand the
            # Jacobian over via a 1-element "ownership box" and drop
            # our own reference now, so the (Ns, Np) Jacobian can be
            # freed inside _solve (right after the all_to_all copy)
            # instead of lingering in this frame until step end.
            if getattr(
                self.preconditioner, '_supports_ownership_box', False,
            ):
                _lpg_arg = [local_lpg]
                del local_lpg
            else:
                _lpg_arg = local_lpg

            dp, t_sr, info = self.solve_sr_step(
                O_loc=_lpg_arg,
                E_loc=local_energies,
                E_mean=energy_mean,
                Ns=total_ns,
                Np=n_params,
                rshift=config.sr_rshift,
                ashift=config.sr_ashift,
                device=device,
            )

            # Free lpg/energies before next step's grad alloc
            del _lpg_arg, local_energies
            torch.cuda.empty_cache()

            # NaN/Inf check on SR gradients
            dp_t = torch.as_tensor(dp, device=device)
            if rank == 0:
                check_NaN_or_inf(dp_t, "SR dp")

            # Diagnostic magnitudes for step logging. Reduce per-param
            # (not parameters_to_vector, whose .view(-1) fails on the
            # non-contiguous TN params when train_tn=True).
            max_pv0 = max(
                p.detach().abs().max().item()
                for p in model.parameters()
            )
            max_dp_t = dp_t.abs().max().item()

            # Apply parameter update
            if config.lr_scheduler is not None:
                global_step = step + config.resume_step
                config.learning_rate = config.lr_scheduler(
                    global_step,
                )

            self.apply_parameter_update(
                model, dp, config.learning_rate, device,
            )

            # Paranoid sync
            self._sync_params(model)

            # End of step diagnostics and logging
            step_time = time.time() - t0

            e_per_site = energy_mean / nsites
            err = (
                np.sqrt(max(energy_var, 0.0) / total_ns)
                / nsites
            )
            energy_history.append(e_per_site)

            global_step = step + config.resume_step
            if rank == 0 and config.show_progress:
                t_s = phase_times.get('t_samp', 0.0)
                t_e = phase_times.get('t_locE', 0.0)
                t_g = phase_times.get('t_grad', 0.0)
                lr_str = (
                    f" lr={config.learning_rate:.2e}"
                    if config.lr_scheduler is not None else ""
                )
                print(
                    f"Step {global_step:3d} | E/site: "
                    f"{e_per_site:.6f} "
                    f"+/- {err:.6f} | N={total_ns}{lr_str} | "
                    f"T_samp={t_s:.1f}s T_locE={t_e:.1f}s "
                    f"T_grad={t_g:.1f}s T_SR={t_sr:.2f}s "
                    f"T_total={step_time:.1f}s | "
                    f"SR_info={info} "
                    f"max_p={max_pv0:.4e} max_dp={max_dp_t:.4e}"
                )
                
                # Always-printed SR mask report (independent of
                # `diagnostics`): fires whenever rows were masked this
                # step — non-finite (NaN/Inf, always checked) and/or
                # |E|>clip outliers (when local_E_clip is set).
                mask_info = getattr(
                    self.preconditioner, '_last_mask_info', None,
                )
                if rank == 0 and mask_info:
                    n_masked = mask_info['n_masked']
                    n_nf = mask_info.get('n_nonfinite', 0)
                    Ns_total = mask_info['Ns_total']
                    clip = mask_info['clip']
                    clip_str = (
                        f"{clip:.3e}" if clip is not None else "off"
                    )
                    # Non-finite rows signal fp32 forward/backward
                    # instability -> louder WARNING tag.
                    tag = "[SR mask WARNING]" if n_nf else "[SR mask]"
                    print(
                        f"  {tag} n_masked={n_masked}/{Ns_total} "
                        f"({100.0 * n_masked / Ns_total:.2f}%) "
                        f"non-finite={n_nf} clip={clip_str} "
                        f"E_mean_clean={mask_info['E_mean_clean']:+.3e}"
                    )
                vmc_pbar.update(1)

            # Gather walker configs from all ranks
            # for checkpoint saving (Ns, N_sites)
            if world_size > 1:
                _fxs_list = [
                    torch.zeros_like(fxs)
                    for _ in range(world_size)
                ]
                dist.all_gather(
                    _fxs_list, fxs.contiguous(),
                )
                all_fxs = torch.cat(_fxs_list, dim=0)
            else:
                all_fxs = fxs

            if on_step_end is not None:
                on_step_end(
                    {
                        "step": global_step,
                        "energy_mean": energy_mean,
                        "energy_var": energy_var,
                        "energy_per_site": e_per_site,
                        "error_per_site": err,
                        "total_samples": total_ns,
                        "sample_time": sample_time,
                        "phase_times": phase_times,
                        "sr_time": t_sr,
                        "total_time": step_time,
                        "solver_info": info,
                        "fxs": all_fxs,
                        "diagnostics": diag_stats,
                    }
                )

        if vmc_pbar is not None:
            vmc_pbar.close()
        return energy_history, fxs

    # ==========================================================
    # BELOW ARE ALL EXPERIMENTAL / UNVERIFIED METHODS.  USE AT YOUR OWN RISK.
    # EXPERIMENTAL: Log-variance optimization (NOT VERIFIED)
    #
    # The pathwise and REINFORCE gradient terms individually
    # pass FD checks, but they nearly perfectly cancel
    # (cos ~ -0.9999), leaving a tiny, noise-dominated
    # residual.  As a result, logvar optimization does NOT
    # converge in practice.  These methods are kept for
    # future investigation but should not be relied upon.
    # The gradient identity below was checked against autograd on
    # small systems.
    # ==========================================================

    def _run_logvar_sampling_phase(
        self,
        fxs: torch.Tensor,
        model: WavefunctionModel_GPU,
        hamiltonian: Hamiltonian,
        graph: Graph,
        ns_per_rank: int,
        burn_in: bool = False,
        burn_in_steps: int = 0,
        use_export_compile: bool = False,
        use_log_amp: bool = False,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """MCMC sampling + energy eval only (no gradient).

        Args:
            fxs: (B, N_sites) int64 walker configurations.
            model: wavefunction model.
            hamiltonian: Hamiltonian for the local energies.
            graph: lattice graph consumed by the sampler's proposals.
            ns_per_rank: samples to collect on this rank (Ns).
            burn_in: if True, run ``burn_in_steps`` sampler sweeps
                first.
            burn_in_steps: number of burn-in sweeps.
            use_export_compile: forwarded to the sampler as
                ``compile``.
            use_log_amp: work in log-amplitude space.
            verbose: forwarded to the sampler / energy function.

        Returns:
            all_fxs: (Ns, N_sites) sampled configs (kept for the
                surrogate-loss forward in ``_compute_logvar_loss``).
            local_energies: (Ns,) local energies.
            fxs: (B, N_sites) updated walker configs.
            phase_times: dict with t_samp, t_locE.

        Currently unreliable (see the EXPERIMENTAL note above).
        """
        B = fxs.shape[0]
        t_samp, t_locE = 0.0, 0.0

        if burn_in:
            t0 = time.time()
            fxs = self.sampler.burn_in(
                fxs, model, graph, burn_in_steps,
                compile=use_export_compile,
                use_log_amp=use_log_amp,
            )
            print(
                f'Burn-in: {burn_in_steps} steps, '
                f'T_b = {time.time()-t0}'
            )

        local_energies_list = []
        all_fxs_list = []
        current_count = 0

        while current_count < ns_per_rank:
            needed = min(B, ns_per_rank - current_count)
            with torch.inference_mode():
                # 1. MCMC sweep
                t0 = time.time()
                fxs, amps_out = self.sampler.step(
                    fxs, model, graph,
                    compile=use_export_compile,
                    use_log_amp=use_log_amp,
                    verbose=verbose,
                )
                t_samp += time.time() - t0

                # 2. Local energy
                t0 = time.time()
                energy_result = self.evaluate_energy_fn(
                    fxs, model, hamiltonian, amps_out,
                    use_log_amp=use_log_amp,
                    verbose=verbose,
                    return_bMPS=False,
                )
                if len(energy_result) == 2:
                    _, local_E = energy_result
                else:
                    _, local_E = (
                        energy_result[0],
                        energy_result[1],
                    )
                t_locE += time.time() - t0

            del amps_out

            local_energies_list.append(
                local_E[:needed].detach(),
            )
            # Save configs for surrogate loss forward pass
            all_fxs_list.append(
                fxs[:needed].detach().clone(),
            )
            current_count += needed
            del local_E

        local_energies = torch.cat(local_energies_list)
        all_fxs = torch.cat(all_fxs_list)

        phase_times = {
            't_samp': t_samp,
            't_locE': t_locE,
        }
        return all_fxs, local_energies, fxs, phase_times

    @staticmethod
    def _compute_logvar_loss(
        all_fxs: torch.Tensor,
        hamiltonian: Hamiltonian,
        gamma: float,
        model: WavefunctionModel_GPU,
        grad_batch_size: int,
        use_log_amp: bool = True,
    ) -> tuple[float, float, float]:
        """Two-pass log-variance gradient: pathwise + REINFORCE.

        Pass 1 (pathwise): Forward through evaluate_energy_grad
        with grad tracking, compute L = log(Var[E_L] + gamma),
        backprop. This captures dE_L/dtheta contributions.

        Pass 2 (REINFORCE / score function): Using detached E_L
        from pass 1, compute centered surrogate loss that
        estimates Cov(O_k, (E_L - Ebar)^2). The centering
        (-sigma^2 baseline) is essential because our psi is
        unnormalized, so <O_k> != 0.

        Full gradient:
          dL/dtheta = (1/(sigma^2+gamma)) * d(sigma^2)/dtheta
        where:
          d(sigma^2)/dtheta = 2*Cov(O, (E_L-Ebar)^2)  [REINFORCE]
                            + 2*<(E_L-Ebar)*dE_L/dtheta> [pathwise]

        Args:
            all_fxs: (Ns, N_sites) frozen sample configurations.
            hamiltonian: Hamiltonian for the local energies.
            gamma: regulariser inside the log,
                L = log(Var[E_L] + gamma).
            model: wavefunction model; ``model.zero_grad()`` is called
                first and both passes accumulate into ``param.grad``.
            grad_batch_size: chunk size for both passes.
            use_log_amp: use ``model.forward_log`` instead of raw
                amplitudes.

        Returns:
            (log_var_loss, energy_mean, energy_var):
                scalars for logging. Grads are in param.grad.
        """
        Ns = all_fxs.shape[0]
        model.zero_grad()

        # === Pass 1: Pathwise (autograd through E_L) ===
        local_E_list = []
        for b_start in range(0, Ns, grad_batch_size):
            b_end = min(b_start + grad_batch_size, Ns)
            fxs_chunk = all_fxs[b_start:b_end]

            if use_log_amp:
                signs, log_abs = model.forward_log(
                    fxs_chunk,
                )
                amps_for_eval = (signs, log_abs)
            else:
                amps_for_eval = model(fxs_chunk)

            _, chunk_E = evaluate_energy_grad(
                fxs_chunk, model, hamiltonian,
                amps_for_eval,
                use_log_amp=use_log_amp,
            )
            local_E_list.append(chunk_E)

        local_E = torch.cat(local_E_list)
        E_mean = local_E.mean()
        E_var = ((local_E - E_mean) ** 2).mean()
        loss = torch.log(E_var + gamma)
        loss.backward()  # pathwise -> param.grad

        # === Pass 2: REINFORCE (score function) ===
        E_mean_val = E_mean.item()
        E_var_val = E_var.item()
        local_E_det = local_E.detach()

        # Centered weights: Cov(O, (E_L-Ebar)^2)
        # The -sigma^2 baseline accounts for <O> != 0
        # from unnormalized psi.
        weights = (
            (local_E_det - E_mean_val) ** 2 - E_var_val
        ) / (E_var_val + gamma)

        for b_start in range(0, Ns, grad_batch_size):
            b_end = min(b_start + grad_batch_size, Ns)
            fxs_chunk = all_fxs[b_start:b_end]
            w = weights[b_start:b_end]

            if use_log_amp:
                _, log_abs = model.forward_log(fxs_chunk)
            else:
                log_abs = torch.log(
                    model(fxs_chunk).abs()
                )

            surr = (2.0 / Ns) * (w * log_abs).sum()
            surr.backward()  # accumulates into param.grad

        return (
            loss.item(),
            E_mean_val,
            E_var_val,
        )

    def run_vmc_loop_logvar(
        self,
        fxs: torch.Tensor,
        model: WavefunctionModel_GPU,
        hamiltonian: Hamiltonian,
        graph: Graph,
        rank: int,
        world_size: int,
        config: VMCConfig,
        n_params: int,
        nsites: int,
        torch_optimizer: torch.optim.Optimizer,
        on_step_end: Optional[
            Callable[[Dict[str, Any]], None]
        ] = None,
    ) -> tuple[list[float], torch.Tensor]:
        """VMC loop using log-variance loss + AdamW.

        Instead of SR, computes a surrogate loss whose
        gradient equals the REINFORCE estimator of
        grad log(Var[E_L] + gamma).  AdamW updates params.

        Args:
            fxs, model, hamiltonian, graph, rank, world_size,
                n_params, nsites: as in ``run_vmc_loop``.
            config: ``VMCConfig``-like; uses ``logvar_gamma`` for the
                loss regulariser plus the sampling / compile fields.
            torch_optimizer: a ``torch.optim`` optimizer (e.g. AdamW)
                over ``model.parameters()``. ``self.preconditioner`` and
                ``self.optimizer`` are not used by this loop.
            on_step_end: per-step callback as in ``run_vmc_loop``
                (``sr_time`` is 0.0, ``solver_info`` None and there is
                no ``diagnostics`` key).

        Returns:
            energy_history: list of per-site energies, one per step.
            fxs: (B, N_sites) final walker configs on this rank.
        """
        warnings.warn(
            "Log-variance optimization is experimental and "
            "does not currently converge in practice. Use "
            "with caution."
        )
        self._sync_params(model)

        if rank == 0 and config.show_progress:
            print(
                f"\n--- VMC logvar "
                f"({config.vmc_steps} steps) ---"
            )
            vmc_pbar = tqdm(
                total=config.vmc_steps,
                desc="VMC LogVar",
            )
        else:
            vmc_pbar = None

        gamma = config.logvar_gamma
        energy_history = []

        for step in range(config.vmc_steps):
            t0 = time.time()

            # 1. MCMC sampling (inference mode)
            all_fxs_local, _, fxs, phase_times = (
                self._run_logvar_sampling_phase(
                    fxs=fxs,
                    model=model,
                    hamiltonian=hamiltonian,
                    graph=graph,
                    ns_per_rank=config.ns_per_rank,
                    burn_in=(step == 0),
                    burn_in_steps=config.burn_in_steps,
                    use_export_compile=(
                        config.use_export_compile
                    ),
                    use_log_amp=config.use_log_amp,
                    verbose=config.verbose,
                )
            )

            # 2. Log-variance loss with full autograd
            #    (re-evaluates amps + E_L with grad tracking)
            t_grad_0 = time.time()
            torch_optimizer.zero_grad()
            with torch.enable_grad():
                log_var_loss, energy_mean, energy_var = (
                    self._compute_logvar_loss(
                        all_fxs_local,
                        hamiltonian,
                        gamma,
                        model,
                        config.grad_batch_size,
                        use_log_amp=config.use_log_amp,
                    )
                )
            t_grad = time.time() - t_grad_0

            total_ns = (
                all_fxs_local.shape[0] * world_size
            )

            # 3. Allreduce gradients
            t_comm_0 = time.time()
            self._allreduce_grads(model, world_size)
            t_comm = time.time() - t_comm_0

            # 4. AdamW step
            torch_optimizer.step()

            # 5. Sync params across ranks
            self._sync_params(model)

            del all_fxs_local

            # Logging
            step_time = time.time() - t0
            e_per_site = energy_mean / nsites
            err = (
                np.sqrt(max(energy_var, 0.0) / total_ns)
                / nsites
            )
            energy_history.append(e_per_site)

            # Gradient norm for diagnostics
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5

            global_step = step + config.resume_step
            if rank == 0 and config.show_progress:
                t_s = phase_times.get('t_samp', 0.0)
                t_e = phase_times.get('t_locE', 0.0)
                print(
                    f"Step {global_step:3d} | "
                    f"E/site: {e_per_site:.6f} "
                    f"+/- {err:.6f} | "
                    f"logvar={log_var_loss:.4f} | "
                    f"|g|={grad_norm:.4e} | "
                    f"N={total_ns} | "
                    f"T_samp={t_s:.1f}s "
                    f"T_locE={t_e:.1f}s "
                    f"T_grad={t_grad:.1f}s "
                    f"T_comm={t_comm:.2f}s "
                    f"T_total={step_time:.1f}s"
                )
                vmc_pbar.update(1)

            # Gather walker configs for checkpoint
            if world_size > 1:
                _fxs_list = [
                    torch.zeros_like(fxs)
                    for _ in range(world_size)
                ]
                dist.all_gather(
                    _fxs_list, fxs.contiguous(),
                )
                all_fxs_ckpt = torch.cat(
                    _fxs_list, dim=0,
                )
            else:
                all_fxs_ckpt = fxs

            if on_step_end is not None:
                on_step_end(
                    {
                        "step": global_step,
                        "energy_mean": energy_mean,
                        "energy_var": energy_var,
                        "energy_per_site": e_per_site,
                        "error_per_site": err,
                        "total_samples": total_ns,
                        "sample_time": (
                            phase_times['t_samp']
                            + phase_times['t_locE']
                        ),
                        "phase_times": phase_times,
                        "sr_time": 0.0,
                        "total_time": step_time,
                        "solver_info": None,
                        "fxs": all_fxs_ckpt,
                    }
                )

        if vmc_pbar is not None:
            vmc_pbar.close()
        return energy_history, fxs

    # ==========================================================
    # Supervised Wavefunction Optimization (SWO)
    # ==========================================================

    def run_SWO_state_fitting_gpu(
        self,
        fxs: torch.Tensor,
        training_model: WavefunctionModel_GPU,
        target_model: WavefunctionModel_GPU,
        graph: Graph,
        ns_per_rank: int,
        *,
        loss: str = 'fidelity',
        sample_times: int = 10,
        SWO_max_iter: int = int(1e3),
        log_fidelity_tol: float = 1e-4,
        ess_tol: float = 0.1,
        burn_in_steps: int = 0,
        grad_batch_size: Optional[int] = None,
        hamiltonian: Optional[Hamiltonian] = None,
        tmpdir: Optional[str] = None,
        save: bool = True,
        save_step_offset: int = 0,
        scheduler: Optional[Callable[[int], float]] = None,
        verbose: bool = False,
        sr_rshift: float = 0.0,
        sr_ashift: float = 1e-4,
        sr_ratio_clip: Optional[float] = None,
        line_search: bool = False,
        line_search_shrink: float = 0.5,
        line_search_max_backtracks: int = 8,
        line_search_min_lr: float = 1e-8,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Fit ``training_model`` to a frozen ``target_model``.

        Outer loop: MCMC-sample a fresh dataset from the training model
        and freeze it. Inner loop: take up to ``SWO_max_iter`` steps on
        that frozen set, importance-reweighting every estimator back to
        the current Born measure (see `swo_utils`).

        Args:
            fxs: (B, N_sites) int64 initial walker configurations.
            training_model: model being fitted (updated in place).
            target_model: frozen model whose state is the fit target.
            graph: lattice graph consumed by the sampler's proposals.
            ns_per_rank: dataset size per rank per outer step.
            loss: ``'fidelity'`` (plain reweighted gradient of -log F)
                or ``'sr'`` (weighted supervised MinSR). Both are built
                from the same ``(O, r, p)``; they differ only in whether
                the QGT is inverted.
            sample_times: number of outer steps (a fresh dataset is
                sampled for each).
            SWO_max_iter: max inner iterations per frozen dataset.
            log_fidelity_tol: leave the inner loop once the reweighted
                ``-log F < log_fidelity_tol``.
            ess_tol: leave the inner loop when the reweighting has
                degenerated, i.e. ``ESS/N < ess_tol``. ``ESS/N`` is
                ``exp(-D_2(p_theta || p_sample))``, so this is a trust
                region on how far theta may drift before the frozen
                samples stop carrying information -- the analogue of
                PPO's early stopping on KL. Set to 0 to disable.
            burn_in_steps: sampler burn-in sweeps before EACH dataset
                collection (after an inner loop the walkers are still
                equilibrated to the old theta).
            grad_batch_size: chunk size for the gradient / refresh
                passes; None -> the walker batch size ``fxs.shape[0]``.
            hamiltonian: if given, the reweighted energy of
                ``training_model`` is measured after every outer step.
            tmpdir: checkpoint directory (rank 0 writes); None disables.
            save: write checkpoints (requires ``tmpdir``).
            save_step_offset: added to the outer step index for
                checkpoint numbering and the lr ``scheduler`` (resume).
            scheduler: optional ``outer_step -> lr`` callable, applied
                to ``self.optimizer.lr`` at the start of every inner
                loop.
            verbose: rank-0 tqdm bar over the inner iterations.
            sr_rshift, sr_ashift: Gram regularisation for
                ``loss='sr'`` (same convention as ``VMCConfig``).
            sr_ratio_clip: ``loss='sr'`` only -- clamp the per-sample
                amplitude-ratio residual ``eps`` to ``[-clip, clip]``
                (opt-in outlier control on top of the overflow drop
                in ``swo_sr_terms``; None disables).
            line_search: accept a step only if it decreases the
                reweighted ``-log F`` on the frozen set.
            line_search_shrink, line_search_max_backtracks,
                line_search_min_lr: backtracking parameters, see
                ``_swo_line_search_step``.

        Returns:
            fxs: (B, N_sites) final walker configs.
            MC_stats: dict with ``sample_size`` (global dataset size)
                and per-outer-step lists ``-logf`` / ``fidelity`` (both
                measured at the START of the inner loop, i.e. before
                fitting on that dataset), ``fidelity_diagnostics``
                (inner-loop summary dicts from ``_run_swo_inner``),
                ``energy`` (per site, only if ``hamiltonian`` is given)
                and, for ``loss='sr'``, ``sr_time`` / ``sr_info``.
                Lists are populated on rank 0 only.
        """
        loss = loss.lower()
        if loss not in ('fidelity', 'sr'):
            raise ValueError(
                f"Unknown SWO loss {loss!r}; expected 'fidelity' or 'sr'."
            )
        if self.optimizer is None:
            raise ValueError("SWO requires an OptimizerGPU, e.g. SGDGPU().")

        rank = dist.get_rank() if dist.is_initialized() else 0
        device = next(training_model.parameters()).device
        Np = sum(p.numel() for p in training_model.parameters())
        if grad_batch_size is None:
            grad_batch_size = fxs.shape[0]

        MC_stats = {
            'sample_size': ns_per_rank * (
                dist.get_world_size() if dist.is_initialized() else 1
            ),
            '-logf': [],
            'fidelity': [],
            'fidelity_diagnostics': [],
            'energy': [],
        }
        if loss == 'sr':
            MC_stats['sr_time'] = []
            MC_stats['sr_info'] = []

        self._sync_params(training_model)
        self._sync_params(target_model)
        label = {'fidelity': 'SWO', 'sr': 'SWO-SR'}[loss]

        for local_step in range(sample_times):
            t_step = save_step_offset + local_step

            # Burn in on EVERY outer step: after an inner loop the
            # walkers are still equilibrated to the old theta, and the
            # collection loop below only runs ceil(ns/batch) sweeps.
            # An unconverged chain is not a known proposal, so no
            # importance weight can correct for it.
            with torch.no_grad():
                fxs, batch = collect_swo_dataset(
                    self.sampler, fxs,
                    training_model, target_model, graph,
                    ns_per_rank,
                    burn_in=True, burn_in_steps=burn_in_steps,
                )
                init_stats = swo_fidelity(batch)

            if rank == 0:
                print(
                    f"[{label} outer {t_step}] init fidelity="
                    f"{float(init_stats['fidelity']):.6e}, "
                    f"-log f={float(init_stats['neg_log_f']):.6e}"
                )

            inner = self._run_swo_inner(
                batch=batch,
                training_model=training_model,
                loss=loss,
                grad_batch_size=grad_batch_size,
                Np=Np,
                device=device,
                t_step=t_step,
                SWO_max_iter=SWO_max_iter,
                log_fidelity_tol=log_fidelity_tol,
                ess_tol=ess_tol,
                scheduler=scheduler,
                sr_rshift=sr_rshift,
                sr_ashift=sr_ashift,
                sr_ratio_clip=sr_ratio_clip,
                rank=rank,
                verbose=verbose,
                line_search=line_search,
                line_search_shrink=line_search_shrink,
                line_search_max_backtracks=line_search_max_backtracks,
                line_search_min_lr=line_search_min_lr,
            )

            if rank == 0:
                MC_stats['-logf'].append(float(init_stats['neg_log_f']))
                MC_stats['fidelity'].append(float(init_stats['fidelity']))
                MC_stats['fidelity_diagnostics'].append(inner['diagnostics'])
                if loss == 'sr':
                    MC_stats['sr_time'].append(inner['sr_time'])
                    MC_stats['sr_info'].append(inner['sr_info'])

            if hamiltonian is not None:
                # batch is refreshed at the post-update theta, so this
                # is <E_loc>_r and not a stale plain mean.
                energy = swo_energy(
                    batch, training_model, hamiltonian,
                    self.evaluate_energy_fn, batch_size=grad_batch_size,
                )
                per_site = float(energy) / graph.n_nodes
                if rank == 0:
                    MC_stats['energy'].append(per_site)
                    print(
                        f"[{label} outer {t_step}] "
                        f"energy={per_site:.12e}\n"
                    )

            if rank == 0 and tmpdir is not None and save:
                save_swo_checkpoint(
                    training_model, MC_stats, tmpdir, t_step,
                    optimizer=self.optimizer,
                )

        return fxs, MC_stats

    def _run_swo_inner(
        self,
        *,
        batch: SWOBatch,
        training_model: WavefunctionModel_GPU,
        loss: str,
        grad_batch_size: int,
        Np: int,
        device: torch.device,
        t_step: int,
        SWO_max_iter: int,
        log_fidelity_tol: float,
        ess_tol: float,
        scheduler: Optional[Callable[[int], float]],
        sr_rshift: float,
        sr_ashift: float,
        sr_ratio_clip: Optional[float],
        rank: int,
        verbose: bool,
        line_search: bool,
        line_search_shrink: float,
        line_search_max_backtracks: int,
        line_search_min_lr: float,
    ) -> dict[str, Any]:
        """Inner loop over the frozen dataset, shared by both losses.

        Each iteration: refresh the importance weights at the current
        theta, build a direction, decide whether to stop, then step.
        Both stopping tests look at the same reweighted estimator that
        produced the direction, so the monitored quantity and the
        optimized quantity can no longer disagree.

        Args:
            batch: frozen ``SWOBatch``; left refreshed at the final
                theta on return.
            training_model: model being fitted (updated in place).
            loss: ``'fidelity'`` or ``'sr'``.
            grad_batch_size, Np, device: as in the caller.
            t_step: global outer step index (fed to ``scheduler``).
            SWO_max_iter, log_fidelity_tol, ess_tol, scheduler,
                sr_rshift, sr_ashift, sr_ratio_clip, rank, verbose,
                line_search, line_search_shrink,
                line_search_max_backtracks, line_search_min_lr: see
                ``run_SWO_state_fitting_gpu``.

        Returns:
            dict with ``sr_time`` (total seconds in the MinSR solves,
            0.0 for ``'fidelity'``), ``sr_info`` (last solver info or
            None) and ``diagnostics``: ``n_inner`` (iterations run),
            ``exit`` (``'tol'`` / ``'ess'`` / ``'maxiter'``),
            ``neg_log_f_final``, ``fidelity_final``, ``ess_frac_final``,
            ``ess_frac_min``, ``coherence_min`` and ``n_over`` (total
            overflow-dropped rows, ``'sr'`` only).
        """
        if scheduler is not None:
            self.optimizer.lr = scheduler(t_step)
        self.optimizer.reset()

        pbar = tqdm(range(SWO_max_iter)) if rank == 0 and verbose else None
        total_sr_time = 0.0
        last_info = None
        ess_min = float('inf')
        coh_min = float('inf')
        n_over_total = 0.0
        stats = None
        exit_reason = 'maxiter'
        n_inner = 0

        for it in range(SWO_max_iter):
            if loss == 'fidelity':
                batch.refresh_from_model(training_model, grad_batch_size)
                direction, stats = swo_fidelity_gradient(
                    batch, training_model,
                    grad_batch_size=grad_batch_size, Np=Np, device=device,
                )
            else:
                # swo_sr_terms refreshes `batch` out of its own gradient
                # pass, so the SR path pays no extra forward.
                terms = swo_sr_terms(
                    batch, training_model,
                    grad_batch_size=grad_batch_size,
                    ratio_clip=sr_ratio_clip,
                )
                stats = terms['stats']
                direction, sr_time, info = weighted_minsr_step(
                    terms['O_loc'], terms['eps'], terms['r'],
                    batch.n_total, Np,
                    rshift=sr_rshift, ashift=sr_ashift, device=device,
                )
                total_sr_time += sr_time
                last_info = info
                n_over_total += float(stats['n_over'])
                del terms
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            n_inner = it + 1
            neg_log_f = float(stats['neg_log_f'])
            ess = float(stats['ess_frac'])
            coh = float(stats['coherence'])
            ess_min = min(ess_min, ess)
            coh_min = min(coh_min, coh)

            if rank == 0 and pbar is not None:
                pbar.set_description(
                    f"{loss} {it} -logF={neg_log_f:.3e} "
                    f"ESS/N={ess:.3f} coh={coh:.3f}"
                )
                pbar.update(1)

            if neg_log_f < log_fidelity_tol:
                exit_reason = 'tol'
                break
            if ess_tol > 0.0 and ess < ess_tol:
                # The frozen samples no longer represent the current
                # state; this direction is noise. Stop and resample
                # rather than take it.
                exit_reason = 'ess'
                if rank == 0:
                    print(
                        f"[SWO inner] ESS/N={ess:.3f} < {ess_tol} at "
                        f"iter {it}; resampling."
                    )
                break
            if rank == 0 and coh < 0.05:
                # <A|B> is a near-cancelling signed sum, so the overlap
                # weight p -- and hence the direction -- is amplified by
                # ~1/coherence. Report it; clamping would silently bend
                # the direction instead.
                print(
                    f"[SWO inner] warning: coherence={coh:.3e} at iter "
                    f"{it}; <A|B> is dominated by sign cancellation."
                )

            if line_search:
                self._swo_line_search_step(
                    batch=batch,
                    training_model=training_model,
                    direction=direction,
                    grad_batch_size=grad_batch_size,
                    max_backtracks=line_search_max_backtracks,
                    shrink=line_search_shrink,
                    min_lr=line_search_min_lr,
                    rank=rank,
                )
            else:
                self.optimizer.step(training_model, direction)

        if pbar is not None:
            pbar.close()

        # Leave `batch` refreshed at the final theta so the caller's
        # energy measurement reweights against the right state.
        batch.refresh_from_model(training_model, grad_batch_size)
        final = swo_fidelity(batch)

        return {
            'sr_time': float(total_sr_time),
            'sr_info': None if last_info is None else int(last_info),
            'diagnostics': {
                'n_inner': n_inner,
                'exit': exit_reason,
                'neg_log_f_final': float(final['neg_log_f']),
                'fidelity_final': float(final['fidelity']),
                'ess_frac_final': float(final['ess_frac']),
                'ess_frac_min': (
                    None if ess_min == float('inf') else ess_min
                ),
                'coherence_min': (
                    None if coh_min == float('inf') else coh_min
                ),
                'n_over': n_over_total,
            },
        }


__all__ = [
    "setup_distributed",
    "print_sampling_settings",
    "VMC_GPU",
]
