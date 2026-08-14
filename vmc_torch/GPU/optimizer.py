import functools
import math
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import numpy as np
import scipy.sparse.linalg as spla
import time
from vmc_torch.GPU.torch_utils import (
    torch_minres,
)

# ============================================================
#  Learning rate schedulers
# ============================================================


class Scheduler:
    """Base LR scheduler: callable(step) -> learning_rate."""

    def __init__(self, init_lr=1e-3):
        self.init_lr = init_lr

    def __call__(self, step):
        raise NotImplementedError


class TrivialScheduler(Scheduler):
    """Constant learning rate."""

    def __call__(self, step):
        return self.init_lr


def continuous_exp_decay(t, patience=50, init_lr=5e-2, rate=0.85):
    return init_lr * math.exp(-math.log(1 / rate) * t / patience)


def discrete_exp_decay(t, patience=50, init_lr=5e-2, rate=0.85):
    return init_lr * rate ** (t // patience)


def polynomial_decay(t, max_iter=1000, init_lr=5e-2, power=1.0):
    return init_lr * (1 - t / max_iter) ** power


def cosine_decay(t, max_iter=1000, init_lr=5e-2):
    return init_lr * 0.5 * (1 + math.cos(math.pi * t / max_iter))


def exponential_decay(t, decay_rate=0.1, decay_step=1, init_lr=5e-2):
    return init_lr * math.exp(-decay_rate * (t / decay_step))


def linear_decay(t, max_iter=1000, init_lr=5e-2):
    return init_lr * (1 - t / max_iter)


class DecayScheduler(Scheduler):
    """Configurable LR decay scheduler.

    Args:
        init_lr: initial learning rate.
        decay_rate: decay rate (meaning depends on type).
        patience: steps between discrete decays.
        min_lr: floor for learning rate.
        type: one of 'continuous_exp', 'discrete_exp', 'polynomial',
              'cosine', 'exponential', 'linear'.
        **kwargs: forwarded to the decay function (e.g. max_iter).
    """

    def __init__(
        self,
        init_lr=1e-3,
        decay_rate=0.9,
        patience=100,
        min_lr=1e-4,
        type='continuous_exp',
        **kwargs,
    ):
        super().__init__(init_lr)
        self.min_lr = min_lr
        if type == 'discrete_exp':
            self.decay_func = functools.partial(
                discrete_exp_decay,
                init_lr=init_lr, rate=decay_rate,
                patience=patience,
            )
        elif type == 'continuous_exp':
            self.decay_func = functools.partial(
                continuous_exp_decay,
                init_lr=init_lr, rate=decay_rate,
                patience=patience,
            )
        elif type == 'polynomial':
            self.decay_func = functools.partial(
                polynomial_decay,
                init_lr=init_lr, power=1 / decay_rate,
                **kwargs,
            )
        elif type == 'cosine':
            self.decay_func = functools.partial(
                cosine_decay, init_lr=init_lr, **kwargs,
            )
        elif type == 'exponential':
            self.decay_func = functools.partial(
                exponential_decay,
                init_lr=init_lr, decay_rate=decay_rate,
                **kwargs,
            )
        elif type == 'linear':
            self.decay_func = functools.partial(
                linear_decay, init_lr=init_lr, **kwargs,
            )
        else:
            raise ValueError(f"Unknown decay type: {type}")

    def __call__(self, step):
        return max(self.decay_func(step), self.min_lr)


class OptimizerGPU:
    """Base optimizer interface for updating model parameters from a direction."""

    # Subclasses with persistent state (momenta, step counter, etc.)
    # override this tuple to list the attribute names that should be
    # round-tripped through checkpoints. Stateless optimizers (e.g.
    # plain SGD) leave it empty and ``state_dict()`` returns ``{}``.
    _STATE_ATTRS: tuple = ()

    def __init__(self, learning_rate: float = 1e-3):
        self.lr = learning_rate

    def state_dict(self) -> dict:
        """Snapshot stateful attributes for checkpointing."""
        return {a: getattr(self, a) for a in self._STATE_ATTRS}

    def load_state_dict(self, state: dict) -> None:
        """Restore stateful attributes from a snapshot."""
        for a, v in state.items():
            setattr(self, a, v)

    def compute_update(
        self,
        params_vec: torch.Tensor,
        direction_vec: torch.Tensor,
        learning_rate: Optional[float] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def step(
        self,
        model,
        direction,
        device: Optional[torch.device] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        with torch.no_grad():
            # reshape(-1) (not parameters_to_vector, whose .view(-1)
            # fails on non-contiguous params) so symmray-backed TN
            # params (train_tn=True) flatten without error.
            current = torch.cat(
                [p.reshape(-1) for p in model.parameters()]
            )
            model_dtype = current.dtype
            target_device = current.device if device is None else device
            # Compute update in float64 for numerical accuracy,
            # then cast result back to model dtype (no-op for f64 models).
            current_f64 = current.to(torch.float64)
            direction_t = torch.as_tensor(
                direction,
                device=target_device,
                dtype=torch.float64,
            )
            updated_f64 = self.compute_update(
                current_f64,
                direction_t,
                learning_rate=learning_rate,
            )
            updated = updated_f64.to(model_dtype)
            # Copy values in-place to preserve storage identity.
            # vector_to_parameters replaces .data with views of one
            # big tensor, which changes storage layout and triggers
            # torch.compile recompilation.
            offset = 0
            for p in model.parameters():
                n = p.numel()
                p.data.copy_(updated[offset:offset + n].view_as(p.data))
                offset += n

    def reset(self) -> None:
        pass


class SGDGPU(OptimizerGPU):
    """SGD with optional Euclidean norm constraint.

    If ``norm_constraint`` is not None, clip the effective learning
    rate per step so that ``||dp_applied|| <= sqrt(norm_constraint)``.
    Concretely, ``eff_lr = min(lr, sqrt(C) / ||direction||)``.  This
    implements the SPRING paper's Eq. 37 norm bound
    (arXiv:2401.10190): ``d_theta = phi * min(eta, sqrt(C)/||phi||)``.

    When ``norm_constraint=None`` (default), this reduces to plain SGD
    and is bit-equivalent to the previous behavior.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        norm_constraint: Optional[float] = None,
    ):
        super().__init__(learning_rate=learning_rate)
        self.norm_constraint = norm_constraint

    def compute_update(
        self,
        params_vec: torch.Tensor,
        direction_vec: torch.Tensor,
        learning_rate: Optional[float] = None,
    ) -> torch.Tensor:
        lr = self.lr if learning_rate is None else learning_rate
        if self.norm_constraint is not None:
            direction_norm = torch.linalg.vector_norm(
                direction_vec
            ).item()
            if direction_norm > 0:
                lr = min(
                    lr,
                    (self.norm_constraint ** 0.5) / direction_norm,
                )
        return params_vec - lr * direction_vec


class AdamGPU(OptimizerGPU):
    _STATE_ATTRS = ('t', 'm', 'v')

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.m = None
        self.v = None

    def compute_update(
        self,
        params_vec: torch.Tensor,
        direction_vec: torch.Tensor,
        learning_rate: Optional[float] = None,
    ) -> torch.Tensor:
        lr = self.lr if learning_rate is None else learning_rate
        grad = direction_vec
        if self.weight_decay != 0.0:
            grad = grad + self.weight_decay * params_vec
        if self.m is None:
            self.m = torch.zeros_like(grad)
            self.v = torch.zeros_like(grad)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        update = lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        return params_vec - update

    def reset(self) -> None:
        self.t = 0
        self.m = None
        self.v = None


def _two_term_shift(trace_gram, n, rshift, ashift):
    """Two-term Tikhonov shift for an (n, n) Gram matrix.

    ``shift = rshift * trace(Gram) / sqrt(n) + ashift``

    ``rshift`` scales with the magnitude of the Gram (relative
    term); ``ashift`` is an absolute floor. ``trace_gram`` must be
    the trace of the matrix actually being shifted.
    """
    return rshift * trace_gram / math.sqrt(n) + ashift


class PreconditionerGPU:
    """Base preconditioner interface for solving gradient update directions."""

    # See OptimizerGPU._STATE_ATTRS for semantics. Stateful subclasses
    # (SPRING/MARCH/AdamSR variants) list their momentum buffers here
    # so checkpoints can round-trip them.
    _STATE_ATTRS: tuple = ()

    # Absolute-value outlier mask threshold on local energies. When
    # set, samples with ``|E_loc| > local_E_clip`` are entirely
    # masked from the SR solve: both their E_loc AND their O_loc row
    # are zeroed, the global mean is recomputed from clean samples,
    # and the effective sample count drops accordingly. ``None``
    # disables. The outer VMC main loop is NOT affected -- the
    # energy displayed / saved to stats is still the unclipped MC
    # mean (outliers are signal about the wavefunction, not noise
    # to be hidden).
    local_E_clip: Optional[float] = None

    # Populated by :meth:`_mask_outlier_samples` on each ``solve``
    # call when ``local_E_clip`` is set. ``None`` if clip disabled
    # or no samples were masked.
    _last_mask_info: Optional[dict] = None

    def state_dict(self) -> dict:
        """Snapshot stateful attributes for checkpointing."""
        return {a: getattr(self, a) for a in self._STATE_ATTRS}

    def load_state_dict(self, state: dict) -> None:
        """Restore stateful attributes from a snapshot."""
        for a, v in state.items():
            setattr(self, a, v)

    def _mask_outlier_samples(self, E_loc, O_loc, E_mean, Ns):
        """Zero out (E_loc, O_loc) rows that are non-finite, or — when
        ``local_E_clip`` is set — have ``|E_loc| > clip``.

        Non-finite masking (NaN/Inf in E, or any NaN/Inf entry in the
        O row) is ALWAYS on: fp32 forward/backward can emit NaN/Inf
        gradients that would otherwise poison the whole Ns x Ns Gram
        (T = O Oᵀ goes all-NaN, and the diagonal shift — applied after
        T — cannot rescue it). It also fixes an ``E_mean`` that was
        already NaN-poisoned upstream, by recomputing the clean mean.
        ``local_E_clip`` is an additional opt-in energy-outlier filter.

        Returns ``(E_loc_clean, O_loc_clean, E_mean_clean, Ns_clean)``.
        Masked samples contribute 0 to ``<E*O>``, ``<O>``, and to the
        QGT ``<O O>``, so direction and centering stay self-consistent
        over the clean subset. ``E_mean_clean`` and ``Ns_clean`` are
        aggregated across MPI ranks; ``self._last_mask_info`` carries
        ``n_masked`` (global) and ``E_mean_clean`` for downstream
        diagnostic printing (``None`` when nothing is masked).
        """
        clip = getattr(self, 'local_E_clip', None)

        # Local mask (per-rank): always drop non-finite rows; optionally
        # also drop |E| > clip outliers. A non-finite entry anywhere in
        # an O row makes its sum(dim=1) non-finite.
        E_t = E_loc if isinstance(E_loc, torch.Tensor) else (
            torch.as_tensor(E_loc, dtype=torch.float64)
        )
        # O_loc may be offloaded to a different device than E_loc (e.g.
        # CPU when offload_grad_to_cpu=True); align its finiteness check
        # to E_t's device before combining to avoid a device mismatch.
        o_nonfinite = (~torch.isfinite(O_loc.sum(dim=1))).to(E_t.device)
        nonfinite = ~torch.isfinite(E_t) | o_nonfinite
        mask = nonfinite if clip is None else (
            nonfinite | (E_t.abs() > clip)
        )

        # Aggregate (total masked, non-finite) counts globally in one
        # all_reduce; bail early if nothing was masked (common case).
        counts = torch.stack([mask.sum(), nonfinite.sum()]).long()
        if self._world_size() > 1:
            counts = counts.to(E_t.device if E_t.is_cuda else 'cpu')
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        n_masked_global, n_nonfinite_global = counts.tolist()

        if n_masked_global == 0:
            self._last_mask_info = None
            return E_loc, O_loc, E_mean, Ns

        # Zero out masked entries on E_loc and the matching O_loc rows.
        E_clean_t = E_t.clone()
        E_clean_t[mask] = 0.0

        O_loc[mask.to(O_loc.device)] = 0.0
        O_clean = O_loc

        # Globally recompute clean mean / sample count
        sum_E_clean_local = E_clean_t.sum()
        if self._world_size() > 1:
            sum_E_clean_t = sum_E_clean_local.to(
                E_t.device if E_t.is_cuda else 'cpu',
            )
            dist.all_reduce(sum_E_clean_t, op=dist.ReduceOp.SUM)
            sum_E_clean = float(sum_E_clean_t.item())
        else:
            sum_E_clean = float(sum_E_clean_local.item())
        Ns_clean = Ns - n_masked_global
        E_mean_clean = (
            sum_E_clean / Ns_clean if Ns_clean > 0 else float(E_mean)
        )

        self._last_mask_info = {
            'n_masked': n_masked_global,
            'n_nonfinite': n_nonfinite_global,
            'Ns_clean': Ns_clean,
            'Ns_total': Ns,
            'E_mean_clean': E_mean_clean,
            'clip': clip,
        }

        # Preserve container type for E_loc.
        if isinstance(E_loc, torch.Tensor):
            out_E = E_clean_t
        else:
            out_E = E_clean_t.cpu().numpy()
        return out_E, O_clean, E_mean_clean, Ns_clean

    def solve(
        self,
        *,
        O_loc,
        E_loc,
        E_mean: float,
        Ns: int,
        Np: int,
        rshift: float,
        ashift: float,
        device: torch.device,
    ) -> Tuple[Any, float, Any]:
        """Public entry: optionally mask outlier samples (when
        ``local_E_clip`` is set) and then delegate to subclass
        :meth:`_solve`. Subclasses should override ``_solve`` only.

        ``O_loc`` may be a tensor, or a 1-element "ownership box"
        ``[tensor]`` (handed down by ``run_vmc_loop`` for
        preconditioners with ``_supports_ownership_box``). When a box
        is given we unwrap it for masking, then forward a *fresh* box
        to ``_solve`` and drop the tensor reference from this frame —
        so ``_solve`` can free the (Ns, Np) Jacobian internally
        instead of it lingering here for the whole solve. Plain-tensor
        callers are forwarded unchanged.
        """
        received_box = isinstance(O_loc, list)
        O_tensor = O_loc[0] if received_box else O_loc
        if received_box:
            O_loc[0] = None  # drop the incoming box's reference

        # Masking zeroes outlier rows in-place but keeps the tensor
        # lengths, so the *physical* sample count ``Ns`` is unchanged —
        # only the statistics use the clean count (recorded in
        # ``_last_mask_info`` for reporting). We keep ``Ns`` physical so
        # the distributed gather / Gram dimension stays consistent
        # (sizing by the clean count would mismatch the full-length
        # E_scaled rows), and use the clean ``E_mean`` for centering.
        E_loc, O_tensor, E_mean, _Ns_clean = self._mask_outlier_samples(
            E_loc, O_tensor, E_mean, Ns,
        )

        if received_box:
            O_arg = [O_tensor]
            del O_tensor  # this frame no longer references the tensor
        else:
            O_arg = O_tensor

        return self._solve(
            O_loc=O_arg,
            E_loc=E_loc,
            E_mean=E_mean,
            Ns=Ns,
            Np=Np,
            rshift=rshift,
            ashift=ashift,
            device=device,
        )

    def _solve(
        self,
        *,
        O_loc,
        E_loc,
        E_mean: float,
        Ns: int,
        Np: int,
        rshift: float,
        ashift: float,
        device: torch.device,
    ) -> Tuple[Any, float, Any]:
        raise NotImplementedError

    @staticmethod
    def _rank() -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    @staticmethod
    def _world_size() -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    @staticmethod
    def _device(device: Optional[torch.device]) -> torch.device:
        return torch.device('cuda') if device is None else device

    @staticmethod
    def _to_f64_tensor(
        x,
        device: torch.device,
        *,
        contiguous: bool = False,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        # Ownership box: a 1-element list ``[tensor]`` handed down by
        # solve() so the Jacobian can be released inside _solve.
        # Unwrap and null the box here so this becomes the only
        # remaining reference to the underlying tensor.
        # ``dtype`` defaults to fp64 (existing callers unchanged); pass
        # fp32 to keep the tensor in single precision (e.g. to do the
        # communication in fp32 and upcast only afterwards).
        if isinstance(x, list):
            t = x[0]
            x[0] = None
            x = t
        if isinstance(x, torch.Tensor):
            y = x.to(device=device, dtype=dtype)
        else:
            y = torch.as_tensor(x, device=device, dtype=dtype)
        return y.contiguous() if contiguous else y

    def _energy_gradient(
        self,
        O_loc: torch.Tensor,
        E_loc: torch.Tensor,
        E_mean: float,
        Ns: int,
        Np: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        n_local = E_loc.shape[0]
        if n_local > 0:
            O_sum = O_loc.sum(dim=0)
            EO_sum = E_loc @ O_loc
        else:
            O_sum = torch.zeros(Np, device=device, dtype=torch.float64)
            EO_sum = torch.zeros(Np, device=device, dtype=torch.float64)

        if self._world_size() > 1:
            dist.all_reduce(O_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(EO_sum, op=dist.ReduceOp.SUM)

        O_mean = O_sum / Ns
        EO_mean = EO_sum / Ns
        return EO_mean - E_mean * O_mean, O_mean, n_local

    def _O_mean(
        self,
        O_loc: torch.Tensor,
        Ns: int,
    ):
        O_sum = O_loc.sum(dim=0)
        if self._world_size() > 1:
            dist.all_reduce(O_sum, op=dist.ReduceOp.SUM)
        return O_sum / Ns

    def _solve_cholesky(
        self,
        T: torch.Tensor,
        rhs: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Solve the SPD system T alpha = rhs by Cholesky.

        No PD check / fallback: T is assumed positive definite after the
        diagonal shift (matches Ao's ``solve(assume_a="pos")``). The
        returned int is always 0, kept only for caller-signature
        compatibility.
        """
        L, _ = torch.linalg.cholesky_ex(T)
        alpha = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        return alpha, 0
    
class TrivialPreconditionerGPU(PreconditionerGPU):
    """No preconditioning: return the raw energy gradient as the update direction."""

    def _solve(
        self,
        *,
        O_loc,
        E_loc,
        E_mean: float,
        Ns: int,
        Np: int,
        rshift: float,
        ashift: float,
        device: torch.device,
    ) -> Tuple[Any, float, Any]:
        t0 = time.time()
        device = self._device(device)
        O_loc = self._to_f64_tensor(O_loc, device)
        E_loc = self._to_f64_tensor(E_loc, device)
        energy_grad, _, _ = self._energy_gradient(
            O_loc, E_loc, E_mean, Ns, Np, device,
        )
        return energy_grad, time.time() - t0, None


class IterSRGPU(PreconditionerGPU):
    """Distributed SR in Np-form using MINRES.

    Matrix-vector products for
    ``S = O.T @ O / Ns - O_mean O_mean.T + shift I`` are formed
    locally and summed with ``all_reduce``; the dense ``Np x Np``
    matrix is never built. The Tikhonov shift is two-term:
    ``shift = rshift * trace(S)/sqrt(Np) + ashift``.
    """

    def __init__(
        self,
        rtol: float = 5e-5,
        maxiter: int = 100,
        use_scipy: bool = False,
    ):
        self.rtol = rtol
        self.maxiter = maxiter
        self.use_scipy = use_scipy

    def _matvec(
        self,
        x: torch.Tensor,
        O_loc: torch.Tensor,
        O_mean: torch.Tensor,
        Ns: int,
        n_local: int,
        shift: float,
    ) -> torch.Tensor:
        if n_local > 0:
            Sx = O_loc.T @ (O_loc @ x)
        else:
            Sx = torch.zeros_like(x)
        if self._world_size() > 1:
            dist.all_reduce(Sx, op=dist.ReduceOp.SUM)
        Sx /= Ns
        Sx -= torch.dot(O_mean, x) * O_mean
        return Sx + shift * x

    def _np_gram_trace(
        self,
        O_loc: torch.Tensor,
        O_mean: torch.Tensor,
        Ns: int,
    ) -> float:
        """Trace of the implicit Np-form Gram
        ``S = O^T O / Ns - O_mean O_mean^T`` (matvec form, the
        matrix is never built)."""
        tr = O_loc.pow(2).sum()
        if self._world_size() > 1:
            dist.all_reduce(tr, op=dist.ReduceOp.SUM)
        return tr.item() / Ns - O_mean.pow(2).sum().item()

    def _rhs(
        self,
        energy_grad: torch.Tensor,
        *,
        shift: float,
        device: torch.device,
    ) -> torch.Tensor:
        return energy_grad

    def _solve_scipy(
        self,
        *,
        O_loc,
        E_loc,
        E_mean: float,
        Ns: int,
        Np: int,
        rshift: float,
        ashift: float,
        device: torch.device,
        t0: float,
    ) -> Tuple[Any, float, Any]:
        # The scipy path builds its RHS from the bare energy gradient and
        # does NOT apply the self._rhs hook. Momentum variants (SPRING)
        # override _rhs to inject their momentum term, which would be
        # silently dropped here -- warn if a subclass overrode it.
        if type(self)._rhs is not IterSRGPU._rhs and self._rank() == 0:
            print(
                f"Warning: {type(self).__name__} with use_scipy=True "
                f"ignores the _rhs momentum hook; the momentum term is "
                f"NOT applied on the scipy MINRES path.",
                flush=True,
            )
        if isinstance(O_loc, torch.Tensor):
            assert O_loc.device.type == 'cpu', (
                "use_scipy=True requires O_loc on CPU"
            )
            O_np = O_loc.numpy()
        else:
            O_np = np.asarray(O_loc, dtype=np.float64)

        E_np = (
            E_loc.cpu().numpy()
            if isinstance(E_loc, torch.Tensor)
            else np.asarray(E_loc, dtype=np.float64)
        )
        n_local = E_np.shape[0]
        if n_local > 0:
            O_sum = O_np.sum(axis=0)
            EO_sum = E_np @ O_np
        else:
            O_sum = np.zeros(Np, dtype=np.float64)
            EO_sum = np.zeros(Np, dtype=np.float64)

        if self._world_size() > 1:
            O_sum_t = torch.tensor(O_sum, device=device)
            EO_sum_t = torch.tensor(EO_sum, device=device)
            dist.all_reduce(O_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(EO_sum_t, op=dist.ReduceOp.SUM)
            O_sum = O_sum_t.cpu().numpy()
            EO_sum = EO_sum_t.cpu().numpy()

        O_mean = O_sum / Ns
        energy_grad = EO_sum / Ns - E_mean * O_mean

        # Two-term Tikhonov shift from the implicit Gram trace:
        # trace(S) = ||O||_F^2 / Ns - ||O_mean||^2.
        tr_local = float((O_np ** 2).sum())
        if self._world_size() > 1:
            tr_t = torch.tensor(tr_local, device=device)
            dist.all_reduce(tr_t, op=dist.ReduceOp.SUM)
            tr_local = tr_t.item()
        trace_S = tr_local / Ns - float(O_mean @ O_mean)
        shift = _two_term_shift(trace_S, Np, rshift, ashift)

        def matvec(x):
            if n_local > 0:
                Sx_local = O_np.T.dot(O_np.dot(x))
            else:
                Sx_local = np.zeros_like(x)
            if self._world_size() > 1:
                Sx_t = torch.tensor(Sx_local, device=device)
                dist.all_reduce(Sx_t, op=dist.ReduceOp.SUM)
                Sx = Sx_t.cpu().numpy()
            else:
                Sx = Sx_local
            Sx /= Ns
            Sx -= np.dot(O_mean, x) * O_mean
            return Sx + shift * x

        A = spla.LinearOperator((Np, Np), matvec=matvec, dtype=np.float64)
        dp, info = spla.minres(
            A, energy_grad, rtol=self.rtol, maxiter=self.maxiter,
        )
        return dp, time.time() - t0, info

    def _solve(
        self,
        *,
        O_loc,
        E_loc,
        E_mean: float,
        Ns: int,
        Np: int,
        rshift: float,
        ashift: float,
        device: torch.device,
    ) -> Tuple[Any, float, Any]:
        t0 = time.time()
        device = self._device(device)
        if self.use_scipy:
            return self._solve_scipy(
                O_loc=O_loc,
                E_loc=E_loc,
                E_mean=E_mean,
                Ns=Ns,
                Np=Np,
                rshift=rshift,
                ashift=ashift,
                device=device,
                t0=t0,
            )

        O_loc = self._to_f64_tensor(O_loc, device)
        E_loc = self._to_f64_tensor(E_loc, device)
        
        O_bar_T_E_bar, O_mean, n_local = self._energy_gradient(
            O_loc, E_loc, E_mean, Ns, Np, device,
        )
        
        O_loc.sub_(O_mean.unsqueeze(0)).div_(math.sqrt(Ns))
        O_bar_loc = O_loc
        # E_bar_loc = (E_loc - E_mean) / math.sqrt(Ns)

        # Two-term Tikhonov shift; the matvec Gram is
        # S = Obar^T Obar (Np x Np), trace(S) = ||Obar||_F^2.
        tr = O_bar_loc.pow(2).sum()
        if self._world_size() > 1:
            dist.all_reduce(tr, op=dist.ReduceOp.SUM)
        shift = _two_term_shift(tr.item(), Np, rshift, ashift)

        # RHS via the polymorphic hook: plain IterSR returns the bare
        # energy gradient; SPRING adds its mu*shift*phi_prev momentum
        # term (Goldshlager et al. 2024). shift is needed here, so this
        # must come after the shift computation above.
        rhs = self._rhs(O_bar_T_E_bar, shift=shift, device=device)

        def matvec(x):
            if n_local > 0:
                Sx_local = O_bar_loc.T @ (O_bar_loc @ x)
            else:
                Sx_local = torch.zeros_like(x)
            Sx = Sx_local
            if self._world_size() > 1:
                dist.all_reduce(Sx, op=dist.ReduceOp.SUM)
            Rx = Sx + shift * x
            return Rx
        
        dp, info = torch_minres(
            matvec,
            b = rhs,
            rtol=self.rtol,
            maxiter=self.maxiter,
        )
        return dp, time.time() - t0, info


class MinSRGPU(PreconditionerGPU):
    r"""
    
    MinSR <https://www.nature.com/articles/s41567-024-02566-1>
    
    Solves :math:`\bar O \dot \theta = \bar \epsilon` in the N_s-form,
    
    where :math:`\bar O = \frac{1}{\sqrt{N_s}}(\frac{1}{\psi} \frac{\partial \psi}{\partial \theta} - \left< \frac{1}{\psi} \frac{\partial \psi}{\partial \theta} \right>)`.
    
    
    Distributed via all_to_all column redistribution.

    Each rank's (B, Np) gradient block is transposed across ranks
    with a single `dist.all_to_all_single` so every rank holds an
    (Ns_total, Np_per_rank) column-block.  The (Ns x Ns) Gram is
    then assembled by a local matmul and one all_reduce SUM, and
    the regularized SPD system is solved via Cholesky.  Per-rank
    peak GPU memory is roughly max(Ns^2, Ns * Np / world_size).

    Implements Eq. 10 + Eq. 17 of Rende et al. 2024 (the
    `parallel_minSR` paper).
    """

    def __init__(self, solver: str = 'direct'):
        """
        Args:
            solver: how the Ns x Ns Gram system is solved.

              - ``'direct'`` (default): shift the diagonal by
                ``rshift * trace(T)/sqrt(Ns) + ashift`` and run a
                Cholesky solve.  Every direction is inverted down to
                that shift, including ones whose eigenvalue is pure
                sampling noise.
              - ``'pinv_eig'``: eigendecomposition pseudo-inverse with
                a SMOOTH relative eigenvalue cutoff at
                ``rtol = rshift``; ``ashift`` is ignored.  Directions
                below the cutoff are suppressed instead of amplified,
                which is the robust choice when the Gram is
                rank-deficient (``Ns > Np``) or its small eigenvalues
                are noise-dominated.

            Same formula and semantics as
            ``AdamSRMinSRGPU(solver=...)``.
        """
        if solver not in ('direct', 'pinv_eig'):
            raise ValueError(
                f"unknown solver {solver!r}; expected "
                "'direct' or 'pinv_eig'"
            )
        self.solver = solver

    @staticmethod
    def _solve_pinv_eig(T, b, rtol):
        """Pseudo-inverse solve of a symmetric PSD Gram via ``eigh``.

        Eigenvalues below ``rtol * lam_max`` are damped smoothly by
        ``1 + (cutoff/lam)**6`` rather than hard-truncated, so the
        update direction varies continuously as the spectrum drifts
        between VMC steps.  Exact zeros map to zero (the ``where``
        guard; the unselected NaN branch is discarded).

        Args:
            T: (Ns, Ns) symmetric PSD Gram matrix.
            b: (Ns,) right-hand side.
            rtol: relative eigenvalue cutoff (``rshift``).

        Returns:
            (Ns,) solution vector.
        """
        evals, U = torch.linalg.eigh(T)
        evals_abs = evals.abs()
        cutoff = rtol * evals_abs.max()
        inv_factor = 1.0 + (cutoff / evals_abs) ** 6
        evals_inv = 1.0 / (evals * inv_factor)
        evals_inv = torch.where(
            evals_abs > 0.0,
            evals_inv,
            torch.zeros_like(evals_inv),
        )
        return U @ (evals_inv * (U.T @ b))

    def _solve(
        self,
        *,
        O_loc,
        E_loc,
        E_mean: float,
        Ns: int,
        Np: int,
        rshift: float,
        ashift: float,
        device: torch.device,
    ) -> Tuple[Any, float, Any]:
        """Solve the MinSR linear system in the Ns-form (Eq. 17 of
        Rende et al. 2024) and return a parameter-update direction.

        Args:
            O_loc: (n_local, Np) per-sample log-derivative matrix
                ``d log psi / d theta`` for this rank's samples,
                uncentered. Will be centered and scaled by
                ``1/sqrt(Ns)`` in place to form ``Obar``. Use case:
                - VMC: ``O_loc[s] = d log psi(x_s) / d theta``
                - Supervised SWO: same, with x_s drawn from |psi_A|^2
            E_loc: (n_local,) per-sample "signal" values for this
                rank, uncentered. Will be centered against
                ``E_mean`` and scaled by ``1/sqrt(Ns)`` to form
                ``Ebar``. Use case:
                - VMC: local energies ``E_loc(x_s)``
            E_mean: globally reduced mean of ``E_loc`` across ALL
                ranks. The caller is responsible for this reduction.
            Ns: total number of samples across all ranks
                (``sum_r n_local_r``). Used for the ``1/sqrt(Ns)``
                normalization and for the all_gather sizing.
            Np: total number of model parameters. Must match
                ``O_loc.shape[1]``. Used to pad columns to a
                multiple of ``world_size`` for the all_to_all
                column redistribution.
            rshift: relative Tikhonov regularization; together
                with ``ashift`` the diagonal of the Ns x Ns Gram
                is shifted by
                ``rshift * trace(T)/sqrt(Ns) + ashift`` before
                the Cholesky solve. Stabilizes ill-conditioned
                systems. Under ``solver='pinv_eig'`` it instead
                means the RELATIVE eigenvalue cutoff
                (``rtol``) of the pseudo-inverse.
            ashift: absolute Tikhonov shift (see ``rshift``).
                Ignored when ``solver='pinv_eig'``.
            device: target device for all tensors. ``None`` defaults
                to ``cuda``.

        Returns:
            (dp, elapsed_time, info) tuple:
                dp: (Np,) float64 update direction on ``device``.
                elapsed_time: wall-clock time of the solve, seconds.
                info: Cholesky info code from
                    ``_solve_cholesky`` (always 0; kept for
                    caller-signature compatibility).
        """
        t0 = time.time()
        device = self._device(device)
        world_size = self._world_size()
        O_loc = self._to_f64_tensor(O_loc, device, contiguous=True)
        E_loc = self._to_f64_tensor(E_loc, device)

        O_mean = self._O_mean(O_loc, Ns)
        n_local = E_loc.shape[0]

        sqrt_Ns = math.sqrt(Ns)
        O_loc.sub_(O_mean.unsqueeze(0)).div_(sqrt_Ns)
        E_scaled = (E_loc - E_mean) / sqrt_Ns
        
        if world_size > 1:
            E_bar = torch.empty(Ns, device=device, dtype=torch.float64)
            dist.all_gather_into_tensor(E_bar, E_scaled)
        else:
            E_bar = E_scaled

        if world_size > 1:
            np_per_rank = (Np + world_size - 1) // world_size
            np_pad = world_size * np_per_rank
            if np_pad != Np:
                O_padded = torch.zeros(
                    (n_local, np_pad),
                    device=device,
                    dtype=torch.float64,
                )
                O_padded[:, :Np].copy_(O_loc)
                del O_loc
            else:
                O_padded = O_loc
                O_loc = None

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

            O_bar = recv_buf.reshape(
                world_size * n_local, np_per_rank,
            )  # shape (Ns, np_per_rank)
            del recv_buf
            T = O_bar @ O_bar.T
            dist.all_reduce(T, op=dist.ReduceOp.SUM)  # T shape (Ns, Ns)
        else:
            np_pad = Np
            np_per_rank = Np
            O_bar = O_loc
            T = O_bar @ O_bar.T  # shape (Ns, Ns) T = O@O.T
        
        # Solve  alpha = (T + shift I)^{-1} E_bar, or the eigh
        # pseudo-inverse of T when solver='pinv_eig'.
        if getattr(self, 'solver', 'direct') == 'pinv_eig':
            alpha = self._solve_pinv_eig(T, E_bar, rshift)
            info = 0
        else:
            shift = _two_term_shift(
                T.trace().item(), T.shape[0], rshift, ashift,
            )
            T_shifted = T + shift * torch.eye(T.shape[0], device=device)
            alpha, info = self._solve_cholesky(T_shifted, E_bar)
        del T

        dp_local = O_bar.T @ alpha  # shape (np_per_rank,)

        if world_size > 1:
            del O_bar
            dp_padded = torch.empty(np_pad, device=device, dtype=torch.float64)
            dist.all_gather_into_tensor(dp_padded, dp_local)
            dp = dp_padded[:Np].contiguous()
        else:
            dp = dp_local
        return dp, time.time() - t0, info


class SPRINGIterGPU(IterSRGPU):
    """SPRING preconditioner using iterative MINRES on the full Np
    operator (N_p-form).

    Holds a persistent ``phi_prev`` iterate between calls that
    implements the Kaczmarz-style recurrence from
    Goldshlager et al. 2024 (arXiv:2401.10190).  With ``mu=0`` this is
    bit-equivalent to ``IterSRGPU``.
    """

    _STATE_ATTRS = ('phi_prev',)

    def __init__(self, mu: float = 0.99, rtol: float = 5e-5, maxiter: int = 100):
        super().__init__(rtol=rtol, maxiter=maxiter, use_scipy=False)
        self.mu = mu
        self.phi_prev: Optional[torch.Tensor] = None

    def _rhs(
        self,
        energy_grad: torch.Tensor,
        *,
        shift: float,
        device: torch.device,
    ) -> torch.Tensor:
        return energy_grad + (self.mu * shift) * self.phi_prev

    def _solve(
        self,
        *,
        O_loc,
        E_loc,
        E_mean: float,
        Ns: int,
        Np: int,
        rshift: float,
        ashift: float,
        device: torch.device,
    ) -> Tuple[Any, float, Any]:
        device = self._device(device)
        if self.phi_prev is None:
            self.phi_prev = torch.zeros(
                Np, device=device, dtype=torch.float64,
            )
        # Call the parent IterSRGPU._solve directly (NOT super().solve):
        # the outer PreconditionerGPU.solve already ran outlier masking
        # before dispatching here, and super().solve would re-dispatch
        # self._solve -> this method -> infinite recursion. IterSR's
        # _solve builds its RHS via self._rhs, which is overridden above
        # to inject the SPRING momentum term.
        dp, t_sr, info = super()._solve(
            O_loc=O_loc,
            E_loc=E_loc,
            E_mean=E_mean,
            Ns=Ns,
            Np=Np,
            rshift=rshift,
            ashift=ashift,
            device=device,
        )
        self.phi_prev = torch.as_tensor(
            dp, device=device, dtype=torch.float64,
        ).clone()
        return dp, t_sr, info

    def reset(self) -> None:
        self.phi_prev = None


class MARCHIterGPU(IterSRGPU):
    r"""MARCH preconditioner using iterative MINRES on the full Np
    operator (N_p-form).

    Bit-equivalent to ``MARCHMinSRGPU`` in exact arithmetic, but
    never forms the (Ns, Ns) Gram matrix and never performs an
    ``all_to_all`` redistribution of ``O``.

    Derivation (see ``refs/theory/MARCH_derivation.ipynb``).  The
    Ns-form MARCH update is

    .. code-block:: text

        alpha = ((Obar/V)(Obar/V).T + lambda I)^{-1} (Ebar - mu Obar phi_prev)
        step  = (Obar/V).T alpha / V + mu phi_prev

    With ``c = step - mu phi_prev = (Obar.T alpha) / V^2``, the
    push-through identity ``A.T (A A.T + lambda I)^{-1} =
    (A.T A + lambda I)^{-1} A.T`` plus the substitution ``z = V c``
    converts the Ns-form solve into the Np-form

    .. code-block:: text

        (S + lambda V^2) step = grad + mu * lambda * V^2 * phi_prev

    where ``S = Obar.T Obar`` (matvec-only, no Ns x Ns matrix) and
    ``grad = Obar.T Ebar`` is the energy gradient.  Structurally
    identical to ``SPRINGIterGPU`` with the scalar Tikhonov shift
    ``lambda`` replaced by the diagonal ``lambda V^2``.

    With ``t = 0`` (so ``V = ones``), one step reduces exactly to
    ``SPRINGIterGPU(mu=mu)``.

    Args:
        mu: first-moment momentum (matches MARCHMinSRGPU.mu).
        beta: second-moment EMA rate (matches MARCHMinSRGPU.beta).
        rtol, maxiter: forwarded to ``torch_minres``.
    """

    _STATE_ATTRS = ('t', 'phi_prev', 'v_prev')

    def __init__(
        self,
        mu: float = 0.95,
        beta: float = 0.995,
        rtol: float = 5e-5,
        maxiter: int = 100,
    ):
        super().__init__(rtol=rtol, maxiter=maxiter, use_scipy=False)
        self.mu = mu
        self.beta = beta
        self.phi_prev: Optional[torch.Tensor] = None
        self.v_prev: Optional[torch.Tensor] = None
        self.t: int = 0

    def _solve(
        self,
        *,
        O_loc,
        E_loc,
        E_mean: float,
        Ns: int,
        Np: int,
        rshift: float,
        ashift: float,
        device: torch.device,
    ) -> Tuple[Any, float, Any]:
        t0 = time.time()
        device = self._device(device)
        if self.phi_prev is None:
            self.phi_prev = torch.zeros(
                Np, device=device, dtype=torch.float64,
            )
            self.v_prev = torch.zeros(
                Np, device=device, dtype=torch.float64,
            )

        O_loc = self._to_f64_tensor(O_loc, device)
        E_loc = self._to_f64_tensor(E_loc, device)
        energy_grad, O_mean, n_local = self._energy_gradient(
            O_loc, E_loc, E_mean, Ns, Np, device,
        )

        # Two-term scalar shift from the implicit Gram trace.
        shift = _two_term_shift(
            self._np_gram_trace(O_loc, O_mean, Ns),
            Np, rshift, ashift,
        )

        # V: first iter -> ones; subsequent -> v_prev^0.25 + eps.
        if self.t == 0:
            V = torch.ones(Np, device=device, dtype=torch.float64)
        else:
            V = self.v_prev.pow(0.25) + 1e-8
        shift_vec = shift * V * V  # shape (Np,)

        # Absorbed rhs: (S + lambda V^2) step = grad + mu lambda V^2 phi_prev
        rhs = energy_grad + self.mu * shift_vec * self.phi_prev

        # _matvec adds shift * x elementwise, so a (Np,) shift_vec gives
        # the diagonal Tikhonov shift lambda*V^2 (no separate matvec).
        dp, info = torch_minres(
            lambda x: self._matvec(
                x, O_loc, O_mean, Ns, n_local, shift_vec,
            ),
            rhs,
            rtol=self.rtol,
            maxiter=self.maxiter,
        )

        # v_new = beta * v_old + |step - phi_old|^2; phi_prev <- step.
        step_t = torch.as_tensor(
            dp, device=device, dtype=torch.float64,
        )
        self.v_prev = (
            self.beta * self.v_prev
            + (step_t - self.phi_prev).abs() ** 2
        )
        self.phi_prev = step_t.clone()
        self.t += 1
        return dp, time.time() - t0, info

    def reset(self) -> None:
        self.phi_prev = None
        self.v_prev = None
        self.t = 0


class AdamSRIterGPU(IterSRGPU):
    r"""AdamSR preconditioner using iterative MINRES on the full Np
    operator (N_p-form).

    Bit-equivalent to ``AdamSRMinSRGPU`` in exact arithmetic, but
    never forms the (Ns, Ns) Gram matrix and never performs an
    ``all_to_all`` redistribution of ``O``.

    Per step it performs two MINRES solves:

    1. **Raw SR direction** (plain MinSR, scalar Tikhonov):

       .. code-block:: text

           (S + lambda I) g = grad

    2. **Column-preconditioned step** (after Adam moments):

       .. code-block:: text

           t      += 1
           m       = mu   * m + (1 - mu)   * g
           v       = beta * v + (1 - beta) * |g|^2
           mhat    = m / (1 - mu**t)
           vhat    = v / (1 - beta**t)
           V       = vhat^{1/4} + eps
           (S + lambda V^2) step = grad + lambda V^2 * mhat

    Note: unlike MARCH, the moment ``v`` is updated **with the
    current step's** ``g`` (after the 1st solve) before computing
    ``V`` — that is why two solves are needed.  The bias-correction
    factors ``1/(1 - mu**t)`` and ``1/(1 - beta**t)`` follow the
    Adam convention.

    Derivation of the 2nd solve is identical to MARCH (see
    ``refs/theory/MARCH_derivation.ipynb``) with ``mhat`` in place
    of ``mu * phi_prev``.

    Args:
        mu: first-moment EMA rate.
        beta: second-moment EMA rate.
        norm_clip: if not None, clip ``||g||_2`` after the 1st solve.
        rtol, maxiter: forwarded to ``torch_minres`` (shared between
            both solves).
    """

    _STATE_ATTRS = ('t', 'm', 'v')

    def __init__(
        self,
        mu: float = 0.95,
        beta: float = 0.995,
        norm_clip: Optional[float] = None,
        rtol: float = 5e-5,
        maxiter: int = 100,
    ):
        super().__init__(rtol=rtol, maxiter=maxiter, use_scipy=False)
        self.mu = mu
        self.beta = beta
        self.norm_clip = norm_clip
        self.m: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.t: int = 0

    def _solve(
        self,
        *,
        O_loc,
        E_loc,
        E_mean: float,
        Ns: int,
        Np: int,
        rshift: float,
        ashift: float,
        device: torch.device,
    ) -> Tuple[Any, float, Any]:
        t0 = time.time()
        device = self._device(device)
        if self.m is None:
            self.m = torch.zeros(
                Np, device=device, dtype=torch.float64,
            )
            self.v = torch.zeros(
                Np, device=device, dtype=torch.float64,
            )

        O_loc = self._to_f64_tensor(O_loc, device)
        E_loc = self._to_f64_tensor(E_loc, device)
        energy_grad, O_mean, n_local = self._energy_gradient(
            O_loc, E_loc, E_mean, Ns, Np, device,
        )

        # Two-term scalar shift from the implicit Gram trace.
        shift = _two_term_shift(
            self._np_gram_trace(O_loc, O_mean, Ns),
            Np, rshift, ashift,
        )

        # ---- 1st solve: g = (S + lambda I)^{-1} energy_grad ----
        g, _ = torch_minres(
            lambda x: self._matvec(
                x, O_loc, O_mean, Ns, n_local, shift,
            ),
            energy_grad,
            rtol=self.rtol,
            maxiter=self.maxiter,
        )

        if self.norm_clip is not None:
            g_norm = torch.linalg.vector_norm(g)
            # branchless renorm: scale = min(clip/||g||, 1)
            scale = (
                self.norm_clip / g_norm.clamp(min=1e-30)
            ).clamp(max=1.0)
            g = g * scale

        # ---- Adam moment updates (current g enters now) ----
        self.t += 1
        self.m = self.mu * self.m + (1.0 - self.mu) * g
        self.v = self.beta * self.v + (1.0 - self.beta) * g.abs() ** 2
        mhat = self.m / (1.0 - self.mu ** self.t)
        vhat = self.v / (1.0 - self.beta ** self.t)
        V = vhat.pow(0.25)
        shift_vec = shift * V * V  # (Np,)

        # ---- 2nd solve: (S + lambda V^2) step = grad + lambda V^2 mhat ----
        rhs2 = energy_grad + shift_vec * mhat
        step, info = torch_minres(
            lambda x: self._matvec(
                x, O_loc, O_mean, Ns, n_local, shift_vec,
            ),
            rhs2,
            rtol=self.rtol,
            maxiter=self.maxiter,
        )
        # If the 2nd solve did not converge, `step` may not be a descent
        # direction. Fall back to the raw SR direction `g` (already
        # computed and norm-clipped above), which is a valid descent
        # direction, instead of risking a bad parameter update.
        if info != 0:
            if self._rank() == 0:
                print(
                    f"Warning: AdamSRIterGPU MINRES did not converge "
                    f"(info={info}). Falling back to the raw SR direction.",
                    flush=True,
                )
            return g, time.time() - t0, info

        if self.norm_clip is not None:
            g_norm = torch.linalg.vector_norm(step)
            # branchless renorm: scale = min(clip/||step||, 1)
            scale = (
                self.norm_clip / g_norm.clamp(min=1e-30)
            ).clamp(max=1.0)
            step = step * scale

        return step, time.time() - t0, info

    def reset(self) -> None:
        self.m = None
        self.v = None
        self.t = 0


class SPRINGMinSRGPU(PreconditionerGPU):
    """SPRING preconditioner using the Tikhonov minSR solve (N_s-form).

    Holds a persistent ``phi_prev`` iterate between calls.  With
    ``mu=0`` this is bit-equivalent to ``MinSRGPU``.
    """

    _STATE_ATTRS = ('phi_prev',)

    def __init__(
        self,
        mu: float = 0.99,
        mixed_precision: bool = False,
    ):
        self.mu = mu
        # If True, the Ns x Ns Gram matmul is done in fp32 (then
        # cast back to fp64). Big speedup on cards with weak fp64
        # throughput (consumer / RTX class); diag_shift dominates
        # the fp32 roundoff. Default False keeps the full Gram in
        # fp64 (recommended on A100/H100 where fp64 is fast).
        self.mixed_precision = mixed_precision
        self.phi_prev: Optional[torch.Tensor] = None

    def _solve(
        self,
        *,
        O_loc,
        E_loc,
        E_mean: float,
        Ns: int,
        Np: int,
        rshift: float,
        ashift: float,
        device: torch.device,
    ) -> Tuple[Any, float, Any]:
        t0 = time.time()
        device = self._device(device)
        rank = self._rank()
        world_size = self._world_size()
        # Mixed precision: fp64 everywhere except the Gram matmul
        # ``T = O_bar @ O_bar.T`` which is cast to fp32 locally.
        # Tikhonov diag_shift dominates the fp32 roundoff.
        if self.phi_prev is None:
            self.phi_prev = torch.zeros(
                Np, device=device, dtype=torch.float64,
            )

        # O_loc (Ns per rank, Np), E_loc (Ns per rank, )
        O_loc = self._to_f64_tensor(O_loc, device, contiguous=True)
        E_loc = self._to_f64_tensor(E_loc, device)

        # minSR builds its force from Obar.T @ alpha, so only the column
        # means of O are needed -- use _O_mean (1 all_reduce) instead of
        # _energy_gradient (which also does the discarded E@O matmul +
        # a 2nd all_reduce). O_mean shape (Np,), n_local = Ns per rank.
        n_local = E_loc.shape[0]
        O_mean = self._O_mean(O_loc, Ns)

        # Get Obar and Ebar by centering and scaling O_loc, E_loc in place.
        sqrt_Ns = math.sqrt(Ns)
        O_loc.sub_(O_mean.unsqueeze(0)).div_(sqrt_Ns)
        E_scaled = (E_loc - E_mean) / sqrt_Ns
        E_all = torch.empty(Ns, device=device, dtype=torch.float64)
        dist.all_gather_into_tensor(E_all, E_scaled)

        if world_size > 1:
            # Column-redistribute O across ranks via all_to_all so each
            # rank holds (Ns, Np/world_size). Peak per-rank memory during
            # this step is ~2 x |O_loc_local|.
            np_per_rank = (Np + world_size - 1) // world_size
            np_pad = world_size * np_per_rank
            col_offset = rank * np_per_rank
            if np_pad != Np:
                O_padded = torch.zeros(
                    (n_local, np_pad),
                    device=device,
                    dtype=torch.float64,
                )
                O_padded[:, :Np].copy_(O_loc)
                del O_loc
            else:
                O_padded = O_loc
                O_loc = None

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
            O_bar = recv_buf.reshape(
                world_size * n_local, np_per_rank,
            )  # shape (Ns, np_per_rank)
            del recv_buf
        else:
            # Single GPU: skip the all_to_all entirely (it is identity
            # but allocates 2 x |O_loc| of transient buffers).
            np_per_rank = Np
            np_pad = Np
            col_offset = 0
            O_bar = O_loc
            O_loc = None

        if self.mixed_precision:
            # Gram matmul in fp32, stored back in fp64.
            O_bar_f32 = O_bar.to(torch.float32)
            T = (O_bar_f32 @ O_bar_f32.T).to(torch.float64)
            del O_bar_f32
        else:
            T = O_bar @ O_bar.T
        dist.all_reduce(T, op=dist.ReduceOp.SUM)  # T shape (Ns, Ns)

        phi_cols = torch.zeros(
            np_per_rank, device=device, dtype=torch.float64,
        )  # phi_prev slice for this rank, shape (np_per_rank,)
        stop = min(col_offset + np_per_rank, Np)
        if stop > col_offset:
            n_live = stop - col_offset
            phi_cols[:n_live].copy_(
                self.phi_prev[col_offset:stop].to(
                    device=device, dtype=torch.float64,
                )
            )

        projection = O_bar @ phi_cols  # local contribution to O_k @ phi_prev, shape (Ns,)
        dist.all_reduce(projection, op=dist.ReduceOp.SUM)
        rhs = E_all - self.mu * projection  # shape (Ns,)

        T.diagonal().add_(_two_term_shift(
            T.trace().item(), T.shape[0], rshift, ashift,
        ))

        # Direct Cholesky solve of the (already-shifted) SPD Gram,
        # matching MinSRGPU. alpha shape (Ns,), alpha = T^{-1} rhs.
        alpha, info = self._solve_cholesky(T, rhs)
        del T

        dp_local = O_bar.T @ alpha + self.mu * phi_cols  # shape (np_per_rank,)
        del O_bar
        dp_padded = torch.empty(np_pad, device=device, dtype=torch.float64)
        dist.all_gather_into_tensor(dp_padded, dp_local) # concatenate dp_local from all ranks into dp_padded
        dp = dp_padded[:Np].contiguous()

        self.phi_prev = dp.detach().clone()
        return dp, time.time() - t0, info

    def reset(self) -> None:
        self.phi_prev = None


class MARCHMinSRGPU(PreconditionerGPU):
    r"""MARCH optimizer (arXiv:2507.02644) in the N_s-form MinSR.

    Adds first/second-order momentum on top of MinSR (Adam-like
    column preconditioning).  With default ``mu=0.95, beta=0.995``,
    the learning rate should be ~1/5 of plain MinSR.

    Per-iter algorithm (matching quantax MARCH):

    .. code-block:: text

        Ebar -= mu * (Obar @ phi_prev)
        V    = ones (first iter) else v_prev**0.25 + eps
        T    = (Obar/V) @ (Obar/V).T;  T += shift * I
        alpha = T^{-1} Ebar
        step = (Obar/V).T @ alpha / V + mu * phi_prev
        v_new = beta * v_prev + |step - phi_prev|**2
        phi_new = step

    Notes:
        - ``v`` is a pure beta-weighted accumulation (no ``(1-beta)``
          factor), matching the reference implementation.
        - No bias correction is applied (MARCH-specific).
    """

    _STATE_ATTRS = ('t', 'phi_prev', 'v_prev')

    def __init__(
        self,
        mu: float = 0.95,
        beta: float = 0.995,
        mixed_precision: bool = False,
    ):
        self.mu = mu
        self.beta = beta
        # fp32 Gram matmul; see SPRINGMinSRGPU for details.
        self.mixed_precision = mixed_precision
        self.phi_prev: Optional[torch.Tensor] = None
        self.v_prev: Optional[torch.Tensor] = None
        self.t: int = 0

    def _solve(
        self,
        *,
        O_loc,
        E_loc,
        E_mean: float,
        Ns: int,
        Np: int,
        rshift: float,
        ashift: float,
        device: torch.device,
    ) -> Tuple[Any, float, Any]:
        t0 = time.time()
        device = self._device(device)
        rank = self._rank()
        world_size = self._world_size()
        # Mixed precision: fp64 everywhere except the Gram matmul
        # ``T = O_pre @ O_pre.T`` which is cast to fp32 locally.
        if self.phi_prev is None:
            self.phi_prev = torch.zeros(
                Np, device=device, dtype=torch.float64,
            )
            self.v_prev = torch.zeros(
                Np, device=device, dtype=torch.float64,
            )

        O_loc = self._to_f64_tensor(O_loc, device, contiguous=True)
        E_loc = self._to_f64_tensor(E_loc, device)

        # Only O's column means are needed (force = Obar.T @ alpha); use
        # _O_mean (1 all_reduce) rather than _energy_gradient (extra E@O
        # matmul + 2nd all_reduce whose gradient output is discarded).
        n_local = E_loc.shape[0]
        O_mean = self._O_mean(O_loc, Ns)

        sqrt_Ns = math.sqrt(Ns)
        O_loc.sub_(O_mean.unsqueeze(0)).div_(sqrt_Ns)
        E_scaled = (E_loc - E_mean) / sqrt_Ns
        E_all = torch.empty(Ns, device=device, dtype=torch.float64)
        dist.all_gather_into_tensor(E_all, E_scaled)

        if world_size > 1:
            # Column-redistribute O across ranks via all_to_all.
            np_per_rank = (Np + world_size - 1) // world_size
            np_pad = world_size * np_per_rank
            col_offset = rank * np_per_rank
            if np_pad != Np:
                O_padded = torch.zeros(
                    (n_local, np_pad),
                    device=device,
                    dtype=torch.float64,
                )
                O_padded[:, :Np].copy_(O_loc)
                del O_loc
            else:
                O_padded = O_loc
                O_loc = None

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
            O_bar = recv_buf.reshape(
                world_size * n_local, np_per_rank,
            )
            del recv_buf
        else:
            # Single GPU: skip the identity all_to_all to avoid the
            # 2 x |O_loc| transient memory spike.
            np_per_rank = Np
            np_pad = Np
            col_offset = 0
            O_bar = O_loc
            O_loc = None

        # Slice phi_prev and v_prev for this rank's column block.
        phi_cols = torch.zeros(
            np_per_rank, device=device, dtype=torch.float64,
        )
        v_cols = torch.zeros(
            np_per_rank, device=device, dtype=torch.float64,
        )
        stop = min(col_offset + np_per_rank, Np)
        if stop > col_offset:
            n_live = stop - col_offset
            phi_cols[:n_live].copy_(
                self.phi_prev[col_offset:stop].to(
                    device=device, dtype=torch.float64,
                )
            )
            v_cols[:n_live].copy_(
                self.v_prev[col_offset:stop].to(
                    device=device, dtype=torch.float64,
                )
            )

        # V: first iter -> ones; subsequent -> v^0.25 + eps.
        if self.t == 0:
            V_cols = torch.ones_like(v_cols)
        else:
            V_cols = v_cols.pow(0.25) + 1e-8

        # rhs = E_all - mu * (Obar @ phi_prev).
        proj = O_bar @ phi_cols  # local contribution, shape (Ns,)
        dist.all_reduce(proj, op=dist.ReduceOp.SUM)
        rhs = E_all - self.mu * proj

        # Column-precondition Obar.
        # In-place: O_bar is not needed in raw form after this point,
        # so divide in place instead of allocating another full
        # (Ns, np_per_rank) copy of the Jacobian.
        O_bar.div_(V_cols.unsqueeze(0))
        O_pre = O_bar
        if self.mixed_precision:
            O_pre_f32 = O_pre.to(torch.float32)
            T = (O_pre_f32 @ O_pre_f32.T).to(torch.float64)
            del O_pre_f32
        else:
            T = O_pre @ O_pre.T
        dist.all_reduce(T, op=dist.ReduceOp.SUM)
        T.diagonal().add_(_two_term_shift(
            T.trace().item(), T.shape[0], rshift, ashift,
        ))

        # Direct Cholesky solve of the (already-shifted) SPD Gram,
        # matching MinSRGPU.
        alpha, info = self._solve_cholesky(T, rhs)
        del T

        # step in original parameter space:
        #   (Obar/V).T @ alpha / V + mu * phi_prev.
        step_pre_local = O_pre.T @ alpha  # shape (np_per_rank,)
        del O_pre, O_bar
        step_local = step_pre_local / V_cols + self.mu * phi_cols

        step_padded = torch.empty(
            np_pad, device=device, dtype=torch.float64,
        )
        dist.all_gather_into_tensor(step_padded, step_local)
        dp = step_padded[:Np].contiguous()

        # Buffer update: v_new = beta * v_old + |step - phi_old|^2.
        new_v = self.beta * self.v_prev + (dp - self.phi_prev).abs() ** 2
        self.phi_prev = dp.detach().clone()
        self.v_prev = new_v
        self.t += 1
        return dp, time.time() - t0, info

    def reset(self) -> None:
        self.phi_prev = None
        self.v_prev = None
        self.t = 0


class AdamSRMinSRGPU(PreconditionerGPU):
    r"""AdamSR optimizer in the N_s-form MinSR.

    Adam-style first/second-moment momentum on top of MinSR.  The
    cost per step is roughly twice that of plain MinSR (two SR
    solves, both Ns x Ns Cholesky/eigh on the column-distributed
    Gram).

    Per-iter algorithm (matching quantax AdamSR):

    .. code-block:: text

        # 1st solve: raw SR direction
        T1 = Obar @ Obar.T
        g  = Obar.T @ solve_gram(T1, Ebar)
        (optional clip ||g|| <= norm_clip)

        # Adam moment updates
        t += 1
        m  = mu * m  + (1-mu)   * g
        v  = beta * v + (1-beta) * |g|^2
        mhat = m / (1 - mu**t);  vhat = v / (1 - beta**t)
        V    = vhat**0.25 + eps

        # 2nd solve: with mhat correction & V column preconditioning
        rhs  = Ebar - Obar @ mhat
        T2   = (Obar/V) @ (Obar/V).T
        step = (Obar/V).T @ solve_gram(T2, rhs) / V + mhat

    where ``solve_gram`` is selected by ``solver`` (``rshift`` and
    ``ashift`` are handed in per solve by the VMC loop):

      - ``'direct'``: direct solve of ``(T + shift*I) x = b`` with
        the two-term shift
        ``shift = rshift * trace(T)/sqrt(Ns) + ashift``.
      - ``'pinv_eig'``: eigendecomposition pseudo-inverse with a
        smooth relative eigenvalue cutoff (``rtol = rshift``):
        ``1/lam -> 1/(lam * (1 + (rtol*lam_max/|lam|)**6))``.
        ``ashift`` is IGNORED in this mode.

    Args:
        mu, beta: first/second moment decay rates.
        norm_clip: if not None, clip the raw direction g to this
            Euclidean norm before the moment updates.
        solver: ``'direct'`` (default) or ``'pinv_eig'`` — how the
            two Ns x Ns Gram systems are solved (see above).
    """

    _STATE_ATTRS = ('t', 'm', 'v')
    # run_vmc_loop hands the Jacobian over via an ownership box so it
    # can be freed inside _solve (right after the all_to_all copy)
    # rather than lingering in the caller's frame for the whole solve.
    _supports_ownership_box = True

    def __init__(
        self,
        mu: float = 0.95,
        beta: float = 0.995,
        norm_clip: Optional[float] = None,
        mixed_precision: bool = False,
        solver: str = 'direct',
    ):
        if solver not in ('direct', 'pinv_eig'):
            raise ValueError(
                f"unknown solver {solver!r}; expected "
                "'direct' or 'pinv_eig'"
            )
        self.mu = mu
        self.beta = beta
        self.norm_clip = norm_clip
        # The all_to_all reshard is always done in fp32 and O_bar is
        # upcast to fp64 afterwards.  mixed_precision only controls the
        # two Gram matmuls (T1, T2): default False keeps them in fp64
        # (matches Ao); True casts O_bar to fp32 just for the matmul.
        self.mixed_precision = mixed_precision
        self.solver = solver
        self.m: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.t: int = 0

    def _solve_gram(
        self,
        T: torch.Tensor,
        b: torch.Tensor,
        rshift: float,
        ashift: float,
    ) -> torch.Tensor:
        """Solve the Ns x Ns Gram system ``T x = b``.

        Dispatches on ``self.solver``:

          - ``'direct'``: in-place diagonal shift
            ``rshift * trace(T)/sqrt(Ns) + ashift``
            followed by a direct solve (``T`` is mutated).
          - ``'pinv_eig'``: eigh pseudo-inverse with a smooth
            relative eigenvalue cutoff
            (``rtol = rshift``, ``atol = 0``); ``ashift`` is
            ignored.

        Args:
            T: (Ns, Ns) symmetric PSD Gram matrix (all-reduced,
                identical on every rank).
            b: (Ns,) right-hand side.
            rshift, ashift: relative / absolute shift terms.

        Returns:
            (Ns,) solution vector.
        """
        if self.solver == 'pinv_eig':
            # Smooth suppression of eigenvalues below
            # rtol * lam_max instead of a hard truncation.
            evals, U = torch.linalg.eigh(T)
            evals_abs = evals.abs()
            cutoff = rshift * evals_abs.max()
            inv_factor = 1.0 + (cutoff / evals_abs) ** 6
            evals_inv = 1.0 / (evals * inv_factor)
            evals_inv = torch.where(
                evals_abs > 0.0,
                evals_inv,
                torch.zeros_like(evals_inv),
            )
            return U @ (evals_inv * (U.T @ b))
        # 'direct': trace-scaled relative shift + absolute shift, then a
        # Cholesky solve (T is SPD after the shift; ~2x fewer flops than
        # LU and matches Ao's `solve(..., assume_a="pos")`).
        T.diagonal().add_(_two_term_shift(
            T.trace().item(), T.shape[0], rshift, ashift,
        ))
        return self._solve_cholesky(T, b)[0]

    def _solve(
        self,
        *,
        O_loc,
        E_loc,
        E_mean: float,
        Ns: int,
        Np: int,
        rshift: float,
        ashift: float,
        device: torch.device,
    ) -> Tuple[Any, float, Any]:
        t0 = time.time()
        device = self._device(device)
        rank = self._rank()
        world_size = self._world_size()

        # Unwrap the ownership box from solve() so this frame holds the
        # only reference (lets the fp32 Jacobian be freed in here).
        if isinstance(O_loc, list):
            box = O_loc
            O_loc = box[0]
            box[0] = None
        jac_dtype = O_loc.dtype
        # Keep the Jacobian in its native (fp32) dtype for the cheap
        # all_to_all; it must be contiguous for the later .view().
        # Energies stay in their native (fp64) dtype so the centering
        # (E_loc - E_mean) keeps full precision, like Ao.
        O_loc = O_loc.to(device).contiguous()
        E_loc = E_loc.to(device)

        # (Non-finite sample rows are already zeroed and E_mean made
        # clean by solve() -> _mask_outlier_samples, before _solve.)

        # minSR only needs the column means of O (the kernel-form force
        # is built from Obar.T @ alpha, not the explicit E@O gradient),
        # so use _O_mean and skip the discarded (Ns x Np) E@O matmul.
        n_local = E_loc.shape[0]
        O_mean = self._O_mean(O_loc, Ns)

        O_loc.sub_(O_mean.unsqueeze(0)).div_(math.sqrt(Ns))
        E_scaled = (E_loc - E_mean) / math.sqrt(Ns)

        E_all = torch.empty(Ns, device=device, dtype=E_scaled.dtype)
        dist.all_gather_into_tensor(E_all, E_scaled)

        if world_size > 1:
            # Column-redistribute O across ranks via all_to_all (fp32).
            np_per_rank = (Np + world_size - 1) // world_size
            np_pad = world_size * np_per_rank
            col_offset = rank * np_per_rank
            if np_pad != Np:
                O_padded = torch.zeros(
                    (n_local, np_pad),
                    device=device,
                    dtype=jac_dtype,
                )
                O_padded[:, :Np].copy_(O_loc)
                del O_loc
            else:
                O_padded = O_loc
                O_loc = None

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
            O_bar = recv_buf.reshape(
                world_size * n_local, np_per_rank,
            )
            del recv_buf
        else:
            # Single GPU: skip the identity all_to_all to avoid the
            # 2 x |O_loc| transient memory spike.
            np_per_rank = Np
            np_pad = Np
            col_offset = 0
            O_bar = O_loc
            O_loc = None

        # ---- O reshard done in fp32; upcast for the fp64 linear
        # algebra (Gram matmul / Cholesky solve / back-projection).
        # E_all is already fp64 (energies kept full precision). ----
        O_bar = O_bar.to(torch.float64)

        stop = min(col_offset + np_per_rank, Np)

        # ---- 1st solve: raw SR direction g = Obar.T @ T1^{-1} Ebar ----
        T1 = O_bar @ O_bar.T

        dist.all_reduce(T1, op=dist.ReduceOp.SUM)
        alpha1 = self._solve_gram(T1, E_all, rshift, ashift)
        del T1
        g_local = O_bar.T @ alpha1  # shape (np_per_rank,)

        g_padded = torch.empty(
            np_pad, device=device, dtype=torch.float64,
        )
        dist.all_gather_into_tensor(g_padded, g_local)
        g = g_padded[:Np].contiguous()

        # clip the solved g
        if self.norm_clip is not None:
            g_norm = torch.linalg.vector_norm(g)
            # branchless renorm: scale = min(clip/||g||, 1)
            scale = (
                self.norm_clip / g_norm.clamp(min=1e-30)
            ).clamp(max=1.0)
            if scale != 1.0 and self._rank() == 0:
                print(
                    f"Clipping AdamSRMinSRGPU raw direction from {g_norm:.4e} "
                    f"to {self.norm_clip:.4e} (scale factor {scale:.4e})",
                    flush=True,
                )
            g = g * scale

        # ---- Adam moment updates (full Np vectors) ----
        if self.m is None:
            self.m = torch.zeros(
                Np, device=device, dtype=torch.float64,
            )
            self.v = torch.zeros(
                Np, device=device, dtype=torch.float64,
            )
        self.t += 1
        self.m = self.mu * self.m + (1.0 - self.mu) * g
        self.v = self.beta * self.v + (1.0 - self.beta) * g.abs() ** 2
        mhat = self.m / (1.0 - self.mu ** self.t)
        vhat = self.v / (1.0 - self.beta ** self.t)
        V = vhat.pow(0.25) + 1e-8

        # Slice mhat, V for this rank's columns.
        mhat_cols = torch.zeros(
            np_per_rank, device=device, dtype=torch.float64,
        )
        V_cols = torch.ones(
            np_per_rank, device=device, dtype=torch.float64,
        )
        if stop > col_offset:
            n_live = stop - col_offset
            mhat_cols[:n_live].copy_(mhat[col_offset:stop])
            V_cols[:n_live].copy_(V[col_offset:stop])

        # ---- 2nd solve: rhs = Ebar - Obar @ mhat, Obar_pre = Obar / V ----
        proj = O_bar @ mhat_cols  # local contribution, shape (Ns,)
        dist.all_reduce(proj, op=dist.ReduceOp.SUM)
        rhs2 = E_all - proj

        # In-place: O_bar is not needed in raw form after this point,
        # so divide in place instead of allocating another full
        # (Ns, np_per_rank) copy of the Jacobian.
        O_bar.div_(V_cols.unsqueeze(0))
        O_pre = O_bar
        T2 = O_pre @ O_pre.T
        dist.all_reduce(T2, op=dist.ReduceOp.SUM)
        alpha2 = self._solve_gram(T2, rhs2, rshift, ashift)
        info2 = 0  # direct solve, so no convergence info
        del T2

        step_pre_local = O_pre.T @ alpha2  # (np_per_rank,) not complete yet
        del O_pre, O_bar
        step_local = step_pre_local / V_cols + mhat_cols

        step_padded = torch.empty(
            np_pad, device=device, dtype=torch.float64,
        )
        dist.all_gather_into_tensor(step_padded, step_local)
        dp = step_padded[:Np].contiguous()
        
        # clip the solved g
        if self.norm_clip is not None:
            dp_norm = torch.linalg.vector_norm(dp)
            # branchless renorm: scale = min(clip/||dp||, 1)
            scale = (
                self.norm_clip / dp_norm.clamp(min=1e-30)
            ).clamp(max=1.0)
            if scale != 1.0 and self._rank() == 0:
                print(
                    f"Clipping AdamSRMinSRGPU step from {dp_norm:.4e} to "
                    f"{self.norm_clip:.4e} (scale factor {scale:.4e})",
                    flush=True,
                )
            dp = dp * scale

        # Cast the step back to the Jacobian (model param) dtype.
        dp = dp.to(jac_dtype)
        return dp, time.time() - t0, info2

    def reset(self) -> None:
        self.m = None
        self.v = None
        self.t = 0


__all__ = [
    "Scheduler",
    "TrivialScheduler",
    "DecayScheduler",
    "OptimizerGPU",
    "SGDGPU",
    "AdamGPU",
    "PreconditionerGPU",
    "IterSRGPU",
    "MinSRGPU",
    "SPRINGIterGPU",
    "MARCHIterGPU",
    "AdamSRIterGPU",
    "SPRINGMinSRGPU",
    "MARCHMinSRGPU",
    "AdamSRMinSRGPU",
]
