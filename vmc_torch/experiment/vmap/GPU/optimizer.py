import functools
import math
from typing import Any, Optional, Tuple

import numpy as np
import torch

from vmc_torch.experiment.vmap.GPU.vmc_modules import (
    distributed_minres_solver_gpu,
    minSR_solver_gpu,
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

    def __init__(self, learning_rate: float = 1e-3):
        self.lr = learning_rate

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
            current = torch.nn.utils.parameters_to_vector(model.parameters())
            target_device = current.device if device is None else device
            direction_t = torch.as_tensor(
                direction,
                device=target_device,
                dtype=current.dtype,
            )
            updated = self.compute_update(
                current,
                direction_t,
                learning_rate=learning_rate,
            )
            torch.nn.utils.vector_to_parameters(updated, model.parameters())

    def reset(self) -> None:
        pass


class SGDGPU(OptimizerGPU):
    def compute_update(
        self,
        params_vec: torch.Tensor,
        direction_vec: torch.Tensor,
        learning_rate: Optional[float] = None,
    ) -> torch.Tensor:
        lr = self.lr if learning_rate is None else learning_rate
        return params_vec - lr * direction_vec


class AdamGPU(OptimizerGPU):
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


class PreconditionerGPU:
    """Base SR/preconditioner interface for solving update directions."""

    def solve(
        self,
        *,
        local_o,
        local_energies,
        energy_mean: float,
        total_samples: int,
        n_params: int,
        diag_shift: float,
        device: torch.device,
        run_sr: bool,
    ) -> Tuple[np.ndarray, float, Any]:
        raise NotImplementedError


class DistributedSRMinresGPU(PreconditionerGPU):
    def __init__(self, rtol: float = 5e-5, maxiter: int = 100):
        self.rtol = rtol
        self.maxiter = maxiter

    def solve(
        self,
        *,
        local_o,
        local_energies,
        energy_mean: float,
        total_samples: int,
        n_params: int,
        diag_shift: float,
        device: torch.device,
        run_sr: bool,
    ) -> Tuple[np.ndarray, float, Any]:
        _ = device
        return distributed_minres_solver_gpu(
            local_O=local_o,
            local_energies=local_energies,
            energy_mean=energy_mean,
            total_samples=total_samples,
            n_params=n_params,
            diag_shift=diag_shift,
            rtol=self.rtol,
            maxiter=self.maxiter,
            run_SR=run_sr,
        )


class MinSRGPU(PreconditionerGPU):
    def solve(
        self,
        *,
        local_o,
        local_energies,
        energy_mean: float,
        total_samples: int,
        n_params: int,
        diag_shift: float,
        device: torch.device,
        run_sr: bool,
    ) -> Tuple[np.ndarray, float, Any]:
        return minSR_solver_gpu(
            local_O=local_o,
            local_energies=local_energies,
            energy_mean=energy_mean,
            total_samples=total_samples,
            n_params=n_params,
            diag_shift=diag_shift,
            device=device,
            do_SR=run_sr,
        )


__all__ = [
    "Scheduler",
    "TrivialScheduler",
    "DecayScheduler",
    "OptimizerGPU",
    "SGDGPU",
    "AdamGPU",
    "PreconditionerGPU",
    "DistributedSRMinresGPU",
    "MinSRGPU",
]
