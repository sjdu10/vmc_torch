from typing import Any, Optional, Tuple

import numpy as np
import torch

from vmc_torch.experiment.vmap.GPU.vmc_modules import (
    distributed_minres_solver_gpu,
    minSR_solver_gpu,
)


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
    "OptimizerGPU",
    "SGDGPU",
    "AdamGPU",
    "PreconditionerGPU",
    "DistributedSRMinresGPU",
    "MinSRGPU",
]
