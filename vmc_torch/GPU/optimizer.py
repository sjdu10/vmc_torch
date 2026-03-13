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

# from vmc_torch.GPU.vmc_modules import (
#     distributed_minres_solver_gpu,
#     distributed_minSR_solver_gpu,
#     minSR_solver_gpu,
# )


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
    ) -> Tuple[Any, float, Any]:
        raise NotImplementedError


# ===========================================================================
# Distributed MINRES SR Solver (via torch.distributed.all_reduce)
# ===========================================================================
def distributed_minres_solver_gpu(
    local_lpg,
    local_energies,
    energy_mean,
    total_samples,
    n_params,
    diag_shift,
    rtol=1e-4,
    maxiter=100,
    run_SR=True,
    use_scipy=False,
    device=None,
):
    """
    Distributed SR solver using MINRES with torch.distributed.all_reduce.

    Each rank holds its local log_psi_grad chunk. The matrix-vector
    products for S = (1/N) O^T O - mean_lpg @ mean_lpg^T + diag_shift * I
    are computed distributedly via matvec (no Np x Np matrix built).

    By default uses pure-PyTorch MINRES (torch_minres) so everything
    stays on GPU.  Set use_scipy=True to fall back to the old CPU
    scipy.sparse.linalg.minres path (for debugging / validation).

    Args:
        local_lpg: (n_local, Np) GPU tensor or numpy array.
        local_energies: (n_local,) GPU tensor or numpy array.
        energy_mean: global mean energy (float).
        total_samples: total samples across all ranks.
        n_params: number of parameters.
        diag_shift: diagonal regularization.
        rtol: MINRES tolerance.
        maxiter: max MINRES iterations.
        run_SR: if False, return only the energy gradient.
        use_scipy: if True, use scipy MINRES on CPU (fallback).

    Returns:
        dp: (Np,) GPU tensor (or numpy if use_scipy), param update.
        sr_time: wall-clock time.
        info: MINRES convergence flag.
    """
    t0 = time.time()
    world_size = dist.get_world_size()
    if device is None:
        device = torch.device('cuda')

    # --- Check if lpg_loc is already on CPU ---
    lpgloc_on_cpu = (
        isinstance(local_lpg, torch.Tensor)
        and local_lpg.device.type == 'cpu'
    ) or isinstance(local_lpg, np.ndarray)

    if use_scipy:
        assert lpgloc_on_cpu, "use_scipy=True requires local_lpg on CPU"
        # --- Fast path: lpg_loc already on CPU ---
        # Compute stats on CPU, only briefly use GPU for
        # all_reduce of (Np,) vectors.
        if isinstance(local_lpg, torch.Tensor):
            local_lpg_np = local_lpg.numpy()
        else:
            local_lpg_np = np.asarray(
                local_lpg, dtype=np.float64,
            )

        if isinstance(local_energies, torch.Tensor):
            local_E_np = local_energies.cpu().numpy()
        else:
            local_E_np = np.asarray(
                local_energies, dtype=np.float64,
            )
        n_local = local_E_np.shape[0]

        if n_local > 0:
            local_sum_lpg_np = local_lpg_np.sum(axis=0)
            local_sum_EO_np = local_E_np @ local_lpg_np
        else:
            local_sum_lpg_np = np.zeros(
                n_params, dtype=np.float64,
            )
            local_sum_EO_np = np.zeros(
                n_params, dtype=np.float64,
            )

        # all_reduce requires GPU tensors (NCCL)
        if world_size > 1:
            sum_lpg_gpu = torch.tensor(
                local_sum_lpg_np, device=device,
            )
            sum_EO_gpu = torch.tensor(
                local_sum_EO_np, device=device,
            )
            dist.all_reduce(
                sum_lpg_gpu, op=dist.ReduceOp.SUM,
            )
            dist.all_reduce(
                sum_EO_gpu, op=dist.ReduceOp.SUM,
            )
            local_sum_lpg_np = sum_lpg_gpu.cpu().numpy()
            local_sum_EO_np = sum_EO_gpu.cpu().numpy()
            del sum_lpg_gpu, sum_EO_gpu

        mean_lpg_np = local_sum_lpg_np / total_samples
        mean_EO_np = local_sum_EO_np / total_samples
        energy_grad_np = (
            mean_EO_np - energy_mean * mean_lpg_np
        )

        if not run_SR:
            t1 = time.time()
            return energy_grad_np, t1 - t0, None

        # scipy MINRES on CPU (lpg_loc stays on CPU)
        if world_size == 1:
            def matvec(x):
                inner = local_lpg_np.dot(x)
                Sx = local_lpg_np.T.dot(inner)
                Sx /= total_samples
                Sx -= np.dot(mean_lpg_np, x) * mean_lpg_np
                return Sx + diag_shift * x
        else:
            def matvec(x):
                if n_local > 0:
                    inner = local_lpg_np.dot(x)
                    local_Sx = local_lpg_np.T.dot(inner)
                else:
                    local_Sx = np.zeros_like(x)
                Sx_t = torch.tensor(
                    local_Sx, device=device,
                )
                dist.all_reduce(
                    Sx_t, op=dist.ReduceOp.SUM,
                )
                Sx = Sx_t.cpu().numpy()
                Sx /= total_samples
                Sx -= np.dot(mean_lpg_np, x) * mean_lpg_np
                return Sx + diag_shift * x

        A = spla.LinearOperator(
            (n_params, n_params), matvec=matvec,
            dtype=np.float64,
        )
        dp, info = spla.minres(
            A, energy_grad_np,
            rtol=rtol, maxiter=maxiter,
        )
        t1 = time.time()
        return dp, t1 - t0, info

    # --- Standard path: ensure inputs are GPU tensors ---
    assert isinstance(local_lpg, torch.Tensor), "local_lpg must be a torch.Tensor"
    assert isinstance(local_energies, torch.Tensor), "local_energies must be a torch.Tensor"
    if lpgloc_on_cpu:
      t0 = time.time()
      local_lpg = local_lpg.to(device=device, dtype=torch.float64)
      if dist.get_rank() == 0:
          print(f'SR: reload lpg from cpu to gpu, time = {time.time() - t0:.2f}s')

    n_local = local_energies.shape[0]

    # --- Compute global statistics via all_reduce on GPU ---
    if n_local > 0:
        local_sum_lpg = local_lpg.sum(dim=0)              # (Np,)
        local_sum_EO = local_energies @ local_lpg       # (Np,)
    else:
        local_sum_lpg = torch.zeros(
            n_params, device=device, dtype=torch.float64,
        )
        local_sum_EO = torch.zeros(
            n_params, device=device, dtype=torch.float64,
        )

    if world_size > 1:
        dist.all_reduce(local_sum_lpg, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sum_EO, op=dist.ReduceOp.SUM)

    mean_lpg = local_sum_lpg / total_samples
    mean_EO = local_sum_EO / total_samples
    energy_grad = mean_EO - energy_mean * mean_lpg  # (Np,) GPU

    if not run_SR:
        t1 = time.time()
        return energy_grad, t1 - t0, None

    # --- Pure-GPU MINRES via torch_minres ---
    if world_size == 1:
        def gpu_matvec(x):
            inner = local_lpg @ x               # (n_local,)
            Sx = local_lpg.T @ inner             # (Np,)
            Sx /= total_samples
            Sx -= torch.dot(mean_lpg, x) * mean_lpg
            return Sx + diag_shift * x
    else:
        def gpu_matvec(x):
            if n_local > 0:
                inner = local_lpg @ x
                local_Sx = local_lpg.T @ inner
            else:
                local_Sx = torch.zeros_like(x)
            dist.all_reduce(local_Sx, op=dist.ReduceOp.SUM)
            local_Sx /= total_samples
            local_Sx -= torch.dot(mean_lpg, x) * mean_lpg
            return local_Sx + diag_shift * x

    dp, info = torch_minres(
        gpu_matvec, energy_grad, rtol=rtol, maxiter=maxiter,
    )

    t1 = time.time()
    return dp, t1 - t0, info


# ===========================================================================
# MinSR Direct Solver (gather to rank 0, GPU linear algebra)
# ===========================================================================
def minSR_solver_gpu(
    local_lpg,
    local_energies,
    energy_mean,
    total_samples,
    n_params,
    diag_shift=1e-4,
    device=None,
    do_SR=True,
):
    """
    MinSR direct solver: gather lpg_loc to rank 0, solve in sample space.

    Efficient when Total_Ns < Np (common for NN-based VMC).
    Uses GPU linear algebra for the (Ns x Ns) solve.

    Args:
        local_lpg: (n_local, Np) numpy array
        local_energies: (n_local,) numpy array
        energy_mean: global mean energy
        total_samples: total across all ranks
        n_params: number of parameters
        diag_shift: regularization
        device: torch device for GPU solve

    Returns:
        dp: (Np,) numpy array
        sr_time: float
        info: 0 (success) or error flag
    """
    if do_SR:
        t0 = time.time()
        if device is None:
            device = torch.device('cuda')

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Accept GPU tensors — skip re-upload if already on device
        if isinstance(local_lpg, torch.Tensor):
            local_lpg_t = local_lpg.to(
                device=device, dtype=torch.float64
            ).contiguous()
            local_E_t = local_energies.to(
                device=device, dtype=torch.float64
            ).contiguous()
        else:
            local_lpg_t = torch.tensor(
                local_lpg, device=device, dtype=torch.float64
            ).contiguous()
            local_E_t = torch.tensor(
                local_energies, device=device, dtype=torch.float64
            ).contiguous()

        if world_size > 1:
            # Gather lpg_loc and energies across ranks
            total_lpg_t = torch.zeros(
                (total_samples, n_params),
                device=device, dtype=torch.float64,
            )
            total_E_t = torch.zeros(
                total_samples, device=device, dtype=torch.float64,
            )
            dist.all_gather_into_tensor(total_lpg_t, local_lpg_t)
            dist.all_gather_into_tensor(total_E_t, local_E_t)
        else:
            total_lpg_t = local_lpg_t
            total_E_t = local_E_t

        info = 0
        dp_t = torch.zeros(
            n_params, device=device, dtype=torch.float64
        )

        if rank == 0:
            lpg_mean = torch.mean(total_lpg_t, dim=0)
            lpg_centered = total_lpg_t - lpg_mean.unsqueeze(0)
            lpg_scaled = lpg_centered / np.sqrt(total_samples)
            E_s = (total_E_t - energy_mean) / np.sqrt(total_samples)

            # Gram matrix T = lpg_scaled @ lpg_scaled^T  (Ns x Ns)
            T = lpg_scaled @ lpg_scaled.T
            T += diag_shift * torch.eye(
                total_samples, device=device, dtype=torch.float64,
            )

            try:
                x = torch.linalg.solve(T, E_s)
            except RuntimeError:
                x = torch.linalg.lstsq(T, E_s).solution
                info = 1

            dp_t = lpg_scaled.T @ x

        if world_size > 1:
            dist.broadcast(dp_t, src=0)

        dp = dp_t.cpu().numpy()
        t1 = time.time()
        return dp, t1 - t0, info
    else:
        # Compute energy gradient only (no SR solve)
        t0 = time.time()
        if device is None:
            device = torch.device('cuda')

        # Accept GPU tensors
        if isinstance(local_lpg, torch.Tensor):
            local_lpg = local_lpg.to(device=device, dtype=torch.float64)
            local_energies = local_energies.to(
                device=device, dtype=torch.float64
            )
        else:
            local_lpg = torch.tensor(
                local_lpg, device=device, dtype=torch.float64
            )
            local_energies = torch.tensor(
                local_energies, device=device, dtype=torch.float64
            )

        world_size = dist.get_world_size()
        n_local = local_energies.shape[0]

        if n_local > 0:
            local_sum_lpg = local_lpg.sum(dim=0)                      # (Np,)
            local_sum_EO = local_energies @ local_lpg               # (Np,)
        else:
            local_sum_lpg = torch.zeros(
                n_params, device=device, dtype=torch.float64
            )
            local_sum_EO = torch.zeros(
                n_params, device=device, dtype=torch.float64
            )

        if world_size > 1:
            dist.all_reduce(local_sum_lpg, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sum_EO, op=dist.ReduceOp.SUM)

        mean_lpg = local_sum_lpg / total_samples
        mean_EO = local_sum_EO / total_samples
        energy_grad = (mean_EO - energy_mean * mean_lpg).cpu().numpy()

        t1 = time.time()
        return energy_grad, t1 - t0, None


# ===========================================================================
# Distributed MinSR Solver (chunked Gram matrix, no full gather)
# ===========================================================================
def distributed_minSR_solver_gpu(
    local_lpg,
    local_energies,
    energy_mean,
    total_samples,
    n_params,
    diag_shift=1e-4,
    param_chunk_size=1024,
    device=None,
    do_SR=True,
):
    """Distributed minSR solver with incremental Gram matrix.

    Builds G = lpg_scaled @ lpg_scaled.T by iterating over parameter chunks
    of size C.  Supports both GPU-resident and CPU-resident
    local_lpg:

    - GPU path: slices local_lpg[:, c:c+C] directly on GPU.
    - CPU path (offload_cpu): local_lpg lives on CPU.  Each
      param chunk is uploaded to GPU on the fly, used for
      G-build and dp-reconstruction, then discarded.  Peak
      GPU memory is O(Ns_total^2 + Ns_total * C) — the
      (Ns_per_rank, Np) matrix never touches GPU.

    Args:
        local_lpg: (Ns_per_rank, Np) tensor — GPU or CPU.
        local_energies: (Ns_per_rank,) tensor — GPU or CPU.
        energy_mean: global mean energy (float).
        total_samples: total samples across all ranks (Ns_total).
        n_params: number of parameters (Np).
        diag_shift: regularization for (G + lambda*I).
        param_chunk_size: number of params to gather at once (C).
        device: torch device for GPU compute.
        do_SR: if False, return only the energy gradient.

    Returns:
        dp: (Np,) GPU tensor, parameter update direction.
        sr_time: wall-clock time (float).
        info: 0 if success, 1 if lstsq fallback.
    """
    t0 = time.time()
    if device is None:
        device = torch.device('cuda')

    world_size = dist.get_world_size()
    C = param_chunk_size

    # --- Detect whether local_lpg lives on CPU ---
    if isinstance(local_lpg, torch.Tensor):
        lpgloc_on_cpu = local_lpg.device.type == 'cpu'
    else:
        lpgloc_on_cpu = True  # numpy

    # --- Normalize inputs ---
    if not isinstance(local_lpg, torch.Tensor):
        local_lpg = torch.as_tensor(
            local_lpg, dtype=torch.float64,
        ).contiguous()
    else:
        local_lpg = local_lpg.to(dtype=torch.float64).contiguous()

    if not isinstance(local_energies, torch.Tensor):
        local_energies = torch.tensor(
            local_energies, device=device, dtype=torch.float64,
        )
    else:
        local_energies = local_energies.to(
            device=device, dtype=torch.float64,
        )

    n_local = local_energies.shape[0]  # Ns_per_rank

    # --- Compute global statistics via all_reduce on (Np,) ---
    # When local_lpg is on CPU, chunk the sum/dot to avoid
    # uploading the full matrix.
    if n_local > 0:
        if lpgloc_on_cpu:
            local_sum_lpg = torch.zeros(
                n_params, device=device, dtype=torch.float64,
            )
            local_sum_EO = torch.zeros(
                n_params, device=device, dtype=torch.float64,
            )
            E_cpu = local_energies.cpu()
            for start in range(0, n_params, C):
                end = min(start + C, n_params)
                chunk_gpu = local_lpg[:, start:end].to(device)
                local_sum_lpg[start:end] = chunk_gpu.sum(dim=0)
                local_sum_EO[start:end] = E_cpu @ local_lpg[:, start:end]
                # sum_EO chunk: (Ns,)@(Ns,c) on CPU is fine,
                # result is small (c,)
            local_sum_EO = local_sum_EO.to(device)
            del E_cpu
        else:
            # local_lpg already on GPU
            if local_lpg.device != device:
                local_lpg = local_lpg.to(device)
            local_sum_lpg = local_lpg.sum(dim=0)
            local_sum_EO = local_energies @ local_lpg
    else:
        local_sum_lpg = torch.zeros(
            n_params, device=device, dtype=torch.float64,
        )
        local_sum_EO = torch.zeros(
            n_params, device=device, dtype=torch.float64,
        )

    if world_size > 1:
        dist.all_reduce(local_sum_lpg, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sum_EO, op=dist.ReduceOp.SUM)

    mean_lpg = local_sum_lpg / total_samples            # (Np,) GPU
    mean_EO = local_sum_EO / total_samples          # (Np,) GPU
    energy_grad = mean_EO - energy_mean * mean_lpg    # (Np,) GPU

    if not do_SR:
        t1 = time.time()
        return energy_grad, t1 - t0, None

    # --- Center and scale local_lpg in-place ---
    # lpg_scaled = (O - mean_lpg) / sqrt(Ns)
    # Safe: caller doesn't reuse local_lpg after solver returns.
    if lpgloc_on_cpu:
        mean_lpg_cpu = mean_lpg.cpu()
        local_lpg -= mean_lpg_cpu.unsqueeze(0)
        local_lpg /= math.sqrt(total_samples)
        del mean_lpg_cpu
    else:
        local_lpg -= mean_lpg.unsqueeze(0)
        local_lpg /= math.sqrt(total_samples)

    # --- Gather energies (cheap: Ns_total floats) ---
    local_E_scaled = (
        (local_energies - energy_mean) / math.sqrt(total_samples)
    )  # (Ns_per_rank,) GPU

    if world_size > 1:
        total_E = torch.zeros(
            total_samples, device=device, dtype=torch.float64,
        )
        dist.all_gather_into_tensor(total_E, local_E_scaled)
    else:
        total_E = local_E_scaled

    # --- Build Gram matrix G = lpg_scaled @ lpg_scaled^T via param chunks ---
    G = torch.zeros(
        (total_samples, total_samples),
        device=device, dtype=torch.float64,
    )

    if world_size > 1:
        gather_buf = torch.zeros(
            (total_samples, C),
            device=device, dtype=torch.float64,
        )

        for start in range(0, n_params, C):
            end = min(start + C, n_params)
            chunk_size = end - start
            # Stream from CPU if needed
            local_chunk = local_lpg[:, start:end].to(
                device=device,
            ).contiguous()

            if chunk_size < C:
                gather_buf_cur = torch.zeros(
                    (total_samples, chunk_size),
                    device=device, dtype=torch.float64,
                )
                dist.all_gather_into_tensor(
                    gather_buf_cur, local_chunk,
                )
            else:
                dist.all_gather_into_tensor(
                    gather_buf, local_chunk,
                )
                gather_buf_cur = gather_buf

            G.addmm_(gather_buf_cur, gather_buf_cur.T)
            del local_chunk

        del gather_buf
    else:
        for start in range(0, n_params, C):
            end = min(start + C, n_params)
            chunk = local_lpg[:, start:end].to(device)
            G.addmm_(chunk, chunk.T)
            del chunk

    # --- Solve (G + lambda*I) alpha = E_s ---
    G.diagonal().add_(diag_shift)
    info = 0
    try:
        alpha = torch.linalg.solve(G, total_E)
    except RuntimeError:
        alpha = torch.linalg.lstsq(G, total_E).solution
        info = 1
    del G

    # --- Reconstruct dp = lpg_scaled^T @ alpha_local ---
    if world_size > 1:
        rank = dist.get_rank()
        alpha_local = alpha[
            rank * n_local:(rank + 1) * n_local
        ]  # (Ns_per_rank,) GPU
    else:
        alpha_local = alpha

    # Chunk the dp reconstruction along params to avoid
    # uploading full (Ns_per_rank, Np) to GPU.
    dp = torch.zeros(
        n_params, device=device, dtype=torch.float64,
    )
    for start in range(0, n_params, C):
        end = min(start + C, n_params)
        chunk = local_lpg[:, start:end].to(device)
        dp[start:end] = chunk.T @ alpha_local
        del chunk

    if world_size > 1:
        dist.all_reduce(dp, op=dist.ReduceOp.SUM)

    t1 = time.time()
    return dp, t1 - t0, info


class DistributedSRMinresGPU(PreconditionerGPU):
    def __init__(
        self,
        rtol: float = 5e-5,
        maxiter: int = 100,
        use_scipy: bool = False,
    ):
        self.rtol = rtol
        self.maxiter = maxiter
        self.use_scipy = use_scipy

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
    ) -> Tuple[Any, float, Any]:
        return distributed_minres_solver_gpu(
            local_lpg=local_o,
            local_energies=local_energies,
            energy_mean=energy_mean,
            total_samples=total_samples,
            n_params=n_params,
            diag_shift=diag_shift,
            rtol=self.rtol,
            maxiter=self.maxiter,
            run_SR=run_sr,
            use_scipy=self.use_scipy,
            device=device,
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
    ) -> Tuple[Any, float, Any]:
        return minSR_solver_gpu(
            local_lpg=local_o,
            local_energies=local_energies,
            energy_mean=energy_mean,
            total_samples=total_samples,
            n_params=n_params,
            diag_shift=diag_shift,
            device=device,
            do_SR=run_sr,
        )


class DistributedMinSRGPU(PreconditionerGPU):
    """Distributed minSR: chunked Gram matrix, no full gather.

    Keeps lpg_loc on each rank's GPU and builds the (Ns x Ns) Gram
    matrix incrementally by gathering param chunks of size C.
    Memory: O(Ns^2 + Ns*C) instead of O(Ns*Np).

    When offload_cpu=True, VMC_GPU eagerly offloads each
    (B_grad, Np) gradient chunk to CPU inside compute_grads_gpu,
    so peak GPU memory is O(B_grad * Np) not O(B * Np).
    The solver then streams param-chunks from CPU to GPU for
    G-build and dp reconstruction.
    """

    def __init__(
        self,
        param_chunk_size: int = 1024,
        offload_cpu: bool = True,
    ):
        self.param_chunk_size = param_chunk_size
        self.offload_cpu = offload_cpu

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
    ) -> Tuple[Any, float, Any]:
        return distributed_minSR_solver_gpu(
            local_lpg=local_o,
            local_energies=local_energies,
            energy_mean=energy_mean,
            total_samples=total_samples,
            n_params=n_params,
            diag_shift=diag_shift,
            param_chunk_size=self.param_chunk_size,
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
    "DistributedMinSRGPU",
]
