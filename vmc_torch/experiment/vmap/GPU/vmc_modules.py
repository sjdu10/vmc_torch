"""
GPU-compatible modular VMC functions using torch.distributed (NCCL).

Mirrors the CPU vmap_modules.py but uses torch.distributed.all_reduce
instead of MPI.Allreduce.  Key memory optimization: O_loc = grad/amp
is computed inline during sampling (Trick #2), so we never hold both
the raw (B, Np) gradient matrix and the (B,) amplitude vector at the
same time across the gathering step.

Functions:
    run_sampling_phase_gpu  — local sampling + energy + inline O_loc
    distributed_minres_solver_gpu — distributed MINRES via all_reduce
    minSR_solver_gpu        — minSR direct solve (gather to rank 0)
"""
import time
import numpy as np
import torch
import torch.distributed as dist
import scipy.sparse.linalg as spla

from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    sample_next,
    evaluate_energy,
    compute_grads_gpu,
)


# ===========================================================================
# Sampling Phase
# ===========================================================================
@torch.inference_mode()
def run_sampling_phase_gpu(
    fxs,
    model,
    hamiltonian,
    graph,
    Ns,
    grad_batch_size=None,
    burn_in=False,
    burn_in_steps=4,
    hopping_rate=0.25,
    verbose=False,
):
    """
    Local sampling phase on GPU with inline O_loc computation.

    All ranks run independently (no master-worker dispatch).
    Runs multiple MCMC sweeps of B walkers to accumulate Ns total
    samples, where B = fxs.shape[0] is the walker batch size.

    Args:
        fxs: (B, N_sites) current walker configs, GPU tensor.
            B is the walker batch size (number of parallel chains).
        model: nn.Module on GPU
        hamiltonian: Hamiltonian object (with .get_conn)
        graph: lattice graph (with .row_edges, .col_edges)
        Ns: total samples this rank should produce (across
            multiple sweeps of B walkers)
        grad_batch_size: chunk size for gradient computation
        burn_in: whether to do burn-in sweeps
        burn_in_steps: number of burn-in sweeps
        hopping_rate: MCMC hopping rate
        verbose: print timing info

    Returns:
        (local_energies, local_O): numpy arrays, shapes
            (n_local,) and (n_local, Np)
        fxs: updated walker configs (GPU tensor)
        sample_time: float
    """
    B = fxs.shape[0]
    t_start = time.time()

    # Burn-in
    if burn_in:
        for _ in range(burn_in_steps):
            fxs, _ = sample_next(
                fxs, model, graph, hopping_rate=hopping_rate
            )

    local_energies_list = []
    local_O_list = []
    current_count = 0

    while current_count < Ns:
        needed = min(B, Ns - current_count)

        # 1. Sample (one MCMC sweep over B walkers)
        fxs, current_amps = sample_next(
            fxs, model, graph, hopping_rate=hopping_rate
        )

        # 2. Energy
        _, local_E = evaluate_energy(
            fxs, model, hamiltonian, current_amps
        )

        # 3. Grads + inline O_loc (Trick #2)
        with torch.enable_grad():
            local_grads, local_amps = compute_grads_gpu(
                fxs, model,
                vectorize=True,
                batch_size=grad_batch_size,
                vmap_grad=True,
            )

        # O_loc = grads / amps (in-place on grads to save memory)
        local_grads /= local_amps.unsqueeze(1)
        # Now local_grads IS O_loc

        # Move to CPU numpy (frees GPU memory for next batch)
        local_energies_list.append(
            local_E[:needed].detach().cpu().numpy()
        )
        local_O_list.append(
            local_grads[:needed].detach().cpu().numpy()
        )

        current_count += needed

        # Explicit cleanup
        del local_grads, local_amps, local_E

    local_energies = np.concatenate(local_energies_list)
    local_O = np.concatenate(local_O_list)

    sample_time = time.time() - t_start

    if verbose:
        rank = dist.get_rank()
        n_sweeps = len(local_energies_list)
        print(
            f"  Rank {rank}: {local_energies.shape[0]} samples "
            f"({n_sweeps} sweeps x B={B}), "
            f"T={sample_time:.2f}s"
        )

    return (local_energies, local_O), fxs, sample_time


# ===========================================================================
# Distributed MINRES SR Solver (via torch.distributed.all_reduce)
# ===========================================================================
def distributed_minres_solver_gpu(
    local_O,
    local_energies,
    energy_mean,
    total_samples,
    n_params,
    diag_shift,
    rtol=1e-4,
    maxiter=100,
):
    """
    Distributed SR solver using MINRES with torch.distributed.all_reduce.

    Each rank holds its local O_loc chunk. The matrix-vector products
    for S = (1/N) O^T O - mean_O @ mean_O^T + diag_shift * I
    are computed distributedly via matvec (no Np x Np matrix built).

    For single-GPU (world_size=1), the all_reduce is a no-op and
    the matvec is pure CPU numpy with zero overhead.

    Args:
        local_O: (n_local, Np) numpy array, pre-computed O_loc
        local_energies: (n_local,) numpy array
        energy_mean: global mean energy (float)
        total_samples: total samples across all ranks
        n_params: number of parameters
        diag_shift: diagonal regularization
        rtol: MINRES tolerance
        maxiter: max MINRES iterations

    Returns:
        dp: (Np,) numpy array, parameter update
        sr_time: wall-clock time
        info: MINRES convergence flag
    """
    t0 = time.time()
    n_local = local_energies.shape[0]
    world_size = dist.get_world_size()

    # --- Compute global statistics via all_reduce ---
    if n_local > 0:
        local_sum_O = np.sum(local_O, axis=0)            # (Np,)
        local_sum_EO = np.dot(local_energies, local_O)   # (Np,)
    else:
        local_sum_O = np.zeros(n_params, dtype=np.float64)
        local_sum_EO = np.zeros(n_params, dtype=np.float64)

    if world_size > 1:
        device = torch.device('cuda')
        sum_O_t = torch.tensor(local_sum_O, device=device)
        sum_EO_t = torch.tensor(local_sum_EO, device=device)
        dist.all_reduce(sum_O_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_EO_t, op=dist.ReduceOp.SUM)
        global_sum_O = sum_O_t.cpu().numpy()
        global_sum_EO = sum_EO_t.cpu().numpy()
    else:
        # Single rank: no communication needed
        global_sum_O = local_sum_O
        global_sum_EO = local_sum_EO

    mean_O = global_sum_O / total_samples
    mean_EO = global_sum_EO / total_samples
    energy_grad = mean_EO - energy_mean * mean_O  # (Np,)

    # --- MINRES solver with matvec (no Np x Np matrix) ---
    if world_size == 1:
        # Single rank: pure numpy matvec, no GPU transfers
        def matvec(x):
            inner = local_O.dot(x)           # (n_local,)
            Sx = local_O.T.dot(inner)         # (Np,)
            Sx /= total_samples
            Sx -= np.dot(mean_O, x) * mean_O
            return Sx + diag_shift * x
    else:
        # Multi-rank: local matmul + all_reduce of (Np,) vector
        device = torch.device('cuda')

        def matvec(x):
            if n_local > 0:
                inner = local_O.dot(x)
                local_Sx = local_O.T.dot(inner)
            else:
                local_Sx = np.zeros_like(x)

            Sx_t = torch.tensor(local_Sx, device=device)
            dist.all_reduce(Sx_t, op=dist.ReduceOp.SUM)
            Sx = Sx_t.cpu().numpy()

            Sx /= total_samples
            Sx -= np.dot(mean_O, x) * mean_O
            return Sx + diag_shift * x

    A = spla.LinearOperator(
        (n_params, n_params), matvec=matvec, dtype=np.float64
    )
    dp, info = spla.minres(A, energy_grad, rtol=rtol, maxiter=maxiter)

    t1 = time.time()
    return dp, t1 - t0, info


# ===========================================================================
# MinSR Direct Solver (gather to rank 0, GPU linear algebra)
# ===========================================================================
def minSR_solver_gpu(
    local_O,
    local_energies,
    energy_mean,
    total_samples,
    n_params,
    diag_shift=1e-4,
    device=None,
):
    """
    MinSR direct solver: gather O_loc to rank 0, solve in sample space.

    Efficient when Total_Ns < Np (common for NN-based VMC).
    Uses GPU linear algebra for the (Ns x Ns) solve.

    Args:
        local_O: (n_local, Np) numpy array
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
    t0 = time.time()
    if device is None:
        device = torch.device('cuda')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Convert to GPU tensors
    local_O_t = torch.tensor(
        local_O, device=device, dtype=torch.float64
    ).contiguous()
    local_E_t = torch.tensor(
        local_energies, device=device, dtype=torch.float64
    ).contiguous()

    if world_size > 1:
        # Gather O_loc and energies across ranks
        total_O_t = torch.zeros(
            (total_samples, n_params),
            device=device, dtype=torch.float64,
        )
        total_E_t = torch.zeros(
            total_samples, device=device, dtype=torch.float64,
        )
        dist.all_gather_into_tensor(total_O_t, local_O_t)
        dist.all_gather_into_tensor(total_E_t, local_E_t)
    else:
        total_O_t = local_O_t
        total_E_t = local_E_t

    info = 0
    dp_t = torch.zeros(
        n_params, device=device, dtype=torch.float64
    )

    if rank == 0:
        O_mean = torch.mean(total_O_t, dim=0)
        O_centered = total_O_t - O_mean.unsqueeze(0)
        O_sk = O_centered / np.sqrt(total_samples)
        E_s = (total_E_t - energy_mean) / np.sqrt(total_samples)

        # Gram matrix T = O_sk @ O_sk^T  (Ns x Ns)
        T = O_sk @ O_sk.T
        T += diag_shift * torch.eye(
            total_samples, device=device, dtype=torch.float64,
        )

        try:
            x = torch.linalg.solve(T, E_s)
        except RuntimeError:
            x = torch.linalg.lstsq(T, E_s).solution
            info = 1

        dp_t = O_sk.T @ x

    if world_size > 1:
        dist.broadcast(dp_t, src=0)

    dp = dp_t.cpu().numpy()
    t1 = time.time()
    return dp, t1 - t0, info
