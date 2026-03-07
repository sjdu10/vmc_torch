"""
GPU-compatible modular VMC functions using torch.distributed (NCCL).

Mirrors the CPU vmap_modules.py but uses torch.distributed.all_reduce
instead of MPI.Allreduce.  Key memory optimization: log_psi_grad = grad/amp
is computed inline during sampling (Trick #2), so we never hold both
the raw (B, Np) gradient matrix and the (B,) amplitude vector at the
same time across the gathering step.

Functions:
    run_sampling_phase_gpu  — local sampling + energy + inline log_psi_grad
    distributed_minres_solver_gpu — distributed MINRES via all_reduce
    minSR_solver_gpu        — minSR direct solve (gather to rank 0)
    torch_minres            — pure-PyTorch MINRES (GPU-native)
"""
import math
import time
import numpy as np
import torch
import torch.distributed as dist
import scipy.sparse.linalg as spla

from vmc_torch.GPU.vmc_utils import (
    sample_next,
    evaluate_energy,
    compute_grads_gpu,
)


# ===========================================================================
# Sampling Phase
# ===========================================================================
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
    compile=False,
    use_log_amp=False,
):
    """Local sampling phase on GPU with inline log_psi_grad.

    All ranks run independently (no master-worker dispatch).
    Runs multiple MCMC sweeps of B walkers to accumulate Ns total
    samples, where B = fxs.shape[0] is the walker batch size.

    Args:
        fxs: (B, N_sites) current walker configs, GPU tensor.
        model: nn.Module on GPU
        hamiltonian: Hamiltonian object (with .get_conn)
        graph: lattice graph (with .row_edges, .col_edges)
        Ns: total samples this rank should produce
        grad_batch_size: chunk size for gradient computation
        burn_in: whether to do burn-in sweeps
        burn_in_steps: number of burn-in sweeps
        hopping_rate: MCMC hopping rate
        verbose: print timing info
        use_log_amp: if True, use log-amplitude space
            throughout sampling, energy, and gradients.

    Returns:
        (local_energies, local_log_psi_grad): GPU tensors,
            shapes (n_local,) and (n_local, Np)
        fxs: updated walker configs (GPU tensor)
        sample_time: float
    """
    B = fxs.shape[0]
    t_samp_total = 0.0
    t_locE_total = 0.0
    t_grad_total = 0.0

    # Burn-in
    if burn_in:
        t_burn = time.time()
        for _ in range(burn_in_steps):
            fxs, _ = sample_next(
                fxs, model, graph,
                hopping_rate=hopping_rate, compile=compile,
            )
        t_samp_total += time.time() - t_burn

    local_energies_list = []
    local_lpg_list = []
    current_count = 0

    while current_count < Ns:
        needed = min(B, Ns - current_count)

        # 1. Sample (one MCMC sweep over B walkers)
        t0 = time.time()
        fxs, current_amps = sample_next(
            fxs, model, graph,
            hopping_rate=hopping_rate, compile=compile,
        )
        t_samp_total += time.time() - t0

        # 2. Energy
        t0 = time.time()
        _, local_E = evaluate_energy(
            fxs, model, hamiltonian, current_amps,
        )
        t_locE_total += time.time() - t0

        # Free sampling/energy tensors so allocator
        # can reuse their blocks for grad computation
        del current_amps

        # 3. Grads -> log_psi_grad
        t0 = time.time()
        with torch.enable_grad():
            local_grads, grads_aux = compute_grads_gpu(
                fxs, model,
                vectorize=True,
                batch_size=grad_batch_size,
                vmap_grad=True,
                use_log_amp=use_log_amp,
            )

        if not use_log_amp:
            # log_psi_grad = grads / amps (in-place to save memory)
            local_grads /= grads_aux.unsqueeze(1)
        # Now local_grads IS log_psi_grad
        t_grad_total += time.time() - t0

        local_energies_list.append(
            local_E[:needed].detach(),
        )
        local_lpg_list.append(
            local_grads[:needed].detach(),
        )

        current_count += needed

        # Explicit cleanup
        del local_grads, grads_aux, local_E

    local_energies = torch.cat(
        local_energies_list, dim=0,
    )  # (Ns,) GPU
    local_log_psi_grad = torch.cat(
        local_lpg_list, dim=0,
    )  # (Ns, Np) GPU

    sample_time = t_samp_total + t_locE_total + t_grad_total
    phase_times = {
        't_samp': t_samp_total,
        't_locE': t_locE_total,
        't_grad': t_grad_total,
    }

    if verbose:
        rank = dist.get_rank()
        n_sweeps = len(local_energies_list)
        print(
            f"  Rank {rank}: "
            f"{local_energies.shape[0]} samples "
            f"({n_sweeps} sweeps x B={B}), "
            f"T={sample_time:.2f}s "
            f"(samp={t_samp_total:.2f} "
            f"locE={t_locE_total:.2f} "
            f"grad={t_grad_total:.2f})"
        )

    return (
        (local_energies, local_log_psi_grad),
        fxs,
        sample_time,
        phase_times,
    )


# ===========================================================================
# Pure-PyTorch MINRES (Paige & Saunders 1975)
# ===========================================================================
def torch_minres(matvec, b, rtol=1e-5, maxiter=100):
    """MINRES solver in pure PyTorch — runs entirely on GPU.

    Solves A x = b where A is symmetric (accessed via matvec).
    Implements Paige & Saunders (1975), mirroring scipy's
    implementation exactly.

    One matvec call per iteration; all other ops are O(Np) vector
    arithmetic.  Scalar extractions (.item()) are negligible
    versus the matvec cost.

    Args:
        matvec: callable, x -> A @ x (GPU tensor in/out).
        b: (Np,) right-hand-side GPU tensor.
        rtol: relative tolerance |r| / |b| < rtol.
        maxiter: maximum Lanczos iterations.

    Returns:
        x: (Np,) solution GPU tensor.
        info: 0 if converged, else maxiter.
    """
    b_norm = torch.linalg.norm(b).item()
    if b_norm == 0:
        return torch.zeros_like(b), 0

    # Lanczos init: r1 = b, beta1 = ||b||
    n = b.shape[0]
    x = torch.zeros_like(b)
    r1 = b.clone()
    r2 = b.clone()
    beta1 = b_norm
    beta = beta1

    # Givens rotation state
    cs = -1.0
    sn = 0.0
    oldb = 0.0
    dbar = 0.0
    epsln = 0.0
    phibar = beta1

    # w vectors for solution update
    w = torch.zeros_like(b)
    w2 = torch.zeros_like(b)

    info = maxiter
    for itn in range(1, maxiter + 1):
        # Lanczos step
        s = 1.0 / beta
        v = s * r2                          # v_k

        y = matvec(v)

        if itn >= 2:
            y = y - (beta / oldb) * r1

        alfa = torch.dot(v, y).item()
        y = y - (alfa / beta) * r2

        r1 = r2
        r2 = y
        oldb = beta
        beta = torch.linalg.norm(r2).item()

        # Apply previous rotation Q_{k-1}
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta

        # Compute new rotation Q_k
        gamma = math.sqrt(gbar ** 2 + beta ** 2)
        gamma = max(gamma, 1e-300)
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar

        # Update x
        denom = 1.0 / gamma
        w1 = w2
        w2 = w
        w = (v - oldeps * w1 - delta * w2) * denom
        x = x + phi * w

        # Convergence: |r| / |b|
        if abs(phibar) < rtol * b_norm:
            info = 0
            break

        if beta == 0.0:
            info = 0
            break

    return x, info


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
):
    """
    Distributed SR solver using MINRES with torch.distributed.all_reduce.

    Each rank holds its local log_psi_grad chunk. The matrix-vector products
    for S = (1/N) lpg^T lpg - mean_lpg @ mean_lpg^T + diag_shift * I
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
    device = torch.device('cuda')

    # --- Check if log_psi_grad is already on CPU ---
    oloc_on_cpu = (
        isinstance(local_lpg, torch.Tensor)
        and local_lpg.device.type == 'cpu'
    ) or isinstance(local_lpg, np.ndarray)

    if use_scipy and oloc_on_cpu:
        # --- Fast path: log_psi_grad already on CPU ---
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

        # scipy MINRES on CPU (log_psi_grad stays on CPU)
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
    if not isinstance(local_lpg, torch.Tensor):
        local_lpg = torch.tensor(
            local_lpg, device=device, dtype=torch.float64,
        )
    else:
        local_lpg = local_lpg.to(
            device=device, dtype=torch.float64,
        )
    if not isinstance(local_energies, torch.Tensor):
        local_energies = torch.tensor(
            local_energies, device=device, dtype=torch.float64,
        )
    else:
        local_energies = local_energies.to(
            device=device, dtype=torch.float64,
        )
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

    # # --- MINRES solve ---
    # if use_scipy:
    #     # Fallback: move to CPU numpy, use scipy MINRES
    #     local_lpg_np = local_lpg.cpu().numpy()
    #     mean_lpg_np = mean_lpg.cpu().numpy()
    #     energy_grad_np = energy_grad.cpu().numpy()

    #     if world_size == 1:
    #         def matvec(x):
    #             inner = local_lpg_np.dot(x)
    #             Sx = local_lpg_np.T.dot(inner)
    #             Sx /= total_samples
    #             Sx -= np.dot(mean_lpg_np, x) * mean_lpg_np
    #             return Sx + diag_shift * x
    #     else:
    #         def matvec(x):
    #             if n_local > 0:
    #                 inner = local_lpg_np.dot(x)
    #                 local_Sx = local_lpg_np.T.dot(inner)
    #             else:
    #                 local_Sx = np.zeros_like(x)
    #             Sx_t = torch.tensor(local_Sx, device=device)
    #             dist.all_reduce(Sx_t, op=dist.ReduceOp.SUM)
    #             Sx = Sx_t.cpu().numpy()
    #             Sx /= total_samples
    #             Sx -= np.dot(mean_lpg_np, x) * mean_lpg_np
    #             return Sx + diag_shift * x

    #     A = spla.LinearOperator(
    #         (n_params, n_params), matvec=matvec,
    #         dtype=np.float64,
    #     )
    #     dp, info = spla.minres(
    #         A, energy_grad_np, rtol=rtol, maxiter=maxiter,
    #     )
    #     t1 = time.time()
    #     return dp, t1 - t0, info

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
    MinSR direct solver: gather log_psi_grad to rank 0, solve in sample space.

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
            # Gather log_psi_grad and energies across ranks
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
    - CPU path (offload_oloc): local_lpg lives on CPU.  Each
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
        oloc_on_cpu = local_lpg.device.type == 'cpu'
    else:
        oloc_on_cpu = True  # numpy

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
        if oloc_on_cpu:
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
    if oloc_on_cpu:
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

