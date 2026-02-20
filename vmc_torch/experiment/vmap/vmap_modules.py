import sys
import numpy as np
import time
import torch
from tqdm import tqdm
from vmc_torch.experiment.vmap.vmap_utils import (
    sample_next,
    evaluate_energy,
    sample_next_vec,
    evaluate_energy_vec,
    sample_next_reuse,
    evaluate_energy_reuse,
)
from vmc_torch.experiment.vmap.vmap_torch_utils import use_jitter_svd
import scipy.sparse.linalg as spla
from mpi4py import MPI

def gentle_barrier(comm, sleep_interval=0.01):
    """
    A non-blocking barrier that periodically yields CPU to avoid busy-waiting.
    """
    # non-blocking barrier
    request = comm.Ibarrier()
    
    # check completion in a loop
    while not request.Test():
        # key: yield CPU while waiting
        time.sleep(sleep_interval)

def distributed_minres_solver_v0(
    local_grads,
    local_amps,
    local_energies,
    energy_mean,
    total_samples,
    n_params,
    diag_shift,
    comm,
    rtol=1e-4,
):
    """
    [BACKUP] Original SR solver that takes separate grads and amps.
    Replaced by distributed_minres_solver which takes pre-computed O_loc.

    使用 MinRes 求解 SR 方程 S * dp = g (分布式 Matrix-Free 版本)

    Args:
        local_grads: (n_local, Np) 本地梯度
        local_amps: (n_local,) 本地振幅
        local_energies: (n_local,) 本地能量
        energy_mean: 全局能量均值
        total_samples: 全局样本总数
        n_params: 参数总数 (Np)
        diag_shift: 对角偏移量 (SR regularization)
        comm: MPI communicator
        rtol: MinRes 相对收敛容限

    Returns:
        dp: (Np,) 参数更新量
        sr_time: 求解耗时
    """
    t0 = MPI.Wtime()
    n_local = local_energies.shape[0]
    
    # 1. 预处理局部数据 (Prepare Local Data)
    if n_local > 0:
        amps_reshaped = local_amps.reshape(-1, 1)
        # O_loc = grad(psi) / psi
        local_O = local_grads / amps_reshaped 
        
        local_sum_O = np.sum(local_O, axis=0)
        local_sum_EO = np.dot(local_energies, local_O)
    else:
        # Rank 0 (Master) 没有任何样本，占位
        local_O = np.zeros((0, n_params), dtype=np.float64)
        local_sum_O = np.zeros(n_params, dtype=np.float64)
        local_sum_EO = np.zeros(n_params, dtype=np.float64)

    # 2. 计算全局梯度 (Compute Global Gradient)
    global_sum_O = np.zeros_like(local_sum_O)
    global_sum_EO = np.zeros_like(local_sum_EO)
    
    comm.Allreduce(local_sum_O, global_sum_O, op=MPI.SUM)
    comm.Allreduce(local_sum_EO, global_sum_EO, op=MPI.SUM)
    
    mean_O = global_sum_O / total_samples
    mean_EO = global_sum_EO / total_samples
    
    # g = <E*O> - <E><O>
    energy_grad = mean_EO - energy_mean * mean_O

    # 3. 定义分布式矩阵乘法 (Distributed MatVec)
    def matvec(x):
        # A. 局部计算
        if n_local > 0:
            # inner = O_i . x  -> shape (n_local,)
            inner = local_O.dot(x)
            # local_res = O_i.T . inner -> shape (Np,)
            local_Sx = local_O.T.dot(inner)
        else:
            local_Sx = np.zeros_like(x)
        
        # B. 全局同步
        global_Sx = np.zeros_like(x)
        comm.Allreduce(local_Sx, global_Sx, op=MPI.SUM)
        
        # C. 均值项修正与归一化
        Sx = global_Sx / total_samples
        # 减去中心化项: <O> * (<O> . x)
        mean_O_dot_x = np.dot(mean_O, x)
        Sx -= mean_O_dot_x * mean_O
        
        # D. 对角正则化
        return Sx + diag_shift * x

    # 4. 运行 MinRes
    A = spla.LinearOperator((n_params, n_params), matvec=matvec, dtype=np.float64)
    dp, info = spla.minres(A, energy_grad, rtol=rtol, maxiter=100)
    
    t1 = MPI.Wtime()
    return dp, t1 - t0, info

def distributed_minres_solver(
    local_O,
    local_energies,
    energy_mean,
    total_samples,
    n_params,
    diag_shift,
    comm,
    rtol=1e-4,
):
    """
    SR solver that takes pre-computed O_loc directly.

    Trick #1: caller already computed O_loc = grad/amp, so no
    extra (n_local, Np) allocation here.

    Args:
        local_O: (n_local, Np) pre-computed O_loc = grad/amp
        local_energies: (n_local,) local energies
        energy_mean: global mean energy
        total_samples: total samples across all ranks
        n_params: number of parameters
        diag_shift: diagonal regularization
        comm: MPI communicator
        rtol: MINRES tolerance

    Returns:
        dp: (Np,) parameter update
        sr_time: wall-clock time
        info: MINRES convergence flag
    """
    t0 = MPI.Wtime()
    n_local = local_energies.shape[0]

    if n_local > 0:
        local_sum_O = np.sum(local_O, axis=0)
        local_sum_EO = np.dot(local_energies, local_O)
    else:
        local_sum_O = np.zeros(n_params, dtype=np.float64)
        local_sum_EO = np.zeros(n_params, dtype=np.float64)

    global_sum_O = np.zeros_like(local_sum_O)
    global_sum_EO = np.zeros_like(local_sum_EO)

    comm.Allreduce(local_sum_O, global_sum_O, op=MPI.SUM)
    comm.Allreduce(local_sum_EO, global_sum_EO, op=MPI.SUM)

    mean_O = global_sum_O / total_samples
    mean_EO = global_sum_EO / total_samples
    energy_grad = mean_EO - energy_mean * mean_O

    def matvec(x):
        if n_local > 0:
            inner = local_O.dot(x)
            local_Sx = local_O.T.dot(inner)
        else:
            local_Sx = np.zeros_like(x)

        global_Sx = np.zeros_like(x)
        comm.Allreduce(local_Sx, global_Sx, op=MPI.SUM)

        Sx = global_Sx / total_samples
        Sx -= np.dot(mean_O, x) * mean_O
        return Sx + diag_shift * x

    A = spla.LinearOperator(
        (n_params, n_params), matvec=matvec, dtype=np.float64
    )
    dp, info = spla.minres(A, energy_grad, rtol=rtol, maxiter=100)

    t1 = MPI.Wtime()
    return dp, t1 - t0, info

def run_sampling_phase_v0(
    svmc: int,
    Ns: int,
    B: int,
    fxs: torch.Tensor,
    model: torch.nn.Module,
    hamiltonian,
    graph,
    get_grads_func: callable,
    comm,
    rank: int,
    size: int,
    should_burn_in=False,
    burn_in_steps=10,
    sampling_hopping_rate=0.25,
    verbose=False
):
    """
    [BACKUP] Original sampling phase that stores grads and amps separately.
    Replaced by run_sampling_phase which computes O_loc inline.

    Returns:
        local_data: (energies, grads, amps), numpy arrays
        fxs: initial configs in a batch
        stats: dict, containing timing statistics and sample counts
    """
    # Define Tags
    TAG_REQ = 100
    TAG_CMD = 101
    CMD_CONTINUE = 0
    CMD_STOP = 1

    sample_time = 0.0
    local_energy_time = 0.0
    grad_time = 0.0
    
    # Pre-allocate buffers: upper bound on samples this worker could collect
    max_n_local = int(np.ceil(Ns / max(size - 1, 1) / B)) * B
    res_energies = None  # lazily allocated after first batch (to learn n_params)
    res_grads = None
    res_amps = None
    n_local = 0

    t0 = MPI.Wtime()

    if should_burn_in:
        if rank != 0:
            current_step = 0
            while current_step < burn_in_steps:
                fxs, _ = sample_next(fxs, model, graph, hopping_rate=sampling_hopping_rate, verbose=False)
                current_step += 1
        else:
            pass

    gentle_barrier(comm, sleep_interval=0.01) # ensure all ranks finish burn-in before master starts dispatching tasks

    # --- Branch A: Master (Rank 0) ---
    if rank == 0:
        pbar = tqdm(total=Ns, desc=f"Step {svmc} Sampling", unit="samples")
        n_collected = 0
        n_dispatched = 0
        active_workers = size - 1
        active_rank_ids = set(range(1, size))

        while active_workers > 0:
            has_message = comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_REQ)
            if has_message:
                status = MPI.Status()
                buf = np.empty(1, dtype=np.int32)
                comm.Recv([buf, MPI.INT], source=MPI.ANY_SOURCE, tag=TAG_REQ, status=status)
                source_rank = status.Get_source()
                finished_batch = buf[0]

                if finished_batch > 0:
                    n_collected += finished_batch
                    pbar.update(finished_batch)

                next_batch = B
                if n_dispatched < Ns:
                    cmd = np.array([CMD_CONTINUE], dtype=np.int32)
                    comm.Send([cmd, MPI.INT], dest=source_rank, tag=TAG_CMD)
                    n_dispatched += next_batch
                    if verbose:
                        print(f"[Master] Dispatched {next_batch} samples to Rank {source_rank}. Total dispatched: {n_dispatched}.", flush=True)
                else:
                    cmd = np.array([CMD_STOP], dtype=np.int32)
                    comm.Send([cmd, MPI.INT], dest=source_rank, tag=TAG_CMD)
                    active_workers -= 1
                    if source_rank in active_rank_ids:
                        active_rank_ids.remove(source_rank)
                        if verbose:
                            print(f'[Master] Kill rank {source_rank} with {finished_batch} samples. Remaining num of active workers: {len(active_rank_ids)}, {active_workers}', flush=True)

            else:
                if n_collected >= Ns:
                    if verbose:
                        print(f"\n[Master] Waiting for {active_workers} workers to finish burn-in.", flush=True)
                        time.sleep(0.1) # yield CPU to let stragglers finish burn-in
                time.sleep(0.001)

        if verbose:
            print('Sampling phase should be done now.', flush=True)

        pbar.close()

        if len(active_rank_ids) > 0:
            print(f"ERROR: Finishing with {len(active_rank_ids)} dead ranks.", flush=True)
            comm.Abort(1)

    # --- Branch B: Worker ---
    else:
        try:
            last_finished_batch = 0
            while True:
                # 1. Request / Report
                buf = np.array([last_finished_batch], dtype=np.int32)
                comm.Send([buf, MPI.INT], dest=0, tag=TAG_REQ)

                # 2. Wait Command
                cmd = np.empty(1, dtype=np.int32)
                comm.Recv([cmd, MPI.INT], source=0, tag=TAG_CMD)

                if cmd[0] == CMD_STOP:
                    break

                # 3. Compute
                # try:
                t00 = MPI.Wtime()
                fxs, current_amps = sample_next(fxs, model, graph, hopping_rate=sampling_hopping_rate, verbose=False)
                t11 = MPI.Wtime()
                energy_batch, local_energies_batch = evaluate_energy(fxs, model, hamiltonian, current_amps, verbose=False)
                t22 = MPI.Wtime()
                grads_vec_batch, amps_batch = get_grads_func(fxs, model)
                t33 = MPI.Wtime()

                sample_time += t11 - t00
                local_energy_time += t22 - t11
                grad_time += t33 - t22

                # Offload into pre-allocated buffers
                b = fxs.shape[0]
                e_np = local_energies_batch.detach().cpu().numpy().ravel()
                g_np = grads_vec_batch.detach().cpu().numpy()
                a_np = amps_batch.detach().cpu().numpy().ravel()

                # Lazy allocation on first batch (now we know n_params)
                if res_energies is None:
                    n_params = g_np.shape[-1]
                    res_energies = np.empty(max_n_local, dtype=np.float64)
                    res_grads = np.empty((max_n_local, n_params), dtype=np.float64)
                    res_amps = np.empty(max_n_local, dtype=np.float64)

                # Grow buffer if needed (rare: worker got more batches than expected)
                if n_local + b > res_energies.shape[0]:
                    new_cap = max(res_energies.shape[0] * 2, n_local + b)
                    res_energies = np.resize(res_energies, new_cap)
                    res_grads.resize((new_cap, res_grads.shape[1]), refcheck=False)
                    res_amps = np.resize(res_amps, new_cap)

                res_energies[n_local:n_local + b] = e_np
                res_grads[n_local:n_local + b] = g_np
                res_amps[n_local:n_local + b] = a_np

                last_finished_batch = b
                n_local += b

                del local_energies_batch, grads_vec_batch, amps_batch, e_np, g_np, a_np
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"!!! Rank {rank} CRASHED with FATAL ERROR !!!\n{error_msg}", flush=True)
            sys.stdout.flush()
            comm.Abort(1)

    # Rest the CPU until all ranks finish
    gentle_barrier(comm, sleep_interval=0.01)

    # Trim buffers to actual size (views, no copy)
    if n_local > 0:
        res_energies = res_energies[:n_local]
        res_grads = res_grads[:n_local]
        res_amps = res_amps[:n_local]
    else:
        res_energies = np.array([], dtype=np.float64)
        res_grads = np.array([], dtype=np.float64)
        res_amps = np.array([], dtype=np.float64)

    stats = {
        'n_local': n_local,
        't_sample': sample_time,
        't_energy': local_energy_time,
        't_grad': grad_time
    }
    t1 = MPI.Wtime()
    total_sample_time = t1 - t0
    return (res_energies, res_grads, res_amps), fxs, stats, total_sample_time

def run_sampling_phase_reuse_v0(
    svmc: int,
    Ns: int,
    B: int,
    fxs: torch.Tensor,
    model: torch.nn.Module,
    hamiltonian,
    graph,
    get_grads_func: callable,
    comm,
    rank: int,
    size: int,
    should_burn_in=False,
    burn_in_steps=10,
    sampling_hopping_rate=0.25,
    verbose=False
):
    """
    [BACKUP] Original reuse sampling phase that stores grads and amps separately.
    Replaced by run_sampling_phase_reuse which computes O_loc inline.

    Returns:
        local_data: (energies, grads, amps), numpy arrays
        fxs: initial configs in a batch
        stats: dict, containing timing statistics and sample counts
    """
    # Define Tags
    TAG_REQ = 100
    TAG_CMD = 101
    CMD_CONTINUE = 0
    CMD_STOP = 1

    sample_time = 0.0
    local_energy_time = 0.0
    grad_time = 0.0

    # Pre-allocate buffers: upper bound on samples this worker could collect
    max_n_local = int(np.ceil(Ns / max(size - 1, 1) / B)) * B
    res_energies = None  # lazily allocated after first batch (to learn n_params)
    res_grads = None
    res_amps = None
    n_local = 0

    t0 = MPI.Wtime()

    if should_burn_in:
        if rank != 0:
            current_step = 0
            while current_step < burn_in_steps:
                fxs, _ = sample_next(fxs, model, graph, hopping_rate=sampling_hopping_rate, verbose=False)
                current_step += 1
        else:
            pass

    gentle_barrier(comm, sleep_interval=0.01) # ensure all ranks finish burn-in before master starts dispatching tasks

    # --- Branch A: Master (Rank 0) ---
    if rank == 0:
        pbar = tqdm(total=Ns, desc=f"Step {svmc} Sampling", unit="samples")
        n_collected = 0
        n_dispatched = 0
        active_workers = size - 1
        active_rank_ids = set(range(1, size))

        while active_workers > 0:
            has_message = comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_REQ)
            if has_message:
                status = MPI.Status()
                buf = np.empty(1, dtype=np.int32)
                comm.Recv([buf, MPI.INT], source=MPI.ANY_SOURCE, tag=TAG_REQ, status=status)
                source_rank = status.Get_source()
                finished_batch = buf[0]

                if finished_batch > 0:
                    n_collected += finished_batch
                    pbar.update(finished_batch)

                next_batch = B
                if n_dispatched < Ns:
                    cmd = np.array([CMD_CONTINUE], dtype=np.int32)
                    comm.Send([cmd, MPI.INT], dest=source_rank, tag=TAG_CMD)
                    n_dispatched += next_batch
                    if verbose:
                        print(f"[Master] Dispatched {next_batch} samples to Rank {source_rank}. Total dispatched: {n_dispatched}.", flush=True)
                else:
                    cmd = np.array([CMD_STOP], dtype=np.int32)
                    comm.Send([cmd, MPI.INT], dest=source_rank, tag=TAG_CMD)
                    active_workers -= 1
                    if source_rank in active_rank_ids:
                        active_rank_ids.remove(source_rank)
                        if verbose:
                            print(f'[Master] Kill rank {source_rank} with {finished_batch} samples. Remaining num of active workers: {len(active_rank_ids)}, {active_workers}', flush=True)

            else:
                if n_collected >= Ns:
                    if verbose:
                        print(f"\n[Master] Waiting for {active_workers} workers to finish burn-in.", flush=True)
                        time.sleep(0.1) # yield CPU to let stragglers finish burn-in
                time.sleep(0.001)

        if verbose:
            print('Sampling phase should be done now.', flush=True)

        pbar.close()

        if len(active_rank_ids) > 0:
            print(f"ERROR: Finishing with {len(active_rank_ids)} dead ranks.", flush=True)
            comm.Abort(1)

    # --- Branch B: Worker ---
    else:
        try:
            last_finished_batch = 0
            while True:
                # 1. Request / Report
                buf = np.array([last_finished_batch], dtype=np.int32)
                comm.Send([buf, MPI.INT], dest=0, tag=TAG_REQ)

                # 2. Wait Command
                cmd = np.empty(1, dtype=np.int32)
                comm.Recv([cmd, MPI.INT], source=0, tag=TAG_CMD)

                if cmd[0] == CMD_STOP:
                    break

                # 3. Compute
                # try:
                t00 = MPI.Wtime()
                fxs, current_amps = sample_next_reuse(fxs, model, graph, hopping_rate=sampling_hopping_rate, verbose=False)
                t11 = MPI.Wtime()
                energy_batch, local_energies_batch = evaluate_energy_reuse(fxs, model, hamiltonian, current_amps, verbose=False)
                t22 = MPI.Wtime()
                grads_vec_batch, amps_batch = get_grads_func(fxs, model)
                t33 = MPI.Wtime()

                sample_time += t11 - t00
                local_energy_time += t22 - t11
                grad_time += t33 - t22

                # Offload into pre-allocated buffers
                b = fxs.shape[0]
                e_np = local_energies_batch.detach().cpu().numpy().ravel()
                g_np = grads_vec_batch.detach().cpu().numpy()
                a_np = amps_batch.detach().cpu().numpy().ravel()

                # Lazy allocation on first batch (now we know n_params)
                if res_energies is None:
                    n_params = g_np.shape[-1]
                    res_energies = np.empty(max_n_local, dtype=np.float64)
                    res_grads = np.empty((max_n_local, n_params), dtype=np.float64)
                    res_amps = np.empty(max_n_local, dtype=np.float64)

                # Grow buffer if needed (rare: worker got more batches than expected)
                if n_local + b > res_energies.shape[0]:
                    new_cap = max(res_energies.shape[0] * 2, n_local + b)
                    res_energies = np.resize(res_energies, new_cap)
                    res_grads.resize((new_cap, res_grads.shape[1]), refcheck=False)
                    res_amps = np.resize(res_amps, new_cap)

                res_energies[n_local:n_local + b] = e_np
                res_grads[n_local:n_local + b] = g_np
                res_amps[n_local:n_local + b] = a_np

                last_finished_batch = b
                n_local += b

                del local_energies_batch, grads_vec_batch, amps_batch, e_np, g_np, a_np
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"!!! Rank {rank} CRASHED with FATAL ERROR !!!\n{error_msg}", flush=True)
            sys.stdout.flush()
            comm.Abort(1)

    # Rest the CPU until all ranks finish
    gentle_barrier(comm, sleep_interval=0.01)

    # Trim buffers to actual size (views, no copy)
    if n_local > 0:
        res_energies = res_energies[:n_local]
        res_grads = res_grads[:n_local]
        res_amps = res_amps[:n_local]
    else:
        res_energies = np.array([], dtype=np.float64)
        res_grads = np.array([], dtype=np.float64)
        res_amps = np.array([], dtype=np.float64)

    stats = {
        'n_local': n_local,
        't_sample': sample_time,
        't_energy': local_energy_time,
        't_grad': grad_time
    }
    t1 = MPI.Wtime()
    total_sample_time = t1 - t0
    return (res_energies, res_grads, res_amps), fxs, stats, total_sample_time


def run_sampling_phase_reuse(
    svmc: int,
    Ns: int,
    B: int,
    fxs: torch.Tensor,
    model: torch.nn.Module,
    hamiltonian,
    graph,
    get_grads_func: callable,
    comm,
    rank: int,
    size: int,
    should_burn_in=False,
    burn_in_steps=10,
    sampling_hopping_rate=0.25,
    verbose=False,
):
    """
    Memory-optimized reuse sampling phase.

    Computes O_loc = grad/amp inline per batch, eliminating
    separate res_grads and res_amps buffers.

    Returns:
        local_data: (energies, O_loc), numpy arrays
        fxs: current config batch
        stats: timing dict
        total_sample_time: float
    """
    TAG_REQ = 100
    TAG_CMD = 101
    CMD_CONTINUE = 0
    CMD_STOP = 1

    sample_time = 0.0
    local_energy_time = 0.0
    grad_time = 0.0

    max_n_local = int(
        np.ceil(Ns / max(size - 1, 1) / B)
    ) * B
    res_energies = None
    res_O = None
    n_local = 0

    t0 = MPI.Wtime()

    if should_burn_in:
        if rank != 0:
            current_step = 0
            while current_step < burn_in_steps:
                fxs, _ = sample_next(
                    fxs, model, graph,
                    hopping_rate=sampling_hopping_rate,
                    verbose=False,
                )
                current_step += 1

    gentle_barrier(comm, sleep_interval=0.01)

    # --- Branch A: Master (Rank 0) ---
    if rank == 0:
        pbar = tqdm(
            total=Ns,
            desc=f"Step {svmc} Sampling",
            unit="samples",
        )
        n_collected = 0
        n_dispatched = 0
        active_workers = size - 1
        active_rank_ids = set(range(1, size))

        while active_workers > 0:
            has_message = comm.Iprobe(
                source=MPI.ANY_SOURCE, tag=TAG_REQ
            )
            if has_message:
                status = MPI.Status()
                buf = np.empty(1, dtype=np.int32)
                comm.Recv(
                    [buf, MPI.INT],
                    source=MPI.ANY_SOURCE,
                    tag=TAG_REQ,
                    status=status,
                )
                source_rank = status.Get_source()
                finished_batch = buf[0]

                if finished_batch > 0:
                    n_collected += finished_batch
                    pbar.update(finished_batch)

                if n_dispatched < Ns:
                    cmd = np.array(
                        [CMD_CONTINUE], dtype=np.int32
                    )
                    comm.Send(
                        [cmd, MPI.INT],
                        dest=source_rank,
                        tag=TAG_CMD,
                    )
                    n_dispatched += B
                    if verbose:
                        print(
                            f"[Master] Dispatched {B} samples "
                            f"to Rank {source_rank}. "
                            f"Total: {n_dispatched}.",
                            flush=True,
                        )
                else:
                    cmd = np.array(
                        [CMD_STOP], dtype=np.int32
                    )
                    comm.Send(
                        [cmd, MPI.INT],
                        dest=source_rank,
                        tag=TAG_CMD,
                    )
                    active_workers -= 1
                    if source_rank in active_rank_ids:
                        active_rank_ids.remove(source_rank)
                        if verbose:
                            print(
                                f'[Master] Kill rank '
                                f'{source_rank}. Remaining: '
                                f'{active_workers}',
                                flush=True,
                            )
            else:
                time.sleep(0.001)

        if verbose:
            print(
                'Sampling phase should be done now.',
                flush=True,
            )

        pbar.close()

        if len(active_rank_ids) > 0:
            print(
                f"ERROR: {len(active_rank_ids)} dead ranks.",
                flush=True,
            )
            comm.Abort(1)

    # --- Branch B: Worker ---
    else:
        try:
            last_finished_batch = 0
            while True:
                buf = np.array(
                    [last_finished_batch], dtype=np.int32
                )
                comm.Send(
                    [buf, MPI.INT], dest=0, tag=TAG_REQ
                )

                cmd = np.empty(1, dtype=np.int32)
                comm.Recv(
                    [cmd, MPI.INT], source=0, tag=TAG_CMD
                )

                if cmd[0] == CMD_STOP:
                    break

                t00 = MPI.Wtime()
                fxs, current_amps = sample_next_reuse(
                    fxs, model, graph,
                    hopping_rate=sampling_hopping_rate,
                    verbose=False,
                )
                t11 = MPI.Wtime()
                energy_batch, local_energies_batch = (
                    evaluate_energy_reuse(
                        fxs, model, hamiltonian,
                        current_amps, verbose=False,
                    )
                )
                t22 = MPI.Wtime()
                grads_vec_batch, amps_batch = get_grads_func(
                    fxs, model
                )
                t33 = MPI.Wtime()

                sample_time += t11 - t00
                local_energy_time += t22 - t11
                grad_time += t33 - t22

                b = fxs.shape[0]
                e_np = (
                    local_energies_batch.detach()
                    .cpu().numpy().ravel()
                )
                # Compute O_loc = grad/amp inline
                g_np = grads_vec_batch.detach().cpu().numpy()
                a_np = (
                    amps_batch.detach()
                    .cpu().numpy().ravel()
                )
                g_np /= a_np.reshape(-1, 1)  # in-place

                if res_energies is None:
                    n_params = g_np.shape[-1]
                    res_energies = np.empty(
                        max_n_local, dtype=np.float64
                    )
                    res_O = np.empty(
                        (max_n_local, n_params),
                        dtype=np.float64,
                    )

                if n_local + b > res_energies.shape[0]:
                    new_cap = max(
                        res_energies.shape[0] * 2, n_local + b
                    )
                    res_energies = np.resize(
                        res_energies, new_cap
                    )
                    res_O.resize(
                        (new_cap, res_O.shape[1]),
                        refcheck=False,
                    )

                res_energies[n_local:n_local + b] = e_np
                res_O[n_local:n_local + b] = g_np

                last_finished_batch = b
                n_local += b

                del (
                    local_energies_batch,
                    grads_vec_batch,
                    amps_batch,
                    e_np,
                    g_np,
                    a_np,
                )
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(
                f"!!! Rank {rank} CRASHED !!!\n{error_msg}",
                flush=True,
            )
            sys.stdout.flush()
            comm.Abort(1)

    gentle_barrier(comm, sleep_interval=0.01)

    if n_local > 0:
        res_energies = res_energies[:n_local]
        res_O = res_O[:n_local]
    else:
        res_energies = np.array([], dtype=np.float64)
        res_O = np.empty((0, 0), dtype=np.float64)

    stats = {
        'n_local': n_local,
        't_sample': sample_time,
        't_energy': local_energy_time,
        't_grad': grad_time,
    }
    t1 = MPI.Wtime()
    total_sample_time = t1 - t0
    return (res_energies, res_O), fxs, stats, total_sample_time


def run_sampling_phase(
    svmc: int,
    Ns: int,
    B: int,
    fxs: torch.Tensor,
    model: torch.nn.Module,
    hamiltonian,
    graph,
    get_grads_func: callable,
    comm,
    rank: int,
    size: int,
    should_burn_in=False,
    burn_in_steps=10,
    sampling_hopping_rate=0.25,
    verbose=False,
):
    """
    Memory-optimized sampling phase.

    Trick #2: computes O_loc = grad/amp inline per batch and
    stores it directly, eliminating the separate res_grads and
    res_amps buffers. Returns (energies, O_loc) instead of
    (energies, grads, amps).

    Returns:
        local_data: (energies, O_loc), numpy arrays
        fxs: current config batch
        stats: timing dict
        total_sample_time: float
    """
    TAG_REQ = 100
    TAG_CMD = 101
    CMD_CONTINUE = 0
    CMD_STOP = 1

    sample_time = 0.0
    local_energy_time = 0.0
    grad_time = 0.0

    max_n_local = int(
        np.ceil(Ns / max(size - 1, 1) / B)
    ) * B
    res_energies = None
    res_O = None  # stores O_loc directly (no separate grads/amps)
    n_local = 0

    t0 = MPI.Wtime()

    if should_burn_in:
        if rank != 0:
            current_step = 0
            while current_step < burn_in_steps:
                fxs, _ = sample_next(
                    fxs, model, graph,
                    hopping_rate=sampling_hopping_rate,
                    verbose=False,
                )
                current_step += 1

    gentle_barrier(comm, sleep_interval=0.01)

    # --- Branch A: Master (Rank 0) ---
    if rank == 0:
        pbar = tqdm(
            total=Ns,
            desc=f"Step {svmc} Sampling",
            unit="samples",
        )
        n_collected = 0
        n_dispatched = 0
        active_workers = size - 1
        active_rank_ids = set(range(1, size))

        while active_workers > 0:
            has_message = comm.Iprobe(
                source=MPI.ANY_SOURCE, tag=TAG_REQ
            )
            if has_message:
                status = MPI.Status()
                buf = np.empty(1, dtype=np.int32)
                comm.Recv(
                    [buf, MPI.INT],
                    source=MPI.ANY_SOURCE,
                    tag=TAG_REQ,
                    status=status,
                )
                source_rank = status.Get_source()
                finished_batch = buf[0]

                if finished_batch > 0:
                    n_collected += finished_batch
                    pbar.update(finished_batch)

                if n_dispatched < Ns:
                    cmd = np.array(
                        [CMD_CONTINUE], dtype=np.int32
                    )
                    comm.Send(
                        [cmd, MPI.INT],
                        dest=source_rank,
                        tag=TAG_CMD,
                    )
                    n_dispatched += B
                    if verbose:
                        print(
                            f"[Master] Dispatched {B} samples "
                            f"to Rank {source_rank}. "
                            f"Total: {n_dispatched}.",
                            flush=True,
                        )
                else:
                    cmd = np.array(
                        [CMD_STOP], dtype=np.int32
                    )
                    comm.Send(
                        [cmd, MPI.INT],
                        dest=source_rank,
                        tag=TAG_CMD,
                    )
                    active_workers -= 1
                    if source_rank in active_rank_ids:
                        active_rank_ids.remove(source_rank)
                        if verbose:
                            print(
                                f'[Master] Kill rank '
                                f'{source_rank}. Remaining: '
                                f'{active_workers}',
                                flush=True,
                            )
            else:
                time.sleep(0.001)

        pbar.close()

        if len(active_rank_ids) > 0:
            print(
                f"ERROR: {len(active_rank_ids)} dead ranks.",
                flush=True,
            )
            comm.Abort(1)

    # --- Branch B: Worker ---
    else:
        try:
            last_finished_batch = 0
            while True:
                buf = np.array(
                    [last_finished_batch], dtype=np.int32
                )
                comm.Send(
                    [buf, MPI.INT], dest=0, tag=TAG_REQ
                )

                cmd = np.empty(1, dtype=np.int32)
                comm.Recv(
                    [cmd, MPI.INT], source=0, tag=TAG_CMD
                )

                if cmd[0] == CMD_STOP:
                    break

                t00 = MPI.Wtime()
                fxs, current_amps = sample_next(
                    fxs, model, graph,
                    hopping_rate=sampling_hopping_rate,
                    verbose=False,
                )
                t11 = MPI.Wtime()
                energy_batch, local_energies_batch = (
                    evaluate_energy(
                        fxs, model, hamiltonian,
                        current_amps, verbose=False,
                    )
                )
                t22 = MPI.Wtime()
                grads_vec_batch, amps_batch = get_grads_func(
                    fxs, model
                )
                t33 = MPI.Wtime()

                sample_time += t11 - t00
                local_energy_time += t22 - t11
                grad_time += t33 - t22

                b = fxs.shape[0]
                e_np = (
                    local_energies_batch.detach()
                    .cpu().numpy().ravel()
                )
                # Compute O_loc = grad/amp inline (trick #2)
                g_np = grads_vec_batch.detach().cpu().numpy()
                a_np = (
                    amps_batch.detach()
                    .cpu().numpy().ravel()
                )
                g_np /= a_np.reshape(-1, 1)  # in-place -> O_loc

                if res_energies is None:
                    n_params = g_np.shape[-1]
                    res_energies = np.empty(
                        max_n_local, dtype=np.float64
                    )
                    res_O = np.empty(
                        (max_n_local, n_params),
                        dtype=np.float64,
                    )

                if n_local + b > res_energies.shape[0]:
                    new_cap = max(
                        res_energies.shape[0] * 2, n_local + b
                    )
                    res_energies = np.resize(
                        res_energies, new_cap
                    )
                    res_O.resize(
                        (new_cap, res_O.shape[1]),
                        refcheck=False,
                    )

                res_energies[n_local:n_local + b] = e_np
                res_O[n_local:n_local + b] = g_np  # O_loc

                last_finished_batch = b
                n_local += b

                del (
                    local_energies_batch,
                    grads_vec_batch,
                    amps_batch,
                    e_np,
                    g_np,
                    a_np,
                )
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(
                f"!!! Rank {rank} CRASHED !!!\n{error_msg}",
                flush=True,
            )
            sys.stdout.flush()
            comm.Abort(1)

    gentle_barrier(comm, sleep_interval=0.01)

    if n_local > 0:
        res_energies = res_energies[:n_local]
        res_O = res_O[:n_local]
    else:
        res_energies = np.array([], dtype=np.float64)
        res_O = np.empty((0, 0), dtype=np.float64)

    stats = {
        'n_local': n_local,
        't_sample': sample_time,
        't_energy': local_energy_time,
        't_grad': grad_time,
    }
    t1 = MPI.Wtime()
    total_sample_time = t1 - t0
    return (res_energies, res_O), fxs, stats, total_sample_time


def run_sampling_phase_vec_v0(
    svmc,
    Ns,
    B,
    fxs,
    model,
    hamiltonian,
    graph,
    get_grads_func,
    comm,
    rank,
    size,
    should_burn_in=False,
    burn_in_steps=10,
):
    """
    [BACKUP] Original vec sampling phase that stores grads and amps separately.
    Replaced by run_sampling_phase_vec which computes O_loc inline.
    WARNING: Unchecked yet
    """
    # 定义 Tag
    TAG_REQ = 100
    TAG_CMD = 101
    CMD_CONTINUE = 0
    CMD_STOP = 1

    sample_time = 0.0
    local_energy_time = 0.0
    grad_time = 0.0

    E_loc_vec = []
    amps_vec = []
    grads_vec_list = []
    n_local = 0

    t0 = MPI.Wtime()

    # --- Branch A: Master (Rank 0) ---
    if rank == 0:
        pbar = tqdm(total=Ns, desc=f"Step {svmc} Sampling", unit="samples")
        n_collected = 0
        n_dispatched = 0
        active_workers = size - 1

        while active_workers > 0:
            status = MPI.Status()
            buf = np.empty(1, dtype=np.int32)
            comm.Recv([buf, MPI.INT], source=MPI.ANY_SOURCE, tag=TAG_REQ, status=status)
            source_rank = status.Get_source()
            finished_batch = buf[0]

            if finished_batch > 0:
                n_collected += finished_batch
                pbar.update(finished_batch)
            elif finished_batch < 0: # worker failed
                n_dispatched -= finished_batch # revert dispatch count
                print(f"Master detected failure from Rank {source_rank}, reverting dispatched count by {-finished_batch}.")

            next_batch = B
            if n_dispatched < Ns:
                cmd = np.array([CMD_CONTINUE], dtype=np.int32)
                comm.Send([cmd, MPI.INT], dest=source_rank, tag=TAG_CMD)
                n_dispatched += next_batch
            else:
                cmd = np.array([CMD_STOP], dtype=np.int32)
                comm.Send([cmd, MPI.INT], dest=source_rank, tag=TAG_CMD)
                active_workers -= 1
        pbar.close()

    # --- Branch B: Worker ---
    else:
        if should_burn_in:
            current_step = 0
            max_resets = 3  # to prevent infinite loops
            reset_count = 0

            while current_step < burn_in_steps:
                try:
                    fxs, _ = sample_next_vec(fxs, model, graph, verbose=False)
                    current_step += 1

                except RuntimeError as e:
                    error_str = str(e).lower()
                    if 'svd' not in error_str and 'converge' not in error_str:
                        raise e # raise non-svd errors immediately

                    # === Jitter SVD ===
                    success_jitter = False
                    try:
                        with use_jitter_svd():
                            print(f"Global SVD Jitter Applied to Rank {comm.Get_rank()}")
                            fxs, _ = sample_next_vec(fxs, model, graph, verbose=False)
                        current_step += 1
                        success_jitter = True
                    except RuntimeError:
                        pass
                    if not success_jitter:
                        if reset_count < max_resets:
                            print(f"Rank {rank}: SVD failed hard. Permuting configs and RESTARTING burn-in.")
                            with torch.no_grad():
                                for i in range(fxs.shape[0]):
                                    perm = torch.randperm(fxs.shape[1])
                                    fxs[i] = fxs[i][perm]
                            current_step = 0
                            reset_count += 1
                        else:
                            raise RuntimeError(f"Rank {rank} failed burn-in too many times.")


        last_finished_batch = 0
        while True:
            buf = np.array([last_finished_batch], dtype=np.int32)
            comm.Send([buf, MPI.INT], dest=0, tag=TAG_REQ)

            cmd = np.empty(1, dtype=np.int32)
            comm.Recv([cmd, MPI.INT], source=0, tag=TAG_CMD)

            if cmd[0] == CMD_STOP:
                break

            t00 = MPI.Wtime()
            fxs, current_amps = sample_next_vec(fxs, model, graph, verbose=False)
            t11 = MPI.Wtime()
            energy_batch, local_energies_batch = evaluate_energy_vec(fxs, model, hamiltonian, current_amps, verbose=False)
            t22 = MPI.Wtime()
            grads_vec_batch, amps_batch = get_grads_func(fxs, model)
            t33 = MPI.Wtime()

            sample_time += t11 - t00
            local_energy_time += t22 - t11
            grad_time += t33 - t22

            E_loc_vec.append(local_energies_batch.detach().cpu().numpy())
            amps_vec.append(amps_batch.detach().cpu().numpy())
            grads_vec_list.append(grads_vec_batch.detach().cpu().numpy())

            last_finished_batch = fxs.shape[0]
            n_local += last_finished_batch

            del local_energies_batch, grads_vec_batch, amps_batch

    comm.Barrier()

    if n_local > 0:
        res_energies = np.concatenate(E_loc_vec)
        res_grads = np.concatenate(grads_vec_list)
        res_amps = np.concatenate(amps_vec)
    else:
        res_energies = np.array([], dtype=np.float64)
        res_grads = np.array([], dtype=np.float64)
        res_amps = np.array([], dtype=np.float64)

    stats = {
        'n_local': n_local,
        't_sample': sample_time,
        't_energy': local_energy_time,
        't_grad': grad_time
    }
    t1 = MPI.Wtime()
    total_sample_time = t1 - t0
    return (res_energies, res_grads, res_amps), fxs, stats, total_sample_time


def run_sampling_phase_vec(
    svmc,
    Ns,
    B,
    fxs,
    model,
    hamiltonian,
    graph,
    get_grads_func,
    comm,
    rank,
    size,
    should_burn_in=False,
    burn_in_steps=10,
    sampling_hopping_rate=0.25,
    verbose=False,
):
    """
    Memory-optimized vec sampling phase.

    Computes O_loc = grad/amp inline per batch.
    WARNING: Unchecked yet (same as original vec variant).

    Returns:
        local_data: (energies, O_loc), numpy arrays
        fxs: current config batch
        stats: timing dict
        total_sample_time: float
    """
    TAG_REQ = 100
    TAG_CMD = 101
    CMD_CONTINUE = 0
    CMD_STOP = 1

    sample_time = 0.0
    local_energy_time = 0.0
    grad_time = 0.0

    E_loc_vec = []
    O_loc_vec = []
    n_local = 0

    t0 = MPI.Wtime()

    # --- Branch A: Master (Rank 0) ---
    if rank == 0:
        pbar = tqdm(
            total=Ns,
            desc=f"Step {svmc} Sampling",
            unit="samples",
        )
        n_collected = 0
        n_dispatched = 0
        active_workers = size - 1

        while active_workers > 0:
            status = MPI.Status()
            buf = np.empty(1, dtype=np.int32)
            comm.Recv(
                [buf, MPI.INT],
                source=MPI.ANY_SOURCE,
                tag=TAG_REQ,
                status=status,
            )
            source_rank = status.Get_source()
            finished_batch = buf[0]

            if finished_batch > 0:
                n_collected += finished_batch
                pbar.update(finished_batch)
            elif finished_batch < 0:
                n_dispatched -= finished_batch
                print(
                    f"Master detected failure from "
                    f"Rank {source_rank}.",
                )

            if n_dispatched < Ns:
                cmd = np.array(
                    [CMD_CONTINUE], dtype=np.int32
                )
                comm.Send(
                    [cmd, MPI.INT],
                    dest=source_rank,
                    tag=TAG_CMD,
                )
                n_dispatched += B
            else:
                cmd = np.array(
                    [CMD_STOP], dtype=np.int32
                )
                comm.Send(
                    [cmd, MPI.INT],
                    dest=source_rank,
                    tag=TAG_CMD,
                )
                active_workers -= 1
        pbar.close()

    # --- Branch B: Worker ---
    else:
        if should_burn_in:
            current_step = 0
            max_resets = 3
            reset_count = 0

            while current_step < burn_in_steps:
                try:
                    fxs, _ = sample_next_vec(
                        fxs, model, graph,
                        hopping_rate=sampling_hopping_rate,
                        verbose=False,
                    )
                    current_step += 1
                except RuntimeError as e:
                    error_str = str(e).lower()
                    if (
                        'svd' not in error_str
                        and 'converge' not in error_str
                    ):
                        raise e

                    success_jitter = False
                    try:
                        with use_jitter_svd():
                            print(
                                f"Global SVD Jitter Applied "
                                f"to Rank {comm.Get_rank()}"
                            )
                            fxs, _ = sample_next_vec(
                                fxs, model, graph,
                                hopping_rate=sampling_hopping_rate,
                                verbose=False,
                            )
                        current_step += 1
                        success_jitter = True
                    except RuntimeError:
                        pass
                    if not success_jitter:
                        if reset_count < max_resets:
                            print(
                                f"Rank {rank}: SVD failed. "
                                f"Permuting & restarting."
                            )
                            with torch.no_grad():
                                for i in range(fxs.shape[0]):
                                    perm = torch.randperm(
                                        fxs.shape[1]
                                    )
                                    fxs[i] = fxs[i][perm]
                            current_step = 0
                            reset_count += 1
                        else:
                            raise RuntimeError(
                                f"Rank {rank} failed burn-in "
                                f"too many times."
                            )

        last_finished_batch = 0
        while True:
            buf = np.array(
                [last_finished_batch], dtype=np.int32
            )
            comm.Send(
                [buf, MPI.INT], dest=0, tag=TAG_REQ
            )

            cmd = np.empty(1, dtype=np.int32)
            comm.Recv(
                [cmd, MPI.INT], source=0, tag=TAG_CMD
            )

            if cmd[0] == CMD_STOP:
                break

            t00 = MPI.Wtime()
            fxs, current_amps = sample_next_vec(
                fxs, model, graph,
                hopping_rate=sampling_hopping_rate,
                verbose=verbose,
            )
            t11 = MPI.Wtime()
            energy_batch, local_energies_batch = (
                evaluate_energy_vec(
                    fxs, model, hamiltonian,
                    current_amps, verbose=verbose,
                )
            )
            t22 = MPI.Wtime()
            grads_vec_batch, amps_batch = get_grads_func(
                fxs, model
            )
            t33 = MPI.Wtime()

            sample_time += t11 - t00
            local_energy_time += t22 - t11
            grad_time += t33 - t22

            if verbose:
                print(
                    f' Rank {rank} | T_samp={t11-t00:.2f} '
                    f'T_E={t22-t11:.2f} T_G={t33-t22:.2f}',
                    flush=True,
                )

            # Compute O_loc = grad/amp inline
            g_np = grads_vec_batch.detach().cpu().numpy()
            a_np = (
                amps_batch.detach().cpu().numpy().ravel()
            )
            g_np /= a_np.reshape(-1, 1)  # in-place

            E_loc_vec.append(
                local_energies_batch.detach()
                .cpu().numpy()
            )
            O_loc_vec.append(g_np)

            last_finished_batch = fxs.shape[0]
            n_local += last_finished_batch

            del (
                local_energies_batch,
                grads_vec_batch,
                amps_batch,
            )

    comm.Barrier()

    if n_local > 0:
        res_energies = np.concatenate(E_loc_vec)
        res_O = np.concatenate(O_loc_vec)
    else:
        res_energies = np.array([], dtype=np.float64)
        res_O = np.empty((0, 0), dtype=np.float64)

    stats = {
        'n_local': n_local,
        't_sample': sample_time,
        't_energy': local_energy_time,
        't_grad': grad_time,
    }
    t1 = MPI.Wtime()
    total_sample_time = t1 - t0
    return (res_energies, res_O), fxs, stats, total_sample_time