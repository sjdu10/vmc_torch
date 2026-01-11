import numpy as np
import torch
from tqdm import tqdm
from vmc_torch.experiment.vmap.vmap_utils import sample_next, evaluate_energy
import scipy.sparse.linalg as spla
from mpi4py import MPI

def distributed_minres_solver(
    local_grads, 
    local_amps, 
    local_energies, 
    energy_mean, 
    total_samples, 
    n_params, 
    diag_shift, 
    comm
):
    """
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
    dp, info = spla.minres(A, energy_grad, rtol=1e-4, maxiter=100)
    
    t1 = MPI.Wtime()
    return dp, t1 - t0


def run_sampling_phase(
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
    运行 Dedicated Master 模式的采样循环
    
    Returns:
        local_data: (energies, grads, amps) 的 numpy 数组
        fxs: 更新后的构型 (用于下一次热启动)
        stats: dict, 包含时间统计和样本数
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
            for _ in range(burn_in_steps):
                fxs, _ = sample_next(fxs, model, graph, verbose=False)

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
            t00 = MPI.Wtime()
            fxs, current_amps = sample_next(fxs, model, graph, verbose=False)
            t11 = MPI.Wtime()
            
            energy_batch, local_energies_batch = evaluate_energy(fxs, model, hamiltonian, current_amps, verbose=False)
            t22 = MPI.Wtime()
            
            grads_vec_batch, amps_batch = get_grads_func(fxs, model)
            t33 = MPI.Wtime()

            sample_time += t11 - t00
            local_energy_time += t22 - t11
            grad_time += t33 - t22

            # Offload
            E_loc_vec.append(local_energies_batch.detach().cpu().numpy())
            amps_vec.append(amps_batch.detach().cpu().numpy())
            grads_vec_list.append(grads_vec_batch.detach().cpu().numpy())
            
            last_finished_batch = fxs.shape[0]
            n_local += last_finished_batch
            
            del local_energies_batch, grads_vec_batch, amps_batch
    
    comm.Barrier()
    
    # Pack Results
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