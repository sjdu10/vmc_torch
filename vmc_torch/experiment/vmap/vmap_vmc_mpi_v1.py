import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ["OMP_NUM_THREADS"] = '1'
from mpi4py import MPI
import numpy as np
import symmray as sr
import quimb.tensor as qtn
import pickle
from autoray import do
from functools import partial
import torch
import json
import time
from tqdm import tqdm
from vmap_utils import sample_next, evaluate_energy, compute_grads, random_initial_config, compute_grads_decoupled
from vmap_utils import NN_fPEPS_Model, fPEPS_Model, Transformer_fPEPS_Model, Transformer_fPEPS_Model_batchedAttn
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.experiment.tn_model import init_weights_to_zero

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

# torch.set_default_device("cuda:0") # GPU
torch.set_default_device("cpu") # CPU
torch.random.manual_seed(42 + RANK)

Lx = 4
Ly = 2
N_f = Lx * Ly - 2 # filling
D = 4
chi = -2
seed = RANK + 42
# only the flat backend is compatible with jax.jit
flat = True
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/data'
u1z2 = True
appendix = '_U1SU' if u1z2 else ''
params = pickle.load(open(pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/peps_su_params{appendix}.pkl', 'rb'))
skeleton = pickle.load(open(pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/peps_skeleton{appendix}.pkl', 'rb'))
peps = qtn.unpack(params, skeleton)
nsites = peps.nsites
for ts in peps.tensors:
    # print(ts.data)
    ts.modify(data=ts.data.to_flat()*10)
for site in peps.sites:
    peps[site].data._label = site
    peps[site].data.indices[-1]._linearmap = ((0, 0), (1, 0), (1, 1), (0, 1)) # Important for U1->Z2 fPEPS

# Select fPEPS-based model

# fpeps_model = NN_fPEPS_Model(
#     tn=peps,
#     max_bond=chi,
#     nn_hidden_dim=D,
#     nn_eta=1,
#     dtype=torch.float64,
# )

fpeps_model = Transformer_fPEPS_Model_batchedAttn(
    tn=peps,
    max_bond=chi,
    embed_dim=16,
    attn_heads=4,
    attn_depth=1,
    nn_hidden_dim=4*peps.nsites,
    nn_eta=1,
    init_perturbation_scale=1e-3,
    dtype=torch.float64,
)

# fpeps_model = fPEPS_Model(
#     peps, max_bond=chi, dtype=torch.float64
# )

n_params = sum(p.numel() for p in fpeps_model.parameters())
if RANK == 0:
    # print model size
    print(f'fPEPS-based model number of parameters: {n_params}')

# Hamiltonian definition
t=1.0
U=8.0
n_fermions_per_spin = (N_f // 2, N_f // 2)
H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx,
    Ly,
    t,
    U,
    N_f,
    pbc=False,
    n_fermions_per_spin=n_fermions_per_spin,
    no_u1_symmetry=False,
)
graph = H.graph

if Lx*Ly <= 6 and RANK == 0:
    H_dense = torch.tensor(H.to_dense())
    psi_vec = fpeps_model(torch.tensor(H.hilbert.all_states(), dtype=torch.int32))
    energies_exact, states_exact = torch.linalg.eigh(H_dense)
    print(f'Exact ground state energy: {energies_exact[0].item()/nsites}')
    SU_E = (psi_vec.conj().T @ H_dense @ psi_vec) / (psi_vec.conj().T @ psi_vec)
    print(f'SU variational energy: {SU_E.item()/nsites}')

    terms = sr.hamiltonians.ham_fermi_hubbard_from_edges(
        "Z2",
        edges=tuple(peps.gen_bond_coos()),
        U=8,
        mu=0.0,
    )
    terms = {k: v.to_flat() for k, v in terms.items()}
    new_peps = peps.copy()
    new_peps.apply_to_arrays(lambda x: np.array(x))
    E_double = new_peps.compute_local_expectation_exact(terms, normalized=True)
    print(f'Double layer energy: {E_double/nsites}')


# Total sample size
Ns = int(2e4) 
# batchsize per rank
B = 1024
B_grad = 256
# Choose gradient computation method
tn_nn_decouple = False
if tn_nn_decouple:
    get_grads = partial(compute_grads_decoupled, verbose=False, batch_size=B_grad)  # use the decoupled gradient computation
else:
    get_grads = partial(compute_grads, vectorize=True, vmap_grad=True, batch_size=B_grad, verbose=False)  # default to use the vectorized gradient computation
vmc_steps = 200
TAG_OFFSET = 424242
vmc_pbar = tqdm(total=vmc_steps, desc="VMC steps")
minSR=False
diag_shift = 1e-5
learning_rate = 0.1
save_state_every = 10

# initial samples for each rank
fxs = []
for _ in range(B):
    fxs.append(random_initial_config(N_f, nsites))
fxs = torch.stack(fxs)
fxs = fxs.to(torch.long)
# burn-in for each rank
t0 = MPI.Wtime()
for _ in range(1):
    fxs, current_amps = sample_next(fxs, fpeps_model, graph)
t1 = MPI.Wtime()
if RANK == 0:
    print(f'Burn-in sampling time: {t1-t0:.4f} s')

stats_file = pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/vmc_mpi_stats_{fpeps_model._get_name()+str(chi)}.json'
stats = {
    'Np': n_params,
    'sample size': Ns,
    'mean': [],
    'error': [],
    'variance': [],
}

# 定义通信 Tag
TAG_UPDATE = 100
TAG_CMD = 101
CMD_CONTINUE = 0
CMD_STOP = 1

for svmc in range(vmc_steps):
    # 计时器
    sample_time = 0.0
    local_energy_time = 0.0
    grad_time = 0.0
    t0 = MPI.Wtime()
    
    # 数据容器 (List of Numpy Arrays)
    E_loc_vec = []
    amps_vec = []
    grads_vec_list = []
    
    # 本地计数
    n_local = 0
    
    # ==========================================================================
    # Phase 1: Sampling & Gradient Computation
    # ==========================================================================
    
    # --- 分支 A: Master (Rank 0) - 专职调度，不计算 ---
    if RANK == 0:
        pbar = tqdm(total=Ns, desc=f"Step {svmc} Sampling", unit="samples")
        n_global = 0
        active_workers = SIZE - 1  # 除去 Rank 0 自身
        
        # 只要还有 Worker 在干活，Master 就必须在线接电话
        while active_workers > 0:
            # 1. 阻塞接收任意 Worker 的进度报告
            status = MPI.Status()
            buf = np.empty(1, dtype=np.int32)
            COMM.Recv([buf, MPI.INT], source=MPI.ANY_SOURCE, tag=TAG_UPDATE, status=status)
            
            source_rank = status.Get_source()
            worker_batch_size = buf[0]
            
            # 2. 更新全局计数
            n_global += worker_batch_size
            pbar.update(worker_batch_size)
            
            # 3. 决策：是否这就够了？
            if n_global >= Ns:
                # 够了，发送停止指令
                cmd = np.array([CMD_STOP], dtype=np.int32)
                COMM.Send([cmd, MPI.INT], dest=source_rank, tag=TAG_CMD)
                active_workers -= 1
            else:
                # 不够，让该 Worker 继续
                cmd = np.array([CMD_CONTINUE], dtype=np.int32)
                COMM.Send([cmd, MPI.INT], dest=source_rank, tag=TAG_CMD)
        
        pbar.close()

    # --- 分支 B: Worker (Rank > 0) - 专职计算 ---
    else:
        while True:
            # 1. 计算 (Computation)
            t00 = MPI.Wtime()
            fxs, current_amps = sample_next(fxs, fpeps_model, graph, verbose=False)
            t11 = MPI.Wtime()
            # 注意：evaluate_energy 和 get_grads 内部最好也都加上 torch.no_grad() 除非必须
            energy_batch, local_energies_batch = evaluate_energy(fxs, fpeps_model, H, current_amps, verbose=False)
            t22 = MPI.Wtime()
            grads_vec_batch, amps_batch = get_grads(fxs, fpeps_model)
            t33 = MPI.Wtime()

            # 计时累计
            sample_time += t11 - t00
            local_energy_time += t22 - t11
            grad_time += t33 - t22

            # 2. 关键：数据落地 (Offload to CPU & Numpy)
            # 立即 detach 并转为 numpy，切断与计算图的联系，防止显存泄漏
            E_loc_batch_np = local_energies_batch.detach().cpu().numpy()
            amps_batch_np = amps_batch.detach().cpu().numpy()
            grads_batch_np = grads_vec_batch.detach().cpu().numpy()

            E_loc_vec.append(E_loc_batch_np)
            amps_vec.append(amps_batch_np)
            grads_vec_list.append(grads_batch_np)
            
            current_batch_size = fxs.shape[0]
            n_local += current_batch_size

            # 3. 显式清理 (Explicit Cleanup)
            # 删除 Tensor 引用，辅助 GC
            del local_energies_batch, grads_vec_batch, amps_batch

            # 4. 通信握手 (Handshake)
            # a. 汇报进度
            buf = np.array([current_batch_size], dtype=np.int32)
            COMM.Send([buf, MPI.INT], dest=0, tag=TAG_UPDATE)
            
            # b. 等待指令 (Stop/Continue)
            # 这里的等待时间极短，因为 Rank 0 没在做计算
            cmd = np.empty(1, dtype=np.int32)
            COMM.Recv([cmd, MPI.INT], source=0, tag=TAG_CMD)
            
            if cmd[0] == CMD_STOP:
                break
    
    # 确保所有进程都退出了循环
    COMM.Barrier()

    # ==========================================================================
    # Phase 2: Aggregation
    # ==========================================================================
    
    # 1. 拼接本地数据
    # Rank 0 是空的，所以要处理空列表的情况
    if n_local > 0:
        local_energies = np.concatenate(E_loc_vec)
        local_grads = np.concatenate(grads_vec_list)
        local_amps = np.concatenate(amps_vec)
    else:
        # Rank 0 占位符
        local_energies = np.array([], dtype=np.float64)
        local_grads = np.array([], dtype=np.float64) # 形状需要在 gather 后处理
        local_amps = np.array([], dtype=np.float64)

    # 2. 收集所有数据到所有节点 (Allgather)
    # 结果是一个 List of numpy arrays, 长度为 SIZE
    # 注意：Rank 0 的数组是空的
    all_energies_list = COMM.allgather(local_energies)
    
    # 过滤掉空的 Rank 0 数据并拼接
    valid_energies = [e for e in all_energies_list if e.size > 0]
    all_energies_global = np.concatenate(valid_energies)
    
    # 计算统计量
    energy_mean = np.mean(all_energies_global)
    energy_var = np.var(all_energies_global)
    
    total_samples_collected = all_energies_global.shape[0]

    # SR to compute parameter update
    if minSR:
        # 1. 收集梯度和振幅到 Rank 0
        # gather 得到 List[Array], 其中 Rank 0 的 Array 是空的
        gathered_grads_list = COMM.gather(local_grads, root=0)
        gathered_amps_list = COMM.gather(local_amps, root=0)
        
        if RANK == 0:
            # 过滤掉空的 Rank 0 数据
            valid_grads = [g for g in gathered_grads_list if g.size > 0]
            valid_amps = [a for a in gathered_amps_list if a.size > 0]
            
            # 拼接大矩阵
            all_grads = np.concatenate(valid_grads, axis=0) # (N_total, Np)
            all_amps = np.concatenate(valid_amps, axis=0)
            
            # 转为 Tensor 进行计算 (放到 GPU 若可用)
            # 注意 dtype 要高精度
            device = fpeps_model.device if hasattr(fpeps_model, 'device') else 'cpu'
            
            grads_vec_t = torch.tensor(all_grads, dtype=torch.float64, device=device)
            amps_t = torch.tensor(all_amps, dtype=torch.float64, device=device)
            energies_t = torch.tensor(all_energies_global, dtype=torch.float64, device=device)
            
            t0_sr = time.time()
            
            # --- SR Logic ---
            with torch.no_grad():
                # Centered Energy
                E_mean = torch.mean(energies_t)
                E_centered = (energies_t - E_mean) / (total_samples_collected**0.5)
                # Log-derivative Gradients (O_k)
                # grad(psi) / psi
                log_grads = grads_vec_t / amps_t # (N, Np)
                # Centered Gradients (O_sk)
                log_grads_mean = torch.mean(log_grads, dim=0)
                O_sk = (log_grads - log_grads_mean.unsqueeze(0)) / (total_samples_collected**0.5)
                
                # S Matrix (implicitly handled or explicit)
                # T = O_sk @ O_sk^T
                # MinSR: Solve T * x = E_centered
                
                # Option A: SVD / Pinv (Robust for N_samples < N_params)
                # dp = O_sk^T @ pinv(O_sk @ O_sk^T + diag) @ E_centered
                
                T = torch.matmul(O_sk, O_sk.T) # (N, N) Gram Matrix
                T += diag_shift * torch.eye(total_samples_collected, device=device, dtype=torch.float64)
                
                # Solve linear system T * x = E_centered
                # using cholesky_solve or linalg.solve is faster than pinv if T is full rank
                # but pinv is safer
                T_inv = torch.linalg.pinv(T,  rtol=1e-12, atol=0, hermitian=True)
                
                dp = torch.matmul(O_sk.T, torch.matmul(T_inv, E_centered)) # (Np,)
                
                # 将 update vector 转回 CPU
                dp_np = dp.cpu().numpy()

                dp = dp_np
        
                print(f"  SR Update Time : {time.time() - t0_sr:.4f}s")
        else:
            dp_np = None # Worker 占位
            dp = None
        
    else:
        # ======================================================================
        # Distributed Iterative Solver (MinRes)
        # Goal: Solve S * dp = g, where S and g are distributed across Ranks
        # ======================================================================
        t0 = MPI.Wtime()
        import scipy.sparse.linalg as spla

        # 1. Prepare Local Data
        # ----------------------------------------------------------------------
        # Rank 0 的 n_local 为 0，需要特殊处理，防止除法报错或维度错误
        if n_local > 0:
            # 确保 amps 维度正确用于广播 (N, 1)
            amps_reshaped = local_amps.reshape(-1, 1)

            # O_loc = grad(psi) / psi
            # shape: (n_local, Np)
            local_O = local_grads / amps_reshaped 
            
            # 局部求和: sum(O_i)
            local_sum_O = np.sum(local_O, axis=0)
            
            # 局部求能量加权和: sum(E_i * O_i)
            # local_energies: (n_local,) -> reshape to (n_local, 1)
            local_sum_EO = np.dot(local_energies, local_O)
            
        else:
            # Rank 0 (Master) 没有任何样本，贡献为 0
            # 使用 n_params 占位 (假设外部已经获取了参数总数)
            n_params = sum(p.numel() for p in fpeps_model.parameters())
            local_O = np.zeros((0, n_params), dtype=np.float64)
            local_sum_O = np.zeros(n_params, dtype=np.float64)
            local_sum_EO = np.zeros(n_params, dtype=np.float64)

        # 2. Compute Global Gradient
        # ----------------------------------------------------------------------
        # Use Allreduce instead of Gather!
        # Allreduce performs sum reduction at the lower level, communication volume is O(Np), much smaller than Gather's O(N_total * Np)
        
        global_sum_O = np.zeros_like(local_sum_O)
        global_sum_EO = np.zeros_like(local_sum_EO)
        
        COMM.Allreduce(local_sum_O, global_sum_O, op=MPI.SUM)
        COMM.Allreduce(local_sum_EO, global_sum_EO, op=MPI.SUM)
        
        # N_total 之前已经计算过了 (total samples collected)
        mean_O = global_sum_O / total_samples_collected
        mean_EO = global_sum_EO / total_samples_collected
        
        # g = <E*O> - <E><O>
        # energy_mean 是之前 calculate 好的全局能量均值
        energy_grad = mean_EO - energy_mean * mean_O

        # 3. S * x (Distributed MatVec)
        # ----------------------------------------------------------------------
        # S = 1/N * sum(O_i * O_i^T) - <O><O>^T + diag
        # S * x = 1/N * sum(O_i * (O_i . x)) - <O> * (<O> . x) + diag * x
        
        def matvec(x):
            # A. 局部计算 (Computation)
            if n_local > 0:
                # 1. inner = O_i . x  -> shape (n_local,)
                #    这里利用矩阵乘法一次算完所有样本的内积
                inner = local_O.dot(x)
                
                # 2. local_res = O_i.T . inner -> shape (Np,)
                #    加权求和
                local_Sx = local_O.T.dot(inner)
            else:
                # Rank 0 贡献 0
                local_Sx = np.zeros_like(x)
            
            # B. 全局同步 (Synchronization)
            #    将所有 Rank 的 local_Sx 加起来
            global_Sx = np.zeros_like(x)
            COMM.Allreduce(local_Sx, global_Sx, op=MPI.SUM)
            
            # C. 均值项修正与归一化
            #    Sx_cov = 1/N * sum(...) - <O><O>.x
            Sx = global_Sx / total_samples_collected
            
            # 减去中心化项: <O> * (<O> . x)
            mean_O_dot_x = np.dot(mean_O, x)
            Sx -= mean_O_dot_x * mean_O
            
            # D. 对角正则化 (Regularization)
            return Sx + diag_shift * x

        # 4. Run MinRes
        n_params = energy_grad.shape[0]
        A = spla.LinearOperator((n_params, n_params), matvec=matvec, dtype=np.float64)
        # solve Ax = b
        dp, info = spla.minres(A, energy_grad, rtol=1e-4, maxiter=100)
        # release memory
        del local_O, global_sum_O, global_sum_EO, A
        t1 = MPI.Wtime()
        SR_time = t1 - t0

    if RANK == 0:
        print(f'\nSTEP {svmc} Summary:')
        print(f'  Mean Energy per site: {energy_mean/nsites:.6f}')
        print(f'  Mean Energy per Site Variance: {energy_var/total_samples_collected/nsites**2:.6f}')
        print(f'  Mean Energy per Site Std : {np.sqrt(energy_var/total_samples_collected/nsites**2):.6f}')
        print(f'  Total Samples  : {total_samples_collected}')
        print(f'  SR/MinRes Time  : {SR_time:.4f} s')
    COMM.Barrier()
    if RANK == 1:
        print(f'  Rank 1 Stats | Samples: {n_local} | Time(s): Sample={sample_time:.2f}, E={local_energy_time:.2f}, Grad={grad_time:.2f}\n')

    if RANK == 0:
        # update params
        params_vec = torch.nn.utils.parameters_to_vector(fpeps_model.parameters())
        new_params_vec = params_vec - learning_rate * torch.tensor(dp, dtype=torch.float64)
    
    COMM.Barrier()
    
    # broadcast the new params to all ranks
    new_params_vec = COMM.bcast(new_params_vec if RANK == 0 else None, root=0)
    # print(f'Rank {RANK} received new params vector of shape: {new_params_vec.shape}')

    # load the new params back to the model
    torch.nn.utils.vector_to_parameters(new_params_vec, fpeps_model.parameters())

    vmc_pbar.update(1)
    t1 = MPI.Wtime()
    if RANK == 0:
        # save step, energy, energy variance to a file (if exists, delete and create a new one)
        
        log_file = pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/vmc_mpi_log_{fpeps_model._get_name()+str(chi)}.txt'
        if os.path.exists(log_file) and svmc == 0:
            os.remove(log_file)
        with open(log_file, 'a') as f:
            f.write(f'STEP {svmc}:\nEnergy per site: {energy_mean/nsites}\nEnergy variance square root: {np.sqrt(energy_var/total_samples_collected)/nsites}\nSample size: {total_samples_collected}\nTime elapsed: {t1 - t0} seconds\n\n')
        # save Np, sample size, mean, error, variance to a json file
        stats['mean'].append(energy_mean/nsites)
        stats['error'].append(np.sqrt(energy_var/total_samples_collected)/nsites)
        stats['variance'].append(energy_var/nsites**2)
        stats['sample size'] = total_samples_collected

        with open(stats_file, 'w') as f:
            json.dump(stats, f)
        
        # save model checkpoint every few steps
        if (svmc + 1) % save_state_every == 0:
            checkpoint_file = pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/checkpoint_step_{fpeps_model._get_name()+str(chi)}_{svmc+1}.pt'
            torch.save(fpeps_model.state_dict(), checkpoint_file)
    

vmc_pbar.close()