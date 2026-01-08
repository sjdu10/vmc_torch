import os
import torch
import torch.distributed as dist
import numpy as np # 仅用于非计算的简单统计或IO
import pickle
import json
import time
from tqdm import tqdm

# 假设这些是你现有的工具库
from vmc_torch.experiment.vmap.GPU_vmap_utils import sample_next, evaluate_energy, compute_grads, random_initial_config
from vmc_torch.experiment.vmap.GPU_vmap_utils import Transformer_fPEPS_Model_batchedAttn, fPEPS_Model
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.experiment.tn_model import init_weights_to_zero
import quimb.tensor as qtn

# ==========================================
# 1. 初始化 Distributed 环境 (GPU Native)
# ==========================================
def setup_distributed():
    if "RANK" not in os.environ:
        # 调试模式：如果没有用 torchrun 启动，默认单卡运行
        print("Warning: Not using torchrun. Defaulting to single device.")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["LOCAL_RANK"] = "0"

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # 核心：设置当前进程使用的 GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device

RANK, WORLD_SIZE, device = setup_distributed()

# 设置默认精度
torch.set_default_dtype(torch.float64)
# 不同 Rank 设置不同随机种子，保证采样独立
torch.manual_seed(42 + RANK)

# ==========================================
# 2. 参数设置与模型加载
# ==========================================
Lx, Ly = 4, 2
nsites = Lx * Ly
N_f = nsites
D = 4
chi = -1

# 路径配置 (保持你的原样)
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap'
u1z2 = True
appendix = '_U1SU' if u1z2 else ''

# 加载骨架 (这部分很快，可以在 CPU 做完再转 GPU)
# 注意：pickle load 最好只在 Rank 0 做然后广播，或者大家各自读文件(如果文件系统支持并发)
# 这里假设大家各自读文件没问题
params_pkl = pickle.load(open(pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/peps_su_params{appendix}.pkl', 'rb'))
skeleton = pickle.load(open(pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/peps_skeleton{appendix}.pkl', 'rb'))
peps = qtn.unpack(params_pkl, skeleton)

# 预处理 (CPU)
for ts in peps.tensors:
    ts.modify(data=ts.data.to_flat() * 10)
for site in peps.sites:
    peps[site].data._label = site
    peps[site].data.indices[-1]._linearmap = ((0, 0), (1, 0), (1, 1), (0, 1))

# 初始化模型并移动到 GPU
# fpeps_model = Transformer_fPEPS_Model_batchedAttn(
#     tn=peps, max_bond=chi, embed_dim=8, attn_heads=4, nn_hidden_dim=16, nn_eta=1, dtype=torch.float64,
# )
fpeps_model = fPEPS_Model(
    tn=peps, max_bond=chi, dtype=torch.float64,
)
fpeps_model.to(device) # <--- 关键：模型全在 GPU

# 初始化权重
model_params_vec = torch.nn.utils.parameters_to_vector(fpeps_model.parameters())
init_std = float(model_params_vec.std().item()) * 0.1
fpeps_model.apply(lambda x: init_weights_to_zero(x, std=init_std))

n_params = sum(p.numel() for p in fpeps_model.parameters())
if RANK == 0:
    print(f'Model parameters: {n_params} | World Size: {WORLD_SIZE} | Device: {device}')

# Hamiltonians
t, U = 1.0, 8.0
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
    gpu=True,
)
graph = H.graph

# ==========================================
# 3. 采样配置
# ==========================================
Total_Ns = int(2e2)  # 总样本数
# 确保每个 Rank 分到的样本数是整数
assert Total_Ns % WORLD_SIZE == 0, f"Total samples {Total_Ns} must be divisible by World Size {WORLD_SIZE}"
samples_per_rank = Total_Ns // WORLD_SIZE

# 并行运行的 Chain 数量 (Batch Size)
# 如果显存够，可以直接设为 samples_per_rank，这样一步到位
# 如果显存不够，可以设小一点，循环多次累积
batch_size_per_rank = 64
# 确保初始化 walkers 在 GPU 上
fxs_list = [random_initial_config(N_f, nsites, seed=42) for _ in range(batch_size_per_rank)]
fxs = torch.stack(fxs_list).to(device)

# Burn-in (Warmup)
for _ in range(4): # 调整你的 burn-in 步数
    fxs, current_amps = sample_next(fxs, fpeps_model, graph, seed=42)

# VMC Settings
vmc_steps = 50
minSR = True # 推荐用 minSR，因为全在 GPU 上很快
learning_rate = 0.1
save_state_every = 10

os.makedirs(pwd + f'/GPU/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}', exist_ok=True)
stats_file = pwd + f'/GPU/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/stats_{fpeps_model._get_name()}.json'
stats = {'Np': n_params, 'sample size': Total_Ns, 'mean': [], 'error': [], 'variance': []}

# ==========================================
# 4. VMC 主循环 (All on GPU)
# ==========================================
if RANK == 0:
    vmc_pbar = tqdm(total=vmc_steps, desc="VMC Steps")

for step in range(vmc_steps):
    t0 = time.time()
    
    # --- A. 本地采样与梯度计算 (Local Sampling & Compute) ---
    # 我们需要在本地累积 samples_per_rank 这么多数据
    local_energies_acc = []
    local_grads_acc = []
    local_amps_acc = []
    
    ##############################################################################
    current_count = 0
    while current_count < samples_per_rank:
        # 1. 采样
        fxs, current_amps = sample_next(fxs, fpeps_model, graph, seed=42)
        
        # 2. 计算能量
        # 注意：evaluate_energy 内部需要确保返回 GPU tensor
        _, local_E = evaluate_energy(fxs, fpeps_model, H, current_amps)
        
        # 3. 计算梯度
        # batch_size=batch_size_per_rank 表示一次处理完，避免 OOM
        local_grads, local_amps = compute_grads(fxs, fpeps_model, vectorize=True)
        
        # 4. 收集 (还是 GPU tensor)
        # 裁剪掉多余的样本 (如果 batch_size 不整除 samples_per_rank)
        needed = min(batch_size_per_rank, samples_per_rank - current_count)
        
        local_energies_acc.append(local_E[:needed])
        local_grads_acc.append(local_grads[:needed])
        local_amps_acc.append(local_amps[:needed])
        
        current_count += needed
    ################################################################################

    # 拼接本地数据
    my_energies = torch.cat(local_energies_acc) # (samples_per_rank, )
    my_grads = torch.cat(local_grads_acc)       # (samples_per_rank, Np)
    my_amps = torch.cat(local_amps_acc)         # (samples_per_rank, )
    
    # 确保内存连续 (通信必须)
    my_energies = my_energies.contiguous()
    my_grads = my_grads.contiguous()
    my_amps = my_amps.contiguous()

    # --- B. 全局聚合 (Global Gather) ---
    # 准备接收容器
    def gather_tensor(tensor):
        gather_list = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]
        dist.all_gather(gather_list, tensor)
        return torch.cat(gather_list)

    total_energies = gather_tensor(my_energies) # (Total_Ns, )
    total_amps = gather_tensor(my_amps)         # (Total_Ns, )
    # 如果 Np 很大，gather grads 可能会显存爆炸。如果炸了需要换策略 (reduce_scatter)。
    # 对于 Transformer fPEPS (Np ~ 10k-100k)，完全没问题。
    total_grads = gather_tensor(my_grads)       # (Total_Ns, Np)

    # --- C. 优化步 (Optimization) ---
    # 为了数值稳定和计算，我们在 Rank 0 上做 SR 的矩阵求逆
    # 其他 Rank 等待广播
    
    # 1. 计算全局能量平均
    E_mean = torch.mean(total_energies)
    E_var = torch.var(total_energies)
    
    # 准备 update 向量容器
    dp = torch.zeros(n_params, device=device, dtype=torch.float64)

    if RANK == 0:
        # SR / MinSR Logic (All GPU)
        # log_psi gradients
        log_grads = total_grads / total_amps # (Total_Ns, Np)
        log_grads_mean = torch.mean(log_grads, dim=0)
        
        # Centering
        O_sk = (log_grads - log_grads_mean.unsqueeze(0)) / np.sqrt(Total_Ns)
        E_s = (total_energies - E_mean) / np.sqrt(Total_Ns)
        
        # SR Matrix T = O * O^dagger
        # (Total_Ns, Np) @ (Np, Total_Ns) -> (Total_Ns, Total_Ns)
        # 如果 Np < Total_Ns (参数少，样本多)，应该算 (Np, Np) 的协方差矩阵
        # 如果 Np > Total_Ns (参数多，样本少)，minSR 算 (Ns, Ns) 的 Gram Matrix
        
        if minSR:
            # T is (Ns, Ns) - usually small
            T = O_sk @ O_sk.conj().T
            # Add small diagonal shift for numerical stability
            diag_shift = 1e-4
            T += diag_shift * torch.eye(T.shape[0], device=T.device, dtype=T.dtype)
            # Pseudo-inverse
            T_inv = torch.linalg.pinv(T, rtol=1e-12, hermitian=True)

            # dp = O^dagger * T_inv * E_s
            # (Np, Ns) @ (Ns, Ns) @ (Ns, ) -> (Np, )
            dp = O_sk.conj().T @ (T_inv @ E_s)
        else:
            # 也可以在这里实现 Iterative Solver (CG/MinRes) using torch.linalg
            # T is (Ns, Ns) - usually small
            T = O_sk @ O_sk.conj().T
            # Add small diagonal shift for numerical stability
            diag_shift = 1e-4
            T += diag_shift * torch.eye(T.shape[0], device=T.device, dtype=T.dtype)
            x = torch.linalg.solve(T, E_s)

            # dp = O^dagger * x
            # (Np, Ns) @ (Ns, ) -> (Np, )
            dp = O_sk.conj().T @ x

        # 打印信息
        print(f"Step {step}: E = {E_mean.item()/nsites:.6f}, Var = {E_var.item()/nsites**2:.2e}, Std of E_mean = {(E_var.item()/(Total_Ns*nsites**2))**0.5:.2e}")
        print(f'SR dp mean: {dp.mean()}, std: {dp.std()}')

    # --- D. 广播更新量 (Broadcast Update) ---
    dist.broadcast(dp, src=0)

    # --- E. 更新模型参数 ---
    # 小技巧：先把 param vector 拿出来，减去 dp，再放回去
    current_params_vec = torch.nn.utils.parameters_to_vector(fpeps_model.parameters())
    new_params_vec = current_params_vec - learning_rate * dp
    torch.nn.utils.vector_to_parameters(new_params_vec, fpeps_model.parameters())

    t1 = time.time()
    
    # --- F. Logging (Rank 0 only) ---
    if RANK == 0:
        stats['mean'].append(E_mean.item()/nsites)
        stats['error'].append(torch.sqrt(E_var).item()/nsites)
        stats['variance'].append(E_var.item())
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
            
        if (step + 1) % save_state_every == 0:
            ckpt_path = pwd + f'/GPU/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/checkpoint_{fpeps_model._get_name()}_{step+1}.pt'
            torch.save(fpeps_model.state_dict(), ckpt_path)
        
        vmc_pbar.update(1)

if RANK == 0:
    vmc_pbar.close()

# 销毁进程组
dist.destroy_process_group()