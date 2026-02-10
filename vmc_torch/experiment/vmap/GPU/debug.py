import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
import torch
from vmc_torch.experiment.vmap.vmap_torch_utils import robust_svd_err_catcher_wrapper
import autoray as ar
import time
import symmray as sr

from vmc_torch.experiment.vmap.GPU.GPU_vmap_utils import random_initial_config
from vmc_torch.experiment.vmap.vmap_models import fPEPS_Model

# # --- Global Configurations ---
# JITTER = 1e-16
# driver = None
# # Register robust SVD for stability
# ar.register_function('torch', 'linalg.svd', lambda x: robust_svd_err_catcher_wrapper(x, jitter=JITTER, driver=driver))


# 设置默认精度
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cuda:0')
# torch.set_default_device(device)

# ==========================================
# 2. 参数设置与模型加载
# ==========================================
Lx, Ly = 8, 8
N_f = Lx * Ly
nsites = Lx * Ly
D = 10
chi = 20

# 路径配置 (保持你的原样)
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/data'
u1z2 = True
appendix = '_U1SU' if u1z2 else ''

# 加载骨架 (这部分很快，可以在 CPU 做完再转 GPU)
# 注意：pickle load 最好只在 Rank 0 做然后广播，或者大家各自读文件(如果文件系统支持并发)
# 这里假设大家各自读文件没问题
peps = sr.networks.PEPS_fermionic_rand(
    "Z2",
    Lx,
    Ly,
    D,
    phys_dim=[
        (0, 0),  # linear index 0 -> charge 0, offset 0
        (1, 1),  # linear index 1 -> charge 1, offset 1
        (1, 0),  # linear index 2 -> charge 1, offset 0
        (0, 1),  # linear index 3 -> charge 0, offset 1
    ],
    subsizes="equal",
    flat=True,
    seed=42,
    dtype=str(dtype).split(".")[-1],
)

# 初始化模型并移动到 GPU
# fpeps_model = Transformer_fPEPS_Model_batchedAttn(
#     tn=peps, max_bond=chi, embed_dim=8, attn_heads=4, nn_hidden_dim=16, nn_eta=1, dtype=torch.float64,
# )
fpeps_model = fPEPS_Model(
    tn=peps, max_bond=chi, dtype=dtype,
)
fpeps_model.to(device) # <--- 关键：模型全在 GPU
# 尝试编译模型
c_fpeps_model = torch.compile(fpeps_model, mode="default", fullgraph=False)


fpeps_model_cpu = fPEPS_Model(
    tn=peps, max_bond=chi, dtype=dtype,
    contract_boundary_opts={
        'mode': 'mps',
        'equalize_norms': 1.0,
        'canonize': True,
    },
)
fpeps_model_cpu.to('cpu')
c_fpeps_model_cpu = torch.compile(fpeps_model_cpu, mode="default", fullgraph=False)
# ==========================================
# 3. 采样配置
# ==========================================

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
def record_time(batch_size, gpu=True):
    # 确保初始化 walkers 在 GPU 上
    fxs_list = [random_initial_config(N_f, nsites, seed=42+_) for _ in range(batch_size)]
    fxs = torch.stack(fxs_list).to(device)
    fxs_cpu = fxs.cpu()

    # 创建 Event
    if gpu:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            
            # Warm up
            fpeps_model(fxs)
            
            # 记录开始
            start_event.record()
            
            fpeps_model(fxs)
            
            # 记录结束
            end_event.record()

        # 依然需要同步，但只是为了在 CPU 侧读取 Event 的结果
        torch.cuda.synchronize()
    
        # 单位是毫秒 (ms)，需要除以 1000 换算成秒
        gpu_time = start_event.elapsed_time(end_event) / 1000.0
        print(f"GPU forward time: {gpu_time:.6f} s")
    else:
        gpu_time = -0.1 # 占位符，表示不测 GPU 时间
        
    # CPU Timing (CPU 不需要 synchronize)
    t3 = time.time()
    with torch.no_grad():
        fpeps_model_cpu(fxs_cpu)
    t4 = time.time()
    cpu_time = t4 - t3
    print(f"CPU forward time: {cpu_time:.6f} s")
    
    return gpu_time, cpu_time


if __name__ == "__main__":
    import numpy as np
    batch_sizes = [1, 4, 16, 64, 128, 256]
    GPU_times = []
    CPU_times = []
    for bs in batch_sizes:
        print(f"Testing batch size: {bs}")
        gpu_time, cpu_time = record_time(bs, gpu=False)
        GPU_times.append(gpu_time)
        CPU_times.append(cpu_time)

    # store the timing results in a .npy file for later plotting
    np.save(f'./{Lx}x{Ly}_D={D}_chi={chi}_timing_results_{str(dtype).split(".")[-1]}.npy', {'batch_sizes': batch_sizes, 'GPU_times': GPU_times, 'CPU_times': CPU_times})
    print(f'./{Lx}x{Ly}_D={D}_chi={chi}_timing_results_{str(dtype).split(".")[-1]}.npy')