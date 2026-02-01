import torch
import pickle
from vmc_torch.experiment.vmap.vmap_torch_utils import RobustSVD
import autoray as ar

import quimb.tensor as qtn
from vmc_torch.experiment.vmap.GPU.GPU_vmap_utils import random_initial_config
from vmc_torch.experiment.vmap.vmap_models import fPEPS_Model

JITTER = 0
driver='gesvd'
ar.register_function('torch', 'linalg.svd', lambda x, **kwargs: RobustSVD.apply(x, jitter_strength=JITTER, driver=driver))


# 设置默认精度
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:0')
torch.set_default_device(device)

# ==========================================
# 2. 参数设置与模型加载
# ==========================================
Lx, Ly = 4, 4
nsites = Lx * Ly
N_f = nsites - 2
D = 4
chi = -1

# 路径配置 (保持你的原样)
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/data'
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
    ts.modify(data=ts.data.to_flat() * 4)
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
# 尝试编译模型
c_fpeps_model = torch.compile(fpeps_model, mode="default", fullgraph=False)

n_params = sum(p.numel() for p in fpeps_model.parameters())
# ==========================================
# 3. 采样配置
# ==========================================

# 并行运行的 Chain 数量 (Batch Size)
# 如果显存够，可以直接设为 samples_per_rank，这样一步到位
# 如果显存不够，可以设小一点，循环多次累积
batch_size_per_rank = 64
# 确保初始化 walkers 在 GPU 上
fxs_list = [random_initial_config(N_f, nsites, seed=42+_) for _ in range(batch_size_per_rank)]
fxs = torch.stack(fxs_list).to(device)

# print(fpeps_model(fxs))
print(c_fpeps_model(fxs))