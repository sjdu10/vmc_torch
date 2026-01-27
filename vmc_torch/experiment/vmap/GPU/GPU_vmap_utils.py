import quimb as qu
import quimb.tensor as qtn
import torch
import torch.nn as nn
import time
import random
from mpi4py import MPI
from vmc_torch.nn_sublayers import SelfAttn_block_pos, SelfAttn_block_pos_batched
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()


class PEPS_Model(nn.Module):
    def __init__(self, tn, max_bond, dtype=torch.float64):
        import quimb as qu
        import quimb.tensor as qtn
        super().__init__()
        
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.chi = max_bond
        # for torch, further flatten pytree into a single list
        params_flat, params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.params_pytree = params_pytree

        # register the flat list parameters
        self.params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in params_flat
        ])

    
    def amplitude(self, x, params):
        tn = qtn.unpack(params, self.skeleton)
        # might need to specify the right site ordering here
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[0, amp.Ly//2-1])
            amp.contract_boundary_from_ymax_(max_bond=self.chi, cutoff=0.0, yrange=[amp.Ly//2, amp.Ly-1])
        return amp.contract()
    
    def vamp(self, x, params):
        params = qu.utils.tree_unflatten(params, self.params_pytree)
        return torch.vmap(
            self.amplitude,
            in_dims=(0, None),
        )(x, params)

    def forward(self, x):
        return self.vamp(x, self.params)



class fPEPS_Model(nn.Module):
    def __init__(self, tn, max_bond, dtype=torch.float64):
        import quimb as qu
        import quimb.tensor as qtn
        super().__init__()
        
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.chi = max_bond
        # for torch, further flatten pytree into a single list
        params_flat, params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.params_pytree = params_pytree

        # register the flat list parameters
        self.params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in params_flat
        ])

    
    def amplitude(self, x, params):
        tn = qtn.unpack(params, self.skeleton)
        # might need to specify the right site ordering here
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[0, amp.Ly//2-1])
            amp.contract_boundary_from_ymax_(max_bond=self.chi, cutoff=0.0, yrange=[amp.Ly//2, amp.Ly-1])
        return amp.contract()
    
    def vamp(self, x, params):
        params = qu.utils.tree_unflatten(params, self.params_pytree)
        return torch.vmap(
            self.amplitude,
            in_dims=(0, None),
        )(x, params)

    def forward(self, x):
        return self.vamp(x, self.params)

class NN_fPEPS_Model(nn.Module):
    def __init__(self, tn, max_bond, nn_eta, nn_hidden_dim, dtype=torch.float64):
        import quimb as qu
        import quimb.tensor as qtn
        super().__init__()
        
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.chi = max_bond
        # for torch, further flatten pytree into a single list
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.ftn_params_pytree = ftn_params_pytree

        # register the flat list parameters
        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_length = nn.utils.parameters_to_vector(self.ftn_params).shape[0]
        
        self.nn_hidden_dim = nn_hidden_dim
        # simplest 2 layer MLP
        self.nn_backflow = nn.Sequential(
            nn.Linear(len(tn.sites), self.nn_hidden_dim, dtype=self.dtype),
            nn.GELU(),
            nn.Linear(self.nn_hidden_dim, self.ftn_params_length, dtype=self.dtype),
        )
        self.nn_eta = nn_eta

        # We use named_parameters() because self.params only contains parameters, not buffers.
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        
        # combine ftn_params and nn_backflow params into a single pytree
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))
        
    def amplitude(self, x, params):
        # split params into ftn_params and nn_backflow params
        ftn_params = params[:len(self.ftn_params)]
        nn_params = params[len(self.ftn_params):]

        nn_params_dict = dict(zip(self.nn_param_names, nn_params))
        # compute nn_backflow output
        # self.nn_backflow.load_state_dict({k: v for k, v in zip(self.nn_backflow.state_dict().keys(), nn_params)})
        nn_output = torch.func.functional_call(self.nn_backflow, nn_params_dict, x.to(self.dtype))
        ftn_params_vector = nn.utils.parameters_to_vector(ftn_params)
        nnftn_params_vector = ftn_params_vector + self.nn_eta * nn_output
        nnftn_params = []
        pointer = 0
        for shape in self.ftn_params_shape:
            length = torch.prod(torch.tensor(shape)).item()
            param = nnftn_params_vector[pointer:pointer+length].view(shape)
            nnftn_params.append(param)
            pointer += length
        nnftn_params = qu.utils.tree_unflatten(nnftn_params, self.ftn_params_pytree)

        tn = qtn.unpack(nnftn_params, self.skeleton)
        # might need to specify the right site ordering here
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[0, amp.Ly//2-1])
            amp.contract_boundary_from_ymax_(max_bond=self.chi, cutoff=0.0, yrange=[amp.Ly//2, amp.Ly-1])
        return amp.contract()
    
    def vamp(self, x, params):
        return torch.vmap(
            self.amplitude,
            in_dims=(0, None),
        )(x, params)

    def forward(self, x):
        return self.vamp(x, self.params)


class PointwiseBackflow(nn.Module):
    """
    一个替换全连接 MLP 的高效模块。
    它对每个 site 独立应用相同的 MLP (Shared Weights),
    然后根据每个 site 实际需要的 TN 参数量进行裁剪和拼接。
    """
    def __init__(self, n_sites, embed_dim, hidden_dim, param_sizes, dtype):
        super().__init__()
        self.param_sizes = param_sizes  # list, e.g., [32, 32, 128, 128, ...]
        self.max_size = max(param_sizes) # e.g., 128
        
        # 这是一个 "Local" MLP，作用于 (Batch, N, embed_dim)
        # 参数量只与 hidden_dim 和 max_size 有关，与 n_sites 无关！
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, self.max_size, dtype=dtype)
        )

    def forward(self, x):
        # x shape: (Batch, n_sites, embed_dim)
        
        # 1. 生成最大可能需要的参数 (Batch, n_sites, max_size)
        raw_out = self.net(x) # Linear layer automatically applies to last dim
        
        # 2. 根据每个 site 实际需要的参数量进行裁剪并拼接
        # 这一步保证了输出形状严格等于 ftn_params_length
        parts = []
        for i, size in enumerate(self.param_sizes):
            # 取出第 i 个 site 的前 size 个参数
            parts.append(raw_out[:, i, :size])
            
        # 3. 拼接成一个大向量: (Batch, Total_TN_Params)
        return torch.cat(parts, dim=1)
    
class Transformer_fPEPS_Model(nn.Module):
    def __init__(self, tn, max_bond, nn_eta, nn_hidden_dim, embed_dim, attn_heads, dtype=torch.float64):
        import quimb as qu
        import quimb.tensor as qtn
        super().__init__()
        
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.chi = max_bond
        # for torch, further flatten pytree into a single list
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.ftn_params_pytree = ftn_params_pytree

        # register the flat list parameters
        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_length = nn.utils.parameters_to_vector(self.ftn_params).shape[0]
        
        self.nn_hidden_dim = nn_hidden_dim
        self.embed_dim = embed_dim
        self.attn_heads = attn_heads
        self.attn = SelfAttn_block_pos(
            n_site=len(tn.sites),
            num_classes=tn.phys_dim(),
            embed_dim=self.embed_dim,
            attention_heads=self.attn_heads,
            dtype=self.dtype,
        )
        # simplest 2 layer MLP
        self.nn_mlp = nn.Sequential(
            nn.Linear(len(tn.sites)*self.embed_dim, self.nn_hidden_dim, dtype=self.dtype),
            nn.GELU(),
            nn.Linear(self.nn_hidden_dim, self.ftn_params_length, dtype=self.dtype),
        )

        # combine attn and mlp into a single nn_backflow
        self.nn_backflow = nn.Sequential(
            self.attn,
            nn.Flatten(start_dim=0),
            self.nn_mlp,
        )
        self.nn_eta = nn_eta


        # We use named_parameters() because self.params only contains parameters, not buffers.
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        
        # combine ftn_params and nn_backflow params into a single pytree
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))
        
    def amplitude(self, x, params):
        # split params into ftn_params and nn_backflow params
        ftn_params = params[:len(self.ftn_params)]
        nn_params = params[len(self.ftn_params):]

        nn_params_dict = dict(zip(self.nn_param_names, nn_params))
        # compute nn_backflow output
        # self.nn_backflow.load_state_dict({k: v for k, v in zip(self.nn_backflow.state_dict().keys(), nn_params)})
        nn_output = torch.func.functional_call(self.nn_backflow, nn_params_dict, x.to(self.dtype))
        ftn_params_vector = nn.utils.parameters_to_vector(ftn_params)
        nnftn_params_vector = ftn_params_vector + self.nn_eta * nn_output
        nnftn_params = []
        pointer = 0
        for shape in self.ftn_params_shape:
            length = torch.prod(torch.tensor(shape)).item()
            param = nnftn_params_vector[pointer:pointer+length].view(shape)
            nnftn_params.append(param)
            pointer += length
        nnftn_params = qu.utils.tree_unflatten(nnftn_params, self.ftn_params_pytree)

        tn = qtn.unpack(nnftn_params, self.skeleton)
        # might need to specify the right site ordering here
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[0, amp.Ly//2-1])
            amp.contract_boundary_from_ymax_(max_bond=self.chi, cutoff=0.0, yrange=[amp.Ly//2, amp.Ly-1])
        return amp.contract()
    
    def vamp(self, x, params):
        return torch.vmap(
            self.amplitude,
            in_dims=(0, None),
        )(x, params)

    def forward(self, x):
        return self.vamp(x, self.params)



class Transformer_fPEPS_Model_batchedAttn(nn.Module):
    def __init__(self, tn, max_bond, nn_eta, nn_hidden_dim, embed_dim, attn_heads, dtype=torch.float64):
        import quimb as qu
        import quimb.tensor as qtn
        super().__init__()
        
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.chi = max_bond
        # for torch, further flatten pytree into a single list
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.ftn_params_pytree = ftn_params_pytree

        # register the flat list parameters
        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])
        # --- NEW: 计算每个 tensor 的参数量 ---
        # e.g., [32, 32, 128, 128, ...]
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_sizes = [p.numel() for p in self.ftn_params] 
        self.ftn_params_length = sum(self.ftn_params_sizes)
        
        self.nn_hidden_dim = nn_hidden_dim
        self.embed_dim = embed_dim
        self.attn_heads = attn_heads
        
        self.attn = SelfAttn_block_pos_batched(
            n_site=len(tn.sites),
            num_classes=tn.phys_dim(),
            embed_dim=self.embed_dim,
            attention_heads=self.attn_heads,
            dtype=self.dtype,
        )

        # --- REPLACED: 使用 PointwiseBackflow 替代原来的 Flatten+MLP ---
        # 你的 ftn_params 列表顺序通常对应 tn.sites 的顺序 (0, 1, 2...)
        # PointwiseBackflow 会自动处理不同大小的 tensor
        self.nn_backflow_generator = PointwiseBackflow(
            n_sites=len(tn.sites),
            embed_dim=self.embed_dim,
            hidden_dim=self.nn_hidden_dim,
            param_sizes=self.ftn_params_sizes,
            dtype=self.dtype
        )

        # combine attn and mlp into a single nn_backflow
        # 注意：移除了 Flatten(start_dim=1)，因为 PointwiseBackflow 需要 (Batch, N, E) 输入
        self.nn_backflow = nn.Sequential(
            self.attn,
            self.nn_backflow_generator
        )

        self.nn_eta = nn_eta


        # We use named_parameters() because self.params only contains parameters, not buffers.
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        
        # combine ftn_params and nn_backflow params into a single pytree
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))
    
    def _get_name(self):
        # Override to provide a custom name for the model
        return 'Transformer_fPEPS_Model_batchedAttn'
    
    def tn_contraction(self, x, ftn_params, nn_output):
        """ This is the part that TRULY needs vmap. """
        # 1. Reconstruct the vector
        ftn_params_vector = nn.utils.parameters_to_vector(ftn_params)
        # 2. Add backflow (nn_output is now a single sample's correction)
        nnftn_params_vector = ftn_params_vector + self.nn_eta * nn_output
        
        # ... Rest of the unpacking and TN contraction logic ...
        nnftn_params = []
        pointer = 0
        for shape in self.ftn_params_shape:
            length = torch.prod(torch.tensor(shape)).item()
            nnftn_params.append(nnftn_params_vector[pointer:pointer+length].view(shape))
            pointer += length
        
        nnftn_params = qu.utils.tree_unflatten(nnftn_params, self.ftn_params_pytree)
        tn = qtn.unpack(nnftn_params, self.skeleton)
        
        # Site indexing and contraction
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[0, amp.Ly//2-1])
            amp.contract_boundary_from_ymax_(max_bond=self.chi, cutoff=0.0, yrange=[amp.Ly//2, amp.Ly-1])
        return amp.contract()

    def vamp(self, x, params):
        # 1. Split params
        ftn_params = params[:len(self.ftn_params)]
        nn_params = params[len(self.ftn_params):]
        nn_params_dict = dict(zip(self.nn_param_names, nn_params))

        # 2. Compute Backflow for the WHOLE BATCH at once
        # This uses the optimized native Attention kernels (No vmap fallback!)
        # Shape: (Batch, ftn_params_length)
        batch_nn_outputs = torch.func.functional_call(self.nn_backflow, nn_params_dict, x.to(self.dtype))

        # 3. Use vmap ONLY for the TN contraction part
        # We map over 'x' (dim 0) and 'batch_nn_outputs' (dim 0)
        # We do NOT map over 'ftn_params' (None)
        amps = torch.vmap(
            self.tn_contraction,
            in_dims=(0, None, 0),
        )(x, ftn_params, batch_nn_outputs)
        return amps

    def forward(self, x):
        return self.vamp(x, self.params)


#=== Utility functions for Metropolis-Hastings sampling ===#

def propose_exchange_or_hopping(i, j, current_config, hopping_rate=0.25, seed=None):
    if seed is not None:
        random.seed(seed)
    ind_n_map = {0: 0, 1: 1, 2: 1, 3: 2}
    if current_config[i] == current_config[j]:
        return current_config, 0
    proposed_config = current_config.clone()
    config_i = current_config[i].item()
    config_j = current_config[j].item()
    if random.random() < 1 - hopping_rate:
        # exchange
        proposed_config[i] = config_j
        proposed_config[j] = config_i
    else:
        # hopping
        n_i = ind_n_map[current_config[i].item()]
        n_j = ind_n_map[current_config[j].item()]
        delta_n = abs(n_i - n_j)
        if delta_n == 1:
            # consider only valid hopping: (0, u) -> (u, 0); (d, ud) -> (ud, d)
            proposed_config[i] = config_j
            proposed_config[j] = config_i
        elif delta_n == 0:
            # consider only valid hopping: (u, d) -> (0, ud) or (ud, 0)
            choices = [(0, 3), (3, 0)]
            choice = random.choice(choices)
            proposed_config[i] = choice[0]
            proposed_config[j] = choice[1]
        elif delta_n == 2:
            # consider only valid hopping: (0, ud) -> (u, d) or (d, u)
            choices = [(1, 2), (2, 1)]
            choice = random.choice(choices)
            proposed_config[i] = choice[0]
            proposed_config[j] = choice[1]
        else:
            raise ValueError("Invalid configuration")
    return proposed_config, 1


def propose_exchange_or_hopping_vec(i, j, current_configs, hopping_rate=0.25, **kwargs):
    """
    完全向量化的 Propose 函数 (GPU Friendly)。
    一次性处理一批构型，无 CPU-GPU 同步。
    
    Args:
        i, j: (int) 发生交换/hopping 的 site 索引
        current_configs: (Batch, N_sites) Tensor, dtype=long/int
        hopping_rate: (float) 跳跃概率
        
    Returns:
        proposed_configs: (Batch, N_sites) 新的构型
        change_mask: (Batch,) bool Tensor, 指示哪些样本发生了有效改变
    """
    B = current_configs.shape[0]
    device = current_configs.device
    
    # 粒子数映射表: 0->0, 1->1, 2->1, 3->2
    # 放到 device 上以便索引
    # 建议在外部定义为全局常量以避免重复创建，但在函数内创建开销也很小
    n_map = torch.tensor([0, 1, 1, 2], device=device, dtype=torch.long)
    
    # 提取第 i 列和第 j 列 (Batch,)
    col_i = current_configs[:, i]
    col_j = current_configs[:, j]
    
    # 1. 基础检查：如果两个位置状态相同，则无法交换或hopping
    # 对应原代码: if current_config[i] == current_config[j]: return ..., 0
    diff_mask = (col_i != col_j)
    
    # 2. 随机决定是 Exchange 还是 Hopping
    # 生成 (Batch,) 的随机数
    rand_vals = torch.rand(B, device=device)
    
    # 只有状态不同 (diff_mask) 的才需要处理
    is_exchange = (rand_vals < (1 - hopping_rate)) & diff_mask
    is_hopping = (~is_exchange) & diff_mask
    
    # 初始化新的一列，默认等于旧的
    new_col_i = col_i.clone()
    new_col_j = col_j.clone()
    
    # --- A. 处理 Exchange (以及 delta_n=1 的 Hopping) ---
    # 计算粒子数
    n_i = n_map[col_i]
    n_j = n_map[col_j]
    delta_n = (n_i - n_j).abs()
    
    # 原逻辑：delta_n == 1 时也是简单的 swap
    mask_swap = is_exchange | (is_hopping & (delta_n == 1))
    
    if mask_swap.any():
        new_col_i[mask_swap] = col_j[mask_swap]
        new_col_j[mask_swap] = col_i[mask_swap]
        
    # --- B. 处理 Hopping (delta_n = 0 或 2) ---
    # 仅当 is_hopping 为 True 时才检查这些条件
    
    # Case: delta_n == 0 (e.g. u,d -> 0,ud)
    # 目标: 随机变为 (0, 3) 或 (3, 0)
    mask_d0 = is_hopping & (delta_n == 0)
    if mask_d0.any():
        # 生成随机位: 0 或 1
        rand_bits = torch.randint(0, 2, (B,), device=device, dtype=torch.bool)
        
        # 准备目标值常量
        val_0 = torch.tensor(0, device=device, dtype=col_i.dtype)
        val_3 = torch.tensor(3, device=device, dtype=col_i.dtype)
        
        # 根据 rand_bits 选择 i 是 0 还是 3
        # rand=0 -> i=0, j=3; rand=1 -> i=3, j=0
        target_i = torch.where(rand_bits, val_3, val_0)
        target_j = torch.where(rand_bits, val_0, val_3)
        
        new_col_i[mask_d0] = target_i[mask_d0]
        new_col_j[mask_d0] = target_j[mask_d0]

    # Case: delta_n == 2 (e.g. 0,ud -> u,d)
    # 目标: 随机变为 (1, 2) 或 (2, 1)
    mask_d2 = is_hopping & (delta_n == 2)
    if mask_d2.any():
        rand_bits_2 = torch.randint(0, 2, (B,), device=device, dtype=torch.bool)
        
        val_1 = torch.tensor(1, device=device, dtype=col_i.dtype)
        val_2 = torch.tensor(2, device=device, dtype=col_i.dtype)
        
        # rand=0 -> i=1, j=2; rand=1 -> i=2, j=1
        target_i_2 = torch.where(rand_bits_2, val_2, val_1)
        target_j_2 = torch.where(rand_bits_2, val_1, val_2)
        
        new_col_i[mask_d2] = target_i_2[mask_d2]
        new_col_j[mask_d2] = target_j_2[mask_d2]
        
    # 3. 组装结果
    # 克隆整个 configs (GPU上内存拷贝很快)
    proposed_configs = current_configs.clone()
    proposed_configs[:, i] = new_col_i
    proposed_configs[:, j] = new_col_j
    
    # diff_mask 基本上覆盖了所有有效变化，
    # 因为 hopping 规则 (u,d)->(0,3) 肯定会改变状态。
    return proposed_configs, diff_mask


# Batched Metropolis-Hastings updates
@torch.inference_mode()
def sample_next(fxs, fpeps_model, graph, hopping_rate=0.25, verbose=False, seed=None):
    current_amps = fpeps_model(fxs)
    B = fxs.shape[0]
    device = fxs.device
    
    n_updates = 0
    t0 = time.time()
    
    # 合并 row_edges 和 col_edges 循环，减少重复代码
    all_edges = []
    for edges in graph.row_edges.values(): 
        all_edges.extend(edges)
    for edges in graph.col_edges.values(): 
        all_edges.extend(edges)
    
    # 如果 edge 很多，考虑 shuffle，防止遍历顺序造成的 bias (虽然 sweeps 多了无所谓)
    # random.shuffle(all_edges) 

    for edge in all_edges:
        n_updates += 1
        i, j = edge
        
        # 1. 直接调用向量化函数，不需要 list comprehension
        proposed_fxs, new_flags = propose_exchange_or_hopping_vec(i, j, fxs, hopping_rate)
        
        # 快速检查: 如果所有 sample 都没有产生有效 update (比如都是同色自旋交换)，直接跳过
        if not new_flags.any():
            continue
        
        # 2. Compute Amplitudes (只计算改变了的)
        # 注意: 即使不需要 update 的，proposed_amps 也要初始化为 current_amps
        proposed_amps = current_amps.clone()
        
        # 仅对 new_flags 为 True 的部分计算模型
        new_proposed_fxs = proposed_fxs[new_flags]
        new_proposed_amps = fpeps_model(new_proposed_fxs)
        proposed_amps[new_flags] = new_proposed_amps
        
        # 3. Accept/Reject (完全向量化，无 .item() 调用)
        # ratio calculation
        # 为了数值稳定性，建议在 log domain 做 (如果 model 输出 log_psi)，如果是 psi，直接平方
        # 避免除以 0: 虽然物理上 psi 不应为 0，但可以加个 epsilon
        ratio = (proposed_amps.abs()**2) / (current_amps.abs()**2 + 1e-18)
        
        # 向量化生成随机数
        probs = torch.rand(B, device=device)
        
        # 生成掩码: 只有 new_flags 为 True 且 随机数 < ratio 的才接受
        # torch.minimum(ratio, 1) 其实不需要显式写，因为 probs 也是 [0, 1)
        accept_mask = new_flags & (probs < ratio)
        
        # 4. Update (In-place update using masking)
        # 使用 torch.where 或者索引赋值，避免 Python loop
        if accept_mask.any():
            fxs[accept_mask] = proposed_fxs[accept_mask]
            current_amps[accept_mask] = proposed_amps[accept_mask]

    t1 = time.time()
    if verbose and RANK == 1:
        print(f"Completed one full sweep of MH updates over {n_updates} edges in time: {t1 - t0:.4f}s")
        
    return fxs, current_amps

@torch.inference_mode()
def evaluate_energy(fxs, fpeps_model, H, current_amps, verbose=False):
    B = fxs.shape[0]
    device = fxs.device
    
    # 1. 准备连接配置 (Hamiltonian Logic - 依然在 CPU)
    # 这一步如果 H.get_conn 是瓶颈，需要重写 H 类使其支持 vectorization
    conn_eta_num = []
    conn_etas = []
    conn_eta_coeffs = []
    
    for fx in fxs:
        eta, coeffs = H.get_conn(fx)
        conn_eta_num.append(len(eta))
        conn_etas.append(torch.as_tensor(eta, device=device))     # 直接转到 device
        conn_eta_coeffs.append(torch.as_tensor(coeffs, device=device))
        
    conn_etas = torch.cat(conn_etas, dim=0)
    conn_eta_coeffs = torch.cat(conn_eta_coeffs, dim=0)
    
    # 记录每个 sample 有多少个连接配置，用于后续还原
    conn_eta_num = torch.tensor(conn_eta_num, device=device) # (B,)

    # 2. Batch 计算 connected amplitudes
    # 这是一个大矩阵运算，效率很高
    chunk_size = B # 或者更大，取决于显存
    conn_amps_list = []
    for i in range(0, conn_etas.shape[0], chunk_size):
        conn_amps_list.append(fpeps_model(conn_etas[i:i+chunk_size]))
    conn_amps = torch.cat(conn_amps_list)

    # 3. 向量化计算 Local Energies (消除 CPU Loop)
    
    # 我们需要构造一个 batch_index 来告诉 GPU 哪些 conn_amps 属于第 b 个样本
    # 例如 conn_eta_num = [2, 3], 那么 batch_ids = [0, 0, 1, 1, 1]
    batch_ids = torch.repeat_interleave(torch.arange(B, device=device), conn_eta_num)
    
    # 扩展 current_amps 以匹配 conn_amps 的长度
    # current_amps_expanded = [amp[0], amp[0], amp[1], amp[1], amp[1]]
    current_amps_expanded = current_amps[batch_ids]
    
    # 计算每一项的 H_s's * (psi_s' / psi_s)
    terms = conn_eta_coeffs * (conn_amps / current_amps_expanded)
    
    # 聚合结果: 将 terms 按照 batch_ids 加回到 local_energies
    local_energies = torch.zeros(B, device=device, dtype=terms.dtype)
    local_energies.index_add_(0, batch_ids, terms)

    energy = torch.mean(local_energies)
    
    if verbose and RANK == 1:
        print(f'Energy: {energy.item()}')

    return energy, local_energies

def flatten_params(parameters):
    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        vec.append(param.reshape(-1))
    return torch.cat(vec)

def compute_grads(fxs, fpeps_model, vectorize=True, batch_size=None, verbose=False):
    if vectorize:
        # Vectorized gradient calculation
        # need per sample gradient of amplitude -- jacobian
        params_pytree = (
            list(fpeps_model.params)
            if type(fpeps_model.params) is torch.nn.ParameterList
            else dict(fpeps_model.params)
            if type(fpeps_model.params) is torch.nn.ParameterDict
            else fpeps_model.params
        )
        # params is a pytree, fxs has shape (B, nsites)
        def g(x, p):
            results = fpeps_model.vamp(x, p)
            return results, results
        if batch_size is None:
            t0 = time.time()
            jac_pytree, amps = torch.func.jacrev(g, argnums=1, has_aux=True)(fxs, params_pytree)
            t1 = time.time()
            if verbose:
                if RANK == 1:
                    print(f"Vectorized jacobian time: {t1 - t0}")
        else:
            B = fxs.shape[0]
            B_grad = batch_size
            jac_pytree_list = []
            amps_list = []
            t0 = time.time()
            for b_start in range(0, B, B_grad):
                b_end = min(b_start + B_grad, B)
                jac_pytree_b, amps_b = torch.func.jacrev(g, argnums=1, has_aux=True)(fxs[b_start:b_end], params_pytree)
                jac_pytree_list.append(jac_pytree_b)
                amps_list.append(amps_b)
            # concatenate jac_pytree_list along batch dimension
            jac_pytree = tree_map(lambda *leaves: torch.cat(leaves, dim=0), *jac_pytree_list)
            amps = torch.cat(amps_list, dim=0)
            t1 = time.time()
            if verbose:
                if RANK == 1:
                    print(f"Batched Vectorized jacobian time: {t1 - t0}")
                    
        # jac_pytree has shape same as params_pytree, each leaf has shape (B, )

        # Get per-sample batched grads in list of pytree format
        leaves, _ = tree_flatten(jac_pytree) # list of leaves in jac_pytree, each leaf shape (B, param_shape)
        leaves_flattend = [leaf.flatten(start_dim=1) for leaf in leaves]  # each leaf shape (B, param_size)
        batched_grads_vec = torch.cat(leaves_flattend, dim=1) # shape (B, Np), Np is number of parameters
        amps.unsqueeze_(1)  # shape (B, 1)
        return batched_grads_vec, amps
    
    else:
        # Sequential non-vectorized gradient calculation
        amps = []
        batched_grads_vec = []
        t0 = time.time()
        for fx in fxs:
            amp = fpeps_model(fx.unsqueeze(0))
            amps.append(amp)
            amp.backward()
            grads = qu.tree_map(lambda x: x.grad, list(fpeps_model.params))
            # batched_grads_vec.append(torch.nn.utils.parameters_to_vector(grads))
            batched_grads_vec.append(flatten_params(grads))
            qu.tree_map(lambda x: x.grad.zero_(), list(fpeps_model.params))
        t1 = time.time()
        if verbose and RANK == 1:
            print(f"Sequential jacobian time: {t1 - t0}")
        amps = torch.stack(amps, dim=0)
        batched_grads_vec = torch.stack(batched_grads_vec, dim=0)
        return batched_grads_vec, amps


def random_initial_config(N_f, N_sites, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    half_filled_config = torch.tensor(
        [1,2] * (N_sites // 2)
    )
    # set first (Lx*Ly - N_f) sites to be empty (0)
    empty_sites = list(range(N_sites-N_f))
    doped_config = half_filled_config.clone()
    doped_config[empty_sites] = 0
    # randomly permute the doped_config
    perm = torch.randperm(N_sites)
    doped_config = doped_config[perm]
    num_1 = torch.sum(doped_config == 1).item()
    num_2 = torch.sum(doped_config == 2).item()
    assert num_1 == N_f//2 and num_2 == N_f//2, f"Number of spin up and spin down fermions should be {N_f//2}, but got {num_1} and {num_2}"

    return doped_config


# =============== Debug ================
def are_pytrees_equal(tree1, tree2):
    from torch.utils import _pytree as pytree
    import torch
    # Flatten both trees
    leaves1, spec1 = pytree.tree_flatten(tree1)
    leaves2, spec2 = pytree.tree_flatten(tree2)
    
    # 1. Compare structure (TreeSpec)
    if spec1 != spec2:
        print("Tree structures differ.")
        return False
    
    # 2. Compare leaves (Tensors/Values)
    if len(leaves1) != len(leaves2):
        print("Number of leaves differ.")
        return False
        
    for l1, l2 in zip(leaves1, leaves2):
        if torch.is_tensor(l1) and torch.is_tensor(l2):
            if not torch.equal(l1, l2):
                print("Tensor leaves differ.")
                return False
        else:
            if (l1 != l2).any():
                print("Non-tensor leaves differ.")
                print("l1:", l1)
                print("l2:", l2)
                return False
                
    return True