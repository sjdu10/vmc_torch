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

    # def forward(self, x):
    #     # x is (Batch, L)
        
    #     # 1. Split params
    #     ftn_params = self.params[:len(self.ftn_params)]
    #     nn_params = self.params[len(self.ftn_params):]
    #     nn_params_dict = dict(zip(self.nn_param_names, nn_params))

    #     # 2. Compute Backflow for the WHOLE BATCH at once
    #     # This uses the optimized native Attention kernels (No vmap fallback!)
    #     # Shape: (Batch, ftn_params_length)
    #     batch_nn_outputs = torch.func.functional_call(self.nn_backflow, nn_params_dict, x.to(self.dtype))

    #     # 3. Use vmap ONLY for the TN contraction part
    #     # We map over 'x' (dim 0) and 'batch_nn_outputs' (dim 0)
    #     # We do NOT map over 'ftn_params' (None)
    #     amps = torch.vmap(
    #         self.tn_contraction,
    #         in_dims=(0, None, 0),
    #     )(x, ftn_params, batch_nn_outputs)

    #     return amps


#=== Utility functions for Metropolis-Hastings sampling ===#

def propose_exchange_or_hopping(i, j, current_config, hopping_rate=0.25):
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


# Batched Metropolis-Hastings updates
@torch.inference_mode()
def sample_next(fxs, fpeps_model, graph, hopping_rate=0.25,verbose=False):
    current_amps = fpeps_model(fxs)
    B = len(fxs)
    n = 0
    t0 = time.time()
    for row, edges in graph.row_edges.items():
        for edge in edges:
            n += 1
            # if verbose:
            #     print(f"Processing edge {edge} in row {row}")
            i, j = edge
            proposed_fxs = []
            new_flags = []
            # t0 = time.time()
            for fx in fxs:
                proposed_fx, new = propose_exchange_or_hopping(i, j, fx, hopping_rate=hopping_rate)
                proposed_fxs.append(proposed_fx)
                new_flags.append(new)
            # t1 = time.time()
            # print(f"Propose time: {t1 - t0}")
            proposed_fxs = torch.stack(proposed_fxs, dim=0)
            if not any(new_flags):
                continue
            # only compute amplitudes for newly proposed configs
            new_proposed_fxs = proposed_fxs[torch.tensor(new_flags, dtype=torch.bool)]
            new_proposed_amps = fpeps_model(new_proposed_fxs)
            # print(f"Number of new proposals: {new_proposed_amps.shape[0]} ({B})" )
            proposed_amps = current_amps.clone()
            proposed_amps[torch.tensor(new_flags, dtype=torch.bool)] = new_proposed_amps
            ratio = proposed_amps**2 / current_amps**2
            accept_prob = torch.minimum(ratio, torch.ones_like(ratio))
            for k in range(B):
                if random.random() < accept_prob[k].item():
                    fxs[k] = proposed_fxs[k]
                    current_amps[k] = proposed_amps[k]

    for col, edges in graph.col_edges.items():
        for edge in edges:
            n += 1
            # if verbose:
            #     print(f"Processing edge {edge} in col {col}")
            i, j = edge
            proposed_fxs = []
            new_flags = []
            for fx in fxs:
                proposed_fx, new = propose_exchange_or_hopping(i, j, fx, hopping_rate=hopping_rate)
                proposed_fxs.append(proposed_fx)
                new_flags.append(new)
            proposed_fxs = torch.stack(proposed_fxs, dim=0)
            if not any(new_flags):
                continue
            # only compute amplitudes for newly proposed configs
            new_proposed_fxs = proposed_fxs[torch.tensor(new_flags, dtype=torch.bool)]
            new_proposed_amps = fpeps_model(new_proposed_fxs)
            # print(f"Number of new proposals: {new_proposed_amps.shape[0]} ({B})" )
            proposed_amps = current_amps.clone()
            proposed_amps[torch.tensor(new_flags, dtype=torch.bool)] = new_proposed_amps
            ratio = proposed_amps**2 / current_amps**2
            accept_prob = torch.minimum(ratio, torch.ones_like(ratio))
            for k in range(B):
                if random.random() < accept_prob[k].item():
                    fxs[k] = proposed_fxs[k]
                    current_amps[k] = proposed_amps[k]
    t1 = time.time()
    if verbose:
        if RANK == 1:
            print(f"Completed one full sweep of MH updates over {n} edges in time: {t1 - t0}")
    return fxs, current_amps

@torch.inference_mode()
def evaluate_energy(fxs, fpeps_model, H, current_amps, verbose=False):
    # TODO: divide the connected configs into chunks of size fxs.shape[0] to avoid OOM
    B = fxs.shape[0]
    # get connected configurations and coefficients
    conn_eta_num = []
    conn_etas = []
    conn_eta_coeffs = []
    for fx in fxs:
        eta, coeffs = H.get_conn(fx)
        conn_eta_num.append(len(eta))
        conn_etas.append(torch.tensor(eta))
        conn_eta_coeffs.append(torch.tensor(coeffs))

    conn_etas = torch.cat(conn_etas, dim=0)
    conn_eta_coeffs = torch.cat(conn_eta_coeffs, dim=0)

    if verbose:
        if RANK == 1:
            print(f'Prepared batched conn_etas and coeffs: {conn_etas.shape}, {conn_eta_coeffs.shape} (batch size {B})')

    # calculate amplitudes for connected configs, in the future consider TN reuse to speed up calculation, TN reuse is controlled by a param that is not batched over (control flow?)
    conn_amps = torch.cat([fpeps_model(conn_etas[i:i+B]) for i in range(0, conn_etas.shape[0], B)])

    # Local energy \sum_{s'} H_{s,s'} <s'|psi>/<s|psi>

    local_energies = []
    offset = 0
    for b in range(B):
        n_conn = conn_eta_num[b]
        amps_ratio = conn_amps[offset:offset+n_conn] / current_amps[b]
        energy_b = torch.sum(conn_eta_coeffs[offset:offset+n_conn] * amps_ratio)
        local_energies.append(energy_b)
        offset += n_conn
    local_energies = torch.stack(local_energies, dim=0)
    if verbose:
        if RANK == 1:
            print(f'Batched local energies: {local_energies.shape}')

    # Energy: (1/N) * \sum_s <s|H|psi>/<s|psi> = (1/N) * \sum_s \sum_{s'} H_{s,s'} <s'|psi>/<s|psi>
    energy = torch.mean(local_energies)
    if verbose:
        if RANK == 1:
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
        batched_grads_vec = []
        for b in range(fxs.shape[0]):
            if isinstance(jac_pytree, dict):
                grad_b_iter = [jac_pytree[k][b] for k in jac_pytree.keys()]
            elif isinstance(jac_pytree, list):
                grad_b_iter = [jac_pytree[k][b] for k in range(len(jac_pytree))]

            batched_grads_vec.append(flatten_params(grad_b_iter))

        batched_grads_vec = torch.stack(batched_grads_vec, dim=0)  # shape (B, Np), Np is number of parameters
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


def random_initial_config(N_f, N_sites):
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