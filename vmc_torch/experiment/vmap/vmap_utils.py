import quimb as qu
import quimb.tensor as qtn
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import math
from mpi4py import MPI
from typing import Optional
from vmc_torch.nn_sublayers import SelfAttn_block_pos
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

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, **mha_kwargs):
        super().__init__()
        # instantiate the real MHA
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, **mha_kwargs
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
    ):
        # internally use x for (query, key, value)
        # x should be of shape (batch_size, seq_length, embed_dim)
        return self.mha(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
    
class TransformerLayer(nn.Module):
    """
    单个 Transformer 层 (Pre-Norm 结构)
    结构: Input -> LayerNorm -> SelfAttention -> Residual Add -> Output
    """
    def __init__(self, embed_dim, num_heads, dtype=torch.float32):
        super().__init__()
        self.embed_dim = embed_dim
        self.dtype = dtype
        
        # Pre-Norm 放在 Attention 之前
        self.norm = nn.LayerNorm(embed_dim, dtype=dtype)
        
        # 你的自定义 Attention 模块
        self.attn = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.attn.to(dtype=dtype)

    def forward(self, x):
        # x: (Batch, L, D)
        
        # Pre-Norm: 先 Norm 再进 Attention
        residual = x
        x_norm = self.norm(x)
        
        # Attention
        attn_out, _ = self.attn(x_norm)
        
        # Residual Connection
        return residual + attn_out

class SelfAttn_block_pos_batched(nn.Module):
    """ 
    支持多层堆叠的 Self-attention block with positional encoding 
    Args:
        depth (int): Transformer block 的层数 (default: 1)
    """
    def __init__(
        self, n_site, num_classes, embed_dim, attention_heads, depth=1, dtype=torch.float32
    ):
        super(SelfAttn_block_pos_batched, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.n_site = n_site
        self.dtype = dtype
        self.depth = depth

        # --- Embedding 部分 (vmap 优化版) ---
        # 直接使用 3D Parameter，避免 forward 里的 stack 操作
        # Shape: (Sites, Embed_Dim, Classes)
        self.pos_weights = nn.Parameter(torch.empty(n_site, embed_dim, num_classes, dtype=dtype))
        self.pos_biases = nn.Parameter(torch.empty(n_site, embed_dim, dtype=dtype))
        
        # 初始化 Embedding 参数
        nn.init.kaiming_uniform_(self.pos_weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.pos_weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.pos_biases, -bound, bound)

        # --- Transformer Layers (支持 depth) ---
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim=embed_dim, 
                num_heads=attention_heads, 
                dtype=dtype
            )
            for _ in range(depth)
        ])
        
        # 可选：最后的 Norm (常见于 Pre-Norm 架构的末尾)
        self.final_norm = nn.LayerNorm(embed_dim, dtype=dtype)

    def forward(self, input_seq):
        # input_seq: (Batch, L) or (L,) inside vmap
        
        # --- Step 1: Vmap-safe One-hot Encoding ---
        # 1. Prepare output container
        out_shape = input_seq.shape + (self.num_classes,)
        one_hot_encoded = torch.zeros(out_shape, device=input_seq.device, dtype=self.dtype)

        # 2. Prepare indices (avoid .long() inside vmap if possible, or do it safely)
        indices = input_seq.unsqueeze(-1)
        if indices.dtype != torch.int64:
             indices = indices.to(torch.int64)

        # 3. Scatter (Out-of-place for vmap safety)
        one_hot_encoded = one_hot_encoded.scatter(-1, indices, 1.0)

        # --- Step 2: Batched Position-wise Embedding ---
        # Contract: (Batch, L, C) * (L, D, C) -> (Batch, L, D)
        # Using '...lc' to handle both batched and unbatched inputs
        embedded = torch.einsum('...lc,ldc->...ld', one_hot_encoded, self.pos_weights) + self.pos_biases

        # --- Step 3: Stacked Transformer Blocks ---
        x = embedded
        
        for layer in self.layers:
            x = layer(x)
            
        # --- Step 4: Final Norm ---
        x = self.final_norm(x)

        return x
    
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

class Conv2dBackflow(nn.Module):
    """
    2D 卷积 Backflow 模块。
    将 (Batch, N_sites, Embed) 还原为 (Batch, Embed, Lx, Ly) 的 2D 图像结构，
    利用 Conv2d 提取近邻信息 (上下左右)，再投影回参数空间。
    """
    def __init__(self, lx, ly, kernel_size, embed_dim, hidden_dim, param_sizes, dtype, pbc=False):
        super().__init__()
        
        # 几何完整性检查
        n_sites = len(param_sizes)
        assert lx * ly == n_sites, f"Lattice dims ({lx}x{ly}) do not match n_sites ({n_sites})"
        
        self.lx = lx
        self.ly = ly
        self.param_sizes = param_sizes
        self.max_size = max(param_sizes)
        self.pbc = pbc  # 是否使用周期性边界条件
        
        # 定义 2D 卷积层
        # Kernel=3, Padding=1 保证输出几何尺寸不变 (Lx, Ly)
        padding_mode = 'circular' if pbc else 'zeros'
        
        self.net = nn.Sequential(
            # Layer 1: 混合邻居信息 (Spatial Mixing)
            nn.Conv2d(
                in_channels=embed_dim, 
                out_channels=hidden_dim, 
                kernel_size=kernel_size, 
                padding=kernel_size // 2, 
                padding_mode=padding_mode,
                dtype=dtype
            ),
            nn.GELU(),
            
            # Layer 2: 投影到参数维度 (Pointwise Projection)
            # Kernel=1 等价于 Pointwise Linear，但在 (C, H, W) 格式下运算
            nn.Conv2d(
                in_channels=hidden_dim, 
                out_channels=self.max_size, 
                kernel_size=1, 
                dtype=dtype
            )
        )

    def forward(self, x):
        # Input x: (Batch, N_sites, Embed_Dim)
        B, N, D = x.shape
        
        # 1. Reshape sequence to image: (B, N, D) -> (B, D, Lx, Ly)
        # PyTorch Conv2d 需要 (Batch, Channel, Height, Width)
        x_2d = x.view(B, self.lx, self.ly, D).permute(0, 3, 1, 2)
        
        # 2. Apply 2D Convolutions
        # Out shape: (Batch, Max_Size, Lx, Ly)
        out_2d = self.net(x_2d)
        
        # 3. Flatten back to sequence: (B, Max_Size, Lx, Ly) -> (Batch, N, Max_Size)
        # 必须先 permute 回 (B, Lx, Ly, Max_Size) 以保证 site 顺序一致
        raw_out = out_2d.permute(0, 2, 3, 1).contiguous().view(B, N, self.max_size)
        
        # 4. 根据每个 site 实际需要的参数量进行裁剪并拼接 (逻辑不变)
        parts = []
        for i, size in enumerate(self.param_sizes):
            # raw_out[:, i, :size] shape is (Batch, size)
            parts.append(raw_out[:, i, :size])
            
        # 5. Concatenate: (Batch, Total_TN_Params)
        return torch.cat(parts, dim=1)


class Transformer_fPEPS_Model_batchedAttn(nn.Module):
    def __init__(self, tn, max_bond, nn_eta, nn_hidden_dim, embed_dim, attn_heads, attn_depth=1, init_perturbation_scale=1e-5, dtype=torch.float64):
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
            depth=attn_depth,
            dtype=self.dtype,
        )
        # PointwiseBackflow 会自动处理不同大小的 tensor
        self.nn_backflow_generator = Conv2dBackflow(
            lx=self.skeleton.Lx,
            ly=self.skeleton.Ly,
            kernel_size=3,
            embed_dim=self.embed_dim,
            hidden_dim=self.nn_hidden_dim,
            param_sizes=self.ftn_params_sizes,
            dtype=self.dtype
        )
        # combine attn and mlp into a single nn_backflow
        self.nn_backflow = nn.Sequential(
            self.attn,
            self.nn_backflow_generator
        )

        self.nn_eta = nn_eta
        # We use named_parameters() because self.params only contains parameters, not buffers.
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        # combine ftn_params and nn_backflow params into a single pytree
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))
        # Initialize weights for perturbative backflow
        self._init_weights_for_perturbation(scale=init_perturbation_scale)
    
    def _init_weights_for_perturbation(self, scale=1e-5):
        """
        Initialize the final output layer of the backflow network to small random values,
        """
        backflow_module = self.nn_backflow_generator
        output_layer = backflow_module.net[-1]
        # 兼容 Linear 和 Conv2d
        if isinstance(output_layer, (torch.nn.Linear, torch.nn.Conv2d)):
            print(f" -> Clamping output layer ({type(output_layer).__name__}) weights to scale {scale}")
            torch.nn.init.normal_(output_layer.weight, mean=0.0, std=scale)
            if output_layer.bias is not None:
                torch.nn.init.zeros_(output_layer.bias)
        # Optional
        for m in self.attn.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
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
        batch_nn_outputs = torch.func.functional_call(self.nn_backflow, nn_params_dict, x)

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


# Batched Metropolis-Hastings updates
@torch.inference_mode()
def sample_next(fxs, fpeps_model, graph, hopping_rate=0.25,verbose=False, seed=None):
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
            fx_id = 0
            for fx in fxs:
                proposed_fx, new = propose_exchange_or_hopping(
                    i,
                    j,
                    fx,
                    hopping_rate=hopping_rate,
                    seed=seed + fx_id if seed is not None else None,
                )
                proposed_fxs.append(proposed_fx)
                new_flags.append(new)
                fx_id += 1
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
            fx_id = 0
            for fx in fxs:
                proposed_fx, new = propose_exchange_or_hopping(
                    i,
                    j,
                    fx,
                    hopping_rate=hopping_rate,
                    seed=seed + fx_id if seed is not None else None,
                )
                proposed_fxs.append(proposed_fx)
                new_flags.append(new)
                fx_id += 1
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

def compute_grads(fxs, fpeps_model, vectorize=True, batch_size=None, verbose=False, vmap_grad=False):
    if vectorize:
        # Vectorized gradient calculation
        # need per sample gradient of amplitude -- jacobian
        if vmap_grad:
            B = fxs.shape[0]
            # 确定 chunk size
            B_grad = batch_size if batch_size is not None else B
            
            # 1. 准备参数 PyTree
            # 兼容 ParameterList, ParameterDict 或直接的 Tensor List
            params_pytree = (
                list(fpeps_model.params)
                if isinstance(fpeps_model.params, torch.nn.ParameterList)
                else dict(fpeps_model.params)
                if isinstance(fpeps_model.params, torch.nn.ParameterDict)
                else fpeps_model.params
            )

            # 2. 定义单样本函数 (Single Sample Function)
            # vmap 要求我们定义处理 "单个样本" 的逻辑
            def single_sample_amp_func(x_i, p):
                # x_i: (L,) 来自 vmap 的切片
                # p: 参数 PyTree (Shared)
                
                # 为了复用现有的 vamp (它可能期待 batch 维度), 我们伪造一个 batch=1
                # 并在输出时 squeeze 回 scalar
                
                # 注意：如果你的 fpeps_model.vamp 内部有 vmap，这里会变成嵌套 vmap
                # 但这是被允许的，只是要注意效率。
                # 如果是 Pure TN，建议确保 vamp 能处理 batch=1 的情况
                
                amp = fpeps_model.vamp(x_i.unsqueeze(0), p).squeeze(0)
                return amp, amp # (Loss target, Aux data)

            # 3. 定义 vmap(grad)
            # argnums=1: 对 params_pytree 求导
            # in_dims=(0, None): 对 x_i 进行 batch 映射，对 p 进行广播
            grad_vmap_fn = torch.vmap(
                torch.func.grad(single_sample_amp_func, argnums=1, has_aux=True),
                in_dims=(0, None)
            )

            # 4. Chunking Loop
            grads_pytree_chunks = []
            amps_chunks = []
            
            t0 = time.time()
            for b_start in range(0, B, B_grad):
                b_end = min(b_start + B_grad, B)
                fxs_chunk = fxs[b_start:b_end]
                grads_chunk, amps_c = grad_vmap_fn(fxs_chunk, params_pytree)
                amps_c = amps_c.detach()
                grads_pytree_chunks.append(grads_chunk)
                amps_chunks.append(amps_c)
                
                del grads_chunk, amps_c

            # 5. 结果拼接
            amps = torch.cat(amps_chunks, dim=0)
            if amps.dim() == 1: amps = amps.unsqueeze(-1)
            def concat_leaves(*leaves):
                return torch.cat(leaves, dim=0)
            full_grads_pytree = tree_map(concat_leaves, *grads_pytree_chunks)

            # 6. Flatten to Vector (B, Np)
            leaves, _ = tree_flatten(full_grads_pytree)
            # 每个 leaf 现在是 (B, Param_Shape)
            # Flatten start_dim=1 -> (B, Param_Size)
            flat_leaves = [leaf.flatten(start_dim=1) for leaf in leaves]
            batched_grads_vec = torch.cat(flat_leaves, dim=1)
            
            batched_grads_vec = batched_grads_vec.detach()
            fpeps_model.zero_grad()
            
            t1 = time.time()
            if verbose and RANK == 1:
                print(f"Single Batched vmap(grad) time: {t1 - t0}")

            return batched_grads_vec, amps
        else:
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
                        print(f"Single Batched Jacobian time: {t1 - t0}")
            else:
                B = fxs.shape[0]
                B_grad = batch_size
                jac_pytree_list = []
                amps_list = []
                t0 = time.time()
                for b_start in range(0, B, B_grad):
                    b_end = min(b_start + B_grad, B)
                    jac_pytree_b, amps_b = torch.func.jacrev(g, argnums=1, has_aux=True)(fxs[b_start:b_end], params_pytree)
                    amps_b.detach_()
                    jac_pytree_b = tree_map(lambda x: x.detach(), jac_pytree_b)
                    jac_pytree_list.append(jac_pytree_b)
                    amps_list.append(amps_b)
                # concatenate jac_pytree_list along batch dimension
                jac_pytree = tree_map(lambda *leaves: torch.cat(leaves, dim=0), *jac_pytree_list)
                amps = torch.cat(amps_list, dim=0)
                t1 = time.time()
                if verbose:
                    if RANK == 1:
                        print(f"Single Batched Jacobian time: {t1 - t0}")
            # jac_pytree has shape same as params_pytree, each leaf has shape (B, )

            # Get per-sample batched grads in list of pytree format
            leaves, _ = tree_flatten(jac_pytree) # list of leaves in jac_pytree, each leaf shape (B, param_shape)
            leaves_flattend = [leaf.flatten(start_dim=1) for leaf in leaves]  # each leaf shape (B, param_size)
            batched_grads_vec = torch.cat(leaves_flattend, dim=1) # shape (B, Np), Np is number of parameters
            amps.unsqueeze_(1)  # shape (B, 1)
            
            # clear grads and cached computational graph to save memory on CPU
            batched_grads_vec = batched_grads_vec.detach()
            amps = amps.detach()
            del jac_pytree
            fpeps_model.zero_grad()
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
            print(f"Single Batched Sequential Jacobian time: {t1 - t0}")
        amps = torch.stack(amps, dim=0)
        batched_grads_vec = torch.stack(batched_grads_vec, dim=0)
        return batched_grads_vec, amps

def compute_grads_decoupled(fxs, fpeps_model, batch_size=None, **kwargs):
    """
    TN-NN decoupled gradient computation:
    Step 1: forward pass to get nn_backflow output (no grad), delta_p
    Step 2: calculate TN gradients (chunked vmap) to get sensitivity vector (dAmp/d(ftn_params) and dAmp/d(delta_p))
    Step 3: propagate sensitivity vector back to NN (VJP, sequential loop)
    """
    B = fxs.shape[0]
    
    B_grad = batch_size if batch_size is not None else B
    
    ftn_params = list(fpeps_model.ftn_params)
    nn_params = list(fpeps_model.nn_backflow.parameters())
    nn_params_dict = dict(zip(fpeps_model.nn_param_names, nn_params))

    with torch.no_grad():
        batch_delta_p = torch.func.functional_call(
            fpeps_model.nn_backflow, nn_params_dict, fxs.long()
        )
    # batch_delta_p shape: (B, ftn_params_length)
    
    def tn_only_func(x_i, ftn_p_list, delta_p_i):
        amp = fpeps_model.tn_contraction(x_i, ftn_p_list, delta_p_i)
        return amp, amp # (Target, Aux)

    tn_grad_vmap_func = torch.vmap(
        torch.func.grad(tn_only_func, argnums=(1, 2), has_aux=True), 
        in_dims=(0, None, 0)
    )

    g_ftn_chunks = []
    g_sensitivity_chunks = []
    amps_chunks = []

    t0 = time.time()
    for b_start in range(0, B, B_grad):
        b_end = min(b_start + B_grad, B)
        
        fxs_chunk = fxs[b_start:b_end]
        delta_p_chunk = batch_delta_p[b_start:b_end]
        
        (g_ftn_c, g_sens_c), amps_c = tn_grad_vmap_func(fxs_chunk, ftn_params, delta_p_chunk)
        
        if amps_c.requires_grad:
            amps_c = amps_c.detach()
            
        g_ftn_chunks.append(g_ftn_c)       
        g_sensitivity_chunks.append(g_sens_c)
        amps_chunks.append(amps_c)
        
        del g_ftn_c, g_sens_c, amps_c
    t1 = time.time()

    g_sensitivity = torch.cat(g_sensitivity_chunks, dim=0)
    
    amps = torch.cat(amps_chunks, dim=0)
    if amps.dim() == 1:
        amps = amps.unsqueeze(-1)

    g_ftn = tree_map(lambda *leaves: torch.cat(leaves, dim=0), *g_ftn_chunks)

    # =================================================================
    g_nn_params_list = []
    t2 = time.time()
    for i in range(B):
        x_i = fxs[i].unsqueeze(0) 
        g_sens_i = g_sensitivity[i].unsqueeze(0) 
        
        fpeps_model.nn_backflow.zero_grad()
        
        with torch.enable_grad():
            out_i = torch.func.functional_call(
                fpeps_model.nn_backflow, 
                nn_params_dict, 
                x_i.long()
            )
            target = torch.sum(out_i * g_sens_i.detach())
            grads_i = torch.autograd.grad(target, nn_params, retain_graph=False)
            
        flat_g = flatten_params(grads_i)
        g_nn_params_list.append(flat_g)
    t3 = time.time()
    if kwargs.get('verbose', True) and RANK == 1:
        print(f"Single Batched grad calc: TN gradient time: {t1 - t0}, Sequential NN VJP time: {t3 - t2}")
    # Stack g_nn_params_list into a tensor of shape (B, Np_nn)
    g_nn_params_vec = torch.stack(g_nn_params_list)


    # Flatten g_ftn
    leaves, _ = tree_flatten(g_ftn)
    flat_g_ftn_list = [leaf.flatten(start_dim=1) for leaf in leaves]
    g_ftn_params_vec = torch.cat(flat_g_ftn_list, dim=1)

    g_params_vec = torch.cat([g_ftn_params_vec, g_nn_params_vec], dim=1) # (B, Np_total)
    
    return g_params_vec, amps

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