import quimb as qu
import quimb.tensor as qtn
import torch
import torch.nn as nn
import math
from typing import Optional
from vmc_torch.nn_sublayers import SelfAttn_block_pos

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