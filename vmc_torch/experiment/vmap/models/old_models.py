from . import torch, qu, qtn, nn, F, BasefPEPSBackflowModel
from vmc_torch.nn_sublayers import SelfAttn_block_pos

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

class Transformer_fPEPS_Model(nn.Module):
    def __init__(self, tn, max_bond, nn_eta, nn_hidden_dim, embed_dim, attn_heads, dtype=torch.float64, **kwargs):
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

# ==============================================================================
# 1. Original Self-Attention Block (Strict Reproduction)
# ==============================================================================
class SelfAttn_block(nn.Module):
    """ Plain self-attention block with one-hot embedding and layer norm"""
    def __init__(
        self, n_site, num_classes, embedding_dim, attention_heads, dtype=torch.float32
    ):
        super(SelfAttn_block, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Linear layer to project one-hot vectors to the embedding dimension
        self.embedding = nn.Linear(num_classes, embedding_dim)

        # Self-attention block
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=attention_heads, batch_first=True
        )

        self.dtype = dtype
        self.embedding.to(dtype=dtype)
        self.self_attention.to(dtype=dtype)

    def forward(self, input_seq):

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

        # Step 2: Embed the one-hot encoded sequence
        embedded = self.embedding(one_hot_encoded) 

        # Step 3: Pass through the self-attention block
        # Note: input_seq is (Batch, Seq_Len), embedded is (Batch, Seq_Len, Embed_Dim)
        attn_output, _ = self.self_attention(
            embedded, embedded, embedded, need_weights=False
        )

        # Step 4: Residual connection and layer normalization
        # Strict reproduction of user's logic: using dynamic shape slicing
        attn_output = F.layer_norm(attn_output + embedded, attn_output.size()[1:])

        return attn_output

# ==============================================================================
# 2. Tensorwise MLP Backflow Module
# ==============================================================================
class TensorwiseMLPBackflow(nn.Module):
    """
    For each on-site tensor, assign a narrow on-site projector MLP.
    Input: Flattened Attention Output (Global Context)
    Output: Tensor Parameters correction
    """
    def __init__(self, input_dim, hidden_dim, param_sizes, dtype):
        super().__init__()
        self.input_dim = input_dim
        self.dtype = dtype
        
        # Use ModuleList instead of ModuleDict for easier batch processing/iteration
        # Each element corresponds to one tensor in the TN
        self.mlps = nn.ModuleList()
        
        for p_size in param_sizes:
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, p_size),
            )
            # Ensure dtype
            mlp.to(dtype=dtype)
            self.mlps.append(mlp)
    
    def initialize_output_scale(self, scale):
        print(f" -> [Init] TensorwiseMLPBackflow: Clamping output weights to scale {scale}")
        for net in self.mlps:
            # The last layer of the MLP is at index 2
            last_layer = net[-1]
            torch.nn.init.normal_(last_layer.weight, mean=0.0, std=scale)
            if last_layer.bias is not None:
                torch.nn.init.zeros_(last_layer.bias)

    def forward(self, attn_features):
        # attn_features: (Batch, N_sites, Embed_Dim)
        B = attn_features.shape[0]
        
        # Flatten the features: (Batch, N_sites * Embed_Dim)
        # This matches the user's logic: "nn_features.view(-1)" applied per sample
        flat_features = attn_features.view(B, -1)
        
        outputs = []
        # Iterate over each unique MLP for each tensor
        for mlp in self.mlps:
            # mlp output: (Batch, p_size)
            outputs.append(mlp(flat_features))
            
        # Concatenate all corrections: (Batch, Total_TN_Params)
        return torch.cat(outputs, dim=1)

# ==============================================================================
# 3. Main Vmap-Compatible Model
# ==============================================================================
class fTN_backflow_attn_Tensorwise_Model_vmap(BasefPEPSBackflowModel):
    def __init__(
        self, 
        ftn, 
        max_bond=None, 
        embed_dim=32, 
        attn_heads=4, 
        nn_hidden_dim=4, 
        nn_eta=1.0, 
        dtype=torch.float32,
        init_perturbation_scale=1e-5,
        **kwargs
    ):
        super().__init__(ftn, max_bond, nn_eta, dtype, kwargs.get('jitter_svd', 0), kwargs.get('debug_file'))
        self.dtype = dtype
        
        # --- 1. TN Parameter Setup ---
        # Extract raw arrays and skeleton
        params, skeleton = qtn.pack(ftn)
        self.skeleton = skeleton
        self.chi = max_bond
        
        # Flatten the nested dictionary structure into a single list of tensors
        # This is crucial for vmap to work (it can't trace ModuleDict)
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.ftn_params_pytree = ftn_params_pytree

        # Register TN parameters
        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])
        
        # Store shapes and sizes to reconstruct/slice later
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_sizes = [p.numel() for p in self.ftn_params] 
        self.ftn_params_length = sum(self.ftn_params_sizes)

        # --- 2. Neural Network Setup ---
        input_dim = ftn.Lx * ftn.Ly
        phys_dim = ftn.phys_dim()
        self.embedding_dim = embed_dim
        
        # A. Attention Block
        self.attn_block = SelfAttn_block(
            n_site=input_dim,
            num_classes=phys_dim,
            embedding_dim=embed_dim,
            attention_heads=attn_heads,
            dtype=self.dtype
        )
        
        # B. Tensorwise MLPs
        # Input to MLP is Flattened Attn Output: Lx * Ly * Embed_Dim
        mlp_input_dim = input_dim * embed_dim
        
        self.nn_backflow_generator = TensorwiseMLPBackflow(
            input_dim=mlp_input_dim,
            hidden_dim=nn_hidden_dim, # Matches user's nn_hidden_dim
            param_sizes=self.ftn_params_sizes,
            dtype=self.dtype
        )
        
        # Combine into sequential
        self.nn_backflow = nn.Sequential(
            self.attn_block,
            self.nn_backflow_generator
        )

        self.nn_eta = nn_eta
        
        # Helper for functional call
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        
        # Combine all parameters for optimizers
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))

        # Initialize the MLP output layers to small random values
        self.nn_backflow_generator.initialize_output_scale(scale=init_perturbation_scale)