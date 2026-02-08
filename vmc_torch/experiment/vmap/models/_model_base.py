import quimb as qu
import quimb.tensor as qtn
import torch
import torch.nn as nn

def is_vmap_compatible(x):
    """
    Check if a node is compatible with vmap (Tensor or Number).
    """
    return isinstance(x, torch.Tensor)

def is_quimb_place_holder(x):
    return isinstance(x, qu.tensor.interface.Placeholder)

def _get_params_ftn_pytree(ftn):
    ftn_params_raw, _ = qtn.pack(ftn)
    ftn_params = {}

    for key in ftn_params_raw.keys():
        # 1. Convert to raw pytree (contains None, 'Z2', etc.)
        raw_tree = ftn.tensor_map[key].data.to_pytree()
        ftn_params[key] = raw_tree
    return ftn_params

def pack_ftn(ftn):
    # Get raw params and skeleton from Quimb
    ftn_params_raw, skeleton = qtn.pack(ftn)
    ftn_params = {}

    for key in ftn_params_raw.keys():
        # 1. Convert to raw pytree (contains None, 'Z2', etc.)
        raw_tree = ftn.tensor_map[key].data.to_pytree()
        ftn_params[key] = raw_tree
    flat_ftn_params, skeleton_tree = qu.utils.tree_flatten(ftn_params,
                                              get_ref=True,
                                              is_leaf=is_vmap_compatible)
    flat_ftn_params = qu.utils.tree_map(lambda x: torch.as_tensor(x), flat_ftn_params, is_leaf=lambda x: isinstance(x, bool))
    return flat_ftn_params, skeleton


def unpack_ftn(flat_ftn_params, skeleton):
    # Create a shallow copy of the skeleton to modify
    ftn = skeleton.copy()
    ftn_params = _get_params_ftn_pytree(ftn)
    _, pytree = qu.utils.tree_flatten(
        ftn_params,
        get_ref=True,
        is_leaf=lambda x: is_vmap_compatible(x) or is_quimb_place_holder(x),
    )
    ftn_params = qu.utils.tree_unflatten(flat_ftn_params, pytree)
    for key in ftn_params.keys():
        new_data = ftn.tensor_map[key].data.from_pytree(ftn_params[key])
        ftn.tensor_map[key].modify(data=new_data)

    return ftn

def get_params_ftn(ftn):
    flat_ftn_params, _ = pack_ftn(ftn)
    return flat_ftn_params


def get_receptive_field_2d(Lx, Ly, r, site_index_map=lambda i, j, Lx, Ly: i * Ly + j):
    """
        Get the receptive field (OBC) for each site in a square lattice graph.
        Default ordering is zig-zag ordering.
        Args:
            Lx (int): Lattice size in x direction.
            Ly (int): Lattice size in y direction.
            r (int): Receptive field radius.
            site_index_map (function): Function to map (i, j) to site index.
        Returns:
            dict: A dictionary mapping site index to list of neighbor site indices.
    """
    receptive_field = {}
    for i in range(Lx):
        for j in range(Ly):
            for ix in range(-r+i, r+1+i):
                for jx in range(-r+j, r+1+j):
                    if ix >= 0 and ix < Lx and jx >= 0 and jx < Ly:
                        site_id = site_index_map(i, j, Lx, Ly)
                        if site_id not in receptive_field:
                            receptive_field[site_id] = []
                        receptive_field[site_id].append(site_index_map(ix, jx, Lx, Ly))
    return receptive_field


class LocalSiteNetwork(nn.Module):
    """
    Independent network for a single site:
    Input (Neighbor Indices) -> Embedding -> Self Attention -> MLP -> Output Params
    """
    def __init__(self, n_neighbors, num_classes, embed_dim, attention_heads, hidden_dim, output_dim, dtype):
        super().__init__()
        self.dtype = dtype
        
        # 1. Independent Embedding for this site (or shared, but independent is more expressive)
        self.embedding = nn.Embedding(num_classes, embed_dim)
        
        # 2. Local Self Attention
        # Input: (Batch, n_neighbors, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            batch_first=True,
            dtype=dtype
        )
        
        # 3. Output MLP
        # Flatten input: n_neighbors * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(n_neighbors * embed_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, dtype=dtype)
        )

    def forward(self, x_local_indices):
        # x_local_indices: (Batch, n_neighbors) int64
        
        # Embed: (Batch, n_neighbors, D)
        h = self.embedding(x_local_indices).to(self.dtype)
        
        # Self Attention
        # attn_output: (Batch, n_neighbors, D)
        h_attn, _ = self.attn(h, h, h)
        
        # Residual + Norm (Optional, simplified here to just Attn output)
        h = h + h_attn
        
        # Flatten
        B = h.shape[0]
        h_flat = h.reshape(B, -1)
        
        # Project to Tensor Params
        return self.mlp(h_flat)


class BasefPEPSBackflowModel(nn.Module):
    """
    Base Class: Handles fPEPS parameter management, vmap contraction logic, 
    and general forward pass flow.
    
    Subclasses must define `self.nn_backflow` in `__init__` and call 
    `self.finish_initialization()`.
    """
    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        dtype=torch.float64,
        debug_file=None,
        contract_boundary_opts={}
    ):
        super().__init__()
        self.contract_boundary_opts = contract_boundary_opts
        self.dtype = dtype
        self.debug_file = debug_file
        self.chi = max_bond
        self.nn_eta = nn_eta

        # --- 1. Tensor Network Setup (Common) ---
        # Extract raw arrays and skeleton
        params, skeleton = qtn.pack(tn)
        self.skeleton = skeleton
        self.skeleton.exponent = 0 
        
        # Flatten TN parameters into a single list and a PyTree structure
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(params, get_ref=True)
        self.ftn_params_pytree = ftn_params_pytree

        # Register TN parameters as a ParameterList so optimizer can see them
        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])
        
        # Metadata for reconstruction inside vmap
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_sizes = [p.numel() for p in self.ftn_params] 
        self.ftn_params_length = sum(self.ftn_params_sizes)

        # Placeholders for Child class to fill
        self.nn_backflow = None 
        self.nn_backflow_generator = None 
        self.nn_param_names = None
        self.params = None

        # vamp func
        self._vamp = torch.vmap(self.tn_contraction, in_dims=(0, None, 0), randomness='different')

    def finish_initialization(self, init_scale=1e-5):
        """
        Must be called by the subclass after `self.nn_backflow` is defined.
        It registers all parameters and initializes weights.
        """
        if self.nn_backflow is None:
            raise ValueError("Child class must define self.nn_backflow before calling finish_initialization")

        # 1. Register NN parameter names for functional_call
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        
        # 2. Combine all params into one list for the optimizer
        # Order: [TN Params ... NN Params]
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))
        
        # 3. Initialize perturbation weights to be small
        self._init_weights_for_perturbation(scale=init_scale)

    def _init_weights_for_perturbation(self, scale=1e-5):
        """
        Delegates initialization to the backflow generator (Generic Interface Pattern).
        """
        target_module = self.nn_backflow_generator if self.nn_backflow_generator else self.nn_backflow
        
        if hasattr(target_module, 'initialize_output_scale'):
            target_module.initialize_output_scale(scale)
        else:
            # Fallback for simple structures
            print(f"Warning: {type(target_module).__name__} does not implement 'initialize_output_scale'.")
            # Try to guess standard layers (Sequential/ModuleList)
            last_layer = None
            if isinstance(target_module, nn.Sequential):
                last_layer = target_module[-1]
            
            if last_layer and hasattr(last_layer, 'weight'):
                 print(f" -> Initializing last layer {type(last_layer).__name__} with scale {scale}")
                 torch.nn.init.normal_(last_layer.weight, mean=0.0, std=scale)
                 if last_layer.bias is not None: 
                     torch.nn.init.zeros_(last_layer.bias)

    def tn_contraction(self, x, ftn_params, nn_output):
        """ 
        Core logic for vmap:
        1. Reconstruct TN parameters from vector.
        2. Add NN backflow correction.
        3. Pack into Quimb TN.
        4. Perform contraction.
        """
        # 1. Reconstruct the vector
        ftn_params_vector = nn.utils.parameters_to_vector(ftn_params)
        
        # 2. Add backflow (NN correction)
        # nn_output is a single sample correction vector
        nnftn_params_vector = ftn_params_vector + self.nn_eta * nn_output
        
        # 3. Unpack to PyTree (list of tensors)
        nnftn_params = []
        pointer = 0
        for shape in self.ftn_params_shape:
            length = torch.prod(torch.tensor(shape)).item()
            nnftn_params.append(nnftn_params_vector[pointer:pointer+length].view(shape))
            pointer += length
        
        # Restore dictionary structure and unpack to Quimb TN
        nnftn_params = qu.utils.tree_unflatten(nnftn_params, self.ftn_params_pytree)
        tn = qtn.unpack(nnftn_params, self.skeleton)
        
        # 4. Contraction
        # Note: x here is a single sample (tensor of indices)
        # Select tensors based on configuration x
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        
        # Contract boundary environments if max_bond is set
        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[0, amp.Ly//2-1], **self.contract_boundary_opts)
            amp.contract_boundary_from_ymax_(max_bond=self.chi, cutoff=0.0, yrange=[amp.Ly//2, amp.Ly-1], **self.contract_boundary_opts)
        
        return amp.contract()

    def vamp(self, x, params):
        """
        Batched computation:
        1. functional_call to compute Backflow (Batch Mode).
        2. vmap to compute TN Contraction.
        """
        # 1. Split params into TN part and NN part
        n_ftn = len(self.ftn_params)
        ftn_params = params[:n_ftn]
        nn_params = params[n_ftn:]
        
        # Reconstruct NN param dict for functional_call
        nn_params_dict = dict(zip(self.nn_param_names, nn_params))

        # 2. Compute Backflow (Batch Mode)
        # nn_backflow handles the logic (Global Attn, Local Conv, or Independent Cluster)
        batch_nn_outputs = torch.func.functional_call(self.nn_backflow, nn_params_dict, x)

        # 3. vmap TN Contraction
        # Map over x (dim 0) and nn_outputs (dim 0)
        # We do NOT map over ftn_params (None)
        amps = self._vamp(x, ftn_params, batch_nn_outputs)
            
        return amps

    def forward(self, x):
        # Ensure inputs are long type for embeddings/indexing
        if x.dtype != torch.long:
             x = x.to(torch.long)
        
        # Forward pass wraps vamp with optional jitter context
        return self.vamp(x, self.params)