from . import torch, qu, qtn, nn, BasefPEPSBackflowModel
from . import pack_ftn, unpack_ftn, get_params_ftn

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', dtype=torch.float64):
        super().__init__()
        
        # 1. Depthwise: groups = in_channels 意味着每个通道独立卷积
        # 输出通道数必须等于输入通道数
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=in_channels, # <--- 关键参数
            dtype=dtype
        )
        
        # 2. Pointwise: 1x1 卷积，用于混合通道
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            dtype=dtype
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CNN_AllTensor_Generator(nn.Module):
    """
    A Conv2D-based Generator that outputs FULL tensor updates (Delta T).
    No LoRA involved.
    
    Inputs: (Batch, Lx, Ly) Integer Configuration
    Outputs: (Batch, Total_Tensor_Params_Concatenated)
    """
    def __init__(self, tn, embed_dim, hidden_dim, kernel_size=3, layers=2, dtype=torch.float64):
        super().__init__()
        self.dtype = dtype
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        
        # --- 1. Pre-computation: Exact Tensor Sizes ---
        # We need to know the number of elements (numel) for every tensor in the network.
        ftn_params, _ = qtn.pack(tn)
        ftn_params_flat, _ = qu.utils.tree_flatten(ftn_params, get_ref=True)
        
        self.param_sizes = [] # Store numel for each site (ordered 0..N-1)
        max_output_dim = 0
        
        for tensor_data in ftn_params_flat:
            # Direct count of elements: e.g., (N_b, D, D, D, D, d) -> Product of all dims
            num_elements = tensor_data.numel()
            
            self.param_sizes.append(num_elements)
            max_output_dim = max(max_output_dim, num_elements)
            
        self.max_output_dim = max_output_dim
        # print(f"CNN Generator initialized. Max Tensor Size: {self.max_output_dim} elements.")
        
        # --- 2. Network Architecture ---
        
        # A. Physical Embedding
        # Maps integer physical state (0..d-1) to vector
        self.embedding = nn.Embedding(tn.phys_dim(), embed_dim)
        
        # B. Coordinate Grids (Buffer)
        # Create (1, 2, Lx, Ly) grid for coordinate awareness
        # Channel 0: x-coord, Channel 1: y-coord
        x_grid = torch.linspace(-1, 1, self.Lx).view(1, 1, self.Lx, 1).expand(1, 1, self.Lx, self.Ly)
        y_grid = torch.linspace(-1, 1, self.Ly).view(1, 1, 1, self.Ly).expand(1, 1, self.Lx, self.Ly)
        self.register_buffer('coord_grid', torch.cat([x_grid, y_grid], dim=1).to(dtype))
        
        # C. CNN Backbone
        # Input channels: embed_dim (physics) + 2 (coordinates)
        in_channels = embed_dim + 2
        
        cnn_layers = []
        # Layer 1: Local perception
        # padding='same' ensures output grid size matches Lx, Ly. 
        # It automatically handles boundary padding with zeros.
        cnn_layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size, padding='same', dtype=dtype))
        cnn_layers.append(nn.GELU())
        
        # Intermediate Layers (Deepening the receptive field)
        for _ in range(layers - 1):
            cnn_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding='same', dtype=dtype))
            cnn_layers.append(nn.GELU())
            
        self.backbone = nn.Sequential(*cnn_layers)
        
        # D. Output Head
        # Projects hidden features to the size of the largest tensor.
        # This is equivalent to a dense MLP acting on every pixel independently.
        self.output_head = nn.Conv2d(hidden_dim, max_output_dim, kernel_size=1, dtype=dtype)

    def initialize_output_scale(self, scale):
        print(f" -> [Init] CNN Dense Generator: Clamping output weights to {scale}")
        # Initialize the last 1x1 Conv layer to start with small perturbations
        torch.nn.init.normal_(self.output_head.weight, mean=0.0, std=scale)
        if self.output_head.bias is not None:
            torch.nn.init.zeros_(self.output_head.bias)

    def forward(self, x):
        """
        x: (Batch, N_sites) Int/Long, flattened site index sequence
        Output: (Batch, Total_Concatenated_Params)
        """
        B = x.shape[0]
        
        # 1. Reshape flat input to grid: (B, Lx, Ly)
        x_grid = x.view(B, self.Lx, self.Ly)
        
        # 2. Embedding
        # (B, Lx, Ly) -> (B, Lx, Ly, Embed_Dim)
        h = self.embedding(x_grid).to(self.dtype)
        
        # Permute to NCHW for Conv2d: (B, Embed_Dim, Lx, Ly)
        h = h.permute(0, 3, 1, 2)
        
        # 3. Concatenate Coordinates
        # coord_grid is (1, 2, Lx, Ly) -> broadcast to (B, 2, Lx, Ly)
        coords = self.coord_grid.expand(B, -1, -1, -1)
        
        # Input to CNN: (B, Embed + 2, Lx, Ly)
        h_in = torch.cat([h, coords], dim=1)
        
        # 4. Run CNN Backbone
        # Feature map: (B, Hidden, Lx, Ly)
        features = self.backbone(h_in)
        
        # 5. Generate Raw Full-Tensor Updates
        # Output: (B, Max_Output_Dim, Lx, Ly)
        raw_outputs = self.output_head(features)
        
        # 6. Flatten and Slice (OBC Handling)
        # We need to flatten the grid back to site index sequence (0..N-1)
        # Permute to (B, Lx, Ly, Max_Output_Dim) first
        raw_outputs = raw_outputs.permute(0, 2, 3, 1)
        
        # Flatten grid: (B, N_sites, Max_Output_Dim)
        raw_outputs_flat = raw_outputs.reshape(B, self.Lx * self.Ly, -1)
        
        # Now we slice strictly what is needed for each site's specific tensor size
        final_parts = []
        for i in range(self.Lx * self.Ly):
            needed = self.param_sizes[i]
            # Slice the vector for site i. 
            # Boundary sites take fewer elements; Bulk sites take max elements.
            site_vec = raw_outputs_flat[:, i, :needed]
            final_parts.append(site_vec)
            
        # Concatenate: (Batch, Total_Network_Params)
        # This vector is ready to be added to the base PEPS parameters.
        return torch.cat(final_parts, dim=1)

class CNN_BulkTensor_Generator(nn.Module):
    """
    A Conv2D-based Generator that outputs bulk tensor updates (Delta T).
    
    Inputs: (Batch, Lx, Ly) Integer Configuration
    Outputs: (Batch, Total_Tensor_Params_Concatenated)
    """
    def __init__(self, tn, embed_dim, hidden_dim, kernel_size=3, layers=2, dtype=torch.float64):
        super().__init__()
        self.dtype = dtype
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        self.bulk_ts_tids = [x*self.Ly + y for x in range(1, self.Lx - 1) for y in range(1, self.Ly - 1)]
        
        # --- 1. Pre-computation: Exact Tensor Sizes ---
        # We need to know the number of elements (numel) for every tensor in the network.
        ftn_params, _ = qtn.pack(tn)
        ftn_params_flat, _ = qu.utils.tree_flatten(ftn_params, get_ref=True)
        
        self.param_sizes = [] # Store numel for each site (ordered 0..N-1)
        max_output_dim = 0
        bulk_output_dim = 0
        for i in range(len(ftn_params_flat)):
            tensor_data = ftn_params_flat[i]
            # Direct count of elements: e.g., (N_b, D, D, D, D, d) -> Product of all dims
            num_elements = tensor_data.numel()
            
            self.param_sizes.append(num_elements)
            if i in self.bulk_ts_tids:
                bulk_output_dim = num_elements
                max_output_dim = max(max_output_dim, bulk_output_dim)
        
        assert bulk_output_dim == max_output_dim, "Bulk tensors should have the largest size in this model design."
            
        self.max_output_dim = max_output_dim
        # print(f"CNN Generator initialized. Bulk Tensor Size: {self.max_output_dim} elements.")
        
        # --- 2. Network Architecture ---
        
        # A. Physical Embedding
        # Maps integer physical state (0..d-1) to vector
        self.embedding = nn.Embedding(tn.phys_dim(), embed_dim)
        
        # B. Coordinate Grids (Buffer)
        # Create (1, 2, Lx, Ly) grid for coordinate awareness
        # Channel 0: x-coord, Channel 1: y-coord
        x_grid = torch.linspace(-1, 1, self.Lx).view(1, 1, self.Lx, 1).expand(1, 1, self.Lx, self.Ly)
        y_grid = torch.linspace(-1, 1, self.Ly).view(1, 1, 1, self.Ly).expand(1, 1, self.Lx, self.Ly)
        self.register_buffer('coord_grid', torch.cat([x_grid, y_grid], dim=1).to(dtype))
        
        # C. CNN Backbone
        # Input channels: embed_dim (physics) + 2 (coordinates)
        in_channels = embed_dim + 2
        
        cnn_layers = []
        # Layer 1: Local perception
        # padding='same' ensures output grid size matches Lx, Ly. 
        # It automatically handles boundary padding with zeros.
        cnn_layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size, padding='same', dtype=dtype))
        cnn_layers.append(nn.GELU())
        
        # Intermediate Layers (Deepening the receptive field)
        for _ in range(layers - 1):
            cnn_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding='same', dtype=dtype))
            cnn_layers.append(nn.GELU())
            
        self.backbone = nn.Sequential(*cnn_layers)
        
        # D. Output Head
        # Projects hidden features to the size of the largest tensor.
        # This is equivalent to a dense MLP acting on every pixel independently.
        self.output_head = nn.Conv2d(hidden_dim, max_output_dim, kernel_size=1, dtype=dtype)

    def initialize_output_scale(self, scale):
        print(f" -> [Init] CNN Dense Generator: Clamping output weights to {scale}")
        # Initialize the last 1x1 Conv layer to start with small perturbations
        torch.nn.init.normal_(self.output_head.weight, mean=0.0, std=scale)
        if self.output_head.bias is not None:
            torch.nn.init.zeros_(self.output_head.bias)

    def forward(self, x):
        """
        x: (Batch, N_sites) Int/Long, flattened site index sequence
        Output: (Batch, Total_Concatenated_Params)
        """
        B = x.shape[0]
        
        # 1. Reshape flat input to grid: (B, Lx, Ly)
        x_grid = x.view(B, self.Lx, self.Ly)
        
        # 2. Embedding
        # (B, Lx, Ly) -> (B, Lx, Ly, Embed_Dim)
        h = self.embedding(x_grid).to(self.dtype)
        
        # Permute to NCHW for Conv2d: (B, Embed_Dim, Lx, Ly)
        h = h.permute(0, 3, 1, 2)
        
        # 3. Concatenate Coordinates
        # coord_grid is (1, 2, Lx, Ly) -> broadcast to (B, 2, Lx, Ly)
        coords = self.coord_grid.expand(B, -1, -1, -1)
        
        # Input to CNN: (B, Embed + 2, Lx, Ly)
        h_in = torch.cat([h, coords], dim=1)
        
        # 4. Run CNN Backbone
        # Feature map: (B, Hidden, Lx, Ly)
        features = self.backbone(h_in)
        
        # 5. Generate Raw Full-Tensor Updates
        # Output: (B, Max_Output_Dim, Lx, Ly)
        raw_outputs = self.output_head(features)
        
        # 6. Flatten and Slice (OBC Handling)
        # We need to flatten the grid back to site index sequence (0..N-1)
        # Permute to (B, Lx, Ly, Max_Output_Dim) first
        raw_outputs = raw_outputs.permute(0, 2, 3, 1)
        
        # Flatten grid: (B, N_sites, Max_Output_Dim)
        raw_outputs_flat = raw_outputs.reshape(B, self.Lx * self.Ly, -1)
        
        # Now we slice strictly the bulk sites to get their updates, and set boundary sites to zero.
        # Effectively, only bulk sites receive non-zero updates from the CNN; boundary sites are unchanged.
        # And this guarantees the useful CNN output is not truncated - matching the bulk ts shape!
        final_parts = []
        for i in range(self.Lx * self.Ly):
            ts_size = self.param_sizes[i]
            # Slice the vector for site i. 
            # Boundary sites: use 0-vector (no update); Bulk sites take all elements.
            if i in self.bulk_ts_tids:
                site_vec = raw_outputs_flat[:, i, :]
            else:
                site_vec = torch.zeros(B, ts_size, dtype=self.dtype)
            final_parts.append(site_vec)
            
        # Concatenate: (Batch, Total_Network_Params)
        # This vector is ready to be added to the base PEPS parameters.
        return torch.cat(final_parts, dim=1)
# ==============================================================================
# Main Model Integration
# ==============================================================================

class Conv2D_Shared_fPEPS_Model_Cluster(BasefPEPSBackflowModel):
    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        nn_hidden_dim,
        embed_dim,
        kernel_size=3, # Controls the "radius": 3x3 sees 1 neighbor, 5x5 sees 2
        layers=2,      # Depth also increases effective receptive field
        init_perturbation_scale=1e-5,
        dtype=torch.float64,
        **kwargs,
    ):
        super().__init__(tn, max_bond, nn_eta, dtype, kwargs.get('debug_file'), contract_boundary_opts=kwargs.get('contract_boundary_opts', {}))
        tn.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=dtype))
        # Use the new Dense CNN Generator
        self.nn_backflow_generator = CNN_AllTensor_Generator(
            tn=tn,
            embed_dim=embed_dim,
            hidden_dim=nn_hidden_dim,
            kernel_size=kernel_size,
            layers=layers,
            dtype=self.dtype
        )
        
        self.nn_backflow = self.nn_backflow_generator
        self.finish_initialization(init_perturbation_scale)


# TODO: add model where NN distinguish bulk vs edge vs corner tensors.
# first step simple model: nn only injects to bulk tensors.
# later steps: add nn to boundary tensors.

class Conv2D_bulk_only_fPEPS_Model_Cluster(BasefPEPSBackflowModel):
    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        nn_hidden_dim,
        embed_dim,
        kernel_size=3, # Controls the "radius": 3x3 sees 1 neighbor, 5x5 sees 2
        layers=2,      # Depth also increases effective receptive field
        init_perturbation_scale=1e-5,
        dtype=torch.float64,
        **kwargs,
    ):
        super().__init__(tn, max_bond, nn_eta, dtype, kwargs.get('debug_file'), contract_boundary_opts=kwargs.get('contract_boundary_opts', {}))
        tn.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=dtype))
        # Use the new Dense CNN Generator
        self.nn_backflow_generator = CNN_BulkTensor_Generator(
            tn=tn,
            embed_dim=embed_dim,
            hidden_dim=nn_hidden_dim,
            kernel_size=kernel_size,
            layers=layers,
            dtype=self.dtype
        )
        
        self.nn_backflow = self.nn_backflow_generator
        self.finish_initialization(init_perturbation_scale)


class CNN_Geometric_Generator(nn.Module):
    """
    A Multi-Head CNN Generator that strictly classifies sites based on Geometry.
    
    Groups (up to 9 types):
    - BULK
    - CORNER_TL, CORNER_TR, CORNER_BL, CORNER_BR
    - EDGE_TOP, EDGE_BOTTOM, EDGE_LEFT, EDGE_RIGHT
    
    Each group gets its own specialized MLP Head.
    """
    def __init__(self, tn, embed_dim, hidden_dim, kernel_size=3, layers=2, dtype=torch.float64):
        super().__init__()
        self.dtype = dtype
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        self.n_sites = self.Lx * self.Ly
        
        # --- 1. Analyze Geometry & Group Sites ---
        ftn_params, _ = qtn.pack(tn)
        ftn_params_flat, _ = qu.utils.tree_flatten(ftn_params, get_ref=True)
        
        # Dictionary to store { GroupName: [site_idx1, site_idx2, ...] }
        self.groups = {}
        # Dictionary to store output size needed for each group { GroupName: numel }
        self.group_output_dims = {}
        
        for i in range(self.n_sites):
            # Calculate coordinates
            x, y = divmod(i, self.Ly)
            
            # Determine Geometric Type
            g_type = self._get_geometric_type(x, y)
            
            if g_type not in self.groups:
                self.groups[g_type] = []
                # Record the required output size from the first tensor of this type
                # We assume all tensors in the same geometric group have the same shape/size
                p = ftn_params_flat[i]
                self.group_output_dims[g_type] = (
                    p.numel() if isinstance(p, torch.Tensor) else p.size
                )
                
            self.groups[g_type].append(i)
            
        print(f" -> [Model] Geometric Grouping: Found {len(self.groups)} active groups.")
        for k, v in self.groups.items():
            print(f"    - {k}: {len(v)} sites (Output Dim: {self.group_output_dims[k]})")

        # Convert indices to tensors for fast gathering
        self.group_indices = {k: torch.tensor(v, dtype=torch.long) for k, v in self.groups.items()}
        
        # --- 2. Network Architecture ---
        
        # A. Shared Embedding & Coordinates (Backbone Input)
        self.embedding = nn.Embedding(tn.phys_dim(), embed_dim)
        
        x_grid = torch.linspace(-1, 1, self.Lx).view(1, 1, self.Lx, 1).expand(1, 1, self.Lx, self.Ly)
        y_grid = torch.linspace(-1, 1, self.Ly).view(1, 1, 1, self.Ly).expand(1, 1, self.Lx, self.Ly)
        self.register_buffer('coord_grid', torch.cat([x_grid, y_grid], dim=1).to(dtype))
        
        # B. Shared CNN Backbone
        in_channels = embed_dim + 2
        cnn_layers = []
        cnn_layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size, padding='same', dtype=dtype))
        cnn_layers.append(nn.GELU())
        
        for _ in range(layers - 1):
            cnn_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding='same', dtype=dtype))
            cnn_layers.append(nn.GELU())
            
        self.backbone = nn.Sequential(*cnn_layers)
        
        # C. Specialized Heads (ModuleDict)
        # Create a linear layer for each active geometric group
        self.heads = nn.ModuleDict()
        for g_name, out_dim in self.group_output_dims.items():
            # Linear: Hidden -> Specific Tensor Size
            self.heads[g_name] = nn.Linear(hidden_dim, out_dim, dtype=dtype)

    def _get_geometric_type(self, x, y):
        """Helper to classify site (x, y)"""
        # x is row index (0..Lx-1), y is col index (0..Ly-1)
        
        is_top = (x == 0)
        is_bottom = (x == self.Lx - 1)
        is_left = (y == 0)
        is_right = (y == self.Ly - 1)
        
        # 1. Corners
        if is_top and is_left: return "CORNER_TL"
        if is_top and is_right: return "CORNER_TR"
        if is_bottom and is_left: return "CORNER_BL"
        if is_bottom and is_right: return "CORNER_BR"
        
        # 2. Edges
        if is_top: return "EDGE_TOP"
        if is_bottom: return "EDGE_BOTTOM"
        if is_left: return "EDGE_LEFT"
        if is_right: return "EDGE_RIGHT"
        
        # 3. Bulk
        return "BULK"

    def initialize_output_scale(self, scale):
        print(f" -> [Init] CNN Geometric Generator: Clamping output weights to {scale}")
        for name, head in self.heads.items():
            torch.nn.init.normal_(head.weight, mean=0.0, std=scale)
            if head.bias is not None:
                torch.nn.init.zeros_(head.bias)

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Prepare Inputs
        x_grid = x.view(B, self.Lx, self.Ly)
        h = self.embedding(x_grid).to(self.dtype).permute(0, 3, 1, 2) # (B, Emb, Lx, Ly)
        coords = self.coord_grid.expand(B, -1, -1, -1)
        h_in = torch.cat([h, coords], dim=1)
        
        # 2. Run Shared Backbone
        # features: (B, Hidden, Lx, Ly)
        features = self.backbone(h_in)
        
        # Flatten spatial dims: (B, Hidden, N_sites) -> Permute -> (B, N_sites, Hidden)
        features_flat = features.flatten(2).permute(0, 2, 1)
        
        # 3. Run Specialized Heads & Reassemble
        # We need to fill the results in the correct order [Site 0, Site 1, ..., Site N]
        results_list = [None] * self.n_sites
        
        for g_name, site_indices in self.group_indices.items():
            # site_indices must be on correct device
            idxs = site_indices.to(x.device)
            
            # A. Gather features for sites in this geometric group
            # (B, N_group, Hidden)
            group_feats = features_flat.index_select(1, idxs)
            
            # B. Run specific Head (e.g., Head_CORNER_TL)
            # (B, N_group, Specific_Size)
            group_outputs = self.heads[g_name](group_feats)
            
            # C. Scatter back to results list
            # Iterate over the sites in this group
            idxs_list = idxs.tolist()
            for i, site_idx in enumerate(idxs_list):
                results_list[site_idx] = group_outputs[:, i, :]
                
        # 4. Final Concatenation
        # Concatenate along dim 1 (Parameter dimension)
        return torch.cat(results_list, dim=1)

# ==============================================================================
# Model Integration
# ==============================================================================

class Conv2D_Geometric_fPEPS_Model_Cluster(BasefPEPSBackflowModel):
    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        nn_hidden_dim,
        embed_dim,
        kernel_size=3,
        layers=2,
        init_perturbation_scale=1e-5,
        dtype=torch.float64,
        **kwargs,
    ):
        super().__init__(tn, max_bond, nn_eta, dtype, kwargs.get('debug_file'), contract_boundary_opts=kwargs.get('contract_boundary_opts', {}))
        tn.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=dtype))
        # Using the Geometric Generator
        self.nn_backflow_generator = CNN_Geometric_Generator(
            tn=tn,
            embed_dim=embed_dim,
            hidden_dim=nn_hidden_dim,
            kernel_size=kernel_size,
            layers=layers,
            dtype=self.dtype
        )

        self.nn_backflow = self.nn_backflow_generator
        self.finish_initialization(init_perturbation_scale)


class Conv2D_Geometric_fPEPS_Model_Cluster_reuse(nn.Module):
    """
    Hybrid CNN-fPEPS model with bMPS boundary caching (reuse interface).

    Uses CNN_Geometric_Generator as the backflow network.  The CNN's
    effective receptive field radius
        radius = layers * (kernel_size - 1) // 2
    plays the same role as `radius` in Transformer_fPEPS_Model_Cluster_reuse:
    it controls how far cached boundary MPS environments must be pulled back
    before a reuse is valid after a Metropolis move.

    Drop-in replacement for Transformer_fPEPS_Model_Cluster_reuse in
    sample_next_reuse / evaluate_energy_reuse.
    """

    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        nn_hidden_dim,
        embed_dim,
        kernel_size=3,
        layers=2,
        init_perturbation_scale=1e-5,
        dtype=torch.float64,
        contract_boundary_opts={},
        **kwargs,
    ):
        import quimb as qu
        import quimb.tensor as qtn
        super().__init__()

        params, skeleton = qtn.pack(tn)
        self.contract_boundary_opts = contract_boundary_opts
        self.dtype = dtype
        self.skeleton = skeleton
        self.skeleton.exponent = 0
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        self.bMPS_x_skeletons = {}
        self.bMPS_y_skeletons = {}
        self.bMPS_params_x_in_dims = None
        self.bMPS_params_y_in_dims = None
        self.chi = max_bond

        # Flatten TN parameters into a single list and a PyTree structure
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(
            params, get_ref=True)
        self.ftn_params_pytree = ftn_params_pytree

        # Register TN parameters as a ParameterList so optimizer can see them
        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])

        # Metadata for reconstruction inside vmap
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_sizes = [p.numel() for p in self.ftn_params]
        self.ftn_params_length = sum(self.ftn_params_sizes)

        # CNN-based backflow generator (geometric heads)
        self.nn_backflow_generator = CNN_Geometric_Generator(
            tn=tn,
            embed_dim=embed_dim,
            hidden_dim=nn_hidden_dim,
            kernel_size=kernel_size,
            layers=layers,
            dtype=self.dtype
        )

        # Receptive field radius for bMPS cache invalidation
        self.radius = layers * (kernel_size - 1) // 2
        self.kernel_size = kernel_size
        self.layers = layers

        self.nn_backflow = self.nn_backflow_generator
        self.nn_param_names = None
        self.nn_eta = nn_eta
        self.params = None
        # 1. Register NN parameter names for functional_call
        self.nn_param_names = [
            name for name, _ in self.nn_backflow.named_parameters()
        ]
        # 2. Combine all params into one list for the optimizer
        # Order: [TN Params ... NN Params]
        self.params = nn.ParameterList(
            list(self.ftn_params) + list(self.nn_backflow.parameters()))

        # Initialize the last NN layer to have small weights
        self._init_weights_for_perturbation(init_perturbation_scale)

    def _init_weights_for_perturbation(self, scale):
        target_module = (self.nn_backflow_generator
                         if self.nn_backflow_generator else self.nn_backflow)
        if hasattr(target_module, 'initialize_output_scale'):
            target_module.initialize_output_scale(scale)
        else:
            print(f"Warning: {type(target_module).__name__} does not implement"
                  " 'initialize_output_scale'.")
            last_layer = None
            if isinstance(target_module, nn.Sequential):
                last_layer = target_module[-1]

            if last_layer and hasattr(last_layer, 'weight'):
                print(f" -> Initializing last layer "
                      f"{type(last_layer).__name__} with scale {scale}")
                torch.nn.init.normal_(last_layer.weight, mean=0.0, std=scale)
                if last_layer.bias is not None:
                    torch.nn.init.zeros_(last_layer.bias)

    def _get_single_amp(self, x, params):
        """
        Get the single amplitude tn for input x.
        """
        n_ftn = len(self.ftn_params)
        ftn_params = params[:n_ftn]
        nn_params = params[n_ftn:]
        nn_params_dict = dict(zip(self.nn_param_names, nn_params))

        # detect if x is batched
        if x.dim() == 1:
            nn_output = torch.func.functional_call(
                self.nn_backflow, nn_params_dict,
                x.unsqueeze(0)).squeeze(0)
        else:
            nn_output = torch.func.functional_call(
                self.nn_backflow, nn_params_dict, x)

        ftn_params_vector = nn.utils.parameters_to_vector(ftn_params)
        nnftn_params_vector = ftn_params_vector + self.nn_eta * nn_output
        nnftn_params = []
        pointer = 0
        for shape in self.ftn_params_shape:
            length = torch.prod(torch.tensor(shape)).item()
            nnftn_params.append(
                nnftn_params_vector[pointer:pointer + length].view(shape))
            pointer += length
        nnftn_params = qu.utils.tree_unflatten(nnftn_params,
                                               self.ftn_params_pytree)
        tns = qtn.unpack(nnftn_params, self.skeleton)
        amp = tns.isel(
            {tns.site_ind(site): x[i]
             for i, site in enumerate(tns.sites)})
        return amp

    @torch.no_grad()
    def cache_bMPS_skeleton(self, x):
        amp = self._get_single_amp(x, self.params)
        env_x = amp.compute_x_environments(max_bond=self.chi, cutoff=0.0)
        bMPS_params_dict = {}
        for key, tn in env_x.items():
            bMPS_params, skeleton = pack_ftn(tn)
            env_x[key] = skeleton
            bMPS_params_dict[key] = bMPS_params

        self.bMPS_x_skeletons = env_x
        bMPS_params_x_in_dims = qu.utils.tree_map(lambda _: 0,
                                                   bMPS_params_dict)
        self.bMPS_params_x_in_dims = bMPS_params_x_in_dims

        env_y = amp.compute_y_environments(max_bond=self.chi, cutoff=0.0)
        bMPS_params_dict = {}
        for key, tn in env_y.items():
            bMPS_params, skeleton = pack_ftn(tn)
            env_y[key] = skeleton
            bMPS_params_dict[key] = bMPS_params
        self.bMPS_y_skeletons = env_y
        bMPS_params_y_in_dims = qu.utils.tree_map(lambda _: 0,
                                                   bMPS_params_dict)
        self.bMPS_params_y_in_dims = bMPS_params_y_in_dims

    @torch.no_grad()
    def cache_bMPS_params_vmap(self, x):
        # return a pytree (dict) of bMPS params for x and y environments
        params = self.params

        def cache_bMPS_params_single(x_single, params):
            amp = self._get_single_amp(x_single, params)
            env_x = amp.compute_x_environments(max_bond=self.chi, cutoff=0.0)
            bMPS_params_x_dict = {}
            for key, btn in env_x.items():
                bMPS_params = get_params_ftn(btn)
                bMPS_params_x_dict[key] = bMPS_params
            bMPS_params_y_dict = {}
            env_y = amp.compute_y_environments(max_bond=self.chi, cutoff=0.0)
            for key, btn in env_y.items():
                bMPS_params = get_params_ftn(btn)
                bMPS_params_y_dict[key] = bMPS_params

            return bMPS_params_x_dict, bMPS_params_y_dict

        return torch.vmap(
            cache_bMPS_params_single,
            in_dims=(0, None),
        )(x, params)

    def cache_bMPS_params_any_direction_vmap(self, x, direction='x'):
        # return a pytree (dict) of bMPS params for x or y environments
        params = self.params

        def cache_bMPS_params_x_single(x_single, params):
            amp = self._get_single_amp(x_single, params)
            env_x = amp.compute_x_environments(max_bond=self.chi, cutoff=0.0)
            amp_val = (env_x[('xmin', self.Lx // 2)]
                       | env_x[('xmax', self.Lx // 2 - 1)]).contract()
            bMPS_params_x_dict = {}
            for key, btn in env_x.items():
                bMPS_params = get_params_ftn(btn)
                bMPS_params_x_dict[key] = bMPS_params
            return bMPS_params_x_dict, amp_val

        def cache_bMPS_params_y_single(x_single, params):
            amp = self._get_single_amp(x_single, params)
            env_y = amp.compute_y_environments(max_bond=self.chi, cutoff=0.0)
            amp_val = (env_y[('ymin', self.Ly // 2)]
                       | env_y[('ymax', self.Ly // 2 - 1)]).contract()
            bMPS_params_y_dict = {}
            for key, btn in env_y.items():
                bMPS_params = get_params_ftn(btn)
                bMPS_params_y_dict[key] = bMPS_params
            return bMPS_params_y_dict, amp_val

        if direction == 'x':
            return torch.vmap(
                cache_bMPS_params_x_single,
                in_dims=(0, None),
            )(x, params)
        else:
            return torch.vmap(
                cache_bMPS_params_y_single,
                in_dims=(0, None),
            )(x, params)

    def update_bMPS_params_to_row_vmap(self,
                                       x,
                                       row_id,
                                       bMPS_params_x_batched,
                                       from_which='xmin'):
        params = self.params

        # update the bMPS params to a specific row_id for all samples in the
        # batch; pull back by radius to find the last safe cached boundary
        if from_which == 'xmin':
            row_edge = max(0, row_id - self.radius)
        else:
            row_edge = min(self.Ly - 1, row_id + self.radius)

        def update_bMPS_params_x_single(x_single, params, row_id,
                                        bMPS_params_x, from_which):
            bMPS_key = (from_which, row_id)
            amp = self._get_single_amp(x_single, params)
            bMPS_to_row = unpack_ftn(bMPS_params_x[bMPS_key],
                                     self.bMPS_x_skeletons[bMPS_key])
            row_tn = amp.select([amp.row_tag(row_id)], which='any')
            # MPO-MPS two row TN
            updated_bMPS = (bMPS_to_row | row_tn)
            # contract to get the updated bMPS
            if from_which == 'xmin':
                if row_id == 0:
                    updated_bMPS = row_tn
                else:
                    updated_bMPS.contract_boundary_from_xmin_(
                        max_bond=self.chi,
                        cutoff=0.0,
                        xrange=[row_id - 1, row_id],
                        **self.contract_boundary_opts)
                updated_bMPS_params = get_params_ftn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(
                    updated_bMPS_params, get_ref=True)
                _, pytree = qu.utils.tree_flatten(
                    bMPS_params_x[(from_which, row_id + 1)], get_ref=True)
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree)
                bMPS_params_x[(from_which,
                               row_id + 1)] = updated_bMPS_params  # inplace
            else:
                if row_id == amp.Ly - 1:
                    updated_bMPS = row_tn
                else:
                    updated_bMPS.contract_boundary_from_xmax_(
                        max_bond=self.chi,
                        cutoff=0.0,
                        xrange=[row_id, row_id + 1],
                        **self.contract_boundary_opts)
                updated_bMPS_params = get_params_ftn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(
                    updated_bMPS_params, get_ref=True)
                _, pytree = qu.utils.tree_flatten(
                    bMPS_params_x[(from_which, row_id - 1)], get_ref=True)
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree)
                bMPS_params_x[(from_which,
                               row_id - 1)] = updated_bMPS_params  # inplace
            return bMPS_params_x

        return torch.vmap(
            update_bMPS_params_x_single,
            in_dims=(0, None, None, self.bMPS_params_x_in_dims, None),
        )(x, params, row_edge, bMPS_params_x_batched, from_which)

    def update_bMPS_params_to_col_vmap(self,
                                       x,
                                       col_id,
                                       bMPS_params_y_batched,
                                       from_which='ymin'):
        params = self.params
        if from_which == 'ymin':
            col_edge = max(0, col_id - self.radius)
        else:
            col_edge = min(self.Lx - 1, col_id + self.radius)

        def update_bMPS_params_y_single(x_single, params, col_id,
                                        bMPS_params_y, from_which):
            bMPS_key = (from_which, col_id)
            amp = self._get_single_amp(x_single, params)
            bMPS_to_col = unpack_ftn(bMPS_params_y[bMPS_key],
                                     self.bMPS_y_skeletons[bMPS_key])
            col_tn = amp.select([amp.col_tag(col_id)], which='any')
            # MPO-MPS two col TN
            updated_bMPS = (bMPS_to_col | col_tn)
            # contract to get the updated bMPS
            if from_which == 'ymin':
                if col_id == 0:
                    updated_bMPS = col_tn
                else:
                    updated_bMPS.contract_boundary_from_ymin_(
                        max_bond=self.chi,
                        cutoff=0.0,
                        yrange=[col_id - 1, col_id],
                        **self.contract_boundary_opts)
                updated_bMPS_params = get_params_ftn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(
                    updated_bMPS_params, get_ref=True)
                _, pytree = qu.utils.tree_flatten(
                    bMPS_params_y[(from_which, col_id + 1)], get_ref=True)
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree)
                bMPS_params_y[(from_which,
                               col_id + 1)] = updated_bMPS_params  # inplace
            else:
                if col_id == amp.Lx - 1:
                    updated_bMPS = col_tn
                else:
                    updated_bMPS.contract_boundary_from_ymax_(
                        max_bond=self.chi,
                        cutoff=0.0,
                        yrange=[col_id, col_id + 1],
                        **self.contract_boundary_opts)
                updated_bMPS_params = get_params_ftn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(
                    updated_bMPS_params, get_ref=True)
                _, pytree = qu.utils.tree_flatten(
                    bMPS_params_y[(from_which, col_id - 1)], get_ref=True)
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree)
                bMPS_params_y[(from_which,
                               col_id - 1)] = updated_bMPS_params  # inplace
            return bMPS_params_y

        return torch.vmap(
            update_bMPS_params_y_single,
            in_dims=(0, None, None, self.bMPS_params_y_in_dims, None),
        )(x, params, col_edge, bMPS_params_y_batched, from_which)

    def amp_tn(self, x):
        return self._get_single_amp(x, self.params)

    def amplitude(
        self,
        x,
        params,
        bMPS_keys=None,
        bMPS_params_xmin=None,
        bMPS_params_xmax=None,
        bMPS_params_ymin=None,
        bMPS_params_ymax=None,
        selected_rows=None,
        selected_cols=None,
    ):
        amp = self._get_single_amp(x, params)

        # replace the x-environment with the cached one
        if (bMPS_params_xmin is not None and bMPS_params_xmax is not None
                and bMPS_keys is not None):
            bMPS_min = unpack_ftn(bMPS_params_xmin,
                                  self.bMPS_x_skeletons[bMPS_keys[0]])
            bMPS_max = unpack_ftn(bMPS_params_xmax,
                                  self.bMPS_x_skeletons[bMPS_keys[1]])
            rows = amp.select(
                [amp.row_tag(row) for row in selected_rows], which='any')
            amp_reuse = (bMPS_min | rows | bMPS_max)
            amp_reuse.view_as_(
                qtn.PEPS,
                site_tag_id=amp._site_tag_id,
                x_tag_id=amp._x_tag_id,
                y_tag_id=amp._y_tag_id,
                Lx=amp._Lx,
                Ly=amp._Ly,
                site_ind_id=amp._site_ind_id,
            )
            num_rows = selected_rows[-1] - selected_rows[0] + 1
            if self.chi > 0:
                amp_reuse.contract_boundary_from_xmin_(
                    max_bond=self.chi,
                    cutoff=0.0,
                    xrange=[
                        bMPS_keys[0][1],
                        bMPS_keys[0][1] + num_rows // 2 - 1
                    ],
                    **self.contract_boundary_opts)
                amp_reuse.contract_boundary_from_xmax_(
                    max_bond=self.chi,
                    cutoff=0.0,
                    xrange=[
                        bMPS_keys[0][1] + num_rows // 2,
                        min(bMPS_keys[1][1] + 1, self.Lx - 1)
                    ],
                    **self.contract_boundary_opts)
            return amp_reuse.contract()

        # replace the y-environment with the cached one
        if (bMPS_params_ymin is not None and bMPS_params_ymax is not None
                and bMPS_keys is not None):
            bMPS_min = unpack_ftn(bMPS_params_ymin,
                                  self.bMPS_y_skeletons[bMPS_keys[0]])
            bMPS_max = unpack_ftn(bMPS_params_ymax,
                                  self.bMPS_y_skeletons[bMPS_keys[1]])
            cols = amp.select(
                [amp.col_tag(col) for col in selected_cols], which='any')
            amp_reuse = (bMPS_min | cols | bMPS_max)
            amp_reuse.view_as_(
                qtn.PEPS,
                site_tag_id=amp._site_tag_id,
                x_tag_id=amp._x_tag_id,
                y_tag_id=amp._y_tag_id,
                Lx=amp._Lx,
                Ly=amp._Ly,
                site_ind_id=amp._site_ind_id,
            )
            num_cols = selected_cols[-1] - selected_cols[0] + 1
            if self.chi > 0:
                amp_reuse.contract_boundary_from_ymin_(
                    max_bond=self.chi,
                    cutoff=0.0,
                    yrange=[
                        bMPS_keys[0][1],
                        bMPS_keys[0][1] + num_cols // 2 - 1
                    ],
                    **self.contract_boundary_opts)
                amp_reuse.contract_boundary_from_ymax_(
                    max_bond=self.chi,
                    cutoff=0.0,
                    yrange=[
                        bMPS_keys[0][1] + num_cols // 2,
                        min(bMPS_keys[1][1] + 1, self.Ly - 1)
                    ],
                    **self.contract_boundary_opts)
            return amp_reuse.contract()

        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi,
                                             cutoff=0.0,
                                             yrange=[0, amp.Ly // 2 - 1],
                                             **self.contract_boundary_opts)
            amp.contract_boundary_from_ymax_(max_bond=self.chi,
                                             cutoff=0.0,
                                             yrange=[amp.Ly // 2, amp.Ly - 1],
                                             **self.contract_boundary_opts)

        return amp.contract()

    def vamp(
        self,
        x,
        params,
        bMPS_keys=None,
        bMPS_params_xmin=None,
        bMPS_params_xmax=None,
        bMPS_params_ymin=None,
        bMPS_params_ymax=None,
        selected_rows=None,
        selected_cols=None,
    ):
        if bMPS_params_xmin is not None and bMPS_params_xmax is not None:
            return torch.vmap(
                self.amplitude,
                in_dims=(
                    0,
                    None,
                    None,
                    self.bMPS_params_x_in_dims[bMPS_keys[0]],
                    self.bMPS_params_x_in_dims[bMPS_keys[1]],
                    None,
                    None,
                    None,
                    None,
                ),
            )(x, params, bMPS_keys, bMPS_params_xmin, bMPS_params_xmax,
              bMPS_params_ymin, bMPS_params_ymax, selected_rows, selected_cols)

        if bMPS_params_ymin is not None and bMPS_params_ymax is not None:
            return torch.vmap(
                self.amplitude,
                in_dims=(
                    0,
                    None,
                    None,
                    None,
                    None,
                    self.bMPS_params_y_in_dims[bMPS_keys[0]],
                    self.bMPS_params_y_in_dims[bMPS_keys[1]],
                    None,
                    None,
                ),
            )(x, params, bMPS_keys, bMPS_params_xmin, bMPS_params_xmax,
              bMPS_params_ymin, bMPS_params_ymax, selected_rows, selected_cols)

        return torch.vmap(
            self.amplitude,
            in_dims=(0, None, None, None, None, None, None, None, None),
        )(
            x,
            params,
            bMPS_keys,
            bMPS_params_xmin,
            bMPS_params_xmax,
            bMPS_params_ymin,
            bMPS_params_ymax,
            selected_rows,
            selected_cols,
        )

    def forward(
        self,
        x,
        bMPS_params_x_batched=None,
        bMPS_params_y_batched=None,
        selected_rows=None,
        selected_cols=None,
    ):
        bMPS_params_xmin = None
        bMPS_params_xmax = None
        bMPS_params_ymin = None
        bMPS_params_ymax = None
        bMPS_keys = None

        if selected_rows is not None:
            bMPS_keys = [('xmin', min(selected_rows)),
                         ('xmax', max(selected_rows))]
            bMPS_params_xmin = bMPS_params_x_batched[bMPS_keys[0]]
            bMPS_params_xmax = bMPS_params_x_batched[bMPS_keys[1]]
        if selected_cols is not None:
            bMPS_keys = [('ymin', min(selected_cols)),
                         ('ymax', max(selected_cols))]
            bMPS_params_ymin = bMPS_params_y_batched[bMPS_keys[0]]
            bMPS_params_ymax = bMPS_params_y_batched[bMPS_keys[1]]

        return self.vamp(
            x,
            self.params,
            bMPS_keys=bMPS_keys,
            bMPS_params_xmin=bMPS_params_xmin,
            bMPS_params_xmax=bMPS_params_xmax,
            bMPS_params_ymin=bMPS_params_ymin,
            bMPS_params_ymax=bMPS_params_ymax,
            selected_rows=selected_rows,
            selected_cols=selected_cols,
        )