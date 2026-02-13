from . import torch, qu, qtn, nn, BasefPEPSBackflowModel

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
                self.group_output_dims[g_type] = ftn_params_flat[i].numel()
                
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