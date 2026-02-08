from . import torch, qu, qtn, nn, Optional, math, List, F, BasefPEPSBackflowModel

class CNN_DenseTensor_Generator(nn.Module):
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
        self.nn_backflow_generator = CNN_DenseTensor_Generator(
            tn=tn,
            embed_dim=embed_dim,
            hidden_dim=nn_hidden_dim,
            kernel_size=kernel_size,
            layers=layers,
            dtype=self.dtype
        )
        
        self.nn_backflow = self.nn_backflow_generator
        self.finish_initialization(init_perturbation_scale)