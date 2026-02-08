from . import torch, qu, qtn, nn, LocalSiteNetwork, BasefPEPSBackflowModel
from . import get_receptive_field_2d
# ==============================================================================
# LoRA NN-fPEPS Models
# ==============================================================================

class LoRA_fPEPS_Model(nn.Module):
    def __init__(self, tn, max_bond, lora_rank=8, dtype=torch.float64, compile=False, contract_boundary_opts={}, **kwargs):
        super().__init__()
        
        # 1. Basic configuration
        self.dtype = dtype
        self.chi = max_bond
        self.contract_boundary_opts = contract_boundary_opts
        self.lora_rank = lora_rank
        
        # 2. Extract TN structure and parameters
        # params_flat: list of numpy arrays representing raw tensor blocks
        params, skeleton = qtn.pack(tn)
        self.skeleton = skeleton
        params_flat, params_pytree = qu.utils.tree_flatten(params, get_ref=True)
        self.params_pytree = params_pytree

        # 3. Register Fixed Tensors as BUFFERS
        # By using register_buffer, these tensors are part of state_dict (for saving/loading)
        # but are NOT returned by model.parameters(), keeping your trainable parameter count minimal.
        self.fixed_tensors = [] 
        for i, x in enumerate(params_flat):
            name = f"fixed_tensor_{i}"
            t = torch.as_tensor(x, dtype=self.dtype).detach()
            self.register_buffer(name, t)
            self.fixed_tensors.append(t) 

        # 4. Register LoRA Parameters (Trainable)
        # We use Batched CP decomposition to maintain the block structure (N_b, D/2, ..., d/2)
        self.lora_sites = nn.ModuleList()
        for tensor_data in params_flat:
            shape = tensor_data.shape
            n_blocks = shape[0]      # Number of symmetry blocks (N_b)
            phys_dims = shape[1:]    # Virtual and physical legs (D/2, ..., d/2)
            
            factors = nn.ParameterList()
            for dim_size in phys_dims:
                # Initialize with small noise so delta_T starts near zero
                # Each factor shape: (N_b, Rank, Dim)
                factor = torch.randn(n_blocks, lora_rank, dim_size, dtype=self.dtype) * 0.01
                factors.append(nn.Parameter(factor))
            self.lora_sites.append(factors)

        # 5. Vmap setup for batched amplitude calculation
        self._vmapped_amplitude = torch.vmap(
            self.amplitude,
            in_dims=(0, None),
            randomness='different',
        )
        if compile:
            self._vmapped_amplitude = torch.compile(
                self._vmapped_amplitude, fullgraph=False, mode="default"
            )

    @property
    def params(self):
        """
        Custom property for compute_grads.
        Returns only the trainable LoRA factors as a nested list (PyTree).
        """
        return [[p for p in site] for site in self.lora_sites]

    def _reconstruct_params(self, lora_params_structure=None):
        """
        Reconstruct the full tensors: T = T_fixed + CP_Reconstruct(LoRA_factors).
        Supports functional injection of parameters for torch.func compatibility.
        """
        # Use provided params (from vmap/grad) or fallback to internal parameters
        iter_params = lora_params_structure if lora_params_structure is not None else self.lora_sites

        current_params_flat = []
        for fixed_t, factors in zip(self.fixed_tensors, iter_params):
            # n_phys_dims excludes the N_b dimension
            n_phys_dims = len(fixed_t.shape) - 1 
            
            # Build Batched CP Einsum string
            # e.g., "xra, xrb, xrc, xrd, xre -> xabcde"
            input_subs = [f"xr{chr(97+i)}" for i in range(n_phys_dims)] 
            output_sub = "x" + "".join([chr(97+i) for i in range(n_phys_dims)])
            eq = f"{','.join(input_subs)}->{output_sub}"
            
            # delta_t matches the shape of fixed_t (N_b, ...)
            delta_t = torch.einsum(eq, *factors)
            current_params_flat.append(fixed_t + delta_t)
            
        return qu.utils.tree_unflatten(current_params_flat, self.params_pytree)

    def amplitude(self, x, params):
        """
        Core physics: compute amplitude for a single configuration x.
        """
        tn = qtn.unpack(params, self.skeleton)
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        
        # Boundary contraction for PEPS
        if self.chi > 0:
            amp.contract_boundary_from_xmin_(
                max_bond=self.chi, cutoff=0.0, xrange=[0, amp.Lx//2-1], **self.contract_boundary_opts
            )
            amp.contract_boundary_from_xmax_(
                max_bond=self.chi, cutoff=0.0, xrange=[amp.Lx//2, amp.Lx-1], **self.contract_boundary_opts
            )
        return amp.contract()
    
    def vamp(self, x, params=None):
        """
        Batched amplitude calculation compatible with compute_grads(fxs, model, p).
        """
        # 1. Reconstruct full tensors (params here refers to LoRA factors)
        combined_params_pytree = self._reconstruct_params(lora_params_structure=params)
        
        # 2. Compute vmapped amplitudes
        return self._vmapped_amplitude(x, combined_params_pytree)

    def forward(self, x):
        return self.vamp(x)
    
class FactorGenerator(nn.Module):
    """
    A lightweight MLP that generates LoRA factors based on the global configuration x.
    """
    def __init__(self, input_dim, output_shapes, hidden_dim=64, dtype=torch.float64):
        super().__init__()
        self.output_shapes = output_shapes # List of (N_b, Rank, Dim)
        
        # Total number of elements to generate across all factors
        self.total_elements = sum(torch.prod(torch.tensor(s)).item() for s in output_shapes)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=dtype),
            nn.Tanh(), # Tanh is often better for wavefunctions than ReLU
            nn.Linear(hidden_dim, int(self.total_elements), dtype=dtype)
        )

    def forward(self, x):
        # x shape: (Batch, N_sites)
        flat_outputs = self.net(x.to(next(self.parameters()).dtype)) # (Batch, Total_Elements)
        
        # Split and reshape into the required factor shapes
        all_factors = []
        cursor = 0
        for shape in self.output_shapes:
            num_el = int(torch.prod(torch.tensor(shape)).item())
            # Reshape to (Batch, N_b, Rank, Dim)
            all_factors.append(flat_outputs[:, cursor:cursor+num_el].view(-1, *shape))
            cursor += num_el
        return all_factors

class LoRA_NNfPEPS_Model(nn.Module):
    def __init__(self, tn, max_bond, lora_rank=8, hidden_dim=64, dtype=torch.float64, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.chi = max_bond
        self.lora_rank = lora_rank
        self.n_sites = len(tn.sites)
        
        # 1. Extract TN structure
        params, skeleton = qtn.pack(tn)
        self.skeleton = skeleton
        params_flat, params_pytree = qu.utils.tree_flatten(params, get_ref=True)
        self.params_pytree = params_pytree

        # 2. Register Fixed Tensors as BUFFERS (Keep parameters minimal)
        self.fixed_tensors = []
        for i, x in enumerate(params_flat):
            self.register_buffer(f"fixed_tensor_{i}", torch.as_tensor(x, dtype=self.dtype).detach())
            self.fixed_tensors.append(getattr(self, f"fixed_tensor_{i}"))

        # 3. Prepare Shapes for the Factor Generator
        # Each tensor site has multiple factors (one per leg)
        self.all_factor_shapes = []
        self.site_factor_counts = []
        for tensor_data in params_flat:
            shape = tensor_data.shape # (N_b, D/2, ..., d/2)
            n_blocks = shape[0]
            phys_dims = shape[1:]
            self.site_factor_counts.append(len(phys_dims))
            for dim_size in phys_dims:
                self.all_factor_shapes.append((n_blocks, lora_rank, dim_size))

        # 4. Neural Factor Generator
        # This is the ONLY part with trainable parameters (besides the optional NN weights)
        self.generator = FactorGenerator(
            input_dim=self.n_sites, 
            output_shapes=self.all_factor_shapes, 
            hidden_dim=hidden_dim, 
            dtype=dtype
        )

    @property
    def params(self):
        """
        Returns the generator's parameters for compute_grads.
        """
        return list(self.generator.parameters())

    def _reconstruct_params_batched(self, x, generator_params=None):
        """
        Reconstruct tensors for each configuration in the batch.
        Output shape for each tensor: (Batch, N_b, D/2, ..., d/2)
        """
        # If generator_params is provided (from torch.func), we use functional call
        if generator_params is not None:
            from torch.func import functional_call
            param_dict = {name: p for name, p in zip([n for n, _ in self.generator.named_parameters()], generator_params)}
            all_generated_factors = functional_call(self.generator, param_dict, (x,))
        else:
            all_generated_factors = self.generator(x)

        current_params_flat_batched = []
        factor_cursor = 0
        
        for fixed_t in self.fixed_tensors:
            num_factors = len(fixed_t.shape) - 1
            factors = all_generated_factors[factor_cursor : factor_cursor + num_factors]
            factor_cursor += num_factors
            
            # --- CORRECTED EINSUM LOGIC ---
            n_phys_dims = len(fixed_t.shape) - 1
            
            # We use UPPERCASE for structural indices to avoid collision with 'a','b','c'...
            # Z: Batch dimension (from x)
            # S: Symmetry Block dimension (N_b)
            # R: LoRA Rank dimension
            # a, b, c... : Physical/Virtual dimensions
            
            # Input subscripts: "ZSRa", "ZSRb", "ZSRc" ...
            input_subs = [f"ZSR{chr(97+i)}" for i in range(n_phys_dims)]
            
            # Output subscript: "ZSabcd..." 
            # Z and S are preserved (batch dims), R is contracted (summed over).
            output_sub = "ZS" + "".join([chr(97+i) for i in range(n_phys_dims)])
            
            eq = f"{','.join(input_subs)}->{output_sub}"
            
            # delta_t shape: (Batch, N_b, D/2, D/2, ..., d/2)
            delta_t = torch.einsum(eq, *factors)
            
            # Broadcast the fixed_t background across the Batch dimension
            # fixed_t: (N_b, ...) -> unsqueeze -> (1, N_b, ...)
            current_params_flat_batched.append(fixed_t.unsqueeze(0) + delta_t)
            
        return current_params_flat_batched

    def amplitude_single(self, x_single, params_single):
        """
        Compute amplitude for a single config with its specific reconstructed tensor.
        """
        # Unflatten the dict/pytree for quimb
        params_dict = qu.utils.tree_unflatten(params_single, self.params_pytree)
        tn = qtn.unpack(params_dict, self.skeleton)
        
        amp = tn.isel({tn.site_ind(site): x_single[i] for i, site in enumerate(tn.sites)})
        if self.chi > 0:
            # Boundary contraction
            amp.contract_boundary_from_xmin_(max_bond=self.chi, xrange=[0, amp.Lx//2-1])
            amp.contract_boundary_from_xmax_(max_bond=self.chi, xrange=[amp.Lx//2, amp.Lx-1])
        return amp.contract()

    def vamp(self, x, params=None):
        """
        Overridden vamp to handle config-dependent tensors.
        """
        # 1. Generate tensors for each x in the batch
        # batched_tensors: List of Tensors, each (Batch, N_b, ...)
        batched_tensors = self._reconstruct_params_batched(x, generator_params=params)
        
        # 2. Map amplitude calculation over the batch
        # We need to slice batched_tensors along the 0-th dimension inside vmap
        def compute_single(x_i, *tensors_i):
            return self.amplitude_single(x_i, tensors_i)

        return torch.vmap(compute_single)(x, *batched_tensors)

    def forward(self, x):
        return self.vamp(x)

# ==============================================================================
# LoRA Local Cluster Generator
# ==============================================================================

class LoRA_LocalClusterGenerator(nn.Module):
    """
    Replaces the simple MLP FactorGenerator.
    Manages N independent LocalSiteNetworks to generate LoRA factors.
    """
    def __init__(self, tn, lora_rank, embed_dim, attn_heads, hidden_dim, radius=1, dtype=torch.float64):
        super().__init__()
        self.dtype = dtype
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        
        # 1. Calculate Receptive Fields
        rf_dict = get_receptive_field_2d(self.Lx, self.Ly, radius)
        n_sites = self.Lx * self.Ly
        self.rf_indices = [rf_dict[i] for i in range(n_sites)]
        
        # 2. Extract on-site tensor shape information for LoRA factors
        # We need to know how many LoRA parameters each site requires.
        ftn_params, _ = qtn.pack(tn)
        ftn_params_flat, _ = qu.utils.tree_flatten(ftn_params, get_ref=True)
        # Record on-site tensor block shapes
        self.ftn_params_shape = [p.shape for p in ftn_params_flat] # p.shape (N_b, D/2, ..., D/2, d/2)
        
        self.site_configs = [] # Stores (output_dim, list_of_factor_shapes) for each site
        
        for tensor_data in ftn_params_flat:
            shape = tensor_data.shape # (N_b, D/2, ..., d/2)
            n_blocks = shape[0]
            phys_dims = shape[1:]
            
            # Calculate total elements needed for LoRA factors at this site
            # Each factor shape is (N_b, Rank, Dim)
            site_factor_shapes = []
            total_site_params = 0
            
            for dim_size in phys_dims:
                f_shape = (n_blocks, lora_rank, dim_size)
                site_factor_shapes.append(f_shape)
                total_site_params += int(torch.prod(torch.tensor(f_shape)).item())
            
            self.site_configs.append({
                "output_dim": total_site_params,
                "fshapes": site_factor_shapes
            })

        # 3. Create N Independent Networks
        self.site_networks = nn.ModuleList()
        for i in range(n_sites):
            n_neighbors = len(self.rf_indices[i])
            output_dim = self.site_configs[i]["output_dim"]
            
            net = LocalSiteNetwork(
                n_neighbors=n_neighbors,
                num_classes=tn.phys_dim(), # e.g. 2 for Spin-1/2, 4 for Fermions
                embed_dim=embed_dim,
                attention_heads=attn_heads,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dtype=dtype
            )
            self.site_networks.append(net)
    
    def initialize_output_scale(self, scale):
        print(f" -> [Init] LocalClusterBackflow: Clamping output weights to scale {scale}")
        for net in self.site_networks:
            # The last layer of the MLP is at index 2
            last_layer = net.mlp[-1]
            torch.nn.init.normal_(last_layer.weight, mean=0.0, std=scale)
            if last_layer.bias is not None:
                torch.nn.init.zeros_(last_layer.bias)
                    

    def forward(self, x):
        # x: (Batch, N_sites) Int/Long
        nn_outputs = []
        # Loop over each site network
        for i, net in enumerate(self.site_networks):
            # 1. Get local neighborhood
            neighbors = self.rf_indices[i]
            x_local = x[:, neighbors] # (Batch, n_neighbors)
            
            # 2. Run local network -> (Batch, Total_Site_Params)
            site_output_flat = net(x_local)
            
            # 3. Split and Reshape into individual factors
            cursor = 0
            site_fshapes = self.site_configs[i]["fshapes"]
            site_ts_shapes = self.ftn_params_shape[i] # (N_b, D/2, ..., d/2)
            
            bfactors = []
            
            for shape in site_fshapes:
                num_el = int(torch.prod(torch.tensor(shape)).item()) # N_b * Rank * Dim
                bfactor = site_output_flat[:, cursor : cursor+num_el].view(-1, *shape) # (Batch, N_b, LoRA_Rank, Dim)
                bfactors.append(bfactor)
                cursor += num_el
            
            # einsum logic to reshape bfactor to match the corresponding tensor leg
            n_factorized_dims = len(site_ts_shapes) - 1
            input_subs = [f'BSR{chr(97+i)}' for i in range(n_factorized_dims)] # e.g. BSRa, BSRb, ...
            output_sub = 'BS' + ''.join([chr(97+i) for i in range(n_factorized_dims)]) # e.g. Bsab..
            eq = f"{','.join(input_subs)}->{output_sub}"
            delta_t = torch.einsum(eq, *bfactors) # (Batch, N_b, D/2, D/2, ..., d/2)
            # flatten to (Batch, Total_Site_Params) for addition to the original tensor vector
            delta_t_flat = delta_t.view(delta_t.size(0), -1) # (Batch, Total_Site_Params)
            nn_outputs.append(delta_t_flat)
        
        nn_outputs = torch.cat(nn_outputs, dim=1) # (Batch, ftn_params_length)
            
        return nn_outputs # shape (Batch, ftn_params_length)

# ==============================================================================
# Main Model Integration
# ==============================================================================

class LoRA_NNfPEPS_Model_Cluster(BasefPEPSBackflowModel):
    
    """
    Subclass B: Local Cluster Backflow.
    Totally independent neural networks per site. No global attention.
    Input for each site is determined strictly by its receptive field.
    """
    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        nn_hidden_dim,
        embed_dim,
        attn_heads,
        radius=1,
        lora_rank=4,
        init_perturbation_scale=1e-5,
        dtype=torch.float64,
        **kwargs,
    ):
        # 1. Call Base Init
        super().__init__(
            tn,
            max_bond,
            nn_eta,
            dtype,
            kwargs.get("debug_file"),
            contract_boundary_opts=kwargs.get("contract_boundary_opts", {}),
        )
        
        self.lora_rank = lora_rank

        # 2. Define NN Architecture (Local & Independent)
        self.nn_backflow_generator = LoRA_LocalClusterGenerator(
            tn=tn,
            radius=radius,
            embed_dim=embed_dim,
            attn_heads=attn_heads,
            hidden_dim=nn_hidden_dim,
            dtype=self.dtype,
            lora_rank=self.lora_rank,
        )
        
        # Direct assignment (no global attn prepended)
        self.nn_backflow = self.nn_backflow_generator

        # 3. Finalize initialization
        self.finish_initialization(init_perturbation_scale)