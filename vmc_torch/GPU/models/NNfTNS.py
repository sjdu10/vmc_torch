"""NN-fTNS models for GPU VMC.

Architecture (two-level evaluation):
    1. NN (batch mode): NN module runs on full batch (B, N_sites) via
       functional_call -> (B, total_ftn_params) backflow corrections.
    2. TN (vmap per-sample): Each sample's corrected TN params are
       unpacked into a quimb fPEPS, isel'd, contracted.
       Vmapped over the batch dim.

Base class NNfTNS_Model_GPU handles all plumbing (param registration,
functional_call, vmap wrappers, export+compile). Subclasses only need
to provide:
    - An NN module (nn.Module: (B, N_sites) -> (B, total_ftn_params))
    - Optionally override _contract_tn / _contract_tn_log for custom
      contraction schemes.
"""
import torch
import torch.nn as nn

from ._base import WavefunctionModel_GPU


# ================================================================
#  CNN Geometric Backflow Generator (GPU port)
# ================================================================


class _CNN_Geometric_Backflow_GPU(nn.Module):
    """Multi-head CNN generator with geometry-aware output heads.

    Classifies lattice sites into up to 9 geometric types
    (4 corners, 4 edges, 1 bulk) and uses a shared CNN backbone
    with per-type Linear output heads.

    Input:  (B, N_sites) int64 configuration
    Output: (B, total_ftn_params) concatenated delta vector

    Ported from vmap/models/ConvNNfTNS.py::CNN_Geometric_Generator.
    """

    def __init__(
        self,
        tn,
        embed_dim,
        hidden_dim,
        kernel_size=3,
        layers=2,
        dtype=torch.float64,
        backbone_dtype=None,
    ):
        import quimb as qu
        import quimb.tensor as qtn

        super().__init__()
        self.dtype = dtype
        # backbone_dtype controls embedding + Conv2d precision.
        # Output heads always use `dtype` so the final output
        # matches TN precision.  Set to torch.float32 for speed.
        self.backbone_dtype = backbone_dtype if backbone_dtype else dtype
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        self.n_sites = self.Lx * self.Ly

        # --- 1. Analyze geometry and group sites ---
        ftn_params, _ = qtn.pack(tn)
        ftn_params_flat, _ = qu.utils.tree_flatten(
            ftn_params, get_ref=True,
        )

        # Collect groups in deterministic order
        groups = {}          # {g_type: [site_idx, ...]}
        group_output_dims = {}  # {g_type: int}

        for i in range(self.n_sites):
            x, y = divmod(i, self.Ly)
            g_type = self._get_geometric_type(x, y)

            if g_type not in groups:
                groups[g_type] = []
                p = ftn_params_flat[i]
                group_output_dims[g_type] = (
                    p.numel()
                    if isinstance(p, torch.Tensor)
                    else p.size
                )
            groups[g_type].append(i)

        # Fixed group ordering (for compile-friendly iteration)
        group_names = list(groups.keys())
        self._n_groups = len(group_names)

        # Register gather indices as buffers (auto .to(device))
        self._gather_buf_names = []
        for k, name in enumerate(group_names):
            buf_name = f'_gather_{k}'
            self._gather_buf_names.append(buf_name)
            self.register_buffer(
                buf_name,
                torch.tensor(groups[name], dtype=torch.long),
            )

        # Build permutation: group-order concat -> site-order concat
        # Group-order: [grp0_site0_params, grp0_site1_params, ...,
        #               grp1_site0_params, ...]
        # Site-order:  site 0, site 1, ..., site N-1
        site_order_idx = []
        group_offset = 0
        # Map site_idx -> (offset_in_group_concat)
        site_to_group_offset = {}
        for name in group_names:
            out_dim = group_output_dims[name]
            for local_i, site_idx in enumerate(groups[name]):
                start = group_offset + local_i * out_dim
                site_to_group_offset[site_idx] = (start, out_dim)
            group_offset += len(groups[name]) * out_dim

        # Build the permutation: for each position j in site-order,
        # which position in group-order provides it?
        for site_idx in range(self.n_sites):
            start, out_dim = site_to_group_offset[site_idx]
            site_order_idx.extend(range(start, start + out_dim))

        self.register_buffer(
            '_site_order_idx',
            torch.tensor(site_order_idx, dtype=torch.long),
        )

        # --- 2. Network architecture ---
        bb_dt = self.backbone_dtype

        # A. Shared embedding + coordinate grid
        self.embedding = nn.Embedding(
            tn.phys_dim(), embed_dim,
        )
        x_grid = (
            torch.linspace(-1, 1, self.Lx)
            .view(1, 1, self.Lx, 1)
            .expand(1, 1, self.Lx, self.Ly)
        )
        y_grid = (
            torch.linspace(-1, 1, self.Ly)
            .view(1, 1, 1, self.Ly)
            .expand(1, 1, self.Lx, self.Ly)
        )
        self.register_buffer(
            'coord_grid',
            torch.cat([x_grid, y_grid], dim=1).to(bb_dt),
        )

        # B. Shared CNN backbone (in backbone_dtype for speed)
        in_channels = embed_dim + 2
        cnn_layers = []
        cnn_layers.append(nn.Conv2d(
            in_channels, hidden_dim, kernel_size,
            padding='same', dtype=bb_dt,
        ))
        cnn_layers.append(nn.GELU())
        for _ in range(layers - 1):
            cnn_layers.append(nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size,
                padding='same', dtype=bb_dt,
            ))
            cnn_layers.append(nn.GELU())
        self.backbone = nn.Sequential(*cnn_layers)

        # C. Per-geometry output heads (in dtype for TN precision).
        #    float32 backbone features get promoted automatically
        #    by the float64 Linear matmul.
        self.heads = nn.ModuleList()
        for name in group_names:
            self.heads.append(nn.Linear(
                hidden_dim, group_output_dims[name], dtype=dtype,
            ))

    def _get_geometric_type(self, x, y):
        """Classify site (x, y) into one of 9 geometric types."""
        is_top = (x == 0)
        is_bottom = (x == self.Lx - 1)
        is_left = (y == 0)
        is_right = (y == self.Ly - 1)

        if is_top and is_left:
            return "CORNER_TL"
        if is_top and is_right:
            return "CORNER_TR"
        if is_bottom and is_left:
            return "CORNER_BL"
        if is_bottom and is_right:
            return "CORNER_BR"
        if is_top:
            return "EDGE_TOP"
        if is_bottom:
            return "EDGE_BOTTOM"
        if is_left:
            return "EDGE_LEFT"
        if is_right:
            return "EDGE_RIGHT"
        return "BULK"

    def initialize_output_scale(self, scale):
        """Small-init output heads so model starts near pure fPEPS."""
        for head in self.heads:
            nn.init.normal_(head.weight, mean=0.0, std=scale)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(self, x):
        """
        Args:
            x: (B, N_sites) int64

        Returns:
            (B, total_ftn_params) float — concatenated backflow delta
        """
        B = x.shape[0]

        # 1. Embed + coordinate features
        x_grid = x.view(B, self.Lx, self.Ly)
        h = (
            self.embedding(x_grid)
            .to(self.backbone_dtype)
            .permute(0, 3, 1, 2)
        )  # (B, embed_dim, Lx, Ly)
        coords = self.coord_grid.expand(B, -1, -1, -1)
        h_in = torch.cat([h, coords], dim=1)

        # 2. Shared backbone -> (B, hidden_dim, Lx, Ly)
        features = self.backbone(h_in)

        # Flatten spatial: (B, N_sites, hidden_dim)
        features_flat = features.flatten(2).permute(0, 2, 1)

        # 3. Cast to output dtype (no-op when backbone_dtype == dtype)
        features_flat = features_flat.to(self.dtype)

        # 4. Per-geometry heads (compile-friendly: fixed-length list)
        group_outputs = []
        for buf_name, head in zip(
            self._gather_buf_names, self.heads,
        ):
            gather_idx = getattr(self, buf_name)
            # (B, N_group, hidden_dim)
            group_feats = features_flat.index_select(1, gather_idx)
            # (B, N_group, out_dim) -> (B, N_group * out_dim)
            group_outputs.append(head(group_feats).flatten(1))

        # 5. Concat in group order, permute to site order
        out_group_order = torch.cat(group_outputs, dim=1)
        return out_group_order[:, self._site_order_idx]


# ================================================================
#  NNfTNS_Model_GPU base class
# ================================================================


class NNfTNS_Model_GPU(WavefunctionModel_GPU):
    """Base class for NN-fTNS models on GPU.

    Handles all plumbing: TN param packing, NN param extraction,
    functional_call, vmap wrappers, export+compile.

    Subclasses provide:
        - An nn_module (nn.Module: (B, N_sites) -> (B, total_ftn_params))
        - Optionally override _contract_tn / _contract_tn_log

    Parameter layout in self.params (ParameterList):
        [0 .. n_ftn-1]   TN parameters (flattened quimb pytree)
        [n_ftn .. end]    NN parameters (from NN module)

    Args:
        tn: quimb tensor network
        nn_module: nn.Module producing backflow corrections
        max_bond: boundary contraction bond dimension (chi)
        nn_eta: scale factor for NN backflow correction
        dtype: parameter dtype (default float64)
        contract_boundary_opts: options for boundary contraction
    """

    def __init__(
        self,
        tn,
        nn_module,
        max_bond,
        nn_eta,
        dtype=torch.float64,
        contract_boundary_opts=None,
    ):
        import quimb as qu
        import quimb.tensor as qtn

        if contract_boundary_opts is None:
            contract_boundary_opts = {}

        if tn.tensors[0].data.indices[-1]._linearmap is not None:
            for ts in tn.tensors:
                ts_data = ts.data
                ts_data.indices[-1]._linearmap = None
                ts.modify(data=ts_data)
            self._loc_basis_perm = torch.tensor(
                [0, 2, 3, 1], dtype=torch.long
            )
        else:
            self._loc_basis_perm = None

        self.dtype = dtype
        self.chi = max_bond
        self.nn_eta = nn_eta
        self.contract_boundary_opts = contract_boundary_opts

        # --- 1. TN parameter setup ---
        params, skeleton = qtn.pack(tn)
        self.skeleton = skeleton

        # Flatten pytree into a list
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(
            params, get_ref=True,
        )
        self.ftn_params_pytree = ftn_params_pytree

        ftn_params_tensors = [
            torch.as_tensor(x, dtype=self.dtype)
            for x in ftn_params_flat
        ]
        self.n_ftn = len(ftn_params_tensors)

        # Precompute shapes/sizes as Python ints (vmap-safe)
        self.ftn_params_shapes = [p.shape for p in ftn_params_tensors]
        self.ftn_params_sizes = [
            int(p.numel()) for p in ftn_params_tensors
        ]
        self.ftn_params_length = sum(self.ftn_params_sizes)

        # --- 2. NN param extraction ---
        nn_param_names = []
        nn_param_dtypes = []
        nn_param_tensors = []
        for name, p in nn_module.named_parameters():
            nn_param_names.append(name)
            nn_param_dtypes.append(p.dtype)
            nn_param_tensors.append(p.data.clone())
        self._nn_param_names = nn_param_names
        self._nn_param_dtypes = nn_param_dtypes

        # --- 3. Register all params: [TN..., NN...] ---
        all_tensors = ftn_params_tensors + nn_param_tensors
        super().__init__(params_list=all_tensors)

        # Hide NN module from nn.Module child scanning
        # (prevents double parameter registration)
        self._nn_container = [nn_module]

        # Pre-vmap the TN contraction (overrides base class default)
        self._vmapped_tn_contraction = torch.vmap(
            self._tn_contraction,
            in_dims=(0, None, 0),
            randomness='different',
        )
        self._vmapped_tn_contraction_log = torch.vmap(
            self._tn_contraction_log,
            in_dims=(0, None, 0),
            randomness='different',
        )

    # ----- Backflow application (shared) -----

    def _apply_backflow(self, x, ftn_params_list, nn_output):
        """Apply additive backflow to TN params and return isel'd TN.

        Args:
            x:              (N_sites,) int64 — one configuration
            ftn_params_list: list of TN parameter tensors
            nn_output:      (total_ftn_params,) — single-sample
                            backflow correction vector

        Returns:
            isel'd quimb TN (ready for contraction)
        """
        import quimb as qu
        import quimb.tensor as qtn

        # 1. Flatten TN params to vector
        ftn_vector = torch.cat([
            p.reshape(-1) for p in ftn_params_list
        ])

        # 2. Add backflow correction
        corrected_vector = ftn_vector + self.nn_eta * nn_output

        # 3. Split back into per-site tensors
        corrected_params = []
        pointer = 0
        for size, shape in zip(
            self.ftn_params_sizes, self.ftn_params_shapes,
        ):
            corrected_params.append(
                corrected_vector[pointer:pointer + size].view(shape)
            )
            pointer += size

        # 4. Reconstruct quimb TN and isel
        params_pytree = qu.utils.tree_unflatten(
            corrected_params, self.ftn_params_pytree,
        )
        tn = qtn.unpack(params_pytree, self.skeleton)

        if self._loc_basis_perm is not None:
            x = self._loc_basis_perm[x]
        amp_tn = tn.isel({
            tn.site_ind(site): x[i]
            for i, site in enumerate(tn.sites)
        })

        return amp_tn

    # ----- Contraction (overridable) -----

    def _contract_tn(self, amp_tn):
        """Contract an isel'd TN to a scalar amplitude.

        Default: boundary contraction from xmin/xmax + contract().
        Override for custom contraction schemes.

        Args:
            amp_tn: isel'd quimb TN

        Returns:
            scalar amplitude
        """
        if self.chi > 0:
            amp_tn.contract_boundary_from_xmin_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[0, amp_tn.Lx // 2 - 1],
                **self.contract_boundary_opts,
            )
            amp_tn.contract_boundary_from_xmax_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[amp_tn.Lx // 2, amp_tn.Lx - 1],
                **self.contract_boundary_opts,
            )
        return amp_tn.contract()

    def _contract_tn_log(self, amp_tn):
        """Contract an isel'd TN to (sign, log_abs) scalars.

        Default: boundary contraction from xmin/xmax +
        contract(strip_exponent=True).
        Override for custom contraction schemes.

        Args:
            amp_tn: isel'd quimb TN

        Returns:
            (sign, log_abs) scalars
        """
        if self.chi > 0:
            amp_tn.contract_boundary_from_xmin_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[0, amp_tn.Lx // 2 - 1],
                **self.contract_boundary_opts,
            )
            amp_tn.contract_boundary_from_xmax_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[amp_tn.Lx // 2, amp_tn.Lx - 1],
                **self.contract_boundary_opts,
            )
        sign, exponent_10 = amp_tn.contract(strip_exponent=True)
        log_abs = exponent_10 * torch.log(torch.tensor(10.0))
        return sign, log_abs

    # ----- TN contraction entry points (compose backflow + contract) ---

    def _tn_contraction(self, x, ftn_params_list, nn_output):
        """Single-sample TN contraction with additive backflow.

        This is the function that gets vmapped in vamp().

        Args:
            x:              (N_sites,) int64 — one configuration
            ftn_params_list: list of TN parameter tensors
            nn_output:      (total_ftn_params,) — single-sample
                            backflow correction vector

        Returns:
            scalar amplitude
        """
        amp_tn = self._apply_backflow(x, ftn_params_list, nn_output)
        return self._contract_tn(amp_tn)

    def _tn_contraction_log(self, x, ftn_params_list, nn_output):
        """Single-sample log TN contraction with additive backflow.

        Returns:
            (sign, log_abs) scalars
        """
        amp_tn = self._apply_backflow(x, ftn_params_list, nn_output)
        return self._contract_tn_log(amp_tn)

    # ----- NN helpers -----

    def _make_nn_params_dict(self, nn_params):
        """Build parameter dict for functional_call, casting to the
        NN module's expected dtypes (handles mixed-precision backbone).
        No-op when backbone_dtype == dtype.
        """
        return {
            name: p.to(dt) if p.dtype != dt else p
            for name, p, dt in zip(
                self._nn_param_names,
                nn_params,
                self._nn_param_dtypes,
            )
        }

    def _run_nn_batch(self, x, params):
        """Split params, run NN in batch mode, return (ftn_params, nn_outputs).

        Shared helper for vamp() and vamp_log().
        """
        if isinstance(params, nn.ParameterList):
            params = list(params)

        ftn_params = params[:self.n_ftn]
        nn_params = params[self.n_ftn:]

        nn_params_dict = self._make_nn_params_dict(nn_params)
        nn_module = self._nn_container[0]
        batch_nn_outputs = torch.func.functional_call(
            nn_module, nn_params_dict, (x,),
        )  # (B, total_ftn_params)

        return ftn_params, batch_nn_outputs

    # ----- Public interface -----

    def amplitude(self, x, params_list):
        """Single-sample amplitude (for export_and_compile compat).

        Runs NN on a single sample (unsqueeze/squeeze), then
        does TN contraction with backflow.

        Args:
            x:           (N_sites,) int64 — one configuration
            params_list: list of all parameter tensors
                         [TN params..., NN params...]

        Returns:
            scalar amplitude
        """
        ftn_params = params_list[:self.n_ftn]
        nn_params = params_list[self.n_ftn:]

        # Run NN on single sample via functional_call
        nn_params_dict = self._make_nn_params_dict(nn_params)
        nn_module = self._nn_container[0]
        x_batch = x.unsqueeze(0)  # (1, N_sites)
        nn_output_batch = torch.func.functional_call(
            nn_module, nn_params_dict, (x_batch,),
        )  # (1, total_ftn_params)
        nn_output = nn_output_batch.squeeze(0)  # (total_ftn_params,)

        return self._tn_contraction(x, ftn_params, nn_output)

    def log_amplitude(self, x, params_list):
        """Single-sample log-amplitude with backflow.

        Same structure as amplitude() but uses _tn_contraction_log.
        """
        ftn_params = params_list[:self.n_ftn]
        nn_params = params_list[self.n_ftn:]

        nn_params_dict = self._make_nn_params_dict(nn_params)
        nn_module = self._nn_container[0]
        x_batch = x.unsqueeze(0)
        nn_output_batch = torch.func.functional_call(
            nn_module, nn_params_dict, (x_batch,),
        )
        nn_output = nn_output_batch.squeeze(0)

        return self._tn_contraction_log(x, ftn_params, nn_output)

    def vamp(self, x, params):
        """Batched amplitude: NN in batch mode + vmap TN contraction.

        Two-level evaluation:
        1. functional_call runs NN on full batch -> (B, total_ftn_params)
        2. vmap runs TN contraction per sample with additive backflow
        """
        ftn_params, batch_nn_outputs = self._run_nn_batch(x, params)
        return self._vmapped_tn_contraction(
            x, ftn_params, batch_nn_outputs,
        )

    def vamp_log(self, x, params):
        """Batched log-amplitude: NN in batch mode + vmap log TN contraction.

        Same two-level pattern as vamp() but uses _tn_contraction_log.
        """
        ftn_params, batch_nn_outputs = self._run_nn_batch(x, params)
        return self._vmapped_tn_contraction_log(
            x, ftn_params, batch_nn_outputs,
        )

    def _tn_contraction_for_export(self, x, nn_output, *ftn_params):
        """Wrapper for torch.export: TN contraction with flat *args.

        Single-sample: x is (N_sites,), nn_output is
        (total_ftn_params,), returns scalar.
        """
        return self._tn_contraction(x, list(ftn_params), nn_output)

    def _tn_contraction_log_for_export(
        self, x, nn_output, *ftn_params,
    ):
        """Wrapper for torch.export: log TN contraction.

        Single-sample: returns (sign, log_abs) scalars.
        """
        return self._tn_contraction_log(
            x, list(ftn_params), nn_output,
        )

    def export_and_compile(
        self, example_x, mode='default',
        use_log_amp=False, **compile_kwargs,
    ):
        """Export TN contraction + compile combined NN+TN forward.

        Strategy:
        1. Export only the TN contraction (quimb/symmray ops) via
           torch.export -> pure aten-ops FX graph.
        2. Vmap the exported TN over the batch dim.
        3. Build a combined forward: NN (batch) + exported TN (vmap).
        4. torch.compile the combined function.

        The NN uses standard PyTorch ops that dynamo traces natively,
        so it does NOT need export.

        Call AFTER .to(device).

        Args:
            example_x: single-sample config (N_sites,) on device.
            mode: torch.compile mode.
            use_log_amp: if True, export log TN contraction.
        """
        from torch.export import export

        device = example_x.device
        ftn_params = [p.data for p in self.params[:self.n_ftn]]

        if use_log_amp:
            tn_export_fn = self._tn_contraction_log_for_export
        else:
            tn_export_fn = self._tn_contraction_for_export

        # --- 1. Generate example nn_output ---
        nn_module = self._nn_container[0]
        with torch.no_grad():
            example_nn_output = nn_module(
                example_x.unsqueeze(0),
            ).squeeze(0)  # (total_ftn_params,)

        # --- 2. Export TN contraction ---
        class _TNModule(nn.Module):
            def __init__(self_, tn_fn):
                super().__init__()
                self_._fn = tn_fn

            def forward(self_, x, nn_output, *flat_ftn_params):
                return self_._fn(x, nn_output, *flat_ftn_params)

        with torch.no_grad():
            exported = export(
                _TNModule(tn_export_fn),
                (example_x, example_nn_output, *ftn_params),
            )
        exported_tn_module = exported.module()

        # Move CPU constants (symmray index tensors) to GPU
        self._exported_tn_module = exported_tn_module
        self._move_exported_tn_constants_to_device(device)

        # --- 3. Vmap the exported TN ---
        n_ftn = self.n_ftn
        vmapped_exported_tn = torch.vmap(
            exported_tn_module,
            # x batched (0), nn_output batched (0),
            # ftn_params broadcast (None)
            in_dims=(0, 0, *([None] * n_ftn)),
        )

        # --- 4. Build combined forward ---
        nn_param_names = self._nn_param_names
        nn_param_dtypes = self._nn_param_dtypes
        nn_container = self._nn_container

        def _compiled_forward(x, *all_params):
            ftn_ps = all_params[:n_ftn]
            nn_ps = all_params[n_ftn:]

            # NN in batch mode (standard PyTorch)
            nn_params_dict = {
                name: p.to(dt) if p.dtype != dt else p
                for name, p, dt in zip(
                    nn_param_names, nn_ps, nn_param_dtypes,
                )
            }
            batch_nn_outputs = torch.func.functional_call(
                nn_container[0], nn_params_dict, (x,),
            )

            # Exported+vmapped TN contraction
            return vmapped_exported_tn(
                x, batch_nn_outputs, *ftn_ps,
            )

        # --- 5. Compile ---
        self._vmapped_compiled = torch.compile(
            _compiled_forward, mode=mode, **compile_kwargs,
        )
        self._vmapped_exported_fn = _compiled_forward

        self._exported = True
        self._compiled = True
        self._exported_log_amp = use_log_amp

    def export_only(self, example_x, use_log_amp=False):
        """Export + vmap without compile. Useful for debugging."""
        from torch.export import export

        device = example_x.device
        ftn_params = [p.data for p in self.params[:self.n_ftn]]

        if use_log_amp:
            tn_export_fn = self._tn_contraction_log_for_export
        else:
            tn_export_fn = self._tn_contraction_for_export

        nn_module = self._nn_container[0]
        with torch.no_grad():
            example_nn_output = nn_module(
                example_x.unsqueeze(0),
            ).squeeze(0)

        class _TNModule(nn.Module):
            def __init__(self_, tn_fn):
                super().__init__()
                self_._fn = tn_fn

            def forward(self_, x, nn_output, *flat_ftn_params):
                return self_._fn(x, nn_output, *flat_ftn_params)

        with torch.no_grad():
            exported = export(
                _TNModule(tn_export_fn),
                (example_x, example_nn_output, *ftn_params),
            )
        exported_tn_module = exported.module()

        self._exported_tn_module = exported_tn_module
        self._move_exported_tn_constants_to_device(device)

        n_ftn = self.n_ftn
        vmapped_exported_tn = torch.vmap(
            exported_tn_module,
            in_dims=(0, 0, *([None] * n_ftn)),
        )

        nn_param_names = self._nn_param_names
        nn_param_dtypes = self._nn_param_dtypes
        nn_container = self._nn_container

        def _exported_forward(x, *all_params):
            ftn_ps = all_params[:n_ftn]
            nn_ps = all_params[n_ftn:]
            nn_params_dict = {
                name: p.to(dt) if p.dtype != dt else p
                for name, p, dt in zip(
                    nn_param_names, nn_ps, nn_param_dtypes,
                )
            }
            batch_nn_outputs = torch.func.functional_call(
                nn_container[0], nn_params_dict, (x,),
            )
            return vmapped_exported_tn(
                x, batch_nn_outputs, *ftn_ps,
            )

        self._vmapped_exported_fn = _exported_forward
        self._exported = True
        self._exported_log_amp = use_log_amp

    def export_grad(
        self, mode='default', use_log_amp=False,
        do_compile=False, **compile_kwargs,
    ):
        """Build vmap(grad) for NNfTNS models.

        Single-sample function: NN (dynamo-traceable) + exported TN
        (pure aten ops).

        Args:
            do_compile: if True, wrap with torch.compile (long
                warmup). Default False.
        """
        assert self._exported, (
            "Call export_and_compile() before export_grad()"
        )
        exported_tn = self._exported_tn_module
        nn_container = self._nn_container
        nn_param_names = self._nn_param_names
        nn_param_dtypes = self._nn_param_dtypes
        n_ftn = self.n_ftn
        n_total = len(list(self.params))
        argnums = tuple(range(1, n_total + 1))
        in_dims = (0,) + (None,) * n_total

        if use_log_amp:
            def single_fn(x_i, *all_params):
                ftn_ps = all_params[:n_ftn]
                nn_ps = all_params[n_ftn:]
                nn_dict = {
                    name: p.to(dt) if p.dtype != dt else p
                    for name, p, dt in zip(
                        nn_param_names, nn_ps,
                        nn_param_dtypes,
                    )
                }
                nn_out = torch.func.functional_call(
                    nn_container[0], nn_dict,
                    (x_i.unsqueeze(0),),
                ).squeeze(0)
                sign, log_abs = exported_tn(
                    x_i, nn_out, *ftn_ps,
                )
                return log_abs, (sign, log_abs)
        else:
            def single_fn(x_i, *all_params):
                ftn_ps = all_params[:n_ftn]
                nn_ps = all_params[n_ftn:]
                nn_dict = {
                    name: p.to(dt) if p.dtype != dt else p
                    for name, p, dt in zip(
                        nn_param_names, nn_ps,
                        nn_param_dtypes,
                    )
                }
                nn_out = torch.func.functional_call(
                    nn_container[0], nn_dict,
                    (x_i.unsqueeze(0),),
                ).squeeze(0)
                amp = exported_tn(
                    x_i, nn_out, *ftn_ps,
                )
                return amp, amp

        grad_fn = torch.func.grad(
            single_fn, argnums=argnums, has_aux=True,
        )
        vmapped = torch.vmap(grad_fn, in_dims=in_dims)

        if do_compile:
            self._exported_grad_fn = torch.compile(
                vmapped, mode=mode, **compile_kwargs,
            )
        else:
            self._exported_grad_fn = vmapped

        self._grad_exported = True
        self._grad_use_log_amp = use_log_amp

    def _move_exported_tn_constants_to_device(self, device):
        """Move CPU constants in the exported TN graph to GPU.

        torch.export captures symmray's block-sparse index tensors
        as CPU int64 constants. This moves them to the target device.
        """
        gm = self._exported_tn_module
        graph = gm.graph

        for node in graph.nodes:
            if node.op != 'get_attr':
                continue
            parts = node.target.split('.')
            parent = gm
            for p in parts[:-1]:
                parent = getattr(parent, p)
            leaf = parts[-1]
            tensor = getattr(parent, leaf)
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.device.type == 'cpu'
            ):
                setattr(parent, leaf, tensor.to(device))

        for node in graph.nodes:
            if node.op != 'call_function':
                continue
            if '_assert_tensor_metadata' not in str(node.target):
                continue
            kw = dict(node.kwargs)
            if kw.get('device') == torch.device('cpu'):
                kw['device'] = device
                node.kwargs = kw

        graph.lint()
        gm.recompile()

    def forward(self, x):
        """Batched forward: compiled -> exported -> eager."""
        if self._exported and not self._exported_log_amp:
            params_list = list(self.params)
            if self._compiled:
                return self._vmapped_compiled(x, *params_list)
            else:
                return self._vmapped_exported_fn(x, *params_list)
        return self.vamp(x, self.params)

    def forward_log(self, x):
        """Batched log-amplitude: compiled -> exported -> eager."""
        if self._exported and self._exported_log_amp:
            params_list = list(self.params)
            if self._compiled:
                return self._vmapped_compiled(x, *params_list)
            return self._vmapped_exported_fn(x, *params_list)
        return self.vamp_log(x, self.params)


# ================================================================
#  Conv2D Geometric fPEPS Model (GPU) — thin subclass
# ================================================================


class Conv2D_Geometric_fPEPS_GPU(NNfTNS_Model_GPU):
    """NN-fPEPS with Conv2D-Geometric backflow for GPU VMC.

    Combines a CNN geometric backflow network with fPEPS tensor
    network contraction.  The CNN produces per-sample additive
    corrections to the TN parameters, which are then contracted
    via quimb.

    Args:
        tn: quimb fPEPS tensor network
        max_bond: boundary contraction bond dimension (chi)
        nn_eta: scale factor for NN backflow correction
        embed_dim: CNN embedding dimension
        hidden_dim: CNN hidden channels
        kernel_size: CNN kernel size (default 3)
        layers: number of CNN layers (default 2)
        init_scale: initial scale for output heads (default 1e-5)
        dtype: parameter dtype (default float64)
        backbone_dtype: dtype for CNN backbone (default None = same
            as dtype). Set to torch.float32 for faster CNN forward.
        contract_boundary_opts: options for boundary contraction
    """

    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        embed_dim,
        hidden_dim,
        kernel_size=3,
        layers=2,
        init_scale=1e-5,
        dtype=torch.float64,
        backbone_dtype=None,
        contract_boundary_opts=None,
    ):
        # 1. Create CNN backflow module
        nn_module = _CNN_Geometric_Backflow_GPU(
            tn=tn,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            layers=layers,
            dtype=dtype,
            backbone_dtype=backbone_dtype,
        )

        # 2. Small-init output heads
        nn_module.initialize_output_scale(init_scale)

        # 3. Delegate to base class
        super().__init__(
            tn=tn,
            nn_module=nn_module,
            max_bond=max_bond,
            nn_eta=nn_eta,
            dtype=dtype,
            contract_boundary_opts=contract_boundary_opts,
        )
