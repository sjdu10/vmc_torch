"""Attention-based NN-fTNS models for GPU VMC.

Architecture:
    Config (B, N_sites) int64
      -> Per-site embedding: (B, N_sites, d_model)
      -> Self-attention blocks x L  (pre-norm, shared, optional local mask)
      -> Final LayerNorm
      -> Cast to output dtype
      -> Geometry-aware output heads (shared per type)
      -> Permute to site order: (B, total_ftn_params)

Key design decisions vs old fTN_BFA_cluster_Model (CPU):
  - Shared attention blocks across all sites (not per-site)
  - Shared output heads per geometry type (not per-site)
  - Optional local attention mask (radius param) for future bMPS reuse
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .NNfTNS import NNfTNS_Model_GPU


class _SelfAttentionBlock(nn.Module):
    """Pre-norm transformer encoder block.

    Pre-norm attention: LN -> QKV -> SDPA (with mask) -> out proj -> +residual
    Pre-norm FFN: LN -> Linear -> GELU -> Linear -> +residual

    Args:
        d_model: embedding dimension
        n_heads: number of attention heads
        dim_feedforward: FFN hidden dimension
        dtype: parameter dtype
    """

    def __init__(self, d_model, n_heads, dim_feedforward, dtype):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model={d_model} not divisible by n_heads={n_heads}"
        )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Pre-norm attention
        self.norm1 = nn.LayerNorm(d_model, dtype=dtype)
        self.qkv_proj = nn.Linear(
            d_model, 3 * d_model, dtype=dtype,
        )
        self.attn_out_proj = nn.Linear(
            d_model, d_model, dtype=dtype,
        )

        # Pre-norm FFN
        self.norm2 = nn.LayerNorm(d_model, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, dtype=dtype),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model, dtype=dtype),
        )

    def forward(self, h, attn_mask=None):
        """
        Args:
            h: (B, S, d_model)
            attn_mask: optional (1, 1, S, S) float additive mask
                (0 = allow, -inf = block). Float additive format
                for vmap compatibility (SDPA's bool mask fails
                under vmap).

        Returns:
            (B, S, d_model)
        """
        B, S, _ = h.shape

        # --- Pre-norm attention + residual ---
        h_norm = self.norm1(h)
        qkv = self.qkv_proj(h_norm)  # (B, S, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, S, head_dim)
        q = q.reshape(B, S, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.reshape(B, S, self.n_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.reshape(B, S, self.n_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)

        if attn_mask is None:
            attn_out = F.scaled_dot_product_attention(q, k, v)
        else:
            # Manual attention with additive mask for vmap compat
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores = scores + attn_mask  # broadcasts (1,1,S,S)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)

        # (B, n_heads, S, head_dim) -> (B, S, d_model)
        attn_out = (
            attn_out.permute(0, 2, 1, 3)
            .reshape(B, S, -1)
        )
        h = h + self.attn_out_proj(attn_out)

        # --- Pre-norm FFN + residual ---
        h = h + self.ffn(self.norm2(h))
        return h


class _Attention_Geometric_Backflow_GPU(nn.Module):
    """Self-attention backflow generator with geometry-aware output heads.

    Same interface as _CNN_Geometric_Backflow_GPU:
        Input:  (B, N_sites) int64 configuration
        Output: (B, total_ftn_params) concatenated delta vector

    Uses shared attention blocks (not per-site) and shared output heads
    per geometry type (corner/edge/bulk).

    Args:
        tn: quimb fPEPS tensor network
        d_model: attention embedding dimension
        n_heads: number of attention heads
        n_layers: number of transformer blocks
        dim_feedforward: FFN hidden dimension
        radius: local attention radius (Chebyshev distance).
            None = global attention, int = local.
        head_hidden_dim: hidden dim for output heads (default None
            = single Linear). When set, heads become
            Linear(d_model, hidden) -> GELU -> Linear(hidden, param_size).
        dtype: output head precision (float64)
        backbone_dtype: attention backbone precision (float32 for speed)
    """

    def __init__(
        self,
        tn,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        radius=None,
        head_hidden_dim=None,
        dtype=torch.float64,
        backbone_dtype=None,
    ):
        import quimb as qu
        import quimb.tensor as qtn

        super().__init__()
        self.dtype = dtype
        self.backbone_dtype = backbone_dtype if backbone_dtype else dtype
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        self.n_sites = self.Lx * self.Ly
        self.nn_radius = radius

        bb_dt = self.backbone_dtype

        # --- 1. Analyze geometry and group sites ---
        # (copied from _CNN_Geometric_Backflow_GPU)
        ftn_params, _ = qtn.pack(tn)
        ftn_params_flat, _ = qu.utils.tree_flatten(
            ftn_params, get_ref=True,
        )

        # Get 2D coordinates from TN sites (not hardcoded divmod)
        sites = list(tn.sites)
        self._site_coords = [(s[0], s[1]) for s in sites]

        groups = {}          # {g_type: [site_idx, ...]}
        group_output_dims = {}  # {g_type: int}

        for i in range(self.n_sites):
            x, y = self._site_coords[i]
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

        group_names = list(groups.keys())
        self._n_groups = len(group_names)

        # Register gather indices as buffers
        self._gather_buf_names = []
        for k, name in enumerate(group_names):
            buf_name = f'_gather_{k}'
            self._gather_buf_names.append(buf_name)
            self.register_buffer(
                buf_name,
                torch.tensor(groups[name], dtype=torch.long),
            )

        # Build permutation: group-order -> site-order
        site_to_group_offset = {}
        group_offset = 0
        for name in group_names:
            out_dim = group_output_dims[name]
            for local_i, site_idx in enumerate(groups[name]):
                start = group_offset + local_i * out_dim
                site_to_group_offset[site_idx] = (start, out_dim)
            group_offset += len(groups[name]) * out_dim

        site_order_idx = []
        for site_idx in range(self.n_sites):
            start, out_dim = site_to_group_offset[site_idx]
            site_order_idx.extend(range(start, start + out_dim))

        self.register_buffer(
            '_site_order_idx',
            torch.tensor(site_order_idx, dtype=torch.long),
        )

        # --- 2. Per-site embedding ---
        # (N_sites, phys_dim, d_model) — each site has its own
        # learned linear map from one-hot config to d_model.
        # Position info encoded implicitly.
        phys_dim = tn.phys_dim()
        self.phys_dim = phys_dim
        self.site_embed = nn.Parameter(
            torch.randn(
                self.n_sites, phys_dim, d_model, dtype=bb_dt,
            ) * 0.02,
        )

        # --- 3. Optional local attention mask ---
        if radius is not None:
            # Additive float mask: 0 = allow, -inf = block.
            # Float additive format instead of bool for vmap compat
            # (SDPA's bool attn_mask fails under vmap).
            mask = torch.zeros(
                self.n_sites, self.n_sites, dtype=bb_dt,
            )
            for i in range(self.n_sites):
                xi, yi = self._site_coords[i]
                for j in range(self.n_sites):
                    xj, yj = self._site_coords[j]
                    if not (
                        abs(xi - xj) <= radius
                        and abs(yi - yj) <= radius
                    ):
                        mask[i, j] = float('-inf')  # block
            # Shape (1, 1, S, S) to broadcast over (B, n_heads, S, S)
            self.register_buffer(
                '_attn_mask', mask.unsqueeze(0).unsqueeze(0),
            )
        else:
            self._attn_mask = None

        # --- 4. Shared attention blocks ---
        self.attn_blocks = nn.ModuleList([
            _SelfAttentionBlock(
                d_model, n_heads, dim_feedforward, dtype=bb_dt,
            )
            for _ in range(n_layers)
        ])

        # --- 5. Final LayerNorm ---
        self.final_norm = nn.LayerNorm(d_model, dtype=bb_dt)

        # --- 6. Per-geometry output heads (in dtype for TN precision) ---
        # Optional hidden layer: d_model -> hidden -> GELU -> param_size
        # Without: d_model -> param_size (single linear)
        self.heads = nn.ModuleList()
        for name in group_names:
            out_dim = group_output_dims[name]
            if head_hidden_dim is not None:
                self.heads.append(nn.Sequential(
                    nn.Linear(
                        d_model, head_hidden_dim, dtype=dtype,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        head_hidden_dim, out_dim, dtype=dtype,
                    ),
                ))
            else:
                self.heads.append(nn.Linear(
                    d_model, out_dim, dtype=dtype,
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
        """Small-init output heads so model starts near pure fPEPS.

        For Sequential heads (with hidden layer), only the last
        Linear gets small init — earlier layers use default init
        so gradients flow through.
        """
        for head in self.heads:
            if isinstance(head, nn.Sequential):
                # Small-init only the last Linear
                last_linear = head[-1]
                nn.init.normal_(
                    last_linear.weight, mean=0.0, std=scale,
                )
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)
            else:
                nn.init.normal_(
                    head.weight, mean=0.0, std=scale,
                )
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

        # 1. Per-site embedding via one-hot + einsum
        # Use identity indexing instead of F.one_hot to avoid
        # scatter_ which is not vmap-compatible
        eye = torch.eye(
            self.phys_dim,
            dtype=self.backbone_dtype,
            device=x.device,
        )
        one_hot = eye[x]  # (B, N, P)
        h = torch.einsum(
            'bnp, npd -> bnd', one_hot, self.site_embed,
        )  # (B, N, d_model)

        # 2. Self-attention blocks (with optional local mask)
        attn_mask = self._attn_mask  # None or (N, N) bool
        for block in self.attn_blocks:
            h = block(h, attn_mask=attn_mask)

        # 3. Final LayerNorm
        h = self.final_norm(h)  # (B, N_sites, d_model)

        # 4. Cast to output dtype (no-op if backbone_dtype == dtype)
        h = h.to(self.dtype)

        # 5. Per-geometry heads (compile-friendly: fixed-length list)
        group_outputs = []
        for buf_name, head in zip(
            self._gather_buf_names, self.heads,
        ):
            gather_idx = getattr(self, buf_name)
            # (B, N_group, d_model)
            group_feats = h.index_select(1, gather_idx)
            # (B, N_group, out_dim) -> (B, N_group * out_dim)
            group_outputs.append(head(group_feats).flatten(1))

        # 6. Concat in group order, permute to site order
        out_group_order = torch.cat(group_outputs, dim=1)
        return out_group_order[:, self._site_order_idx]


class Attention_Geometric_fPEPS_GPU(NNfTNS_Model_GPU):
    """NN-fPEPS with self-attention geometric backflow for GPU VMC.

    Combines a shared self-attention backbone with geometry-aware
    output heads and fPEPS tensor network contraction.

    Args:
        tn: quimb fPEPS tensor network
        max_bond: boundary contraction bond dimension (chi)
        nn_eta: scale factor for NN backflow correction
        d_model: attention embedding dimension (default 64)
        n_heads: number of attention heads (default 4)
        n_layers: number of transformer blocks (default 2)
        dim_feedforward: FFN hidden dimension (default 256)
        radius: local attention radius (None=global, int=local)
        head_hidden_dim: hidden dim for output heads (default None
            = single Linear). When set, heads become
            Linear(d_model, hidden) -> GELU -> Linear(hidden, param_size).
        init_scale: initial scale for output heads (default 1e-5)
        dtype: parameter dtype (default float64)
        backbone_dtype: dtype for attention backbone (default None)
        contract_boundary_opts: options for boundary contraction
    """

    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        radius=None,
        head_hidden_dim=None,
        init_scale=1e-5,
        dtype=torch.float64,
        backbone_dtype=None,
        contract_boundary_opts=None,
    ):
        # 1. Create attention backflow module
        nn_module = _Attention_Geometric_Backflow_GPU(
            tn=tn,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            radius=radius,
            head_hidden_dim=head_hidden_dim,
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

        # 4. Store radius info for future bMPS reuse
        self.nn_radius = radius
        self.effective_radius = (
            n_layers * radius if radius is not None else None
        )
