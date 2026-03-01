"""Neural Network Backflow (NNBF) wavefunction models for GPU VMC.

    psi(x) = det( M(x)[occupied, :] )

where M(x) = M_base + bf_scale * NN(n(x)), i.e. a Slater determinant with
configuration-dependent orbitals produced by a neural network backflow.

Two models are provided:

1. NNBF_GPU — MLP backflow (manual params_list, no nn.Module dependency)
2. AttentionNNBF_GPU — Self-attention backflow (uses torch.func.functional_call
   to integrate standard nn.Module layers with the params_list interface)

For spinful fermions with quimb encoding {0=empty, 1=down, 2=up, 3=updown}:
    1. Convert x to binary occupation: n = [spin_up | spin_dn]  (2*N_sites,)
    2. NN maps n -> delta_M correction
    3. occupied = argsort(n, descending)[:N_f]  (vmap-friendly)
    4. psi = det((M_base + bf_scale * delta_M)[occupied])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import WavefunctionModel_GPU

# Map string names to activation functions (vmap-safe)
ACTIVATIONS = {
    'tanh': torch.tanh,
    'relu': F.relu,
    'gelu': F.gelu,
    'silu': F.silu,
    'sigmoid': torch.sigmoid,
}


class NNBF_GPU(WavefunctionModel_GPU):
    """Neural Network Backflow: psi(x) = det((M + MLP(n))[occupied, :]).

    The MLP takes the binary occupation vector n = [spin_up | spin_dn]
    (length 2*N_sites) and outputs a correction to the base orbital
    matrix M. No embedding layer needed — input is already binary.

    Parameter layout in self.params (ParameterList indices):
        [0]        M_base     (n_orbitals, n_fermions)
        [1+2k]     w_k        (hidden, in_dim)    k=0..n_layers-1
        [2+2k]     b_k        (hidden,)
        [-2]       out_w      (n_orbitals * n_fermions, hidden)
        [-1]       out_b      (n_orbitals * n_fermions,)

    Args:
        n_sites: number of spatial sites
        n_fermions: number of fermions
        hidden_dim: neurons per hidden layer (default 64)
        n_layers: number of hidden layers (default 2)
        activation: activation function name (default 'tanh')
        bf_scale: initial scale for backflow output (default 0.01)
        dtype: parameter dtype (default float64)
    """

    def __init__(
        self,
        n_sites,
        n_fermions,
        hidden_dim=64,
        n_layers=2,
        activation='tanh',
        bf_scale=0.01,
        dtype=torch.float64,
    ):
        self.n_sites = n_sites
        self.n_fermions = n_fermions
        self.n_orbitals = 2 * n_sites  # spin-up + spin-down
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bf_scale = bf_scale

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from {list(ACTIVATIONS.keys())}"
            )
        self.activation_name = activation

        all_tensors = []

        # [0] Base orbital matrix: (n_orbitals, n_fermions)
        M_base = torch.randn(
            self.n_orbitals, n_fermions, dtype=dtype,
        )
        all_tensors.append(M_base)

        # Hidden layers: input is n = [spin_up | spin_dn], dim = 2*N_sites
        in_dim = 2 * n_sites
        for _ in range(n_layers):
            w = torch.empty(hidden_dim, in_dim, dtype=dtype)
            nn.init.kaiming_uniform_(
                w, a=0, mode='fan_in', nonlinearity='tanh',
            )
            b = torch.zeros(hidden_dim, dtype=dtype)
            all_tensors.append(w)
            all_tensors.append(b)
            in_dim = hidden_dim

        # Output layer: hidden -> n_orbitals * n_fermions
        out_dim = self.n_orbitals * n_fermions
        out_w = torch.zeros(out_dim, hidden_dim, dtype=dtype)
        out_b = torch.zeros(out_dim, dtype=dtype)
        all_tensors.append(out_w)
        all_tensors.append(out_b)

        super().__init__(params_list=all_tensors)

    def amplitude(self, x, params_list):
        """Single-sample NNBF evaluation.

        1. Convert x to binary occupation n = [spin_up | spin_dn]
        2. MLP(n) -> delta_M  (n_orbitals, n_fermions)
        3. M(x) = M_base + bf_scale * delta_M
        4. Select occupied rows, compute det.

        Args:
            x:           (N_sites,) int64 — one configuration
            params_list: list of parameter tensors

        Returns:
            scalar determinant value
        """
        act_fn = ACTIVATIONS[self.activation_name]

        # Unpack base orbital matrix
        idx = 0
        M_base = params_list[idx]; idx += 1

        # --- Convert x to binary occupation vector ---
        # quimb encoding: 0=empty, 1=down only, 2=up only, 3=up+down
        spin_up = ((x == 2) | (x == 3)).to(M_base.dtype)  # (N_sites,)
        spin_dn = ((x == 1) | (x == 3)).to(M_base.dtype)  # (N_sites,)
        n = torch.cat([spin_up, spin_dn])  # (2*N_sites,)

        # --- MLP forward on binary input ---
        h = n  # (2*N_sites,)

        for _ in range(self.n_layers):
            w = params_list[idx]; idx += 1
            b = params_list[idx]; idx += 1
            h = act_fn(h @ w.T + b)

        # Output layer -> delta_M
        out_w = params_list[idx]; idx += 1
        out_b = params_list[idx]
        delta_M = (h @ out_w.T + out_b).reshape(
            self.n_orbitals, self.n_fermions,
        )

        # --- Slater determinant with backflow ---
        M = M_base + self.bf_scale * delta_M

        # Select occupied orbitals via argsort (vmap-friendly)
        Nf = self.n_fermions
        occupied = n.argsort(descending=True)[:Nf]  # (N_f,)
        A = M[occupied]  # (N_f, N_f)
        return torch.linalg.det(A)


# ------------------------------------------------------------------ #
#  Attention-based backflow — demonstrates functional_call pattern    #
# ------------------------------------------------------------------ #


class _AttentionLayer(nn.Module):
    """Single self-attention layer: QKV + SDPA + residual + LayerNorm."""

    def __init__(self, d_model, n_heads, dtype=torch.float64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(
            d_model, 3 * d_model, dtype=dtype,
        )
        self.attn_out_proj = nn.Linear(
            d_model, d_model, dtype=dtype,
        )
        self.norm = nn.LayerNorm(d_model, dtype=dtype)

    def forward(self, h):
        """
        Args:
            h: (seq_len, d_model)
        Returns:
            (seq_len, d_model)
        """
        seq_len = h.shape[0]

        qkv = self.qkv_proj(h)  # (seq, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # Multi-head reshape: (seq, d_model) -> (n_heads, seq, head_dim)
        q = q.reshape(seq_len, self.n_heads, self.head_dim)
        q = q.permute(1, 0, 2)
        k = k.reshape(seq_len, self.n_heads, self.head_dim)
        k = k.permute(1, 0, 2)
        v = v.reshape(seq_len, self.n_heads, self.head_dim)
        v = v.permute(1, 0, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        # (n_heads, seq, head_dim) -> (seq, d_model)
        attn_out = attn_out.permute(1, 0, 2).reshape(seq_len, -1)
        attn_out = self.attn_out_proj(attn_out)

        return self.norm(h + attn_out)


class _AttentionBackflow(nn.Module):
    """Multi-layer self-attention backflow built from standard nn modules.

    Each orbital (2*N_sites tokens) attends to every other orbital
    through n_layers attention layers, then projects to n_fermions
    columns — producing the delta_M correction of shape
    (n_orbitals, n_fermions).

    Architecture:
        n (n_orbitals,) binary
        -> Linear(1, d_model)                 input projection
        -> [QKV + SDPA + residual + LN] x L   attention layers
        -> Linear(d_model, n_fermions)         output projection
    """

    def __init__(
        self, n_orbitals, n_fermions, d_model=32, n_heads=4,
        n_layers=1, dtype=torch.float64,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model={d_model} not divisible by n_heads={n_heads}"
        )
        self.input_proj = nn.Linear(1, d_model, dtype=dtype)
        self.layers = nn.ModuleList([
            _AttentionLayer(d_model, n_heads, dtype=dtype)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(
            d_model, n_fermions, dtype=dtype,
        )
        # Small-init output so model starts near plain Slater det
        # but gradients still flow through all layers.
        # (Exact zeros kill gradient for all preceding layers.)
        nn.init.normal_(self.output_proj.weight, std=1e-3)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, n):
        """
        Args:
            n: (n_orbitals,) binary occupation vector
        Returns:
            (n_orbitals, n_fermions) backflow correction delta_M
        """
        # Input projection: (seq, 1) -> (seq, d_model)
        h = self.input_proj(n.unsqueeze(-1))

        # Attention layers
        for layer in self.layers:
            h = layer(h)

        # Output: (seq, d_model) -> (n_orbitals, n_fermions)
        return self.output_proj(h)


class AttentionNNBF_GPU(WavefunctionModel_GPU):
    """Attention-based Neural Network Backflow.

    psi(x) = det((M_base + bf_scale * Attention(n))[occupied, :])

    Demonstrates how to use standard nn.Module layers (nn.Linear,
    nn.LayerNorm, F.scaled_dot_product_attention) within the
    WavefunctionModel_GPU framework via torch.func.functional_call.

    The pattern:
        1. Build the NN as a normal nn.Module (_AttentionBackflow).
        2. Extract its named_parameters() into a flat list.
        3. Pass [M_base] + nn_params to super().__init__ (registers
           everything in self.params ParameterList).
        4. Hide the module object from nn.Module's child scanning
           by storing it in a plain list (self._nn_container = [mod]).
        5. In amplitude(), reconstruct the param dict from names +
           params_list, then call torch.func.functional_call().

    This works with vmap and torch.func.grad because functional_call
    threads explicit parameters through the module's forward().

    Args:
        n_sites: number of spatial sites
        n_fermions: number of fermions
        d_model: attention embedding dimension (default 32)
        n_heads: number of attention heads (default 4)
        n_layers: number of attention layers (default 1)
        bf_scale: initial scale for backflow output (default 0.01)
        dtype: parameter dtype (default float64)
    """

    def __init__(
        self,
        n_sites,
        n_fermions,
        d_model=32,
        n_heads=4,
        n_layers=1,
        bf_scale=0.01,
        dtype=torch.float64,
    ):
        self.n_sites = n_sites
        self.n_fermions = n_fermions
        self.n_orbitals = 2 * n_sites
        self.bf_scale = bf_scale

        # [0] Base orbital matrix
        M_base = torch.randn(
            self.n_orbitals, n_fermions, dtype=dtype,
        )

        # Build NN as a standard nn.Module
        nn_module = _AttentionBackflow(
            self.n_orbitals, n_fermions, d_model, n_heads,
            n_layers, dtype,
        )

        # Extract named parameters for functional_call
        nn_param_names = []
        nn_param_tensors = []
        for name, p in nn_module.named_parameters():
            nn_param_names.append(name)
            nn_param_tensors.append(p.data.clone())
        self._nn_param_names = nn_param_names

        # All params: [M_base, nn_param_0, nn_param_1, ...]
        all_tensors = [M_base] + nn_param_tensors
        super().__init__(params_list=all_tensors)

        # Hide module from nn.Module's parameter scanning.
        # Storing in a plain list prevents it from being registered
        # as a child module (which would double-count parameters).
        # The module is only used as a "computation template" —
        # its own .parameters() are never used; functional_call
        # substitutes them with our explicitly passed tensors.
        self._nn_container = [nn_module]

    def amplitude(self, x, params_list):
        """Single-sample attention backflow evaluation.

        Args:
            x:           (N_sites,) int64 — one configuration
            params_list: list of parameter tensors

        Returns:
            scalar determinant value
        """
        M_base = params_list[0]
        nn_params = params_list[1:]

        # Convert x to binary occupation vector
        spin_up = ((x == 2) | (x == 3)).to(M_base.dtype)
        spin_dn = ((x == 1) | (x == 3)).to(M_base.dtype)
        n = torch.cat([spin_up, spin_dn])  # (2*N_sites,)

        # Run attention block via functional_call
        param_dict = dict(zip(self._nn_param_names, nn_params))
        nn_module = self._nn_container[0]
        delta_M = torch.func.functional_call(
            nn_module, param_dict, (n,),
        )  # (n_orbitals, n_fermions)

        # Slater determinant with backflow
        M = M_base + self.bf_scale * delta_M

        Nf = self.n_fermions
        occupied = n.argsort(descending=True)[:Nf]
        A = M[occupied]  # (N_f, N_f)
        return torch.linalg.det(A)
