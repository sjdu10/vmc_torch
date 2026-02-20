"""
GPU-specific fPEPS models with torch.compile support.

These are adapted from models/pureTNS.py for GPU usage,
with compile-friendly defaults and GPU-specific optimizations.

Key: torch.export captures the quimb/symmray computation as
a pure aten-ops FX graph, which is then compatible with both
torch.vmap and torch.compile. This gives ~13x speedup for
exact contraction (chi=-1) and ~1.4x for boundary contraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import quimb as qu
import quimb.tensor as qtn


class fPEPS_Model_GPU(nn.Module):
    """
    fPEPS model optimized for GPU with torch.export + compile.

    Pipeline: torch.export (trace) -> torch.vmap (batch) ->
              torch.compile (fuse CUDA kernels)

    This eliminates all Python overhead from quimb/symmray at
    forward time. The export step captures the TN contraction
    as a fixed graph of aten ops.

    Usage:
        model = fPEPS_Model_GPU(tn=peps, max_bond=chi, ...)
        model.to(device)
        model.export_and_compile(example_x)  # one-time setup
        out = model(fxs)  # fast batched forward
    """

    def __init__(
        self,
        tn,
        max_bond,
        dtype=torch.float64,
        contract_boundary_opts=None,
        **kwargs,
    ):
        super().__init__()

        if contract_boundary_opts is None:
            contract_boundary_opts = {}

        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.contract_boundary_opts = contract_boundary_opts
        self.chi = max_bond

        # Flatten pytree into a single list for torch
        params_flat, params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.params_pytree = params_pytree

        # Register as ParameterList
        self.params = nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype)
            for x in params_flat
        ])

        # Pre-vmap the amplitude function (eager fallback)
        self._vmapped_amplitude = torch.vmap(
            self.amplitude,
            in_dims=(0, None),
            randomness='different',
        )

        self._compiled = False
        self._exported = False

    def amplitude(self, x, params):
        """Single-sample amplitude via quimb TN contraction."""
        tn = qtn.unpack(params, self.skeleton)
        amp = tn.isel({
            tn.site_ind(site): x[i]
            for i, site in enumerate(tn.sites)
        })
        if self.chi > 0:
            amp.contract_boundary_from_xmin_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[0, amp.Lx // 2 - 1],
                **self.contract_boundary_opts,
            )
            amp.contract_boundary_from_xmax_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[amp.Lx // 2, amp.Lx - 1],
                **self.contract_boundary_opts,
            )
        return amp.contract()

    def _amplitude_for_export(self, x, *flat_params):
        """Wrapper for torch.export: takes flat params."""
        p = qu.utils.tree_unflatten(
            list(flat_params), self.params_pytree
        )
        return self.amplitude(x, p)

    def export_and_compile(
        self, example_x, mode='default', **compile_kwargs,
    ):
        """
        Export + compile the amplitude function for GPU speedup.

        1. torch.export traces the amplitude function with a
           concrete example input, capturing all quimb/symmray
           operations as a pure aten-ops FX graph.
        2. torch.vmap batches the exported graph.
        3. torch.compile fuses the batched ops into CUDA kernels.

        Call AFTER .to(device).

        Args:
            example_x: single-sample config tensor (N_sites,)
                on the target device.
            mode: torch.compile mode ('default', 'reduce-overhead',
                'max-autotune').

        Note:
            One-time cost: ~2s export + ~10-40s compile.
            Must re-export if the tensor network structure changes
            (e.g., different Lx/Ly or different contraction path).
            Safe to call with different parameter VALUES — only the
            graph structure is baked in.
        """
        from torch.export import export

        # Step 1: Export — trace amplitude with concrete example
        params_list = list(self.params)

        class _AmpModule(nn.Module):
            def __init__(self_, amp_fn):
                super().__init__()
                self_._fn = amp_fn

            def forward(self_, x, *flat_params):
                return self_._fn(x, *flat_params)

        with torch.no_grad():
            exported = export(
                _AmpModule(self._amplitude_for_export),
                (example_x, *params_list),
            )
        self._exported_module = exported.module()

        # Step 2: vmap over the exported module
        n_params = len(params_list)
        self._vmapped_exported = torch.vmap(
            self._exported_module,
            in_dims=(0, *([None] * n_params)),
        )

        # Step 3: Compile the vmapped exported function
        self._vmapped_compiled = torch.compile(
            self._vmapped_exported,
            mode=mode,
            **compile_kwargs,
        )

        self._exported = True
        self._compiled = True

    def export_only(self, example_x):
        """
        Export + vmap without compile. Useful for debugging
        or when compile time is too long.
        """
        from torch.export import export

        params_list = list(self.params)

        class _AmpModule(nn.Module):
            def __init__(self_, amp_fn):
                super().__init__()
                self_._fn = amp_fn

            def forward(self_, x, *flat_params):
                return self_._fn(x, *flat_params)

        with torch.no_grad():
            exported = export(
                _AmpModule(self._amplitude_for_export),
                (example_x, *params_list),
            )
        self._exported_module = exported.module()

        n_params = len(params_list)
        self._vmapped_exported = torch.vmap(
            self._exported_module,
            in_dims=(0, *([None] * n_params)),
        )
        self._exported = True

    def compile_model(self, mode='reduce-overhead', **kwargs):
        """
        Legacy compile: wraps vmap(eager) with torch.compile.
        Does NOT use export — less effective, kept for
        compatibility.
        """
        self._vmapped_amplitude = torch.compile(
            self._vmapped_amplitude,
            fullgraph=False,
            mode=mode,
            **kwargs,
        )
        self._compiled = True

    def vamp(self, x, params):
        params = qu.utils.tree_unflatten(params, self.params_pytree)
        return self._vmapped_amplitude(x, params)

    def forward(self, x):
        if self._exported:
            params_list = list(self.params)
            if self._compiled:
                return self._vmapped_compiled(x, *params_list)
            else:
                return self._vmapped_exported(x, *params_list)
        return self.vamp(x, self.params)


class PureNN_GPU(nn.Module):
    """
    Pure neural-network wavefunction for GPU VMC benchmarking.

    Drop-in replacement for fPEPS_Model_GPU — exposes the same
    interface (params, vamp, forward, _compiled, _exported) so it
    can be swapped into vmc_run.py without other changes.

    Architecture:
        x: (B, n_sites) int64
        → embedding lookup    (B, n_sites, embed_dim)
        → flatten             (B, n_sites * embed_dim)
        → [Linear → Tanh] x n_layers
        → Linear              (B, 1) → squeeze → (B,) real

    Parameter layout in self.params (ParameterList indices):
        [0]        emb_w  (phys_dim, embed_dim)
        [1+2k]     w_k    (hidden_dim, in_dim)   k=0..n_layers-1
        [2+2k]     b_k    (hidden_dim,)
        [-2]       out_w  (1, hidden_dim)
        [-1]       out_b  (1,)
    """

    def __init__(
        self,
        n_sites,
        phys_dim=4,
        embed_dim=16,
        hidden_dim=256,
        n_layers=2,
        dtype=torch.float64,
    ):
        super().__init__()
        self.n_sites = n_sites
        self.phys_dim = phys_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dtype = dtype

        all_tensors = []

        # Embedding: (phys_dim, embed_dim)
        # Scale 0.3 gives hidden-layer inputs of O(0.4) std,
        # keeping tanh activations in the linear regime at init.
        emb_w = torch.randn(phys_dim, embed_dim, dtype=dtype) * 0.3
        all_tensors.append(emb_w)

        # Hidden layers
        in_dim = n_sites * embed_dim
        for _ in range(n_layers):
            w = torch.empty(hidden_dim, in_dim, dtype=dtype)
            nn.init.kaiming_uniform_(w, a=0, mode='fan_in',
                                     nonlinearity='tanh')
            b = torch.zeros(hidden_dim, dtype=dtype)
            all_tensors.append(w)
            all_tensors.append(b)
            in_dim = hidden_dim

        # Output layer: (1, hidden_dim) → scalar per sample
        out_w = torch.empty(1, hidden_dim, dtype=dtype)
        nn.init.kaiming_uniform_(out_w, a=0, mode='fan_in',
                                  nonlinearity='linear')
        out_b = torch.zeros(1, dtype=dtype)
        all_tensors.append(out_w)
        all_tensors.append(out_b)

        # Single registration — no submodule + ParameterList duplication
        self.params = nn.ParameterList([
            nn.Parameter(t) for t in all_tensors
        ])

        self._compiled = False
        self._exported = False

    def _amp_from_params(self, x, params_list):
        """
        Pure tensor-op forward pass.

        Args:
            x:           (B, n_sites) int64 configurations
            params_list: list of parameter tensors

        Returns:
            (B,) real amplitudes
        """
        idx = 0
        emb_w = params_list[idx]; idx += 1

        # F.embedding has explicit vmap/grad functional-transform support;
        # plain emb_w[x] (aten::index) does not propagate grads through vmap.
        h = F.embedding(x, emb_w).reshape(x.shape[0], -1)

        # Hidden layers
        for _ in range(self.n_layers):
            w = params_list[idx]; idx += 1
            b = params_list[idx]; idx += 1
            h = torch.tanh(h @ w.T + b)

        # Output: (B, 1) → (B,)
        out_w = params_list[idx]; idx += 1
        out_b = params_list[idx]
        return (h @ out_w.T + out_b).squeeze(-1)

    def vamp(self, x, params):
        """
        Batched amplitude evaluation compatible with torch.vmap and
        torch.func.grad contexts.

        Args:
            x:      (B, n_sites) int64 configurations
            params: nn.ParameterList or list of parameter tensors

        Returns:
            (B,) amplitudes
        """
        if isinstance(params, nn.ParameterList):
            params = list(params)
        return self._amp_from_params(x, params)

    def forward(self, x):
        """
        Args:
            x: (B, n_sites) int64 configurations

        Returns:
            (B,) amplitudes
        """
        if self._compiled:
            return self._compiled_fn(x, list(self.params))
        return self._amp_from_params(x, list(self.params))

    def compile_model(self, mode='default', **kwargs):
        """
        Compile _amp_from_params with torch.compile.

        No export step needed — torch.compile can trace standard
        PyTorch ops directly without quimb/symmray Python dispatch.
        """
        self._compiled_fn = torch.compile(
            self._amp_from_params,
            mode=mode,
            **kwargs,
        )
        self._compiled = True
