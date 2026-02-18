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
