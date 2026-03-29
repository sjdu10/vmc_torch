"""GPU wavefunction base class with single-sample amplitude + auto-vmap.

All subclasses define a single-sample amplitude:

    amplitude(x, params_list)
        x:           (N_sites,) int64 — one configuration
        params_list: list of parameter tensors
        returns:     scalar amplitude

The base class vmaps it automatically.  No model ever sees (B, N_sites).

WavefunctionModel_GPU also provides:
  - forward(x): compiled -> exported -> eager dispatch
  - vamp(x, params): batched amplitude for torch.func.grad
  - export_and_compile / export_only / compile_model
"""
import torch
import torch.nn as nn


class WavefunctionModel_GPU(nn.Module):
    """Base class for GPU wavefunction models.

    Subclasses must implement:
        amplitude(x, params_list) -> scalar
            x is (N_sites,) int64, params_list is list[Tensor].

    Optionally override:
        vamp(x, params) — default normalizes ParameterList -> list,
            calls _vmapped_amplitude.  Override for model-specific
            param handling (e.g. quimb pytree unflatten for TN).
    """

    def __init__(self, params_list):
        """
        Args:
            params_list: list of Tensor — the learnable parameters.
                Each tensor is registered as an nn.Parameter.
        """
        super().__init__()
        self.params = nn.ParameterList([
            nn.Parameter(t) if not isinstance(t, nn.Parameter) else t
            for t in params_list
        ])
        self._compiled = False
        self._exported = False
        self._exported_log_amp = False

        # Pre-vmap the single-sample amplitude function
        self._vmapped_amplitude = torch.vmap(
            self.amplitude,
            in_dims=(0, None),
            randomness='different',
        )

        # Pre-vmap the single-sample log-amplitude function
        self._vmapped_log_amplitude = torch.vmap(
            self.log_amplitude,
            in_dims=(0, None),
            randomness='different',
        )

    # ----- Must implement -----

    def amplitude(self, x, params_list):
        """Single-sample amplitude evaluation.

        Args:
            x:           (N_sites,) int64 — one configuration
            params_list: list of parameter tensors

        Returns:
            scalar amplitude
        """
        raise NotImplementedError

    # ----- Optionally override -----
    
    def _vamp_params_preprocess(self, params):
        """Preprocess params for vamp.  Default: normalize ParameterList -> list.
        
        Override if vamp needs different param handling than forward.
        """
        if isinstance(params, nn.ParameterList):
            return list(params)
        return params

    def vamp(self, x, params):
        """Batched amplitude compatible with torch.vmap / torch.func.grad.

        Default: normalize ParameterList -> list, call
        _vmapped_amplitude (vmap over single-sample amplitude).

        Override for model-specific param handling.
        """
        params = self._vamp_params_preprocess(params)
        return self._vmapped_amplitude(x, params)

    # ----- Log-amplitude interface -----

    def log_amplitude(self, x, params_list):
        """Single-sample: returns (sign, log_abs) scalars.

        Default wraps amplitude(). Override for native log-space.
        """
        amp = self.amplitude(x, params_list)
        sign = torch.sign(amp)
        log_abs = torch.log(amp.abs().clamp(min=1e-45))
        return sign, log_abs
    
    def vamp_log(self, x, params):
        """Batched log-amplitude: returns (signs, log_abs) each (B,).

        Default: vmap over log_amplitude(). If a subclass overrides
        log_amplitude (e.g. for native log-space TN contraction),
        this automatically picks it up.
        
        Override for model-specific param handling.
        """
        params = self._vamp_params_preprocess(params)
        return self._vmapped_log_amplitude(x, params)

    def forward_log(self, x):
        """Dispatch: compiled -> exported -> eager for log-amplitude."""
        if self._exported and self._exported_log_amp:
            params_list = list(self.params)
            if self._compiled:
                return self._vmapped_compiled(x, *params_list)
            return self._vmapped_exported(x, *params_list)
        return self.vamp_log(x, self.params)

    # ----- Provided for free -----

    def forward(self, x):
        """Dispatch: compiled -> exported -> eager."""
        if self._exported and not self._exported_log_amp:
            params_list = list(self.params)
            if self._compiled:
                return self._vmapped_compiled(x, *params_list)
            else:
                return self._vmapped_exported(x, *params_list)
        return self.vamp(x, self.params)

    def _amplitude_for_export(self, x, *flat_params):
        """Wrapper for torch.export: takes flat *args params.

        Single-sample: x is (N_sites,), returns scalar.
        """
        p = self._vamp_params_preprocess(list(flat_params))
        return self.amplitude(x, p)

    def _log_amplitude_for_export(self, x, *flat_params):
        """Wrapper for torch.export: log_amplitude with flat *args.

        Single-sample: x is (N_sites,), returns (sign, log_abs).
        """
        p = self._vamp_params_preprocess(list(flat_params))
        return self.log_amplitude(x, p)

    def _move_exported_constants_to_device(self, device):
        """Move CPU constants in the exported graph to GPU.

        torch.export captures symmray's block-sparse index tensors
        as CPU int64 constants.  Without this step, torch.compile
        inserts a DeviceCopy (H2D) for each one on every forward call.
        """
        gm = self._exported_module
        graph = gm.graph

        # Move CPU constant tensors to the target device
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

        # Patch _assert_tensor_metadata nodes that still reference
        # device='cpu' — update them to match the new device
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

    def export_and_compile(
        self, example_x, mode='default',
        use_log_amp=False, **compile_kwargs,
    ):
        """Export + compile the amplitude function for GPU speedup.

        1. torch.export traces the amplitude with a concrete example,
           capturing all ops as a pure aten-ops FX graph.
        2. torch.vmap batches the exported graph.
        3. torch.compile fuses the batched ops into CUDA kernels.

        Call AFTER .to(device).

        Args:
            example_x: single-sample config tensor (N_sites,)
                on the target device.
            mode: torch.compile mode ('default', 'reduce-overhead',
                'max-autotune').
            use_log_amp: if True, export log_amplitude instead of
                amplitude. forward_log() will dispatch to compiled
                path; forward() will fall back to eager.
        """
        from torch.export import export

        params_list = list(self.params)

        if use_log_amp:
            export_fn = self._log_amplitude_for_export
        else:
            export_fn = self._amplitude_for_export

        class _AmpModule(nn.Module):
            def __init__(self_, amp_fn):
                super().__init__()
                self_._fn = amp_fn

            def forward(self_, x, *flat_params):
                return self_._fn(x, *flat_params)

        with torch.no_grad():
            exported = export(
                _AmpModule(export_fn),
                (example_x, *params_list),
            )
        self._exported_module = exported.module()
        self._move_exported_constants_to_device(example_x.device)

        n_params = len(params_list)
        self._vmapped_exported = torch.vmap(
            self._exported_module,
            in_dims=(0, *([None] * n_params)),
        )

        self._vmapped_compiled = torch.compile(
            self._vmapped_exported,
            mode=mode,
            **compile_kwargs,
        )

        self._exported = True
        self._compiled = True
        self._exported_log_amp = use_log_amp

    def export_only(self, example_x, use_log_amp=False):
        """Export + vmap without compile.  Useful for debugging."""
        from torch.export import export

        params_list = list(self.params)

        if use_log_amp:
            export_fn = self._log_amplitude_for_export
        else:
            export_fn = self._amplitude_for_export

        class _AmpModule(nn.Module):
            def __init__(self_, amp_fn):
                super().__init__()
                self_._fn = amp_fn

            def forward(self_, x, *flat_params):
                return self_._fn(x, *flat_params)

        with torch.no_grad():
            exported = export(
                _AmpModule(export_fn),
                (example_x, *params_list),
            )
        self._exported_module = exported.module()
        self._move_exported_constants_to_device(example_x.device)

        n_params = len(params_list)
        self._vmapped_exported = torch.vmap(
            self._exported_module,
            in_dims=(0, *([None] * n_params)),
        )
        self._exported = True
        self._exported_log_amp = use_log_amp

    def compile_model(self, mode='reduce-overhead', **kwargs):
        """Wrap vmap(eager) with torch.compile (no export step)."""
        self._vmapped_amplitude = torch.compile(
            self._vmapped_amplitude,
            fullgraph=False,
            mode=mode,
            **kwargs,
        )
        self._compiled = True

    def export_grad(
        self, mode='default', use_log_amp=False,
        do_compile=False, **compile_kwargs,
    ):
        """Build vmap(grad(exported_fn)) for fast grads.

        Requires export_and_compile() or export_only() first.
        Uses the exported aten-ops FX graph so vmap/grad bypass
        quimb/symmray Python dispatch entirely.

        Args:
            mode: torch.compile mode (only used if do_compile).
            use_log_amp: must match the export's use_log_amp.
            do_compile: if True, wrap with torch.compile for
                further kernel fusion (adds long warmup).
                Default False — export-only is usually enough.
            **compile_kwargs: passed to torch.compile.
        """
        assert self._exported, (
            "Call export_and_compile() before export_grad()"
        )
        exported_module = self._exported_module
        params_list = list(self.params)
        n_params = len(params_list)
        argnums = tuple(range(1, n_params + 1))
        in_dims = (0,) + (None,) * n_params

        if use_log_amp:
            def single_fn(x_i, *flat_params):
                sign, log_abs = exported_module(
                    x_i, *flat_params,
                )
                return log_abs, (sign, log_abs)
        else:
            def single_fn(x_i, *flat_params):
                amp = exported_module(x_i, *flat_params)
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
