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

        # Pre-vmap the single-sample amplitude function
        self._vmapped_amplitude = torch.vmap(
            self.amplitude,
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

    def vamp(self, x, params):
        """Batched amplitude compatible with torch.vmap / torch.func.grad.

        Default: normalize ParameterList -> list, call
        _vmapped_amplitude (vmap over single-sample amplitude).

        Override for model-specific param handling.
        """
        if isinstance(params, nn.ParameterList):
            params = list(params)
        return self._vmapped_amplitude(x, params)

    # ----- Provided for free -----

    def forward(self, x):
        """Dispatch: compiled -> exported -> eager."""
        if self._exported:
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
        return self.amplitude(x, list(flat_params))

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
        self, example_x, mode='default', **compile_kwargs,
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

    def export_only(self, example_x):
        """Export + vmap without compile.  Useful for debugging."""
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
        self._move_exported_constants_to_device(example_x.device)

        n_params = len(params_list)
        self._vmapped_exported = torch.vmap(
            self._exported_module,
            in_dims=(0, *([None] * n_params)),
        )
        self._exported = True

    def compile_model(self, mode='reduce-overhead', **kwargs):
        """Wrap vmap(eager) with torch.compile (no export step)."""
        self._vmapped_amplitude = torch.compile(
            self._vmapped_amplitude,
            fullgraph=False,
            mode=mode,
            **kwargs,
        )
        self._compiled = True
