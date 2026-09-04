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
import os
import warnings

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
        # Model-specific architecture info for printing (optional)
        self.model_arch = None

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

    @staticmethod
    def _reject_cudagraph_mode(mode, what):
        """Downgrade a cudagraph compile mode to 'default', loudly.

        The environment-reuse and bMPS-cache compile paths do NOT set
        ``_uses_cudagraph``, so their dispatch neither clones the
        compiled outputs nor calls ``_maybe_mark_cudagraph_step``.
        Worse, those paths CACHE boundary-MPS tensors across forwards
        (``cache_bMPS_params_*``): a cached cudagraph output buffer is
        overwritten by the next replay, which would silently corrupt
        every amplitude for the rest of the sweep.  So refuse the mode
        instead of accepting it unsafely.

        Returns the mode to actually compile with.
        """
        if mode in ('reduce-overhead', 'max-autotune'):
            warnings.warn(
                f"torch.compile mode {mode!r} requests CUDA graphs, "
                f"which {what} does not support yet: its cached "
                f"boundary-MPS tensors would alias a graph output "
                f"buffer and be overwritten by the next replay. "
                f"Falling back to mode='default'.",
                RuntimeWarning, stacklevel=3,
            )
            return 'default'
        return mode

    def _maybe_mark_cudagraph_step(self):
        """Signal a new iteration to inductor, if a graph was captured.

        Only does anything when ``export_and_compile`` was given a
        cudagraph mode ('reduce-overhead' / 'max-autotune'), which is
        what sets ``_uses_cudagraph`` -- so enabling CUDA graphs is
        auto-detected from the compile mode, with nothing for the caller
        to remember.

        Without this call the captured graph is NEVER replayed:
        inductor's ``CUDAGraphTreeManager.try_end_curr_warmup`` only
        leaves its WARMUP state when a new generation starts or the
        previous node's outputs are all dead, and a plain inference loop
        satisfies neither, so every call silently falls through to
        ``run_eager``.  There is no warning on that path.  Measured on
        4x8 D=4 chi=-1, B=256: 9.24 ms/forward without the mark vs
        1.65 ms with it (GPU utilisation 9.4% -> 68.6%, median
        inter-kernel gap 37 us -> 0 us).

        Safe here because forward/forward_log clone the compiled outputs
        below, so nothing the caller keeps still lives in the graph's
        static output buffer.  Callers that hold model outputs across
        later forwards -- e.g. the Metropolis samplers, which compare a
        proposal against a stored log|psi| -- therefore stay correct.
        """
        if getattr(self, '_uses_cudagraph', False):
            torch.compiler.cudagraph_mark_step_begin()

    def forward_log(self, x):
        """Dispatch: compiled -> exported -> eager for log-amplitude."""
        if self._exported and self._exported_log_amp:
            params_list = list(self.params)
            if self._compiled:
                self._maybe_mark_cudagraph_step()
                out = self._vmapped_compiled(x, *params_list)
                if getattr(self, '_uses_cudagraph', False):
                    # cudagraph static buffer; (sign, log_abs) tuple
                    out = tuple(o.clone() for o in out)
                return out
            return self._vmapped_exported(x, *params_list)
        return self.vamp_log(x, self.params)

    # ----- Provided for free -----

    def forward(self, x):
        """Dispatch: compiled -> exported -> eager."""
        if self._exported and not self._exported_log_amp:
            params_list = list(self.params)
            if self._compiled:
                self._maybe_mark_cudagraph_step()
                out = self._vmapped_compiled(x, *params_list)
                if getattr(self, '_uses_cudagraph', False):
                    out = out.clone()  # cudagraph reuses a static buffer
                return out
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
        self, example_x, mode='reduce-overhead',
        use_log_amp=True, cache_dir=None,
        export_grad_fn=True, grad_graph_capture=True,
        capture_safe_linalg=True,
        **compile_kwargs,
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
            cache_dir: if provided, save/load the ExportedProgram
                to/from this directory (alongside checkpoints).
                Avoids re-running torch.export on restarts.
            export_grad_fn: if True (default), also build the
                vmap(grad) function over the exported graph
                (``export_grad``) so ``compute_grads_gpu`` uses the
                exported gradient path instead of the eager vamp
                fallback. For chi > 0 contractions pair this with
                capture-safe linalg hooks (aten eigh backward is
                nan-prone on rank-deficient truncation Grams).
            grad_graph_capture: if True (default), mark the grad fn
                for lazy CUDA-graph capture on its first
                ``compute_grads_gpu`` call (see
                ``torch_utils.GraphedGradFn``).
            capture_safe_linalg: if True (default), ensure the
                capture-safe Jacobi linalg hooks are installed
                BEFORE tracing (torch.export bakes the registered
                svd/qr implementation into the graph). A prior
                manual ``install_capture_safe_linalg`` call with
                custom settings wins.
        """
        from torch.export import export

        if capture_safe_linalg:
            from vmc_torch.GPU.torch_utils import (
                ensure_capture_safe_linalg,
            )
            ensure_capture_safe_linalg()

        params_list = list(self.params)
        n_params = len(params_list)

        # --- Determine cache path ---
        if cache_dir is not None:
            amp_tag = "logamp" if use_log_amp else "amp"
            cache_path = os.path.join(
                cache_dir, f"exported_{amp_tag}.pt2",
            )
        else:
            cache_path = None

        # --- Load compiler artifacts before torch.compile ---
        # Must be loaded before torch.compile so dynamo finds
        # cached Triton kernels instead of recompiling from scratch.
        if cache_dir is not None:
            artifacts_path = os.path.join(
                cache_dir, "compiler_artifacts.bin",
            )
            if os.path.exists(artifacts_path):
                with open(artifacts_path, 'rb') as f:
                    artifact_bytes = f.read()
                torch.compiler.load_cache_artifacts(artifact_bytes)
                print(
                    f"Loaded compiler artifacts from "
                    f"{artifacts_path}"
                )
        else:
            artifacts_path = None
        self._compiler_artifacts_path = artifacts_path

        # --- Export or load from cache ---
        if cache_path is not None and os.path.exists(cache_path):
            try:
                # torch>=2.14: pt2 loader allocates with torch.empty(0) (default
                # device) then set_()s a CPU storage -> breaks under a cuda
                # default device. FIX: Load on cpu
                with torch.device('cpu'):
                    exported = torch.export.load(cache_path)
            except Exception as e:
                warnings.warn(f"cached ExportedProgram unusable ({e!r}); re-exporting")
                os.remove(cache_path)
                exported = None
        else:
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

            print("Running torch.export (this may take a while)...")
            with torch.no_grad():
                exported = export(
                    _AmpModule(export_fn),
                    (example_x, *params_list),
                )

            # Save before device move so constants remain on CPU
            # (portable across devices / restarts).
            if cache_path is not None:
                os.makedirs(cache_dir, exist_ok=True)
                torch.export.save(exported, cache_path)
                print(f"Saved ExportedProgram to {cache_path}")

        self._exported_module = exported.module()
        self._move_exported_constants_to_device(example_x.device)

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
        # cudagraph modes return outputs in a static buffer that the
        # next forward call overwrites in-place -> must clone outputs.
        self._uses_cudagraph = mode in ('reduce-overhead', 'max-autotune')

        if export_grad_fn:
            self.export_grad(use_log_amp=use_log_amp)
            # lazy capture happens on the first compute_grads_gpu
            # call (only there is the grad chunk shape known)
            self._grad_graph_capture = grad_graph_capture


    def save_compiler_artifacts(self):
        """Save compiled Triton kernel artifacts to cache_dir.

        Call this AFTER the first forward pass (warmup) so that all
        kernels have been compiled and can be captured. On the next
        run, export_and_compile will load them automatically and the
        first forward should be fast.
        """
        path = getattr(self, '_compiler_artifacts_path', None)
        if path is None:
            print(
                "save_compiler_artifacts: no cache_dir set, "
                "skipping."
            )
            return
        try:
            artifact_bytes, _ = (
                torch.compiler.save_cache_artifacts()
            )
            with open(path, 'wb') as f:
                f.write(artifact_bytes)
            print(
                f"Saved compiler artifacts to {path} "
                f"({len(artifact_bytes) / 1e6:.1f} MB)"
            )
        except Exception as e:
            print(f"save_compiler_artifacts failed: {e}")

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
