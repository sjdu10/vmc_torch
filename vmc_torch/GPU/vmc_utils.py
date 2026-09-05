import quimb as qu
import torch
import torch.distributed as dist
import time
from torch.utils._pytree import tree_map, tree_flatten
import os

"""Utility functions for GPU-accelerated VMC with PEPS wavefunctions.
Mainly includes:
- energy evaluation with GPU-batched get_conn and bMPS reuse
- gradient calculation"""
# ==========================================================
# Utilities
# ==========================================================

def check_NaN_or_inf(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        tensor_nan = torch.isnan(tensor).sum().item()
        print(f"Warning: {name} contains {tensor_nan} NaN values.")
    if torch.isinf(tensor).any():
        tensor_inf = torch.isinf(tensor).sum().item()
        print(f"Warning: {name} contains {tensor_inf} Inf values.")

def has_precomputed_gpu_conn(H):
    """Return whether H is a precomputed GPU-batched Hamiltonian."""
    from vmc_torch.hamiltonian_torch import GPUMixin

    return isinstance(H, GPUMixin) and hasattr(H, '_hop_list')

def cpu_mem_gb():
    """Resident memory of this process in GB."""
    with open(f'/proc/{os.getpid()}/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024**2  # kB -> GB
    return 0.0


def random_initial_config(N_f, N_sites, seed=None):
    # Use a local torch.Generator so this seed doesn't pollute the
    # global RNG. Otherwise every subsequent torch.rand* in the
    # process inherits the state we left here.
    #
    # All tensor ops here are pinned to CPU explicitly: the caller
    # (initialize_walkers) does the .to(device) transfer. Without
    # device='cpu', a global torch.set_default_device('cuda') would
    # make torch.randperm try to use a CUDA generator and crash with
    # "Expected a 'cuda' device type for generator but found 'cpu'".
    gen = torch.Generator(device='cpu')
    if seed is not None:
        gen.manual_seed(int(seed))
    half_filled_config = torch.tensor(
        [1, 2] * (N_sites // 2), device='cpu',
    )
    # Set first (Lx*Ly - N_f) sites to be empty (0)
    empty_sites = list(range(N_sites - N_f))
    doped_config = half_filled_config.clone()
    doped_config[empty_sites] = 0
    # Randomly permute the doped_config using the local generator
    perm = torch.randperm(N_sites, generator=gen, device='cpu')
    doped_config = doped_config[perm]
    num_1 = torch.sum(doped_config == 1).item()
    num_2 = torch.sum(doped_config == 2).item()
    assert num_1 == N_f // 2 and num_2 == N_f // 2, f"Number of spin up and spin down fermions should be {N_f // 2}, but got {num_1} and {num_2}"

    return doped_config


def are_pytrees_equal(tree1, tree2):
    from torch.utils import _pytree as pytree
    # Flatten both trees
    leaves1, spec1 = pytree.tree_flatten(tree1)
    leaves2, spec2 = pytree.tree_flatten(tree2)

    # 1. Compare structure (TreeSpec)
    if spec1 != spec2:
        print("Tree structures differ.")
        return False

    # 2. Compare leaves (Tensors/Values)
    if len(leaves1) != len(leaves2):
        print("Number of leaves differ.")
        return False

    for l1, l2 in zip(leaves1, leaves2):
        if torch.is_tensor(l1) and torch.is_tensor(l2):
            if not torch.equal(l1, l2):
                print("Tensor leaves differ.")
                return False
        else:
            if (l1 != l2).any():
                print("Non-tensor leaves differ.")
                print("l1:", l1)
                print("l2:", l2)
                return False

    return True

# ============================================================
# Explicit parameter update (the step normally inside OptimizerGPU)
# ============================================================
def apply_update(model, direction, lr):
    """In-place SGD step:  theta <- theta - lr * direction.

    Identical arithmetic to ``SGDGPU`` / ``OptimizerGPU.step`` in
    optimizer.py, written out so the update is visible.  Computed in
    float64, cast back to the model dtype, and copied IN PLACE so the
    parameter storage (and any compiled graph) is preserved.

    ``direction`` is identical on every rank (the all_reduce happened
    inside ``preconditioner.solve``), so applying it independently on
    each rank keeps the replicas in sync.
    """
    direction = torch.as_tensor(direction, dtype=torch.float64)
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            n = p.numel()
            step = direction[offset:offset + n].view_as(p)
            p.data.copy_((p.data.to(torch.float64) - lr * step).to(p.dtype))
            offset += n


def global_energy_stats(local_energies, world_size):
    """Mean / variance of E_loc across ALL ranks (data-parallel).

    Same reduction as ``VMC_GPU.compute_global_energy_stats``.
    """
    total_ns = local_energies.shape[0] * world_size
    e_sum = local_energies.sum()
    e_sq_sum = (local_energies ** 2).sum()
    if world_size > 1:
        dist.all_reduce(e_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(e_sq_sum, op=dist.ReduceOp.SUM)
    e_mean = e_sum.item() / total_ns
    e_var = e_sq_sum.item() / total_ns - e_mean ** 2
    return total_ns, e_mean, e_var


def broadcast_params(model):
    """Copy rank-0 params to every rank (paranoid resync)."""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    for p in model.parameters():
        if not p.data.is_contiguous():
            p.data = p.data.contiguous()
        dist.broadcast(p.data, src=0)


# ==========================================================
# Energy evaluation
# ==========================================================

def _evaluate_energy_impl(
    fxs, fpeps_model, H, current_amps,
    verbose=False, use_log_amp=False, **kwargs,
):
    """Core energy evaluation (no grad context decorator).

    Shared by evaluate_energy (inference_mode) and
    evaluate_energy_grad (grad-enabled).

    For each config fxs[b], obtains connected configs and matrix
    elements via H.get_conn, evaluates amplitudes on all connected
    configs, and assembles E_loc[b] = sum_s' H_{s,s'} psi(s')/psi(s).

    Uses GPU-batched get_conn when a GPUMixin Hamiltonian has been
    precomputed, otherwise falls back to per-sample CPU computation. Connected amplitudes
    are evaluated in size-B chunks with padding on the last chunk
    to keep input shapes fixed for torch.compile.

    Args:
        fxs: Configurations, (B, N_sites) int64.
        fpeps_model: Batched wavefunction model, (B, N_sites) -> (B,).
        H: Hamiltonian with get_conn (or get_conn_batch_gpu) method.
        current_amps: Amplitudes at fxs, (B,). When
            use_log_amp=True, this is (signs, log_abs).
        verbose: Print timing breakdown.
        use_log_amp: If True, current_amps is a
            (signs, log_abs) tuple and connected amps
            are evaluated in log-space.

    Returns:
        energy: Mean local energy, scalar.
        local_energies: Per-sample local energies, (B,).
    """
    import numpy as np
    B = fxs.shape[0]
    device = fxs.device
    
    # --- GPU-batched path: zero CPU round-trips ---
    if has_precomputed_gpu_conn(H):
        if verbose:
            t0 = time.time()
        conn_etas, conn_eta_coeffs, batch_ids = H.get_conn_batch_gpu(fxs)
        conn_eta_num = torch.bincount(batch_ids, minlength=B)
        if verbose:
            t1 = time.time()
            print(f"GPU get_conn_batch time: {t1 - t0:.4f}s")

    # --- Fallback: one bulk CPU→GPU transfer instead of per-sample uploads ---
    else:
        print("Warning: H does not support get_conn_batch_gpu, falling back to CPU computation for connected configurations. This may be slow.")
        fxs_cpu = fxs.cpu()
        all_etas_np, all_coeffs_np, conn_eta_num_list = [], [], []
        for fx in fxs_cpu:
            eta, coeffs = H.get_conn(fx)
            conn_eta_num_list.append(len(eta))
            all_etas_np.append(np.asarray(eta))
            all_coeffs_np.append(np.asarray(coeffs))
        conn_etas = torch.tensor(
            np.concatenate(all_etas_np), device=device
        )
        conn_eta_coeffs = torch.tensor(
            np.concatenate(all_coeffs_np), device=device, dtype=torch.float64
        )
        conn_eta_num = torch.tensor(conn_eta_num_list, device=device)
        batch_ids = torch.repeat_interleave(
            torch.arange(B, device=device), conn_eta_num
        )

    # Unpack log-amp current state if needed
    if use_log_amp:
        cur_signs, cur_log_abs = current_amps

    # Batch compute connected amplitudes — pad last chunk to
    # fixed size B to avoid torch.compile recompilation
    if verbose:
        t0 = time.time()
    chunk_size = B
    total_conn = conn_etas.shape[0]

    if use_log_amp:
        conn_signs_list = []
        conn_log_abs_list = []
    else:
        conn_amps_list = []

    for i in range(0, total_conn, chunk_size):
        if verbose:
            t00 = time.time()
        chunk = conn_etas[i:i + chunk_size]
        actual = chunk.shape[0]
        if actual < chunk_size:
            # Pad with copies of first row (result discarded)
            pad = chunk_size - actual
            chunk = torch.cat([
                chunk,
                chunk[:1].expand(pad, -1),
            ], dim=0)
            if use_log_amp:
                cs, cla = fpeps_model.forward_log(chunk)
                conn_signs_list.append(cs[:actual])
                conn_log_abs_list.append(cla[:actual])
            else:
                out = fpeps_model(chunk)
                conn_amps_list.append(out[:actual])
        else:
            if use_log_amp:
                cs, cla = fpeps_model.forward_log(chunk)
                conn_signs_list.append(cs)
                conn_log_abs_list.append(cla)
            else:
                conn_amps_list.append(fpeps_model(chunk))
        if verbose:
            print(
                f"  Evaluating connected amplitudes: "
                f"chunk {i // chunk_size + 1} / "
                f"{(total_conn + chunk_size - 1) // chunk_size}, "
                f"delta t_forward: {time.time() - t00:.4f}s, "
                f"total t_forward: {time.time() - t0:.4f}s"
            )

    if verbose:
        t1 = time.time()
        print(
            f"GPU forward for connected configs time: "
            f"{t1 - t0:.4f}s"
        )

    # Vectorized local energy calculation
    if use_log_amp:
        conn_signs = torch.cat(conn_signs_list)
        conn_log_abs = torch.cat(conn_log_abs_list)
        # amp_ratio = sign' * sign * exp(log_abs' - log_abs)
        amp_ratio = (
            conn_signs
            * cur_signs[batch_ids]
            * torch.exp(
                conn_log_abs - cur_log_abs[batch_ids],
            )
        )
        terms = conn_eta_coeffs * amp_ratio
    else:
        conn_amps = torch.cat(conn_amps_list)
        current_amps_expanded = current_amps[batch_ids]
        terms = conn_eta_coeffs * (
            conn_amps / current_amps_expanded
        )

    # Aggregate results (out-of-place for autograd support)
    local_energies = torch.zeros(
        B, device=device, dtype=terms.dtype,
    ).index_add(0, batch_ids, terms)

    energy = torch.mean(local_energies)

    return energy, local_energies


@torch.inference_mode()
def evaluate_energy(
    fxs, fpeps_model, H, current_amps,
    verbose=False, use_log_amp=False, **kwargs,
):
    """Compute local energies (inference mode, no grad)."""
    return _evaluate_energy_impl(
        fxs, fpeps_model, H, current_amps,
        verbose=verbose, use_log_amp=use_log_amp,
        **kwargs,
    )


def evaluate_energy_grad(
    fxs, fpeps_model, H, current_amps,
    verbose=False, use_log_amp=False, **kwargs,
):
    """Compute local energies with grad tracking.

    Same as evaluate_energy but without inference_mode,
    so the computation graph is kept for backprop
    (needed by the log-variance loss path).
    """
    return _evaluate_energy_impl(
        fxs, fpeps_model, H, current_amps,
        verbose=verbose, use_log_amp=use_log_amp,
        **kwargs,
    )


# ==========================================================
# ==========================================================
# Gradient computation
# ==========================================================

def flatten_params(parameters):
    vec = []
    for param in parameters:
        # Ensure parameters are on the same device
        vec.append(param.reshape(-1))
    return torch.cat(vec)

def _check_grads_amps(batched_grads_vec, amps, fpeps_model, configs=None):
    """Raise ValueError if amps or grads contain NaN/Inf."""
    if torch.isnan(amps).any() or torch.isinf(amps).any():
        nan_count = torch.isnan(amps).sum().item()
        inf_count = torch.isinf(amps).sum().item()
        # Print ill configs for amplitudes
        if configs is not None:
            bad_mask = torch.isnan(amps) | torch.isinf(amps)
            # Handle both 1D amps (B,) and 2D amps (B, 1)
            if bad_mask.dim() > 1:
                bad_mask = bad_mask.any(dim=1)
            bad_indices = bad_mask.nonzero(as_tuple=True)[0]
            torch.set_printoptions(threshold=10_000_000)
            for idx in bad_indices[:20].tolist():
                print(f"  amp NaN/Inf sample[{idx}]: {configs[idx].tolist()}")
            torch.set_printoptions(profile="default")
        raise ValueError(
            f"NaN/Inf in amplitudes: {nan_count} NaN, "
            f"{inf_count} Inf out of {amps.numel()} samples"
        )
    B = batched_grads_vec.shape[0]
    Np = batched_grads_vec.shape[1]
    max_check_elems = 10_000_000
    rows_per_check = max(1, max_check_elems // max(Np, 1))
    bad_samples = torch.zeros(
        B, dtype=torch.bool, device=batched_grads_vec.device,
    )
    bad_params = torch.zeros(
        Np, dtype=torch.bool, device=batched_grads_vec.device,
    )

    for start in range(0, B, rows_per_check):
        stop = min(start + rows_per_check, B)
        bad_chunk = ~torch.isfinite(batched_grads_vec[start:stop])
        if bad_chunk.any():
            bad_samples[start:stop] = bad_chunk.any(dim=1)
            bad_params |= bad_chunk.any(dim=0)
        del bad_chunk

    if bad_samples.any():
        n_bad = bad_samples.sum().item()
        bad_param_ids = bad_params.nonzero(as_tuple=True)[0]
        # Print ill configs that produce NaN/Inf gradients
        if configs is not None:
            bad_indices = bad_samples.nonzero(as_tuple=True)[0]
            print(f"Ill configs with NaN/Inf grads ({n_bad} total):")
            for idx in bad_indices[:20].tolist():
                print(f"  sample[{idx}]: {configs[idx].tolist()}")
        # Map flat param index to (param_idx, offset) in ParameterList
        param_ranges = []
        offset = 0
        for i, p in enumerate(fpeps_model.params):
            size = p.numel()
            param_ranges.append((i, offset, offset + size, p.shape))
            offset += size
        bad_param_info = []
        for pid in bad_param_ids[:10].tolist():
            for (idx, lo, hi, shape) in param_ranges:
                if lo <= pid < hi:
                    bad_param_info.append(
                        f"  flat[{pid}] -> params[{idx}]"
                        f"{list(shape)} offset {pid - lo}"
                    )
                    break
        raise ValueError(
            f"NaN/Inf in gradients: {n_bad}/{batched_grads_vec.shape[0]} "
            f"samples affected, "
            f"{bad_params.sum().item()}/{batched_grads_vec.shape[1]}"
            f" params affected.\n"
            f"First bad params:\n"
            + "\n".join(bad_param_info)
        )


def compute_grads_gpu(
    fxs, fpeps_model, batch_size=None,
    verbose=False, offload_to_cpu=False,
    use_log_amp=False, **kwargs,  # kwargs for backward compat
):
    """Per-sample gradient computation via vmap(grad) on GPU.

    Uses compiled grad path when model.compile_grad() has been
    called (torch.compile over vmap(grad(exported_fn))), otherwise
    falls back to eager vmap(grad(vamp)).

    Args:
        fxs: (B, N_sites) int64 configurations.
        fpeps_model: model with .params and .vamp/.vamp_log.
        batch_size: chunk size for gradient computation (B_grad).
        offload_to_cpu: if True, each (B_grad, Np) gradient chunk
            is moved to pinned CPU memory immediately after GPU
            computation.  Keeps GPU peak memory at O(B_grad * Np).
        use_log_amp: if True, compute d(log|psi|)/d(params)
            directly. Returns (log_psi_grad, (signs, log_abs))
            instead of (grads, amps).
    """
    B = fxs.shape[0]
    B_grad = batch_size if batch_size is not None else B

    # ------------------------------------------------------------------
    # Check for compiled grad path
    # ------------------------------------------------------------------
    use_exported_grad = (
        getattr(fpeps_model, '_grad_exported', False)
        and getattr(
            fpeps_model, '_grad_use_log_amp', False,
        ) == use_log_amp
    )

    if use_exported_grad:
        exported_grad_fn = fpeps_model._exported_grad_fn
        params_list = list(fpeps_model.params)

        # Lazy CUDA-graph capture of the grad fn (first call only).
        # Deferred to here because the captured shape must be the
        # padded chunk (B_grad, N_sites) — exactly what every chunk
        # below is padded to, so all chunks replay. Removes the
        # per-kernel launch overhead of the eager fwd+bwd execution
        # (~13x measured); see torch_utils.GraphedGradFn.
        from vmc_torch.GPU.torch_utils import GraphedGradFn
        if (
            getattr(fpeps_model, '_grad_graph_capture', False)
            and not isinstance(exported_grad_fn, GraphedGradFn)
            and fxs.is_cuda
        ):
            example = fxs[:B_grad]
            if example.shape[0] < B_grad:
                pad = example[0:1].expand(
                    B_grad - example.shape[0], -1,
                )
                example = torch.cat([example, pad], dim=0)
            try:
                exported_grad_fn = GraphedGradFn(
                    exported_grad_fn, example, params_list,
                    clone_outputs=False,  # chunks consumed below
                )
                fpeps_model._exported_grad_fn = exported_grad_fn
                print(
                    "[compute_grads] grad fn captured into a CUDA "
                    f"graph (chunk shape {tuple(example.shape)}); "
                    "subsequent calls replay. NOTE: calls with a "
                    "different grad_batch_size fall back to eager."
                )
            except Exception as e:
                import warnings
                torch.cuda.synchronize()
                exported_grad_fn = fpeps_model._exported_grad_fn
                warnings.warn(
                    "CUDA-graph capture of the grad fn failed "
                    f"({type(e).__name__}: {e}); falling back to "
                    "the uncaptured exported grad fn. Some op in "
                    "the exported grad graph is not capture-safe "
                    "(host sync such as .item()/nonzero/unique, "
                    "pageable H2D copy, or a raw eigh/svd when "
                    "chi > 0). To locate it, run the grad fn once "
                    "eagerly under "
                    "torch.cuda.set_sync_debug_mode('warn')."
                )
                fpeps_model._grad_graph_capture = False
    else:
        # Eager path: build vmap(grad(vamp)) function
        params_pytree = (
            list(fpeps_model.params)
            if isinstance(
                fpeps_model.params, torch.nn.ParameterList,
            )
            else dict(fpeps_model.params)
            if isinstance(
                fpeps_model.params, torch.nn.ParameterDict,
            )
            else fpeps_model.params
        )

        if use_log_amp:
            def single_sample_log_amp_func(x_i, p):
                sign, log_abs = fpeps_model.vamp_log(
                    x_i.unsqueeze(0), p,
                )
                sign = sign.squeeze(0)
                log_abs = log_abs.squeeze(0)
                return log_abs, (sign, log_abs)

            grad_vmap_fn = torch.vmap(
                torch.func.grad(
                    single_sample_log_amp_func,
                    argnums=1, has_aux=True,
                ),
                in_dims=(0, None),
            )
        else:
            def single_sample_amp_func(x_i, p):
                amp = fpeps_model.vamp(
                    x_i.unsqueeze(0), p,
                ).squeeze(0)
                return amp, amp

            grad_vmap_fn = torch.vmap(
                torch.func.grad(
                    single_sample_amp_func,
                    argnums=1, has_aux=True,
                ),
                in_dims=(0, None),
            )

    # ------------------------------------------------------------------
    # Pre-allocate output buffers
    # ------------------------------------------------------------------
    if use_exported_grad:
        Np = sum(p.numel() for p in params_list)
        p_dtype = params_list[0].dtype
    else:
        leaves_p, _ = tree_flatten(params_pytree)
        Np = sum(p.numel() for p in leaves_p)
        p_dtype = leaves_p[0].dtype
        del leaves_p

    if offload_to_cpu:
        # Pinned memory for faster D2H transfer
        batched_grads_vec = torch.empty(
            B, Np, dtype=p_dtype, device='cpu',
            pin_memory=True,
        )
        if use_log_amp:
            signs = torch.empty(
                B, dtype=p_dtype, device='cpu',
                pin_memory=True,
            )
            log_abs = torch.empty(
                B, dtype=p_dtype, device='cpu',
                pin_memory=True,
            )
        else:
            amps = torch.empty(
                B, dtype=p_dtype, device='cpu',
                pin_memory=True,
            )
    else:
        device = fxs.device
        batched_grads_vec = torch.empty(
            B, Np, dtype=p_dtype, device=device,
        )
        if use_log_amp:
            signs = torch.empty(
                B, dtype=p_dtype, device=device,
            )
            log_abs = torch.empty(
                B, dtype=p_dtype, device=device,
            )
        else:
            amps = torch.empty(
                B, dtype=p_dtype, device=device,
            )

    # ------------------------------------------------------------------
    # Chunked gradient computation
    # ------------------------------------------------------------------
    t0 = time.time()

    for b_start in range(0, B, B_grad):
        b_end = min(b_start + B_grad, B)
        if verbose:
            print(
                f"Processing grad chunk: "
                f"{b_start} to {b_end} / {B}"
            )
        fxs_chunk = fxs[b_start:b_end]
        actual_size = b_end - b_start

        if use_exported_grad:
            # Pad last chunk to B_grad to avoid recompilation
            if actual_size < B_grad:
                pad = fxs_chunk[0:1].expand(
                    B_grad - actual_size, -1,
                )
                fxs_chunk = torch.cat(
                    [fxs_chunk, pad], dim=0,
                )

            grads_tuple, aux_c = exported_grad_fn(
                fxs_chunk, *params_list,
            )
            # grads_tuple: tuple of (B_grad, *param_shape)
            flat_c = torch.cat(
                [g.detach().flatten(start_dim=1)
                 for g in grads_tuple],
                dim=1,
            )[:actual_size]

            if use_log_amp:
                sc, lac = aux_c
                sc = sc.detach()[:actual_size]
                lac = lac.detach()[:actual_size]
            else:
                aux_c = aux_c.detach()[:actual_size]
        else:
            grads_chunk, aux_c = grad_vmap_fn(
                fxs_chunk, params_pytree,
            )
            grads_chunk = tree_map(
                lambda x: x.detach(), grads_chunk,
            )
            leaves_c, _ = tree_flatten(grads_chunk)
            flat_c = torch.cat(
                [lf.flatten(start_dim=1) for lf in leaves_c],
                dim=1,
            )
            if use_log_amp:
                sc, lac = aux_c
                sc = sc.detach()
                lac = lac.detach()
            else:
                aux_c = aux_c.detach()
            del grads_chunk, leaves_c

        # Write to output buffer
        if offload_to_cpu:
            if verbose:
                time_to_cpu = time.time()
            batched_grads_vec[b_start:b_end].copy_(flat_c, non_blocking=True) 
            if use_log_amp:
                signs[b_start:b_end] = sc.cpu()
                log_abs[b_start:b_end] = lac.cpu()
            else:
                amps[b_start:b_end] = aux_c.cpu()
            if verbose:
                print(
                    f"  GPU to CPU transfer time: "
                    f"{time.time() - time_to_cpu:.4f}s"
                )
        else:
            batched_grads_vec[b_start:b_end] = flat_c
            if use_log_amp:
                signs[b_start:b_end] = sc
                log_abs[b_start:b_end] = lac
            else:
                amps[b_start:b_end] = aux_c

        del flat_c
        if use_log_amp:
            del sc, lac
        else:
            del aux_c

    # ------------------------------------------------------------------
    # Final cleanup and return
    # ------------------------------------------------------------------
    batched_grads_vec = batched_grads_vec.detach()
    fpeps_model.zero_grad()

    t1 = time.time()
    if verbose:
        label = "exported" if use_exported_grad else "eager"
        print(
            f"GPU vmap(grad) [{label}] time: "
            f"{t1 - t0:.4f}s"
        )

    if use_log_amp:
        _check_grads_amps(
            batched_grads_vec, log_abs, fpeps_model,
        )
        return batched_grads_vec, (signs, log_abs)
    else:
        _check_grads_amps(
            batched_grads_vec, amps, fpeps_model,
        )
        return batched_grads_vec, amps


def run_config_sampling(
    fxs,
    model,
    sampler,
    graph,
    *,
    n_burn_in,
    n_sweeps,
    thin=1,
    use_export_compile=False,
    use_log_amp=False,
    rank=0,
    world_size=1,
    verbose=False,
):
    """Sample-only MCMC loop for measuring diagonal observables.

    Drives ``sampler`` for ``n_burn_in`` burn-in sweeps, then ``n_sweeps``
    measurement sweeps, snapshotting the full ``(B, N_sites)`` walker batch
    every ``thin``-th sweep. Configs are stored as ``int8`` (values in
    ``{0, 1, 2, 3}``) on CPU to keep memory and disk footprint small.

    On multi-rank runs the per-rank configs are concatenated via
    ``dist.all_gather`` so rank 0 ends up holding the full sample.

    Args:
        fxs: ``(B, N_sites)`` int64 walker configs on the device.
        model: nn.Module with ``forward(x) -> (B,)`` amplitudes.
        sampler: ``SamplerGPU`` instance exposing ``step`` and ``burn_in``.
        graph: lattice graph with ``row_edges`` / ``col_edges``.
        n_burn_in: number of burn-in sweeps before any snapshots.
        n_sweeps: number of measurement sweeps after burn-in.
        thin: keep every ``thin``-th sweep snapshot. Default 1.
        use_export_compile: pass-through to sampler (compile mode).
        use_log_amp: pass-through to sampler (log-amplitude mode).
        rank: torch.distributed rank.
        world_size: torch.distributed world size.
        verbose: print per-sweep timing.

    Returns:
        all_configs: ``(Ns_total, N_sites)`` int8 CPU tensor on rank 0,
            ``None`` on other ranks. ``Ns_total = world_size *
            ceil(n_sweeps / thin) * B``.
        fxs: walker state after the final sweep (still on device).
    """
    sampler_kwargs = dict(
        compile=use_export_compile,
        use_log_amp=use_log_amp,
    )

    if n_burn_in > 0:
        t0 = time.time()
        fxs = sampler.burn_in(
            fxs, model, graph, n_burn_in, **sampler_kwargs,
        )
        if rank == 0 and verbose:
            print(
                f"Burn-in: {n_burn_in} sweeps "
                f"in {time.time() - t0:.2f}s"
            )

    local_chunks = []
    t_loop = time.time()
    for sweep in range(n_sweeps):
        fxs, _ = sampler.step(
            fxs, model, graph, verbose=verbose, **sampler_kwargs,
        )
        if sweep % thin == 0:
            # int8 is enough for {0, 1, 2, 3} and 8x smaller than int64.
            local_chunks.append(
                fxs.detach().to('cpu', dtype=torch.int8).clone(),
            )
    if rank == 0 and verbose:
        print(
            f"Sampling: {n_sweeps} sweeps "
            f"in {time.time() - t_loop:.2f}s"
        )

    local_configs = torch.cat(local_chunks, dim=0)

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        gathered = [
            torch.zeros_like(local_configs) for _ in range(world_size)
        ]
        dist.all_gather(gathered, local_configs.contiguous())
        all_configs = (
            torch.cat(gathered, dim=0) if rank == 0 else None
        )
    else:
        all_configs = local_configs

    return all_configs, fxs


# ==========================================================
# Backward compatibility
# ==========================================================
# The bMPS-reuse code below moved to
# vmc_torch/GPU/tensor_network/reuse.py.  Forwarded lazily (PEP 562)
# rather than re-exported at the top of this module: reuse.py imports
# FROM here, so an eager import would be circular.
_MOVED_TO_TN_REUSE = (
    'detect_changed_row_col_pair',
    '_slice_env_dict',
    'evaluate_energy_reuse',
    'detect_changed_rows',
    'evaluate_energy_reuse_x',
    'compute_grads_cheap_gpu',
)


def __getattr__(name):
    if name in _MOVED_TO_TN_REUSE:
        from vmc_torch.GPU.tensor_network import reuse
        return getattr(reuse, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
