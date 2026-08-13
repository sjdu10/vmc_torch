"""Boundary-MPS environment reuse for GPU VMC.

A Metropolis sweep changes only one or two sites per proposal, so most
of the boundary-MPS environment around a walker is still valid after a
move.  This module caches those environments and reuses them, instead
of recontracting the whole 2D network per amplitude:

    energy      evaluate_energy_reuse (x and y envs),
                evaluate_energy_reuse_x (x only)
    gradients   compute_grads_cheap_gpu
    samplers    the four *Reuse_GPU / *XReuse_GPU Metropolis samplers
    helpers     detect_changed_row_col_pair, detect_changed_rows,
                _slice_env_dict

Moved verbatim out of ``GPU/vmc_utils.py`` and ``GPU/sampler.py`` so
those stay TN-agnostic; both keep a lazy ``__getattr__`` forwarder, so
the old import paths still work.

Requires a model that can cache environments -- ``fPEPS_Model_reuse_GPU``
/ ``PEPS_Model_reuse_GPU`` in ``GPU/models/`` -- not the plain
full-contraction models.

Dependency direction is one-way: this module imports from ``vmc_utils``
and ``sampler``, never the reverse.
"""
import time
from typing import Tuple

import quimb as qu
import torch
from torch.utils._pytree import tree_flatten

from vmc_torch.GPU.sampler import (
    SamplerGPU,
    propose_exchange_or_hopping_vec,
    propose_spin_exchange_vec,
)
from vmc_torch.GPU.vmc_utils import (
    _check_grads_amps,
    has_precomputed_gpu_conn,
)


# Energy evaluation with bMPS reuse
# ==========================================================

def detect_changed_row_col_pair(fx1, fx2, Ly):
    """Classify which row(s) or col(s) differ between two configs.

    Compares two single-sample configurations, finds the (at most 2)
    sites that differ, converts to 2D coordinates. If the change spans
    fewer rows than cols, it's a "row change" (reuse x-direction bMPS);
    otherwise a "col change" (reuse y-direction bMPS).

    Args:
        fx1: (N_sites,) int64 — original config
        fx2: (N_sites,) int64 — connected config
        Ly: int — number of columns in the lattice

    Returns:
        (is_row, is_col, affected_indices):
            is_row=True, is_col=False, indices=list of row indices
            is_row=False, is_col=True, indices=list of col indices
            is_row=False, is_col=False, indices=None  (diagonal term)
    """
    changed_pos = torch.nonzero(fx1 - fx2)
    if changed_pos.shape[0] == 0:
        return False, False, None

    changed_pos_2d = []
    assert changed_pos.shape[0] <= 2, (
        "Expect at most 2 on-site config changes"
    )
    for pos in changed_pos:
        flat = pos.item()
        x, y = flat // Ly, flat % Ly
        changed_pos_2d.append((x, y))

    if len(changed_pos_2d) == 2:
        delta_row = abs(
            changed_pos_2d[0][0] - changed_pos_2d[1][0]
        )
        delta_col = abs(
            changed_pos_2d[0][1] - changed_pos_2d[1][1]
        )
        if delta_row <= delta_col:
            x1 = min(changed_pos_2d, key=lambda t: t[0])[0]
            return True, False, list(
                range(x1, x1 + delta_row + 1)
            )
        else:
            y1 = min(changed_pos_2d, key=lambda t: t[1])[1]
            return False, True, list(
                range(y1, y1 + delta_col + 1)
            )
    else:
        # Single-site change — treat as diagonal
        return False, False, None


def _slice_env_dict(env_dict, idxs):
    """Slice each pytree leaf tensor in env_dict by sample indices.

    Args:
        env_dict: {key: PyTree_of_Tensors} — batched bMPS params
        idxs: tensor of indices to slice

    Returns:
        {key: sliced_PyTree_of_Tensors}
    """
    return {
        k: qu.utils.tree_map(lambda x: x[idxs], v)
        for k, v in env_dict.items()
    }


@torch.inference_mode()
def evaluate_energy_reuse(
    fxs, model, H, current_amps,
    verbose=False, use_log_amp=False,
    return_bMPS=False, **kwargs,
):
    """Compute local energies using bMPS environment reuse.

    Groups connected configurations by which row(s)/col(s) change,
    then evaluates each group with the appropriate cached bMPS
    environments. Diagonal terms (x' == x) reuse current_amps
    directly.

    Args:
        fxs: (B, N_sites) int64 configurations.
        model: PEPS_Model_reuse_GPU with cache_bMPS_skeleton
            called.
        H: Hamiltonian with get_conn or get_conn_batch_gpu.
        current_amps: (B,) amplitudes at fxs. When
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
    Ly = model.Ly

    # Unpack log-amp current state if needed
    if use_log_amp:
        cur_signs, cur_log_abs = current_amps

    if verbose:
        t0 = time.time()

    # 1. Cache both x and y bMPS environments
    bMPS_x, bMPS_y = model.cache_bMPS_params_vmap(fxs)

    if verbose:
        t1 = time.time()
        print(f"  cache bMPS: {t1 - t0:.4f}s")

    # 2. Get connected configurations
    # --- GPU-batched path: zero CPU round-trips ---
    if has_precomputed_gpu_conn(H):
        if verbose:
            t0 = time.time()
        conn_etas, conn_eta_coeffs, batch_ids = (
            H.get_conn_batch_gpu(fxs)
        )
        conn_eta_num = torch.bincount(
            batch_ids, minlength=B,
        )
        if verbose:
            t1 = time.time()
            print(
                f"  GPU get_conn_batch time: "
                f"{t1 - t0:.4f}s"
                f" ({conn_etas.shape[0]} connected)"
            )

    # --- Fallback: CPU get_conn ---
    else:
        if verbose:
            t0 = time.time()
            print(
                "  Warning: falling back to CPU "
                "get_conn (no _hop_list)"
            )
        fxs_cpu = fxs.cpu()
        all_etas_np, all_coeffs_np = [], []
        conn_eta_num_list = []
        for fx in fxs_cpu:
            eta, coeffs = H.get_conn(fx)
            conn_eta_num_list.append(len(eta))
            all_etas_np.append(np.asarray(eta))
            all_coeffs_np.append(np.asarray(coeffs))
        conn_etas = torch.tensor(
            np.concatenate(all_etas_np), device=device,
        )
        conn_eta_coeffs = torch.tensor(
            np.concatenate(all_coeffs_np),
            device=device, dtype=torch.float64,
        )
        conn_eta_num = torch.tensor(
            conn_eta_num_list, device=device,
        )
        batch_ids = torch.repeat_interleave(
            torch.arange(B, device=device), conn_eta_num,
        )
        if verbose:
            t1 = time.time()
            print(
                f"  CPU get_conn: {t1 - t0:.4f}s"
                f" ({conn_etas.shape[0]} connected)"
            )

    # 3. Classify connected configs by change type
    # Vectorized: compare all conn configs vs parents at once
    if verbose:
        t0 = time.time()

    total_conn = conn_etas.shape[0]
    Lx = model.Lx
    radius = model.radius

    # (total_conn, N_sites) bool: which sites differ
    parent_fxs = fxs[batch_ids]  # (total_conn, N_sites)
    diff = (conn_etas != parent_fxs)  # on GPU

    # Reshape to (total_conn, Lx, Ly)
    diff_2d = diff.view(total_conn, Lx, Ly)
    row_changed = diff_2d.any(dim=2)  # (total_conn, Lx)
    col_changed = diff_2d.any(dim=1)  # (total_conn, Ly)

    # Count changed sites per config
    n_changed = diff.sum(dim=1)  # (total_conn,)

    # Diagonal: 0 changed sites, or exactly 1 (single-site
    # change treated as diagonal by detect_changed_row_col_pair)
    diagonal_mask = (n_changed <= 1)

    offdiag_mask = ~diagonal_mask
    offdiag_idxs = torch.nonzero(
        offdiag_mask,
    ).squeeze(-1)  # (n_offdiag,)

    batch_ids_cpu = batch_ids.cpu()

    tasks_map = {}
    if offdiag_idxs.numel() > 0:
        rc = row_changed[offdiag_idxs]  # (n_offdiag, Lx)
        cc = col_changed[offdiag_idxs]  # (n_offdiag, Ly)

        # Count changed rows/cols per config
        n_rows = rc.sum(dim=1)  # (n_offdiag,)
        n_cols = cc.sum(dim=1)  # (n_offdiag,)

        # delta_row = n_rows - 1, delta_col = n_cols - 1
        # is_row when delta_row <= delta_col, i.e. n_rows <= n_cols
        is_row_mask = (n_rows <= n_cols)  # (n_offdiag,)

        row_arange = torch.arange(Lx, device=device)
        col_arange = torch.arange(Ly, device=device)

        # For row changes: find min/max changed row
        row_vals_min = torch.where(
            rc, row_arange, torch.tensor(Lx, device=device),
        )
        rmin = row_vals_min.min(dim=1).values
        row_vals_max = torch.where(
            rc, row_arange, torch.tensor(-1, device=device),
        )
        rmax = row_vals_max.max(dim=1).values

        # For col changes: find min/max changed col
        col_vals_min = torch.where(
            cc, col_arange, torch.tensor(Ly, device=device),
        )
        cmin = col_vals_min.min(dim=1).values
        col_vals_max = torch.where(
            cc, col_arange, torch.tensor(-1, device=device),
        )
        cmax = col_vals_max.max(dim=1).values

        # Expand by radius and clamp
        row_lo = (rmin - radius).clamp(min=0)
        row_hi = (rmax + radius + 1).clamp(max=Lx)
        col_lo = (cmin - radius).clamp(min=0)
        col_hi = (cmax + radius + 1).clamp(max=Ly)

        # Move to CPU for dict grouping
        is_row_cpu = is_row_mask.cpu()
        row_lo_cpu = row_lo.cpu()
        row_hi_cpu = row_hi.cpu()
        col_lo_cpu = col_lo.cpu()
        col_hi_cpu = col_hi.cpu()
        offdiag_idxs_cpu = offdiag_idxs.cpu()

        for i in range(offdiag_idxs_cpu.shape[0]):
            k = offdiag_idxs_cpu[i].item()
            b = batch_ids_cpu[k].item()
            if is_row_cpu[i]:
                lo = row_lo_cpu[i].item()
                hi = row_hi_cpu[i].item()
                group_key = ('row', tuple(range(lo, hi)))
            else:
                lo = col_lo_cpu[i].item()
                hi = col_hi_cpu[i].item()
                group_key = ('col', tuple(range(lo, hi)))
            if group_key not in tasks_map:
                tasks_map[group_key] = {
                    'global_idxs': [],
                    'parent_idxs': [],
                }
            tasks_map[group_key]['global_idxs'].append(k)
            tasks_map[group_key]['parent_idxs'].append(b)

    n_diag = int(diagonal_mask.sum())
    n_groups = len(tasks_map)
    n_offdiag = total_conn - n_diag
    if verbose:
        t1 = time.time()
        print(
            f"  classify: {t1 - t0:.4f}s "
            f"({n_groups} groups, {n_diag} diagonal, "
            f"{n_offdiag} off-diagonal)"
        )

    # 4. Evaluate connected amplitudes
    if verbose:
        t0 = time.time()

    _amp_dtype = (
        cur_log_abs.dtype if use_log_amp
        else current_amps.dtype
    )

    if use_log_amp:
        conn_signs = torch.zeros(
            total_conn, dtype=_amp_dtype, device=device,
        )
        conn_log_abs = torch.zeros(
            total_conn, dtype=_amp_dtype, device=device,
        )
    else:
        conn_amps = torch.zeros(
            total_conn, dtype=_amp_dtype, device=device,
        )

    # A. Diagonal terms — direct copy (no forward pass)
    if n_diag > 0:
        diag_locs = torch.nonzero(
            diagonal_mask,
        ).squeeze(-1)
        parents = batch_ids[diag_locs]
        if use_log_amp:
            conn_signs[diag_locs] = cur_signs[parents]
            conn_log_abs[diag_locs] = cur_log_abs[parents]
        else:
            conn_amps[diag_locs] = current_amps[parents]

    # B. Non-diagonal terms — grouped reuse contraction.
    # Pad each chunk to fixed size B to avoid torch.compile
    # recompilation on varying batch sizes.
    chunk_counter = 0
    total_chunks = sum(
        (len(d['global_idxs']) + B - 1) // B
        for d in tasks_map.values()
    )

    for (mode, indices), data in tasks_map.items():
        global_idxs = data['global_idxs']
        parent_idxs = data['parent_idxs']

        for start in range(0, len(global_idxs), B):
            if verbose:
                t00 = time.time()
            chunk_counter += 1
            end = min(start + B, len(global_idxs))
            batch_global = global_idxs[start:end]
            batch_parents = parent_idxs[start:end]
            actual = len(batch_global)

            target_configs = conn_etas[batch_global]
            subset_parents = torch.tensor(
                batch_parents, device=device,
            )

            # Pad to fixed size B if needed
            if actual < B:
                pad = B - actual
                target_configs = torch.cat([
                    target_configs,
                    target_configs[:1].expand(pad, -1),
                ], dim=0)
                subset_parents = torch.cat([
                    subset_parents,
                    subset_parents[:1].expand(pad),
                ], dim=0)

            subset_env_x = _slice_env_dict(
                bMPS_x, subset_parents,
            )
            subset_env_y = _slice_env_dict(
                bMPS_y, subset_parents,
            )

            if use_log_amp:
                if mode == 'row':
                    cs, cla = model.forward_reuse_log(
                        target_configs,
                        bMPS_params_x_batched=subset_env_x,
                        selected_rows=list(indices),
                    )
                else:
                    cs, cla = model.forward_reuse_log(
                        target_configs,
                        bMPS_params_y_batched=subset_env_y,
                        selected_cols=list(indices),
                    )
                locs = torch.tensor(
                    batch_global, device=device,
                )
                conn_signs[locs] = cs[:actual]
                conn_log_abs[locs] = cla[:actual]
            else:
                if mode == 'row':
                    amps_chunk = model.forward_reuse(
                        target_configs,
                        bMPS_params_x_batched=subset_env_x,
                        selected_rows=list(indices),
                    )
                else:
                    amps_chunk = model.forward_reuse(
                        target_configs,
                        bMPS_params_y_batched=subset_env_y,
                        selected_cols=list(indices),
                    )
                locs = torch.tensor(
                    batch_global, device=device,
                )
                conn_amps[locs] = amps_chunk[:actual]

            if verbose:
                print(
                    f"  Evaluating connected amplitudes: "
                    f"chunk {chunk_counter} / "
                    f"{total_chunks} "
                    f"({mode} {list(indices)}, "
                    f"{actual} configs), "
                    f"delta t_forward: "
                    f"{time.time() - t00:.4f}s, "
                    f"total t_forward: "
                    f"{time.time() - t0:.4f}s"
                )

    if verbose:
        t1 = time.time()
        print(
            f"  GPU forward for connected configs "
            f"time: {t1 - t0:.4f}s"
        )

    # 5. Compute local energies via vectorized index_add_
    if use_log_amp:
        amp_ratio = (
            conn_signs
            * cur_signs[batch_ids]
            * torch.exp(
                conn_log_abs - cur_log_abs[batch_ids],
            )
        )
        terms = conn_eta_coeffs * amp_ratio
    else:
        current_amps_expanded = current_amps[batch_ids]
        terms = conn_eta_coeffs * (
            conn_amps / current_amps_expanded
        )

    local_energies = torch.zeros(
        B, device=device, dtype=terms.dtype,
    )
    local_energies.index_add_(0, batch_ids, terms)

    energy = torch.mean(local_energies)

    if verbose:
        print(f"  E_loc mean: {energy.item():.6f}")

    if return_bMPS:
        return energy, local_energies, bMPS_x, bMPS_y
    return energy, local_energies


def detect_changed_rows(fx1, fx2, Ly):
    """Find which rows differ between two configs.

    Returns sorted list of changed row indices,
    or None if configs are identical (diagonal term).
    """
    changed_pos = torch.nonzero(fx1 - fx2)
    if changed_pos.shape[0] == 0:
        return None
    rows = set()
    for pos in changed_pos:
        rows.add(pos.item() // Ly)
    return sorted(rows)


@torch.inference_mode()
def evaluate_energy_reuse_x(
    fxs, model, H, current_amps,
    verbose=False, use_log_amp=False,
    return_bMPS=False, **kwargs,
):
    """Compute local energies using x-only bMPS reuse.

    Like evaluate_energy_reuse but caches only x-direction bMPS
    and handles all connected configs (row AND column edge hops)
    via selected_rows. Matches the XReuse sampler approach.

    Args:
        fxs: (B, N_sites) int64 configurations.
        model: PEPS model with cache_bMPS_params_any_direction_vmap.
        H: Hamiltonian with get_conn or get_conn_batch_gpu.
        current_amps: (B,) amplitudes at fxs. When
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
    Ly = model.Ly

    # Unpack log-amp current state if needed
    if use_log_amp:
        cur_signs, cur_log_abs = current_amps

    if verbose:
        t0 = time.time()

    # 1. Cache x-direction bMPS only
    bMPS_x, current_amps_from_cache = (
        model.cache_bMPS_params_any_direction_vmap(
            fxs, direction='x',
        )
    )

    if verbose:
        t1 = time.time()
        print(f"  cache bMPS (x-only): {t1 - t0:.4f}s")

    # 2. Get connected configurations
    # --- GPU-batched path: zero CPU round-trips ---
    if has_precomputed_gpu_conn(H):
        if verbose:
            t0 = time.time()
        conn_etas, conn_eta_coeffs, batch_ids = (
            H.get_conn_batch_gpu(fxs)
        )
        conn_eta_num = torch.bincount(
            batch_ids, minlength=B,
        )
        if verbose:
            t1 = time.time()
            print(
                f"  GPU get_conn_batch time: "
                f"{t1 - t0:.4f}s"
                f" ({conn_etas.shape[0]} connected)"
            )

    # --- Fallback: CPU get_conn ---
    else:
        if verbose:
            t0 = time.time()
            print(
                "  Warning: falling back to CPU "
                "get_conn (no _hop_list)"
            )
        fxs_cpu = fxs.cpu()
        all_etas_np, all_coeffs_np = [], []
        conn_eta_num_list = []
        for fx in fxs_cpu:
            eta, coeffs = H.get_conn(fx)
            conn_eta_num_list.append(len(eta))
            all_etas_np.append(np.asarray(eta))
            all_coeffs_np.append(np.asarray(coeffs))
        conn_etas = torch.tensor(
            np.concatenate(all_etas_np), device=device,
        )
        conn_eta_coeffs = torch.tensor(
            np.concatenate(all_coeffs_np),
            device=device, dtype=torch.float64,
        )
        conn_eta_num = torch.tensor(
            conn_eta_num_list, device=device,
        )
        batch_ids = torch.repeat_interleave(
            torch.arange(B, device=device), conn_eta_num,
        )
        if verbose:
            t1 = time.time()
            print(
                f"  CPU get_conn: {t1 - t0:.4f}s"
                f" ({conn_etas.shape[0]} connected)"
            )

    # 3. Classify connected configs — all as row changes
    # Vectorized: compare all conn configs vs parents at once
    if verbose:
        t0 = time.time()

    total_conn = conn_etas.shape[0]
    Lx = model.Lx
    radius = model.radius

    # (total_conn, N_sites) bool: which sites differ
    parent_fxs = fxs[batch_ids]  # (total_conn, N_sites)
    diff = (conn_etas != parent_fxs)  # on GPU

    # Reshape to (total_conn, Lx, Ly), check which rows changed
    diff_2d = diff.view(total_conn, Lx, Ly)
    row_changed = diff_2d.any(dim=2)  # (total_conn, Lx)

    # Diagonal: no sites differ
    diagonal_mask = ~diff.any(dim=1)  # (total_conn,)

    # For non-diagonal, encode row pattern as integer key
    # row_changed is (total_conn, Lx) bool on GPU
    # Compute min/max changed row per config, then expand
    # by radius to get the group key.
    # Use argmax tricks on GPU to find min/max row.
    offdiag_mask = ~diagonal_mask
    offdiag_idxs = torch.nonzero(
        offdiag_mask,
    ).squeeze(-1)  # (n_offdiag,)

    batch_ids_cpu = batch_ids.cpu()

    tasks_map = {}
    if offdiag_idxs.numel() > 0:
        rc = row_changed[offdiag_idxs]  # (n_offdiag, Lx)
        # row indices: 0..Lx-1
        row_arange = torch.arange(Lx, device=device)
        # min changed row: first True in each row
        # Set False positions to Lx so they don't win min
        row_vals = torch.where(
            rc, row_arange, torch.tensor(Lx, device=device),
        )
        rmin = row_vals.min(dim=1).values  # (n_offdiag,)
        # max changed row
        row_vals_max = torch.where(
            rc, row_arange, torch.tensor(-1, device=device),
        )
        rmax = row_vals_max.max(dim=1).values  # (n_offdiag,)

        # Expand by radius and clamp
        pos_min = (rmin - radius).clamp(min=0)
        pos_max = (rmax + radius + 1).clamp(max=Lx)

        # Encode group key as (pos_min, pos_max) pair
        # Move to CPU for dict grouping
        pos_min_cpu = pos_min.cpu()
        pos_max_cpu = pos_max.cpu()
        offdiag_idxs_cpu = offdiag_idxs.cpu()

        for i in range(offdiag_idxs_cpu.shape[0]):
            k = offdiag_idxs_cpu[i].item()
            b = batch_ids_cpu[k].item()
            lo = pos_min_cpu[i].item()
            hi = pos_max_cpu[i].item()
            group_key = ('row', tuple(range(lo, hi)))
            if group_key not in tasks_map:
                tasks_map[group_key] = {
                    'global_idxs': [],
                    'parent_idxs': [],
                }
            tasks_map[group_key]['global_idxs'].append(k)
            tasks_map[group_key]['parent_idxs'].append(b)

    n_diag = int(diagonal_mask.sum())
    n_groups = len(tasks_map)
    n_offdiag = total_conn - n_diag
    if verbose:
        t1 = time.time()
        print(
            f"  classify (x-only): {t1 - t0:.4f}s "
            f"({n_groups} groups, {n_diag} diagonal, "
            f"{n_offdiag} off-diagonal)"
        )

    # 4. Evaluate connected amplitudes
    if verbose:
        t0 = time.time()

    _amp_dtype = (
        cur_log_abs.dtype if use_log_amp
        else current_amps.dtype
    )

    if use_log_amp:
        conn_signs = torch.zeros(
            total_conn, dtype=_amp_dtype, device=device,
        )
        conn_log_abs = torch.zeros(
            total_conn, dtype=_amp_dtype, device=device,
        )
    else:
        conn_amps = torch.zeros(
            total_conn, dtype=_amp_dtype, device=device,
        )

    # A. Diagonal terms — direct copy (no forward pass)
    if n_diag > 0:
        diag_locs = torch.nonzero(
            diagonal_mask,
        ).squeeze(-1)
        parents = batch_ids[diag_locs]
        if use_log_amp:
            conn_signs[diag_locs] = cur_signs[parents]
            conn_log_abs[diag_locs] = cur_log_abs[parents]
        else:
            conn_amps[diag_locs] = current_amps[parents]

    # B. Non-diagonal terms — grouped x-direction reuse.
    # Pad each chunk to fixed size B to avoid torch.compile
    # recompilation on varying batch sizes.
    chunk_counter = 0
    total_chunks = sum(
        (len(d['global_idxs']) + B - 1) // B
        for d in tasks_map.values()
    )

    for (mode, indices), data in tasks_map.items():
        global_idxs = data['global_idxs']
        parent_idxs = data['parent_idxs']

        for start in range(0, len(global_idxs), B):
            if verbose:
                t00 = time.time()
            chunk_counter += 1
            end = min(start + B, len(global_idxs))
            batch_global = global_idxs[start:end]
            batch_parents = parent_idxs[start:end]
            actual = len(batch_global)

            target_configs = conn_etas[batch_global]
            subset_parents = torch.tensor(
                batch_parents, device=device,
            )

            # Pad to fixed size B if needed
            if actual < B:
                pad = B - actual
                target_configs = torch.cat([
                    target_configs,
                    target_configs[:1].expand(pad, -1),
                ], dim=0)
                subset_parents = torch.cat([
                    subset_parents,
                    subset_parents[:1].expand(pad),
                ], dim=0)

            subset_env_x = _slice_env_dict(
                bMPS_x, subset_parents,
            )

            if use_log_amp:
                cs, cla = model.forward_reuse_log(
                    target_configs,
                    bMPS_params_x_batched=subset_env_x,
                    selected_rows=list(indices),
                )
                locs = torch.tensor(
                    batch_global, device=device,
                )
                conn_signs[locs] = cs[:actual]
                conn_log_abs[locs] = cla[:actual]
            else:
                amps_chunk = model.forward_reuse(
                    target_configs,
                    bMPS_params_x_batched=subset_env_x,
                    selected_rows=list(indices),
                )
                locs = torch.tensor(
                    batch_global, device=device,
                )
                conn_amps[locs] = amps_chunk[:actual]

            if verbose:
                print(
                    f"  Evaluating connected amplitudes: "
                    f"chunk {chunk_counter} / "
                    f"{total_chunks} "
                    f"({mode} {list(indices)}, "
                    f"{actual} configs), "
                    f"delta t_forward: "
                    f"{time.time() - t00:.4f}s, "
                    f"total t_forward: "
                    f"{time.time() - t0:.4f}s"
                )

    if verbose:
        t1 = time.time()
        print(
            f"  GPU forward for connected configs "
            f"time: {t1 - t0:.4f}s"
        )

    # 5. Compute local energies via vectorized index_add_
    if use_log_amp:
        amp_ratio = (
            conn_signs
            * cur_signs[batch_ids]
            * torch.exp(
                conn_log_abs - cur_log_abs[batch_ids],
            )
        )
        terms = conn_eta_coeffs * amp_ratio
    else:
        current_amps_expanded = current_amps[batch_ids]
        terms = conn_eta_coeffs * (
            conn_amps / current_amps_expanded
        )

    local_energies = torch.zeros(
        B, device=device, dtype=terms.dtype,
    )
    local_energies.index_add_(0, batch_ids, terms)

    energy = torch.mean(local_energies)

    if verbose:
        print(f"  E_loc mean: {energy.item():.6f}")

    if return_bMPS:
        return energy, local_energies, bMPS_x
    return energy, local_energies


# ==========================================================
# Gradient computation with cached environments
# ==========================================================


def compute_grads_cheap_gpu(
    fxs, fpeps_model, batch_size=None,
    offload_to_cpu=False, use_log_amp=False,
    verbose=False, bMPS_params_x=None,
    **kwargs,
):
    """Cheap gradient via per-row hole contraction for pure fTNS.

    Avoids backprop through SVDs in boundary contraction by treating
    cached bMPS environments as constants. Requires the model to be
    an fPEPS_Model_reuse_GPU with bMPS skeletons initialized.

    Interface matches compute_grads_gpu for drop-in replacement.

    Args:
        fxs: (B, N_sites) int64 configurations.
        fpeps_model: fPEPS_Model_reuse_GPU with cache_bMPS_skeleton
            called.
        batch_size: gradient chunk size (like grad_batch_size).
        offload_to_cpu: move grad chunks to CPU eagerly.
        use_log_amp: if True, return log-amplitude gradients.
        verbose: print timing info.
        bMPS_params_x: pre-computed batched x-env params from
            evaluate_energy_reuse. If None, recomputes them.

    Returns:
        use_log_amp=False: (grads (B, Np), amps (B,))
        use_log_amp=True:  (grads (B, Np), (signs (B,), log_abs (B,)))
    """
    B = fxs.shape[0]
    device = fxs.device
    B_grad = batch_size if batch_size is not None else B

    # Compute bMPS environments if not provided
    with torch.no_grad():
        if bMPS_params_x is None:
            if verbose:
                t0 = time.time()
            bMPS_x, _ = fpeps_model.cache_bMPS_params_vmap(fxs)
            bMPS_params_x = bMPS_x
            if verbose:
                print(
                    f"  cheap_grad: cache bMPS: "
                    f"{time.time() - t0:.4f}s"
                )

    t0 = time.time()

    # Pre-allocate output buffers
    leaves_p, _ = tree_flatten(
        list(fpeps_model.params),
    )
    Np = sum(p.numel() for p in leaves_p)
    dtype = leaves_p[0].dtype
    del leaves_p

    if offload_to_cpu:
        pin_cpu = torch.cuda.is_available()
        batched_grads_vec = torch.empty(
            B, Np, dtype=dtype, device='cpu',
            pin_memory=pin_cpu,
        )
        if use_log_amp:
            signs = torch.empty(
                B, dtype=dtype, device='cpu',
                pin_memory=pin_cpu,
            )
            log_abs = torch.empty(
                B, dtype=dtype, device='cpu',
                pin_memory=pin_cpu,
            )
        else:
            amps = torch.empty(
                B, dtype=dtype, device='cpu',
                pin_memory=pin_cpu,
            )
    else:
        batched_grads_vec = torch.empty(
            B, Np, dtype=dtype, device=device,
        )
        if use_log_amp:
            signs = torch.empty(B, dtype=dtype, device=device)
            log_abs = torch.empty(
                B, dtype=dtype, device=device,
            )
        else:
            amps = torch.empty(B, dtype=dtype, device=device)

    for b_start in range(0, B, B_grad):
        b_end = min(b_start + B_grad, B)
        if verbose:
            print(
                f"  cheap_grad chunk: "
                f"{b_start} to {b_end} / {B}"
            )

        fxs_chunk = fxs[b_start:b_end]
        bMPS_chunk = _slice_env_dict(
            bMPS_params_x,
            torch.arange(
                b_start, b_end, device=device,
            ),
        )

        grads_chunk, amps_chunk = (
            fpeps_model.compute_cheap_grads_vmap(
                fxs_chunk, bMPS_chunk,
            )
        )
        grads_chunk = grads_chunk.detach()
        amps_chunk = amps_chunk.detach()

        if use_log_amp:
            s_chunk = torch.sign(amps_chunk)
            la_chunk = torch.log(
                amps_chunk.abs().clamp(min=1e-45),
            )
            # Convert raw grad to log-amplitude grad:
            # d(log|psi|)/dp = (1/psi) * dpsi/dp
            grads_chunk = (
                grads_chunk / amps_chunk.unsqueeze(1)
            )

        if offload_to_cpu:
            batched_grads_vec[b_start:b_end].copy_(
                grads_chunk, non_blocking=True,
            )
            if use_log_amp:
                signs[b_start:b_end].copy_(
                    s_chunk, non_blocking=True,
                )
                log_abs[b_start:b_end].copy_(
                    la_chunk, non_blocking=True,
                )
            else:
                amps[b_start:b_end].copy_(
                    amps_chunk, non_blocking=True,
                )
        else:
            batched_grads_vec[b_start:b_end] = grads_chunk
            if use_log_amp:
                signs[b_start:b_end] = s_chunk
                log_abs[b_start:b_end] = la_chunk
            else:
                amps[b_start:b_end] = amps_chunk

        del grads_chunk, amps_chunk

    t1 = time.time()
    if verbose:
        print(f"  cheap_grad total: {t1 - t0:.4f}s")

    _check_grads_amps(
        batched_grads_vec,
        log_abs if use_log_amp else amps,
        fpeps_model,
    )

    if use_log_amp:
        return batched_grads_vec, (signs, log_abs)
    return batched_grads_vec, amps


# ==========================================================
# Metropolis samplers with bMPS reuse
# ==========================================================


class MetropolisExchangeSpinfulSamplerReuse_GPU(SamplerGPU):
    """Metropolis exchange sampler with bMPS environment reuse.

    Two-phase sweep: first x-direction (row edges) with cached
    x-boundary MPS, then y-direction (col edges) with cached
    y-boundary MPS. After processing each row/col, the boundary
    MPS is incrementally updated rather than recomputed from
    scratch.

    Requires model to be fPEPS_Model_reuse_GPU with
    cache_bMPS_skeleton() already called.

    Args:
        hopping_rate: Fraction of proposals that are
            hoppings (vs exchanges). Default 0.25.
    """

    def __init__(self, hopping_rate: float = 0.25):
        self.hopping_rate = hopping_rate

    @torch.inference_mode()
    def step(
        self,
        fxs: torch.Tensor,
        model,
        graph,
        compile: bool = False,
        verbose: bool = False,
        use_log_amp: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One MCMC sweep with bMPS environment reuse.

        Phase 1: sweep row edges using x-direction bMPS.
        Phase 2: sweep col edges using y-direction bMPS.

        Args:
            fxs: (B, N_sites) int64 walker configs.
            model: fPEPS_Model_reuse_GPU with cache_bMPS_skeleton
                already called.
            graph: Lattice graph with .row_edges, .col_edges.
            compile: Unused (kept for interface compat).
            verbose: Print per-phase timing info.
            use_log_amp: If True, work in log-space and
                return (signs, log_abs) instead of amps.

        Returns:
            fxs: (B, N_sites) int64 updated configs.
            amps_out: (B,) amplitudes, or (signs, log_abs)
                tuple when use_log_amp=True.
        """
        B = fxs.shape[0]
        device = fxs.device

        # Collect all edges for progress tracking
        all_edges = []
        for edges in graph.row_edges.values():
            all_edges.extend(edges)
        for edges in graph.col_edges.values():
            all_edges.extend(edges)
        total_edges = len(all_edges)

        n_updates = 0
        if verbose:
            t_total_start = time.time()
            t_propose = 0.0
            t_forward = 0.0

        # ---- Phase 1: x-direction (row edges) ----
        if verbose:
            t0 = time.time()
        bMPS_x, current_amps = (
            model.cache_bMPS_params_any_direction_vmap(
                fxs, direction='x', sides='xmax',
            )
        )
        if use_log_amp:
            cur_signs = torch.sign(current_amps)
            cur_log_abs = torch.log(
                current_amps.abs().clamp(min=1e-45),
            )
        if verbose:
            print(
                f" cache bMPS x: "
                f"{time.time() - t0:.4f}s"
            )

        for row, edges in graph.row_edges.items():
            for edge in edges:
                n_updates += 1
                i, j = edge

                if verbose:
                    t00 = time.time()
                proposed_fxs, new_flags = (
                    propose_exchange_or_hopping_vec(
                        i, j, fxs, self.hopping_rate,
                    )
                )
                if verbose:
                    t11 = time.time()
                    t_propose += t11 - t00

                if not new_flags.any():
                    continue

                n_changed = new_flags.sum().item()

                # Determine which rows to contract
                selected_rows = list(range(
                    max(0, row - model.radius),
                    min(model.Lx, row + model.radius + 1),
                ))

                if verbose:
                    t10 = time.time()

                if use_log_amp:
                    prop_signs, prop_log_abs = (
                        model.forward_reuse_log(
                            proposed_fxs,
                            bMPS_params_x_batched=bMPS_x,
                            selected_rows=selected_rows,
                        )
                    )
                    ratio = torch.exp(
                        2.0 * (prop_log_abs - cur_log_abs),
                    )
                else:
                    proposed_amps = model.forward_reuse(
                        proposed_fxs,
                        bMPS_params_x_batched=bMPS_x,
                        selected_rows=selected_rows,
                    )
                    ratio = (
                        (proposed_amps.abs() ** 2)
                        / (current_amps.abs() ** 2)
                    )

                if verbose:
                    t11 = time.time()
                    t_forward += t11 - t10
                    print(
                        f" Edge ({i}, {j}): "
                        f"{n_changed} / {B} "
                        f"proposed, forward: "
                        f"{t11-t10:.4f}s, "
                        f"total forward: "
                        f"{t_forward:.4f}s, "
                        f"progress: "
                        f"{n_updates}/{total_edges}"
                    )

                probs = torch.rand(B, device=device)
                accept_mask = new_flags & (probs < ratio)

                if accept_mask.any():
                    fxs[accept_mask] = (
                        proposed_fxs[accept_mask]
                    )
                    if use_log_amp:
                        cur_signs[accept_mask] = (
                            prop_signs[accept_mask]
                        )
                        cur_log_abs[accept_mask] = (
                            prop_log_abs[accept_mask]
                        )
                    else:
                        current_amps[accept_mask] = (
                            proposed_amps[accept_mask]
                        )

            # Update bMPS to next row
            if row < model.Lx - 1:
                bMPS_x = model.update_bMPS_params_to_row_vmap(
                    fxs, row, bMPS_x, from_which='xmin',
                )

        # ---- Phase 2: y-direction (col edges) ----
        if verbose:
            t0 = time.time()
        bMPS_y, current_amps = (
            model.cache_bMPS_params_any_direction_vmap(
                fxs, direction='y', sides='ymax',
            )
        )
        if use_log_amp:
            cur_signs = torch.sign(current_amps)
            cur_log_abs = torch.log(
                current_amps.abs().clamp(min=1e-45),
            )
        if verbose:
            print(
                f" cache bMPS y: "
                f"{time.time() - t0:.4f}s"
            )

        for col, edges in graph.col_edges.items():
            for edge in edges:
                n_updates += 1
                i, j = edge

                if verbose:
                    t00 = time.time()
                proposed_fxs, new_flags = (
                    propose_exchange_or_hopping_vec(
                        i, j, fxs, self.hopping_rate,
                    )
                )
                if verbose:
                    t11 = time.time()
                    t_propose += t11 - t00

                if not new_flags.any():
                    continue

                n_changed = new_flags.sum().item()

                selected_cols = list(range(
                    max(0, col - model.radius),
                    min(model.Ly, col + model.radius + 1),
                ))

                if verbose:
                    t10 = time.time()

                if use_log_amp:
                    prop_signs, prop_log_abs = (
                        model.forward_reuse_log(
                            proposed_fxs,
                            bMPS_params_y_batched=bMPS_y,
                            selected_cols=selected_cols,
                        )
                    )
                    ratio = torch.exp(
                        2.0 * (prop_log_abs - cur_log_abs),
                    )
                else:
                    proposed_amps = model.forward_reuse(
                        proposed_fxs,
                        bMPS_params_y_batched=bMPS_y,
                        selected_cols=selected_cols,
                    )
                    ratio = (
                        (proposed_amps.abs() ** 2)
                        / (current_amps.abs() ** 2)
                    )

                if verbose:
                    t11 = time.time()
                    t_forward += t11 - t10
                    print(
                        f" Edge ({i}, {j}): "
                        f"{n_changed} / {B} "
                        f"proposed, forward: "
                        f"{t11-t10:.4f}s, "
                        f"total forward: "
                        f"{t_forward:.4f}s, "
                        f"progress: "
                        f"{n_updates}/{total_edges}"
                    )

                probs = torch.rand(B, device=device)
                accept_mask = new_flags & (probs < ratio)

                if accept_mask.any():
                    fxs[accept_mask] = (
                        proposed_fxs[accept_mask]
                    )
                    if use_log_amp:
                        cur_signs[accept_mask] = (
                            prop_signs[accept_mask]
                        )
                        cur_log_abs[accept_mask] = (
                            prop_log_abs[accept_mask]
                        )
                    else:
                        current_amps[accept_mask] = (
                            proposed_amps[accept_mask]
                        )

            # Update bMPS to next col
            if col < model.Ly - 1:
                bMPS_y = model.update_bMPS_params_to_col_vmap(
                    fxs, col, bMPS_y, from_which='ymin',
                )

        if verbose:
            t1 = time.time()
            print(
                f"Sample next: "
                f"{t1-t_total_start:.4f}s for "
                f"{n_updates} edges "
                f"(avg "
                f"{(t1-t_total_start)/n_updates:.4f}"
                f"s/edge, B={B})"
            )
            print(
                f"  Propose: {t_propose:.4f}s "
                f"(avg "
                f"{t_propose/n_updates:.4f}s/edge)"
            )
            print(
                f"  Forward: {t_forward:.4f}s "
                f"(avg "
                f"{t_forward/n_updates:.4f}s/edge)"
            )

        if use_log_amp:
            return fxs, (cur_signs, cur_log_abs)
        return fxs, current_amps


class MetropolisExchangeSpinSamplerReuse_GPU(SamplerGPU):
    """Metropolis exchange sampler with bMPS reuse for spins.

    Two-phase sweep: x-direction (row edges) with cached
    x-boundary MPS, then y-direction (col edges) with cached
    y-boundary MPS. After each row/col, the boundary MPS is
    incrementally updated.

    Requires model to be PEPS_Model_reuse_GPU with
    cache_bMPS_skeleton() already called.
    """

    @torch.inference_mode()
    def step(
        self,
        fxs: torch.Tensor,
        model,
        graph,
        compile: bool = False,
        verbose: bool = False,
        use_log_amp: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One MCMC sweep with bMPS environment reuse.

        Phase 1: sweep row edges using x-direction bMPS.
        Phase 2: sweep col edges using y-direction bMPS.

        Args:
            fxs: (B, N_sites) int64 walker configs.
            model: PEPS_Model_reuse_GPU with
                cache_bMPS_skeleton already called.
            graph: Lattice graph with .row_edges,
                .col_edges.
            compile: Unused (kept for interface compat).
            verbose: Print per-phase timing info.
            use_log_amp: If True, work in log-space and
                return (signs, log_abs) instead of amps.

        Returns:
            fxs: (B, N_sites) int64 updated configs.
            amps_out: (B,) amplitudes, or (signs, log_abs)
                tuple when use_log_amp=True.
        """
        B = fxs.shape[0]
        device = fxs.device

        # Collect all edges for progress tracking
        all_edges = []
        for edges in graph.row_edges.values():
            all_edges.extend(edges)
        for edges in graph.col_edges.values():
            all_edges.extend(edges)
        total_edges = len(all_edges)

        n_updates = 0
        if verbose:
            t_total_start = time.time()
            t_propose = 0.0
            t_forward = 0.0

        # ---- Phase 1: x-direction (row edges) ----
        if verbose:
            t0 = time.time()
        bMPS_x, current_amps = (
            model.cache_bMPS_params_any_direction_vmap(
                fxs, direction='x',
            )
        )
        if use_log_amp:
            cur_signs = torch.sign(current_amps)
            cur_log_abs = torch.log(
                current_amps.abs().clamp(min=1e-45),
            )
        if verbose:
            print(
                f" cache bMPS x: "
                f"{time.time() - t0:.4f}s"
            )

        for row, edges in graph.row_edges.items():
            for edge in edges:
                n_updates += 1
                i, j = edge

                if verbose:
                    t00 = time.time()
                proposed_fxs, new_flags = (
                    propose_spin_exchange_vec(i, j, fxs)
                )
                if verbose:
                    t11 = time.time()
                    t_propose += t11 - t00

                if not new_flags.any():
                    continue

                n_changed = new_flags.sum().item()

                selected_rows = list(range(
                    max(0, row - model.radius),
                    min(model.Lx, row + model.radius + 1),
                ))

                if verbose:
                    t10 = time.time()

                if use_log_amp:
                    prop_signs, prop_log_abs = (
                        model.forward_reuse_log(
                            proposed_fxs,
                            bMPS_params_x_batched=bMPS_x,
                            selected_rows=selected_rows,
                        )
                    )
                    ratio = torch.exp(
                        2.0 * (prop_log_abs - cur_log_abs),
                    )
                else:
                    proposed_amps = model.forward_reuse(
                        proposed_fxs,
                        bMPS_params_x_batched=bMPS_x,
                        selected_rows=selected_rows,
                    )
                    ratio = (
                        (proposed_amps.abs() ** 2)
                        / (current_amps.abs() ** 2)
                    )

                if verbose:
                    t11 = time.time()
                    t_forward += t11 - t10
                    print(
                        f" Edge ({i}, {j}): "
                        f"{n_changed} / {B} "
                        f"proposed, forward: "
                        f"{t11-t10:.4f}s, "
                        f"total forward: "
                        f"{t_forward:.4f}s, "
                        f"progress: "
                        f"{n_updates}/{total_edges}"
                    )

                probs = torch.rand(B, device=device)
                accept_mask = new_flags & (probs < ratio)

                if accept_mask.any():
                    fxs[accept_mask] = (
                        proposed_fxs[accept_mask]
                    )
                    if use_log_amp:
                        cur_signs[accept_mask] = (
                            prop_signs[accept_mask]
                        )
                        cur_log_abs[accept_mask] = (
                            prop_log_abs[accept_mask]
                        )
                    else:
                        current_amps[accept_mask] = (
                            proposed_amps[accept_mask]
                        )

            # Update bMPS to next row
            if row < model.Lx - 1:
                bMPS_x = (
                    model.update_bMPS_params_to_row_vmap(
                        fxs, row, bMPS_x,
                        from_which='xmin',
                    )
                )

        # ---- Phase 2: y-direction (col edges) ----
        if verbose:
            t0 = time.time()
        bMPS_y, current_amps = (
            model.cache_bMPS_params_any_direction_vmap(
                fxs, direction='y',
            )
        )
        if use_log_amp:
            cur_signs = torch.sign(current_amps)
            cur_log_abs = torch.log(
                current_amps.abs().clamp(min=1e-45),
            )
        if verbose:
            print(
                f" cache bMPS y: "
                f"{time.time() - t0:.4f}s"
            )

        for col, edges in graph.col_edges.items():
            for edge in edges:
                n_updates += 1
                i, j = edge

                if verbose:
                    t00 = time.time()
                proposed_fxs, new_flags = (
                    propose_spin_exchange_vec(i, j, fxs)
                )
                if verbose:
                    t11 = time.time()
                    t_propose += t11 - t00

                if not new_flags.any():
                    continue

                n_changed = new_flags.sum().item()

                selected_cols = list(range(
                    max(0, col - model.radius),
                    min(model.Ly, col + model.radius + 1),
                ))

                if verbose:
                    t10 = time.time()

                if use_log_amp:
                    prop_signs, prop_log_abs = (
                        model.forward_reuse_log(
                            proposed_fxs,
                            bMPS_params_y_batched=bMPS_y,
                            selected_cols=selected_cols,
                        )
                    )
                    ratio = torch.exp(
                        2.0 * (prop_log_abs - cur_log_abs),
                    )
                else:
                    proposed_amps = model.forward_reuse(
                        proposed_fxs,
                        bMPS_params_y_batched=bMPS_y,
                        selected_cols=selected_cols,
                    )
                    ratio = (
                        (proposed_amps.abs() ** 2)
                        / (current_amps.abs() ** 2)
                    )

                if verbose:
                    t11 = time.time()
                    t_forward += t11 - t10
                    print(
                        f" Edge ({i}, {j}): "
                        f"{n_changed} / {B} "
                        f"proposed, forward: "
                        f"{t11-t10:.4f}s, "
                        f"total forward: "
                        f"{t_forward:.4f}s, "
                        f"progress: "
                        f"{n_updates}/{total_edges}"
                    )

                probs = torch.rand(B, device=device)
                accept_mask = new_flags & (probs < ratio)

                if accept_mask.any():
                    fxs[accept_mask] = (
                        proposed_fxs[accept_mask]
                    )
                    if use_log_amp:
                        cur_signs[accept_mask] = (
                            prop_signs[accept_mask]
                        )
                        cur_log_abs[accept_mask] = (
                            prop_log_abs[accept_mask]
                        )
                    else:
                        current_amps[accept_mask] = (
                            proposed_amps[accept_mask]
                        )

            # Update bMPS to next col
            if col < model.Ly - 1:
                bMPS_y = (
                    model.update_bMPS_params_to_col_vmap(
                        fxs, col, bMPS_y,
                        from_which='ymin',
                    )
                )

        if verbose:
            t1 = time.time()
            print(
                f"Sample next: "
                f"{t1-t_total_start:.4f}s for "
                f"{n_updates} edges "
                f"(avg "
                f"{(t1-t_total_start)/n_updates:.4f}"
                f"s/edge, B={B})"
            )
            print(
                f"  Propose: {t_propose:.4f}s "
                f"(avg "
                f"{t_propose/n_updates:.4f}s/edge)"
            )
            print(
                f"  Forward: {t_forward:.4f}s "
                f"(avg "
                f"{t_forward/n_updates:.4f}s/edge)"
            )

        if use_log_amp:
            return fxs, (cur_signs, cur_log_abs)
        return fxs, current_amps


class MetropolisExchangeSpinSamplerXReuse_GPU(SamplerGPU):
    """Metropolis exchange sampler with x-only bMPS reuse.

    Interleaved sweep: for each row, processes row edges
    (horizontal) then col edges (vertical to next row),
    all using x-direction boundary MPS only. Eliminates
    the expensive y-direction bMPS caching.
    """

    @torch.inference_mode()
    def step(
        self,
        fxs: torch.Tensor,
        model,
        graph,
        compile: bool = False,
        verbose: bool = False,
        use_log_amp: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One MCMC sweep with x-only bMPS reuse.

        For each row r:
          (a) sweep row edges in row r
          (b) sweep col edges between rows r and r+1
          (c) update xmin bMPS past row r

        Args:
            fxs: (B, N_sites) int64 walker configs.
            model: PEPS_Model_reuse_GPU with
                cache_bMPS_skeleton already called.
            graph: Lattice graph with .row_edges,
                .col_edges.
            compile: Unused (kept for interface compat).
            verbose: Print per-phase timing info.
            use_log_amp: If True, work in log-space and
                return (signs, log_abs) instead of amps.

        Returns:
            fxs: (B, N_sites) int64 updated configs.
            amps_out: (B,) amplitudes, or (signs, log_abs)
                tuple when use_log_amp=True.
        """
        B = fxs.shape[0]
        device = fxs.device
        Ly = model.Ly
        Lx = model.Lx

        # Pre-group col edges by row pair
        col_edges_by_row_pair = {}
        for col, edges in graph.col_edges.items():
            for (i, j) in edges:
                r = min(i // Ly, j // Ly)
                col_edges_by_row_pair.setdefault(
                    r, []
                ).append((i, j))

        # Count edges for progress tracking
        total_edges = sum(
            len(e) for e in graph.row_edges.values()
        ) + sum(
            len(e) for e in graph.col_edges.values()
        )

        n_updates = 0
        if verbose:
            t_total_start = time.time()
            t_propose = 0.0
            t_forward = 0.0

        # Cache x-direction bMPS only
        if verbose:
            t0 = time.time()
        bMPS_x, current_amps = (
            model.cache_bMPS_params_any_direction_vmap(
                fxs, direction='x',
            )
        )
        if use_log_amp:
            cur_signs = torch.sign(current_amps)
            cur_log_abs = torch.log(
                current_amps.abs().clamp(min=1e-45),
            )
        if verbose:
            print(
                f" cache bMPS x: "
                f"{time.time() - t0:.4f}s"
            )

        for row in range(Lx):
            # (a) Row edges in this row
            if row in graph.row_edges:
                selected_rows = list(range(
                    max(0, row - model.radius),
                    min(Lx, row + model.radius + 1),
                ))
                for edge in graph.row_edges[row]:
                    n_updates += 1
                    i, j = edge

                    if verbose:
                        t00 = time.time()
                    proposed_fxs, new_flags = (
                        propose_spin_exchange_vec(
                            i, j, fxs,
                        )
                    )
                    if verbose:
                        t11 = time.time()
                        t_propose += t11 - t00

                    if not new_flags.any():
                        continue

                    n_changed = new_flags.sum().item()

                    if verbose:
                        t10 = time.time()

                    if use_log_amp:
                        prop_signs, prop_log_abs = (
                            model.forward_reuse_log(
                                proposed_fxs,
                                bMPS_params_x_batched=(
                                    bMPS_x
                                ),
                                selected_rows=(
                                    selected_rows
                                ),
                            )
                        )
                        ratio = torch.exp(
                            2.0
                            * (prop_log_abs - cur_log_abs),
                        )
                    else:
                        proposed_amps = (
                            model.forward_reuse(
                                proposed_fxs,
                                bMPS_params_x_batched=(
                                    bMPS_x
                                ),
                                selected_rows=(
                                    selected_rows
                                ),
                            )
                        )
                        ratio = (
                            (proposed_amps.abs() ** 2)
                            / (current_amps.abs() ** 2)
                        )

                    if verbose:
                        t11 = time.time()
                        t_forward += t11 - t10
                        print(
                            f" Edge ({i}, {j}): "
                            f"{n_changed} / {B} "
                            f"proposed, forward: "
                            f"{t11-t10:.4f}s, "
                            f"total forward: "
                            f"{t_forward:.4f}s, "
                            f"progress: "
                            f"{n_updates}/{total_edges}"
                        )

                    probs = torch.rand(B, device=device)
                    accept_mask = (
                        new_flags & (probs < ratio)
                    )

                    if accept_mask.any():
                        fxs[accept_mask] = (
                            proposed_fxs[accept_mask]
                        )
                        if use_log_amp:
                            cur_signs[accept_mask] = (
                                prop_signs[accept_mask]
                            )
                            cur_log_abs[accept_mask] = (
                                prop_log_abs[accept_mask]
                            )
                        else:
                            current_amps[accept_mask] = (
                                proposed_amps[accept_mask]
                            )

            # (b) Col edges: row -> row+1
            if row in col_edges_by_row_pair:
                selected_rows = list(range(
                    max(0, row - model.radius),
                    min(
                        Lx,
                        row + 1 + model.radius + 1,
                    ),
                ))
                for edge in col_edges_by_row_pair[row]:
                    n_updates += 1
                    i, j = edge

                    if verbose:
                        t00 = time.time()
                    proposed_fxs, new_flags = (
                        propose_spin_exchange_vec(
                            i, j, fxs,
                        )
                    )
                    if verbose:
                        t11 = time.time()
                        t_propose += t11 - t00

                    if not new_flags.any():
                        continue

                    n_changed = new_flags.sum().item()

                    if verbose:
                        t10 = time.time()

                    if use_log_amp:
                        prop_signs, prop_log_abs = (
                            model.forward_reuse_log(
                                proposed_fxs,
                                bMPS_params_x_batched=(
                                    bMPS_x
                                ),
                                selected_rows=(
                                    selected_rows
                                ),
                            )
                        )
                        ratio = torch.exp(
                            2.0
                            * (prop_log_abs - cur_log_abs),
                        )
                    else:
                        proposed_amps = (
                            model.forward_reuse(
                                proposed_fxs,
                                bMPS_params_x_batched=(
                                    bMPS_x
                                ),
                                selected_rows=(
                                    selected_rows
                                ),
                            )
                        )
                        ratio = (
                            (proposed_amps.abs() ** 2)
                            / (current_amps.abs() ** 2)
                        )

                    if verbose:
                        t11 = time.time()
                        t_forward += t11 - t10
                        print(
                            f" Edge ({i}, {j}): "
                            f"{n_changed} / {B} "
                            f"proposed, forward: "
                            f"{t11-t10:.4f}s, "
                            f"total forward: "
                            f"{t_forward:.4f}s, "
                            f"progress: "
                            f"{n_updates}/{total_edges}"
                        )

                    probs = torch.rand(B, device=device)
                    accept_mask = (
                        new_flags & (probs < ratio)
                    )

                    if accept_mask.any():
                        fxs[accept_mask] = (
                            proposed_fxs[accept_mask]
                        )
                        if use_log_amp:
                            cur_signs[accept_mask] = (
                                prop_signs[accept_mask]
                            )
                            cur_log_abs[accept_mask] = (
                                prop_log_abs[accept_mask]
                            )
                        else:
                            current_amps[accept_mask] = (
                                proposed_amps[accept_mask]
                            )

            # (c) Update bMPS xmin past this row
            if row < Lx - 1:
                bMPS_x = (
                    model.update_bMPS_params_to_row_vmap(
                        fxs, row, bMPS_x,
                        from_which='xmin',
                    )
                )

        if verbose:
            t1 = time.time()
            print(
                f"Sample next: "
                f"{t1-t_total_start:.4f}s for "
                f"{n_updates} edges "
                f"(avg "
                f"{(t1-t_total_start)/n_updates:.4f}"
                f"s/edge, B={B})"
            )
            print(
                f"  Propose: {t_propose:.4f}s "
                f"(avg "
                f"{t_propose/n_updates:.4f}s/edge)"
            )
            print(
                f"  Forward: {t_forward:.4f}s "
                f"(avg "
                f"{t_forward/n_updates:.4f}s/edge)"
            )

        if use_log_amp:
            return fxs, (cur_signs, cur_log_abs)
        return fxs, current_amps


class MetropolisExchangeSpinfulSamplerXReuse_GPU(SamplerGPU):
    """Metropolis exchange sampler with x-only bMPS reuse
    for spinful fermions.

    Interleaved sweep: for each row, processes row edges
    (horizontal) then col edges (vertical to next row),
    all using x-direction boundary MPS only. Eliminates
    the expensive y-direction bMPS caching.

    Args:
        hopping_rate: Fraction of proposals that are
            hoppings (vs exchanges). Default 0.25.
    """

    def __init__(self, hopping_rate: float = 0.25):
        self.hopping_rate = hopping_rate

    @torch.inference_mode()
    def step(
        self,
        fxs: torch.Tensor,
        model,
        graph,
        compile: bool = False,
        verbose: bool = False,
        use_log_amp: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One MCMC sweep with x-only bMPS reuse.

        For each row r:
          (a) sweep row edges in row r
          (b) sweep col edges between rows r and r+1
          (c) update xmin bMPS past row r

        Args:
            fxs: (B, N_sites) int64 walker configs.
            model: fPEPS_Model_reuse_GPU with
                cache_bMPS_skeleton already called.
            graph: Lattice graph with .row_edges,
                .col_edges.
            compile: Unused (kept for interface compat).
            verbose: Print per-phase timing info.
            use_log_amp: If True, work in log-space and
                return (signs, log_abs) instead of amps.

        Returns:
            fxs: (B, N_sites) int64 updated configs.
            amps_out: (B,) amplitudes, or (signs, log_abs)
                tuple when use_log_amp=True.
        """
        B = fxs.shape[0]
        device = fxs.device
        Ly = model.Ly
        Lx = model.Lx

        # Pre-group col edges by row pair
        col_edges_by_row_pair = {}
        for col, edges in graph.col_edges.items():
            for (i, j) in edges:
                r = min(i // Ly, j // Ly)
                col_edges_by_row_pair.setdefault(
                    r, []
                ).append((i, j))

        # Count edges for progress tracking
        total_edges = sum(
            len(e) for e in graph.row_edges.values()
        ) + sum(
            len(e) for e in graph.col_edges.values()
        )

        n_updates = 0
        if verbose:
            t_total_start = time.time()
            t_propose = 0.0
            t_forward = 0.0

        # Cache x-direction bMPS only
        if verbose:
            t0 = time.time()
        bMPS_x, current_amps = (
            model.cache_bMPS_params_any_direction_vmap(
                fxs, direction='x', sides='xmax',
            )
        )
        if use_log_amp:
            cur_signs = torch.sign(current_amps)
            cur_log_abs = torch.log(
                current_amps.abs().clamp(min=1e-45),
            )
        if verbose:
            print(
                f" cache bMPS x: "
                f"{time.time() - t0:.4f}s"
            )

        for row in range(Lx):
            # (a) Row edges in this row
            if row in graph.row_edges:
                selected_rows = list(range(
                    max(0, row - model.radius),
                    min(Lx, row + model.radius + 1),
                ))
                for edge in graph.row_edges[row]:
                    n_updates += 1
                    i, j = edge

                    if verbose:
                        t00 = time.time()
                    proposed_fxs, new_flags = (
                        propose_exchange_or_hopping_vec(
                            i, j, fxs,
                            self.hopping_rate,
                        )
                    )
                    if verbose:
                        t11 = time.time()
                        t_propose += t11 - t00

                    if not new_flags.any():
                        continue

                    n_changed = new_flags.sum().item()

                    if verbose:
                        t10 = time.time()

                    if use_log_amp:
                        prop_signs, prop_log_abs = (
                            model.forward_reuse_log(
                                proposed_fxs,
                                bMPS_params_x_batched=(
                                    bMPS_x
                                ),
                                selected_rows=(
                                    selected_rows
                                ),
                            )
                        )
                        ratio = torch.exp(
                            2.0
                            * (prop_log_abs - cur_log_abs),
                        )
                    else:
                        proposed_amps = (
                            model.forward_reuse(
                                proposed_fxs,
                                bMPS_params_x_batched=(
                                    bMPS_x
                                ),
                                selected_rows=(
                                    selected_rows
                                ),
                            )
                        )
                        ratio = (
                            (proposed_amps.abs() ** 2)
                            / (current_amps.abs() ** 2)
                        )

                    if verbose:
                        t11 = time.time()
                        t_forward += t11 - t10
                        print(
                            f" Edge ({i}, {j}) (single row): "
                            f"{n_changed} / {B} "
                            f"proposed, forward: "
                            f"{t11-t10:.4f}s, "
                            f"total forward: "
                            f"{t_forward:.4f}s, "
                            f"progress: "
                            f"{n_updates}/{total_edges}"
                        )

                    probs = torch.rand(B, device=device)
                    accept_mask = (
                        new_flags & (probs < ratio)
                    )

                    if accept_mask.any():
                        fxs[accept_mask] = (
                            proposed_fxs[accept_mask]
                        )
                        if use_log_amp:
                            cur_signs[accept_mask] = (
                                prop_signs[accept_mask]
                            )
                            cur_log_abs[accept_mask] = (
                                prop_log_abs[accept_mask]
                            )
                        else:
                            current_amps[accept_mask] = (
                                proposed_amps[accept_mask]
                            )

            # (b) Col edges: row -> row+1
            if row in col_edges_by_row_pair:
                selected_rows = list(range(
                    max(0, row - model.radius),
                    min(
                        Lx,
                        row + 1 + model.radius + 1,
                    ),
                ))
                for edge in col_edges_by_row_pair[row]:
                    n_updates += 1
                    i, j = edge

                    if verbose:
                        t00 = time.time()
                    proposed_fxs, new_flags = (
                        propose_exchange_or_hopping_vec(
                            i, j, fxs,
                            self.hopping_rate,
                        )
                    )
                    if verbose:
                        t11 = time.time()
                        t_propose += t11 - t00

                    if not new_flags.any():
                        continue

                    n_changed = new_flags.sum().item()

                    if verbose:
                        t10 = time.time()

                    if use_log_amp:
                        prop_signs, prop_log_abs = (
                            model.forward_reuse_log(
                                proposed_fxs,
                                bMPS_params_x_batched=(
                                    bMPS_x
                                ),
                                selected_rows=(
                                    selected_rows
                                ),
                            )
                        )
                        ratio = torch.exp(
                            2.0
                            * (prop_log_abs - cur_log_abs),
                        )
                    else:
                        proposed_amps = (
                            model.forward_reuse(
                                proposed_fxs,
                                bMPS_params_x_batched=(
                                    bMPS_x
                                ),
                                selected_rows=(
                                    selected_rows
                                ),
                            )
                        )
                        ratio = (
                            (proposed_amps.abs() ** 2)
                            / (current_amps.abs() ** 2)
                        )

                    if verbose:
                        t11 = time.time()
                        t_forward += t11 - t10
                        print(
                            f" Edge ({i}, {j}) (two rows): "
                            f"{n_changed} / {B} "
                            f"proposed, forward: "
                            f"{t11-t10:.4f}s, "
                            f"total forward: "
                            f"{t_forward:.4f}s, "
                            f"progress: "
                            f"{n_updates}/{total_edges}"
                        )

                    probs = torch.rand(B, device=device)
                    accept_mask = (
                        new_flags & (probs < ratio)
                    )

                    if accept_mask.any():
                        fxs[accept_mask] = (
                            proposed_fxs[accept_mask]
                        )
                        if use_log_amp:
                            cur_signs[accept_mask] = (
                                prop_signs[accept_mask]
                            )
                            cur_log_abs[accept_mask] = (
                                prop_log_abs[accept_mask]
                            )
                        else:
                            current_amps[accept_mask] = (
                                proposed_amps[accept_mask]
                            )

            # (c) Update bMPS xmin past this row
            if row < Lx - 1:
                bMPS_x = (
                    model.update_bMPS_params_to_row_vmap(
                        fxs, row, bMPS_x,
                        from_which='xmin',
                    )
                )

        if verbose:
            t1 = time.time()
            print(
                f"Sample next: "
                f"{t1-t_total_start:.4f}s for "
                f"{n_updates} edges "
                f"(avg "
                f"{(t1-t_total_start)/n_updates:.4f}"
                f"s/edge, B={B})"
            )
            print(
                f"  Propose: {t_propose:.4f}s "
                f"(avg "
                f"{t_propose/n_updates:.4f}s/edge)"
            )
            print(
                f"  Forward: {t_forward:.4f}s "
                f"(avg "
                f"{t_forward/n_updates:.4f}s/edge)"
            )

        if use_log_amp:
            return fxs, (cur_signs, cur_log_abs)
        return fxs, current_amps
