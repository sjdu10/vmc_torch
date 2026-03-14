import quimb as qu
import torch
import time
import random
# from mpi4py import MPI
from torch.utils._pytree import tree_map, tree_flatten
import os                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                
def cpu_mem_gb():                                                                                                                                                                                                                                           
    """Resident memory of this process in GB."""
    with open(f'/proc/{os.getpid()}/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024**2  # kB -> GB
    return 0.0

#=== Utility functions for Metropolis-Hastings sampling ===#

def propose_exchange_or_hopping(i, j, current_config, hopping_rate=0.25, seed=None):
    if seed is not None:
        random.seed(seed)
    ind_n_map = {0: 0, 1: 1, 2: 1, 3: 2}
    if current_config[i] == current_config[j]:
        return current_config, 0
    proposed_config = current_config.clone()
    config_i = current_config[i].item()
    config_j = current_config[j].item()
    if random.random() < 1 - hopping_rate:
        # exchange
        proposed_config[i] = config_j
        proposed_config[j] = config_i
    else:
        # hopping
        n_i = ind_n_map[current_config[i].item()]
        n_j = ind_n_map[current_config[j].item()]
        delta_n = abs(n_i - n_j)
        if delta_n == 1:
            # consider only valid hopping: (0, u) -> (u, 0); (d, ud) -> (ud, d)
            proposed_config[i] = config_j
            proposed_config[j] = config_i
        elif delta_n == 0:
            # consider only valid hopping: (u, d) -> (0, ud) or (ud, 0)
            choices = [(0, 3), (3, 0)]
            choice = random.choice(choices)
            proposed_config[i] = choice[0]
            proposed_config[j] = choice[1]
        elif delta_n == 2:
            # consider only valid hopping: (0, ud) -> (u, d) or (d, u)
            choices = [(1, 2), (2, 1)]
            choice = random.choice(choices)
            proposed_config[i] = choice[0]
            proposed_config[j] = choice[1]
        else:
            raise ValueError("Invalid configuration")
    return proposed_config, 1


def propose_exchange_or_hopping_vec(i, j, current_configs, hopping_rate=0.25):
    """
    Fully vectorized propose function (GPU Friendly).
    Processes a batch of configurations at once without CPU-GPU synchronization.
    
    Args:
        i, j: (int) site indices for exchange/hopping
        current_configs: (Batch, N_sites) Tensor, dtype=long/int
        hopping_rate: (float) hopping probability
        
    Returns:
        proposed_configs: (Batch, N_sites) new configurations
        change_mask: (Batch,) bool Tensor indicating which samples have valid changes
    """
    B = current_configs.shape[0]
    device = current_configs.device
    
    # Particle number mapping: 0->0, 1->1, 2->1, 3->2
    n_map = torch.tensor([0, 1, 1, 2], device=device, dtype=torch.long)
    
    # Extract column i and j (Batch,)
    col_i = current_configs[:, i]
    col_j = current_configs[:, j]
    
    # 1. Basic check: if both positions have same state, cannot exchange or hop
    diff_mask = (col_i != col_j)
    
    # 2. Random decision between Exchange and Hopping
    rand_vals = torch.rand(B, device=device)
    
    # Only positions with different states need processing
    is_exchange = (rand_vals < (1 - hopping_rate)) & diff_mask
    is_hopping = (~is_exchange) & diff_mask
    
    # Initialize new columns, default equals old ones
    new_col_i = col_i.clone()
    new_col_j = col_j.clone()
    
    # --- A. Handle Exchange (and delta_n=1 Hopping) ---
    # Compute particle numbers
    n_i = n_map[col_i]
    n_j = n_map[col_j]
    delta_n = (n_i - n_j).abs()
    
    # Original logic: simple swap when delta_n == 1
    mask_swap = is_exchange | (is_hopping & (delta_n == 1))
    
    if mask_swap.any():
        new_col_i[mask_swap] = col_j[mask_swap]
        new_col_j[mask_swap] = col_i[mask_swap]
        
    # --- B. Handle Hopping (delta_n = 0 or 2) ---
    
    # Case: delta_n == 0 (e.g. u,d -> 0,ud)
    # Target: randomly become (0, 3) or (3, 0)
    mask_d0 = is_hopping & (delta_n == 0)
    if mask_d0.any():
        rand_bits = torch.randint(0, 2, (B,), device=device, dtype=torch.bool)
        
        val_0 = torch.tensor(0, device=device, dtype=col_i.dtype)
        val_3 = torch.tensor(3, device=device, dtype=col_i.dtype)
        
        # rand=0 -> i=0, j=3; rand=1 -> i=3, j=0
        target_i = torch.where(rand_bits, val_3, val_0)
        target_j = torch.where(rand_bits, val_0, val_3)
        
        new_col_i[mask_d0] = target_i[mask_d0]
        new_col_j[mask_d0] = target_j[mask_d0]

    # Case: delta_n == 2 (e.g. 0,ud -> u,d)
    # Target: randomly become (1, 2) or (2, 1)
    mask_d2 = is_hopping & (delta_n == 2)
    if mask_d2.any():
        rand_bits_2 = torch.randint(0, 2, (B,), device=device, dtype=torch.bool)
        
        val_1 = torch.tensor(1, device=device, dtype=col_i.dtype)
        val_2 = torch.tensor(2, device=device, dtype=col_i.dtype)
        
        # rand=0 -> i=1, j=2; rand=1 -> i=2, j=1
        target_i_2 = torch.where(rand_bits_2, val_2, val_1)
        target_j_2 = torch.where(rand_bits_2, val_1, val_2)
        
        new_col_i[mask_d2] = target_i_2[mask_d2]
        new_col_j[mask_d2] = target_j_2[mask_d2]
        
    # 3. Assemble results
    proposed_configs = current_configs.clone()
    proposed_configs[:, i] = new_col_i
    proposed_configs[:, j] = new_col_j
    
    return proposed_configs, diff_mask


# Batched Metropolis-Hastings updates
@torch.inference_mode()
def sample_next(fxs, fpeps_model, graph, hopping_rate=0.25, verbose=False, compile=False, **kwargs):
    """One full Metropolis-Hastings sweep over all lattice edges.

    Iterates over every edge in the lattice graph. At each edge (i, j),
    proposes an exchange or hopping for all B walkers simultaneously,
    evaluates amplitudes on the proposed configs, and accepts/rejects
    via the |psi'|^2 / |psi|^2 ratio.

    Args:
        fxs: Current configurations, (B, N_sites) int64. Modified in-place.
        fpeps_model: Batched wavefunction model, (B, N_sites) -> (B,).
        graph: Lattice graph with .row_edges and .col_edges dicts,
            each mapping direction to list of (i, j) edge tuples.
        hopping_rate: Probability of proposing a hopping (vs exchange)
            when sites i, j have different occupations.
        verbose: Print per-sweep timing breakdown.
        compile: If True, always evaluate all B configs (no partial
            batching), suitable for use with torch.compile.

    Returns:
        fxs: Updated configurations, (B, N_sites) int64.
        current_amps: Amplitudes at updated configs, (B,).
    """
    current_amps = fpeps_model(fxs)
    B = fxs.shape[0]
    device = fxs.device
    
    n_updates = 0 
    if verbose:
        t0 = time.time()
        t_propose = 0.0
        t_forward = 0.0
    # Merge row_edges and col_edges loops to reduce duplicate code
    all_edges = []
    for edges in graph.row_edges.values(): 
        all_edges.extend(edges)
    for edges in graph.col_edges.values(): 
        all_edges.extend(edges)

    for edge in all_edges:
        n_updates += 1
        i, j = edge
        
        # Call vectorized function directly without list comprehension
        if verbose:
            t00 = time.time()
        proposed_fxs, new_flags = propose_exchange_or_hopping_vec(i, j, fxs, hopping_rate)
        if verbose:
            t11 = time.time()
            t_propose += (t11 - t00)
        
        # Quick check: if all samples have no valid update, skip
        if not new_flags.any():
            continue
        
        # Compute Amplitudes — pad to fixed batch size B to
        # avoid torch.compile recompilation on varying shapes
        proposed_amps = current_amps.clone()
        n_changed = new_flags.sum().item()

        if verbose:
            t10 = time.time()
        if compile:
            new_proposed_amps = fpeps_model(proposed_fxs)
            proposed_amps = new_proposed_amps
        else:
            if n_changed == B:
                # All changed
                new_proposed_amps = fpeps_model(proposed_fxs)
                proposed_amps = new_proposed_amps
            else:
                changed_fxs = proposed_fxs[new_flags]
                changed_amps = fpeps_model(changed_fxs)
                proposed_amps[new_flags] = changed_amps
        
        if verbose:
            t11 = time.time()
            t_forward += (t11 - t10)
            print(f' Edge ({i}, {j}): {n_changed} / {B} samples proposed changes, time for forward pass: {t11-t10:.4f}s, total forward time: {t_forward:.4f}s')
        # Accept/Reject (fully vectorized, no .item() calls)
        ratio = (proposed_amps.abs()**2) / (current_amps.abs()**2 + 1e-18)
        
        # Vectorized random number generation
        probs = torch.rand(B, device=device)
        
        # Accept mask: only accept if new_flags is True and random < ratio
        accept_mask = new_flags & (probs < ratio)
        
        # Update using masking
        if accept_mask.any():
            fxs[accept_mask] = proposed_fxs[accept_mask]
            current_amps[accept_mask] = proposed_amps[accept_mask]
    if verbose:
        t1 = time.time()
        print(
            f"Sample next time: {t1 - t0:.4f}s for {n_updates} edge updates" \
            f' (avg {((t1 - t0) / n_updates):.4f}s per edge)' \
            f' (Batch size: {B})'
        )
        print(f"  Propose time: {t_propose:.4f}s (avg {t_propose / n_updates:.4f}s per edge)")
        print(f"  Forward time: {t_forward:.4f}s (avg {t_forward / n_updates:.4f}s per edge)")
    return fxs, current_amps

@torch.inference_mode()
def evaluate_energy(
    fxs, fpeps_model, H, current_amps,
    verbose=False, use_log_amp=False, **kwargs,
):
    """Compute local energies for a batch of configurations.

    For each config fxs[b], obtains connected configs and matrix
    elements via H.get_conn, evaluates amplitudes on all connected
    configs, and assembles E_loc[b] = sum_s' H_{s,s'} psi(s')/psi(s).

    Uses GPU-batched get_conn when available (H._hop_list), otherwise
    falls back to per-sample CPU computation. Connected amplitudes
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
    if hasattr(H, '_hop_list'):
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

    # Aggregate results
    local_energies = torch.zeros(
        B, device=device, dtype=terms.dtype,
    )
    local_energies.index_add_(0, batch_ids, terms)

    energy = torch.mean(local_energies)

    return energy, local_energies

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
    if hasattr(H, '_hop_list'):
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
    if hasattr(H, '_hop_list'):
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
    if torch.isnan(batched_grads_vec).any() or torch.isinf(batched_grads_vec).any():
        nan_mask = torch.isnan(batched_grads_vec)
        inf_mask = torch.isinf(batched_grads_vec)
        bad_samples = (nan_mask | inf_mask).any(dim=1)
        n_bad = bad_samples.sum().item()
        bad_params = (nan_mask | inf_mask).any(dim=0)
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
    fxs, fpeps_model, vectorize=True, batch_size=None,
    verbose=False, vmap_grad=False, offload_to_cpu=False,
    use_log_amp=False, **kwargs,
):
    """Vectorized gradient computation optimized for GPU.

    Args:
        offload_to_cpu: if True, each (B_grad, Np) gradient chunk
            is flattened and moved to CPU immediately after GPU
            computation.  The final returned tensors live on CPU.
            This keeps GPU peak memory at O(B_grad * Np) instead
            of O(B * Np).
        use_log_amp: if True, compute d(log|psi|)/d(params)
            directly. Returns (log_psi_grad, (signs, log_abs))
            instead of (grads, amps). log_psi_grad is already the
            log-derivative — no division by amps needed.
    """
    if vectorize:
        # 1. Prepare parameters PyTree structure
        # Compatible with ParameterList, ParameterDict, or direct Tensor List
        params_pytree = (
            list(fpeps_model.params)
            if isinstance(fpeps_model.params, torch.nn.ParameterList)
            else dict(fpeps_model.params)
            if isinstance(fpeps_model.params, torch.nn.ParameterDict)
            else fpeps_model.params
        )

        # ------------------------------------------------------------------
        # Path A: vmap(grad) - Usually efficient for per-sample scalar grad
        # ------------------------------------------------------------------
        if vmap_grad:
            B = fxs.shape[0]
            # Determine chunk size to avoid OOM
            B_grad = batch_size if batch_size is not None else B

            if use_log_amp:
                # Differentiate log|psi| directly — grad is O(1)
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
                # Original: differentiate amp directly
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

            t0 = time.time()

            if offload_to_cpu:
                # Pre-allocate CPU buffer and write chunks
                # directly to avoid torch.cat peak doubling.
                leaves_p, _ = tree_flatten(params_pytree)
                Np = sum(p.numel() for p in leaves_p)
                p_dtype = leaves_p[0].dtype
                del leaves_p

                batched_grads_vec = torch.empty(
                    B, Np, dtype=p_dtype, device='cpu',
                )
                if use_log_amp:
                    signs = torch.empty(
                        B, dtype=p_dtype, device='cpu',
                    )
                    log_abs = torch.empty(
                        B, dtype=p_dtype, device='cpu',
                    )
                else:
                    amps = torch.empty(
                        B, dtype=p_dtype, device='cpu',
                    )

                for b_start in range(0, B, B_grad):
                    if verbose:
                        print(f"Processing grad chunk: {b_start} to {min(b_start + B_grad, B)} / {B}")
                    b_end = min(b_start + B_grad, B)
                    fxs_chunk = fxs[b_start:b_end]

                    grads_chunk, aux_c = grad_vmap_fn(
                        fxs_chunk, params_pytree,
                    )

                    grads_chunk = tree_map(
                        lambda x: x.detach(), grads_chunk,
                    )

                    # Flatten to (B_grad, Np) on GPU, then offload
                    leaves_c, _ = tree_flatten(grads_chunk)
                    flat_c = torch.cat(
                        [l.flatten(start_dim=1) for l in leaves_c],
                        dim=1,
                    )
                    batched_grads_vec[b_start:b_end] = flat_c.cpu()

                    if use_log_amp:
                        sc, lac = aux_c
                        signs[b_start:b_end] = sc.detach().cpu()
                        log_abs[b_start:b_end] = lac.detach().cpu()
                        del sc, lac, aux_c
                    else:
                        amps[b_start:b_end] = aux_c.detach().cpu()
                        del aux_c

                    del grads_chunk, leaves_c, flat_c
            else:
                # Standard path: pre-allocate and fill in-place.
                leaves_p, _ = tree_flatten(params_pytree)
                Np = sum(p.numel() for p in leaves_p)
                dtype = leaves_p[0].dtype
                device = leaves_p[0].device
                del leaves_p

                batched_grads_vec = torch.empty(
                    B, Np, dtype=dtype, device=device,
                )
                if use_log_amp:
                    signs = torch.empty(
                        B, dtype=dtype, device=device,
                    )
                    log_abs = torch.empty(
                        B, dtype=dtype, device=device,
                    )
                else:
                    amps = torch.empty(
                        B, dtype=dtype, device=device,
                    )

                for b_start in range(0, B, B_grad):
                    if verbose:
                        print(f"Processing grad chunk: {b_start} to {min(b_start + B_grad, B)} / {B}")
                    b_end = min(b_start + B_grad, B)
                    fxs_chunk = fxs[b_start:b_end]

                    grads_chunk, aux_c = grad_vmap_fn(
                        fxs_chunk, params_pytree,
                    )

                    grads_chunk = tree_map(
                        lambda x: x.detach(), grads_chunk,
                    )

                    # Flatten and write directly into buffer
                    leaves_c, _ = tree_flatten(grads_chunk)
                    flat_c = torch.cat(
                        [leaf.flatten(start_dim=1)
                         for leaf in leaves_c],
                        dim=1,
                    )
                    batched_grads_vec[b_start:b_end] = flat_c
                    if use_log_amp:
                        sc, lac = aux_c
                        signs[b_start:b_end] = sc.detach()
                        log_abs[b_start:b_end] = lac.detach()
                        del sc, lac, aux_c
                    else:
                        amps[b_start:b_end] = aux_c.detach()
                        del aux_c
                    del grads_chunk, leaves_c, flat_c

            # Final cleanup
            batched_grads_vec = batched_grads_vec.detach()
            fpeps_model.zero_grad()

            t1 = time.time()
            if verbose:
                print(f"GPU Batched vmap(grad) time: {t1 - t0:.4f}s")

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

        # ------------------------------------------------------------------
        # Path B: jacrev - Standard Jacobian Reverse Mode
        # ------------------------------------------------------------------
        else:
            # Deprecated warning: jacrev path is less memory efficient and may OOM on large batches, prefer vmap(grad) with chunking
            Warning = ("jacrev path is less memory efficient and may OOM on large batches, "
                       "prefer vmap(grad) with chunking. This path will be removed in future versions.")
            def g(x, p):
                results = fpeps_model.vamp(x, p)
                return results, results

            # If no batch_size limit, try to compute all at once (RISKY on GPU)
            if batch_size is None:
                t0 = time.time()
                jac_pytree, amps = torch.func.jacrev(g, argnums=1, has_aux=True)(fxs, params_pytree)
                t1 = time.time()
                if verbose:
                    try: 
                        print(f"GPU Full Batch Jacobian time: {t1 - t0:.4f}s")
                    except NameError: pass
            
            # Chunked execution for jacrev
            else:
                B = fxs.shape[0]
                B_grad = batch_size
                jac_pytree_list = []
                amps_list = []
                
                t0 = time.time()
                for b_start in range(0, B, B_grad):
                    b_end = min(b_start + B_grad, B)
                    
                    # jacrev computation
                    jac_pytree_b, amps_b = torch.func.jacrev(g, argnums=1, has_aux=True)(
                        fxs[b_start:b_end], params_pytree
                    )
                    
                    # Detach to free compute graph
                    amps_b = amps_b.detach()
                    jac_pytree_b = tree_map(lambda x: x.detach(), jac_pytree_b)

                    jac_pytree_list.append(jac_pytree_b)
                    amps_list.append(amps_b)

                # Concatenate results
                jac_pytree = tree_map(lambda *leaves: torch.cat(leaves, dim=0), *jac_pytree_list)
                amps = torch.cat(amps_list, dim=0)
                t1 = time.time()
                if verbose:
                    try:
                        if RANK == 0: print(f"GPU Chunked Jacobian time: {t1 - t0:.4f}s")
                    except NameError: pass

            # Process jac_pytree to flat vector
            leaves, _ = tree_flatten(jac_pytree)
            leaves_flattend = [leaf.flatten(start_dim=1) for leaf in leaves]
            batched_grads_vec = torch.cat(leaves_flattend, dim=1)
            
            if amps.dim() == 1: amps.unsqueeze_(1)
            
            # Cleanup
            batched_grads_vec = batched_grads_vec.detach()
            amps = amps.detach()
            if 'jac_pytree' in locals(): del jac_pytree
            fpeps_model.zero_grad()

            _check_grads_amps(batched_grads_vec, amps, fpeps_model)
            return batched_grads_vec, amps

    else:
        # ------------------------------------------------------------------
        # Non-Vectorized Sequential Fallback
        # ------------------------------------------------------------------
        amps = []
        batched_grads_vec = []
        t0 = time.time()
        
        # Helper to flatten gradients from .grad attributes
        def flatten_grads(model):
            grads_list = []
            for p in model.parameters():
                if p.grad is not None:
                    grads_list.append(p.grad.flatten())
                else:
                    grads_list.append(torch.zeros_like(p).flatten())
            return torch.cat(grads_list)

        for fx in fxs:
            amp = fpeps_model(fx.unsqueeze(0))
            amps.append(amp)
            
            # Standard backward
            amp.backward()
            
            # Collect gradients
            # Assuming params is flat list or consistent iteration order
            current_grads = flatten_grads(fpeps_model)
            batched_grads_vec.append(current_grads)
            
            fpeps_model.zero_grad()
            
        t1 = time.time()
        if verbose:
            try:
                if RANK == 0: print(f"GPU Sequential time: {t1 - t0:.4f}s")
            except NameError: pass
            
        amps = torch.stack(amps, dim=0)
        batched_grads_vec = torch.stack(batched_grads_vec, dim=0)
        
        return batched_grads_vec, amps


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
        flat_vec_chunks = []
        if use_log_amp:
            signs_chunks, log_abs_chunks = [], []
        else:
            amps_chunks = []
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
            flat_vec_chunks.append(grads_chunk.cpu())
            if use_log_amp:
                signs_chunks.append(s_chunk.cpu())
                log_abs_chunks.append(la_chunk.cpu())
            else:
                amps_chunks.append(amps_chunk.cpu())
        else:
            batched_grads_vec[b_start:b_end] = grads_chunk
            if use_log_amp:
                signs[b_start:b_end] = s_chunk
                log_abs[b_start:b_end] = la_chunk
            else:
                amps[b_start:b_end] = amps_chunk

        del grads_chunk, amps_chunk

    if offload_to_cpu:
        batched_grads_vec = torch.cat(
            flat_vec_chunks, dim=0,
        )
        if use_log_amp:
            signs = torch.cat(signs_chunks, dim=0)
            log_abs = torch.cat(log_abs_chunks, dim=0)
        else:
            amps = torch.cat(amps_chunks, dim=0)

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


def random_initial_config(N_f, N_sites, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    half_filled_config = torch.tensor(
        [1, 2] * (N_sites // 2)
    )
    # Set first (Lx*Ly - N_f) sites to be empty (0)
    empty_sites = list(range(N_sites - N_f))
    doped_config = half_filled_config.clone()
    doped_config[empty_sites] = 0
    # Randomly permute the doped_config
    perm = torch.randperm(N_sites)
    doped_config = doped_config[perm]
    num_1 = torch.sum(doped_config == 1).item()
    num_2 = torch.sum(doped_config == 2).item()
    assert num_1 == N_f // 2 and num_2 == N_f // 2, f"Number of spin up and spin down fermions should be {N_f // 2}, but got {num_1} and {num_2}"

    return doped_config


# =============== Debug ================
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
