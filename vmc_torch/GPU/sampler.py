import time
from typing import Tuple

import torch
import random

#=== Utility functions for Metropolis-Hastings sampling on fermionic systems ===#
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


class SamplerGPU:
    """Base sampler interface — MCMC only.

    The sampler only handles Markov chain Monte Carlo
    (proposing moves, accepting/rejecting). It does NOT
    evaluate energies or gradients — that is the VMC
    driver's responsibility.

    Subclasses must implement step(). burn_in() has a
    default implementation that calls step() repeatedly.
    """

    def step(
        self,
        fxs: torch.Tensor,
        model,
        graph,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One MCMC sweep over all walkers.

        Args:
            fxs: (B, N_sites) int64 walker configs.
            model: nn.Module with .forward(x) -> (B,).
            graph: Lattice graph with .row_edges,
                .col_edges.
            **kwargs: Sampler-specific options (compile,
                verbose, etc.)

        Returns:
            fxs_new: (B, N_sites) int64 updated configs.
            amps: (B,) float64 amplitudes at fxs_new.
        """
        raise NotImplementedError

    def burn_in(
        self,
        fxs: torch.Tensor,
        model,
        graph,
        n_steps: int,
        **kwargs,
    ) -> torch.Tensor:
        """Run multiple MCMC sweeps without collecting.

        Args:
            fxs: (B, N_sites) int64 walker configs.
            model: nn.Module with .forward(x) -> (B,).
            graph: Lattice graph.
            n_steps: Number of burn-in sweeps.
            **kwargs: Forwarded to step().

        Returns:
            fxs: (B, N_sites) int64 after burn-in.
        """
        for _ in range(n_steps):
            fxs, _ = self.step(fxs, model, graph, **kwargs)
        return fxs


class MetropolisExchangeSpinfulSamplerGPU(SamplerGPU):
    """Metropolis exchange sampler for spinful fermions.

    Proposes particle exchanges and hoppings on a lattice
    graph. To create a sampler for different physics
    (bosons, spins), subclass SamplerGPU and implement
    step().

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
        """One Metropolis sweep over all lattice edges.

        Iterates over graph.row_edges + graph.col_edges,
        proposes exchange or hopping for all B walkers at
        each edge, evaluates amplitudes, and
        accepts/rejects via |psi'|^2 / |psi|^2.

        Args:
            fxs: (B, N_sites) int64 walker configs.
            model: nn.Module with .forward(x) -> (B,).
            graph: Lattice graph.
            compile: If True, always evaluate all B configs
                (no partial batching) for torch.compile.
            verbose: Print per-edge timing info.
            use_log_amp: If True, work in log-space and
                return (signs, log_abs) instead of amps.

        Returns:
            fxs: (B, N_sites) int64 updated configs.
            amps_out: (B,) amplitudes, or (signs, log_abs)
                tuple when use_log_amp=True.
        """
        if use_log_amp:
            cur_signs, cur_log_abs = model.forward_log(fxs)
        else:
            current_amps = model(fxs)
        B = fxs.shape[0]
        device = fxs.device

        n_updates = 0
        if verbose:
            t0 = time.time()
            t_propose = 0.0
            t_forward = 0.0

        # Collect all edges
        all_edges = []
        for edges in graph.row_edges.values():
            all_edges.extend(edges)
        for edges in graph.col_edges.values():
            all_edges.extend(edges)

        for edge in all_edges:
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

            # Skip if no valid proposals
            if not new_flags.any():
                continue

            n_changed = new_flags.sum().item()

            if verbose:
                t10 = time.time()

            if use_log_amp:
                # Evaluate proposed log-amplitudes
                prop_signs = cur_signs.clone()
                prop_log_abs = cur_log_abs.clone()
                if compile:
                    ps, pla = model.forward_log(proposed_fxs)
                    prop_signs = ps
                    prop_log_abs = pla
                else:
                    if n_changed == B:
                        ps, pla = model.forward_log(
                            proposed_fxs,
                        )
                        prop_signs = ps
                        prop_log_abs = pla
                    else:
                        changed_fxs = proposed_fxs[new_flags]
                        ps, pla = model.forward_log(
                            changed_fxs,
                        )
                        prop_signs[new_flags] = ps
                        prop_log_abs[new_flags] = pla
            else:
                # Evaluate proposed amplitudes
                proposed_amps = current_amps.clone()
                if compile:
                    new_proposed_amps = model(proposed_fxs)
                    proposed_amps = new_proposed_amps
                else:
                    if n_changed == B:
                        new_proposed_amps = model(
                            proposed_fxs,
                        )
                        proposed_amps = new_proposed_amps
                    else:
                        changed_fxs = proposed_fxs[new_flags]
                        changed_amps = model(changed_fxs)
                        proposed_amps[new_flags] = changed_amps

            if verbose:
                t11 = time.time()
                t_forward += t11 - t10
                print(
                    f" Edge ({i}, {j}): {n_changed} / {B} "
                    f"proposed, forward: {t11-t10:.4f}s, "
                    f"total forward: {t_forward:.4f}s, "
                    f"progress: {n_updates}/{len(all_edges)}"
                )

            # Metropolis accept/reject
            if use_log_amp:
                ratio = torch.exp(
                    2.0 * (prop_log_abs - cur_log_abs),
                )
            else:
                ratio = (
                    (proposed_amps.abs() ** 2)
                    / (current_amps.abs() ** 2)
                )
            probs = torch.rand(B, device=device)
            accept_mask = new_flags & (probs < ratio)

            if accept_mask.any():
                fxs[accept_mask] = proposed_fxs[accept_mask]
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

        if verbose:
            t1 = time.time()
            print(
                f"Sample next: {t1-t0:.4f}s for "
                f"{n_updates} edges "
                f"(avg {(t1-t0)/n_updates:.4f}s/edge, "
                f"B={B})"
            )
            print(
                f"  Propose: {t_propose:.4f}s "
                f"(avg {t_propose/n_updates:.4f}s/edge)"
            )
            print(
                f"  Forward: {t_forward:.4f}s "
                f"(avg {t_forward/n_updates:.4f}s/edge)"
            )

        if use_log_amp:
            return fxs, (cur_signs, cur_log_abs)
        return fxs, current_amps


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

#=== Utility functions for Metropolis-Hastings sampling on spin systems ===#
def propose_spin_exchange_vec(i, j, current_configs):
    """Propose spin exchange on edge (i,j) for all walkers.

    For spin-1/2 configs encoded as {0, 1}, swaps
    the values at sites i and j when they differ.

    Args:
        i, j: int, site indices.
        current_configs: (B, N_sites) int64.

    Returns:
        proposed_configs: (B, N_sites) int64.
        new_flags: (B,) bool — True where a swap occurred.
    """
    proposed = current_configs.clone()
    si = current_configs[:, i]
    sj = current_configs[:, j]
    diff = si != sj
    proposed[diff, i] = sj[diff]
    proposed[diff, j] = si[diff]
    return proposed, diff


class MetropolisExchangeSpinSamplerGPU(SamplerGPU):
    """Metropolis exchange sampler for spin-1/2 systems.

    Proposes nearest-neighbor spin exchanges on a lattice
    graph. Conserves total Sz. For Heisenberg models.
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
        """One Metropolis sweep over all lattice edges.

        Args:
            fxs: (B, N_sites) int64 walker configs.
            model: nn.Module with .forward(x) -> (B,).
            graph: Lattice graph with .row_edges,
                .col_edges.
            compile: If True, evaluate all B configs per
                edge (no partial batching).
            verbose: Print per-edge timing info.
            use_log_amp: If True, work in log-space and
                return (signs, log_abs) instead of amps.

        Returns:
            fxs: (B, N_sites) int64 updated configs.
            amps_out: (B,) amplitudes, or (signs, log_abs)
                tuple when use_log_amp=True.
        """
        if use_log_amp:
            cur_signs, cur_log_abs = model.forward_log(fxs)
        else:
            current_amps = model(fxs)
        B = fxs.shape[0]
        device = fxs.device

        n_updates = 0
        if verbose:
            t0 = time.time()
            t_propose = 0.0
            t_forward = 0.0

        all_edges = []
        for edges in graph.row_edges.values():
            all_edges.extend(edges)
        for edges in graph.col_edges.values():
            all_edges.extend(edges)

        for edge in all_edges:
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

            if verbose:
                t10 = time.time()

            if use_log_amp:
                prop_signs = cur_signs.clone()
                prop_log_abs = cur_log_abs.clone()
                if compile:
                    ps, pla = model.forward_log(proposed_fxs)
                    prop_signs = ps
                    prop_log_abs = pla
                else:
                    if n_changed == B:
                        ps, pla = model.forward_log(
                            proposed_fxs,
                        )
                        prop_signs = ps
                        prop_log_abs = pla
                    else:
                        changed_fxs = proposed_fxs[new_flags]
                        ps, pla = model.forward_log(
                            changed_fxs,
                        )
                        prop_signs[new_flags] = ps
                        prop_log_abs[new_flags] = pla
            else:
                proposed_amps = current_amps.clone()
                if compile:
                    new_proposed_amps = model(proposed_fxs)
                    proposed_amps = new_proposed_amps
                else:
                    if n_changed == B:
                        new_proposed_amps = model(
                            proposed_fxs,
                        )
                        proposed_amps = new_proposed_amps
                    else:
                        changed_fxs = proposed_fxs[new_flags]
                        changed_amps = model(changed_fxs)
                        proposed_amps[new_flags] = changed_amps

            if verbose:
                t11 = time.time()
                t_forward += t11 - t10
                print(
                    f" Edge ({i}, {j}): {n_changed} / {B}"
                    f" proposed, forward: {t11-t10:.4f}s,"
                    f" total forward: {t_forward:.4f}s,"
                    f" progress: {n_updates}/{len(all_edges)}"
                )

            if use_log_amp:
                ratio = torch.exp(
                    2.0 * (prop_log_abs - cur_log_abs),
                )
            else:
                ratio = (
                    (proposed_amps.abs() ** 2)
                    / (current_amps.abs() ** 2)
                )
            probs = torch.rand(B, device=device)
            accept_mask = new_flags & (probs < ratio)

            if accept_mask.any():
                fxs[accept_mask] = proposed_fxs[accept_mask]
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

        if verbose:
            t1 = time.time()
            print(
                f"Sample next: {t1-t0:.4f}s for "
                f"{n_updates} edges "
                f"(avg {(t1-t0)/n_updates:.4f}s/edge, "
                f"B={B})"
            )
            print(
                f"  Propose: {t_propose:.4f}s "
                f"(avg {t_propose/n_updates:.4f}s/edge)"
            )
            print(
                f"  Forward: {t_forward:.4f}s "
                f"(avg {t_forward/n_updates:.4f}s/edge)"
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


__all__ = [
    "SamplerGPU",
    "MetropolisExchangeSpinfulSamplerGPU",
    "MetropolisExchangeSpinfulSamplerReuse_GPU",
    "MetropolisExchangeSpinfulSamplerXReuse_GPU",
    "MetropolisExchangeSpinSamplerGPU",
    "MetropolisExchangeSpinSamplerReuse_GPU",
    "MetropolisExchangeSpinSamplerXReuse_GPU",
    "propose_spin_exchange_vec",
]
