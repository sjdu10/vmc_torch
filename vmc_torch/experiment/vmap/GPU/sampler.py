import time
from typing import Tuple

import torch

from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    propose_exchange_or_hopping_vec,
)


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

        Returns:
            fxs: (B, N_sites) int64 updated configs.
            current_amps: (B,) amplitudes at updated configs.
        """
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

            # Evaluate amplitudes — pad to B for compile
            proposed_amps = current_amps.clone()
            n_changed = new_flags.sum().item()

            if verbose:
                t10 = time.time()
            if compile:
                new_proposed_amps = model(proposed_fxs)
                proposed_amps = new_proposed_amps
            else:
                if n_changed == B:
                    new_proposed_amps = model(proposed_fxs)
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
                    f"total forward: {t_forward:.4f}s"
                )

            # Metropolis accept/reject
            ratio = (
                (proposed_amps.abs() ** 2)
                / (current_amps.abs() ** 2 + 1e-18)
            )
            probs = torch.rand(B, device=device)
            accept_mask = new_flags & (probs < ratio)

            if accept_mask.any():
                fxs[accept_mask] = proposed_fxs[accept_mask]
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

        Returns:
            fxs: (B, N_sites) int64 updated configs.
            current_amps: (B,) amplitudes at updated configs.
        """
        B = fxs.shape[0]
        device = fxs.device

        if verbose:
            t0 = time.time()

        # ---- Phase 1: x-direction (row edges) ----
        bMPS_x, current_amps = (
            model.cache_bMPS_params_any_direction_vmap(
                fxs, direction='x',
            )
        )

        for row, edges in graph.row_edges.items():
            for edge in edges:
                i, j = edge
                proposed_fxs, new_flags = (
                    propose_exchange_or_hopping_vec(
                        i, j, fxs, self.hopping_rate,
                    )
                )
                if not new_flags.any():
                    continue

                # Determine which rows to contract
                selected_rows = list(range(
                    max(0, row - model.radius),
                    min(model.Lx, row + model.radius + 1),
                ))

                # Evaluate all B configs (must match bMPS batch)
                proposed_amps = model.forward_reuse(
                    proposed_fxs,
                    bMPS_params_x_batched=bMPS_x,
                    selected_rows=selected_rows,
                )

                # Vectorized Metropolis accept/reject
                ratio = (
                    (proposed_amps.abs() ** 2)
                    / (current_amps.abs() ** 2 + 1e-18)
                )
                probs = torch.rand(B, device=device)
                accept_mask = new_flags & (probs < ratio)

                if accept_mask.any():
                    fxs[accept_mask] = proposed_fxs[accept_mask]
                    current_amps[accept_mask] = (
                        proposed_amps[accept_mask]
                    )

            # Update bMPS to next row
            if row < model.Lx - 1:
                bMPS_x = model.update_bMPS_params_to_row_vmap(
                    fxs, row, bMPS_x, from_which='xmin',
                )

        if verbose:
            t1 = time.time()
            print(
                f"Phase 1 (x-dir row edges): {t1 - t0:.4f}s"
            )

        # ---- Phase 2: y-direction (col edges) ----
        if verbose:
            t0 = time.time()

        bMPS_y, current_amps = (
            model.cache_bMPS_params_any_direction_vmap(
                fxs, direction='y',
            )
        )

        for col, edges in graph.col_edges.items():
            for edge in edges:
                i, j = edge
                proposed_fxs, new_flags = (
                    propose_exchange_or_hopping_vec(
                        i, j, fxs, self.hopping_rate,
                    )
                )
                if not new_flags.any():
                    continue

                selected_cols = list(range(
                    max(0, col - model.radius),
                    min(model.Ly, col + model.radius + 1),
                ))

                proposed_amps = model.forward_reuse(
                    proposed_fxs,
                    bMPS_params_y_batched=bMPS_y,
                    selected_cols=selected_cols,
                )

                ratio = (
                    (proposed_amps.abs() ** 2)
                    / (current_amps.abs() ** 2 + 1e-18)
                )
                probs = torch.rand(B, device=device)
                accept_mask = new_flags & (probs < ratio)

                if accept_mask.any():
                    fxs[accept_mask] = proposed_fxs[accept_mask]
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
                f"Phase 2 (y-dir col edges): {t1 - t0:.4f}s"
            )

        return fxs, current_amps


__all__ = [
    "SamplerGPU",
    "MetropolisExchangeSpinfulSamplerGPU",
    "MetropolisExchangeSpinfulSamplerReuse_GPU",
]
