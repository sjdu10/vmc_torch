"""GPU MCMC samplers for tensor-network variational Monte Carlo.

Every sampler exposes ``step(fxs, model, graph, ...)`` -> ``(fxs, amps)``
(one sweep over all ``B`` walkers) and ``burn_in(...)``; energies and
gradients are the driver's job (see ``vmc_torch.GPU.VMC``).  Walker
configurations are ``(B, N_sites)`` int64 tensors and amplitudes are
``(B,)`` tensors (or a ``(sign, log_abs)`` pair of ``(B,)`` tensors when
``use_log_amp=True``).  Spinful fermion sites are encoded as
0=empty, 1=down, 2=up, 3=doubly occupied; spin-1/2 sites as {0, 1}.

Metropolis acceptance always uses ``|psi(x')|^2 / |psi(x)|^2`` (in log
form ``exp(2 * (log|psi'| - log|psi|))``).  The exchange / hopping
proposal kernels are symmetric, so detailed balance holds without a
Hastings correction.  The direct-proposal samplers are independence
samplers and include the ``p_c(S) / p_c(S')`` Hastings factor
explicitly.

The bMPS-reuse samplers live in ``tensor_network/reuse.py`` and are
forwarded lazily through the module-level ``__getattr__`` at the bottom
of this file.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

import torch
import random

if TYPE_CHECKING:
    # Annotation-only imports: keep the heavy models package and the
    # quimb-backed Hamiltonian module out of the runtime import graph.
    from vmc_torch.GPU.models._base import WavefunctionModel_GPU
    from vmc_torch.hamiltonian_torch import Graph

#=== Utility functions for Metropolis-Hastings sampling on fermionic systems ===#
def propose_exchange_or_hopping(
    i: int,
    j: int,
    current_config: torch.Tensor,
    hopping_rate: float = 0.25,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, int]:
    """Propose an exchange or hopping move on bond (i, j) for ONE config.

    Scalar (non-vectorized) reference version of
    :func:`propose_exchange_or_hopping_vec`; it is not used by any
    sampler in this module.  Sites use the spinful encoding 0=empty,
    1=down, 2=up, 3=doubly occupied.  If the two sites hold the same
    state the config is returned unchanged.  Otherwise, with probability
    ``1 - hopping_rate`` the two site states are swapped (exchange);
    with probability ``hopping_rate`` a hopping move is made, chosen by
    the local particle-number difference ``delta_n = |n_i - n_j|``:

    * ``delta_n == 1`` (e.g. (0, u) or (d, ud)): swap the two sites,
      which is a single-particle hop;
    * ``delta_n == 0`` (i.e. (d, u) or (u, d)): become (0, ud) or
      (ud, 0) with equal probability;
    * ``delta_n == 2`` (i.e. (0, ud) or (ud, 0)): become (d, u) or
      (u, d) with equal probability.

    Every branch conserves N_up and N_down separately, and the kernel is
    symmetric, so the Metropolis ratio is just |psi'|^2 / |psi|^2.

    Args:
        i: first site index.
        j: second site index.
        current_config: (N_sites,) int64 configuration of one walker.
        hopping_rate: probability of proposing a hopping instead of an
            exchange when the two site states differ.
        seed: if given, reseeds Python's global ``random`` module
            (side effect) before drawing.

    Returns:
        proposed_config: (N_sites,) int64 proposal; the input tensor
            itself (not a copy) when no move is possible.
        changed: 1 if a move was proposed, 0 if sites i and j were equal.
    """
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


def propose_exchange_or_hopping_vec(
    i: int,
    j: int,
    current_configs: torch.Tensor,
    hopping_rate: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propose exchange/hopping moves on bond (i, j) for all walkers.

    Fully vectorized (GPU friendly) version of
    :func:`propose_exchange_or_hopping`: every walker draws an
    independent coin (exchange vs hopping) and, for the two-outcome
    hopping cases, an independent random bit, all from the global torch
    RNG.  The only CPU-GPU syncs are the ``.any()`` guards.  Walkers
    whose sites i and j hold the same state are left unchanged.  All
    moves conserve N_up and N_down separately and the kernel is
    symmetric (see the scalar version for the case table).

    Args:
        i: first site index.
        j: second site index.
        current_configs: (B, N_sites) int64 spinful configurations.
        hopping_rate: probability of proposing a hopping instead of an
            exchange for a walker whose two site states differ.

    Returns:
        proposed_configs: (B, N_sites) int64 proposals (a new tensor).
        change_mask: (B,) bool, True where sites i and j differed, i.e.
            where the proposal differs from the input.
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


#=== Ao-style (quantax) proposals: mode-occupation representation ===#
# Our site config encodes 0=empty, 1=up, 2=down, 3=doublon. Ao works
# in a mode-occupation vector of length 2*N (first N = spin-up modes,
# next N = spin-down modes), each mode occupied/empty. These helpers
# convert to/from that picture so ParticleHop matches quantax exactly.

def _site_to_mode_occ(current_configs: torch.Tensor) -> torch.Tensor:
    """Convert site configs (B, N) to a mode-occupation vector (B, 2N).

    The first N modes hold the species encoded by site value 1 and the
    next N modes the species encoded by site value 2 (a doublon, 3,
    occupies both).  The quantax port labels these blocks "up" and
    "down"; in the codebase convention 1=down, 2=up, but the two
    species are treated symmetrically so only the label differs.

    Args:
        current_configs: (B, N) int64 site configs in {0, 1, 2, 3}.

    Returns:
        occ: (B, 2N) int64 occupations in {0, 1}, modes ordered as
            ``[m1_0 .. m1_{N-1}, m2_0 .. m2_{N-1}]`` where ``m1`` /
            ``m2`` are the species encoded by 1 / 2.
    """
    up_occ = (current_configs == 1) | (current_configs == 3)
    dn_occ = (current_configs == 2) | (current_configs == 3)
    return torch.cat([up_occ, dn_occ], dim=1).to(torch.int64)


def _mode_occ_to_site(occ: torch.Tensor, N: int) -> torch.Tensor:
    """Inverse of :func:`_site_to_mode_occ`.

    Args:
        occ: (B, 2N) int64 mode occupations in {0, 1}.
        N: number of lattice sites.

    Returns:
        configs: (B, N) int64 site configs
            ``occ[:, :N] + 2 * occ[:, N:]`` in {0, 1, 2, 3}.
    """
    up_new = occ[:, :N]
    dn_new = occ[:, N:]
    return up_new + 2 * dn_new


def propose_particle_hop(
    current_configs: torch.Tensor,
    mode_neighbors: torch.Tensor,
    pick_occupied: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ParticleHop proposal — faithful port of quantax ParticleHop.

    Picks a random occupied (or, if the lattice is more than
    half-filled, a random empty) mode per walker, then a random
    same-spin-sector neighbor mode, and swaps their occupations.
    This hops a single spin-species particle by one lattice bond, so
    N_up and N_down are conserved separately.  When the chosen
    neighbor is also occupied or is the -1 fill (no valid neighbor),
    the swap is a no-op -> a null move S'=S, exactly as in quantax (no
    skip, no resample).

    The kernel is symmetric: the picked mode is uniform over the
    minority set and the slot is uniform over ``max_deg`` entries of a
    symmetric adjacency table, so the reverse hop has the same
    probability and plain Metropolis acceptance applies.  Randomness
    comes from the global torch RNG (``multinomial`` / ``randint``).

    Args:
        current_configs: (B, N) int64 site configs in {0,1,2,3}.
        mode_neighbors: (2N, max_deg) int64 same-sector neighbor mode
            table, -1 padded (built by
            ``MetropolisAoMixSpinfulSamplerGPU._build``).
        pick_occupied: If True pick among occupied modes (Ntotal <=
            Nmodes/2), else pick among empty modes (the minority set,
            matching quantax's hopping_particle choice).

    Returns:
        proposed_configs: (B, N) int64 proposals (a new tensor, same
            dtype as the input).
        change_mask: (B,) bool — walkers whose config actually changed.
    """
    B, N = current_configs.shape
    device = current_configs.device
    arange = torch.arange(B, device=device)

    occ = _site_to_mode_occ(current_configs)   # (B, 2N)

    # Pick a random mode from the minority set (uniform over its
    # members), matching jr.choice(Nmodes, p=(spins == hopping_particle)).
    p = occ if pick_occupied else (1 - occ)
    particle_idx = torch.multinomial(p.float(), 1).squeeze(1)   # (B,)

    # Same-sector neighbor modes, then a uniform slot (incl. -1 fill).
    cand = mode_neighbors[particle_idx]        # (B, max_deg)
    max_deg = cand.shape[1]
    slot = torch.randint(0, max_deg, (B,), device=device)
    neighbor_idx = cand[arange, slot]
    neighbor_idx = torch.where(
        neighbor_idx == -1, particle_idx, neighbor_idx,
    )

    # Swap occupations of the two modes.
    val_p = occ[arange, particle_idx].clone()
    val_n = occ[arange, neighbor_idx].clone()
    occ[arange, particle_idx] = val_n
    occ[arange, neighbor_idx] = val_p

    proposed = _mode_occ_to_site(occ, N).to(current_configs.dtype)
    change_mask = (proposed != current_configs).any(dim=1)
    return proposed, change_mask


def propose_site_exchange(
    current_configs: torch.Tensor,
    edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SiteExchange proposal — faithful port of quantax SiteExchange.

    Picks a random lattice edge per walker (uniform over ``E`` edges,
    global torch RNG) and swaps the full site states of its two
    endpoints (both spin sectors at once), which conserves N_up and
    N_down.  Equal endpoints give a null move S'=S, as in quantax.  The
    kernel is symmetric (the same edge undoes the swap).

    Args:
        current_configs: (B, N) int64 site configs in {0,1,2,3}.
        edges: (E, 2) int64 lattice edge table.

    Returns:
        proposed_configs: (B, N) int64 proposals (a new tensor).
        change_mask: (B,) bool — walkers whose config actually changed.
    """
    B = current_configs.shape[0]
    device = current_configs.device
    arange = torch.arange(B, device=device)

    E = edges.shape[0]
    eidx = torch.randint(0, E, (B,), device=device)
    chosen = edges[eidx]                        # (B, 2)
    i = chosen[:, 0]
    j = chosen[:, 1]

    vi = current_configs[arange, i]
    vj = current_configs[arange, j]
    proposed = current_configs.clone()
    proposed[arange, i] = vj
    proposed[arange, j] = vi

    change_mask = (vi != vj)
    return proposed, change_mask


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
        model: WavefunctionModel_GPU,
        graph: Graph,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """One MCMC sweep over all walkers.

        Subclasses update ``fxs`` in place for accepted walkers and
        return it together with the amplitudes at the new configs, so
        the driver never re-evaluates ``psi`` at the sampled states.

        Args:
            fxs: (B, N_sites) int64 walker configs.
            model: wavefunction with ``.forward(x) -> (B,)`` and, for
                log-space sampling, ``.forward_log(x) -> (sign, log_abs)``.
            graph: Lattice graph with ``.row_edges`` / ``.col_edges``
                dicts of ``(i, j)`` site-index pairs.
            **kwargs: Sampler-specific options (``compile``,
                ``verbose``, ``use_log_amp``, ...).

        Returns:
            fxs_new: (B, N_sites) int64 updated configs.
            amps: (B,) amplitudes at fxs_new, or a ``(sign, log_abs)``
                tuple of (B,) tensors when ``use_log_amp=True``.
        """
        raise NotImplementedError

    def burn_in(
        self,
        fxs: torch.Tensor,
        model: WavefunctionModel_GPU,
        graph: Graph,
        n_steps: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run multiple MCMC sweeps without collecting.

        Args:
            fxs: (B, N_sites) int64 walker configs.
            model: wavefunction with ``.forward(x) -> (B,)``.
            graph: Lattice graph.
            n_steps: Number of burn-in sweeps.
            **kwargs: Forwarded to step().

        Returns:
            fxs: (B, N_sites) int64 after burn-in (the amplitudes
                returned by each ``step()`` are discarded).
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

    def __init__(self, hopping_rate: float = 0.25) -> None:
        """Set the hopping-vs-exchange proposal mix."""
        self.hopping_rate = hopping_rate

    @torch.inference_mode()
    def step(
        self,
        fxs: torch.Tensor,
        model: WavefunctionModel_GPU,
        graph: Graph,
        compile: bool = False,
        verbose: bool = False,
        use_log_amp: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """One Metropolis sweep over all lattice edges.

        Iterates over ``graph.row_edges`` then ``graph.col_edges`` in
        dict order.  At each edge (i, j) every walker gets an
        exchange-or-hopping proposal from
        :func:`propose_exchange_or_hopping_vec` (conserves N_up and
        N_down; symmetric kernel), the model is evaluated on the
        changed walkers only (or on the full batch when ``compile`` /
        ``model._exported``), and each changed walker is accepted with
        probability ``min(1, |psi'|^2 / |psi|^2)`` (log form:
        ``exp(2 * (log|psi'| - log|psi|))``).  Accepted walkers have
        their configs and cached amplitudes updated in place.

        Side effects: ``fxs`` is modified in place (and returned);
        ``self._last_accepted_moves`` is set to the number of accepted,
        config-changing micro-moves in this sweep; the global torch RNG
        advances.

        Args:
            fxs: (B, N_sites) int64 walker configs (updated in place).
            model: wavefunction with ``.forward(x) -> (B,)`` and, when
                ``use_log_amp``, ``.forward_log(x) -> (sign, log_abs)``.
            graph: Lattice graph with ``.row_edges`` / ``.col_edges``.
            compile: If True, always evaluate all B configs (no partial
                batching) so torch.compile / export sees a fixed shape.
                Forced on when ``model._exported`` is set.
            verbose: Print per-edge timing info.
            use_log_amp: If True, work in log-space and return
                (signs, log_abs) instead of amps.
            **kwargs: Ignored (accepted for interface compatibility).

        Returns:
            fxs: (B, N_sites) int64 updated configs.
            amps_out: (B,) amplitudes, or (signs, log_abs) tuple of
                (B,) tensors when use_log_amp=True.
        """
        if use_log_amp:
            cur_signs, cur_log_abs = model.forward_log(fxs)
        else:
            current_amps = model(fxs)
        B = fxs.shape[0]
        device = fxs.device

        # Forward the FULL (fixed-size) batch every edge when the model
        # is exported/compiled: the default path only forwards the
        # changed subset proposed_fxs[new_flags], whose row count varies
        # per edge and would force torch.compile to recompile each time.
        # Auto-detect so callers don't have to pass compile=True after
        # model.export_and_compile().
        compile = compile or getattr(model, '_exported', False)

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

        # Count effective micro-moves (accepted, config-changing) over
        # the sweep; stored in self._last_accepted_moves for diagnostics.
        accepted = torch.zeros((), device=device, dtype=torch.long)

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
            # A walker on an exact node (|psi(x)| = 0) has zero
            # stationary weight: let it leave on any proposal. Without
            # this a 0/0 ratio is nan and ``probs < nan`` is always
            # False, freezing the walker for the rest of the run.
            if use_log_amp:
                cur_zero = torch.isneginf(cur_log_abs)
            else:
                cur_zero = current_amps == 0
            accept_mask = new_flags & ((probs < ratio) | cur_zero)
            accepted += accept_mask.sum()

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

        self._last_accepted_moves = int(accepted)

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


class MetropolisAoMixSpinfulSamplerGPU(SamplerGPU):
    """Faithful port of Ao's quantax ``MixSampler([ParticleHop,
    SiteExchange])`` for spinful fermions.

    Each micro-move flips one coin (shared by all walkers, matching
    quantax's per-sweep-step sampler choice) to pick a kernel:

    * with prob ``hopping_rate`` -> ParticleHop: pick a random
      occupied mode (a particle), then a random same-spin-sector
      neighbor mode, and swap -> a single spin hops one bond;
    * else -> SiteExchange: pick a random lattice edge per walker
      and swap the two full site states.

    One ``step()`` runs ``n_moves_per_step`` such moves (default =
    ``2 * N_sites``, matching Ao's ``sweep_steps = 2 * N``). Both
    kernels conserve N_up and N_down separately and allow null moves
    S'=S (no skip / resample), exactly as quantax does.

    Args:
        hopping_rate: Probability of a ParticleHop move per step.
            Default 0.5 (Ao mixes ParticleHop/SiteExchange 50/50).
        n_moves_per_step: Moves per ``step()``. None -> 2 * N_sites.
    """

    def __init__(
        self,
        hopping_rate: float = 0.5,
        n_moves_per_step: Optional[int] = None,
    ) -> None:
        """Store proposal settings; device buffers are built lazily."""
        self.hopping_rate = hopping_rate
        self.n_moves_per_step = n_moves_per_step
        self._edges = None        # (E, 2) long
        self._mode_nb = None      # (2N, max_deg) long, -1 padded
        self._pick_occupied = None

    def _build(self, graph: Graph, N: int, device: torch.device) -> None:
        """Cache the edge table and the same-sector mode-neighbor
        table on ``device`` (mirrors quantax _get_site_neighbors).

        Collects ``graph.row_edges`` + ``graph.col_edges`` into
        ``self._edges`` ((E, 2) long) and builds a symmetric, deduped
        site adjacency padded with -1 to the max degree.  Spin-up
        modes (0..N-1) use the site neighbors; spin-down modes
        (N..2N-1) use the same neighbors shifted by N.  The result is
        stored in ``self._mode_nb`` ((2N, max_deg) long, -1 padded).

        Args:
            graph: Lattice graph with ``.row_edges`` / ``.col_edges``.
            N: number of lattice sites.
            device: device on which to allocate the tables.
        """
        all_edges = []
        for edges in graph.row_edges.values():
            all_edges.extend(edges)
        for edges in graph.col_edges.values():
            all_edges.extend(edges)
        self._edges = torch.tensor(
            all_edges, dtype=torch.long, device=device,
        )

        # Site adjacency (symmetric, deduped) -> padded neighbor table.
        adj = [set() for _ in range(N)]
        for (i, j) in all_edges:
            adj[i].add(j)
            adj[j].add(i)
        max_deg = max(len(a) for a in adj)
        site_nb = torch.full(
            (N, max_deg), -1, dtype=torch.long, device=device,
        )
        for s in range(N):
            ns = sorted(adj[s])
            site_nb[s, :len(ns)] = torch.tensor(
                ns, dtype=torch.long, device=device,
            )
        # Spin-up modes use site neighbors; spin-down modes (s + N)
        # use the same neighbors shifted by N (keep -1 fills as -1).
        dn_nb = torch.where(site_nb == -1, site_nb, site_nb + N)
        self._mode_nb = torch.cat([site_nb, dn_nb], dim=0)  # (2N,maxdeg)

    @torch.inference_mode()
    def step(
        self,
        fxs: torch.Tensor,
        model: WavefunctionModel_GPU,
        graph: Graph,
        compile: bool = False,
        verbose: bool = False,
        use_log_amp: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """One MCMC sweep of ``n_moves_per_step`` mixed moves.

        Each move draws one shared coin: with probability
        ``hopping_rate`` all walkers get a :func:`propose_particle_hop`
        proposal, otherwise a :func:`propose_site_exchange` proposal.
        The full batch is forwarded through the model and each walker
        whose config changed is accepted with probability
        ``min(1, |psi'|^2 / |psi|^2)`` (log form
        ``exp(2 * (log|psi'| - log|psi|))``); both kernels are
        symmetric so no Hastings factor is needed.

        Side effects: on the first call (or after a device change) the
        edge / mode-neighbor tables are built via ``_build`` and
        ``self._pick_occupied`` is fixed from the particle number of
        ``fxs[0]`` (all walkers are assumed to share it); ``fxs`` is
        modified in place; ``self._last_accepted_moves`` is set to the
        number of accepted, config-changing moves; the global torch RNG
        advances.

        Args:
            fxs: (B, N_sites) int64 walker configs (updated in place).
            model: wavefunction with ``.forward(x) -> (B,)`` and, when
                ``use_log_amp``, ``.forward_log(x) -> (sign, log_abs)``.
            graph: Lattice graph with ``.row_edges`` / ``.col_edges``.
            compile: Unused (every move forwards the full batch).
            verbose: Print acceptance-rate summary.
            use_log_amp: If True, work in log-space and return
                (signs, log_abs) instead of amps.
            **kwargs: Ignored (accepted for interface compatibility).

        Returns:
            fxs: (B, N_sites) int64 updated configs.
            amps_out: (B,) amplitudes, or (signs, log_abs) tuple of
                (B,) tensors when use_log_amp=True.
        """
        device = fxs.device
        B, N = fxs.shape

        if self._edges is None or self._edges.device != device:
            self._build(graph, N, device)
        if self._pick_occupied is None:
            # Pick from the minority set (matches quantax
            # hopping_particle = 1 if 2*Ntotal <= Nmodes else -1).
            occ = _site_to_mode_occ(fxs)
            n_particles = int(occ[0].sum().item())
            self._pick_occupied = (2 * n_particles <= 2 * N)

        n_moves = self.n_moves_per_step or (2 * N)

        if use_log_amp:
            cur_signs, cur_log_abs = model.forward_log(fxs)
        else:
            current_amps = model(fxs)

        # One coin per micro-step, shared by all walkers (as quantax).
        use_hop = (
            torch.rand(n_moves, device=device) < self.hopping_rate
        ).tolist()

        if verbose:
            t0 = time.time()
            n_accept = 0
            n_propose = 0
            n_hop = 0

        # Count effective micro-moves (accepted, config-changing) over
        # the sweep; stored in self._last_accepted_moves for diagnostics.
        accepted = torch.zeros((), device=device, dtype=torch.long)

        for hop in use_hop:
            if hop:
                proposed_fxs, new_flags = propose_particle_hop(
                    fxs, self._mode_nb, self._pick_occupied,
                )
            else:
                proposed_fxs, new_flags = propose_site_exchange(
                    fxs, self._edges,
                )
            if not new_flags.any():
                continue

            # Proposals differ per walker, so just forward the full
            # batch (unchanged walkers give ratio 1, harmless).
            if use_log_amp:
                prop_signs, prop_log_abs = model.forward_log(proposed_fxs)
                ratio = torch.exp(
                    2.0 * (prop_log_abs - cur_log_abs),
                )
            else:
                proposed_amps = model(proposed_fxs)
                ratio = (
                    (proposed_amps.abs() ** 2)
                    / (current_amps.abs() ** 2)
                )

            probs = torch.rand(B, device=device)
            # A walker on an exact node (|psi(x)| = 0) has zero
            # stationary weight: let it leave on any proposal. Without
            # this a 0/0 ratio is nan and ``probs < nan`` is always
            # False, freezing the walker for the rest of the run.
            if use_log_amp:
                cur_zero = torch.isneginf(cur_log_abs)
            else:
                cur_zero = current_amps == 0
            accept_mask = new_flags & ((probs < ratio) | cur_zero)
            accepted += accept_mask.sum()

            if verbose:
                n_hop += int(hop)
                n_propose += int(new_flags.sum().item())
                n_accept += int(accept_mask.sum().item())

            if accept_mask.any():
                fxs[accept_mask] = proposed_fxs[accept_mask]
                if use_log_amp:
                    cur_signs[accept_mask] = prop_signs[accept_mask]
                    cur_log_abs[accept_mask] = prop_log_abs[accept_mask]
                else:
                    current_amps[accept_mask] = (
                        proposed_amps[accept_mask]
                    )

        self._last_accepted_moves = int(accepted)

        if verbose:
            t1 = time.time()
            acc_rate = (
                n_accept / n_propose if n_propose > 0 else 0.0
            )
            print(
                f"AoMix sweep: {t1-t0:.4f}s, {n_moves} moves "
                f"({n_hop} hop / {n_moves-n_hop} exchange), B={B}, "
                f"accept {n_accept}/{n_propose} ({acc_rate:.3f})"
            )

        if use_log_amp:
            return fxs, (cur_signs, cur_log_abs)
        return fxs, current_amps


#=== Utility functions for Metropolis-Hastings sampling on spin systems ===#
def propose_spin_exchange_vec(
    i: int,
    j: int,
    current_configs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propose spin exchange on edge (i,j) for all walkers.

    For spin-1/2 configs encoded as {0, 1}, swaps the values at sites
    i and j when they differ (conserves total Sz).  The move is
    deterministic given the edge, hence a symmetric kernel.  Walkers
    with equal spins at i and j are left unchanged.

    Args:
        i: first site index.
        j: second site index.
        current_configs: (B, N_sites) int64 spin configs in {0, 1}.

    Returns:
        proposed_configs: (B, N_sites) int64 proposals (a new tensor).
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
        model: WavefunctionModel_GPU,
        graph: Graph,
        compile: bool = False,
        verbose: bool = False,
        use_log_amp: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """One Metropolis sweep over all lattice edges.

        Iterates over ``graph.row_edges`` then ``graph.col_edges``; at
        each edge (i, j) every walker gets a
        :func:`propose_spin_exchange_vec` proposal (swap the two spins
        if they differ; conserves Sz; symmetric kernel).  The model is
        evaluated on the changed walkers only (or on the full batch
        when ``compile`` / ``model._exported``), and each changed
        walker is accepted with probability
        ``min(1, |psi'|^2 / |psi|^2)`` (log form
        ``exp(2 * (log|psi'| - log|psi|))``).

        Side effects: ``fxs`` is modified in place (and returned); the
        global torch RNG advances.  Unlike the spinful samplers this
        class does not record ``_last_accepted_moves``.

        Args:
            fxs: (B, N_sites) int64 walker configs (updated in place).
            model: wavefunction with ``.forward(x) -> (B,)`` and, when
                ``use_log_amp``, ``.forward_log(x) -> (sign, log_abs)``.
            graph: Lattice graph with ``.row_edges`` / ``.col_edges``.
            compile: If True, evaluate all B configs per edge (no
                partial batching).  Forced on when ``model._exported``.
            verbose: Print per-edge timing info.
            use_log_amp: If True, work in log-space and return
                (signs, log_abs) instead of amps.
            **kwargs: Ignored (accepted for interface compatibility).

        Returns:
            fxs: (B, N_sites) int64 updated configs.
            amps_out: (B,) amplitudes, or (signs, log_abs) tuple of
                (B,) tensors when use_log_amp=True.
        """
        if use_log_amp:
            cur_signs, cur_log_abs = model.forward_log(fxs)
        else:
            current_amps = model(fxs)
        B = fxs.shape[0]
        device = fxs.device

        # See MetropolisExchangeSpinfulSamplerGPU.step: auto-use the
        # full-batch path when the model is exported/compiled, so the
        # variable-length proposed_fxs[new_flags] forward doesn't force
        # torch.compile to recompile every edge.
        compile = compile or getattr(model, '_exported', False)

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
            # A walker on an exact node (|psi(x)| = 0) has zero
            # stationary weight: let it leave on any proposal. Without
            # this a 0/0 ratio is nan and ``probs < nan`` is always
            # False, freezing the walker for the rest of the run.
            if use_log_amp:
                cur_zero = torch.isneginf(cur_log_abs)
            else:
                cur_zero = current_amps == 0
            accept_mask = new_flags & ((probs < ratio) | cur_zero)

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


class DirectProposalMCMCSamplerSpinGPU(SamplerGPU):
    """Direct-sample-as-proposal Metropolis-Hastings sampler for spin PEPS.

    At each step, draws B fresh proposals via `model.direct_sample_vmap(...)`
    (vectorized over walkers) and accepts each independently with probability
    min(1, w(S')/w(S)) where w(S) = |Ψ(S)|² / p_c(S). Targets p_Ψ.
    Per-walker `log p_c` is cached across steps (and across `burn_in()`,
    which calls `step()` repeatedly).

    Requires `model` to have a `direct_sample_vmap(u_batch, chi_s, total_sz,
    chi_m, forced_configs)` method (e.g. `PEPS_Model_GPU` in
    pureTNS_spin.py).

    Args:
        chi_s:    Boundary MPS bond for direct sampling. Defaults to model.chi.
        chi_m:    Bottom marginal MPO bond (None or 0 = chi_m=0 approx).
        total_sz: If given, enforce U(1) Sz constraint at sampling time.
        direct_thermal_steps: Number of propose+accept/reject MH sub-steps to
            perform per `step()` call (advances the chain by this many fresh
            direct-sample proposals). Defaults to 1 (original behavior).
        base_seed: Unused — kept for backward compat. Per-walker randomness
            now comes from `torch.rand(B, N, device=...)` driven by
            PyTorch's RNG (seeded once in run scripts).
    """

    def __init__(
        self,
        chi_s: Optional[int] = None,
        chi_m: Optional[int] = None,
        total_sz: Optional[int] = None,
        direct_thermal_steps: int = 1,
        base_seed: Optional[int] = None,
    ) -> None:
        """Store direct-sampling options; the log p_c cache starts empty."""
        self.chi_s = chi_s
        self.chi_m = chi_m
        self.total_sz = total_sz
        self.direct_thermal_steps = direct_thermal_steps
        self.base_seed = base_seed   # unused; kept for backward compat
        self._log_pc = None      # (B,) float64 on device, current chain state
        self._step_idx = 0       # used only for verbose-print labelling

    @torch.inference_mode()
    def step(
        self,
        fxs: torch.Tensor,
        model: WavefunctionModel_GPU,
        graph: Graph,
        compile: bool = False,
        verbose: bool = False,
        use_log_amp: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """One MH step: B fresh direct-sample proposals + accept/reject.

        Runs ``direct_thermal_steps`` sub-steps.  Each draws B
        independent proposals ``S' ~ p_c`` from
        ``model.direct_sample_vmap`` (uniforms ``u`` of shape (B, N)
        from the global torch RNG) together with ``log p_c(S')``, then
        accepts walker-wise with the independence-sampler rule
        ``log u < log w(S') - log w(S)``,
        ``log w(S) = 2 log|psi(S)| - log p_c(S)``.  ``log p_c`` of the
        incoming walkers is evaluated at the start of every call (one
        extra direct-sampling pass with the configs forced), so the
        ratio is exact even right after a parameter update.

        Side effects: ``fxs`` is modified in place (and returned);
        ``self._log_pc`` ((B,) float64) holds ``log p_c`` of the
        current chain state (re-evaluated at every call, so external
        model / walker changes are safe); ``self._step_idx`` counts
        sub-steps for verbose labels.

        Args:
            fxs: (B, N_sites) int64 walker configs on device (updated
                in place).
            model: spin PEPS model with
                ``.direct_sample_vmap(u, chi_s, total_sz, chi_m)`` ->
                ``((B, N) int64, (B,) log p_c)`` and ``.forward(x) ->
                (B,)`` (or ``.forward_log(x) -> (signs, log_abs)`` for
                use_log_amp=True), e.g. ``PEPS_Model_GPU``.
            graph: ignored (kept for SamplerGPU contract).
            compile: ignored (every sub-step forwards the full batch).
            verbose: print per-sub-step timing and acceptance counts.
            use_log_amp: if True, work in log-amplitude space.
            **kwargs: Ignored (accepted for interface compatibility).

        Returns:
            fxs: (B, N_sites) int64 updated configs.
            amps_out: (B,) amplitudes, or (signs, log_abs) tuple of
                (B,) tensors when use_log_amp.
        """
        device = fxs.device
        B, _N = fxs.shape

        # Evaluate current-config amplitudes once; they are kept in sync with
        # the chain as walkers accept, so each thermalization sub-step only
        # needs a fresh forward pass on the proposals.
        if use_log_amp:
            cur_signs, cur_log_abs = model.forward_log(fxs)
        else:
            current_amps = model(fxs)

        # log p_c of the CURRENT walkers under the CURRENT parameters,
        # re-evaluated every call (one extra direct-sampling pass with
        # the configs forced). A value cached from the previous call
        # would be stale after the optimizer step, and the
        # independence-sampler ratio needs w(S) and w(S') at the same
        # theta. It also gives incoming walkers a proper MH treatment
        # instead of an unconditional first accept.
        u_dummy = torch.zeros(
            B, fxs.shape[1], device=device, dtype=torch.float64,
        )
        _, self._log_pc = model.direct_sample_vmap(
            u_dummy,
            chi_s=self.chi_s,
            total_sz=self.total_sz,
            chi_m=self.chi_m,
            forced_configs=fxs,
        )
        del u_dummy

        # `direct_thermal_steps` propose+accept/reject MH sub-steps per call.
        for _ in range(self.direct_thermal_steps):
            # ---- 1. Draw B fresh proposals via direct_sample_vmap ----
            if verbose:
                t0 = time.time()
            u_batch = torch.rand(
                B, fxs.shape[1], device=device, dtype=torch.float64,
            )
            proposed_fxs_int, proposed_log_pc = model.direct_sample_vmap(
                u_batch,
                chi_s=self.chi_s,
                total_sz=self.total_sz,
                chi_m=self.chi_m,
            )
            proposed_fxs = proposed_fxs_int.to(fxs.dtype)
            if verbose:
                t_propose = time.time() - t0

            # ---- 2. Evaluate proposal amplitudes via batched forward ----
            if verbose:
                t0 = time.time()
            if use_log_amp:
                prop_signs, prop_log_abs = model.forward_log(proposed_fxs)
            else:
                proposed_amps = model(proposed_fxs)
            if verbose:
                t_forward = time.time() - t0

            # ---- 3. MH accept/reject ----
            # log w = 2 log|Ψ| - log p_c
            if use_log_amp:
                log_w_prop = 2.0 * prop_log_abs - proposed_log_pc
            else:
                log_w_prop = (
                    2.0 * torch.log(proposed_amps.abs() + 1e-300)
                    - proposed_log_pc
                )

            if use_log_amp:
                log_w_curr = 2.0 * cur_log_abs - self._log_pc
            else:
                log_w_curr = (
                    2.0 * torch.log(current_amps.abs() + 1e-300)
                    - self._log_pc
                )
            log_u = torch.log(torch.rand(B, device=device) + 1e-300)
            accept_mask = log_u < (log_w_prop - log_w_curr)

            # ---- 4. In-place updates for accepted walkers ----
            fxs[accept_mask] = proposed_fxs[accept_mask]
            self._log_pc[accept_mask] = proposed_log_pc[accept_mask]

            # Keep current-config amplitudes in sync for the next sub-step.
            if use_log_amp:
                cur_signs[accept_mask] = prop_signs[accept_mask]
                cur_log_abs[accept_mask] = prop_log_abs[accept_mask]
            else:
                current_amps[accept_mask] = proposed_amps[accept_mask]

            if verbose:
                n_acc = int(accept_mask.sum().item())
                print(
                    f"  DirectMH step {self._step_idx}: "
                    f"propose {t_propose:.3f}s, "
                    f"forward {t_forward:.3f}s, "
                    f"accept {n_acc}/{B}",
                )
            self._step_idx += 1

        if use_log_amp:
            return fxs, (cur_signs, cur_log_abs)
        else:
            return fxs, current_amps


class DirectProposalMCMCSamplerSpinfulGPU(SamplerGPU):
    """Direct-sample-as-proposal Metropolis-Hastings sampler for fPEPS.

    Spinful-fermionic analogue of `DirectProposalMCMCSamplerSpinGPU`.
    At each step, draws B fresh proposals via `model.direct_sample_vmap(...)`
    (vectorized over walkers) and accepts each independently with probability
    min(1, w(S')/w(S)) where w(S) = |Ψ(S)|² / p_c(S). Targets p_Ψ.
    Per-walker `log p_c` is cached across steps (and across `burn_in()`,
    which calls `step()` repeatedly).

    Requires `model` to have a `direct_sample_vmap(u_batch, chi_s,
    n_up_target, n_dn_target, chi_m, forced_configs)` method (e.g.
    `fPEPS_Model_GPU`).

    Args:
        chi_s:       Boundary MPS bond for direct sampling. Defaults to
            model.chi.
        chi_m:       Bottom marginal MPO bond (must be None/0; chi_m>0 not
            implemented for fermions).
        n_up_target: If given, enforce U(1) up-count constraint at sampling.
        n_dn_target: If given, enforce U(1) down-count constraint.
        direct_thermal_steps: Number of propose+accept/reject MH sub-steps to
            perform per `step()` call (advances the chain by this many fresh
            direct-sample proposals). Defaults to 1 (original behavior).
        base_seed:   Unused — kept for backward compat. Per-walker randomness
            comes from `torch.rand(B, N, device=...)`.
    """

    def __init__(
        self,
        chi_s: Optional[int] = None,
        chi_m: Optional[int] = None,
        n_up_target: Optional[int] = None,
        n_dn_target: Optional[int] = None,
        direct_thermal_steps: int = 1,
        base_seed: Optional[int] = None,
    ) -> None:
        """Store direct-sampling options; the log p_c cache starts empty."""
        self.chi_s = chi_s
        self.chi_m = chi_m
        self.n_up_target = n_up_target
        self.n_dn_target = n_dn_target
        self.direct_thermal_steps = direct_thermal_steps
        self.base_seed = base_seed   # unused; kept for backward compat
        self._log_pc = None      # (B,) float64 on device, current chain state
        self._step_idx = 0       # used only for verbose-print labelling

    @torch.inference_mode()
    def step(
        self,
        fxs: torch.Tensor,
        model: WavefunctionModel_GPU,
        graph: Graph,
        compile: bool = False,
        verbose: bool = False,
        use_log_amp: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """One MH step: B fresh direct-sample proposals + accept/reject.

        Same algorithm as ``DirectProposalMCMCSamplerSpinGPU.step`` for
        spinful fermions: ``direct_thermal_steps`` sub-steps, each
        drawing B independent proposals ``S' ~ p_c`` (with
        ``log p_c(S')``) from ``model.direct_sample_vmap`` and
        accepting walker-wise via ``log u < log w(S') - log w(S)``,
        ``log w(S) = 2 log|psi(S)| - log p_c(S)``.  ``log p_c`` of the
        incoming walkers is evaluated at the start of every call (one
        extra direct-sampling pass with the configs forced), so the
        ratio is exact even right after a parameter update.

        Side effects: ``fxs`` is modified in place (and returned);
        ``self._log_pc`` ((B,) float64) holds ``log p_c`` of the
        current chain state (re-evaluated at every call, so external
        model / walker changes are safe);
        ``self._step_idx`` counts sub-steps for verbose labels.

        Args:
            fxs: (B, N_sites) int64 walker configs on device (updated
                in place).
            model: fPEPS model with ``.direct_sample_vmap(u, chi_s,
                n_up_target, n_dn_target, chi_m)`` ->
                ``((B, N) int64, (B,) log p_c)`` and ``.forward(x) ->
                (B,)`` (or ``.forward_log(x) -> (signs, log_abs)`` for
                use_log_amp=True), e.g. ``fPEPS_Model_GPU``.
            graph: ignored (kept for SamplerGPU contract).
            compile: ignored (every sub-step forwards the full batch).
            verbose: print per-sub-step timing and acceptance counts.
            use_log_amp: if True, work in log-amplitude space.
            **kwargs: Ignored (accepted for interface compatibility).

        Returns:
            fxs: (B, N_sites) int64 updated configs.
            amps_out: (B,) amplitudes, or (signs, log_abs) tuple of
                (B,) tensors when use_log_amp.
        """
        device = fxs.device
        B, _N = fxs.shape

        # Evaluate current-config amplitudes once; they are kept in sync with
        # the chain as walkers accept, so each thermalization sub-step only
        # needs a fresh forward pass on the proposals.
        if use_log_amp:
            cur_signs, cur_log_abs = model.forward_log(fxs)
        else:
            current_amps = model(fxs)

        # log p_c of the CURRENT walkers under the CURRENT parameters,
        # re-evaluated every call (one extra direct-sampling pass with
        # the configs forced). A value cached from the previous call
        # would be stale after the optimizer step, and the
        # independence-sampler ratio needs w(S) and w(S') at the same
        # theta. It also gives incoming walkers a proper MH treatment
        # instead of an unconditional first accept.
        u_dummy = torch.zeros(
            B, fxs.shape[1], device=device, dtype=torch.float64,
        )
        _, self._log_pc = model.direct_sample_vmap(
            u_dummy,
            chi_s=self.chi_s,
            n_up_target=self.n_up_target,
            n_dn_target=self.n_dn_target,
            chi_m=self.chi_m,
            forced_configs=fxs,
        )
        del u_dummy

        # `direct_thermal_steps` propose+accept/reject MH sub-steps per call.
        for _ in range(self.direct_thermal_steps):
            # ---- 1. Draw B fresh proposals via direct_sample_vmap ----
            if verbose:
                t0 = time.time()
            u_batch = torch.rand(
                B, fxs.shape[1], device=device, dtype=torch.float64,
            )
            proposed_fxs_int, proposed_log_pc = model.direct_sample_vmap(
                u_batch,
                chi_s=self.chi_s,
                n_up_target=self.n_up_target,
                n_dn_target=self.n_dn_target,
                chi_m=self.chi_m,
            )
            proposed_fxs = proposed_fxs_int.to(fxs.dtype)
            if verbose:
                t_propose = time.time() - t0

            # ---- 2. Evaluate proposal amplitudes via batched forward ----
            if verbose:
                t0 = time.time()
            if use_log_amp:
                prop_signs, prop_log_abs = model.forward_log(proposed_fxs)
            else:
                proposed_amps = model(proposed_fxs)
            if verbose:
                t_forward = time.time() - t0

            # ---- 3. MH accept/reject ----
            # log w = 2 log|Ψ| - log p_c
            if use_log_amp:
                log_w_prop = 2.0 * prop_log_abs - proposed_log_pc
            else:
                log_w_prop = (
                    2.0 * torch.log(proposed_amps.abs() + 1e-300)
                    - proposed_log_pc
                )

            if use_log_amp:
                log_w_curr = 2.0 * cur_log_abs - self._log_pc
            else:
                log_w_curr = (
                    2.0 * torch.log(current_amps.abs() + 1e-300)
                    - self._log_pc
                )
            log_u = torch.log(torch.rand(B, device=device) + 1e-300)
            accept_mask = log_u < (log_w_prop - log_w_curr)

            # ---- 4. In-place updates for accepted walkers ----
            fxs[accept_mask] = proposed_fxs[accept_mask]
            self._log_pc[accept_mask] = proposed_log_pc[accept_mask]

            # Keep current-config amplitudes in sync for the next sub-step.
            if use_log_amp:
                cur_signs[accept_mask] = prop_signs[accept_mask]
                cur_log_abs[accept_mask] = prop_log_abs[accept_mask]
            else:
                current_amps[accept_mask] = proposed_amps[accept_mask]

            if verbose:
                n_acc = int(accept_mask.sum().item())
                print(
                    f"  DirectMH(spinful) step {self._step_idx}: "
                    f"propose {t_propose:.3f}s, "
                    f"forward {t_forward:.3f}s, "
                    f"accept {n_acc}/{B}",
                )
            self._step_idx += 1

        if use_log_amp:
            return fxs, (cur_signs, cur_log_abs)
        else:
            return fxs, current_amps


__all__ = [
    "SamplerGPU",
    "MetropolisExchangeSpinfulSamplerGPU",
    "MetropolisAoMixSpinfulSamplerGPU",
    "propose_particle_hop",
    "propose_site_exchange",
    "MetropolisExchangeSpinfulSamplerReuse_GPU",
    "MetropolisExchangeSpinfulSamplerXReuse_GPU",
    "MetropolisExchangeSpinSamplerGPU",
    "MetropolisExchangeSpinSamplerReuse_GPU",
    "MetropolisExchangeSpinSamplerXReuse_GPU",
    "DirectProposalMCMCSamplerSpinGPU",
    "DirectProposalMCMCSamplerSpinfulGPU",
    "propose_spin_exchange_vec",
]


# ==========================================================
# Backward compatibility
# ==========================================================
# The bMPS-reuse code below moved to
# vmc_torch/GPU/tensor_network/reuse.py.  Forwarded lazily (PEP 562)
# rather than re-exported at the top of this module: reuse.py imports
# FROM here, so an eager import would be circular.
_MOVED_TO_TN_REUSE = (
    'MetropolisExchangeSpinfulSamplerReuse_GPU',
    'MetropolisExchangeSpinSamplerReuse_GPU',
    'MetropolisExchangeSpinSamplerXReuse_GPU',
    'MetropolisExchangeSpinfulSamplerXReuse_GPU',
)


def __getattr__(name: str) -> Any:
    """Lazily forward the bMPS-reuse sampler names (PEP 562).

    Looks ``name`` up in ``vmc_torch.GPU.tensor_network.reuse`` when it
    is one of ``_MOVED_TO_TN_REUSE``, importing that module on first
    use so the old ``from vmc_torch.GPU.sampler import ...Reuse_GPU``
    paths keep working without a circular top-level import.

    Args:
        name: attribute requested on this module.

    Returns:
        The forwarded class from ``reuse``.

    Raises:
        AttributeError: for any other missing attribute.
    """
    if name in _MOVED_TO_TN_REUSE:
        from vmc_torch.GPU.tensor_network import reuse
        return getattr(reuse, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
