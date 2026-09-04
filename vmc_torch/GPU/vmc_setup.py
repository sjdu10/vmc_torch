"""VMC run setup: walker initialization.

TN-specific setup moved to ``tensor_network/utils.py`` (``setup_linalg_hooks``,
``load_or_generate_peps``, ``generate_random_spin_peps``); it is
re-exported below so existing imports keep working.  New code should
take those from ``vmc_torch.GPU.tensor_network.utils``.
"""
from typing import Callable, Optional

import torch

from vmc_torch.GPU.tensor_network.utils import (  # noqa: F401  (re-export)
    generate_random_spin_mps,
    generate_random_spin_peps,
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.fermion_utils import (  # noqa: F401  (re-export)
    standardize_pbc_peps_leg_order,
    make_pbc_dual_uniform,
)


def initialize_walkers(
    init_fn: Callable[..., torch.Tensor],
    batch_size: int,
    seed: int = 42,
    rank: int = 0,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Create a batch of initial configurations.

    Walker ``i`` on rank ``r`` gets seed ``seed + r * batch_size + i``,
    so walkers are distinct across ranks without any communication.

    Args:
        init_fn: ``Callable(seed=int) -> (N_sites,)`` config tensor /
            array for one walker (must accept ``seed`` as a keyword).
            Examples:
              - lambda seed: H.hilbert.random_state(key=seed)
              - lambda seed: random_initial_config(N_f, N, seed=seed)
              - lambda seed: neel_state(N_sites)  # ignores seed
        batch_size: Number of walkers (parallel Markov chains).
        seed: Base random seed.
        rank: Distributed rank (offsets seed per rank).
        device: Target device; None keeps the tensors on CPU.

    Returns:
        (batch_size, N_sites) int64 configurations on ``device``.
    """
    configs = []
    for i in range(batch_size):
        state = init_fn(seed=seed + rank * batch_size + i)
        configs.append(torch.as_tensor(state, dtype=torch.int64))
    return torch.stack(configs).to(device)



def random_spin_config_sz0(
    N_sites: int, seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate a random spin-1/2 config with Sz=0.

    Returns a 1D int64 CPU tensor with exactly N_sites//2
    up-spins (1) and N_sites//2 down-spins (0).

    Args:
        N_sites: number of sites (must be even).
        seed: optional random seed.

    Returns:
        (N_sites,) int64 tensor (CPU).
    """
    if seed is not None:
        gen = torch.Generator(device='cpu').manual_seed(seed)
    else:
        gen = None
    n_up = N_sites // 2
    config = torch.cat([
        torch.ones(n_up, dtype=torch.int64, device='cpu'),
        torch.zeros(
            N_sites - n_up, dtype=torch.int64, device='cpu',
        ),
    ])
    perm = torch.randperm(
        N_sites, generator=gen, device='cpu',
    )
    return config[perm]


__all__ = [
    'setup_linalg_hooks',
    'load_or_generate_peps',
    'standardize_pbc_peps_leg_order',
    'make_pbc_dual_uniform',
    'initialize_walkers',
    'generate_random_spin_peps',
    'generate_random_spin_mps',
    'random_spin_config_sz0',
]
