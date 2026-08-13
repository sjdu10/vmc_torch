"""VMC run setup: walker initialization.

TN-specific setup moved to ``tensor_network/utils.py`` (``setup_linalg_hooks``,
``load_or_generate_peps``, ``generate_random_spin_peps``); it is
re-exported below so existing imports keep working.  New code should
take those from ``vmc_torch.GPU.tensor_network.utils``.
"""
import torch

from vmc_torch.GPU.tensor_network.utils import (  # noqa: F401  (re-export)
    generate_random_spin_peps,
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.fermion_utils import (  # noqa: F401  (re-export)
    standardize_pbc_peps_leg_order,
    make_pbc_dual_uniform,
)


def initialize_walkers(
    init_fn, batch_size, seed=42, rank=0, device=None,
):
    """Create a batch of initial configurations.

    Args:
        init_fn: Callable(seed) -> 1D config tensor/array for one
            walker.  Examples:
              - lambda seed: H.hilbert.random_state(key=seed)
              - lambda seed: random_initial_config(N_f, N, seed=seed)
              - lambda seed: neel_state(N_sites)  # ignores seed
        batch_size: Number of walkers.
        seed: Base random seed.
        rank: Distributed rank (offsets seed per rank).
        device: Target device.
    """
    configs = []
    for i in range(batch_size):
        state = init_fn(seed=seed + rank * batch_size + i)
        configs.append(torch.as_tensor(state, dtype=torch.int64))
    return torch.stack(configs).to(device)



def random_spin_config_sz0(N_sites, seed=None):
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
    'random_spin_config_sz0',
]
