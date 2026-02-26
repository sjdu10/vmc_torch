import os
import pickle

import autoray as ar
import quimb.tensor as qtn
import torch

from vmc_torch.experiment.vmap.GPU.torch_utils import (
    size_aware_qr,
    size_aware_svd,
)
from vmc_torch.experiment.vmap.GPU.vmc_utils import random_initial_config

DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
    'vmc_torch/experiment/vmap/data'
)


def setup_linalg_hooks(jitter=1e-16, driver=None):
    ar.register_function(
        'torch',
        'linalg.svd',
        lambda x: size_aware_svd(x, jitter=jitter, driver=driver),
    )
    ar.register_function(
        'torch',
        'linalg.qr',
        lambda x: size_aware_qr(x, via_eigh=True, jitter=jitter),
    )


def load_or_generate_peps(
    Lx, Ly, t, U, N_f, D, seed=42, dtype=torch.float64,
    data_root=DEFAULT_DATA_ROOT,
):
    """Load a pre-trained fPEPS from disk, or generate a random one."""
    try:
        u1z2 = True
        appendix = '_U1SU' if u1z2 else ''

        base = (
            f"{data_root}/{Lx}x{Ly}/t={t}_U={U}"
            f"/N={N_f}/Z2/D={D}/"
        )
        params_path = base + f"peps_su_params{appendix}.pkl"
        skeleton_path = base + f"peps_skeleton{appendix}.pkl"

        with open(params_path, 'rb') as f:
            params_pkl = pickle.load(f)
        with open(skeleton_path, 'rb') as f:
            skeleton = pickle.load(f)

        peps = qtn.unpack(params_pkl, skeleton)

        for ts in peps.tensors:
            ts.modify(data=ts.data.to_flat() * 4)
        for site in peps.sites:
            peps[site].data._label = site
            peps[site].data.indices[-1]._linearmap = (
                (0, 0), (1, 0), (1, 1), (0, 1)
            )
    except Exception as e:
        import symmray as sr

        print(
            f'Could not load PEPS from pickle: {e}. '
            f'Generating random PEPS instead.'
        )
        peps = sr.networks.PEPS_fermionic_rand(
            "Z2",
            Lx,
            Ly,
            D,
            phys_dim=[
                (0, 0),
                (1, 1),
                (1, 0),
                (0, 1),
            ],
            subsizes="equal",
            flat=True,
            seed=seed,
            dtype=str(dtype).split(".")[-1],
        )
    return peps


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


def ensure_output_dir(
    Lx, Ly, t, U, N_f, D, data_root=DEFAULT_DATA_ROOT,
):
    """Create output directory and return its path."""
    output_dir = (
        f"{data_root}/GPU/{Lx}x{Ly}/"
        f"t={t}_U={U}/N={N_f}/Z2/D={D}"
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


__all__ = [
    'setup_linalg_hooks',
    'load_or_generate_peps',
    'initialize_walkers',
    'ensure_output_dir',
]
