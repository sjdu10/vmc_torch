"""Tensor-network layer for the GPU VMC pipeline.

Everything TN-specific lives here, so ``VMC.py`` / ``vmc_utils.py`` /
``sampler.py`` / ``vmc_setup.py`` stay high-level and TN-agnostic:

    utils.py   linalg dispatch (setup_linalg_hooks), fermionic TN
               pack/unpack, PEPS construction / loading
    reuse.py   boundary-MPS environment reuse (energy, gradients,
               samplers)

``utils`` is re-exported eagerly.  ``reuse`` is exposed lazily, because
it imports from ``vmc_utils`` / ``sampler`` and an eager import here
would drag that dependency into every consumer of this package (and
risk an import cycle for anything those two modules touch).  Access is
identical either way::

    from vmc_torch.GPU.tensor_network import setup_linalg_hooks
    from vmc_torch.GPU.tensor_network import evaluate_energy_reuse
"""
from .utils import (  # noqa: F401
    generate_random_spin_mps,
    generate_random_spin_peps,
    get_params_ftn,
    load_or_generate_peps,
    pack_ftn,
    setup_linalg_hooks,
    strip_phys_linearmap,
    unpack_ftn,
)

_REUSE_NAMES = (
    'compute_grads_cheap_gpu',
    'detect_changed_row_col_pair',
    'detect_changed_rows',
    'evaluate_energy_reuse',
    'evaluate_energy_reuse_x',
    'MetropolisExchangeSpinfulSamplerReuse_GPU',
    'MetropolisExchangeSpinfulSamplerXReuse_GPU',
    'MetropolisExchangeSpinSamplerReuse_GPU',
    'MetropolisExchangeSpinSamplerXReuse_GPU',
)


def __getattr__(name):
    if name in _REUSE_NAMES:
        from . import reuse
        return getattr(reuse, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    'setup_linalg_hooks',
    'pack_ftn',
    'unpack_ftn',
    'get_params_ftn',
    'load_or_generate_peps',
    'generate_random_spin_peps',
    'generate_random_spin_mps',
    *_REUSE_NAMES,
]
