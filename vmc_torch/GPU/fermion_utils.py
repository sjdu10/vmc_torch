"""Backward-compat shim -- the contents moved to ``tensor_network/utils.py``.

The fermionic TN pack/unpack helpers and ``load_or_generate_peps`` now
live in :mod:`vmc_torch.GPU.tensor_network.utils`, which is the single home for
TN-specific code.  This module re-exports them so existing imports
(``from vmc_torch.GPU.fermion_utils import pack_ftn, ...``) keep
working.  New code should import from ``vmc_torch.GPU.tensor_network.utils``.
"""
from vmc_torch.GPU.tensor_network.utils import (  # noqa: F401
    _is_quimb_placeholder,
    _is_vmap_compatible,
    get_params_ftn,
    load_or_generate_peps,
    pack_ftn,
    unpack_ftn,
)

__all__ = [
    'pack_ftn',
    'unpack_ftn',
    'get_params_ftn',
    'load_or_generate_peps',
]
