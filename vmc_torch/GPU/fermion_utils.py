"""Fermionic tensor network pack/unpack utilities.

These handle the pytree structure of fermionic TNs (symmray blocks
with Z2/U1 symmetry labels, Placeholders, etc.) that quimb's
standard pack/unpack doesn't cover.

Copied from vmap/models/_model_base.py to make GPU/ self-contained.
"""
import torch
import quimb as qu
import quimb.tensor as qtn


def _is_vmap_compatible(x):
    """Check if a node is compatible with vmap (Tensor)."""
    return isinstance(x, torch.Tensor)


def _is_quimb_placeholder(x):
    return isinstance(x, qu.tensor.interface.Placeholder)


def pack_ftn(ftn):
    """Pack a fermionic TN into (flat_params, skeleton).

    flat_params: list of Tensors (vmap-compatible leaves).
    skeleton: quimb TN skeleton for unpack_ftn.
    """
    ftn_params_raw, skeleton = qtn.pack(ftn)
    ftn_params = {}
    for key in ftn_params_raw.keys():
        raw_tree = ftn.tensor_map[key].data.to_pytree()
        ftn_params[key] = raw_tree
    flat_ftn_params, _ = qu.utils.tree_flatten(
        ftn_params, get_ref=True, is_leaf=_is_vmap_compatible,
    )
    flat_ftn_params = qu.utils.tree_map(
        lambda x: torch.as_tensor(x),
        flat_ftn_params,
        is_leaf=lambda x: isinstance(x, bool),
    )
    return flat_ftn_params, skeleton


def unpack_ftn(flat_ftn_params, skeleton):
    """Unpack flat params + skeleton back into a fermionic TN."""
    ftn = skeleton.copy()
    # Rebuild pytree structure from the skeleton's current data
    ftn_params_raw, _ = qtn.pack(ftn)
    ftn_params = {}
    for key in ftn_params_raw.keys():
        ftn_params[key] = ftn.tensor_map[key].data.to_pytree()
    _, pytree = qu.utils.tree_flatten(
        ftn_params,
        get_ref=True,
        is_leaf=lambda x: (
            _is_vmap_compatible(x) or _is_quimb_placeholder(x)
        ),
    )
    ftn_params = qu.utils.tree_unflatten(flat_ftn_params, pytree)
    for key in ftn_params.keys():
        new_data = ftn.tensor_map[key].data.from_pytree(
            ftn_params[key],
        )
        ftn.tensor_map[key].modify(data=new_data)
    return ftn


def get_params_ftn(ftn):
    """Get flat parameter list from a fermionic TN."""
    flat_ftn_params, _ = pack_ftn(ftn)
    return flat_ftn_params
