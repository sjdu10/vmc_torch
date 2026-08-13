"""Tensor-network utilities for the GPU VMC pipeline.

Single home for the TN-specific code the VMC framework needs, so that
``VMC.py`` / ``vmc_utils.py`` / ``sampler.py`` / ``vmc_setup.py`` stay
high-level and TN-agnostic:

    linalg dispatch     setup_linalg_hooks
    fermionic TN pack   pack_ftn, unpack_ftn, get_params_ftn
    TN construction     load_or_generate_peps (Z2-fPEPS),
                        generate_random_spin_peps (dense spin PEPS)

The boundary-MPS *environment reuse* machinery is deliberately NOT
here -- it lives in ``reuse.py``, which is a subsystem rather than a
grab bag of helpers.

Moved here from ``GPU/fermion_utils.py`` and ``GPU/vmc_setup.py``; both
keep thin re-export shims so existing imports still work.  New code
should import from ``vmc_torch.GPU.tensor_network``.
"""
import autoray as ar
import quimb as qu
import quimb.tensor as qtn
import torch

from vmc_torch.GPU.torch_utils import (
    size_aware_qr,
    size_aware_svd,
    qr_via_cholesky,
)


# =================================================================
#  Linalg dispatch for TN contraction
# =================================================================


def setup_linalg_hooks(
    random_truncated_svd=False,
    jitter=1e-16,
    driver=None,

    qr_via_eigh=True,
    cholesky_qr=False,
    cholesky_qr_adaptive_jitter=False,
    nonuniform_diag=False,
):
    """Register autoray hooks for SVD and QR dispatch.

    Args:
        nonuniform_diag: if True, use non-uniform diagonal jitter (instead of
            identity) in EIG-based SVD/QR to lift singular value
            degeneracies.  Stabilizes backward for matrices with
            repeated or near-degenerate singular values.
    """
    if random_truncated_svd:
        from symmray.linalg import svd_rand_truncated
        from functools import partial
        svd_rand_truncated_new = partial(
            svd_rand_truncated,
            seed=42,
        )
        ar.register_function("symmray", "svd_truncated", svd_rand_truncated_new)
    else:
        ar.register_function(
            'torch',
            'linalg.svd',
            lambda x: size_aware_svd(
                x, jitter=jitter, driver=driver,
                nonuniform_diag=nonuniform_diag,
            ),
        )
    if qr_via_eigh and cholesky_qr:
        raise ValueError(
            "Cannot use both qr_via_eigh and cholesky_qr."
        )
    if cholesky_qr:
        ar.register_function(
            "torch",
            "linalg.qr",
            lambda x: qr_via_cholesky(
                x, jitter=jitter,
                adaptive_jitter=cholesky_qr_adaptive_jitter,
            ),
        )
    elif qr_via_eigh:
        ar.register_function(
            'torch',
            'linalg.qr',
            lambda x: size_aware_qr(
                x, via_eigh=True, jitter=jitter,
                nonuniform_diag=nonuniform_diag,
            ),
        )
    else: # both False: use default torch.linalg.qr
        # Use default torch.linalg.qr
        pass


# =================================================================
#  Fermionic TN pack / unpack
#
#  These handle the pytree structure of fermionic TNs (symmray blocks
#  with Z2/U1 symmetry labels, Placeholders, etc.) that quimb's
#  standard pack/unpack doesn't cover.
# =================================================================


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


# =================================================================
#  TN construction / loading
# =================================================================


def load_or_generate_peps(
    Lx,
    Ly,
    t,
    U,
    N_f,
    D,
    seed=42,
    dtype=torch.float64,
    scale_factor=4,
    data_root='./',
    file_path=None,
    random_init=False,
    pbc=False,
    saved_peps_name="peps",
    appendix="_U1SU", # for GPU workflow, by default we use U1SU peps
):
    """Load a pre-trained Z2-fPEPS from disk, or generate a random one.

    Args:
        pbc: if True, the random-init branch produces a cyclic
            (torus) PEPS via ``PEPS_fermionic_rand(cyclic=True, ...)``;
            the disk-load branch is structure-agnostic (whatever was
            pickled — OBC or PBC — is returned as-is). When the
            returned PEPS is cyclic (in either branch),
            :func:`standardize_pbc_peps_leg_order` is applied so all
            site tensors share a uniform (UP, LEFT, RIGHT, DOWN, PHYS)
            leg order — required by uniform-channel NN backflows.
    """
    import pickle
    import symmray as sr
    from vmc_torch.fermion_utils import (
        standardize_pbc_peps_leg_order,
        make_pbc_dual_uniform,
    )
    try:
        if random_init:
            raise ValueError("random_init=True: skipping loading from disk.")


        if file_path is not None:
            base = file_path
        else:
            base = (
                f"{data_root}/{Lx}x{Ly}/t={t}_U={U}"
                f"/N={N_f}/Z2/D={D}/"
            )
        params_path = base + f"{saved_peps_name}_su_params{appendix}.pkl"
        skeleton_path = base + f"{saved_peps_name}_skeleton{appendix}.pkl"

        with open(params_path, 'rb') as f:
            params_pkl = pickle.load(f)
        with open(skeleton_path, 'rb') as f:
            skeleton = pickle.load(f)

        peps = qtn.unpack(params_pkl, skeleton)

        for ts in peps.tensors:
            ts.modify(data=ts.data.to_flat() * scale_factor)
            sorted_data = ts.data.sort_stack(inplace=False)
            ts.modify(data=sorted_data)
        for site in peps.sites:
            peps[site].data._label = site
            peps[site].data.indices[-1]._linearmap = (
                (0, 0), (1, 0), (1, 1), (0, 1)
            )
    except Exception as e:
        import symmray as sr

        print(
            f'Could not load Z2-fPEPS from pickle: {e}. '
            f'Generating random Z2-fPEPS instead.'
        )
        peps = sr.networks.PEPS_fermionic_rand(
            "Z2",
            Lx,
            Ly,
            D,
            # Same (Z2 charge, slot-in-sector) order as the phys-leg
            # `_linearmap` of the SU-loaded PEPS above, so that both
            # branches need the SAME basis permutation downstream
            # (fPEPS_Model_GPU hardcodes one for both).  Swapping the
            # middle two entries relabels up <-> down.
            phys_dim=[
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1),
            ],
            subsizes="equal",
            flat=True,
            seed=seed,
            dtype=str(dtype).split(".")[-1],
            cyclic=pbc,
        )
        # Defense-in-depth: sort sectors on every tensor right at
        # generation so the returned PEPS is in canonical form from
        # the source. The disk-load branch above and the PBC post-
        # processing below also sort_stack; this ensures the random-
        # init path is symmetric and consumers that bypass the model
        # (e.g. direct symmray exact contraction for ED comparison)
        # always see a consistent sector layout across sites.
        for ts in peps.tensors:
            ts.modify(data=ts.data.sort_stack(inplace=False))

    # For PBC, normalize:
    #   1) leg order -> (UP, LEFT, RIGHT, DOWN, PHYS), required so
    #      uniform NN backflows map flat output to consistent legs;
    #   2) dual pattern -> (T, T, F, F, F) everywhere, so the
    #      parametrization is manifestly translation-equivariant
    #      (otherwise wrap-affected sites carry site-dependent
    #      fermion phases);
    #   3) sort_stack after the leg transpose: standardize_..._leg_order
    #      permutes leg axes differently per site (each site needed a
    #      different transpose to reach ULRD), which permutes the
    #      stored _sectors array per site. Without re-sorting, two sites
    #      end up with the same SET of sectors but in different stored
    #      ORDER. Uniform NN backflows assume sector index k means the
    #      same (charge tuple) on every site -- sort_stack restores
    #      that by putting every site's _sectors in canonical lex order.
    if pbc:
        peps = standardize_pbc_peps_leg_order(peps)
        peps = make_pbc_dual_uniform(peps)
        for ts in peps.tensors:
            ts.modify(data=ts.data.sort_stack(inplace=False))
            ts.data.phase_sync(inplace=True)

    return peps


def generate_random_spin_peps(
    Lx, Ly, D, seed=42, dtype=torch.float64,
):
    """Generate a random PEPS for spin-1/2 systems.

    Creates a quimb PEPS with physical dimension 2 (spin
    states {0, 1}) and bond dimension D.

    Args:
        Lx, Ly: lattice dimensions.
        D: bond dimension.
        seed: random seed.
        dtype: torch dtype.

    Returns:
        quimb PEPS tensor network.
    """
    dtype_str = str(dtype).split('.')[-1]
    peps = qtn.PEPS.rand(
        Lx, Ly,
        bond_dim=D,
        phys_dim=2,
        dtype=dtype_str,
        seed=seed,
    )
    return peps


__all__ = [
    'setup_linalg_hooks',
    'pack_ftn',
    'unpack_ftn',
    'get_params_ftn',
    'load_or_generate_peps',
    'generate_random_spin_peps',
]
