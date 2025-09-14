from quimb.tensor.tensor_1d_compress import *
from quimb.tensor.tensor_1d_compress import _tn1d_fit_sum_sweep_1site

def _tn1d_fit_sum_sweep_2site_fermion(
    tn_fit,
    tn_overlaps,
    site_tags,
    max_bond=None,
    cutoff=1e-10,
    envs=None,
    prepare=True,
    reverse=False,
    optimize="auto-hq",
    compute_tdiff=True,
    tns=None,
    **compress_opts,
):
    """Core sweep of the 2-site 1D fit algorithm."""

    N = len(site_tags)
    K = len(tn_overlaps)

    if envs is None:
        envs = {}
        prepare = True

    if prepare:
        for k in range(K):
            envs.setdefault(("L", 0, k), TensorNetwork())
            envs.setdefault(("R", N - 1, k), TensorNetwork())

        if not reverse:
            # move canonical center to left
            tn_fit.canonize_around_(site_tags[0])
            # compute each of K right environments
            for i in range(N - 2, 0, -1):
                site_r = site_tags[i + 1]
                for k, tn_overlap in enumerate(tn_overlaps):
                    tni = envs["R", i + 1, k] | tn_overlap.select(site_r)
                    envs["R", i, k] = tni.contract(all, optimize=optimize)
        else:
            # move canonical center to right
            tn_fit.canonize_around_(site_tags[-1])
            # compute each of K left environments
            for i in range(1, N - 1):
                site_l = site_tags[i - 1]
                for k, tn_overlap in enumerate(tn_overlaps):
                    tni = envs["L", i - 1, k] | tn_overlap.select(site_l)
                    envs["L", i, k] = tni.contract(all, optimize=optimize)

    # track the maximum change in any tensor norm
    max_tdiff = -1.0

    sweep = range(N - 1)
    if reverse:
        sweep = reversed(sweep)

    for i in sweep:
        # print(f"fitting site {i} {'<-' if reverse else '->'} {i+1}")
        site0 = site_tags[i]
        site1 = site_tags[i + 1]

        if not reverse:
            if i > 0:
                site_l = site_tags[i - 1]
                # recalculate K left environments
                for k, tn_overlap in enumerate(tn_overlaps):
                    tni = envs["L", i - 1, k] | tn_overlap.select(site_l)
                    envs["L", i, k] = tni.contract(all, optimize=optimize)
        else:
            if i < N - 2:
                site_r = site_tags[i + 2]
                # recalculate right environment
                for k, tn_overlap in enumerate(tn_overlaps):
                    tni = envs["R", i + 2, k] | tn_overlap.select(site_r)
                    envs["R", i + 1, k] = tni.contract(all, optimize=optimize)

        tfi0 = tn_fit[site0]
        tfi1 = tn_fit[site1]
        (bond,) = tfi0.bonds(tfi1)
        left_inds = tuple(ix for ix in tfi0.inds if ix != bond)
        right_inds = tuple(ix for ix in tfi1.inds if ix != bond)
        tfinew = None

        for k, tn_overlap in enumerate(tn_overlaps):
            # form local overlap
            tnik = (
                envs["L", i, k]
                | tn_overlap.select_any((site0, site1))
                | envs["R", i + 1, k]
            )

            left_env_ind, right_env_ind = None, None
            if type(envs["L", i, k]) is Tensor:
                (left_env_ind,) = tfi0.bonds(envs["L", i, k].contract())
            if type(envs["R", i + 1, k]) is Tensor:
                (right_env_ind,) = tfi1.bonds(envs["R", i + 1, k].contract())

            # remove old tensors
            del tnik["__FIT__", site0]
            del tnik["__FIT__", site1]

            # contract its new value, maintaining index order
            tfiknew = tnik.contract(
                all, optimize=optimize, output_inds=left_inds + right_inds
            )#XXX: insert the parity tensor for fermion TN legs? 

            lind_id, rind_id = None, None
            lind_id = tfiknew.inds.index(left_env_ind) if left_env_ind is not None else None
            rind_id = tfiknew.inds.index(right_env_ind) if right_env_ind is not None else None
            # print(f'duals of env legs: left {lind_id} right {rind_id}')
            # print(tfiknew.data.duals[lind_id] if lind_id is not None else 'None', tfiknew.data.duals[rind_id] if rind_id is not None else 'None')

            if lind_id is not None and rind_id is not None:
                if tfiknew.data.duals[lind_id]:
                    tfiknew.data.phase_flip(lind_id, inplace=True)
                else:
                    tfiknew.data.phase_flip(rind_id, inplace=True)
            elif lind_id is not None and rind_id is None:
                if tfiknew.data.duals[lind_id]:
                    tfiknew.data.phase_flip(lind_id, inplace=True)
            elif lind_id is None and rind_id is not None:
                if tfiknew.data.duals[rind_id]:
                    tfiknew.data.phase_flip(rind_id, inplace=True)

            # sum into fitted tensor
            if tfinew is None:
                tfinew = tfiknew
            else:
                tfinew += tfiknew

        tfinew0, tfinew1 = tfinew.split(
            max_bond=max_bond,
            cutoff=cutoff,
            absorb="left" if reverse else "right",
            left_inds=left_inds,
            right_inds=right_inds,
            bond_ind=bond,
            get="tensors",
            **compress_opts,
        )

        if compute_tdiff:
            # track change in tensor norm
            dt = (tfi0 | tfi1).distance_normalized(tfinew0 | tfinew1)
            max_tdiff = max(max_tdiff, dt)

        # reinsert into all viewing tensor networks
        tfinew0.transpose_like_(tfi0)
        tfinew1.transpose_like_(tfi1)
        tfi0.modify(data=tfinew0.data, left_inds=tfinew0.left_inds)
        tfi1.modify(data=tfinew1.data, left_inds=tfinew1.left_inds)

        # Update the tn_overlaps to use the new tn_fit
        tn_fit_conj = tn_fit.conj()
        tn_fit_conj.add_tag("__FIT__")
        for ts in (tn_fit_conj[site0]|tn_fit_conj[site1]).tensors:
            if len(ts.data._oddpos) % 2 == 1:
                ts.data.phase_global(inplace=True)
        tn_overlaps = [(tn_fit_conj | tn) for tn in tns]

    return max_tdiff

def tensor_network_1d_compress_fit_fermion(
    tns,
    max_bond=None,
    cutoff=None,
    tn_fit=None,
    bsz="auto",
    initial_bond_dim=8,
    max_iterations=10,
    tol=0.0,
    site_tags=None,
    cutoff_mode="rsum2",
    sweep_sequence="RL",
    normalize=False,
    permute_arrays=True,
    optimize="auto-hq",
    canonize=True,
    sweep_reverse=False,
    equalize_norms=False,
    inplace_fit=False,
    inplace=False,
    progbar=False,
    **compress_opts,
):
    """Compress any 1D-like (can have multiple tensors per site) tensor network
    or sum of tensor networks to an exactly 1D (one tensor per site) tensor
    network of bond dimension `max_bond` using the 1-site or 2-site variational
    fitting (or 'DMRG-style') method. The tensor network(s) can have arbitrary
    inner and outer structure.

    This method has the lowest scaling of the standard 1D compression methods
    and can also provide the most accurate compression, but the actual speed
    and accuracy depend on the number of iterations required and initial guess,
    making it a more 'hands-on' method.

    It's also the only method to support fitting to a sum of tensor networks
    directly, rather than having to forming the explicitly summed TN first.

    Parameters
    ----------
    tns : TensorNetwork or Sequence[TensorNetwork]
        The tensor network or tensor networks to compress. Each tensor network
        should have the same outer index structure, and within each tensor
        network every tensor should have exactly one of the site tags.
    max_bond : int
        The maximum bond dimension to compress to. If not given, this is set
        as the maximum bond dimension of the initial guess tensor network, if
        any, else infinite for ``bsz=2``.
    cutoff : float, optional
        A dynamic threshold for discarding singular values when compressing.
        This is only relevant for the 2-site sweeping algorithm (``bsz=2``),
        where it defaults to 1e-10.
    tn_fit : TensorNetwork, dict, or str, optional
        An initial guess for the compressed tensor network. It should matching
        outer indices and site tags with ``tn``. If a `dict`, this is assumed
        to be options to supply to `tensor_network_1d_compress` to construct
        the initial guess, inheriting various defaults like `initial_bond_dim`.
        If a string, e.g. ``"zipup"``, this is shorthand for that compression
        method with default settings. If not given, a random 1D tensor network
        will be used.
    bsz : {"auto", 1, 2}, optional
        The size of the block to optimize while sweeping. If ``"auto"``, this
        will be inferred from the value of ``max_bond`` and ``cutoff``.
    initial_bond_dim : int, optional
        The initial bond dimension to use when creating the initial guess. This
        is only relevant if ``tn_fit`` is not given. For each sweep the allowed
        bond dimension is doubled, up to ``max_bond``. For 1-site this occurs
        via explicit bond expansion, while for 2-site it occurs during the
        2-site tensor decomposition.
    max_iterations : int, optional
        The maximum number of variational sweeps to perform.
    tol : float, optional
        The convergence tolerance, in terms of local tensor distance
        normalized. If zero, there will be exactly ``max_iterations`` sweeps.
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site, in the order given by ``site_tags``.
    cutoff_mode : {"rsum2", "rel", ...}, optional
        The mode to use when truncating the singular values of the decomposed
        tensors. See :func:`~quimb.tensor.tensor_split`, if using the 2-site
        sweeping algorithm.
    sweep_sequence : str, optional
        The sequence of sweeps to perform, e.g. ``"LR"`` means first sweep left
        to right, then right to left. The sequence is cycled.
    normalize : bool, optional
        Whether to normalize the final tensor network, making use of the fact
        that the output tensor network is in left or right canonical form.
    permute_arrays : bool or str, optional
        Whether to permute the array indices of the final tensor network into
        canonical order. If ``True`` will use the default order, otherwise if a
        string this specifies a custom order.
    optimize : str, optional
        The contraction path optimizer to use.
    canonize : bool, optional
        Dummy argument to match the signature of other compression methods.
    sweep_reverse : bool, optional
        Whether to sweep in the reverse direction, swapping whether the final
        tensor network is in right or left canonical form, which also depends
        on the last sweep direction.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    inplace_fit : bool, optional
        Whether to perform the compression inplace on the initial guess tensor
        network, ``tn_fit``, if supplied.
    inplace : bool, optional
        Whether to perform the compression inplace on the target tensor network
        supplied, or ``tns[0]`` if a sequence to sum is supplied.
    progbar : bool, optional
        Whether to show a progress bar. Note the progress bar shows the maximum
        change of any single tensor norm, *not* the global change in norm or
        truncation error.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split`, if using the 2-site
        sweeping algorithm.

    Returns
    -------
    TensorNetwork
        The compressed tensor network. Depending on ``sweep_reverse`` and the
        last sweep direction, the canonical center will be at either L:
        ``site_tags[0]`` or R: ``site_tags[-1]``, or the opposite if
        ``sweep_reverse``.
    """
    if not canonize:
        warnings.warn("`canonize=False` is ignored for the `fit` method.")

    if isinstance(tns, TensorNetwork):
        # fit to single tensor network
        tns = (tns,)
    else:
        # fit to sum of tensor networks
        tns = tuple(tns)

    # how to partition the tensor network(s)
    if site_tags is None:
        site_tags = next(
            tn.site_tags for tn in tns if hasattr(tn, "site_tags")
        )

    tns = tuple(
        enforce_1d_like(tn, site_tags=site_tags, inplace=inplace) for tn in tns
    )

    # choose the block size of the sweeping function
    if bsz == "auto":
        if max_bond is not None:
            if (cutoff is None) or (cutoff == 0.0):
                # max_bond specified, no cutoff -> 1-site
                bsz = 1
            else:
                # max_bond and cutoff specified -> 2-site
                bsz = 2
        else:
            if cutoff == 0.0:
                # no max_bond or cutoff -> 1-site
                bsz = 1
            else:
                # no max_bond, but cutoff -> 2-site
                bsz = 2
    f_sweep = {
        1: _tn1d_fit_sum_sweep_1site,
        2: _tn1d_fit_sum_sweep_2site_fermion,
    }[bsz]

    if cutoff is None:
        # set default cutoff
        cutoff = 1e-10 if bsz == 2 else 0.0

    if bsz == 2:
        compress_opts["cutoff_mode"] = cutoff_mode

    # choose our initial guess
    if not isinstance(tn_fit, TensorNetwork):
        if max_bond is None:
            if bsz == 1:
                raise ValueError(
                    "Need to specify at least one of `max_bond` "
                    "or `tn_fit` when using 1-site sweeping."
                )
            max_bond = float("inf")
            current_bond_dim = initial_bond_dim
        else:
            # don't start larger than the target bond dimension
            current_bond_dim = max(initial_bond_dim, max_bond) #XXX: should be min in final code

        if tn_fit is None:
            # random initial guess
            tn_fit = TN_matching(
                tns[0], max_bond=current_bond_dim, site_tags=site_tags
            )
        else:
            if isinstance(tn_fit, str):
                tn_fit = {"method": tn_fit}
            tn_fit.setdefault("max_bond", current_bond_dim)
            tn_fit.setdefault("cutoff", cutoff)
            tn_fit.setdefault("site_tags", site_tags)
            tn_fit.setdefault("optimize", optimize)
            tn_fit = tensor_network_1d_compress(tns[0], **tn_fit)
            inplace_fit = True
    else:
        # a guess was supplied
        current_bond_dim = tn_fit.max_bond()
        if max_bond is None:
            # assume we want to limit bond dimension to the initial guess
            max_bond = current_bond_dim

    # choose to conjugte the smaller fitting network
    # tn_fit = tn_fit.conj(inplace=inplace_fit)
    # tn_fit.add_tag("__FIT__")
    tn_fit_conj = tn_fit.conj()
    tn_fit_conj.add_tag("__FIT__")
    for ts in tn_fit_conj.tensors:
        if len(ts.data._oddpos) % 2 == 1:
            ts.data.phase_global(inplace=True)

    # note these are all views of `tn_fit` and thus will update as it does
    # tn_overlaps = [(tn_fit | tn) for tn in tns]

    # this is not views of `tn_fit` and thus will NOT update as it does
    tn_overlaps = [(tn_fit_conj | tn) for tn in tns]
    

    if any(tn_overlap.outer_inds() for tn_overlap in tn_overlaps):
        raise ValueError(
            "The outer indices of one or more of "
            "`tns` and `tn_fit` don't seem to match."
        )

    sweeps = itertools.cycle(sweep_sequence)
    if max_iterations is None:
        its = itertools.count()
    else:
        its = range(max_iterations)

    envs = {}
    old_direction = ""

    if progbar:
        from quimb.utils import progbar as ProgBar

        its = ProgBar(its, total=max_iterations)

    # whether to compute the maximum change in tensor norm
    compute_tdiff = (tol != 0.0) or progbar

    try:
        for i in its:
            next_direction = next(sweeps)
            reverse = {"R": False, "L": True}[next_direction]
            if sweep_reverse:
                reverse = not reverse

            if current_bond_dim < max_bond:
                # double bond dimension, up to max_bond
                current_bond_dim = min(2 * current_bond_dim, max_bond)

            # perform a single sweep
            max_tdiff = f_sweep(
                tn_fit,
                tn_overlaps,
                max_bond=current_bond_dim,
                cutoff=cutoff,
                envs=envs,
                prepare=(i == 0) or (next_direction == old_direction),
                site_tags=site_tags,
                reverse=reverse,
                optimize=optimize,
                compute_tdiff=compute_tdiff,
                tns=tns,
                **compress_opts,
            )

            if progbar:
                its.set_description(f"max_tdiff={max_tdiff:.2e}")
            if tol != 0.0 and max_tdiff < tol:
                # converged
                break

            old_direction = next_direction

    except KeyboardInterrupt:
        pass
    finally:
        if progbar:
            its.close()

    # tn_fit.drop_tags("__FIT__")
    # tn_fit.conj_()

    if normalize:
        if reverse:
            tn_fit[site_tags[0]].normalize_()
        else:
            tn_fit[site_tags[-1]].normalize_()
        
    if inplace:
        tn0 = tns[0]
        tn0.remove_all_tensors()
        tn0.add_tensor_network(
            tn_fit, virtual=not inplace_fit, check_collisions=False
        )
        tn_fit = tn0

    # possibly put the array indices in canonical order (e.g. when MPS or MPO)
    possibly_permute_(tn_fit, permute_arrays)

    # XXX: do better than simply waiting til the end to equalize norms
    if equalize_norms is True:
        tn_fit.equalize_norms_()
    elif equalize_norms:
        tn_fit.equalize_norms_(value=equalize_norms)

    return tn_fit