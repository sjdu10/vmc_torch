"""Pure PEPS model with bMPS environment reuse for spin systems.

Uses standard quimb pack/unpack (dense tensors) instead of
the fermionic pack_ftn/unpack_ftn (symmray block-sparse).

PEPS_Model_reuse_GPU  — bMPS environment caching for spin PEPS
"""
import torch
import torch.nn as nn

from ._base import WavefunctionModel_GPU


# =================================================================
#  Dense TN pack/unpack helpers (spin-compatible)
# =================================================================


def pack_tn(tn):
    """Pack a regular TN into (flat_params, skeleton_info).

    Returns:
        flat_params: list of Tensors.
        skeleton_info: (skeleton, pytree_ref) for unpack_tn.
    """
    import quimb as qu
    import quimb.tensor as qtn

    params, skeleton = qtn.pack(tn)
    flat_params, pytree_ref = qu.utils.tree_flatten(
        params, get_ref=True,
    )
    flat_params = [torch.as_tensor(x) for x in flat_params]
    return flat_params, (skeleton, pytree_ref)


def unpack_tn(flat_params, skeleton_info):
    """Unpack flat params back into a TN.

    Args:
        flat_params: list of Tensors.
        skeleton_info: (skeleton, pytree_ref) from pack_tn.

    Returns:
        quimb TN.
    """
    import quimb as qu
    import quimb.tensor as qtn

    skeleton, pytree_ref = skeleton_info
    params = qu.utils.tree_unflatten(flat_params, pytree_ref)
    return qtn.unpack(params, skeleton)


def get_params_tn(tn):
    """Get flat parameter list from a regular TN."""
    flat_params, _ = pack_tn(tn)
    return flat_params


# =================================================================
#  PEPS with bMPS environment reuse (spin-compatible)
# =================================================================


class PEPS_Model_reuse_GPU(WavefunctionModel_GPU):
    """PEPS model with bMPS environment caching and reuse.

    Spin-compatible version of fPEPS_Model_reuse_GPU. Uses
    standard quimb pack/unpack for dense tensors instead of
    fermionic pack_ftn/unpack_ftn.

    Caches boundary MPS environments along x and y directions.
    Supports incremental updates when only a few rows/cols
    change.

    Full contraction path (forward / vamp) goes through the
    base class export+compile machinery. Reuse path is eager
    vmap.
    """

    def __init__(
        self,
        tn,
        max_bond,
        dtype=torch.float64,
        contract_boundary_opts=None,
        **kwargs,
    ):
        import quimb as qu
        import quimb.tensor as qtn

        if contract_boundary_opts is None:
            contract_boundary_opts = {}

        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.contract_boundary_opts = contract_boundary_opts
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        self.chi = max_bond
        self.debug = kwargs.get('debug', False)

        # bMPS skeleton/in_dims dicts
        # Populated by cache_bMPS_skeleton.
        # skeleton_info = (skeleton, pytree_ref)
        self.bMPS_x_skeletons = {}
        self.bMPS_y_skeletons = {}
        self.bMPS_params_x_in_dims = None
        self.bMPS_params_y_in_dims = None

        # Keys whose bMPS is a raw single row/col (D bonds,
        # not chi bonds). For these, skip boundary contraction
        # inside amplitude_reuse to avoid slow small-tensor
        # SVDs under torch.vmap.
        self._raw_bMPS_x_keys = set()
        self._raw_bMPS_y_keys = set()

        # Flatten pytree into a single list for torch
        params_flat, params_pytree = qu.utils.tree_flatten(
            params, get_ref=True,
        )
        self.params_pytree = params_pytree

        params_tensors = [
            torch.as_tensor(x, dtype=self.dtype)
            for x in params_flat
        ]

        self.radius = 0

        # Cache for vmapped reuse functions, keyed by
        # (direction, bMPS_keys_tuple).
        self._vamp_reuse_cache = {}

        super().__init__(params_list=params_tensors)

    # ----- Single-sample amplitude (full contraction) -----

    def amplitude(self, x, params):
        """Single-sample full TN contraction.

        Args:
            x:      (N_sites,) int64 — one configuration
            params: quimb pytree of parameter tensors

        Returns:
            scalar amplitude
        """
        import quimb.tensor as qtn

        tn = qtn.unpack(params, self.skeleton)
        amp = tn.isel({
            tn.site_ind(site): x[i]
            for i, site in enumerate(tn.sites)
        })
        if self.chi > 0:
            amp.contract_boundary_from_xmin_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[0, amp.Lx // 2 - 1],
                **self.contract_boundary_opts,
            )
            amp.contract_boundary_from_xmax_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[amp.Lx // 2, amp.Lx - 1],
                **self.contract_boundary_opts,
            )
        return amp.contract()

    # ----- Single-sample amplitude with reuse -----

    def amplitude_reuse(
        self,
        x,
        params,
        bMPS_keys=None,
        bMPS_params_xmin=None,
        bMPS_params_xmax=None,
        bMPS_params_ymin=None,
        bMPS_params_ymax=None,
        selected_rows=None,
        selected_cols=None,
    ):
        """Single-sample amplitude with cached bMPS environments.

        Falls back to full contraction when no caches given.
        """
        import quimb.tensor as qtn

        tns = qtn.unpack(params, self.skeleton)
        amp = tns.isel({
            tns.site_ind(site): x[i]
            for i, site in enumerate(tns.sites)
        })

        # x-environment reuse
        if (bMPS_params_xmin is not None
                and bMPS_params_xmax is not None
                and bMPS_keys is not None):
            bMPS_min = unpack_tn(
                bMPS_params_xmin,
                self.bMPS_x_skeletons[bMPS_keys[0]],
            )
            bMPS_max = unpack_tn(
                bMPS_params_xmax,
                self.bMPS_x_skeletons[bMPS_keys[1]],
            )
            rows = amp.select(
                [tns.row_tag(row) for row in selected_rows],
                which='any',
            )
            amp_reuse = (bMPS_min | rows | bMPS_max)
            amp_reuse.view_as_(
                qtn.PEPS,
                site_tag_id=tns._site_tag_id,
                x_tag_id=tns._x_tag_id,
                y_tag_id=tns._y_tag_id,
                Lx=tns._Lx,
                Ly=tns._Ly,
                site_ind_id=tns._site_ind_id,
            )
            if self.chi > 0:
                if len(amp_reuse.tensors) > 2 * self.Ly:
                    # Skip boundary contraction when either
                    # env is a raw row (D bonds). The small
                    # SVDs are very slow under torch.vmap.
                    # Let cotengra contract all tensors
                    # directly instead.
                    has_raw = (
                        bMPS_keys[0] in self._raw_bMPS_x_keys
                        or bMPS_keys[1]
                        in self._raw_bMPS_x_keys
                    )
                    if not has_raw:
                        amp_reuse.contract_boundary_from_xmin_(
                            max_bond=self.chi, cutoff=0.0,
                            xrange=[
                                bMPS_keys[0][1],
                                min(
                                    bMPS_keys[0][1] + 1,
                                    self.Lx - 1,
                                ),
                            ],
                            **self.contract_boundary_opts,
                        )
            return amp_reuse.contract()

        # y-environment reuse
        if (bMPS_params_ymin is not None
                and bMPS_params_ymax is not None
                and bMPS_keys is not None):
            bMPS_min = unpack_tn(
                bMPS_params_ymin,
                self.bMPS_y_skeletons[bMPS_keys[0]],
            )
            bMPS_max = unpack_tn(
                bMPS_params_ymax,
                self.bMPS_y_skeletons[bMPS_keys[1]],
            )
            cols = amp.select(
                [tns.col_tag(col) for col in selected_cols],
                which='any',
            )
            amp_reuse = (bMPS_min | cols | bMPS_max)
            amp_reuse.view_as_(
                qtn.PEPS,
                site_tag_id=tns._site_tag_id,
                x_tag_id=tns._x_tag_id,
                y_tag_id=tns._y_tag_id,
                Lx=tns._Lx,
                Ly=tns._Ly,
                site_ind_id=tns._site_ind_id,
            )
            if self.chi > 0:
                if len(amp_reuse.tensors) > 2 * self.Lx:
                    has_raw = (
                        bMPS_keys[0] in self._raw_bMPS_y_keys
                        or bMPS_keys[1]
                        in self._raw_bMPS_y_keys
                    )
                    if not has_raw:
                        amp_reuse.contract_boundary_from_ymin_(
                            max_bond=self.chi, cutoff=0.0,
                            yrange=[
                                bMPS_keys[0][1],
                                min(
                                    bMPS_keys[0][1] + 1,
                                    self.Ly - 1,
                                ),
                            ],
                            **self.contract_boundary_opts,
                        )
            return amp_reuse.contract()

        # Full contraction fallback
        if self.chi > 0:
            amp.contract_boundary_from_xmin_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[0, amp.Lx // 2 - 1],
                **self.contract_boundary_opts,
            )
            amp.contract_boundary_from_xmax_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[amp.Lx // 2, amp.Lx - 1],
                **self.contract_boundary_opts,
            )
        return amp.contract()

    # ----- vamp overrides -----

    def vamp(self, x, params):
        """Batched full-contraction amplitude with pytree
        unflatten."""
        import quimb as qu

        if isinstance(params, nn.ParameterList):
            params = list(params)
        params = qu.utils.tree_unflatten(
            params, self.params_pytree,
        )
        return self._vmapped_amplitude(x, params)

    def _get_cached_vamp_reuse(self, cache_key, in_dims):
        """Get or create a cached vmapped amplitude_reuse fn."""
        if cache_key not in self._vamp_reuse_cache:
            self._vamp_reuse_cache[cache_key] = torch.vmap(
                self.amplitude_reuse, in_dims=in_dims,
            )
        return self._vamp_reuse_cache[cache_key]

    def vamp_reuse(
        self,
        x,
        params,
        bMPS_keys=None,
        bMPS_params_xmin=None,
        bMPS_params_xmax=None,
        bMPS_params_ymin=None,
        bMPS_params_ymax=None,
        selected_rows=None,
        selected_cols=None,
    ):
        """Batched amplitude with bMPS environment reuse."""
        import quimb as qu

        if isinstance(params, nn.ParameterList):
            params = list(params)
        params = qu.utils.tree_unflatten(
            params, self.params_pytree,
        )

        if (bMPS_params_xmin is not None
                and bMPS_params_xmax is not None):
            cache_key = ('x', bMPS_keys[0], bMPS_keys[1])
            in_dims = (
                0, None, None,
                self.bMPS_params_x_in_dims[
                    bMPS_keys[0]
                ],
                self.bMPS_params_x_in_dims[
                    bMPS_keys[1]
                ],
                None, None, None, None,
            )
            fn = self._get_cached_vamp_reuse(
                cache_key, in_dims,
            )
            return fn(
                x, params, bMPS_keys,
                bMPS_params_xmin, bMPS_params_xmax,
                bMPS_params_ymin, bMPS_params_ymax,
                selected_rows, selected_cols,
            )

        if (bMPS_params_ymin is not None
                and bMPS_params_ymax is not None):
            cache_key = ('y', bMPS_keys[0], bMPS_keys[1])
            in_dims = (
                0, None, None, None, None,
                self.bMPS_params_y_in_dims[
                    bMPS_keys[0]
                ],
                self.bMPS_params_y_in_dims[
                    bMPS_keys[1]
                ],
                None, None,
            )
            fn = self._get_cached_vamp_reuse(
                cache_key, in_dims,
            )
            return fn(
                x, params, bMPS_keys,
                bMPS_params_xmin, bMPS_params_xmax,
                bMPS_params_ymin, bMPS_params_ymax,
                selected_rows, selected_cols,
            )

        # No reuse — full contraction via vmap
        cache_key = ('full', None, None)
        in_dims = (
            0, None, None,
            None, None, None, None, None, None,
        )
        fn = self._get_cached_vamp_reuse(
            cache_key, in_dims,
        )
        return fn(
            x, params, bMPS_keys,
            bMPS_params_xmin, bMPS_params_xmax,
            bMPS_params_ymin, bMPS_params_ymax,
            selected_rows, selected_cols,
        )

    # ----- Export override for pytree unflatten -----

    def _amplitude_for_export(self, x, *flat_params):
        """Wrapper for torch.export."""
        import quimb as qu

        p = qu.utils.tree_unflatten(
            list(flat_params), self.params_pytree,
        )
        return self.amplitude(x, p)

    # ----- forward -----

    def forward(self, x):
        """Compiled full-contraction forward (no reuse)."""
        if self._exported:
            params_list = list(self.params)
            if self._compiled:
                return self._vmapped_compiled(
                    x, *params_list,
                )
            else:
                return self._vmapped_exported(
                    x, *params_list,
                )
        return self.vamp(x, self.params)

    def forward_reuse(
        self,
        x,
        bMPS_params_x_batched=None,
        bMPS_params_y_batched=None,
        selected_rows=None,
        selected_cols=None,
    ):
        """Reuse forward path (eager vmap)."""
        bMPS_params_xmin = None
        bMPS_params_xmax = None
        bMPS_params_ymin = None
        bMPS_params_ymax = None
        bMPS_keys = None

        if selected_rows is not None:
            bMPS_keys = [
                ('xmin', min(selected_rows)),
                ('xmax', max(selected_rows)),
            ]
            bMPS_params_xmin = bMPS_params_x_batched[
                bMPS_keys[0]
            ]
            bMPS_params_xmax = bMPS_params_x_batched[
                bMPS_keys[1]
            ]
        if selected_cols is not None:
            bMPS_keys = [
                ('ymin', min(selected_cols)),
                ('ymax', max(selected_cols)),
            ]
            bMPS_params_ymin = bMPS_params_y_batched[
                bMPS_keys[0]
            ]
            bMPS_params_ymax = bMPS_params_y_batched[
                bMPS_keys[1]
            ]

        return self.vamp_reuse(
            x, self.params,
            bMPS_keys=bMPS_keys,
            bMPS_params_xmin=bMPS_params_xmin,
            bMPS_params_xmax=bMPS_params_xmax,
            bMPS_params_ymin=bMPS_params_ymin,
            bMPS_params_ymax=bMPS_params_ymax,
            selected_rows=selected_rows,
            selected_cols=selected_cols,
        )

    # ----- bMPS caching -----

    @torch.no_grad()
    def cache_bMPS_skeleton(self, x):
        """One-time init: compute bMPS skeletons and in_dims.

        Must be called before any reuse operations.

        Args:
            x: (N_sites,) int64 — a single example config
        """
        import quimb as qu
        import quimb.tensor as qtn

        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree,
        )
        tns = qtn.unpack(params, self.skeleton)
        amp = tns.isel({
            tns.site_ind(site): x[i]
            for i, site in enumerate(tns.sites)
        })

        # x-direction environments
        env_x = amp.compute_x_environments(
            max_bond=self.chi, cutoff=0.0,
            **self.contract_boundary_opts,
        )
        bMPS_params_dict = {}
        for key, tn_env in env_x.items():
            flat_params, skeleton_info = pack_tn(tn_env)
            env_x[key] = skeleton_info
            bMPS_params_dict[key] = flat_params
        self.bMPS_x_skeletons = env_x
        self.bMPS_params_x_in_dims = qu.utils.tree_map(
            lambda _: 0, bMPS_params_dict,
        )

        # Detect raw (single-row, D-bonded) x-envs:
        # ('xmin', 1) = row 0 only, ('xmax', Lx-2) = last row
        # only. These have D bonds, not chi bonds. Skip boundary
        # contraction for them to avoid slow small SVDs under vmap.
        self._raw_bMPS_x_keys = set()
        if self.Lx >= 3:
            self._raw_bMPS_x_keys.add(('xmin', 1))
            self._raw_bMPS_x_keys.add(('xmax', self.Lx - 2))

        # y-direction environments
        env_y = amp.compute_y_environments(
            max_bond=self.chi, cutoff=0.0,
            **self.contract_boundary_opts,
        )
        bMPS_params_dict = {}
        for key, tn_env in env_y.items():
            flat_params, skeleton_info = pack_tn(tn_env)
            env_y[key] = skeleton_info
            bMPS_params_dict[key] = flat_params
        self.bMPS_y_skeletons = env_y
        self.bMPS_params_y_in_dims = qu.utils.tree_map(
            lambda _: 0, bMPS_params_dict,
        )

        self._raw_bMPS_y_keys = set()
        if self.Ly >= 3:
            self._raw_bMPS_y_keys.add(('ymin', 1))
            self._raw_bMPS_y_keys.add(('ymax', self.Ly - 2))

    def cache_bMPS_params_vmap(self, x):
        """Compute batched bMPS params for all x and y envs.

        Args:
            x: (B, N_sites) int64 — batch of configurations

        Returns:
            (bMPS_x_dict, bMPS_y_dict) — batched pytrees
        """
        import quimb as qu
        import quimb.tensor as qtn

        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree,
        )

        def cache_single(x_single, params):
            tns = qtn.unpack(params, self.skeleton)
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            env_x = amp.compute_x_environments(
                max_bond=self.chi, cutoff=0.0,
                **self.contract_boundary_opts,
            )
            bMPS_x = {}
            for key, btn in env_x.items():
                bMPS_x[key] = get_params_tn(btn)
            env_y = amp.compute_y_environments(
                max_bond=self.chi, cutoff=0.0,
                **self.contract_boundary_opts,
            )
            bMPS_y = {}
            for key, btn in env_y.items():
                bMPS_y[key] = get_params_tn(btn)
            return bMPS_x, bMPS_y

        return torch.vmap(
            cache_single, in_dims=(0, None),
        )(x, params)

    def cache_bMPS_params_any_direction_vmap(
        self, x, direction='x',
    ):
        """Compute batched bMPS params for one direction
        + amplitudes.

        Args:
            x: (B, N_sites) int64
            direction: 'x' or 'y'

        Returns:
            (bMPS_params_dict, amp_vals)
        """
        import quimb as qu
        import quimb.tensor as qtn

        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree,
        )

        def cache_x_single(x_single, params):
            tns = qtn.unpack(params, self.skeleton)
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            env_x = amp.compute_x_environments(
                max_bond=self.chi, cutoff=0.0,
                **self.contract_boundary_opts,
            )
            amp_val = (
                env_x[('xmin', self.Lx // 2)]
                | env_x[('xmax', self.Lx // 2 - 1)]
            ).contract()
            bMPS_x = {}
            for key, btn in env_x.items():
                bMPS_x[key] = get_params_tn(btn)
            return bMPS_x, amp_val

        def cache_y_single(x_single, params):
            tns = qtn.unpack(params, self.skeleton)
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            env_y = amp.compute_y_environments(
                max_bond=self.chi, cutoff=0.0,
                **self.contract_boundary_opts,
            )
            amp_val = (
                env_y[('ymin', self.Ly // 2)]
                | env_y[('ymax', self.Ly // 2 - 1)]
            ).contract()
            bMPS_y = {}
            for key, btn in env_y.items():
                bMPS_y[key] = get_params_tn(btn)
            return bMPS_y, amp_val

        if direction == 'x':
            return torch.vmap(
                cache_x_single, in_dims=(0, None),
            )(x, params)
        else:
            return torch.vmap(
                cache_y_single, in_dims=(0, None),
            )(x, params)

    # ----- Incremental bMPS updates -----

    def update_bMPS_params_to_row_vmap(
        self, x, row_id, bMPS_params_x_batched,
        from_which='xmin',
    ):
        """Update batched bMPS x-params to a specific row."""
        import quimb as qu
        import quimb.tensor as qtn

        bMPS_key = (from_which, row_id)
        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree,
        )

        def update_single(
            x_single, params, row_id,
            bMPS_params_x, from_which,
        ):
            tns = qtn.unpack(params, self.skeleton)
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            bMPS_to_row = unpack_tn(
                bMPS_params_x[bMPS_key],
                self.bMPS_x_skeletons[bMPS_key],
            )
            row_tn = amp.select(
                [tns.row_tag(row_id)], which='any',
            )
            updated_bMPS = (bMPS_to_row | row_tn)

            if from_which == 'xmin':
                if row_id == 0:
                    updated_bMPS = row_tn
                else:
                    updated_bMPS.contract_boundary_from_xmin_(
                        max_bond=self.chi, cutoff=0.0,
                        xrange=[row_id - 1, row_id],
                        **self.contract_boundary_opts,
                    )
                updated_params = get_params_tn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(
                    updated_params, get_ref=True,
                )
                _, pytree = qu.utils.tree_flatten(
                    bMPS_params_x[
                        (from_which, row_id + 1)
                    ],
                    get_ref=True,
                )
                updated_params = qu.utils.tree_unflatten(
                    pytree_params, pytree,
                )
                bMPS_params_x[
                    (from_which, row_id + 1)
                ] = updated_params
            else:
                if row_id == amp.Ly - 1:
                    updated_bMPS = row_tn
                else:
                    updated_bMPS.contract_boundary_from_xmax_(
                        max_bond=self.chi, cutoff=0.0,
                        xrange=[row_id, row_id + 1],
                        **self.contract_boundary_opts,
                    )
                updated_params = get_params_tn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(
                    updated_params, get_ref=True,
                )
                _, pytree = qu.utils.tree_flatten(
                    bMPS_params_x[
                        (from_which, row_id - 1)
                    ],
                    get_ref=True,
                )
                updated_params = qu.utils.tree_unflatten(
                    pytree_params, pytree,
                )
                bMPS_params_x[
                    (from_which, row_id - 1)
                ] = updated_params
            return bMPS_params_x

        return torch.vmap(
            update_single,
            in_dims=(
                0, None, None,
                self.bMPS_params_x_in_dims, None,
            ),
        )(
            x, params, row_id,
            bMPS_params_x_batched, from_which,
        )

    def update_bMPS_params_to_col_vmap(
        self, x, col_id, bMPS_params_y_batched,
        from_which='ymin',
    ):
        """Update batched bMPS y-params to a specific col."""
        import quimb as qu
        import quimb.tensor as qtn

        bMPS_key = (from_which, col_id)
        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree,
        )

        def update_single(
            x_single, params, col_id,
            bMPS_params_y, from_which,
        ):
            tns = qtn.unpack(params, self.skeleton)
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            bMPS_to_col = unpack_tn(
                bMPS_params_y[bMPS_key],
                self.bMPS_y_skeletons[bMPS_key],
            )
            col_tn = amp.select(
                [tns.col_tag(col_id)], which='any',
            )
            updated_bMPS = (bMPS_to_col | col_tn)

            if from_which == 'ymin':
                if col_id == 0:
                    updated_bMPS = col_tn
                else:
                    updated_bMPS.contract_boundary_from_ymin_(
                        max_bond=self.chi, cutoff=0.0,
                        yrange=[col_id - 1, col_id],
                        **self.contract_boundary_opts,
                    )
                updated_params = get_params_tn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(
                    updated_params, get_ref=True,
                )
                _, pytree = qu.utils.tree_flatten(
                    bMPS_params_y[
                        (from_which, col_id + 1)
                    ],
                    get_ref=True,
                )
                updated_params = qu.utils.tree_unflatten(
                    pytree_params, pytree,
                )
                bMPS_params_y[
                    (from_which, col_id + 1)
                ] = updated_params
            else:
                if col_id == amp.Lx - 1:
                    updated_bMPS = col_tn
                else:
                    updated_bMPS.contract_boundary_from_ymax_(
                        max_bond=self.chi, cutoff=0.0,
                        yrange=[col_id, col_id + 1],
                        **self.contract_boundary_opts,
                    )
                updated_params = get_params_tn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(
                    updated_params, get_ref=True,
                )
                _, pytree = qu.utils.tree_flatten(
                    bMPS_params_y[
                        (from_which, col_id - 1)
                    ],
                    get_ref=True,
                )
                updated_params = qu.utils.tree_unflatten(
                    pytree_params, pytree,
                )
                bMPS_params_y[
                    (from_which, col_id - 1)
                ] = updated_params
            return bMPS_params_y

        return torch.vmap(
            update_single,
            in_dims=(
                0, None, None,
                self.bMPS_params_y_in_dims, None,
            ),
        )(
            x, params, col_id,
            bMPS_params_y_batched, from_which,
        )
