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

    # ----- Param preprocessing -----

    def _vamp_params_preprocess(self, params):
        """Unflatten ParameterList -> quimb pytree."""
        import quimb as qu

        if isinstance(params, nn.ParameterList):
            params = list(params)
        params = qu.utils.tree_unflatten(
            params, self.params_pytree,
        )
        return params

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

    def log_amplitude(self, x, params):
        """Single-sample log-amplitude via strip_exponent.

        Args:
            x:      (N_sites,) int64 — one configuration
            params: quimb pytree of parameter tensors

        Returns:
            (sign, log_abs) scalars
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
        sign, exponent_10 = amp.contract(strip_exponent=True)
        log_abs = exponent_10 * torch.log(torch.tensor(10.0))
        return sign, log_abs

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

    def amplitude_reuse_log(
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
        """Single-sample log-amplitude with cached bMPS environments.

        Same as amplitude_reuse but uses strip_exponent for
        numerically stable log-amplitude.

        Returns:
            (sign, log_abs) scalars
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
                    has_raw = (
                        bMPS_keys[0]
                        in self._raw_bMPS_x_keys
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
            sign, exp10 = amp_reuse.contract(
                strip_exponent=True,
            )
            log_abs = exp10 * torch.log(torch.tensor(10.0))
            return sign, log_abs

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
                        bMPS_keys[0]
                        in self._raw_bMPS_y_keys
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
            sign, exp10 = amp_reuse.contract(
                strip_exponent=True,
            )
            log_abs = exp10 * torch.log(torch.tensor(10.0))
            return sign, log_abs

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
        sign, exp10 = amp.contract(strip_exponent=True)
        log_abs = exp10 * torch.log(torch.tensor(10.0))
        return sign, log_abs

    # ----- vamp overrides -----

    def vamp(self, x, params):
        """Batched full-contraction amplitude with pytree
        unflatten."""
        params = self._vamp_params_preprocess(params)
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
        params = self._vamp_params_preprocess(params)

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

    def vamp_reuse_log(
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
        """Batched log-amplitude with bMPS environment reuse.

        Uses torch.vmap over amplitude_reuse_log with appropriate
        in_dims for the batched bMPS parameter pytrees.

        Returns:
            (signs, log_abs): each (B,)
        """
        params = self._vamp_params_preprocess(params)

        if (bMPS_params_xmin is not None
                and bMPS_params_xmax is not None):
            return torch.vmap(
                self.amplitude_reuse_log,
                in_dims=(
                    0, None, None,
                    self.bMPS_params_x_in_dims[
                        bMPS_keys[0]
                    ],
                    self.bMPS_params_x_in_dims[
                        bMPS_keys[1]
                    ],
                    None, None, None, None,
                ),
            )(
                x, params, bMPS_keys,
                bMPS_params_xmin, bMPS_params_xmax,
                bMPS_params_ymin, bMPS_params_ymax,
                selected_rows, selected_cols,
            )

        if (bMPS_params_ymin is not None
                and bMPS_params_ymax is not None):
            return torch.vmap(
                self.amplitude_reuse_log,
                in_dims=(
                    0, None, None, None, None,
                    self.bMPS_params_y_in_dims[
                        bMPS_keys[0]
                    ],
                    self.bMPS_params_y_in_dims[
                        bMPS_keys[1]
                    ],
                    None, None,
                ),
            )(
                x, params, bMPS_keys,
                bMPS_params_xmin, bMPS_params_xmax,
                bMPS_params_ymin, bMPS_params_ymax,
                selected_rows, selected_cols,
            )

        # No reuse — full contraction via vmap
        return torch.vmap(
            self.amplitude_reuse_log,
            in_dims=(
                0, None, None,
                None, None, None, None, None, None,
            ),
        )(
            x, params, bMPS_keys,
            bMPS_params_xmin, bMPS_params_xmax,
            bMPS_params_ymin, bMPS_params_ymax,
            selected_rows, selected_cols,
        )

    # ----- Export + compile for reuse patterns -----

    def export_and_compile_reuse(
        self, example_x, mode='default', verbose=True,
        use_log_amp=False,
    ):
        """Pre-export amplitude_reuse for all reuse patterns.

        For a nearest-neighbor Hamiltonian on Lx x Ly lattice,
        the possible reuse patterns are:
          - x-dir, single row r: selected_rows=[r]   (Lx)
          - x-dir, two rows r,r+1: selected_rows=[r,r+1] (Lx-1)
          - y-dir, single col c: selected_cols=[c]   (Ly)
          - y-dir, two cols c,c+1: selected_cols=[c,c+1] (Ly-1)
        Total: 2*(Lx + Ly) - 2 patterns.

        Args:
            example_x: (N_sites,) int64 on the target device.
            mode: torch.compile mode.
            verbose: Print progress.
            use_log_amp: if True, export amplitude_reuse_log.
        """
        import quimb as qu
        from torch.export import export

        self._compiled_reuse = {}
        self._compiled_reuse_log_amp = use_log_amp
        device = example_x.device

        # Compute example bMPS params for slicing
        x_batch = example_x.unsqueeze(0)  # (1, N_sites)
        bMPS_x, bMPS_y = self.cache_bMPS_params_vmap(x_batch)

        params_list = list(self.params)
        n_tn_params = len(params_list)

        patterns = []
        for direction in ('x', 'y'):
            L = self.Lx if direction == 'x' else self.Ly
            for width in (1, 2):
                for start in range(L - width + 1):
                    indices = tuple(
                        range(start, start + width)
                    )
                    patterns.append((direction, indices))

        if verbose:
            print(
                f"Exporting {len(patterns)} reuse patterns "
                f"for {self.Lx}x{self.Ly} lattice..."
            )

        for pat_idx, (direction, indices) in enumerate(
            patterns
        ):
            if direction == 'x':
                bMPS_keys = [
                    ('xmin', min(indices)),
                    ('xmax', max(indices)),
                ]
                bMPS_min_batched = bMPS_x[bMPS_keys[0]]
                bMPS_max_batched = bMPS_x[bMPS_keys[1]]
                in_dims_min = (
                    self.bMPS_params_x_in_dims[bMPS_keys[0]]
                )
                in_dims_max = (
                    self.bMPS_params_x_in_dims[bMPS_keys[1]]
                )
            else:
                bMPS_keys = [
                    ('ymin', min(indices)),
                    ('ymax', max(indices)),
                ]
                bMPS_min_batched = bMPS_y[bMPS_keys[0]]
                bMPS_max_batched = bMPS_y[bMPS_keys[1]]
                in_dims_min = (
                    self.bMPS_params_y_in_dims[bMPS_keys[0]]
                )
                in_dims_max = (
                    self.bMPS_params_y_in_dims[bMPS_keys[1]]
                )

            # Extract single-sample bMPS params for export
            bMPS_min_single = qu.utils.tree_map(
                lambda t: t[0], bMPS_min_batched,
            )
            bMPS_max_single = qu.utils.tree_map(
                lambda t: t[0], bMPS_max_batched,
            )

            # Flatten bMPS params for export
            bMPS_min_flat, bMPS_min_pytree = (
                qu.utils.tree_flatten(
                    bMPS_min_single, get_ref=True,
                )
            )
            bMPS_max_flat, bMPS_max_pytree = (
                qu.utils.tree_flatten(
                    bMPS_max_single, get_ref=True,
                )
            )

            n_min = len(bMPS_min_flat)
            n_max = len(bMPS_max_flat)

            selected_rows = (
                list(indices) if direction == 'x' else None
            )
            selected_cols = (
                list(indices) if direction == 'y' else None
            )
            bMPS_keys_frozen = list(bMPS_keys)

            def make_wrapper(
                _bMPS_keys, _bMPS_min_pytree,
                _bMPS_max_pytree,
                _n_tn, _n_min, _selected_rows,
                _selected_cols, _direction, _use_log_amp,
            ):
                if _use_log_amp:
                    reuse_fn = self.amplitude_reuse_log
                else:
                    reuse_fn = self.amplitude_reuse

                def wrapper(x, *flat_args):
                    tn_params = list(flat_args[:_n_tn])
                    min_params = list(
                        flat_args[_n_tn:_n_tn + _n_min]
                    )
                    max_params = list(
                        flat_args[_n_tn + _n_min:]
                    )
                    bMPS_min = qu.utils.tree_unflatten(
                        min_params, _bMPS_min_pytree,
                    )
                    bMPS_max = qu.utils.tree_unflatten(
                        max_params, _bMPS_max_pytree,
                    )
                    if _direction == 'x':
                        return reuse_fn(
                            x,
                            qu.utils.tree_unflatten(
                                tn_params,
                                self.params_pytree,
                            ),
                            bMPS_keys=_bMPS_keys,
                            bMPS_params_xmin=bMPS_min,
                            bMPS_params_xmax=bMPS_max,
                            selected_rows=_selected_rows,
                        )
                    else:
                        return reuse_fn(
                            x,
                            qu.utils.tree_unflatten(
                                tn_params,
                                self.params_pytree,
                            ),
                            bMPS_keys=_bMPS_keys,
                            bMPS_params_ymin=bMPS_min,
                            bMPS_params_ymax=bMPS_max,
                            selected_cols=_selected_cols,
                        )
                return wrapper

            wrapper_fn = make_wrapper(
                bMPS_keys_frozen, bMPS_min_pytree,
                bMPS_max_pytree,
                n_tn_params, n_min,
                selected_rows, selected_cols, direction,
                use_log_amp,
            )

            # Export
            import torch.nn as nn_mod

            class _ReuseModule(nn_mod.Module):
                def __init__(self_, fn):
                    super().__init__()
                    self_._fn = fn

                def forward(self_, x, *flat_args):
                    return self_._fn(x, *flat_args)

            all_flat_args = (
                params_list + bMPS_min_flat + bMPS_max_flat
            )

            try:
                with torch.no_grad():
                    exported = export(
                        _ReuseModule(wrapper_fn),
                        (example_x, *all_flat_args),
                    )
                exported_module = exported.module()

                # Move CPU constants to GPU
                self._move_exported_constants_to_device(
                    device,
                    exported_module=exported_module,
                )

                # Flatten in_dims for bMPS params
                in_dims_min_flat, _ = (
                    qu.utils.tree_flatten(
                        in_dims_min, get_ref=True,
                    )
                )
                in_dims_max_flat, _ = (
                    qu.utils.tree_flatten(
                        in_dims_max, get_ref=True,
                    )
                )

                in_dims_tuple = (
                    (0,)  # x batched
                    + tuple([None] * n_tn_params)
                    + tuple(in_dims_min_flat)
                    + tuple(in_dims_max_flat)
                )

                vmapped = torch.vmap(
                    exported_module,
                    in_dims=in_dims_tuple,
                )
                compiled = torch.compile(
                    vmapped, mode=mode,
                )

                key = (direction, indices)
                self._compiled_reuse[key] = {
                    'fn': compiled,
                    'bMPS_min_pytree': bMPS_min_pytree,
                    'bMPS_max_pytree': bMPS_max_pytree,
                    'n_tn': n_tn_params,
                    'n_min': n_min,
                    'bMPS_keys': bMPS_keys_frozen,
                }

                if verbose:
                    print(
                        f"  [{pat_idx+1}/{len(patterns)}] "
                        f"({direction}, {indices}) OK"
                    )
            except Exception as e:
                if verbose:
                    print(
                        f"  [{pat_idx+1}/{len(patterns)}] "
                        f"({direction}, {indices}) FAILED: "
                        f"{e}"
                    )

        if verbose:
            print(
                f"Exported {len(self._compiled_reuse)}"
                f"/{len(patterns)} patterns"
            )

    def _move_exported_constants_to_device(
        self, device, exported_module=None,
    ):
        """Move CPU constants in exported graph to GPU.

        Extended to accept an external exported_module for
        reuse patterns.
        """
        if exported_module is None:
            super()._move_exported_constants_to_device(
                device
            )
            return

        gm = exported_module
        graph = gm.graph
        for node in graph.nodes:
            if node.op != 'get_attr':
                continue
            parts = node.target.split('.')
            parent = gm
            for p in parts[:-1]:
                parent = getattr(parent, p)
            leaf = parts[-1]
            tensor = getattr(parent, leaf)
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.device.type == 'cpu'
            ):
                setattr(parent, leaf, tensor.to(device))

        for node in graph.nodes:
            if node.op != 'call_function':
                continue
            if '_assert_tensor_metadata' not in str(
                node.target
            ):
                continue
            kw = dict(node.kwargs)
            if kw.get('device') == torch.device('cpu'):
                kw['device'] = device
                node.kwargs = kw

        graph.lint()
        gm.recompile()

    # ----- forward -----

    def forward(self, x):
        """Compiled full-contraction forward (no reuse)."""
        if self._exported and not self._exported_log_amp:
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
        """Reuse forward — compiled if available, else eager."""
        import quimb as qu

        # Determine the reuse key
        if selected_rows is not None:
            key = ('x', tuple(selected_rows))
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
        elif selected_cols is not None:
            key = ('y', tuple(selected_cols))
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
        else:
            key = None

        # Try compiled path (only for amplitude, not log)
        if (
            key is not None
            and hasattr(self, '_compiled_reuse')
            and key in self._compiled_reuse
            and not getattr(
                self, '_compiled_reuse_log_amp', False
            )
        ):
            entry = self._compiled_reuse[key]
            params_list = list(self.params)

            if selected_rows is not None:
                bMPS_min_flat, _ = qu.utils.tree_flatten(
                    bMPS_params_xmin, get_ref=True,
                )
                bMPS_max_flat, _ = qu.utils.tree_flatten(
                    bMPS_params_xmax, get_ref=True,
                )
            else:
                bMPS_min_flat, _ = qu.utils.tree_flatten(
                    bMPS_params_ymin, get_ref=True,
                )
                bMPS_max_flat, _ = qu.utils.tree_flatten(
                    bMPS_params_ymax, get_ref=True,
                )

            return entry['fn'](
                x,
                *params_list,
                *bMPS_min_flat,
                *bMPS_max_flat,
            )

        # Fallback to eager
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

    def forward_reuse_log(
        self,
        x,
        bMPS_params_x_batched=None,
        bMPS_params_y_batched=None,
        selected_rows=None,
        selected_cols=None,
    ):
        """Log-amplitude reuse — compiled if available."""
        import quimb as qu

        # Determine the reuse key
        if selected_rows is not None:
            key = ('x', tuple(selected_rows))
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
        elif selected_cols is not None:
            key = ('y', tuple(selected_cols))
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
        else:
            key = None

        # Try compiled path (only for log-amp)
        if (
            key is not None
            and hasattr(self, '_compiled_reuse')
            and key in self._compiled_reuse
            and getattr(
                self, '_compiled_reuse_log_amp', False
            )
        ):
            entry = self._compiled_reuse[key]
            params_list = list(self.params)

            if selected_rows is not None:
                bMPS_min_flat, _ = qu.utils.tree_flatten(
                    bMPS_params_xmin, get_ref=True,
                )
                bMPS_max_flat, _ = qu.utils.tree_flatten(
                    bMPS_params_xmax, get_ref=True,
                )
            else:
                bMPS_min_flat, _ = qu.utils.tree_flatten(
                    bMPS_params_ymin, get_ref=True,
                )
                bMPS_max_flat, _ = qu.utils.tree_flatten(
                    bMPS_params_ymax, get_ref=True,
                )

            return entry['fn'](
                x,
                *params_list,
                *bMPS_min_flat,
                *bMPS_max_flat,
            )

        # Fallback to eager
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

        return self.vamp_reuse_log(
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

    # ----- Cheap gradient via hole contraction -----

    def cheap_grad_single(self, x, params, bMPS_params_x):
        """Single-sample cheap gradient via per-row hole contraction.

        For pure TNS, psi(x) is linear in each row's tensors.
        The bMPS environments are treated as constants (no grad
        through SVDs). For each row, we differentiate the small
        contraction (bMPS_below | row_tensors | bMPS_above) w.r.t.
        the full params pytree. Only the current row's params get
        non-zero gradients.

        Requires chi >= 4*D for accurate boundary environments.

        Args:
            x: (N_sites,) int64 — one configuration
            params: quimb pytree of TN params
            bMPS_params_x: dict {('xmin', r): flat_params,
                ('xmax', r): flat_params} for all rows r

        Returns:
            (grad_flat, amp): gradient (Np,) and scalar amplitude
        """
        import quimb as qu
        import quimb.tensor as qtn

        # Compute amplitude from midpoint bMPS envs
        mid = self.Lx // 2
        bMPS_mid_below = unpack_tn(
            bMPS_params_x[('xmin', mid)],
            self.bMPS_x_skeletons[('xmin', mid)],
        )
        bMPS_mid_above = unpack_tn(
            bMPS_params_x[('xmax', mid - 1)],
            self.bMPS_x_skeletons[('xmax', mid - 1)],
        )
        amp_val = (bMPS_mid_below | bMPS_mid_above).contract()

        # Accumulate gradients row by row.
        # Use flatten/unflatten since qu.utils.tree_map
        # only maps over a single tree.
        accum_flat, accum_ref = qu.utils.tree_flatten(
            qu.utils.tree_map(torch.zeros_like, params),
            get_ref=True,
        )

        for row in range(self.Lx):
            bk_below = ('xmin', row)
            bk_above = ('xmax', row)
            bp_below = bMPS_params_x[bk_below]
            bp_above = bMPS_params_x[bk_above]
            sk_below = self.bMPS_x_skeletons[bk_below]
            sk_above = self.bMPS_x_skeletons[bk_above]

            # Capture loop vars via defaults
            def row_fn(
                p,
                _bp_below=bp_below,
                _bp_above=bp_above,
                _sk_below=sk_below,
                _sk_above=sk_above,
                _row=row,
            ):
                tns = qtn.unpack(p, self.skeleton)
                amp = tns.isel({
                    tns.site_ind(site): x[i]
                    for i, site in enumerate(tns.sites)
                })
                b_below = unpack_tn(
                    _bp_below, _sk_below,
                )
                b_above = unpack_tn(
                    _bp_above, _sk_above,
                )
                row_ts = amp.select(
                    tns.row_tag(_row), which='any',
                )
                amp_reuse = (
                    b_below | row_ts | b_above
                )
                return amp_reuse.contract()

            row_grad = torch.func.grad(row_fn)(params)
            row_grad_flat = qu.utils.tree_flatten(
                row_grad,
            )
            accum_flat = [
                a + g
                for a, g in zip(accum_flat, row_grad_flat)
            ]

        # Flatten to (Np,)
        grad_flat = torch.cat(
            [g.flatten() for g in accum_flat],
        )
        return grad_flat, amp_val

    def compute_cheap_grads_vmap(
        self, fxs, bMPS_params_x,
    ):
        """Batched cheap gradient via vmap over cheap_grad_single.

        Args:
            fxs: (B, N_sites) int64
            bMPS_params_x: batched dict from cache_bMPS_params_vmap

        Returns:
            (grads, amps): (B, Np) and (B,)
        """
        params = self._vamp_params_preprocess(self.params)
        return torch.vmap(
            self.cheap_grad_single,
            in_dims=(
                0, None, self.bMPS_params_x_in_dims,
            ),
        )(fxs, params, bMPS_params_x)
