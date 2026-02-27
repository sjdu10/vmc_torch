"""Pure tensor-network models for GPU VMC.

fPEPS_Model_GPU        — full contraction with export+compile support
fPEPS_Model_reuse_GPU  — bMPS environment caching for incremental updates
"""
import torch
import torch.nn as nn

from ._base import WavefunctionModel_GPU

# Fermionic pack/unpack utilities
from vmc_torch.experiment.vmap.GPU.fermion_utils import (
    pack_ftn,
    unpack_ftn,
    get_params_ftn,
)


# =================================================================
#  fPEPS full-contraction model
# =================================================================


class fPEPS_Model_GPU(WavefunctionModel_GPU):
    """fPEPS model optimized for GPU with torch.export + compile.

    Pipeline: torch.export (trace) -> torch.vmap (batch) ->
              torch.compile (fuse CUDA kernels)

    Usage:
        model = fPEPS_Model_GPU(tn=peps, max_bond=chi, ...)
        model.to(device)
        model.export_and_compile(example_x)  # one-time setup
        out = model(fxs)  # fast batched forward
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
        self.chi = max_bond

        # Flatten pytree into a single list for torch
        params_flat, params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.params_pytree = params_pytree

        params_tensors = [
            torch.as_tensor(x, dtype=self.dtype) for x in params_flat
        ]

        super().__init__(params_list=params_tensors)

    def amplitude(self, x, params):
        """Single-sample amplitude via quimb TN contraction.

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

    def vamp(self, x, params):
        """Batched amplitude with quimb pytree unflatten.

        Override: unflatten ParameterList -> quimb pytree, then
        call _vmapped_amplitude (vmap over single-sample TN contraction).
        """
        import quimb as qu

        if isinstance(params, nn.ParameterList):
            params = list(params)
        params = qu.utils.tree_unflatten(params, self.params_pytree)
        return self._vmapped_amplitude(x, params)

    def _amplitude_for_export(self, x, *flat_params):
        """Wrapper for torch.export: unflatten then TN contraction."""
        import quimb as qu

        p = qu.utils.tree_unflatten(
            list(flat_params), self.params_pytree
        )
        return self.amplitude(x, p)


# =================================================================
#  fPEPS with bMPS environment reuse
# =================================================================


class fPEPS_Model_reuse_GPU(WavefunctionModel_GPU):
    """fPEPS model with bMPS environment caching and reuse.

    Caches boundary MPS environments along x and y directions.
    Supports incremental updates when only a few rows/cols change.

    Full contraction path (forward / vamp) goes through the base
    class export+compile machinery.  Reuse path is eager vmap.

    Ported from vmap/models/pureTNS.py::fPEPS_Model_reuse.
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

        # bMPS skeleton/in_dims dicts — populated by cache_bMPS_skeleton
        self.bMPS_x_skeletons = {}
        self.bMPS_y_skeletons = {}
        self.bMPS_params_x_in_dims = None
        self.bMPS_params_y_in_dims = None

        # Flatten pytree into a single list for torch
        params_flat, params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.params_pytree = params_pytree

        params_tensors = [
            torch.as_tensor(x, dtype=self.dtype) for x in params_flat
        ]

        self.radius = 0

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

        When bMPS_params_xmin/xmax or ymin/ymax are provided, uses
        cached boundary MPS instead of full contraction.  Falls back
        to full contraction when no caches are given.

        Args:
            x:      (N_sites,) int64 — one configuration
            params: quimb pytree of parameter tensors
            bMPS_keys: list of 2 tuples, e.g.
                [('xmin', row_min), ('xmax', row_max)]
            bMPS_params_xmin/xmax: pytree of cached x-boundary params
            bMPS_params_ymin/ymax: pytree of cached y-boundary params
            selected_rows/cols: list of row/col indices to contract

        Returns:
            scalar amplitude
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
            bMPS_min = unpack_ftn(
                bMPS_params_xmin,
                self.bMPS_x_skeletons[bMPS_keys[0]],
            )
            bMPS_max = unpack_ftn(
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
                    amp_reuse.contract_boundary_from_xmin_(
                        max_bond=self.chi, cutoff=0.0,
                        xrange=[
                            bMPS_keys[0][1],
                            min(bMPS_keys[0][1] + 1, self.Lx - 1),
                        ],
                        **self.contract_boundary_opts,
                    )
            return amp_reuse.contract()

        # y-environment reuse
        if (bMPS_params_ymin is not None
                and bMPS_params_ymax is not None
                and bMPS_keys is not None):
            bMPS_min = unpack_ftn(
                bMPS_params_ymin,
                self.bMPS_y_skeletons[bMPS_keys[0]],
            )
            bMPS_max = unpack_ftn(
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
                    amp_reuse.contract_boundary_from_ymin_(
                        max_bond=self.chi, cutoff=0.0,
                        yrange=[
                            bMPS_keys[0][1],
                            min(bMPS_keys[0][1] + 1, self.Ly - 1),
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
        """Batched full-contraction amplitude with pytree unflatten."""
        import quimb as qu

        if isinstance(params, nn.ParameterList):
            params = list(params)
        params = qu.utils.tree_unflatten(params, self.params_pytree)
        return self._vmapped_amplitude(x, params)

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
        """Batched amplitude with bMPS environment reuse.

        Uses torch.vmap over amplitude_reuse with appropriate in_dims
        for the batched bMPS parameter pytrees.
        """
        import quimb as qu

        if isinstance(params, nn.ParameterList):
            params = list(params)
        params = qu.utils.tree_unflatten(params, self.params_pytree)

        if (bMPS_params_xmin is not None
                and bMPS_params_xmax is not None):
            return torch.vmap(
                self.amplitude_reuse,
                in_dims=(
                    0,
                    None,
                    None,
                    self.bMPS_params_x_in_dims[bMPS_keys[0]],
                    self.bMPS_params_x_in_dims[bMPS_keys[1]],
                    None,
                    None,
                    None,
                    None,
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
                self.amplitude_reuse,
                in_dims=(
                    0,
                    None,
                    None,
                    None,
                    None,
                    self.bMPS_params_y_in_dims[bMPS_keys[0]],
                    self.bMPS_params_y_in_dims[bMPS_keys[1]],
                    None,
                    None,
                ),
            )(
                x, params, bMPS_keys,
                bMPS_params_xmin, bMPS_params_xmax,
                bMPS_params_ymin, bMPS_params_ymax,
                selected_rows, selected_cols,
            )

        # No reuse — full contraction via vmap
        return torch.vmap(
            self.amplitude_reuse,
            in_dims=(
                0, None, None, None, None, None, None, None, None,
            ),
        )(
            x, params, bMPS_keys,
            bMPS_params_xmin, bMPS_params_xmax,
            bMPS_params_ymin, bMPS_params_ymax,
            selected_rows, selected_cols,
        )

    # ----- Export override for pytree unflatten -----

    def _amplitude_for_export(self, x, *flat_params):
        """Wrapper for torch.export: unflatten then TN contraction."""
        import quimb as qu

        p = qu.utils.tree_unflatten(
            list(flat_params), self.params_pytree
        )
        return self.amplitude(x, p)

    # ----- forward -----

    def forward(self, x):
        """Compiled full-contraction forward (no reuse)."""
        if self._exported:
            params_list = list(self.params)
            if self._compiled:
                return self._vmapped_compiled(x, *params_list)
            else:
                return self._vmapped_exported(x, *params_list)
        return self.vamp(x, self.params)

    def forward_reuse(
        self,
        x,
        bMPS_params_x_batched=None,
        bMPS_params_y_batched=None,
        selected_rows=None,
        selected_cols=None,
    ):
        """Eager reuse forward path."""
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
            bMPS_params_xmin = bMPS_params_x_batched[bMPS_keys[0]]
            bMPS_params_xmax = bMPS_params_x_batched[bMPS_keys[1]]
        if selected_cols is not None:
            bMPS_keys = [
                ('ymin', min(selected_cols)),
                ('ymax', max(selected_cols)),
            ]
            bMPS_params_ymin = bMPS_params_y_batched[bMPS_keys[0]]
            bMPS_params_ymax = bMPS_params_y_batched[bMPS_keys[1]]

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
        """One-time init: compute bMPS skeletons and in_dims specs.

        Must be called before any reuse operations.

        Args:
            x: (N_sites,) int64 — a single example configuration
        """
        import quimb as qu
        import quimb.tensor as qtn

        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree
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
        for key, tn in env_x.items():
            bMPS_params, skeleton = pack_ftn(tn)
            env_x[key] = skeleton
            bMPS_params_dict[key] = bMPS_params
        self.bMPS_x_skeletons = env_x
        self.bMPS_params_x_in_dims = qu.utils.tree_map(
            lambda _: 0, bMPS_params_dict
        )

        # y-direction environments
        env_y = amp.compute_y_environments(
            max_bond=self.chi, cutoff=0.0,
        )
        bMPS_params_dict = {}
        for key, tn in env_y.items():
            bMPS_params, skeleton = pack_ftn(tn)
            env_y[key] = skeleton
            bMPS_params_dict[key] = bMPS_params
        self.bMPS_y_skeletons = env_y
        self.bMPS_params_y_in_dims = qu.utils.tree_map(
            lambda _: 0, bMPS_params_dict
        )

    def cache_bMPS_params_vmap(self, x):
        """Compute batched bMPS params for all x and y environments.

        Args:
            x: (B, N_sites) int64 — batch of configurations

        Returns:
            (bMPS_params_x_dict, bMPS_params_y_dict) — batched pytrees
        """
        import quimb as qu
        import quimb.tensor as qtn

        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree
        )

        def cache_bMPS_params_single(x_single, params):
            tns = qtn.unpack(params, self.skeleton)
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            env_x = amp.compute_x_environments(
                max_bond=self.chi, cutoff=0.0,
            )
            bMPS_params_x_dict = {}
            for key, btn in env_x.items():
                bMPS_params_x_dict[key] = get_params_ftn(btn)
            env_y = amp.compute_y_environments(
                max_bond=self.chi, cutoff=0.0,
            )
            bMPS_params_y_dict = {}
            for key, btn in env_y.items():
                bMPS_params_y_dict[key] = get_params_ftn(btn)
            return bMPS_params_x_dict, bMPS_params_y_dict

        return torch.vmap(
            cache_bMPS_params_single,
            in_dims=(0, None),
        )(x, params)

    def cache_bMPS_params_any_direction_vmap(
        self, x, direction='x',
    ):
        """Compute batched bMPS params for one direction + amplitudes.

        Args:
            x: (B, N_sites) int64 — batch of configurations
            direction: 'x' or 'y'

        Returns:
            (bMPS_params_dict, amp_vals) — batched pytree + (B,) amps
        """
        import quimb as qu
        import quimb.tensor as qtn

        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree
        )

        def cache_bMPS_params_x_single(x_single, params):
            tns = qtn.unpack(params, self.skeleton)
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            env_x = amp.compute_x_environments(
                max_bond=self.chi, cutoff=0.0,
            )
            amp_val = (
                env_x[('xmin', self.Lx // 2)]
                | env_x[('xmax', self.Lx // 2 - 1)]
            ).contract()
            bMPS_params_x_dict = {}
            for key, btn in env_x.items():
                bMPS_params_x_dict[key] = get_params_ftn(btn)
            return bMPS_params_x_dict, amp_val

        def cache_bMPS_params_y_single(x_single, params):
            tns = qtn.unpack(params, self.skeleton)
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            env_y = amp.compute_y_environments(
                max_bond=self.chi, cutoff=0.0,
            )
            amp_val = (
                env_y[('ymin', self.Ly // 2)]
                | env_y[('ymax', self.Ly // 2 - 1)]
            ).contract()
            bMPS_params_y_dict = {}
            for key, btn in env_y.items():
                bMPS_params_y_dict[key] = get_params_ftn(btn)
            return bMPS_params_y_dict, amp_val

        if direction == 'x':
            return torch.vmap(
                cache_bMPS_params_x_single,
                in_dims=(0, None),
            )(x, params)
        else:
            return torch.vmap(
                cache_bMPS_params_y_single,
                in_dims=(0, None),
            )(x, params)

    # ----- Incremental bMPS updates -----

    def update_bMPS_params_to_row_vmap(
        self, x, row_id, bMPS_params_x_batched, from_which='xmin',
    ):
        """Update batched bMPS x-params to a specific row.

        Args:
            x: (B, N_sites) int64
            row_id: int — target row
            bMPS_params_x_batched: batched pytree of x-env params
            from_which: 'xmin' or 'xmax'

        Returns:
            updated bMPS_params_x_batched
        """
        import quimb as qu
        import quimb.tensor as qtn

        bMPS_key = (from_which, row_id)
        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree
        )

        def update_bMPS_params_x_single(
            x_single, params, row_id, bMPS_params_x, from_which,
        ):
            tns = qtn.unpack(params, self.skeleton)
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            bMPS_to_row = unpack_ftn(
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
                updated_bMPS_params = get_params_ftn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(
                    updated_bMPS_params, get_ref=True,
                )
                _, pytree = qu.utils.tree_flatten(
                    bMPS_params_x[(from_which, row_id + 1)],
                    get_ref=True,
                )
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree,
                )
                bMPS_params_x[(from_which, row_id + 1)] = (
                    updated_bMPS_params
                )
            else:
                if row_id == amp.Ly - 1:
                    updated_bMPS = row_tn
                else:
                    updated_bMPS.contract_boundary_from_xmax_(
                        max_bond=self.chi, cutoff=0.0,
                        xrange=[row_id, row_id + 1],
                        **self.contract_boundary_opts,
                    )
                updated_bMPS_params = get_params_ftn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(
                    updated_bMPS_params, get_ref=True,
                )
                _, pytree = qu.utils.tree_flatten(
                    bMPS_params_x[(from_which, row_id - 1)],
                    get_ref=True,
                )
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree,
                )
                bMPS_params_x[(from_which, row_id - 1)] = (
                    updated_bMPS_params
                )
            return bMPS_params_x

        return torch.vmap(
            update_bMPS_params_x_single,
            in_dims=(
                0, None, None,
                self.bMPS_params_x_in_dims, None,
            ),
        )(x, params, row_id, bMPS_params_x_batched, from_which)

    def update_bMPS_params_to_col_vmap(
        self, x, col_id, bMPS_params_y_batched, from_which='ymin',
    ):
        """Update batched bMPS y-params to a specific column.

        Args:
            x: (B, N_sites) int64
            col_id: int — target column
            bMPS_params_y_batched: batched pytree of y-env params
            from_which: 'ymin' or 'ymax'

        Returns:
            updated bMPS_params_y_batched
        """
        import quimb as qu
        import quimb.tensor as qtn

        bMPS_key = (from_which, col_id)
        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree
        )

        def update_bMPS_params_y_single(
            x_single, params, col_id, bMPS_params_y, from_which,
        ):
            tns = qtn.unpack(params, self.skeleton)
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            bMPS_to_col = unpack_ftn(
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
                updated_bMPS_params = get_params_ftn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(
                    updated_bMPS_params, get_ref=True,
                )
                _, pytree = qu.utils.tree_flatten(
                    bMPS_params_y[(from_which, col_id + 1)],
                    get_ref=True,
                )
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree,
                )
                bMPS_params_y[(from_which, col_id + 1)] = (
                    updated_bMPS_params
                )
            else:
                if col_id == amp.Lx - 1:
                    updated_bMPS = col_tn
                else:
                    updated_bMPS.contract_boundary_from_ymax_(
                        max_bond=self.chi, cutoff=0.0,
                        yrange=[col_id, col_id + 1],
                        **self.contract_boundary_opts,
                    )
                updated_bMPS_params = get_params_ftn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(
                    updated_bMPS_params, get_ref=True,
                )
                _, pytree = qu.utils.tree_flatten(
                    bMPS_params_y[(from_which, col_id - 1)],
                    get_ref=True,
                )
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree,
                )
                bMPS_params_y[(from_which, col_id - 1)] = (
                    updated_bMPS_params
                )
            return bMPS_params_y

        return torch.vmap(
            update_bMPS_params_y_single,
            in_dims=(
                0, None, None,
                self.bMPS_params_y_in_dims, None,
            ),
        )(x, params, col_id, bMPS_params_y_batched, from_which)

    # ----- Debug helper -----

    def amp_tn(self, x):
        """Return the raw TN for a single config (debug helper)."""
        import quimb as qu
        import quimb.tensor as qtn

        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree
        )
        tns = qtn.unpack(params, self.skeleton)
        amp = tns.isel({
            tns.site_ind(site): x[i]
            for i, site in enumerate(tns.sites)
        })
        return amp
