"""Pure tensor-network models for GPU VMC.

fPEPS_Model_GPU        — full contraction with export+compile support
fPEPS_Model_reuse_GPU  — bMPS environment caching for incremental updates
"""
import torch
import torch.nn as nn

from ._base import WavefunctionModel_GPU

# Fermionic pack/unpack utilities
from vmc_torch.GPU.fermion_utils import (
    pack_ftn,
    unpack_ftn,
    get_params_ftn,
    strip_phys_linearmap,
)

# Fermionic physical-state encoding for direct sampling:
#   0=empty, 1=up, 2=down, 3=double.
# Per-state contribution to the up / down particle counts.
_FPEPS_DU = (0, 1, 0, 1)   # n_up contribution per state
_FPEPS_DD = (0, 0, 1, 1)   # n_dn contribution per state


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
        tn = tn.copy()  # don't modify original TN
        import quimb as qu
        import quimb.tensor as qtn

        if contract_boundary_opts is None:
            contract_boundary_opts = {}
        
        # config value k -> physical index perm[k]
        self._loc_basis_perm = strip_phys_linearmap(tn)

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
        # apply local basis permutation (no-op if _loc_basis_perm is None)
        if self._loc_basis_perm is not None:
            x = self._loc_basis_perm[x]
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
        """Single-sample log-amplitude via quimb TN contraction.

        Args:
            x:      (N_sites,) int64 — one configuration
            params: quimb pytree of parameter tensors

        Returns:
            scalar log-amplitude (mantissa, exponent)
        """
        import quimb.tensor as qtn

        tn = qtn.unpack(params, self.skeleton)
        if self._loc_basis_perm is not None:
            x = self._loc_basis_perm[x]
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
        exp = exponent_10*torch.log(torch.tensor(10.0))
        return sign, exp

    def _vamp_params_preprocess(self, params):
        """Preprocess params for vamp.  Default: unflatten ParameterList -> quimb pytree."""
        import quimb as qu

        if isinstance(params, nn.ParameterList):
            params = list(params)
        params = qu.utils.tree_unflatten(params, self.params_pytree)
        return params

    # -----------------------------------------------------------------
    #  Direct sampler (torch.vmap over walkers), chi_m=0, Z2.
    #  Mirrors the CPU fermionic sampler `fTNModel.direct_sample`
    #  (vmc_torch/model.py) but built on the flat fermionic
    #  representation so the whole thing is vmap-traceable.
    # -----------------------------------------------------------------

    def _phys_linear(self, v):
        """Map a semantic state v in {0,1,2,3} to the flat phys leg's
        linear index. The model nulls the phys `_linearmap` and records
        `_loc_basis_perm` (see __init__); the same perm `amplitude` uses.
        For a None perm (no remap needed) semantic == linear.
        """
        if self._loc_basis_perm is not None:
            return self._loc_basis_perm[v]
        return v

    @staticmethod
    def _site_label(tns, x, y):
        """Fermionic ordering label for site (x, y)'s physical leg.

        Flat fermionic `squeeze` turns a squeezed odd-parity phys leg into a
        dummy mode keyed by the tensor's `.label`; when several dummy modes
        accumulate (rows below the top boundary) they are SORTED by label,
        each reorder contributing a fermionic sign. So the label MUST sort in
        the peps's true fermionic site order, otherwise the signs are wrong.
        Use the original (uncompressed) site tensor's label; fall back to the
        site coordinate.
        """
        lbl = tns[tns.site_tag(x, y)].data.label
        return lbl if lbl is not None else (x, y)

    @staticmethod
    def _shared_ind(t_y, tn, y_tags, y, direction, Ly):
        """Bond index name shared between column tensor t_y and its
        neighbour. direction = -1 (left) / +1 (right). None at boundary.
        """
        if direction == -1 and y == 0:
            return None
        if direction == +1 and y == Ly - 1:
            return None
        nbr = tn[y_tags[y + direction]]
        shared = set(t_y.inds) & set(nbr.inds)
        return next(iter(shared)) if shared else None

    @staticmethod
    def _inverse_cdf_sample(probs, u, d):
        """Sample v in {0,...,d-1} from probs via inverse CDF with u."""
        cs = torch.cumsum(probs, dim=0)
        v = (cs <= u).sum()
        v = torch.clamp(v, max=d - 1)
        return v

    @staticmethod
    def _apply_u1u1_mask(probs, x, y, Ly, N, n_up_so_far, n_dn_so_far,
                         n_up_target, n_dn_target):
        """Branchless U(1)xU(1) charge mask for the d=4 phys states.

        Torch port of `_fpeps_u1u1_mask` (vmc_torch/model.py): zero out
        states that overshoot a particle target or cannot reach it with
        the remaining sites. `n_*_so_far` are traced int64 scalars;
        `n_*_target` are Python ints or None (static config args, so the
        `is None` branches are not data-dependent). Returns normalized
        probs (4,).
        """
        probs = torch.clamp(probs, min=0.0)
        if n_up_target is None and n_dn_target is None:
            return probs / (probs.sum() + 1e-300)
        site_idx = x * Ly + y                  # Python int
        n_remain_after = N - site_idx - 1      # Python int
        du = torch.tensor(_FPEPS_DU, dtype=torch.int64, device=probs.device)
        dd = torch.tensor(_FPEPS_DD, dtype=torch.int64, device=probs.device)
        ok = torch.ones(4, dtype=probs.dtype, device=probs.device)
        if n_up_target is not None:
            n_up_a = n_up_so_far + du          # (4,) traced
            ok = ok * (n_up_a <= n_up_target).to(probs.dtype)
            ok = ok * ((n_up_target - n_up_a) <= n_remain_after).to(probs.dtype)
        if n_dn_target is not None:
            n_dn_a = n_dn_so_far + dd
            ok = ok * (n_dn_a <= n_dn_target).to(probs.dtype)
            ok = ok * ((n_dn_target - n_dn_a) <= n_remain_after).to(probs.dtype)
        probs = probs * ok
        return probs / (probs.sum() + 1e-300)

    def _sample_row_single(self, combined_rc, tns, x, d, u_rand,
                           n_up_so_far, n_dn_so_far,
                           n_up_target, n_dn_target, N, forced=None):
        """Sample row x site-by-site (LTR) on the flat fermionic TN.

        chi_m=0: right-canonicalize then propagate a left density matrix
        rho_L. Mirrors the CPU `_fpeps_sample_row` but with torch ops so
        it is vmap-traceable. Returns
        (sampled_row [SEMANTIC int64 scalars], log_p_row,
         n_up_so_far, n_dn_so_far).
        """
        Ly = tns.Ly
        y_tags = [tns.y_tag(y) for y in range(Ly)]
        log_p_row = torch.zeros(
            (), dtype=u_rand.dtype, device=u_rand.device,
        )
        sampled_row = []
        du = torch.tensor(_FPEPS_DU, dtype=torch.int64, device=u_rand.device)
        dd = torch.tensor(_FPEPS_DD, dtype=torch.int64, device=u_rand.device)

        for y in range(Ly - 1, 0, -1):
            combined_rc.canonize_between(y_tags[y], y_tags[y - 1])
        rho_L = None

        for y in range(Ly):
            phys_ind = tns.site_ind(x, y)
            t_y = combined_rc[y_tags[y]]
            # Flat fermionic squeeze of an odd-parity phys leg needs an
            # ordering .label (compression drops it); it must sort in the
            # peps's true fermionic site order (see _site_label).
            t_y.data._label = self._site_label(tns, x, y)

            left_ind = self._shared_ind(t_y, combined_rc, y_tags, y, -1, Ly)
            right_ind = self._shared_ind(t_y, combined_rc, y_tags, y, +1, Ly)

            probs_list = []
            for v in range(d):
                tv = t_y.isel({phys_ind: self._phys_linear(v)})
                if rho_L is None:
                    p = (tv.H | tv).contract().real
                else:
                    tv_c = tv.conj().reindex({left_ind: left_ind + '_c'})
                    F_v = (tv | tv_c).contract(
                        output_inds=[left_ind, left_ind + '_c'],
                    )
                    p = (rho_L | F_v).contract().real
                probs_list.append(p)
            probs = torch.stack(probs_list)
            probs = self._apply_u1u1_mask(
                probs, x, y, Ly, N, n_up_so_far, n_dn_so_far,
                n_up_target, n_dn_target,
            )

            if forced is not None:
                v = forced[x * Ly + y]
            else:
                v = self._inverse_cdf_sample(
                    probs, u_rand[x * Ly + y], d,
                )
            sampled_row.append(v)
            log_p_row = log_p_row + torch.log(probs[v] + 1e-300)
            n_up_so_far = n_up_so_far + du[v]
            n_dn_so_far = n_dn_so_far + dd[v]

            # rho_L update for next site (uses torch-scalar v)
            if right_ind is not None:
                tv = t_y.isel({phys_ind: self._phys_linear(v)})
                if rho_L is None:
                    tv_c = tv.conj().reindex(
                        {right_ind: right_ind + '_c'},
                    )
                    rho_L = (tv | tv_c).contract(
                        output_inds=[right_ind, right_ind + '_c'],
                    )
                else:
                    tv_c = tv.conj().reindex({
                        left_ind: left_ind + '_c',
                        right_ind: right_ind + '_c',
                    })
                    rho_L = (rho_L | tv | tv_c).contract(
                        output_inds=[right_ind, right_ind + '_c'],
                    )
                # Normalize rho_L by the POSITIVE raw data norm (||blocks||).
                # NOT by the fermionic inner product (rho_L.H | rho_L): that
                # carries fermionic signs and CAN GO NEGATIVE -> rsqrt -> NaN
                # (observed for >=3 rows). The true trace would be positive
                # but routes to the unsupported einsum 'aa->'. Any positive
                # scale is fine here: it cancels in the per-site probs
                # renormalization and in log_p_c. Phases are +/-1 so the
                # block norm is phase-independent.
                scale = torch.linalg.vector_norm(rho_L.data.blocks)
                rho_L = rho_L / (scale + 1e-300)

        return sampled_row, log_p_row, n_up_so_far, n_dn_so_far

    def direct_sample_single(self, u_rand, params, chi_s,
                             n_up_target=None, n_dn_target=None,
                             forced=None):
        """One walker's fermionic direct sample. vmap-traceable.

        Args:
            u_rand:      (Lx*Ly,) torch float uniforms — pre-drawn.
            params:      quimb pytree of torch tensors (unflattened).
            chi_s:       Python int or None, boundary MPS bond.
            n_up_target: Python int or None, U(1) up-count constraint.
            n_dn_target: Python int or None, U(1) down-count constraint.
            forced:      (Lx*Ly,) int64 SEMANTIC config or None. If given,
                no sampling: site values are taken from ``forced`` and
                ``log_p_c`` is the direct-sampler log-probability of that
                config (``u_rand`` only sets dtype/device).

        Returns:
            config:  (Lx*Ly,) int64 SEMANTIC encoding {0,1,2,3}
            log_p_c: scalar float
        """
        import quimb.tensor as qtn
        from quimb.tensor.tn1d.compress import (
            tensor_network_1d_compress,
        )

        tns = qtn.unpack(params, self.skeleton)
        Lx, Ly = tns.Lx, tns.Ly
        d = tns.phys_dim()
        N = Lx * Ly
        y_tags = [tns.y_tag(y) for y in range(Ly)]

        n_up_so_far = torch.zeros(
            (), dtype=torch.int64, device=u_rand.device,
        )
        n_dn_so_far = torch.zeros(
            (), dtype=torch.int64, device=u_rand.device,
        )
        config_list = []
        log_p_c = torch.zeros(
            (), dtype=u_rand.dtype, device=u_rand.device,
        )
        boundary = None

        for x in range(Lx):
            row_tn = tns.select(tns.x_tag(x)).copy()
            combined = (
                (boundary | row_tn) if boundary is not None else row_tn
            )
            if chi_s is not None and chi_s > 0:
                combined_rc = tensor_network_1d_compress(
                    combined, max_bond=chi_s, cutoff=0.0,
                    method='direct', site_tags=y_tags,
                )
            else:
                combined_rc = combined
                for i in range(Ly):
                    combined_rc.contract_tags_(y_tags[i])

            sampled_row, log_p_row, n_up_so_far, n_dn_so_far = (
                self._sample_row_single(
                    combined_rc, tns, x, d, u_rand,
                    n_up_so_far, n_dn_so_far,
                    n_up_target, n_dn_target, N, forced,
                )
            )
            config_list.extend(sampled_row)
            log_p_c = log_p_c + log_p_row

            # Fix this row's phys legs (semantic -> linear) -> new boundary.
            isel = {}
            for y in range(Ly):
                t_col = combined_rc[y_tags[y]]
                t_col.data._label = self._site_label(tns, x, y)
                isel[tns.site_ind(x, y)] = self._phys_linear(sampled_row[y])
            boundary = combined_rc.isel(isel)

        return torch.stack(config_list), log_p_c

    @torch.no_grad()
    def direct_sample_vmap(self, u_batch, chi_s=None,
                           n_up_target=None, n_dn_target=None, chi_m=None,
                           forced_configs=None):
        """Vectorized fermionic direct sampler over a batch of B walkers.

        Args:
            u_batch:     (B, Lx*Ly) torch float uniforms in [0, 1).
            chi_s:       boundary MPS bond. None -> self.chi (or no
                compression if self.chi <= 0).
            n_up_target: U(1) up-count constraint or None.
            n_dn_target: U(1) down-count constraint or None.
            chi_m:       must be None or 0 (chi_m>0 not implemented).
            forced_configs: (B, Lx*Ly) int64 SEMANTIC configs or None.
                If given, return them unchanged with ``log p_c`` of each
                under the current parameters (no sampling; ``u_batch``
                only sets dtype/device). Used to refresh the MH cache.

        Returns:
            configs: (B, Lx*Ly) int64 SEMANTIC encoding {0,1,2,3}
            log_pcs: (B,) float
        """
        import quimb as qu

        if chi_m is not None and chi_m > 0:
            raise NotImplementedError(
                "fPEPS direct sampling supports chi_m=0 only."
            )
        if chi_s is None:
            chi_s = self.chi if self.chi > 0 else None

        params = qu.utils.tree_unflatten(
            list(self.params), self.params_pytree,
        )

        def _single(u, params, forced):
            return self.direct_sample_single(
                u, params, chi_s, n_up_target, n_dn_target, forced,
            )

        forced_in_dim = None if forced_configs is None else 0
        return torch.vmap(_single, in_dims=(0, None, forced_in_dim))(
            u_batch, params, forced_configs,
        )


# =================================================================
#  fPEPS with HOTRG contraction
# =================================================================


class fPEPS_Model_HOTRG_GPU(WavefunctionModel_GPU):
    """fPEPS model using HOTRG contraction.

    Uses quimb's contract_hotrg for amplitude evaluation instead
    of boundary MPS contraction. HOTRG coarse-grains the 2D TN
    by inserting oblique projectors between plaquettes.

    Same interface as fPEPS_Model_GPU — drop-in replacement.
    """

    def __init__(
        self,
        tn,
        max_bond,
        dtype=torch.float64,
        contract_hotrg_opts=None,
        **kwargs,
    ):
        import quimb as qu
        import quimb.tensor as qtn

        if contract_hotrg_opts is None:
            contract_hotrg_opts = {}

        tn = tn.copy()  # don't modify original TN
        # config value k -> physical index perm[k]
        self._loc_basis_perm = strip_phys_linearmap(tn)

        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.contract_hotrg_opts = contract_hotrg_opts
        self.chi = max_bond

        params_flat, params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.params_pytree = params_pytree

        params_tensors = [
            torch.as_tensor(x, dtype=self.dtype)
            for x in params_flat
        ]

        super().__init__(params_list=params_tensors)

    def amplitude(self, x, params):
        """Single-sample amplitude via HOTRG contraction.

        Args:
            x:      (N_sites,) int64 — one configuration
            params: quimb pytree of parameter tensors

        Returns:
            scalar amplitude
        """
        import quimb.tensor as qtn

        tn = qtn.unpack(params, self.skeleton)
        if self._loc_basis_perm is not None:
            x = self._loc_basis_perm[x]
        amp = tn.isel({
            tn.site_ind(site): x[i]
            for i, site in enumerate(tn.sites)
        })
        return amp.contract_hotrg(
            max_bond=self.chi,
            cutoff=0.0,
            **self.contract_hotrg_opts,
        )

    def log_amplitude(self, x, params):
        """Single-sample log-amplitude via HOTRG contraction.

        Returns:
            (sign, log_abs) scalars
        """
        import quimb.tensor as qtn

        tn = qtn.unpack(params, self.skeleton)
        if self._loc_basis_perm is not None:
            x = self._loc_basis_perm[x]
        amp = tn.isel({
            tn.site_ind(site): x[i]
            for i, site in enumerate(tn.sites)
        })
        sign, exp10 = amp.contract_hotrg(
            max_bond=self.chi,
            cutoff=0.0,
            equalize_norms=1.0,
            strip_exponent=True,
            **self.contract_hotrg_opts,
        )
        log_abs = exp10 * torch.log(torch.tensor(10.0))
        return sign, log_abs

    def _vamp_params_preprocess(self, params):
        """Unflatten ParameterList -> quimb pytree."""
        import quimb as qu

        if isinstance(params, nn.ParameterList):
            params = list(params)
        params = qu.utils.tree_unflatten(
            params, self.params_pytree,
        )
        return params


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
        bold=True,
        **kwargs,
    ):
        import quimb as qu
        import quimb.tensor as qtn

        if contract_boundary_opts is None:
            contract_boundary_opts = {}

        tn = tn.copy()  # don't modify original TN
        # config value k -> physical index perm[k]
        self._loc_basis_perm = strip_phys_linearmap(tn)

        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.contract_boundary_opts = contract_boundary_opts
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        self.chi = max_bond
        self.debug = kwargs.get('debug', False)
        self.bold = bold # whether to aggresively do exact contraction, if True: do exact contraction for any assembly of bMPS and row TN (can have 3,4 rows); if 3, exact contract only when 3 rows

        # bMPS skeleton/in_dims dicts — populated by cache_bMPS_skeleton
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
            params, get_ref=True
        )
        self.params_pytree = params_pytree

        params_tensors = [
            torch.as_tensor(x, dtype=self.dtype) for x in params_flat
        ]

        self.radius = 0

        # Cache for vmapped reuse functions, keyed by
        # (direction, bMPS_keys_tuple).
        self._vamp_reuse_cache = {}

        # Compiled cache for export+compile bMPS caching.
        # Populated by export_and_compile_cache().
        self._compiled_cache = {}
        self._cache_manifest_x = None
        self._cache_manifest_y = None

        super().__init__(params_list=params_tensors)

    # ----- Param preprocessing -----

    def _vamp_params_preprocess(self, params):
        """Unflatten ParameterList -> quimb pytree."""
        import quimb as qu

        if isinstance(params, nn.ParameterList):
            params = list(params)
        params = qu.utils.tree_unflatten(params, self.params_pytree)
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
        if self._loc_basis_perm is not None:
            x = self._loc_basis_perm[x]
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
        if self._loc_basis_perm is not None:
            x = self._loc_basis_perm[x]
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
        if self._loc_basis_perm is not None:
            x = self._loc_basis_perm[x]
        amp = tns.isel({
            tns.site_ind(site): x[i]
            for i, site in enumerate(tns.sites)
        })

        # x-environment reuse
        if (bMPS_params_xmin is not None
                and bMPS_params_xmax is not None
                and bMPS_keys is not None):
            bMPS_min = unpack_ftn(
                bMPS_params_xmin[0],
                self.bMPS_x_skeletons[bMPS_keys[0]],
            )
            bMPS_min.exponent = bMPS_params_xmin[1]
            bMPS_max = unpack_ftn(
                bMPS_params_xmax[0],
                self.bMPS_x_skeletons[bMPS_keys[1]],
            )
            bMPS_max.exponent = bMPS_params_xmax[1]
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
                if len(amp_reuse.tensors) > 2 * self.Ly and (self.bold is not True):
                    has_raw = (
                        bMPS_keys[0] in self._raw_bMPS_x_keys
                        or bMPS_keys[1]
                        in self._raw_bMPS_x_keys
                    )

                    if has_raw:
                        if self.bold == 3 and len(amp_reuse.tensors) == 3 * self.Ly:
                            pass
                        else:
                            is_xmin_bmps = bMPS_keys[0] in self._raw_bMPS_x_keys
                            is_xmax_bmps = bMPS_keys[1] in self._raw_bMPS_x_keys
                            if is_xmin_bmps:
                                amp_reuse.contract_boundary_from_xmin_(
                                    max_bond=self.chi, cutoff=0.0,
                                    xrange=[
                                        bMPS_keys[0][1]-1,
                                        bMPS_keys[0][1],
                                    ],
                                    **self.contract_boundary_opts,
                                )
                            elif is_xmax_bmps:
                                amp_reuse.contract_boundary_from_xmax_(
                                    max_bond=self.chi, cutoff=0.0,
                                    xrange=[
                                        bMPS_keys[1][1]+1,
                                        bMPS_keys[1][1],
                                    ],
                                    **self.contract_boundary_opts,
                                )

                    if len(amp_reuse.tensors) == 4 * self.Ly:  # exactly 4 rows
                        amp_reuse.contract_boundary_from_xmax_(
                            max_bond=self.chi, cutoff=0.0,
                            xrange=[
                                bMPS_keys[1][1]+1,
                                max(
                                    bMPS_keys[1][1],
                                    0,
                                ),
                            ],
                            **self.contract_boundary_opts,
                        )

                    if len(amp_reuse.tensors) == 3 * self.Ly:  # exactly 3 rows
                        if self.bold == 3:
                            pass
                        else:
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
            bMPS_min = unpack_ftn(
                bMPS_params_ymin[0],
                self.bMPS_y_skeletons[bMPS_keys[0]],
            )
            bMPS_min.exponent = bMPS_params_ymin[1]
            bMPS_max = unpack_ftn(
                bMPS_params_ymax[0],
                self.bMPS_y_skeletons[bMPS_keys[1]],
            )
            bMPS_max.exponent = bMPS_params_ymax[1]
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
                if len(amp_reuse.tensors) > 2 * self.Lx and (self.bold is not True):
                    has_raw = (
                        bMPS_keys[0] in self._raw_bMPS_y_keys
                        or bMPS_keys[1]
                        in self._raw_bMPS_y_keys
                    )

                    if has_raw:
                        if self.bold == 3 and len(amp_reuse.tensors) == 3 * self.Lx:
                            pass
                        else:
                            is_ymin_bmps = bMPS_keys[0] in self._raw_bMPS_y_keys
                            is_ymax_bmps = bMPS_keys[1] in self._raw_bMPS_y_keys
                            if is_ymin_bmps:
                                amp_reuse.contract_boundary_from_ymin_(
                                    max_bond=self.chi, cutoff=0.0,
                                    yrange=[
                                        bMPS_keys[0][1]-1,
                                        bMPS_keys[0][1],
                                    ],
                                    **self.contract_boundary_opts,
                                )
                            elif is_ymax_bmps:
                                amp_reuse.contract_boundary_from_ymax_(
                                    max_bond=self.chi, cutoff=0.0,
                                    yrange=[
                                        bMPS_keys[1][1]+1,
                                        bMPS_keys[1][1],
                                    ],
                                    **self.contract_boundary_opts,
                                )

                    if len(amp_reuse.tensors) == 4 * self.Lx:  # exactly 4 cols
                        amp_reuse.contract_boundary_from_ymax_(
                            max_bond=self.chi, cutoff=0.0,
                            yrange=[
                                bMPS_keys[1][1]+1,
                                max(
                                    bMPS_keys[1][1],
                                    0,
                                ),
                            ],
                            **self.contract_boundary_opts,
                        )

                    if len(amp_reuse.tensors) == 3 * self.Lx:  # exactly 3 cols
                        if self.bold == 3:
                            pass
                        else:
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
        if self._loc_basis_perm is not None:
            x = self._loc_basis_perm[x]
        amp = tns.isel({
            tns.site_ind(site): x[i]
            for i, site in enumerate(tns.sites)
        })

        # x-environment reuse
        if (bMPS_params_xmin is not None
                and bMPS_params_xmax is not None
                and bMPS_keys is not None):
            bMPS_min = unpack_ftn(
                bMPS_params_xmin[0],
                self.bMPS_x_skeletons[bMPS_keys[0]],
            )
            bMPS_min.exponent = bMPS_params_xmin[1]
            bMPS_max = unpack_ftn(
                bMPS_params_xmax[0],
                self.bMPS_x_skeletons[bMPS_keys[1]],
            )
            bMPS_max.exponent = bMPS_params_xmax[1]
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
                if len(amp_reuse.tensors) > 2 * self.Ly and (self.bold is not True):
                    has_raw = (
                        bMPS_keys[0] in self._raw_bMPS_x_keys
                        or bMPS_keys[1]
                        in self._raw_bMPS_x_keys
                    )
                            
                    if has_raw:
                        if self.bold == 3 and len(amp_reuse.tensors) == 3 * self.Ly:
                            # if bold==3, for 3-row assembly of bMPS+rowTN, do exact contraction without SVDs even if one bMPS is raw (small bond) - often faster on GPU than doing SVD with small bond dimension
                            pass
                        else:
                            is_xmin_bmps = bMPS_keys[0] in self._raw_bMPS_x_keys
                            is_xmax_bmps = bMPS_keys[1] in self._raw_bMPS_x_keys
                            if is_xmin_bmps:
                                amp_reuse.contract_boundary_from_xmin_(
                                    max_bond=self.chi, cutoff=0.0,
                                    xrange=[
                                        bMPS_keys[0][1]-1,
                                        bMPS_keys[0][1],
                                    ],
                                    **self.contract_boundary_opts,
                                )
                            elif is_xmax_bmps:
                                amp_reuse.contract_boundary_from_xmax_(
                                    max_bond=self.chi, cutoff=0.0,
                                    xrange=[
                                        bMPS_keys[1][1]+1,
                                        bMPS_keys[1][1],
                                    ],
                                    **self.contract_boundary_opts,
                                )
                            
                    if len(amp_reuse.tensors) == 4 * self.Ly:  # exactly 4 rows
                        amp_reuse.contract_boundary_from_xmax_(
                            max_bond=self.chi, cutoff=0.0,
                            xrange=[
                                bMPS_keys[1][1]+1,
                                max(
                                    bMPS_keys[1][1],
                                    0,
                                ),
                            ],
                            **self.contract_boundary_opts,
                        )
                        
                    if len(amp_reuse.tensors) == 3 * self.Ly:  # exactly 3 rows
                        if self.bold == 3:
                            # exactly contract the 3-row TN without SVDs - often faster on GPU than doing SVD with small bond dimension
                            pass
                        else:
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
            bMPS_min = unpack_ftn(
                bMPS_params_ymin[0],
                self.bMPS_y_skeletons[bMPS_keys[0]],
            )
            bMPS_min.exponent = bMPS_params_ymin[1]
            bMPS_max = unpack_ftn(
                bMPS_params_ymax[0],
                self.bMPS_y_skeletons[bMPS_keys[1]],
            )
            bMPS_max.exponent = bMPS_params_ymax[1]
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
                if len(amp_reuse.tensors) > 2 * self.Lx and (self.bold is not True):
                    has_raw = (
                        bMPS_keys[0] in self._raw_bMPS_y_keys
                        or bMPS_keys[1]
                        in self._raw_bMPS_y_keys
                    )

                    if has_raw:
                        if self.bold == 3 and len(amp_reuse.tensors) == 3 * self.Lx:
                            # if bold==3, for 3-col assembly of bMPS+colTN, do exact contraction without SVDs even if one bMPS is raw (small bond) - often faster on GPU than doing SVD with small bond dimension
                            pass
                        else:
                            is_ymin_bmps = bMPS_keys[0] in self._raw_bMPS_y_keys
                            is_ymax_bmps = bMPS_keys[1] in self._raw_bMPS_y_keys
                            if is_ymin_bmps:
                                amp_reuse.contract_boundary_from_ymin_(
                                    max_bond=self.chi, cutoff=0.0,
                                    yrange=[
                                        bMPS_keys[0][1]-1,
                                        bMPS_keys[0][1],
                                    ],
                                    **self.contract_boundary_opts,
                                )
                            elif is_ymax_bmps:
                                amp_reuse.contract_boundary_from_ymax_(
                                    max_bond=self.chi, cutoff=0.0,
                                    yrange=[
                                        bMPS_keys[1][1]+1,
                                        bMPS_keys[1][1],
                                    ],
                                    **self.contract_boundary_opts,
                                )

                    if len(amp_reuse.tensors) == 4 * self.Lx:  # exactly 4 cols
                        amp_reuse.contract_boundary_from_ymax_(
                            max_bond=self.chi, cutoff=0.0,
                            yrange=[
                                bMPS_keys[1][1]+1,
                                max(
                                    bMPS_keys[1][1],
                                    0,
                                ),
                            ],
                            **self.contract_boundary_opts,
                        )

                    if len(amp_reuse.tensors) == 3 * self.Lx:  # exactly 3 cols
                        if self.bold == 3:
                            # exactly contract the 3-col TN without SVDs - often faster on GPU than doing SVD with small bond dimension
                            pass
                        else:
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
        """Batched full-contraction amplitude with pytree unflatten."""
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
                self.bMPS_params_x_in_dims[bMPS_keys[0]],
                self.bMPS_params_x_in_dims[bMPS_keys[1]],
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
                self.bMPS_params_y_in_dims[bMPS_keys[0]],
                self.bMPS_params_y_in_dims[bMPS_keys[1]],
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
                self.amplitude_reuse_log,
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
            self.amplitude_reuse_log,
            in_dims=(
                0, None, None, None, None, None, None, None, None,
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
        use_log_amp=False, directions=('x', 'y'),
    ):
        """Pre-export amplitude_reuse for all reuse patterns.

        For a nearest-neighbor Hamiltonian on Lx x Ly lattice,
        the possible reuse patterns are:
          - x-dir, single row r: selected_rows=[r]    (Lx patterns)
          - x-dir, two rows r,r+1: selected_rows=[r,r+1]  (Lx-1)
          - y-dir, single col c: selected_cols=[c]    (Ly patterns)
          - y-dir, two cols c,c+1: selected_cols=[c,c+1]  (Ly-1)
        Total: 2*(Lx + Ly) - 2 patterns.

        Each pattern is exported via torch.export, vmapped over the
        batch dimension, and compiled via torch.compile.

        Stores compiled functions in:
            self._compiled_reuse[(direction, indices_tuple)]

        Args:
            example_x: (N_sites,) int64 on the target device.
            mode: torch.compile mode.
            verbose: Print progress.
            use_log_amp: if True, export amplitude_reuse_log instead
                of amplitude_reuse. forward_reuse_log() will dispatch
                to compiled path; forward_reuse() to eager.
            directions: tuple of directions to compile,
                e.g. ('x',) or ('x', 'y').
        """
        import quimb as qu
        from torch.export import export
        mode = self._reject_cudagraph_mode(mode, 'environment reuse')

        self._compiled_reuse = {}
        self._compiled_reuse_log_amp = use_log_amp
        device = example_x.device

        # Compute example bMPS params for slicing
        x_batch = example_x.unsqueeze(0)  # (1, N_sites)
        bMPS_x, bMPS_y = self.cache_bMPS_params_vmap(x_batch)

        params_list = list(self.params)
        n_tn_params = len(params_list)

        patterns = []
        for direction in directions:
            L = self.Lx if direction == 'x' else self.Ly
            for width in (1, 2):
                for start in range(L - width + 1):
                    indices = tuple(range(start, start + width))
                    patterns.append((direction, indices))

        if verbose:
            print(
                f"Exporting {len(patterns)} reuse patterns "
                f"for {self.Lx}x{self.Ly} lattice..."
            )

        for pat_idx, (direction, indices) in enumerate(patterns):
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

            # Build the single-sample wrapper
            selected_rows = (
                list(indices) if direction == 'x' else None
            )
            selected_cols = (
                list(indices) if direction == 'y' else None
            )
            bMPS_keys_frozen = list(bMPS_keys)

            def make_wrapper(
                _bMPS_keys, _bMPS_min_pytree, _bMPS_max_pytree,
                _n_tn, _n_min, _selected_rows, _selected_cols,
                _direction, _use_log_amp,
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
                                tn_params, self.params_pytree,
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
                                tn_params, self.params_pytree,
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
                in_dims_min_flat, _ = qu.utils.tree_flatten(
                    in_dims_min, get_ref=True,
                )
                in_dims_max_flat, _ = qu.utils.tree_flatten(
                    in_dims_max, get_ref=True,
                )

                n_total_args = n_tn_params + n_min + n_max
                in_dims_tuple = (
                    (0,)  # x batched
                    + tuple([None] * n_tn_params)  # TN params
                    + tuple(in_dims_min_flat)  # bMPS_min
                    + tuple(in_dims_max_flat)  # bMPS_max
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
                        f"({direction}, {indices}) FAILED: {e}"
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
            # Use the base class version for the main module
            super()._move_exported_constants_to_device(device)
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
                self._maybe_mark_cudagraph_step()
                out = self._vmapped_compiled(x, *params_list)
                if getattr(self, '_uses_cudagraph', False):
                    out = out.clone()  # cudagraph reuses a static buffer
                return out
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
        """Reuse forward path — compiled if available, else eager.

        Dispatches to pre-compiled reuse function when
        export_and_compile_reuse has been called and the pattern
        matches. Falls back to eager vamp_reuse otherwise.
        """
        import quimb as qu

        # Determine the reuse key and extract bMPS boundary params
        key = None
        bMPS_keys = None
        bMPS_params_xmin = bMPS_params_xmax = None
        bMPS_params_ymin = bMPS_params_ymax = None

        if selected_rows is not None:
            key = ('x', tuple(selected_rows))
            bMPS_keys = [
                ('xmin', min(selected_rows)),
                ('xmax', max(selected_rows)),
            ]
            bMPS_params_xmin = bMPS_params_x_batched[bMPS_keys[0]]
            bMPS_params_xmax = bMPS_params_x_batched[bMPS_keys[1]]
        elif selected_cols is not None:
            key = ('y', tuple(selected_cols))
            bMPS_keys = [
                ('ymin', min(selected_cols)),
                ('ymax', max(selected_cols)),
            ]
            bMPS_params_ymin = bMPS_params_y_batched[bMPS_keys[0]]
            bMPS_params_ymax = bMPS_params_y_batched[bMPS_keys[1]]

        # Try compiled path (only when exported for amplitude, not log)
        if (
            key is not None
            and hasattr(self, '_compiled_reuse')
            and key in self._compiled_reuse
            and not getattr(self, '_compiled_reuse_log_amp', False)
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
        """Log-amplitude variant of forward_reuse.

        Uses native strip_exponent via vamp_reuse_log for
        numerically stable log-amplitude.  Dispatches to
        compiled path when export_and_compile_reuse was called
        with use_log_amp=True.

        Returns:
            (signs, log_abs): each (B,) float64.
        """
        import quimb as qu

        # Determine the reuse key and extract bMPS boundary params
        key = None
        bMPS_keys = None
        bMPS_params_xmin = bMPS_params_xmax = None
        bMPS_params_ymin = bMPS_params_ymax = None

        if selected_rows is not None:
            key = ('x', tuple(selected_rows))
            bMPS_keys = [
                ('xmin', min(selected_rows)),
                ('xmax', max(selected_rows)),
            ]
            bMPS_params_xmin = bMPS_params_x_batched[bMPS_keys[0]]
            bMPS_params_xmax = bMPS_params_x_batched[bMPS_keys[1]]
        elif selected_cols is not None:
            key = ('y', tuple(selected_cols))
            bMPS_keys = [
                ('ymin', min(selected_cols)),
                ('ymax', max(selected_cols)),
            ]
            bMPS_params_ymin = bMPS_params_y_batched[bMPS_keys[0]]
            bMPS_params_ymax = bMPS_params_y_batched[bMPS_keys[1]]

        # Try compiled path (only when exported for log-amp)
        if (
            key is not None
            and hasattr(self, '_compiled_reuse')
            and key in self._compiled_reuse
            and getattr(self, '_compiled_reuse_log_amp', False)
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
        if self._loc_basis_perm is not None:
            x = self._loc_basis_perm[x]
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
            bMPS_params_dict[key] = (
                bMPS_params,
                torch.tensor(0.0, dtype=self.dtype),
            )
        self.bMPS_x_skeletons = env_x
        self.bMPS_params_x_in_dims = qu.utils.tree_map(
            lambda _: 0, bMPS_params_dict
        )
        # Reference params for creating placeholders with
        # matching pytree structure (used by sides= option)
        self._bMPS_x_ref_params = bMPS_params_dict

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
        )
        bMPS_params_dict = {}
        for key, tn in env_y.items():
            bMPS_params, skeleton = pack_ftn(tn)
            env_y[key] = skeleton
            bMPS_params_dict[key] = (
                bMPS_params,
                torch.tensor(0.0, dtype=self.dtype),
            )
        self.bMPS_y_skeletons = env_y
        self.bMPS_params_y_in_dims = qu.utils.tree_map(
            lambda _: 0, bMPS_params_dict
        )
        self._bMPS_y_ref_params = bMPS_params_dict

        self._raw_bMPS_y_keys = set()
        if self.Ly >= 3:
            self._raw_bMPS_y_keys.add(('ymin', 1))
            self._raw_bMPS_y_keys.add(('ymax', self.Ly - 2))

    def _cache_bMPS_params_vmap_eager(self, x):
        """Eager vmap implementation of cache_bMPS_params_vmap."""
        import quimb as qu
        import quimb.tensor as qtn

        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree,
        )

        def cache_single(x_single, params):
            tns = qtn.unpack(params, self.skeleton)
            if self._loc_basis_perm is not None:
                x_single = self._loc_basis_perm[x_single]
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
                bMPS_x[key] = (
                    get_params_ftn(btn),
                    torch.as_tensor(btn.exponent),
                )
            env_y = amp.compute_y_environments(
                max_bond=self.chi, cutoff=0.0,
                **self.contract_boundary_opts,
            )
            bMPS_y = {}
            for key, btn in env_y.items():
                bMPS_y[key] = (
                    get_params_ftn(btn),
                    torch.as_tensor(btn.exponent),
                )
            return bMPS_x, bMPS_y

        return torch.vmap(
            cache_single, in_dims=(0, None),
        )(x, params)

    def cache_bMPS_params_vmap(self, x):
        """Compute batched bMPS params for all x and y envs.

        Uses compiled path if available, else eager vmap.

        Args:
            x: (B, N_sites) int64 — batch of configurations

        Returns:
            (bMPS_x_dict, bMPS_y_dict) — batched pytrees
        """
        if not self._compiled_cache:
            return self._cache_bMPS_params_vmap_eager(x)

        params_list = list(self.params)
        results = {}

        for direction in ('x', 'y'):
            if direction in self._compiled_cache:
                entry = self._compiled_cache[direction]
                flat_out = entry['fn'](x, *params_list)
                results[direction] = (
                    self._unflatten_cache_output(
                        flat_out, entry['manifest'],
                    )
                )
            else:
                # Fallback: run eager for this direction
                bMPS_x, bMPS_y = (
                    self._cache_bMPS_params_vmap_eager(x)
                )
                if direction == 'x':
                    results['x'] = bMPS_x
                    results['y'] = bMPS_y
                else:
                    results['y'] = bMPS_y
                break

        return results['x'], results['y']

    def _cache_bMPS_params_any_direction_vmap_eager(
        self, x, direction='x', sides='both',
    ):
        """Eager vmap implementation of
        cache_bMPS_params_any_direction_vmap.

        Args:
            x: (B, N_sites) int64
            direction: 'x' or 'y'
            sides: 'both', 'xmin', 'xmax', 'ymin', 'ymax'
                If not 'both', only compute envs from that side
                and fill the opposite side with empty-TN
                placeholders.
        """
        import quimb as qu
        import quimb.tensor as qtn

        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree,
        )

        def cache_x_single(x_single, params):
            tns = qtn.unpack(params, self.skeleton)
            if self._loc_basis_perm is not None:
                x_single = self._loc_basis_perm[x_single]
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            if sides == 'both':
                env_x = amp.compute_x_environments(
                    max_bond=self.chi, cutoff=0.0,
                    **self.contract_boundary_opts,
                )
            elif sides == 'xmax':
                env_x = amp.compute_xmax_environments(
                    max_bond=self.chi, cutoff=0.0,
                    **self.contract_boundary_opts,
                )
            elif sides == 'xmin':
                env_x = amp.compute_xmin_environments(
                    max_bond=self.chi, cutoff=0.0,
                    **self.contract_boundary_opts,
                )

            # Amplitude: use the non-empty side
            if sides == 'xmax':
                amp_val = (
                    amp.select(tns.row_tag(0))
                    | env_x[('xmax', 0)]
                ).contract()
            elif sides == 'xmin':
                amp_val = (
                    env_x[('xmin', self.Lx - 1)]
                    | amp.select(tns.row_tag(self.Lx - 1))
                ).contract()
            else:
                amp_val = (
                    env_x[('xmin', self.Lx // 2)]
                    | env_x[('xmax', self.Lx // 2 - 1)]
                ).contract()

            # Start from ref params (correct pytree shape),
            # overwrite with real computed envs
            bMPS_x = dict(self._bMPS_x_ref_params)
            for key, btn in env_x.items():
                bMPS_x[key] = (
                    get_params_ftn(btn),
                    torch.as_tensor(btn.exponent),
                )
            return bMPS_x, amp_val

        def cache_y_single(x_single, params):
            tns = qtn.unpack(params, self.skeleton)
            if self._loc_basis_perm is not None:
                x_single = self._loc_basis_perm[x_single]
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            if sides == 'both':
                env_y = amp.compute_y_environments(
                    max_bond=self.chi, cutoff=0.0,
                    **self.contract_boundary_opts,
                )
            elif sides == 'ymax':
                env_y = amp.compute_ymax_environments(
                    max_bond=self.chi, cutoff=0.0,
                    **self.contract_boundary_opts,
                )
            elif sides == 'ymin':
                env_y = amp.compute_ymin_environments(
                    max_bond=self.chi, cutoff=0.0,
                    **self.contract_boundary_opts,
                )

            if sides == 'ymax':
                amp_val = (
                    amp.select(tns.col_tag(0))
                    | env_y[('ymax', 0)]
                ).contract()
            elif sides == 'ymin':
                amp_val = (
                    env_y[('ymin', self.Ly - 1)]
                    | amp.select(tns.col_tag(self.Ly - 1))
                ).contract()
            else:
                amp_val = (
                    env_y[('ymin', self.Ly // 2)]
                    | env_y[('ymax', self.Ly // 2 - 1)]
                ).contract()

            # Start from ref params (correct pytree shape),
            # overwrite with real computed envs
            bMPS_y = dict(self._bMPS_y_ref_params)
            for key, btn in env_y.items():
                bMPS_y[key] = (
                    get_params_ftn(btn),
                    torch.as_tensor(btn.exponent),
                )
            return bMPS_y, amp_val

        if direction == 'x':
            return torch.vmap(
                cache_x_single, in_dims=(0, None),
            )(x, params)
        else:
            return torch.vmap(
                cache_y_single, in_dims=(0, None),
            )(x, params)

    def cache_bMPS_params_any_direction_vmap(
        self, x, direction='x', sides='both',
    ):
        """Compute batched bMPS params for one direction
        + amplitudes.

        Uses compiled path if available, else eager vmap.

        Args:
            x: (B, N_sites) int64
            direction: 'x' or 'y'
            sides: 'both', 'xmin', 'xmax', 'ymin', 'ymax'
                If not 'both', only compute envs from that
                side (saves ~half the caching time).

        Returns:
            (bMPS_params_dict, amp_vals)
        """
        if direction not in self._compiled_cache:
            return (
                self
                ._cache_bMPS_params_any_direction_vmap_eager(
                    x, direction, sides=sides,
                )
            )

        # Compiled path
        entry = self._compiled_cache[direction]
        params_list = list(self.params)
        flat_out = entry['fn'](x, *params_list)
        bMPS_dict = self._unflatten_cache_output(
            flat_out, entry['manifest'],
        )

        # Compute amplitude via forward_reuse (cheap: 1 row)
        if direction == 'x':
            amps = self.forward_reuse(
                x,
                bMPS_params_x_batched=bMPS_dict,
                selected_rows=[self.Lx // 2],
            )
        else:
            amps = self.forward_reuse(
                x,
                bMPS_params_y_batched=bMPS_dict,
                selected_cols=[self.Ly // 2],
            )

        return bMPS_dict, amps

    # ----- Compiled bMPS cache export -----

    def _build_cache_manifest(self, example_x):
        """Build output manifests for x and y cache functions.

        Runs _cache_bMPS_params_vmap_eager on a single-sample
        batch to discover the output structure (keys and tensor
        counts).

        Args:
            example_x: (N_sites,) int64 — single config

        Sets:
            self._cache_manifest_x: list of (key, n_tensors)
            self._cache_manifest_y: list of (key, n_tensors)
        """
        x_batch = example_x.unsqueeze(0)  # (1, N_sites)
        bMPS_x, bMPS_y = (
            self._cache_bMPS_params_vmap_eager(x_batch)
        )

        # x-direction manifest: sorted keys, count tensors
        self._cache_manifest_x = []
        for key in sorted(bMPS_x.keys()):
            params_list, _exponent = bMPS_x[key]
            self._cache_manifest_x.append(
                (key, len(params_list))
            )

        # y-direction manifest
        self._cache_manifest_y = []
        for key in sorted(bMPS_y.keys()):
            params_list, _exponent = bMPS_y[key]
            self._cache_manifest_y.append(
                (key, len(params_list))
            )

    def _unflatten_cache_output(self, flat_outputs, manifest):
        """Unflatten a flat tuple of tensors back into bMPS dict.

        Args:
            flat_outputs: tuple/list of tensors in manifest order.
                For each key: n_tensors param tensors + 1 exponent.
            manifest: list of (key, n_tensors) from
                _build_cache_manifest.

        Returns:
            dict {key: (params_list, exponent)}
        """
        result = {}
        idx = 0
        for key, n_tensors in manifest:
            params = list(flat_outputs[idx:idx + n_tensors])
            exponent = flat_outputs[idx + n_tensors]
            result[key] = (params, exponent)
            idx += n_tensors + 1
        return result

    def _make_cache_wrapper(self, direction):
        """Create a flat-in/flat-out wrapper for cache export.

        The wrapper takes (x_single, *flat_tn_params) and returns
        a flat tuple of tensors: for each bMPS key in manifest
        order, (param_tensors..., exponent_scalar).

        Args:
            direction: 'x' or 'y'

        Returns:
            (wrapper_fn, manifest) — the wrapper function and
            corresponding manifest.
        """
        import quimb as qu
        import quimb.tensor as qtn

        manifest = (
            self._cache_manifest_x
            if direction == 'x'
            else self._cache_manifest_y
        )
        skeleton = self.skeleton
        chi = self.chi
        contract_boundary_opts = self.contract_boundary_opts
        params_pytree = self.params_pytree
        dtype = self.dtype
        loc_basis_perm = self._loc_basis_perm

        def wrapper(x_single, *flat_tn_params):
            params = qu.utils.tree_unflatten(
                list(flat_tn_params), params_pytree,
            )
            tns = qtn.unpack(params, skeleton)
            if loc_basis_perm is not None:
                x_single = loc_basis_perm[x_single]
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            if direction == 'x':
                envs = amp.compute_x_environments(
                    max_bond=chi, cutoff=0.0,
                    **contract_boundary_opts,
                )
            else:
                envs = amp.compute_y_environments(
                    max_bond=chi, cutoff=0.0,
                    **contract_boundary_opts,
                )

            # Flatten output in manifest order
            flat_out = []
            for key, _n_tensors in manifest:
                btn = envs[key]
                flat_params = get_params_ftn(btn)
                flat_out.extend(flat_params)
                flat_out.append(
                    torch.as_tensor(
                        btn.exponent, dtype=dtype,
                    )
                )
            return tuple(flat_out)

        return wrapper, manifest

    def export_and_compile_cache(
        self,
        example_x,
        mode='default',
        verbose=True,
        directions=('x', 'y'),
    ):
        """Export+compile the bMPS cache functions.

        Exports compute_x_environments and compute_y_environments
        separately via torch.export -> torch.vmap -> torch.compile.

        Must be called after cache_bMPS_skeleton().

        Args:
            example_x: (N_sites,) int64 on target device.
            mode: torch.compile mode.
            verbose: print progress.
            directions: tuple of directions to compile,
                e.g. ('x',) or ('x', 'y').
        """
        import torch.nn as nn_mod
        from torch.export import export

        mode = self._reject_cudagraph_mode(mode, 'the bMPS cache path')

        device = example_x.device
        params_list = list(self.params)
        n_tn = len(params_list)

        # Build manifests if not done
        if self._cache_manifest_x is None:
            if verbose:
                print("Building cache manifests...")
            self._build_cache_manifest(example_x)

        for direction in directions:
            if verbose:
                print(
                    f"Exporting cache_{direction} "
                    f"({len(params_list)} TN tensors)..."
                )

            wrapper_fn, manifest = (
                self._make_cache_wrapper(direction)
            )

            class _CacheModule(nn_mod.Module):
                def __init__(self_, fn):
                    super().__init__()
                    self_._fn = fn

                def forward(self_, x, *flat_args):
                    return self_._fn(x, *flat_args)

            try:
                with torch.no_grad():
                    exported = export(
                        _CacheModule(wrapper_fn),
                        (example_x, *params_list),
                    )
                exported_module = exported.module()

                # Move CPU constants to GPU
                self._move_exported_constants_to_device(
                    device,
                    exported_module=exported_module,
                )

                # vmap over batch dim (x batched, params not)
                vmapped = torch.vmap(
                    exported_module,
                    in_dims=(0, *([None] * n_tn)),
                )

                compiled = torch.compile(
                    vmapped, mode=mode,
                )

                self._compiled_cache[direction] = {
                    'fn': compiled,
                    'manifest': manifest,
                    'exported_module': exported_module,
                }

                if verbose:
                    n_out = sum(
                        nt + 1 for _, nt in manifest
                    )
                    print(
                        f"  cache_{direction}: OK "
                        f"({len(manifest)} keys, "
                        f"{n_out} output tensors)"
                    )

            except Exception as e:
                if verbose:
                    print(
                        f"  cache_{direction}: FAILED: {e}"
                    )
                import traceback
                traceback.print_exc()

        if verbose:
            print(
                f"Compiled cache: "
                f"{list(self._compiled_cache.keys())}"
            )

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
            if self._loc_basis_perm is not None:
                x_single = self._loc_basis_perm[x_single]
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            bMPS_to_row = unpack_ftn(
                bMPS_params_x[bMPS_key][0],
                self.bMPS_x_skeletons[bMPS_key],
            )
            bMPS_to_row.exponent = bMPS_params_x[bMPS_key][1]
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
                    bMPS_params_x[(from_which, row_id + 1)][0],
                    get_ref=True,
                )
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree,
                )
                bMPS_params_x[(from_which, row_id + 1)] = (
                    updated_bMPS_params,
                    torch.as_tensor(updated_bMPS.exponent),
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
                    bMPS_params_x[(from_which, row_id - 1)][0],
                    get_ref=True,
                )
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree,
                )
                bMPS_params_x[(from_which, row_id - 1)] = (
                    updated_bMPS_params,
                    torch.as_tensor(updated_bMPS.exponent),
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
            if self._loc_basis_perm is not None:
                x_single = self._loc_basis_perm[x_single]
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            bMPS_to_col = unpack_ftn(
                bMPS_params_y[bMPS_key][0],
                self.bMPS_y_skeletons[bMPS_key],
            )
            bMPS_to_col.exponent = bMPS_params_y[bMPS_key][1]
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
                    bMPS_params_y[(from_which, col_id + 1)][0],
                    get_ref=True,
                )
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree,
                )
                bMPS_params_y[(from_which, col_id + 1)] = (
                    updated_bMPS_params,
                    torch.as_tensor(updated_bMPS.exponent),
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
                    bMPS_params_y[(from_which, col_id - 1)][0],
                    get_ref=True,
                )
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree,
                )
                bMPS_params_y[(from_which, col_id - 1)] = (
                    updated_bMPS_params,
                    torch.as_tensor(updated_bMPS.exponent),
                )
            return bMPS_params_y

        return torch.vmap(
            update_bMPS_params_y_single,
            in_dims=(
                0, None, None,
                self.bMPS_params_y_in_dims, None,
            ),
        )(x, params, col_id, bMPS_params_y_batched, from_which)

    # ----- Cheap gradient via hole contraction -----

    def cheap_grad_single(self, x, params, bMPS_params_x):
        """Single-sample cheap gradient via per-row hole contraction.

        For pure fTNS, psi(x) is linear in each row's tensors.
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

        if self._loc_basis_perm is not None:
            x = self._loc_basis_perm[x]

        # Compute amplitude from midpoint bMPS envs
        mid = self.Lx // 2
        bMPS_mid_below = unpack_ftn(
            bMPS_params_x[('xmin', mid)][0],
            self.bMPS_x_skeletons[('xmin', mid)],
        )
        bMPS_mid_below.exponent = (
            bMPS_params_x[('xmin', mid)][1]
        )
        bMPS_mid_above = unpack_ftn(
            bMPS_params_x[('xmax', mid - 1)][0],
            self.bMPS_x_skeletons[('xmax', mid - 1)],
        )
        bMPS_mid_above.exponent = (
            bMPS_params_x[('xmax', mid - 1)][1]
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
                b_below = unpack_ftn(
                    _bp_below[0], _sk_below,
                )
                b_below.exponent = _bp_below[1]
                b_above = unpack_ftn(
                    _bp_above[0], _sk_above,
                )
                b_above.exponent = _bp_above[1]
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
        import quimb as qu

        params = self._vamp_params_preprocess(self.params)
        return torch.vmap(
            self.cheap_grad_single,
            in_dims=(
                0, None, self.bMPS_params_x_in_dims,
            ),
        )(fxs, params, bMPS_params_x)

    # ----- Debug helper -----

    def amp_tn(self, x):
        """Return the raw TN for a single config (debug helper)."""
        import quimb as qu
        import quimb.tensor as qtn

        params = qu.utils.tree_unflatten(
            self.params, self.params_pytree
        )
        tns = qtn.unpack(params, self.skeleton)
        if self._loc_basis_perm is not None:
            x = self._loc_basis_perm[x]
        amp = tns.isel({
            tns.site_ind(site): x[i]
            for i, site in enumerate(tns.sites)
        })
        return amp
