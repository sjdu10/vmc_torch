"""Symmetry-projected wavefunction wrapper for GPU VMC.

Spin systems (SymmetryProjectedModel):
    psi^r(s) = (d_r / |G|) * sum_g chi^{(r)}(g) * psi(s[pi_g])

Fermion systems (FermionSymmetryProjectedModel):
    psi^r(n) = (d_r / |G|) * sum_g sign(g,n) * chi^{(r)}(g) * psi(D_g^{-1} n)

where sign(g,n) is the fermionic sign from reordering creation operators
after the site permutation g. For the symmray interleaved convention:
    sign(g, n) = (-1)^{sum_{i<j, g(i)>g(j)} n_i * n_j}
where n_i is the number of fermions at site i (0, 1, or 2).

Supports the C4v point group on a square lattice (Lx = Ly),
with 1D irreps: A1, A2, B1, B2.

Reference: Chen, "Large-scale simulation of deep neural quantum states"
(Dissertation, 2025), Chapter 4.
"""
import torch
import torch.nn as nn

from ._base import WavefunctionModel_GPU


# =================================================================
#  C4v symmetry group
# =================================================================


# Character table for C4v (all 8 elements listed individually).
# Order: [e, C4, C4^2, C4^3, sigma_v, sigma_v', sigma_d, sigma_d']
C4V_CHARACTERS = {
    'A1': [1, 1, 1, 1, 1, 1, 1, 1],
    'A2': [1, 1, 1, 1, -1, -1, -1, -1],
    'B1': [1, -1, 1, -1, 1, 1, -1, -1],
    'B2': [1, -1, 1, -1, -1, -1, 1, 1],
}

# Group element names (for reference / debugging)
C4V_ELEMENT_NAMES = [
    'e', 'C4', 'C4^2', 'C4^3',
    'sigma_v', "sigma_v'", 'sigma_d', "sigma_d'",
]


# =================================================================
#  C2v symmetry group (rectangular lattice)
# =================================================================


# Character table for C2v (4 elements: e, C2, sigma_v, sigma_v')
C2V_CHARACTERS = {
    'A1': [1, 1, 1, 1],
    'A2': [1, 1, -1, -1],
    'B1': [1, -1, 1, -1],
    'B2': [1, -1, -1, 1],
}

C2V_ELEMENT_NAMES = ['e', 'C2', 'sigma_v', "sigma_v'"]


def build_c2v_permutations(Lx, Ly):
    """Build the 4 site permutations for C2v on Lx x Ly lattice.

    C2v has 4 elements: identity, 180° rotation, and two
    reflections (x-axis and y-axis). Works for any Lx, Ly.

    Uses row-major ordering: site at (ix, iy) has flat index
    i = ix * Ly + iy.

    Returns:
        perms: (4, N_sites) int64 tensor.
    """
    N = Lx * Ly
    sites = torch.arange(N, dtype=torch.long)
    ix = sites // Ly
    iy = sites % Ly

    def flat(r, c):
        return r * Ly + c

    perms = torch.stack([
        flat(ix, iy),                       # e: identity
        flat(Lx - 1 - ix, Ly - 1 - iy),    # C2: 180 rotation
        flat(ix, Ly - 1 - iy),              # sigma_v: reflect y
        flat(Lx - 1 - ix, iy),              # sigma_v': reflect x
    ])  # (4, N)

    return perms


def build_symmetry_group(Lx, Ly):
    """Build symmetry group permutations and character table.

    For square lattice (Lx == Ly): C4v (8 elements).
    For rectangular lattice (Lx != Ly): C2v (4 elements).

    Returns:
        perms: (|G|, N_sites) int64 tensor.
        char_table: dict mapping irrep name -> list of characters.
        element_names: list of group element names.
    """
    if Lx == Ly:
        perms = build_c4v_permutations(Lx)
        return perms, C4V_CHARACTERS, C4V_ELEMENT_NAMES
    else:
        perms = build_c2v_permutations(Lx, Ly)
        return perms, C2V_CHARACTERS, C2V_ELEMENT_NAMES


def build_c4v_permutations(L):
    """Build the 8 site permutations for C4v on an L x L lattice.

    Uses row-major ordering: site at (ix, iy) has flat index
    i = ix * L + iy, where ix is the row and iy is the column.

    The permutation pi_g maps: pi_g[i] = g(i), so that
    s_transformed = s[pi_g] gives D_g^{-1} s.

    Rotations are about the lattice center ((L-1)/2, (L-1)/2).
    90 deg CCW rotation: (ix, iy) -> (L-1-iy, ix).

    Returns:
        perms: (8, N_sites) int64 tensor.
    """
    N = L * L
    sites = torch.arange(N, dtype=torch.long)
    ix = sites // L
    iy = sites % L

    def flat(r, c):
        return r * L + c

    perms = torch.stack([
        flat(ix, iy),                   # e: identity
        flat(L - 1 - iy, ix),           # C4: 90 CCW
        flat(L - 1 - ix, L - 1 - iy),  # C4^2: 180
        flat(iy, L - 1 - ix),           # C4^3: 270 CCW
        flat(ix, L - 1 - iy),           # sigma_v: reflect y-axis
        flat(L - 1 - ix, iy),           # sigma_v': reflect x-axis
        flat(iy, ix),                   # sigma_d: main diagonal
        flat(L - 1 - iy, L - 1 - ix),  # sigma_d': anti-diagonal
    ])  # (8, N)

    return perms


def verify_c4v_group(perms):
    """Verify that the permutations form a valid C4v group.

    Checks:
    1. Each row is a valid permutation.
    2. Group closure under composition.
    3. Key multiplication table entries.

    Returns True if all checks pass, raises AssertionError otherwise.
    """
    n_group, N = perms.shape

    # Each row should be a permutation of 0..N-1
    for g in range(n_group):
        assert torch.equal(
            perms[g].sort().values,
            torch.arange(N, dtype=perms.dtype),
        ), f"Element {g} is not a valid permutation"

    # Compose all pairs, check closure
    # pi_{g1 g2}[i] = pi_{g1}[pi_{g2}[i]]
    perm_set = set()
    for g in range(n_group):
        perm_set.add(tuple(perms[g].tolist()))

    for g1 in range(n_group):
        for g2 in range(n_group):
            composed = perms[g1][perms[g2]]
            assert tuple(composed.tolist()) in perm_set, (
                f"Composition of elements {g1} and {g2} "
                f"is not in the group"
            )

    # Check specific relations:
    # C4 * C4 = C4^2 (index 1*1 = 2)
    assert torch.equal(
        perms[1][perms[1]], perms[2]
    ), "C4 * C4 != C4^2"
    # C4^3 * C4 = e (index 3*1 = 0)
    assert torch.equal(
        perms[3][perms[1]], perms[0]
    ), "C4^3 * C4 != e"
    # sigma_d * sigma_d = e (index 6*6 = 0)
    assert torch.equal(
        perms[6][perms[6]], perms[0]
    ), "sigma_d^2 != e"

    return True


# =================================================================
#  Symmetry-projected model wrapper
# =================================================================


class SymmetryProjectedModel(WavefunctionModel_GPU):
    """Symmetry-projected wavefunction model.

    Wraps an existing WavefunctionModel_GPU and projects onto a
    chosen irrep of the lattice point group:
    - Square lattice (Lx == Ly): C4v (8 elements)
    - Rectangular lattice (Lx != Ly): C2v (4 elements)

        psi^r(s) = (1/|G|) sum_g chi^{(r)}(g) * psi(s[pi_g])

    The wrapper implements the full model interface so it is a
    drop-in replacement everywhere the pipeline expects a model.

    Args:
        base_model: WavefunctionModel_GPU instance.
        Lx: lattice rows.
        Ly: lattice columns.
        irrep: str, one of 'A1', 'A2', 'B1', 'B2'.
    """

    def __init__(self, base_model, Lx, Ly, irrep='A1'):
        # Build symmetry group (C4v or C2v)
        perms, char_table, _ = build_symmetry_group(
            Lx, Ly,
        )
        group_name = 'C4v' if Lx == Ly else 'C2v'
        assert irrep in char_table, (
            f"Unknown irrep '{irrep}' for {group_name}. "
            f"Choose from {list(char_table.keys())}"
        )

        chars = char_table[irrep]
        characters = torch.tensor(
            chars, dtype=torch.float64,
        )

        # Initialize nn.Module directly (not WavefunctionModel_GPU
        # __init__) because we don't own the parameters — the
        # base_model does.
        nn.Module.__init__(self)

        # Register base_model as submodule so its parameters
        # are visible via self.parameters()
        self.base_model = base_model
        self.Lx = Lx
        self.Ly = Ly
        self.irrep = irrep
        self.n_group = perms.shape[0]

        # Register buffers (non-learnable, moved with .to())
        self.register_buffer('perms', perms)
        self.register_buffer('characters', characters)

        # Mirror compile status from base model
        self._compiled = base_model._compiled
        self._exported = base_model._exported
        self._exported_log_amp = getattr(
            base_model, '_exported_log_amp', False,
        )

    def __getattr__(self, name):
        """Delegate unknown attributes to base_model.

        This ensures attributes like `radius`, `Lx`, `Ly` etc.
        are accessible even when accessed on the wrapper
        (e.g., by reuse samplers/energy evaluators).
        """
        # Avoid infinite recursion: nn.Module uses __getattr__
        # for registered modules/buffers/params, so only
        # delegate when the normal lookup fails.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    @property
    def params(self):
        """Expose base model's params for gradient computation."""
        return self.base_model.params

    @params.setter
    def params(self, value):
        self.base_model.params = value

    # ----- Single-sample interface -----

    def amplitude(self, x, params_list):
        """Single-sample projected amplitude.

        Args:
            x: (N_sites,) int64.
            params_list: list of parameter tensors.

        Returns:
            Scalar projected amplitude.
        """
        total = torch.tensor(
            0.0, dtype=self.characters.dtype,
            device=x.device,
        )
        for g in range(self.n_group):
            x_g = x[self.perms[g]]
            amp_g = self.base_model.amplitude(x_g, params_list)
            total = total + self.characters[g] * amp_g
        return total / self.n_group

    def log_amplitude(self, x, params_list):
        """Single-sample log projected amplitude.

        Uses logsumexp trick for numerical stability.

        Returns:
            (sign, log_abs) scalars.
        """
        signs = []
        log_abs_vals = []
        for g in range(self.n_group):
            x_g = x[self.perms[g]]
            s_g, la_g = self.base_model.log_amplitude(
                x_g, params_list,
            )
            signs.append(self.characters[g] * s_g)
            log_abs_vals.append(la_g)

        signs_t = torch.stack(signs)
        log_abs_t = torch.stack(log_abs_vals)

        # logsumexp: sum_g sign_g * exp(log_abs_g)
        max_la = log_abs_t.max()
        weighted_sum = (
            signs_t * torch.exp(log_abs_t - max_la)
        ).sum()

        result_sign = torch.sign(weighted_sum)
        result_log_abs = (
            max_la
            + torch.log(weighted_sum.abs().clamp(min=1e-45))
            - torch.log(
                torch.tensor(
                    float(self.n_group),
                    dtype=max_la.dtype,
                    device=max_la.device,
                )
            )
        )
        return result_sign, result_log_abs

    # ----- Batched interface -----

    def forward(self, x):
        """Batched projected amplitude: (B, N) -> (B,).

        Reshapes (B, N) -> (B*|G|, N) with all permuted configs,
        calls base model once, reshapes back and sums with chars.
        """
        B, N = x.shape
        # x[:, self.perms] -> (B, |G|, N)
        x_all = x[:, self.perms]
        x_flat = x_all.reshape(B * self.n_group, N)

        # Single batched forward on all permuted configs
        amps_flat = self.base_model(x_flat)
        amps_grouped = amps_flat.reshape(B, self.n_group)

        # Weight by characters and average
        projected = (
            (amps_grouped * self.characters).sum(dim=1)
            / self.n_group
        )
        return projected

    def forward_log(self, x):
        """Batched log projected amplitude with logsumexp.

        Returns:
            (signs, log_abs) each (B,).
        """
        B, N = x.shape
        x_all = x[:, self.perms]
        x_flat = x_all.reshape(B * self.n_group, N)

        signs_flat, log_abs_flat = (
            self.base_model.forward_log(x_flat)
        )
        signs_g = signs_flat.reshape(B, self.n_group)
        log_abs_g = log_abs_flat.reshape(B, self.n_group)

        # Effective signs: character * base_sign
        eff_signs = signs_g * self.characters

        # Logsumexp per sample
        max_la = log_abs_g.max(dim=1, keepdim=True).values
        weighted_sum = (
            eff_signs * torch.exp(log_abs_g - max_la)
        ).sum(dim=1)

        result_sign = torch.sign(weighted_sum)
        result_log_abs = (
            max_la.squeeze(1)
            + torch.log(
                weighted_sum.abs().clamp(min=1e-45)
            )
            - torch.log(
                torch.tensor(
                    float(self.n_group),
                    dtype=max_la.dtype,
                    device=max_la.device,
                )
            )
        )
        return result_sign, result_log_abs

    # ----- vamp interface for torch.func.grad -----

    def _vamp_params_preprocess(self, params):
        """Delegate to base model's preprocessing."""
        return self.base_model._vamp_params_preprocess(params)

    def vamp(self, x, params):
        """Batched projected amplitude for torch.func.grad.

        Vectorized: reshapes (B, N) -> (B*|G|, N), single
        call to base_model._vmapped_amplitude, reshape back.
        """
        params_pp = self._vamp_params_preprocess(params)
        B, N = x.shape
        # (B, |G|, N) -> (B*|G|, N)
        x_flat = x[:, self.perms].reshape(
            B * self.n_group, N,
        )
        amps_flat = self.base_model._vmapped_amplitude(
            x_flat, params_pp,
        )
        amps_grouped = amps_flat.reshape(B, self.n_group)
        return (
            (amps_grouped * self.characters).sum(dim=1)
            / self.n_group
        )

    def vamp_log(self, x, params):
        """Batched log projected amplitude for torch.func.grad.

        Vectorized: single call to base_model._vmapped_log_amplitude.

        Returns:
            (signs, log_abs) each (B,).
        """
        params_pp = self._vamp_params_preprocess(params)
        B, N = x.shape
        x_flat = x[:, self.perms].reshape(
            B * self.n_group, N,
        )
        signs_flat, log_abs_flat = (
            self.base_model._vmapped_log_amplitude(
                x_flat, params_pp,
            )
        )
        signs_g = signs_flat.reshape(B, self.n_group)
        log_abs_g = log_abs_flat.reshape(B, self.n_group)

        eff_signs = signs_g * self.characters
        max_la = log_abs_g.max(dim=1, keepdim=True).values
        weighted_sum = (
            eff_signs * torch.exp(log_abs_g - max_la)
        ).sum(dim=1)

        result_sign = torch.sign(weighted_sum)
        result_log_abs = (
            max_la.squeeze(1)
            + torch.log(
                weighted_sum.abs().clamp(min=1e-45)
            )
            - torch.log(
                torch.tensor(
                    float(self.n_group),
                    dtype=max_la.dtype,
                    device=max_la.device,
                )
            )
        )
        return result_sign, result_log_abs

    # ----- Export + compile -----

    def export_and_compile(self, example_x, **kwargs):
        """Export+compile the base model, then mark self."""
        self.base_model.export_and_compile(
            example_x, **kwargs,
        )
        self._compiled = True
        self._exported = True
        self._exported_log_amp = (
            self.base_model._exported_log_amp
        )

    def export_only(self, example_x, **kwargs):
        """Export without compile."""
        self.base_model.export_only(example_x, **kwargs)
        self._exported = True
        self._exported_log_amp = (
            self.base_model._exported_log_amp
        )

    def export_grad(
        self, mode='default', use_log_amp=False,
        do_compile=False, **compile_kwargs,
    ):
        """Build vmap(grad) with symmetry projection.

        The single-sample function loops over |G| group elements,
        calling the base model's exported single-sample function
        for each permuted config, weighted by characters.

        Args:
            do_compile: if True, wrap with torch.compile.
                Default False.
        """
        assert self.base_model._exported, (
            "Call export_and_compile() before export_grad()"
        )
        base = self.base_model
        perms = self.perms        # (|G|, N)
        chars = self.characters   # (|G|,)
        n_group = self.n_group
        params_list = list(base.params)
        n_params = len(params_list)
        argnums = tuple(range(1, n_params + 1))
        in_dims = (0,) + (None,) * n_params

        # Get base model's exported single-sample fn
        base_fn = self._get_base_single_sample_fn(
            use_log_amp,
        )

        if use_log_amp:
            def single_fn(x_i, *flat_params):
                # Evaluate each group element
                signs_list = []
                log_abs_list = []
                for g in range(n_group):
                    x_g = x_i[perms[g]]
                    s_g, la_g = base_fn(x_g, *flat_params)
                    signs_list.append(
                        chars[g] * s_g
                    )
                    log_abs_list.append(la_g)
                signs_t = torch.stack(signs_list)
                log_abs_t = torch.stack(log_abs_list)
                max_la = log_abs_t.max()
                weighted = (
                    signs_t
                    * torch.exp(log_abs_t - max_la)
                ).sum()
                result_sign = torch.sign(weighted)
                result_log_abs = (
                    max_la
                    + torch.log(
                        weighted.abs().clamp(min=1e-45)
                    )
                    - torch.log(
                        torch.tensor(
                            float(n_group),
                            dtype=max_la.dtype,
                            device=max_la.device,
                        )
                    )
                )
                return result_log_abs, (
                    result_sign, result_log_abs,
                )
        else:
            def single_fn(x_i, *flat_params):
                total = torch.tensor(
                    0.0, dtype=chars.dtype,
                    device=x_i.device,
                )
                for g in range(n_group):
                    x_g = x_i[perms[g]]
                    amp_g = base_fn(x_g, *flat_params)
                    total = total + chars[g] * amp_g
                result = total / n_group
                return result, result

        grad_fn = torch.func.grad(
            single_fn, argnums=argnums, has_aux=True,
        )
        vmapped = torch.vmap(grad_fn, in_dims=in_dims)

        if do_compile:
            self._exported_grad_fn = torch.compile(
                vmapped, mode=mode, **compile_kwargs,
            )
        else:
            self._exported_grad_fn = vmapped

        self._grad_exported = True
        self._grad_use_log_amp = use_log_amp

    def _get_base_single_sample_fn(self, use_log_amp):
        """Return the base model's exported single-sample fn.

        Handles both pure TNS (has _exported_module) and
        NNfTNS (has _exported_tn_module + NN).
        """
        base = self.base_model
        if hasattr(base, '_exported_tn_module'):
            # NNfTNS: NN + exported TN
            exported_tn = base._exported_tn_module
            nn_container = base._nn_container
            nn_param_names = base._nn_param_names
            nn_param_dtypes = base._nn_param_dtypes
            n_ftn = base.n_ftn

            if use_log_amp:
                def fn(x_i, *all_params):
                    ftn_ps = all_params[:n_ftn]
                    nn_ps = all_params[n_ftn:]
                    nn_dict = {
                        name: p.to(dt)
                        if p.dtype != dt else p
                        for name, p, dt in zip(
                            nn_param_names, nn_ps,
                            nn_param_dtypes,
                        )
                    }
                    nn_out = torch.func.functional_call(
                        nn_container[0], nn_dict,
                        (x_i.unsqueeze(0),),
                    ).squeeze(0)
                    return exported_tn(
                        x_i, nn_out, *ftn_ps,
                    )
            else:
                def fn(x_i, *all_params):
                    ftn_ps = all_params[:n_ftn]
                    nn_ps = all_params[n_ftn:]
                    nn_dict = {
                        name: p.to(dt)
                        if p.dtype != dt else p
                        for name, p, dt in zip(
                            nn_param_names, nn_ps,
                            nn_param_dtypes,
                        )
                    }
                    nn_out = torch.func.functional_call(
                        nn_container[0], nn_dict,
                        (x_i.unsqueeze(0),),
                    ).squeeze(0)
                    return exported_tn(
                        x_i, nn_out, *ftn_ps,
                    )
            return fn
        else:
            # Pure TNS: _exported_module
            exported = base._exported_module
            if use_log_amp:
                def fn(x_i, *flat_params):
                    return exported(x_i, *flat_params)
            else:
                def fn(x_i, *flat_params):
                    return exported(x_i, *flat_params)
            return fn


# =================================================================
#  Fermionic sign computation
# =================================================================


def build_mode_perms(perms):
    """Build 2N-mode permutations from N-site permutations.

    For the symmray interleaved convention [d0, u0, d1, u1, ...],
    a site permutation g that maps site i -> g(i) induces a
    mode permutation:
        mode 2*i   (down at site i) -> mode 2*g(i)   (down at g(i))
        mode 2*i+1 (up at site i)   -> mode 2*g(i)+1 (up at g(i))

    Args:
        perms: (|G|, N_sites) int64 site permutations.

    Returns:
        mode_perms: (|G|, 2*N_sites) int64 mode permutations.
    """
    n_group, N = perms.shape
    mode_perms = torch.empty(
        n_group, 2 * N, dtype=torch.long,
    )
    # mode 2i -> 2*g(i), mode 2i+1 -> 2*g(i)+1
    mode_perms[:, 0::2] = 2 * perms        # down modes
    mode_perms[:, 1::2] = 2 * perms + 1    # up modes
    return mode_perms


def build_mode_inv_masks(mode_perms):
    """Precompute inversion masks at the 2N-mode level.

    inv_mask[g, m1, m2] = True iff m1 < m2 and
    mode_perm_g[m1] > mode_perm_g[m2].

    Args:
        mode_perms: (|G|, 2*N_sites) int64.

    Returns:
        inv_masks: (|G|, 2*N, 2*N) bool.
    """
    n_group, M = mode_perms.shape
    upper = torch.triu(
        torch.ones(M, M, dtype=torch.bool), diagonal=1,
    )
    g_i = mode_perms.unsqueeze(2)  # (|G|, 2N, 1)
    g_j = mode_perms.unsqueeze(1)  # (|G|, 1, 2N)
    inv_masks = (g_i > g_j) & upper  # (|G|, 2N, 2N)
    return inv_masks


def compute_fermion_signs(fxs, mode_inv_masks):
    """Compute fermionic sign(g, n) for all group elements.

    Works at the 2N-mode level in the symmray interleaved
    convention [d0, u0, d1, u1, ...]. The sign is the parity
    of the mode permutation restricted to occupied modes:

        sign(g, n) = (-1)^{# inversions among occupied modes}

    This correctly accounts for cross-spin ordering — e.g.,
    an up fermion at site i crossing a down fermion at site j.

    Args:
        fxs: (B, N_sites) int64, quimb encoding {0,1,2,3}.
        mode_inv_masks: (|G|, 2N, 2N) bool, precomputed.

    Returns:
        signs: (B, |G|) float64, each entry +1.0 or -1.0.
    """
    B, N = fxs.shape

    # Build (B, 2N) occupation in symmray order [d0,u0,d1,u1,...]
    occ_down = ((fxs == 1) | (fxs == 3)).long()  # (B, N)
    occ_up = ((fxs == 2) | (fxs == 3)).long()     # (B, N)
    # Interleave: [d0, u0, d1, u1, ...]
    occ = torch.stack(
        [occ_down, occ_up], dim=-1,
    ).reshape(B, 2 * N)  # (B, 2N)

    # Occupied-pair matrix: occ_m1 * occ_m2 for all mode pairs
    occ_pairs = (
        occ.unsqueeze(2) * occ.unsqueeze(1)
    )  # (B, 2N, 2N)

    # Count inversions among occupied modes for each g
    # mode_inv_masks: (|G|, 2N, 2N), occ_pairs: (B, 2N, 2N)
    exponents = torch.einsum(
        'gmn,bmn->bg',
        mode_inv_masks.float(), occ_pairs.float(),
    ).long()

    signs = 1.0 - 2.0 * (exponents % 2).to(torch.float64)
    return signs


# =================================================================
#  Fermionic symmetry-projected model
# =================================================================


class FermionSymmetryProjectedModel(SymmetryProjectedModel):
    """Symmetry-projected wavefunction for fermionic systems.

    Extends SymmetryProjectedModel with the fermionic sign factor:

        psi^r(n) = (1/|G|) sum_g sign(g,n) chi^(r)(g) psi(D_g^{-1} n)

    The sign is computed at the 2N-mode level in the symmray
    interleaved convention [d0, u0, d1, u1, ...], properly
    accounting for cross-spin ordering.

    Args:
        base_model: WavefunctionModel_GPU instance (e.g. fPEPS).
        Lx: lattice rows.
        Ly: lattice columns (must equal Lx for C4v).
        irrep: str, one of 'A1', 'A2', 'B1', 'B2'.
    """

    def __init__(self, base_model, Lx, Ly, irrep='A1'):
        super().__init__(base_model, Lx, Ly, irrep)

        # The projection formula requires sign(g^{-1}, n), NOT
        # sign(g, n). These differ for non-self-inverse elements
        # (e.g., C4 vs C4^3). We build mode masks from the
        # INVERSE site permutations so that mode_inv_masks[g]
        # gives the inversions for g^{-1}.
        N = self.perms.shape[1]
        inv_perms = torch.empty_like(self.perms)
        for g in range(self.n_group):
            inv_perms[g][self.perms[g]] = torch.arange(
                N, dtype=torch.long,
            )

        mode_perms = build_mode_perms(inv_perms)
        mode_inv_masks = build_mode_inv_masks(mode_perms)
        self.register_buffer('mode_perms', mode_perms)
        self.register_buffer('mode_inv_masks', mode_inv_masks)

    # ----- Helper: single-sample sign -----

    def _single_sample_signs(self, x):
        """Compute sign(g, x) for all g, single sample.

        Args:
            x: (N_sites,) int64, quimb encoding.

        Returns:
            signs: (|G|,) float64.
        """
        N = x.shape[0]
        # Build symmray occupation: [d0, u0, d1, u1, ...]
        occ_d = ((x == 1) | (x == 3)).long()  # (N,)
        occ_u = ((x == 2) | (x == 3)).long()  # (N,)
        occ = torch.stack(
            [occ_d, occ_u], dim=-1,
        ).reshape(2 * N)  # (2N,)

        # Occupied-pair products
        occ_pairs = occ.unsqueeze(1) * occ.unsqueeze(0)  # (2N, 2N)

        # Count inversions for each g
        exponents = (
            self.mode_inv_masks.long() * occ_pairs
        ).sum(dim=(1, 2))  # (|G|,)

        return 1.0 - 2.0 * (exponents % 2).to(
            torch.float64,
        )

    # ----- Single-sample interface -----

    def amplitude(self, x, params_list):
        """Single-sample projected amplitude with fermionic sign."""
        fsigns = self._single_sample_signs(x)  # (|G|,)

        total = torch.tensor(
            0.0, dtype=self.characters.dtype,
            device=x.device,
        )
        for g in range(self.n_group):
            x_g = x[self.perms[g]]
            amp_g = self.base_model.amplitude(
                x_g, params_list,
            )
            total = (
                total
                + fsigns[g] * self.characters[g] * amp_g
            )
        return total / self.n_group

    def log_amplitude(self, x, params_list):
        """Single-sample log projected amplitude with sign."""
        fsigns = self._single_sample_signs(x)  # (|G|,)

        signs = []
        log_abs_vals = []
        for g in range(self.n_group):
            x_g = x[self.perms[g]]
            s_g, la_g = self.base_model.log_amplitude(
                x_g, params_list,
            )
            signs.append(
                fsigns[g] * self.characters[g] * s_g
            )
            log_abs_vals.append(la_g)

        signs_t = torch.stack(signs)
        log_abs_t = torch.stack(log_abs_vals)

        max_la = log_abs_t.max()
        weighted_sum = (
            signs_t * torch.exp(log_abs_t - max_la)
        ).sum()

        result_sign = torch.sign(weighted_sum)
        result_log_abs = (
            max_la
            + torch.log(
                weighted_sum.abs().clamp(min=1e-45)
            )
            - torch.log(
                torch.tensor(
                    float(self.n_group),
                    dtype=max_la.dtype,
                    device=max_la.device,
                )
            )
        )
        return result_sign, result_log_abs

    # ----- Batched interface -----

    def forward(self, x):
        """Batched projected amplitude with fermionic sign."""
        B, N = x.shape
        x_all = x[:, self.perms]
        x_flat = x_all.reshape(B * self.n_group, N)

        amps_flat = self.base_model(x_flat)
        amps_grouped = amps_flat.reshape(B, self.n_group)

        # Fermionic signs: (B, |G|)
        fsigns = compute_fermion_signs(x, self.mode_inv_masks)

        projected = (
            (amps_grouped * self.characters * fsigns)
            .sum(dim=1) / self.n_group
        )
        return projected

    def forward_log(self, x):
        """Batched log projected amplitude with fermionic sign."""
        B, N = x.shape
        x_all = x[:, self.perms]
        x_flat = x_all.reshape(B * self.n_group, N)

        signs_flat, log_abs_flat = (
            self.base_model.forward_log(x_flat)
        )
        signs_g = signs_flat.reshape(B, self.n_group)
        log_abs_g = log_abs_flat.reshape(B, self.n_group)

        # Fermionic signs: (B, |G|)
        fsigns = compute_fermion_signs(x, self.mode_inv_masks)

        eff_signs = signs_g * self.characters * fsigns

        max_la = log_abs_g.max(dim=1, keepdim=True).values
        weighted_sum = (
            eff_signs * torch.exp(log_abs_g - max_la)
        ).sum(dim=1)

        result_sign = torch.sign(weighted_sum)
        result_log_abs = (
            max_la.squeeze(1)
            + torch.log(
                weighted_sum.abs().clamp(min=1e-45)
            )
            - torch.log(
                torch.tensor(
                    float(self.n_group),
                    dtype=max_la.dtype,
                    device=max_la.device,
                )
            )
        )
        return result_sign, result_log_abs

    # ----- vamp interface for torch.func.grad -----

    def vamp(self, x, params):
        """Batched projected amplitude for torch.func.grad."""
        params_pp = self._vamp_params_preprocess(params)
        B, N = x.shape
        x_flat = x[:, self.perms].reshape(
            B * self.n_group, N,
        )
        amps_flat = self.base_model._vmapped_amplitude(
            x_flat, params_pp,
        )
        amps_grouped = amps_flat.reshape(B, self.n_group)

        # Fermionic signs (no grad needed — config-only)
        fsigns = compute_fermion_signs(x, self.mode_inv_masks)

        return (
            (amps_grouped * self.characters * fsigns)
            .sum(dim=1) / self.n_group
        )

    def vamp_log(self, x, params):
        """Batched log projected amplitude for torch.func.grad."""
        params_pp = self._vamp_params_preprocess(params)
        B, N = x.shape
        x_flat = x[:, self.perms].reshape(
            B * self.n_group, N,
        )
        signs_flat, log_abs_flat = (
            self.base_model._vmapped_log_amplitude(
                x_flat, params_pp,
            )
        )
        signs_g = signs_flat.reshape(B, self.n_group)
        log_abs_g = log_abs_flat.reshape(B, self.n_group)

        fsigns = compute_fermion_signs(x, self.mode_inv_masks)
        eff_signs = signs_g * self.characters * fsigns

        max_la = log_abs_g.max(dim=1, keepdim=True).values
        weighted_sum = (
            eff_signs * torch.exp(log_abs_g - max_la)
        ).sum(dim=1)

        result_sign = torch.sign(weighted_sum)
        result_log_abs = (
            max_la.squeeze(1)
            + torch.log(
                weighted_sum.abs().clamp(min=1e-45)
            )
            - torch.log(
                torch.tensor(
                    float(self.n_group),
                    dtype=max_la.dtype,
                    device=max_la.device,
                )
            )
        )
        return result_sign, result_log_abs

    def export_grad(
        self, mode='default', use_log_amp=False,
        do_compile=False, **compile_kwargs,
    ):
        """Build vmap(grad) with fermionic symmetry.

        Same as SymmetryProjectedModel.export_grad but includes
        fermionic sign factors.

        Args:
            do_compile: if True, wrap with torch.compile.
                Default False.
        """
        assert self.base_model._exported, (
            "Call export_and_compile() before export_grad()"
        )
        base = self.base_model
        perms = self.perms
        chars = self.characters
        n_group = self.n_group
        # Capture buffers as local vars for torch.func compat
        mode_inv_masks = self.mode_inv_masks
        params_list = list(base.params)
        n_params = len(params_list)
        argnums = tuple(range(1, n_params + 1))
        in_dims = (0,) + (None,) * n_params

        base_fn = self._get_base_single_sample_fn(
            use_log_amp,
        )

        # Single-sample fermionic sign (captured via closure)
        def _fsigns(x_i):
            N = x_i.shape[0]
            occ_d = ((x_i == 1) | (x_i == 3)).long()
            occ_u = ((x_i == 2) | (x_i == 3)).long()
            occ = torch.stack(
                [occ_d, occ_u], dim=-1,
            ).reshape(2 * N)
            occ_pairs = (
                occ.unsqueeze(1) * occ.unsqueeze(0)
            )
            exponents = (
                mode_inv_masks.long() * occ_pairs
            ).sum(dim=(1, 2))
            return 1.0 - 2.0 * (exponents % 2).to(
                torch.float64,
            )

        if use_log_amp:
            def single_fn(x_i, *flat_params):
                fsigns = _fsigns(x_i)
                signs_list = []
                log_abs_list = []
                for g in range(n_group):
                    x_g = x_i[perms[g]]
                    s_g, la_g = base_fn(
                        x_g, *flat_params,
                    )
                    signs_list.append(
                        fsigns[g] * chars[g] * s_g
                    )
                    log_abs_list.append(la_g)
                signs_t = torch.stack(signs_list)
                log_abs_t = torch.stack(log_abs_list)
                max_la = log_abs_t.max()
                weighted = (
                    signs_t
                    * torch.exp(log_abs_t - max_la)
                ).sum()
                result_sign = torch.sign(weighted)
                result_log_abs = (
                    max_la
                    + torch.log(
                        weighted.abs().clamp(min=1e-45)
                    )
                    - torch.log(
                        torch.tensor(
                            float(n_group),
                            dtype=max_la.dtype,
                            device=max_la.device,
                        )
                    )
                )
                return result_log_abs, (
                    result_sign, result_log_abs,
                )
        else:
            def single_fn(x_i, *flat_params):
                fsigns = _fsigns(x_i)
                total = torch.tensor(
                    0.0, dtype=chars.dtype,
                    device=x_i.device,
                )
                for g in range(n_group):
                    x_g = x_i[perms[g]]
                    amp_g = base_fn(x_g, *flat_params)
                    total = (
                        total
                        + fsigns[g] * chars[g] * amp_g
                    )
                result = total / n_group
                return result, result

        grad_fn = torch.func.grad(
            single_fn, argnums=argnums, has_aux=True,
        )
        vmapped = torch.vmap(grad_fn, in_dims=in_dims)

        if do_compile:
            self._exported_grad_fn = torch.compile(
                vmapped, mode=mode, **compile_kwargs,
            )
        else:
            self._exported_grad_fn = vmapped

        self._grad_exported = True
        self._grad_use_log_amp = use_log_amp
