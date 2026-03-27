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


# =================================================================
#  Fermionic sign computation
# =================================================================


# Quimb encoding -> fermion count per site
# 0=empty->0, 1=down->1, 2=up->1, 3=both->2
_QUIMB_TO_NFERMIONS = torch.tensor(
    [0, 1, 1, 2], dtype=torch.long,
)


def build_inversion_masks(perms):
    """Precompute inversion masks for all group elements.

    For each group element g with site permutation perm_g,
    inv_mask[g, i, j] = True iff i < j and perm_g[i] > perm_g[j].

    These pairs are the sites whose fermions need to swap
    past each other under the permutation g.

    Args:
        perms: (|G|, N_sites) int64 site permutations.

    Returns:
        inv_masks: (|G|, N_sites, N_sites) bool.
    """
    n_group, N = perms.shape
    # upper triangle: i < j
    upper = torch.triu(
        torch.ones(N, N, dtype=torch.bool), diagonal=1,
    )
    # perm_g[i] > perm_g[j] for each group element
    # perms: (|G|, N), compare (|G|, N, 1) vs (|G|, 1, N)
    g_i = perms.unsqueeze(2)  # (|G|, N, 1)
    g_j = perms.unsqueeze(1)  # (|G|, 1, N)
    inv_masks = (g_i > g_j) & upper  # (|G|, N, N)
    return inv_masks


def compute_fermion_signs(fxs, inv_masks):
    """Compute fermionic sign(g, n) for all group elements.

    Uses the formula for the symmray interleaved convention:
        sign(g, n) = (-1)^{sum_{inversions (i,j)} n_i * n_j}

    where n_i is the number of fermions at site i (0, 1, or 2),
    and the sum is over pairs (i < j) with g(i) > g(j).

    Args:
        fxs: (B, N_sites) int64, quimb encoding {0,1,2,3}.
        inv_masks: (|G|, N, N) bool, from build_inversion_masks.

    Returns:
        signs: (B, |G|) float64, each entry +1.0 or -1.0.
    """
    B, N = fxs.shape
    n_group = inv_masks.shape[0]

    # Fermion count per site: (B, N)
    lookup = _QUIMB_TO_NFERMIONS.to(fxs.device)
    n_counts = lookup[fxs]  # (B, N) long

    # Outer product: n_i * n_j -> (B, N, N)
    n_pairs = n_counts.unsqueeze(2) * n_counts.unsqueeze(1)

    # For each g: exponent = sum of n_i*n_j over inversion pairs
    # inv_masks: (|G|, N, N), n_pairs: (B, N, N)
    # Result: (B, |G|)
    # Cast to float for CUDA einsum (baddbmm not impl for Long)
    exponents = torch.einsum(
        'gnm,bnm->bg',
        inv_masks.float(), n_pairs.float(),
    ).long()

    # sign = (-1)^exponent
    signs = 1.0 - 2.0 * (exponents % 2).to(torch.float64)
    return signs


# =================================================================
#  Fermionic symmetry-projected model
# =================================================================


class FermionSymmetryProjectedModel(SymmetryProjectedModel):
    """Symmetry-projected wavefunction for fermionic systems.

    Extends SymmetryProjectedModel with the fermionic sign factor:

        psi^r(n) = (1/|G|) sum_g sign(g,n) chi^(r)(g) psi(D_g^{-1} n)

    where sign(g,n) = (-1)^{sum_{i<j, g(i)>g(j)} n_i * n_j}
    accounts for reordering fermionic creation operators after
    the site permutation.

    Args:
        base_model: WavefunctionModel_GPU instance (e.g. fPEPS).
        Lx: lattice rows.
        Ly: lattice columns (must equal Lx for C4v).
        irrep: str, one of 'A1', 'A2', 'B1', 'B2'.
    """

    def __init__(self, base_model, Lx, Ly, irrep='A1'):
        super().__init__(base_model, Lx, Ly, irrep)

        # Precompute inversion masks for fermionic sign
        inv_masks = build_inversion_masks(self.perms)
        self.register_buffer('inv_masks', inv_masks)

    # ----- Single-sample interface -----

    def amplitude(self, x, params_list):
        """Single-sample projected amplitude with fermionic sign.

        Args:
            x: (N_sites,) int64, quimb encoding.
            params_list: list of parameter tensors.

        Returns:
            Scalar projected amplitude.
        """
        lookup = _QUIMB_TO_NFERMIONS.to(x.device)
        n_counts = lookup[x]  # (N,)

        total = torch.tensor(
            0.0, dtype=self.characters.dtype,
            device=x.device,
        )
        for g in range(self.n_group):
            x_g = x[self.perms[g]]
            amp_g = self.base_model.amplitude(
                x_g, params_list,
            )
            # Fermionic sign for this group element
            n_pairs = (
                n_counts.unsqueeze(1) * n_counts.unsqueeze(0)
            )
            exp_g = (
                n_pairs * self.inv_masks[g]
            ).sum()
            sign_g = 1.0 - 2.0 * (exp_g % 2).to(
                self.characters.dtype,
            )
            total = (
                total
                + sign_g * self.characters[g] * amp_g
            )
        return total / self.n_group

    def log_amplitude(self, x, params_list):
        """Single-sample log projected amplitude with sign."""
        lookup = _QUIMB_TO_NFERMIONS.to(x.device)
        n_counts = lookup[x]
        n_pairs = (
            n_counts.unsqueeze(1) * n_counts.unsqueeze(0)
        )

        signs = []
        log_abs_vals = []
        for g in range(self.n_group):
            x_g = x[self.perms[g]]
            s_g, la_g = self.base_model.log_amplitude(
                x_g, params_list,
            )
            exp_g = (n_pairs * self.inv_masks[g]).sum()
            fsign_g = 1.0 - 2.0 * (exp_g % 2).to(
                s_g.dtype,
            )
            signs.append(
                fsign_g * self.characters[g] * s_g
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
        fsigns = compute_fermion_signs(x, self.inv_masks)

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
        fsigns = compute_fermion_signs(x, self.inv_masks)

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
        fsigns = compute_fermion_signs(x, self.inv_masks)

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

        fsigns = compute_fermion_signs(x, self.inv_masks)
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
