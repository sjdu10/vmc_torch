# Fermionic Sign for Symmetry Projection

## Motivation

When projecting a fermionic wavefunction onto a symmetry irrep, we compute

$$
\psi^{(r)}(n) = \frac{d_r}{|G|} \sum_{g \in G} \chi^{(r)}(g)\, \text{sign}(g, n)\, \psi(D_g^{-1}\, n)
$$

where $D_g^{-1} n$ is the permuted configuration and $\text{sign}(g, n)$ is a fermionic phase.  This document derives $\text{sign}(g, n)$ from first principles using creation operator language.

---

## Part 1: Spinless Fermions

### Setup

Consider $N$ sites labeled $0, 1, \ldots, N-1$ (0-indexed, matching the codebase). A Fock state with $k$ fermions occupying sites $r_0 < r_1 < \cdots < r_{k-1}$ is

$$
|n\rangle = c^\dagger_{r_0}\, c^\dagger_{r_1} \cdots c^\dagger_{r_{k-1}} |0\rangle
$$

The **canonical ordering convention** is that the creation operators are written in ascending site order. This convention is a choice, but once made, it must be used consistently.

### Action of a site permutation

A spatial symmetry $g$ acts on creation operators by relabeling the site index:

$$
T_g\, c^\dagger_i\, T_g^{-1} = c^\dagger_{g(i)}
$$

Therefore:

$$
T_g |n\rangle = c^\dagger_{g(r_0)}\, c^\dagger_{g(r_1)} \cdots c^\dagger_{g(r_{k-1})} |0\rangle
$$

Note: $g$ replaces each site index $r_j \to g(r_j)$, but does **not** change the position of each operator in the string. The operators are still in the order determined by the **original** site ordering $r_0 < r_1 < \cdots < r_{k-1}$.

To identify which Fock state this is, we must **reorder** the creation operators back to canonical (ascending) order. Each swap of adjacent fermionic operators picks up a factor of $-1$.

**Definition.** $\text{sign}(g, n)$ is the parity of the permutation that sorts the list $[g(r_0), g(r_1), \ldots, g(r_{k-1})]$ into ascending order.

Equivalently, it is $(-1)$ raised to the number of **inversions** among the mapped sites: pairs $(a, b)$ with $a < b$ in the original ordering but $g(r_a) > g(r_b)$.

### Example: 2x2 lattice (Sijing's example)

Sites in row-major order:

```
0  1
2  3
```

#### Example 1: 90-degree CCW rotation, config |1,0,0,1>

**Config**: $n = (1, 0, 0, 1)$. Occupied sites: $\{0, 3\}$.

$$
|n\rangle = c^\dagger_0\, c^\dagger_3\, |0\rangle
$$

**Rotation $g$** (90 deg CCW): $(x, y) \to (L-1-y, x)$ with $L = 2$.

| site | $(i_x, i_y)$ | $(L-1-i_y, i_x)$ | $g(\text{site})$ |
|------|-------------|-------------------|-------------------|
| 0    | (0, 0)      | (1, 0)            | 2                 |
| 1    | (0, 1)      | (0, 0)            | 0                 |
| 2    | (1, 0)      | (1, 1)            | 3                 |
| 3    | (1, 1)      | (0, 1)            | 1                 |

Apply $g$ to occupied sites: $[g(0), g(3)] = [2, 1]$.

To sort $[2, 1]$ into ascending order $[1, 2]$, we need **one swap**. So $\text{sign}(g, n) = (-1)^1 = -1$.

Check:
$$
T_g |n\rangle = c^\dagger_{g(0)}\, c^\dagger_{g(3)}\, |0\rangle = c^\dagger_2\, c^\dagger_1\, |0\rangle = -c^\dagger_1\, c^\dagger_2\, |0\rangle
$$

Correct.

#### Example 2: 180-degree rotation, config |1,0,1,0>

**Config**: $n = (1, 0, 1, 0)$. Occupied sites: $\{0, 2\}$.

$$
|n\rangle = c^\dagger_0\, c^\dagger_2\, |0\rangle
$$

**Rotation $g$**: 180 degrees maps site $i \to$ site $(L-1-i_x, L-1-i_y)$.

| site | $g(\text{site})$ |
|------|-------------------|
| 0    | 3                 |
| 2    | 1                 |

Apply to occupied sites: $[g(0), g(2)] = [3, 1]$.

Sort $[3, 1] \to [1, 3]$: one swap. $\text{sign} = -1$.

Check:
$$
T_g |n\rangle = c^\dagger_3\, c^\dagger_1\, |0\rangle = -c^\dagger_1\, c^\dagger_3\, |0\rangle
$$

Correct.

#### Example 3: Reflection (y-axis), config |1,1,0,0>

**Config**: $n = (1, 1, 0, 0)$. Occupied sites: $\{0, 1\}$.

$$
|n\rangle = c^\dagger_0\, c^\dagger_1\, |0\rangle
$$

**Reflection $\sigma_v$**: $(i_x, i_y) \to (i_x, L-1-i_y)$, so $0 \to 1$, $1 \to 0$.

Apply to occupied sites: $[g(0), g(1)] = [1, 0]$.

Sort $[1, 0] \to [0, 1]$: one swap. $\text{sign} = -1$.

#### Example 4: 90-degree CCW rotation, config |1,1,1,0>

**Config**: $n = (1, 1, 1, 0)$. Occupied sites: $\{0, 1, 2\}$.

Apply 90 CCW: $[g(0), g(1), g(2)] = [2, 0, 3]$.

Sort $[2, 0, 3]$: the sorted order is $[0, 2, 3]$. The sorting permutation takes $(2, 0, 3) \to (0, 2, 3)$, which is a cyclic shift of the first two elements, equivalent to one transposition. So $\text{sign} = -1$.

Alternatively, count inversions in $[2, 0, 3]$:
- $(2, 0)$: $2 > 0$, inversion. Count = 1.
- $(2, 3)$: $2 < 3$, no inversion.
- $(0, 3)$: $0 < 3$, no inversion.

Total inversions = 1. $\text{sign} = (-1)^1 = -1$.

---

## Part 2: Spinful Fermions

### Mode ordering: the symmray interleaved convention

For spinful fermions with $N$ sites, each site has two modes: spin-down ($d$) and spin-up ($u$). The codebase uses the **symmray interleaved convention**, where modes are ordered as:

$$
[d_0,\, u_0,\, d_1,\, u_1,\, \ldots,\, d_{N-1},\, u_{N-1}]
$$

That is, modes alternate down/up within each site, and sites are ordered by site index. The total number of modes is $M = 2N$.

Mode indices (0-indexed):
- **Down at site $i$**: mode index $2i$
- **Up at site $i$**: mode index $2i + 1$

A Fock state is built by applying creation operators in ascending **mode index** order:

$$
|n\rangle = c^\dagger_{m_0}\, c^\dagger_{m_1}\, \cdots\, c^\dagger_{m_{k-1}}\, |0\rangle, \qquad m_0 < m_1 < \cdots < m_{k-1}
$$

### Quimb encoding

The codebase represents site occupancy with the quimb encoding:

| quimb value | meaning           | occupied modes at site $i$  |
|-------------|-------------------|-----------------------------|
| 0           | empty             | none                        |
| 1           | spin-down only    | $d_i$ (mode $2i$)          |
| 2           | spin-up only      | $u_i$ (mode $2i+1$)        |
| 3           | doubly occupied   | $d_i$ and $u_i$ (modes $2i$ and $2i+1$) |

### Site permutation induces a mode permutation

A site permutation $g$ that maps site $i \to g(i)$ induces a mode permutation $\tilde{g}$:

$$
\tilde{g}(2i) = 2\,g(i), \qquad \tilde{g}(2i+1) = 2\,g(i) + 1
$$

That is, $g$ moves the **pair** of modes at site $i$ to site $g(i)$, preserving the down-before-up ordering within the pair. This is exactly what `build_mode_perms` in `symmetry.py` computes.

### Deriving sign(g, n) for spinful fermions

The logic is identical to the spinless case, but now at the **mode** level.

Given a Fock state $|n\rangle = c^\dagger_{m_0}\, c^\dagger_{m_1}\, \cdots\, c^\dagger_{m_{k-1}}\, |0\rangle$ with $m_0 < m_1 < \cdots < m_{k-1}$, the symmetry acts as:

$$
T_g |n\rangle = c^\dagger_{\tilde{g}(m_0)}\, c^\dagger_{\tilde{g}(m_1)}\, \cdots\, c^\dagger_{\tilde{g}(m_{k-1})}\, |0\rangle
$$

To return this to canonical order, we must sort $[\tilde{g}(m_0), \tilde{g}(m_1), \ldots, \tilde{g}(m_{k-1})]$ into ascending order.

**Definition.** $\text{sign}(g, n) = (-1)^{I}$, where $I$ is the number of inversions among the mapped mode indices:

$$
I = \#\{(a, b) : a < b \text{ and } \tilde{g}(m_a) > \tilde{g}(m_b)\}
$$

Equivalently, summing over all mode pairs (not just occupied ones):

$$
\text{sign}(g, n) = (-1)^{\sum_{i < j,\, \tilde{g}(i) > \tilde{g}(j)}\, n_i\, n_j}
$$

where $n_i \in \{0, 1\}$ is the occupation of mode $i$.

### Why cross-spin terms matter

Consider two sites $i < j$ where site $i$ has a spin-up fermion (mode $2i+1$) and site $j$ has a spin-down fermion (mode $2j$). In the symmray ordering, mode $2i+1 < 2j$ (since $i < j$), so the canonical state has $c^\dagger_{2i+1}$ to the left of $c^\dagger_{2j}$.

If $g$ maps site $i$ to a larger site and site $j$ to a smaller one, these two fermions can swap their relative ordering, producing a sign. This cross-spin contribution is correctly captured by working at the mode level.

### Worked examples

#### Example 1: 2 sites, config |0, 3>

**System**: $N = 2$ sites, quimb config $n = (0, 3)$.

Site 1 ($i = 1$) is doubly occupied. Occupied modes: $d_1 = 2$, $u_1 = 3$.

$$
|n\rangle = c^\dagger_2\, c^\dagger_3\, |0\rangle
$$

**Symmetry**: swap sites $0 \leftrightarrow 1$, i.e., $g(0) = 1$, $g(1) = 0$.

Mode permutation: $\tilde{g}(0) = 2,\; \tilde{g}(1) = 3,\; \tilde{g}(2) = 0,\; \tilde{g}(3) = 1$.

Apply to occupied modes: $[\tilde{g}(2), \tilde{g}(3)] = [0, 1]$.

Already in ascending order. Zero inversions. $\text{sign} = +1$.

**Physical check**: $T_g |n\rangle = c^\dagger_0\, c^\dagger_1\, |0\rangle = |3, 0\rangle$. The pair $(d, u)$ at site 1 maps to the pair $(d, u)$ at site 0. Since both modes move together, no relative reordering occurs.

#### Example 2: 2 sites, config |1, 2>

**System**: $N = 2$, quimb config $n = (1, 2)$.

Occupied modes: $d_0 = 0$ (down at site 0), $u_1 = 3$ (up at site 1).

$$
|n\rangle = c^\dagger_0\, c^\dagger_3\, |0\rangle
$$

**Symmetry**: swap $0 \leftrightarrow 1$.

Apply mode permutation: $[\tilde{g}(0), \tilde{g}(3)] = [2, 1]$.

Sort $[2, 1] \to [1, 2]$: one swap. $\text{sign} = -1$.

**Physical check**: $T_g |n\rangle = c^\dagger_2\, c^\dagger_1\, |0\rangle = -c^\dagger_1\, c^\dagger_2\, |0\rangle$. The down fermion from site 0 (now at mode 2) had to cross the up fermion from site 1 (now at mode 1). This is a cross-spin crossing.

#### Example 3: 2x2 lattice, config |3, 0, 1, 2>, 90-deg CCW

**System**: $N = 4$ sites on $2 \times 2$ lattice. Config $n = (3, 0, 1, 2)$.

Occupied modes:
- Site 0 ($n_0 = 3$): modes 0 ($d_0$) and 1 ($u_0$)
- Site 2 ($n_2 = 1$): mode 4 ($d_2$)
- Site 3 ($n_3 = 2$): mode 7 ($u_3$)

Canonical state: $|n\rangle = c^\dagger_0\, c^\dagger_1\, c^\dagger_4\, c^\dagger_7\, |0\rangle$.

**90 CCW rotation** (from Part 1): $g(0) = 2,\; g(1) = 0,\; g(2) = 3,\; g(3) = 1$.

Mode permutation $\tilde{g}$:

| mode | site, spin | $\tilde{g}(\text{mode})$ |
|------|-----------|--------------------------|
| 0    | $d_0$     | $2 \cdot 2 = 4$          |
| 1    | $u_0$     | $2 \cdot 2 + 1 = 5$      |
| 4    | $d_2$     | $2 \cdot 3 = 6$          |
| 7    | $u_3$     | $2 \cdot 1 + 1 = 3$      |

Apply to occupied modes: $[\tilde{g}(0), \tilde{g}(1), \tilde{g}(4), \tilde{g}(7)] = [4, 5, 6, 3]$.

Sort $[4, 5, 6, 3]$: the element $3$ must move past $4, 5, 6$, requiring 3 swaps (via bubble sort). $\text{sign} = (-1)^3 = -1$.

Alternatively, count inversions in $[4, 5, 6, 3]$:
- $(4, 3)$: inversion
- $(5, 3)$: inversion
- $(6, 3)$: inversion
- All other pairs are in order

Total inversions = 3. $\text{sign} = -1$.

#### Example 4: 2x2 lattice, config |2, 1, 0, 0>, reflection $\sigma_v$ (y-axis)

**Config**: $n = (2, 1, 0, 0)$. Occupied modes: $u_0 = 1$, $d_1 = 2$.

$$
|n\rangle = c^\dagger_1\, c^\dagger_2\, |0\rangle
$$

**Reflection** $\sigma_v$: $(i_x, i_y) \to (i_x, L-1-i_y)$, so $g(0) = 1,\; g(1) = 0$.

Mode permutation: $\tilde{g}(1) = 2 \cdot 1 + 1 = 3$, $\tilde{g}(2) = 2 \cdot 0 = 0$.

Apply: $[\tilde{g}(1), \tilde{g}(2)] = [3, 0]$.

Sort $[3, 0] \to [0, 3]$: one swap. $\text{sign} = -1$.

---

## Part 3: Algorithm (Pseudocode)

Given a batch of quimb configs `fxs` of shape `(B, N_sites)` and a site permutation `g`:

```
function compute_sign(fxs, g):
    # Step 1: Build symmray binary occupation (B, 2N)
    for each sample b and site i:
        occ[b, 2*i]     = 1 if fxs[b, i] in {1, 3} else 0    # down mode
        occ[b, 2*i + 1] = 1 if fxs[b, i] in {2, 3} else 0    # up mode

    # Step 2: Build mode permutation from site permutation
    for each mode m:
        if m is even:  mode_perm[m] = 2 * g[m // 2]        # down mode at site m//2
        if m is odd:   mode_perm[m] = 2 * g[m // 2] + 1    # up mode at site m//2

    # Step 3: Count inversions among occupied modes
    for each sample b:
        I = 0
        for each pair of modes (m1, m2) with m1 < m2:
            if occ[b, m1] == 1 and occ[b, m2] == 1:
                if mode_perm[m1] > mode_perm[m2]:
                    I += 1
        sign[b] = (-1)^I

    return sign
```

**Vectorized implementation** (as in `compute_fermion_signs` in `symmetry.py`):

1. Precompute `mode_inv_masks[g, m1, m2]` = True iff $m_1 < m_2$ and $\tilde{g}(m_1) > \tilde{g}(m_2)$.
2. Build `occ_pairs[b, m1, m2]` = `occ[b, m1] * occ[b, m2]`.
3. The exponent for group element $g$ and sample $b$ is $\sum_{m_1, m_2} \text{inv\_mask}[g, m_1, m_2] \cdot \text{occ\_pairs}[b, m_1, m_2]$.
4. Sign = $(-1)^{\text{exponent}}$.

This is computed in one `einsum('gmn,bmn->bg', ...)` call.

---

## Part 4: Consistency with `calc_phase_symmray`

The function `calc_phase_symmray` in `fermion_utils.py` computes the fermionic phase for **hopping** between two configurations that differ by a single fermion hop. It uses the same symmray interleaved mode ordering and the same sign logic, just specialized to the two-configuration case.

### What `calc_phase_symmray` does

Given two configs `configi` and `configj` that differ by a single-fermion hop:

1. Convert both configs to netket format (all spin-up sites, then all spin-down sites).
2. Re-interleave to symmray format: `[down_0, up_0, down_1, up_1, ...]`.
3. Find the two mode indices where the symmray binary configs differ: call them `m_src` (where fermion departs) and `m_dst` (where it arrives), with `m_src < m_dst`.
4. Count the number of occupied modes **between** `m_src` and `m_dst` in the initial config.
5. Phase = $(-1)^{\text{count}}$.

### Why this is the same sign logic

The hopping operator $c^\dagger_{m_\text{dst}}\, c_{m_\text{src}}$ applied to $|n\rangle$ requires anticommuting $c^\dagger_{m_\text{dst}}$ past all occupied modes between positions $m_\text{src}$ and $m_\text{dst}$ to reach its canonical position. Each occupied mode in between contributes one sign flip.

This is exactly the "inversion count" logic restricted to the special case where a permutation swaps exactly one pair of modes. In our symmetry projection, the permutation can move **all** occupied modes simultaneously, so we need the full inversion count, but the underlying principle is the same: each time two occupied fermionic modes swap their relative ordering, we pick up a factor of $-1$.

### Verifying the mode ordering matches

Both functions use the same convention:

- `calc_phase_symmray` builds `(config_i_spin_down[i], config_i_spin_up[i])` interleaved, which gives `[d_0, u_0, d_1, u_1, ...]`.
- `build_mode_perms` assigns mode $2i$ to down at site $i$ and mode $2i+1$ to up at site $i$, giving the same `[d_0, u_0, d_1, u_1, ...]`.
- `compute_fermion_signs` builds `occ` by stacking `[occ_down, occ_up]` along the last axis then reshaping, which produces `[d_0, u_0, d_1, u_1, ...]`.

All three are consistent: the symmray interleaved convention, 0-indexed, with down before up at each site.

### Explicit cross-check

Take the example from Part 2, Example 2: $N = 2$, config $|1, 2\rangle$, swap $0 \leftrightarrow 1$.

Symmray binary config: `[1, 0, 0, 1]` (modes $d_0, u_0, d_1, u_1$).

Now consider this as a "hop" scenario: the **permuted** config is $|2, 1\rangle$, symmray binary `[0, 1, 1, 0]`.

Using `calc_phase_symmray` logic: the two configs differ at modes 0, 1, 2, 3 (all four modes changed). This is actually a **two-fermion** rearrangement, not a single hop, so `calc_phase_symmray` would return 1 (the "no hopping detected" branch). The symmetry sign computation handles the general case that `calc_phase_symmray` doesn't need to.

For a true single-hop cross-check, consider $|1, 0\rangle$ and $|0, 1\rangle$ on 2 sites. Symmray binary: `[1,0,0,0]` vs `[0,0,1,0]`. The hop is from mode 0 to mode 2. Between them sits mode 1, which is unoccupied. So the phase = $(-1)^0 = +1$.

Now check via symmetry sign: the swap $g: 0 \leftrightarrow 1$ maps mode 0 to mode 2. The occupied modes are $\{0\}$, mapped to $\{2\}$. Only one occupied mode, so zero inversions. Sign = $+1$. Consistent.

---

## Summary

| Quantity | Formula |
|----------|---------|
| Occupied modes | From quimb config: mode $2i$ occupied if $n_i \in \{1, 3\}$, mode $2i+1$ if $n_i \in \{2, 3\}$ |
| Mode permutation | $\tilde{g}(2i) = 2g(i)$, $\tilde{g}(2i+1) = 2g(i)+1$ |
| Inversion count | $I = \#\{(a, b) : a < b,\; n_a = 1,\; n_b = 1,\; \tilde{g}(a) > \tilde{g}(b)\}$ (modes) |
| Fermionic sign | $\text{sign}(g, n) = (-1)^I$ |

The sign arises purely from the anticommutation of creation operators when restoring canonical (ascending mode index) order after the symmetry operation has relabeled all mode indices.
