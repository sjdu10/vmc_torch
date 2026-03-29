# Symmetry Projection for Spin VMC: Complete Theory

## 1. Setup and Notation

We consider a spin-1/2 system on an Lx x Ly square lattice with N = Lx * Ly sites. Configurations are encoded as `s = (s_0, s_1, ..., s_{N-1})` with `s_i in {0, 1}` (spin down, spin up). We use **row-major ordering**: site at grid position `(ix, iy)` has flat index

    site_index = ix * Ly + iy

where `ix in {0, ..., Lx-1}` is the row and `iy in {0, ..., Ly-1}` is the column.

The unprojected wavefunction is `psi(s; theta)`, a PEPS model evaluated via `PEPS_Model_GPU.forward()`.


## 2. Symmetry Projection (Point Group)

### 2.1 General Formula (Chen Dissertation Eq. 4.25)

For a symmetry group G acting on the lattice, the projected wavefunction in irreducible representation r is:

    psi^r(s) = (d_r / |G|) * sum_{g in G} chi^{(r)*}(g) * psi(D_g^{-1} s)

where:
- `d_r` = dimension of irrep r (1 for all 1D irreps of C4v)
- `|G|` = order of the group
- `chi^{(r)}(g)` = character of irrep r at group element g
- `D_g` = representation of g on configuration space: permutes site indices
- `D_g^{-1} s` = configuration with sites permuted by the inverse of g

**Key point**: `D_g^{-1} s` means "the configuration you get by applying g^{-1} to the site indices." Concretely, if g maps site i to site pi_g(i), then:

    (D_g s)_i = s_{pi_g^{-1}(i)}     (value at site i comes from site pi_g^{-1}(i))

equivalently:

    (D_g^{-1} s)_i = s_{pi_g(i)}      (value at site i comes from site pi_g(i))

So in practice, `D_g^{-1} s` is obtained by indexing `s` with the permutation `pi_g`:

    s_transformed[i] = s[pi_g[i]]

or in PyTorch: `s_transformed = s[perm_g]` where `perm_g[i] = pi_g(i)`.


### 2.2 The C4v Group for a Square Lattice

The C4v group has 8 elements: {e, C4, C4^2, C4^3, sigma_v, sigma_v', sigma_d, sigma_d'}.

**Convention**: We place the origin at the center of the lattice. For an Lx x Ly lattice (we require Lx = Ly = L for C4v), the center is at `((L-1)/2, (L-1)/2)`.

The 8 group elements act on grid coordinates `(ix, iy)` as follows:

| Element | Geometric action | (ix, iy) -> (ix', iy') |
|---------|-----------------|----------------------|
| e (identity) | do nothing | (ix, iy) |
| C4 (90 deg CCW) | rotate 90 deg counterclockwise | (iy, L-1-ix) |
| C4^2 (180 deg) | rotate 180 deg | (L-1-ix, L-1-iy) |
| C4^3 (270 deg CCW) | rotate 270 deg counterclockwise = 90 CW | (L-1-iy, ix) |
| sigma_v (reflect across vertical axis) | flip iy | (ix, L-1-iy) |
| sigma_v' (reflect across horizontal axis) | flip ix | (L-1-ix, iy) |
| sigma_d (reflect across main diagonal) | swap ix <-> iy | (iy, ix) |
| sigma_d' (reflect across anti-diagonal) | swap and flip | (L-1-iy, L-1-ix) |

**Derivation of rotations**: Under 90 deg CCW rotation about center `(c, c)` where `c = (L-1)/2`:
- Shift to center: `(ix - c, iy - c)`
- Rotate: `(ix - c, iy - c) -> (-(iy - c), ix - c)` = `(c - iy, ix - c)`
- Shift back: `(c - iy + c, ix - c + c)` = `(L-1-iy, ix)`

Wait, let me redo this carefully. The standard 90 deg CCW rotation matrix is:
```
[cos(90)  -sin(90)] = [0  -1]
[sin(90)   cos(90)]   [1   0]
```
So `(x, y) -> (-y, x)`. Shifting:
- `(ix - c, iy - c) -> (-(iy - c), ix - c) = (c - iy, ix - c)`
- Shift back: `(c - iy + c, ix - c + c) = (2c - iy, ix) = (L-1-iy, ix)`

So C4 (90 CCW): `(ix, iy) -> (L-1-iy, ix)`. Let me correct the table:

| Element | (ix, iy) -> (ix', iy') |
|---------|----------------------|
| e | (ix, iy) |
| C4 (90 CCW) | (L-1-iy, ix) |
| C4^2 (180) | (L-1-ix, L-1-iy) |
| C4^3 (270 CCW = 90 CW) | (iy, L-1-ix) |
| sigma_v (y-axis reflection, flips columns) | (ix, L-1-iy) |
| sigma_v' (x-axis reflection, flips rows) | (L-1-ix, iy) |
| sigma_d (main diagonal reflection) | (iy, ix) |
| sigma_d' (anti-diagonal reflection) | (L-1-iy, L-1-ix) |

**Verification of group multiplication (sanity checks)**:
- C4 * C4 should give C4^2: `(ix,iy) -C4-> (L-1-iy, ix) -C4-> (L-1-ix, L-1-iy)`. Correct, matches C4^2.
- C4^3 * C4 should give e: `(ix,iy) -C4-> (L-1-iy, ix) -C4^3-> (ix, L-1-(L-1-iy)) = (ix, iy)`. Correct.
- sigma_d * sigma_d should give e: `(ix,iy) -sigma_d-> (iy, ix) -sigma_d-> (ix, iy)`. Correct.
- C4 * sigma_v should give sigma_d or sigma_d': `(ix,iy) -sigma_v-> (ix, L-1-iy) -C4-> (L-1-(L-1-iy), ix) = (iy, ix)`. That's sigma_d. This is consistent with C4v group structure.


### 2.3 Site Permutation Maps

The permutation `pi_g` maps flat site indices. Given `(ix, iy) -> (ix', iy')` under group element g:

    pi_g(ix * L + iy) = ix' * L + iy'

**For the projected wavefunction, we need `D_g^{-1} s = s[pi_g]`**, i.e., we need the forward permutation of each group element. In PyTorch:

```python
def build_c4v_permutations(L):
    """Build the 8 site permutations for C4v on an L x L lattice.

    Returns:
        perms: dict mapping group element name to (N,) int64 tensor
               such that s_transformed = s[perm] gives D_g^{-1} s.
    """
    N = L * L
    sites = torch.arange(N)
    ix = sites // L  # row indices
    iy = sites % L   # col indices

    def flat(r, c):
        return r * L + c

    perms = {
        'e':        flat(ix, iy),                    # identity
        'C4':       flat(L-1-iy, ix),                # 90 CCW
        'C4_2':     flat(L-1-ix, L-1-iy),            # 180
        'C4_3':     flat(iy, L-1-ix),                # 270 CCW
        'sigma_v':  flat(ix, L-1-iy),                # y-axis reflection
        'sigma_vp': flat(L-1-ix, iy),                # x-axis reflection
        'sigma_d':  flat(iy, ix),                    # main diagonal
        'sigma_dp': flat(L-1-iy, L-1-ix),            # anti-diagonal
    }
    return perms
```

**Important**: These permutations `pi_g` represent where each site *goes to* under g. When we compute `s[pi_g]`, we get `(D_g^{-1} s)_i = s_{pi_g(i)}`, which is the value at the destination site pi_g(i) of the original configuration. This is exactly what appears in the projection formula.

**Verification for 2x2 lattice** (L=2, sites 0,1,2,3 at positions (0,0),(0,1),(1,0),(1,1)):
- C4: (0,0)->(1,0) site 0->2, (0,1)->(0,0) site 1->0, (1,0)->(1,1) site 2->3, (1,1)->(0,1) site 3->1. perm = [2,0,3,1].
- Check: `s[perm]` for s=[a,b,c,d] gives [c,a,d,b]. Site 0 gets value from site 2, site 1 from site 0, etc. This corresponds to "unrotating" the config, which is D_{C4}^{-1} s. Correct.


### 2.4 Character Table

| C4v | e | C4 | C4^2 | C4^3 | sigma_v | sigma_v' | sigma_d | sigma_d' |
|-----|---|----|------|------|---------|----------|---------|----------|
| A1  | 1 | 1  | 1    | 1    | 1       | 1        | 1       | 1        |
| A2  | 1 | 1  | 1    | 1    | -1      | -1       | -1      | -1       |
| B1  | 1 | -1 | 1    | -1   | 1       | 1        | -1      | -1       |
| B2  | 1 | -1 | 1    | -1   | -1      | -1       | 1       | 1        |
| E   | 2 | 0  | -2   | 0    | 0       | 0        | 0       | 0        |

Notes:
- All 1D irreps have `d_r = 1`.
- The E irrep has `d_r = 2` and is 2-dimensional, requiring partner functions. For the ground state of the Heisenberg model, the relevant irrep is typically **A1** (fully symmetric).
- The character table uses conjugacy classes: {e}, {C4, C4^3}, {C4^2}, {sigma_v, sigma_v'}, {sigma_d, sigma_d'}. For a 1D irrep, elements in the same class share the same character, but we list all 8 separately for implementation clarity.

For the **A1 irrep** (ground state of Heisenberg), all characters are +1, so:

    psi^{A1}(s) = (1/8) * sum_{g in C4v} psi(s[pi_g])

This is simply the uniform average of psi over all 8 symmetry-related configurations.


## 3. Translation Symmetry (Optional Extension)

### 3.1 Translation Group on a Periodic Lattice

For a periodic Lx x Ly lattice, the translation group G' has |G'| = Lx * Ly elements T_{tx, ty} with tx in {0,...,Lx-1}, ty in {0,...,Ly-1}.

The action on site coordinates:

    T_{tx, ty}: (ix, iy) -> ((ix + tx) mod Lx, (iy + ty) mod Ly)

The translation permutation in flat indices:

    pi_{T_{tx,ty}}(ix * Ly + iy) = ((ix + tx) mod Lx) * Ly + ((iy + ty) mod Ly)

### 3.2 Combined Point Group + Translation (Chen Eq. 4.26)

    psi^{r,k}(s) = (d_r / (|G| * |G'|)) * sum_{g in G} sum_{g' in G'} chi^{(r)*}(g) * chi^{(k)*}(g') * psi(D_{gg'}^{-1} s)

where:
- `chi^{(k)}(T_{tx,ty}) = exp(i k . t)` = `exp(i (kx * tx * 2pi/Lx + ky * ty * 2pi/Ly))`
- `k = (kx, ky)` labels the momentum sector (integers mod Lx, Ly)
- For ground state at Gamma point: `k = (0, 0)`, so all `chi^{(k)} = 1`

**At the Gamma point with A1 irrep**, the combined projection is:

    psi^{A1, Gamma}(s) = (1 / (8 * Lx * Ly)) * sum_{g in C4v} sum_{tx,ty} psi(D_{g T_{tx,ty}}^{-1} s)

This requires `8 * Lx * Ly` forward passes per configuration.

### 3.3 Practical Consideration

For a 4x4 lattice: 8 * 16 = 128 forward passes per config. For 6x6: 8 * 36 = 288. This is expensive. **Start with point group only (8 forward passes) and add translation later if needed.**


## 4. VMC with Symmetry Projection

### 4.1 Sampling

We sample from:

    p(s) proportional to |psi^r(s)|^2

The Metropolis acceptance ratio becomes:

    A(s -> s') = min(1, |psi^r(s')|^2 / |psi^r(s)|^2)

**Implementation**: In the sampler, replace `model(fxs)` with the projected amplitude. Each call to the projected amplitude requires |G| = 8 forward passes of the base model.

Specifically, for a batch of B configurations `fxs` of shape (B, N_sites):

```python
def projected_amplitude(fxs, model, perms, characters):
    """
    Args:
        fxs: (B, N_sites) int64
        model: base model with forward(x) -> (B,)
        perms: list of 8 permutation tensors, each (N_sites,)
        characters: list of 8 floats (chi^{(r)*}(g))
    Returns:
        psi_proj: (B,) projected amplitudes
    """
    psi_proj = torch.zeros(B, device=fxs.device, dtype=model_dtype)
    for perm, chi in zip(perms, characters):
        fxs_g = fxs[:, perm]        # (B, N_sites) — permuted configs
        psi_g = model(fxs_g)         # (B,) — base amplitudes
        psi_proj += chi * psi_g      # accumulate
    psi_proj /= 8  # divide by |G|; d_r = 1 for 1D irreps
    return psi_proj
```

### 4.2 Local Energy

The local energy is:

    E_loc(s) = sum_{s'} H_{ss'} * psi^r(s') / psi^r(s)

where the sum is over connected configurations from H.get_conn(s).

**Expanding**:

    E_loc(s) = sum_{s'} H_{ss'} * [sum_g chi^{(r)*}(g) psi(D_g^{-1} s')] / [sum_g chi^{(r)*}(g) psi(D_g^{-1} s)]

For each connected config s', we need to evaluate psi on all 8 symmetry-transformed versions of s'. The denominator (projected amplitude at s) should be cached from sampling.

**Implementation**: In `evaluate_energy`, after obtaining connected configs `conn_etas` of shape (total_conn, N_sites):
1. For the denominator: use the cached `psi^r(s)` from sampling (already computed as `current_amps`).
2. For the numerator: compute projected amplitudes on all connected configs.

```python
# conn_etas: (total_conn, N_sites)
# Compute projected amps for connected configs
conn_proj_amps = torch.zeros(total_conn, device=device, dtype=dtype)
for perm, chi in zip(perms, characters):
    conn_g = conn_etas[:, perm]     # (total_conn, N_sites)
    # Evaluate in chunks of size B (with padding for torch.compile)
    amps_g = batched_forward(model, conn_g, chunk_size=B)  # (total_conn,)
    conn_proj_amps += chi * amps_g
conn_proj_amps /= 8

# Local energy
terms = conn_eta_coeffs * (conn_proj_amps / current_proj_amps[batch_ids])
local_energies = torch.zeros(B, ...).index_add(0, batch_ids, terms)
```

**Cost**: 8x more forward passes for connected configs compared to unprojected. If there are ~N_conn connected configs per sample and B samples, we go from `B * N_conn` to `8 * B * N_conn` forward passes total. However, since we chunk in batches of B, the number of chunks increases by 8x.

**Optimization**: We can batch all 8 permutations together. Instead of 8 separate forward passes over (total_conn, N_sites), concatenate into one (8 * total_conn, N_sites) batch and do a single chunked evaluation, then reshape and sum with characters. This avoids Python loop overhead and can better utilize GPU parallelism.


### 4.3 Gradients (O_k for Stochastic Reconfiguration)

The log-derivative for SR is:

    O_k(s) = d/d(theta_k) ln psi^r(s)
           = [sum_g chi^{(r)*}(g) * d/d(theta_k) psi(D_g^{-1} s)] / [sum_g chi^{(r)*}(g) * psi(D_g^{-1} s)]
           = [sum_g chi^{(r)*}(g) * (d psi / d theta_k)(D_g^{-1} s)] / psi^r(s)

**Derivation**: Since `psi^r(s) = (d_r / |G|) sum_g chi^{(r)*}(g) psi(D_g^{-1} s)`, and the parameters theta only appear inside the base model psi:

    d/d(theta_k) psi^r(s) = (d_r / |G|) sum_g chi^{(r)*}(g) * (d psi / d theta_k)(D_g^{-1} s)

So:

    O_k(s) = d/d(theta_k) ln psi^r(s) = [d/d(theta_k) psi^r(s)] / psi^r(s)

This is a weighted sum of gradients of the base model at 8 transformed configurations, divided by the projected amplitude.

**Implementation with vmap(grad)**:

The existing `compute_grads_gpu` computes `d psi / d theta_k` for a batch of configs via `torch.vmap(torch.func.grad(single_sample_amp_func))`. For symmetry projection:

```python
# For each sample s, we need grads at 8 transformed configs
# fxs: (B, N_sites) — the sampled configs
# For the gradient, we differentiate psi^r(s) w.r.t. theta

def single_sample_projected_amp(x_i, params):
    """Projected amplitude for one sample."""
    amp = 0.0
    for perm, chi in zip(perms, characters):
        x_g = x_i[perm]
        amp += chi * model.vamp(x_g.unsqueeze(0), params).squeeze(0)
    return amp / |G|

# Then vmap(grad(single_sample_projected_amp)) over the batch
```

However, this puts the symmetry loop inside vmap, which may cause issues with torch.vmap (it doesn't handle Python control flow over tensors well).

**Alternative (recommended)**: Compute gradients at all 8 transformed configs separately, then combine:

```python
# For each g in G:
#   1. Transform all B configs: fxs_g = fxs[:, perm_g]  -> (B, N_sites)
#   2. Compute per-sample grads: grads_g = vmap(grad(psi))(fxs_g)  -> (B, Np)
#   3. Compute amps: amps_g = model(fxs_g)  -> (B,)
# Then combine:
#   numerator = sum_g chi_g * grads_g   (B, Np)
#   O_k = numerator / psi^r(s)          (B, Np)

grad_projected = torch.zeros(B, Np, device=device, dtype=dtype)
for perm, chi in zip(perms, characters):
    fxs_g = fxs[:, perm]
    grads_g, amps_g = compute_grads_gpu(fxs_g, model, ...)  # (B, Np), (B,)
    grad_projected += chi * grads_g   # note: grads_g is d(psi)/d(theta), NOT d(ln psi)/d(theta)

# The projected O_k (log-derivative):
O_k = grad_projected / (|G| * psi_proj[:, None])  # (B, Np)
```

where `psi_proj` is the cached projected amplitude from sampling.

**Critical subtlety**: The existing `compute_grads_gpu` with `use_log_amp=False` returns `(grads, amps)` where `grads[b, k] = d psi(s_b) / d theta_k` (the raw derivative, not the log-derivative). The log-derivative is then computed in the optimizer as `O_k = grads / amps`. For symmetry projection, we need the raw derivatives `d psi / d theta_k` at each transformed config, which is exactly what `compute_grads_gpu` returns.

**Alternative using log-amp path**: If `use_log_amp=True`, the existing code returns `d(log|psi|) / d theta_k`. This is NOT directly usable for symmetry projection because:

    d(log|psi^r|) / d theta_k ≠ sum_g chi_g * d(log|psi|)(D_g^{-1} s) / d theta_k

The log of a sum is not the sum of logs. We must use the raw amplitude gradient path.


### 4.4 The lpg Matrix for SR

In the SR (Stochastic Reconfiguration) solver, the key quantity is:

    lpg[b, k] = O_k(s_b) = d/d(theta_k) ln psi^r(s_b)

The S matrix is: `S_{kl} = <O_k^* O_l> - <O_k^*><O_l>`
The force is: `f_k = <E_loc^* O_k> - <E_loc^*><O_k>`

These formulas remain unchanged — only the definition of O_k changes. The distributed MINRES solver in `vmc_modules.py` operates on `lpg` and `local_energies`, so it needs no modification as long as we provide the correct `lpg`.


## 5. Integration with the Existing GPU Pipeline

### 5.1 What Changes

| Component | Change Required |
|-----------|----------------|
| `sampler.py` | Use projected amplitude in accept/reject ratio |
| `vmc_utils.py: evaluate_energy` | Compute projected amps for connected configs |
| `vmc_utils.py: compute_grads_gpu` | Compute grads at 8 transformed configs, combine |
| `vmc_modules.py` | No change — receives lpg and local_energies as before |
| `models/pureTNS_spin.py` | No change — base model stays the same |
| `VMC.py` | Pass symmetry info (perms, characters) to sampler/evaluator/grad |

### 5.2 Implementation Strategy: Wrapper Model

The cleanest approach is a **wrapper model** that takes a base model and applies projection:

```python
class SymmetryProjectedModel(nn.Module):
    """Wraps a base wavefunction model with symmetry projection.

    forward(fxs) returns projected amplitude (B,).
    The base model's .params is exposed for gradient computation.
    """
    def __init__(self, base_model, perms, characters):
        super().__init__()
        self.base_model = base_model
        self.perms = perms          # list of (N_sites,) int64 tensors
        self.characters = characters # list of floats
        self.group_order = len(perms)
        self.params = base_model.params  # expose for optimizer

    def forward(self, fxs):
        """Projected amplitude: (B, N_sites) -> (B,)."""
        result = torch.zeros(fxs.shape[0], device=fxs.device, dtype=...)
        for perm, chi in zip(self.perms, self.characters):
            fxs_g = fxs[:, perm]
            result += chi * self.base_model(fxs_g)
        return result / self.group_order
```

**Advantages**:
- Sampler calls `model(fxs)` and automatically gets projected amplitudes.
- No changes to sampler code at all.
- Energy evaluation also works transparently — connected configs get projected amplitudes.

**Disadvantage**:
- Gradient computation needs special handling (can't just wrap `compute_grads_gpu`).

### 5.3 Gradient Computation with Projection

For gradients, we cannot simply call `compute_grads_gpu(fxs, projected_model)` because the vmap machinery traces through `model.vamp()`. We need to either:

**(A) Make vamp projection-aware**: Define `SymmetryProjectedModel.vamp()` that loops over permutations inside the vmap-able function. This requires the permutations to be captured as constants (tensors) rather than Python-level control flow.

```python
def vamp(self, x, params):
    """Batched projected amplitude for torch.func.grad."""
    result = torch.zeros(x.shape[0], ...)
    for perm, chi in zip(self.perms, self.characters):
        x_g = x[:, perm]
        result += chi * self.base_model.vamp(x_g, params)
    return result / self.group_order
```

Since `self.perms` and `self.characters` are fixed tensors/constants, the Python for-loop unrolls at trace time, and each iteration calls `self.base_model.vamp`. This should work with `torch.vmap(torch.func.grad(...))` because the loop is over a fixed, small number of iterations (8) with no data-dependent control flow.

**(B) Compute base grads separately, then combine** (as described in Section 4.3). This is more verbose but avoids any potential vmap issues.

**Recommendation**: Start with approach (A) — it's simpler and the Python loop should unroll cleanly. Fall back to (B) if vmap has issues.


### 5.4 Computational Cost Summary

| Operation | Without projection | With C4v projection | Ratio |
|-----------|-------------------|-------------------|-------|
| Sampling (per edge, per sweep) | B forward passes | 8B forward passes | 8x |
| Energy (per sample, N_conn connected) | N_conn forward passes | 8 * N_conn forward passes | 8x |
| Gradients (per sample) | 1 grad evaluation | 8 grad evaluations | 8x |

The total wall-clock overhead depends on how well the 8x more forward/grad calls can be parallelized on GPU. Since each permuted config is independent, they can be batched together (concatenate 8 permuted versions -> 8B batch -> single forward pass -> split).


## 6. Log-Space Formulation (Numerical Stability)

For large systems, amplitudes can vary over many orders of magnitude. The projected amplitude involves a sum of terms that may partially cancel. Working in log-space:

    psi^r(s) = (1/|G|) sum_g chi_g * psi(D_g^{-1} s)

Let `a_g = log|psi(D_g^{-1} s)|` and `phi_g = sign(psi(D_g^{-1} s))` (or phase for complex psi). Then:

    psi^r(s) = (1/|G|) sum_g chi_g * phi_g * exp(a_g)

To evaluate this stably, factor out the maximum:

    a_max = max_g(a_g)
    psi^r(s) = (exp(a_max) / |G|) * sum_g chi_g * phi_g * exp(a_g - a_max)

The log-projected amplitude:

    log|psi^r(s)| = a_max + log|sum_g chi_g * phi_g * exp(a_g - a_max)| - log(|G|)
    sign(psi^r(s)) = sign(sum_g chi_g * phi_g * exp(a_g - a_max))

This requires the base model to support `forward_log(fxs) -> (signs, log_abs)`, which `PEPS_Model_GPU` already does via `WavefunctionModel_GPU.forward_log`.


## 7. Correctness Checks

### 7.1 Symmetry Verification

For any g in G: `psi^r(D_g s)` should equal `chi^{(r)}(g) * psi^r(s)` (for 1D irreps).

For A1: `psi^{A1}(D_g s) = psi^{A1}(s)` for all g. This can be verified numerically.

### 7.2 Energy Invariance

The projected energy `<E>` should be independent of which symmetry sector we project into (within the ground state sector). The variance should decrease compared to unprojected VMC.

### 7.3 Gradient Correctness

Finite-difference check: perturb `theta_k` by small epsilon, recompute projected amplitude, verify:

    [psi^r(s; theta + eps * e_k) - psi^r(s; theta - eps * e_k)] / (2 * eps) ≈ d/d(theta_k) psi^r(s)

### 7.4 Permutation Verification

For each pair of group elements g1, g2 with g1 * g2 = g3:

    perm_g3 = perm_g1[perm_g2]    (composition: first apply g2, then g1)

Wait — need to be careful about composition order. If `pi_g(i)` is where site i goes under g, then:

    pi_{g1 g2}(i) = pi_{g1}(pi_{g2}(i))

In PyTorch: `perm_g1g2 = perm_g1[perm_g2]`.

Verify all group multiplication table entries (8x8 = 64 checks, but only need to check generators C4 and sigma_v).


## 8. Summary of Formulas for Implementation

### Projected amplitude:
    psi^r(s) = (1/|G|) sum_g chi^{(r)}(g) * psi(s[pi_g])

### Local energy:
    E_loc(s) = sum_{s'} H_{ss'} * psi^r(s') / psi^r(s)

where `psi^r(s')` is computed by the same projection formula applied to each connected config s'.

### Log-derivative for SR:
    O_k(s) = [sum_g chi^{(r)}(g) * (d psi/d theta_k)(s[pi_g])] / [sum_g chi^{(r)}(g) * psi(s[pi_g])]

equivalently:
    O_k(s) = d/d(theta_k) ln psi^r(s)

### Permutation for C4v (site index i at row ix = i // L, col iy = i % L):
    pi_e(i) = i
    pi_{C4}(i) = (L-1-iy)*L + ix
    pi_{C4^2}(i) = (L-1-ix)*L + (L-1-iy)
    pi_{C4^3}(i) = iy*L + (L-1-ix)
    pi_{sigma_v}(i) = ix*L + (L-1-iy)
    pi_{sigma_v'}(i) = (L-1-ix)*L + iy
    pi_{sigma_d}(i) = iy*L + ix
    pi_{sigma_d'}(i) = (L-1-iy)*L + (L-1-ix)

### Characters for A1 (ground state):
    chi(g) = 1 for all g in C4v

### Characters for B1 (d-wave):
    chi(e) = chi(C4^2) = chi(sigma_v) = chi(sigma_v') = 1
    chi(C4) = chi(C4^3) = chi(sigma_d) = chi(sigma_d') = -1
