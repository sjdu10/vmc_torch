# Run Scripts Notebook

## 2026-03-04: Distributed minSR solver with chunked Gram matrix

### Problem
The existing `minSR_solver_gpu` gathers the full `O_loc (Ns_total, Np)` to rank 0 via `all_gather_into_tensor`. With Np~1M (float64), this is ~32 GB for 4096 samples — doesn't fit on a single GPU.

### Solution: Incremental Gram matrix
Each rank keeps `O_local (Ns_per_rank, Np)` on GPU. The Gram matrix `G = O_sk @ O_sk.T` (small: Ns x Ns) is built by iterating over param chunks of size C.

**Math**: Partition O_sk (Ns, Np) along the param axis into K = ceil(Np/C) chunks:
O_sk = [O^(1) | O^(2) | ... | O^(K)], each O^(k) is (Ns, C). Then:
G = sum_k O^(k) @ O^(k).T — exact decomposition of the inner-product sum
G_ij = sum_{p=1}^{Np} O_{ip} O_{jp}, split into K partial sums over p.
Each `G.addmm_(chunk, chunk.T)` accumulates one term. No approximation.

**Memory**: only ever hold (Ns, C) in the gather buffer instead of (Ns, Np):
- Full gather: 8 * Ns * Np bytes (32 GB for Ns=4096, Np=1M)
- Chunked: 8 * (Ns^2 + Ns * C) bytes (168 MB for C=1024)

**Communication**: K all-gather calls, each transferring 8 * Ns * C bytes.
Total bytes = same as one full gather, just spread across K rounds.

### Eager CPU offload for O_loc
The original `offload_oloc` in `VMC.py` only moved O_loc to CPU *after* `compute_grads_gpu`
returned the full `(B, Np)` matrix on GPU — OOM already happened there.
Fix: added `offload_to_cpu` param to `compute_grads_gpu` in `vmc_utils.py`. When True,
each `(B_grad, Np)` chunk is flattened and `.cpu()`'d immediately inside the grad loop.
Peak GPU mem = `O(B_grad * Np)` instead of `O(B * Np)`.

The solver (`distributed_minSR_solver_gpu`) detects CPU-resident `local_O` and streams
param chunks `(Ns_per_rank, C)` to GPU on the fly for G-build, stats, and dp reconstruction.

### Changes
1. **`vmc_utils.py`**: `compute_grads_gpu()` — added `offload_to_cpu` param for eager CPU offload.
2. **`vmc_modules.py`**: `distributed_minSR_solver_gpu()` — handles both GPU and CPU local_O.
   Streams param chunks from CPU→GPU for stats, G-build, and dp reconstruction.
3. **`VMC.py`**: `_run_sampling_phase()` passes `offload_to_cpu=offload_oloc` to `compute_grads_fn`.
   Detection logic now also checks `preconditioner.offload_oloc`.
4. **`optimizer.py`**: `DistributedMinSRGPU` — added `offload_oloc=True` attribute (default on).
5. **`vmc_run_nnfpeps_4x4.py`** and **`vmc_run_nnfpeps.py`**: Added `use_distributed_min_sr`
   and `param_chunk_size` config options with preconditioner selection logic.

### Usage
Set `use_distributed_min_sr=True` in `VMCConfig`. Default `param_chunk_size=1024`.
CPU offload is on by default (`DistributedMinSRGPU(offload_oloc=True)`).

### Verification (single GPU, 4x4 FH, D=4, chi=-1, Np=25456, Ns=64)
All tests vs reference `minSR_solver_gpu`:

**GPU-resident local_O**:
- C=32: rel_diff=1.07e-14, C=128: 6.56e-15, C=1024: 3.66e-15 — all PASS

**CPU-offloaded local_O**:
- C=32: rel_diff=1.04e-14, C=128: 6.56e-15, C=1024: 3.66e-15 — all PASS

**Gradient-only (CPU vs GPU)**: rel_diff=6.99e-16 — PASS

Test script: `run_scripts/test_distributed_minsr.py`

## 2026-03-04: Debugging vmc_run_spin_reuse warmup issues

### System
- 8x8 Heisenberg, D=4, chi=16, B=256, single GPU

### Observations
1. **0 acceptance on rows 2-5 and row 7** during x-direction sweep in warmup.
   Rows 0-1 and 6 accept normally. Statistically impossible for correct amplitudes.
2. **Row 6 is slow** (~1.35s/edge vs ~0.1s for rows 2-5).
   Likely due to different bMPS skeleton structure at `('xmax', 6)` = raw row 7
   tensors (bond dim D) vs contracted boundary MPS (bond dim chi) for other rows,
   causing torch.vmap to retrace.
3. **Energy evaluation is slow** (~3.1s per chunk, 170 chunks) because it uses
   full contraction, not the reuse path.

### Changes made
1. **sampler.py**: Added `[DBG]` prints comparing reuse vs full contraction
   amplitudes on first edge of each row with 0 acceptance. This will show
   whether `forward_reuse` returns wrong values.
2. **pureTNS_spin.py**: Added `_vamp_reuse_cache` dict to pre-cache vmapped
   functions keyed by `(direction, bMPS_key_min, bMPS_key_max)`. Avoids
   re-creating `torch.vmap` wrapper each call.
3. **vmc_run_spin_reuse.py**: Wired up `evaluate_energy_reuse` from `vmc_utils.py`
   (was already implemented for fPEPS). Passes it to `VMC_GPU` constructor via
   `evaluate_energy_fn=evaluate_energy_reuse`.

### Root cause
`equalize_norms=1.0` in `contract_boundary_opts` rescales bMPS tensors in-place
during boundary contraction, corrupting cached environments. Fix: remove
`equalize_norms` from contract_boundary_opts for reuse models.

### Row 6 / Col 6 slowdown: root cause and fix

**Observation**: With B=16, rows 2-5 take ~0.04s, but row 6 takes ~0.13s (3x slower).
Single-sample timing (no vmap) shows ALL rows take ~0.04s — slowdown is vmap-specific.

**Root cause**: `('xmax', 6)` is just raw row 7 (D=4 bonds), not a boundary-contracted MPS
(chi=16 bonds). The `contract_boundary_from_xmin_` step inside `amplitude_reuse` operates
on tiny D-bonded tensors. Under `torch.vmap`, the Python/dispatch overhead per operation
dominates over actual computation for these tiny tensors, making batched execution ~3x slower.

Same issue for `('xmin', 1)` (raw row 0, D bonds) and symmetric y-direction keys.

**bMPS skeleton shapes** (8x8 PEPS, D=4, chi=16):
- `('xmax', 2-5)`: (16, 4, 16) — chi-bonded boundary MPS
- `('xmax', 6)`: (4, 4, 4) — raw row 7, D bonds only
- `('xmin', 1)`: (4, 4, 4) — raw row 0, D bonds only

**Fix** (`pureTNS_spin.py`): Detect "raw" bMPS environments (max tensor dim < chi)
in `cache_bMPS_skeleton`. In `amplitude_reuse`, skip `contract_boundary_from_xmin_`
for rows where either env is raw. Let `cotengra` contract all 3*Ly tensors directly.

**Correctness**:
- Row 6: D²=16=chi, so boundary step was NOT truncating. Skip gives exact same result (rel_diff=4.5e-14).
- Row 1: D*chi=64>chi, so boundary step WAS truncating. Skip gives more accurate result (avoids extra truncation).

**Timing** (8x8 Heisenberg, D=4, chi=16, B=16):

| Row | Before | After | Speedup |
|-----|--------|-------|---------|
| Row 1 | 0.048s | 0.007s | 6.8x |
| Row 2-5 | ~0.04s | ~0.04s | (same) |
| Row 6 | 0.130s | 0.008s | 16x |
| Col 6 | 0.135s | 0.008s | 17x |

Scripts: `GPU/scripts/diag_row6_slowdown.py`, `GPU/scripts/verify_skip_boundary.py`

### Remaining
- `vmc_run_fpeps_reuse.py` still has `equalize_norms: 1.0` — same bug as Issue 1

## 2026-03-04: Inspecting quimb compute_x_environments tag structure

### Question
What is the tag structure of the dict returned by `compute_x_environments` on a PEPS after isel?

### Findings (4x4 PEPS, D=4, chi=8)
- Returns 8 keys: `('xmin', i)` and `('xmax', i)` for i=0..3.
- `('xmin', 0)` and `('xmax', 3)` are **empty** TensorNetworks (0 tensors) — boundary has nothing to contract.
- Each non-empty environment has exactly `Ly=4` tensors (one per column).
- **Tag accumulation**: boundary contraction merges tags from all contracted rows. E.g.:
  - `('xmin', 1)` = row 0 only: tags are `{'I0,j', 'X0', 'Yj'}` — original row-0 tags.
  - `('xmin', 2)` = rows 0+1 contracted: tags are `{'I0,j', 'X0', 'Yj', 'I1,j', 'X1'}`.
  - `('xmin', 3)` = rows 0+1+2: tags are `{'I0,j', 'X0', ..., 'I2,j', 'X2', 'Yj'}`.
  - `('xmax', 0)` = rows 1+2+3 contracted: tags are `{'I1,j', 'X1', ..., 'I3,j', 'X3', 'Yj'}`.
  - `('xmax', 2)` = row 3 only: tags are `{'I3,j', 'X3', 'Yj'}`.
- The `Yj` tag is always present (column identity preserved).
- After isel, tensors have no physical index — shapes are pure virtual bonds (e.g. `(4,4)` for corners, `(8,4,4)` for boundary-contracted MPS).
- Boundary MPS bond dimension can grow up to `max_bond=8` (visible in shapes like `(8,4,4)`).
- Script: `GPU/scripts/test_env_tags.py`
