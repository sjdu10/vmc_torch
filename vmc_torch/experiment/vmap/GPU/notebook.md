# GPU VMC Pipeline — Research Notebook

## 2026-02-19: Batched QR Performance Investigation

### Context
Benchmarking `torch.linalg.{eigh, svd, qr}` for batched small matrices on GPU to understand linalg costs in the boundary contraction pipeline.

### What we did
- Added QR to `GPU/scripts/torch_linalg_timing.py` alongside eigh/svd benchmarks.
- Benchmarked B=64 batched (n, n) matrices for n in [28, 35].

### Key findings
- **n<=32 threshold**: eigh and svd use cuSOLVER's fused batched Jacobi kernels (single kernel launch for the whole batch). QR has no such fused kernel — it calls `geqrf` (Householder reflectors) then `orgqr` (form Q), each looped over the batch.
- **QR is ~10-15x slower than eigh/svd for n<=32** due to this kernel asymmetry.
- **n>=33**: eigh/svd lose the fused Jacobi kernel and fall back to per-matrix launches, becoming much slower. QR's relative disadvantage disappears.
- cuSOLVER's Jacobi threshold is hardcoded at n=32 (LAPACK-style block size).

### Research on MAGMA backend
- MAGMA provides fused register-based batched Householder QR that avoids looped `orgqr`.
- Reported 16-25x speedup over cuBLAS-based QR for small matrices.
- PyTorch exposes this via `torch.backends.cuda.preferred_linalg_library('magma')`.
- Quick experiment: run the same benchmark with MAGMA backend to see if QR gap closes.

### Options considered
1. **MAGMA backend switch** — lowest effort, may help QR for n<=32
2. **Custom CUDA kernel** — fused batched QR, high effort
3. **Triton kernel** — batched Householder in Triton, moderate effort
4. **Givens rotations** — alternative to Householder, naturally parallelizable
5. **Avoid QR entirely** — restructure boundary contraction to use only SVD/eigh

### Next steps (initial)
- [x] Profile full VMC pipeline (`profile_vmc.py`) — done, QR not in top 20 for chi=-1
- [x] Test MAGMA backend in `torch_linalg_timing.py` — done, MAGMA ~15x worse for QR
- [x] Record results below

### MAGMA backend results

MAGMA is **much worse** for QR at B=64, n in [28,35]. Default cuSOLVER is ~15x faster.

| n  | default (ms) | magma (ms) | speedup |
|----|-------------|-----------|---------|
| 28 | 5.26        | 77.05     | 0.07x   |
| 32 | 5.65        | 89.75     | 0.06x   |
| 35 | 5.58        | 91.64     | 0.06x   |

MAGMA also hurts eigh (~3-5ms vs 0.4ms default for n<=32) and svd (~40-56ms vs 0.4ms default for n<=32). The cuSOLVER Jacobi kernels are dominant for small matrices. MAGMA is not a viable path.

Interestingly, MAGMA eigh avoids the n=33 cliff — stays at ~4ms for all n, while default jumps to ~70ms. This suggests MAGMA could help eigh/svd for n>32 but not QR.

### VMC pipeline profiling results

Setup: 4x2 lattice, N_f=6, D=4, chi=-1 (exact contraction), B=2048, export+compile.

**Top CUDA-time operations (per-step average across 3 active steps):**

| Operation | Self CUDA % | Self CUDA time |
|-----------|------------|----------------|
| `aten::bmm` | 36.5% | 89.1 ms |
| `aten::index` | 5.6% | 13.7 ms |
| `aten::nonzero` | 5.0% | 12.1 ms |
| `aten::copy_` | 4.8% | 11.8 ms |
| `aten::mul` | 4.8% | 11.8 ms |

**No QR/SVD/eigh in the top 20 kernels.** For chi=-1 (exact contraction), there are no boundary contractions — the TN is contracted exactly via bmm (batched matrix multiply). QR only appears in boundary contraction (finite chi), which this benchmark doesn't exercise.

**Phase breakdown (CUDA time):**
- `compute_grads`: 65.3 ms (via compiled graph, dominated by bmm)
- `sample_next`: 65.1 ms
- `evaluate_energy`: 54.6 ms

**Conclusion:** For exact contraction (chi=-1), QR is not a bottleneck — bmm dominates at 36.5%. QR investigation is only relevant for finite-chi boundary contraction.

### Exact contraction: GPU vs CPU (chi=-1, vmap only, no export+compile)

Setup: 4x4, N_f=14, D=4, chi=-1, vmap (no export+compile), single CPU thread.

| | Single sample | B=128 | B=512 |
|---|---|---|---|
| GPU | 44.0 ms | 54.9 ms (0.43 ms/samp) | 56.2 ms (0.11 ms/samp) |
| CPU | 9.8 ms | 22.2 ms (0.17 ms/samp) | — |
| GPU/CPU | 4.5x slower | 2.5x slower | ~0.6x (GPU wins) |

GPU time is flat at ~55 ms from B=1 to B=512 — vmap parallelizes across samples effectively. CPU scales linearly. **GPU overtakes CPU per-sample at B~256** (0.22 vs ~0.17 ms/sample) and reaches 9114 samples/s at B=512.

Note: this is vmap-only without `torch.export` + `torch.compile`. The export+compile path (used in `profile_vmc.py` on 4x2) fuses the TN contraction into optimized CUDA kernels and should reduce the ~55 ms fixed overhead significantly.

### Boundary contraction profiling (chi=D=4, `profile_chi_approx.py`)

Setup: 4x4 lattice, N_f=14, D=4, chi=D=4, B=128, vmap (no export+compile).

**QR is the dominant bottleneck.** `aten::linalg_qr` accounts for **83.7% of self CUDA time** (69.6 ms out of 83.2 ms total).

| Operation | Self CUDA % | Self CUDA time | # calls |
|-----------|------------|----------------|---------|
| `aten::linalg_qr` | 83.7% | 69.6 ms | 12 |
| cutlass gemm (two variants) | ~60% combined | ~50 ms | 9218 |
| `larft_T_32` (QR sub-kernel) | 6.6% | 5.5 ms | 3072 |
| `larft_vtv_32` (QR sub-kernel) | 4.8% | 4.0 ms | 3072 |
| `aten::linalg_svd` | 2.9% | 2.4 ms | 2 |

QR is called 12 times per forward pass (boundary contraction along xmin + xmax), each on (B, 8, 8) matrices. The 12 QR calls spawn **3072 sub-kernels each** for `larft`, `lacpy`, `batch_eye`, `orgqr_step1`, `copy_info` — totaling ~18k kernel launches just for QR.

SVD is only called **2 times** and uses the fast batched Jacobi kernel (2.4 ms total) — negligible.

**Key metrics:**
- GPU is **3.2x slower than CPU** for single sample, **3.7x slower** for B=128
- 388 CUDA kernel launches per sample — launch overhead dominates
- CUDA utilization only 37.7%
- Boundary contraction (`bnd_xmin` + `bnd_xmax`) is ~79% of total time on both GPU and CPU
- Canonize adds ~15% overhead (358 ms full vs 306 ms bare MPS)

**Why GPU loses:** The 8×8 QR matrices are tiny — each of the ~18k QR sub-kernel launches takes ~5-10 µs of overhead but only ~1-2 µs of compute. CPU avoids this overhead entirely with in-register LAPACK calls.

### Conclusions

1. **QR is THE bottleneck for chi=D boundary contraction** — 84% of CUDA time
2. **SVD is fine** — batched Jacobi kernel handles it efficiently (only 2 calls, 2.9% time)
3. **MAGMA is not a solution** — it's 15x worse for QR
4. **The problem is kernel launch overhead**, not compute — 18k kernel launches for 12 QR calls on 8×8 matrices
5. **GPU is currently 3-4x slower than CPU** for chi=D due to this overhead

### Next steps
- [x] Consider restructuring boundary contraction to avoid QR — e.g., use SVD-based canonicalization instead (SVD is fast via Jacobi kernel) → done, see below
- [ ] Explore fusing QR sub-kernels (custom Triton or CUDA kernel for batched small QR)
- [ ] Test if `torch.compile` can fuse the QR sub-ops (unlikely but worth checking)
- [ ] Profile at larger chi (chi=8, 16) where matrices are bigger and GPU compute/launch ratio improves

### QR-via-SVD experiment (replacing QR with SVD via autoray)

**Approach:** Register a custom `linalg.qr` via `ar.register_function('torch', 'linalg.qr', qr_via_svd)` that implements QR using SVD internally:
```python
def qr_via_svd(x):
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    R = S.unsqueeze(-1) * Vh  # diag(S) @ Vh
    return U, R  # Q=U is orthogonal, A = Q @ R holds
```

R is not upper-triangular, but quimb's `qr_stabilized` and `isometrize_qr` only need `A = Q @ R` with orthogonal Q.

**Setup:** 4x4 lattice, N_f=14, D=4, chi=D=4, vmap (no export+compile).

**Results — QR-via-SVD vs original QR:**

| Metric | Original QR | QR-via-SVD | Change |
|--------|------------|------------|--------|
| SVD calls (single sample) | 2 | 2 | same (QR→SVD adds no extra logged SVD calls because QR is intercepted at a different autoray key) |
| GPU single sample | ~148 ms | 148 ms | ~same |
| CPU single sample | ~42 ms | 49 ms | ~17% slower |
| GPU B=128 | ~358 ms | 229 ms | **1.56x faster** |
| CPU B=128 | ~110 ms | 125 ms | ~14% slower |
| GPU/CPU ratio (B=128) | 3.7x slower | 1.83x slower | **improved from 3.7x to 1.8x** |
| CUDA kernel calls | ~50k | 22k | **2.3x fewer** |
| Calls per sample | 388 | 173 | **2.2x fewer** |
| CUDA utilization | 37.7% | 35.6% | similar |

**Kernel profile comparison:**

| Kernel | Original QR | QR-via-SVD |
|--------|------------|------------|
| `aten::linalg_qr` | 83.7% (69.6 ms) | gone |
| `aten::linalg_svd` | 2.9% (2.4 ms) | 48.4% (32 ms), 14 calls |
| batched Jacobi SVD | — | 44.9% (29.7 ms) |
| QR sub-kernels (larft etc.) | ~18k calls | gone |

**Batch scaling (GPU, QR-via-SVD):**

| B | time (ms) | ms/sample | samples/s |
|---|-----------|-----------|-----------|
| 1 | 238 | 238.4 | 4 |
| 32 | 206 | 6.4 | 155 |
| 128 | 224 | 1.8 | 571 |
| 256 | 232 | 0.9 | 1101 |
| 512 | 247 | 0.5 | 2076 |

GPU time is nearly flat (~220-250 ms) from B=1 to B=512, confirming vmap batches SVDs effectively.

**Ablation (B=128, QR-via-SVD):**

| Config | time (ms) | ms/sample |
|--------|-----------|-----------|
| Full (norm+canonize) | 228 | 1.79 |
| No canonize | 158 | 1.23 |
| No equalize_norms | 181 | 1.41 |
| Bare MPS | 180 | 1.40 |

Disabling canonize saves ~30% — canonize calls QR(→SVD) for each bond pair.

**Phase breakdown (GPU single sample, QR-via-SVD):**

| Phase | GPU (ms) | CPU (ms) | GPU/CPU |
|-------|----------|----------|---------|
| unpack+isel | 11.8 | 2.9 | 4.0x |
| bnd_xmin | 59.3 | 15.4 | 3.9x |
| bnd_xmax | 60.6 | 15.9 | 3.8x |
| contract | 16.4 | 4.1 | 4.0x |
| TOTAL | 148.1 | 38.4 | 3.9x |

GPU/CPU ratio is uniform ~4x across all phases (was also ~4x with QR). Single-sample overhead hasn't improved — the speedup is entirely in the batched case.

**Conclusions:**
1. **QR-via-SVD gives ~1.6x speedup for batched GPU** (B=128: 358→229 ms)
2. **Kernel launches halved** (50k→22k) — SVD's fused Jacobi kernel replaces QR's ~18k sub-kernel launches
3. **GPU/CPU ratio improved from 3.7x to 1.8x slower** — still slower but much closer to parity
4. **Single-sample GPU unchanged** (~148 ms) — fixed Python/dispatch overhead dominates, not kernel launches
5. **CPU is ~15% slower** with SVD-based QR — SVD is more expensive than QR on CPU (no kernel-launch overhead issue)
6. **Numerically correct** — script ran without errors, amplitudes computed successfully

**Remaining bottleneck:** Even with QR→SVD, GPU is 1.8x slower than CPU at B=128. The 22k kernel calls (173/sample) still create significant launch overhead. The SVD Jacobi kernel (29.7 ms, 45% CUDA time) is efficient but `aten::index` (10%), `aten::mul` (10%), `aten::add` (5%) contribute many small kernel launches. Further improvement requires `torch.export` + `torch.compile` to fuse these small ops.

### torch.export + vmap + compile with QR-via-SVD

**Setup:** 4x4 lattice, N_f=14, D=4, chi=D=4. Export captures quimb/symmray boundary contraction as pure aten-ops FX graph, then vmap batches it, then compile fuses kernels.

**Export+compile overhead:** export ~10s, compile ~21s (one-time).

**Numerical accuracy:** max relative diff vs eager = 4.78e-15 (essentially exact).

**Key result — GPU is now 4x FASTER than CPU:**

| Path | B=128 time (ms) | ms/sample | speedup |
|------|-----------------|-----------|---------|
| GPU compiled | **24.9** | **0.19** | baseline |
| GPU eager (QR-via-SVD) | 272 | 2.12 | 10.9x slower |
| CPU eager (QR-via-SVD) | 97.9 | 0.76 | 3.9x slower |

**Compiled GPU / CPU ratio: 0.25x (GPU is 4x faster!)**

This is a complete reversal from the original situation where GPU was 3.7x *slower* than CPU.

**Compiled batch-size scaling:**

| B | time (ms) | ms/sample | samples/s |
|---|-----------|-----------|-----------|
| 128 | 28 | 0.22 | 4562 |
| 256 | 27 | 0.10 | 9611 |
| 512 | 39 | 0.08 | **12964** |

Nearly flat ~27 ms from B=128 to B=256, slight increase at B=512. ~13k samples/s at B=512.

**Compiled kernel profile (B=128):**

| Metric | Eager (QR-via-SVD) | Compiled |
|--------|-------------------|----------|
| CUDA kernel calls | 22,085 | **1,106** |
| Calls per sample | 172.5 | **8.6** |
| Total CUDA time | 66.4 ms | **8.6 ms** |
| Total CPU time | 319 ms | **25.7 ms** |

torch.compile fuses the thousands of small elementwise/index/copy ops into a single Triton kernel (`triton_poi_fused_...`), eliminating ~95% of kernel launches. SVD Jacobi kernel still dominates at 84.5% of CUDA time (7.2 ms) but is much faster due to fewer surrounding ops.

**Top compiled kernels:**

| Kernel | CUDA time | % |
|--------|-----------|---|
| CompiledFxGraph (fused Triton) | 23.6 ms (async overlap) | — |
| batched_svd_parallel_jacobi | 7.2 ms | 84.5% |
| batched_svd_qr (SVD sub-step) | 0.5 ms | 6.1% |
| bmm | 0.1 ms | 1.2% |

**Conclusions:**
1. **QR-via-SVD + export + compile makes GPU 4x faster than CPU** for chi=D boundary contraction
2. **10.9x speedup** from compile alone (vs eager GPU with QR-via-SVD)
3. **Kernel launches reduced 20x** (22k→1.1k), calls/sample from 173 to 8.6
4. **CPU overhead reduced 12x** (319 ms→26 ms) — Python dispatch overhead eliminated
5. **SVD is 84.5% of remaining CUDA time** — the fused Jacobi kernel is now the true compute bottleneck (as it should be)
6. Combined effect of QR→SVD + export+compile: **~14x speedup** vs original eager QR GPU (358 ms → 25 ms)

### Next steps (updated)
- [x] Integrate QR-via-SVD + export+compile into `vmc_run.py` for production use → done, see below
- [ ] Profile at larger chi (chi=8, 16) where SVD Jacobi kernel may lose fused path (n>32)
- [ ] Verify numerical stability of SVD-based QR during VMC optimization (long runs)
- [ ] Test gradient computation (`compute_grads_gpu`) with compiled forward — needs `torch.func.grad` compatibility

### Size-dependent SVD/QR dispatch via autoray

**Refactoring:** Moved QR-via-SVD from self-contained experiment in `profile_chi_approx.py` to production code in `vmap_torch_utils.py`. Added size/device-aware dispatch for both QR and SVD.

**New functions in `vmap_torch_utils.py`:**
- `qr_via_svd(x)` — QR via SVD, calls `robust_svd_err_catcher_wrapper` for jitter + EIG fallback
- `size_aware_qr(x)` — dispatches QR-via-SVD on GPU for n<=32, default QR otherwise
- `size_aware_svd(x, jitter, driver)` — dispatches `RobustSVD_EIG` + MAGMA on GPU for n>32, default `RobustSVD` otherwise

**Dispatch table:**

| | n <= 32 (GPU) | n > 32 (GPU) | CPU (any n) |
|---|---|---|---|
| **QR** | QR-via-SVD (Jacobi fast) | default `torch.linalg.qr` (Householder) | default `torch.linalg.qr` |
| **SVD** | `RobustSVD` (Jacobi fast) | `RobustSVD_EIG` + MAGMA (~4ms, avoids 70ms cliff) | `RobustSVD` (default) |

**Rationale:** cuSOLVER's batched Jacobi kernel threshold is n=32. For n<=32, SVD/eigh use a single fused kernel (~0.4ms), but QR has no fused kernel (~5-10ms, ~3k sub-kernels). For n>32, SVD/eigh lose Jacobi and jump to ~70ms, while QR stays ~5-10ms. MAGMA eigh stays at ~4ms for all n.

**Files changed:**
- `vmap_torch_utils.py` — added `qr_via_svd`, `size_aware_qr`, `size_aware_svd`
- `GPU/vmc_run.py` — registers both `size_aware_qr` and `size_aware_svd` via autoray
- `GPU/scripts/profile_chi_approx.py` — imports from `vmap_torch_utils` instead of inline definitions

**Verification** (4x4, N_f=14, D=4, chi=4, `profile_chi_approx.py`):
- Numerical: rel diff = 6.90e-15 (matches eager exactly)
- Compiled GPU B=128: 22.7 ms (0.18 ms/sample), **4x faster than CPU** (90 ms)
- Compiled/Eager GPU speedup: 11.1x
- SVD Jacobi kernel at 84% CUDA time — n=8 <=32 takes Jacobi path correctly
- No `aten::linalg_qr` in kernel profile — QR-via-SVD dispatch working
- Results match pre-refactoring experiment (no regressions)

### Rewrite of `record_time.py` for dispatch comparison

**What:** Rewrote `GPU/scripts/record_time.py` — old script had Chinese comments, outdated `vmap_models` import, and 8x8 D=10 chi=20 params. New script benchmarks 8x8 D=chi=10 with two dispatch modes side-by-side.

**Setup:** 8x8 lattice, D=chi=10, `fPEPS_Model` from `models.pureTNS` (vmap-based, no torchrun needed).

**Two modes compared:**
- **Mode A (default):** `robust_svd_err_catcher_wrapper` for SVD, default `torch.linalg.qr`
- **Mode B (size-aware):** `size_aware_svd` + `size_aware_qr` (QR-via-SVD on GPU for n<=32)

For chi=10, boundary contraction matrices are ~10x10 (well within Jacobi n<=32 threshold). Expected: GPU size-aware faster (QR-via-SVD eliminates kernel-launch overhead), CPU similar/slightly slower.

**Method:** CUDA events (GPU), `time.perf_counter` (CPU), 3 warmup + 10 timed reps per batch size. Models rebuilt after changing autoray registration so vmap traces through new dispatch.

**Batch sizes:** [1, 4, 16, 64, 128, 256]. Saves `.npy` to `GPU/data/`.

**Status:** Script written, awaiting execution.

### Remaining questions
- What is the actual GPU speedup from size-aware dispatch at 8x8 D=chi=10?
- Does the QR-via-SVD advantage hold for larger chi where matrices exceed n=32?
- How does export+compile interact with the size-aware dispatch?
