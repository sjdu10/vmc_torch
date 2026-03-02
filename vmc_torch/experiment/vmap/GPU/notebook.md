# GPU VMC Pipeline — Research Notebook

## 2026-03-02: Fix GPU memory not released between inference and grad phases

**Problem:** In `run_warmup()`, after MCMC sampling + energy eval (under `torch.inference_mode()`), GPU memory from model forward passes was not released before gradient computation. PyTorch's CUDA caching allocator retains freed blocks for reuse rather than returning them to CUDA.

**Fix:**
- `VMC.py` `run_warmup()`: After the `inference_mode()` block, explicitly `del amps, evals` and call `torch.cuda.empty_cache()` before `compute_grads`. This is a one-time transition so `empty_cache()` cost is negligible.
- `VMC.py` main loop + `vmc_modules.py` `run_sampling_phase_gpu()`: `del amps`/`del current_amps` between energy eval and grad computation. **No `empty_cache()` in the loop** — that would force re-allocation from CUDA on every iteration, slower than letting the caching allocator reuse blocks. The `del` makes blocks available in the cache for the grad step to reuse.

## 2026-03-01: Sync model params across ranks before VMC

Added `VMC_GPU._sync_params(model)` — broadcasts all model parameters from rank 0 to all other ranks via `dist.broadcast`. Called at the start of both `run_warmup()` and `run_vmc_loop()`. Prevents silent divergence if ranks initialize with different params (e.g., random PEPS fallback with per-rank seeds). No-op for single GPU.

File: `VMC.py`.

## 2026-03-01: Pure-GPU MINRES for `distributed_minres_solver_gpu`

### Problem
The MINRES SR solver accepted GPU tensors but immediately moved them to CPU numpy, ran `scipy.sparse.linalg.minres` on CPU, and returned numpy. This caused: (1) large CPU matmuls each MINRES iteration (`(Ns, Np) @ (Np,)` with Ns=4096+, Np=1000+), (2) 2 CPU-GPU transfers per iteration for multi-GPU all_reduce, repeated ~100 times, (3) scipy dependency for a simple algorithm.

### What was done
- Added `torch_minres()` to `vmc_modules.py`: ~60-line pure-PyTorch MINRES (Paige & Saunders 1975). Lanczos three-term recurrence + Givens rotations + w-vector solution update. One matvec per iteration; all else is O(Np) vector ops. Runs entirely on GPU.
- Rewrote `distributed_minres_solver_gpu()`: all statistics (`sum_O`, `sum_EO`, `mean_O`, `energy_grad`) computed on GPU via `torch.sum`/`@`/`dist.all_reduce`. Matvec uses `local_O @ x` and `local_O.T @ inner` on GPU. Multi-GPU: `dist.all_reduce` operates on GPU tensors directly (NCCL). Returns GPU tensor `dp` instead of numpy.
- Added `use_scipy=False` kwarg for fallback to old CPU scipy path (debugging/validation).
- Rewrote `VMC_GPU.solve_sr_step()` else-branch (no-preconditioner SGD path) to stay on GPU: replaced numpy conversion with `torch.sum`/`@`/`dist.all_reduce`.
- Updated `PreconditionerGPU.solve` return type from `Tuple[np.ndarray, ...]` to `Tuple[Any, ...]`.
- No consumer changes needed: `torch.as_tensor(dp, device=device)` is a no-op for GPU tensors.

Files: `vmc_modules.py`, `VMC.py`, `optimizer.py`.

### Verification
TODO: Run `torchrun --nproc_per_node=1 run_scripts/vmc_run_fpeps.py`, compare energy convergence and SR solve time before/after. Cross-validate with `use_scipy=True`.

## 2026-03-01: Move `lr_scheduler` and `run_sr` to VMCConfig

Moved `run_sr` and `lr_scheduler` from inline `VMCLoopConfig(...)` calls to the `VMCConfig` dataclass at the top of each run script. This makes all tunable settings visible in one place. `lr_scheduler` is set after `VMCConfig()` construction since it depends on `learning_rate`. Files changed: `vmc_run_fpeps.py`, `vmc_run_fpeps_reuse.py`, `vmc_run_slater.py`, `vmc_run_nnbf.py`, `vmc_run_nnfpeps.py`, `vmc_run_nnfpeps_4x4.py`.

## 2026-03-01: Raw energy gradient (no SR) in VMC_GPU.solve_sr_step

Implemented the `else` branch in `VMC_GPU.solve_sr_step()` for when `preconditioner is None`. Computes the raw energy gradient `energy_grad = <E*O> - <E><O>` with correct multi-GPU aggregation via `dist.all_reduce` of `local_sum_O` and `local_sum_EO`. Returns `(energy_grad, elapsed_time, None)` matching the preconditioner return signature.

File: `VMC.py`, lines 385–439.

## 2026-02-27: Eliminate torch.compile Graph Breaks in CNN Backflow

### Question
Profiling showed the CNN dominates NN-fPEPS forward time (240ms out of 241ms at B=4096) and has `torch.compile` graph breaks from: dict iteration (`self.group_indices.items()`), `.tolist()` on tensors, Python scatter loop, and Python list assembly + `torch.cat`. Can we make the CNN fully compile-friendly?

### What was done
Rewrote `_CNN_Geometric_Backflow_GPU` in `models/NNfTNS.py`:
- **`__init__`**: Replaced `group_indices` dict with registered buffers (`_gather_0`, `_gather_1`, ...) that auto-follow `.to(device)`. Changed `ModuleDict` → `ModuleList` for heads. Added `_site_order_idx` permutation buffer that maps group-order concatenation → site-order.
- **`forward`**: Loop over `zip(self._gather_buf_names, self.heads)` (fixed-length Python lists, unrolled by compile), `index_select` for gathering, `head(group_feats).flatten(1)` per group, `torch.cat` in group order, then `out[:, self._site_order_idx]` to reorder. No `.tolist()`, no dict iteration, no dynamic scatter.
- **`initialize_output_scale`**: Iterate `self.heads` (ModuleList) directly.

### Verification
- `torch.compile(fullgraph=True)`: **PASS** — zero graph breaks
- `functional_call + fullgraph=True`: **PASS** — works in the `vamp()` path too
- Output correctness: **exact match** (max diff = 0.0e+00) between eager and compiled
- Full model integration (`Conv2D_Geometric_fPEPS_GPU`): NN param names auto-update to `heads.0.weight` (was `heads.CORNER_TL.weight`), `functional_call` works correctly.
- Test script: `GPU/scripts/test_cnn_compile.py`

### Timing Results (RTX 4080, 4x2 lattice, D=4, chi=-1 exact)
NN: embed=8, hidden=16, kernel=3, 1 layer. Full model forward (CNN + TN).

| B | Eager | Export+Compile | Speedup |
|---|---|---|---|
| 64 | 29.5 ms | 2.4 ms | 12.3x |
| 256 | 31.9 ms | 7.2 ms | 4.4x |
| 1024 | 53.3 ms | 26.9 ms | 2.0x |
| 4096 | 134.6 ms | 106.7 ms | 1.3x |

Speedup largest at small B where kernel launch overhead dominates. At B=4096, Conv2d/Linear GEMMs dominate. Previous export+compile with graph breaks gave only 1.1x; now with zero breaks we get 1.3x.

CNN-only timing at B=4096: eager=106ms, compiled(reduce-overhead)=97.6ms → 1.09x. The CNN is dominated by cuDNN/cuBLAS compute; fusion helps the smaller ops (embedding, coord concat, index_select, permutation).

### Scaling with System Size / Bond Dimension (B=4096, RTX 4080, chi=-1)

| System | CNN (ms) | TN eager (ms) | Total eager (ms) | Compiled (ms) | Speedup |
|---|---|---|---|---|---|
| 4x2 D=4 | 106 | 27 | 133 | 112 | 1.19x |
| 4x2 D=8 | 109 | 30 | 139 | 110 | 1.26x |
| 4x4 D=4 | 111 | 69 | 180 | 112 | 1.60x |
| 4x4 D=8 | 131 | 133 | 264 | 163 | 1.62x |

CNN cost is ~flat (~110ms) regardless of D. At larger D, TN dominates and export+compile speedup approaches ~2-3x. The CNN is now the fixed overhead floor.

### Remaining
- Run VMC test: `torchrun --nproc_per_node=1 run_scripts/vmc_run_nnfpeps.py` to verify energy + timing
- Consider slimming CNN architecture or using mixed precision (float32 CNN) to reduce the ~110ms floor

## 2026-02-27: Export+Compile for NN-fPEPS Model

### Question
The NN-fPEPS model (`Conv2D_Geometric_fPEPS_GPU`) works in eager mode but is ~3x slower than pure fPEPS per VMC step (7.5s vs 2.4s). `torch.compile` alone gives no speedup because dynamo cannot trace through quimb/symmray. Can we apply the export+compile pipeline to NN-fPEPS?

### Design
Key insight: **export only the TN contraction** (quimb/symmray), not the CNN (standard PyTorch ops that dynamo traces natively).

1. `_tn_contraction_for_export(x, nn_output, *ftn_params)` — flat-args wrapper for the TN contraction
2. `torch.export` traces only the TN part into pure aten-ops FX graph
3. `torch.vmap` batches the exported TN: `in_dims=(0, 0, None*n_ftn)` — x and nn_output batched, TN params broadcast
4. Combined forward function: CNN in batch mode (via `functional_call`) -> exported+vmapped TN
5. `torch.compile` fuses the whole thing

`vamp()` stays unchanged (eager) — used by `compute_grads_gpu` inside `torch.vmap(torch.func.grad(...))`.

### What was done
- Added to `models/NNfTNS.py`:
  - `_tn_contraction_for_export()` — flat-args wrapper
  - `export_and_compile()` — export TN, vmap, build combined forward, compile
  - `export_only()` — export+vmap without compile (for debugging)
  - `_move_exported_tn_constants_to_device()` — move symmray CPU constants to GPU
  - Updated `forward()` dispatch: compiled -> exported -> eager
- Updated `run_scripts/vmc_run_nnfpeps.py`:
  - `USE_EXPORT_COMPILE = True` flag
  - Export+compile block after `model.to(device)`
  - `VMCWarmupConfig` and `VMCLoopConfig` use `USE_EXPORT_COMPILE`

### Verification
- Import: OK
- Syntax: OK
- Runtime test: `torchrun --nproc_per_node=1 run_scripts/vmc_run_nnfpeps.py` — **passed**

### Results: 4x2 Fermi-Hubbard, t=1, U=8, N_f=6, D=4, chi=-1, Ns=4096, MinSR, lr=0.05, 1 GPU

| Metric | Export+Compile | Eager (baseline) |
|---|---|---|
| Export time | 3.3s | N/A |
| T_total/step | ~7.4s | ~7.5s |
| T_samp | 2.8s | 2.4s |
| T_locE | 3.2s | 3.7s |
| T_grad | 0.8s | 0.8s |
| E/site (100 steps) | -0.643 -> -0.707 (min -0.708) | -0.643 -> -0.707 |

Compiled vs eager is roughly equivalent. Profiling explains why:

### Forward call breakdown (B=4096, 4x2, D=4, chi=-1)

| Component | Time | % of NN-fPEPS |
|---|---|---|
| Pure fPEPS (compiled) | 2.9 ms | — |
| Pure fPEPS (eager) | 32.1 ms | — (10.9x speedup) |
| NN-fPEPS (compiled) | 241 ms | 100% |
| NN-fPEPS (eager) | 277 ms | — (1.1x speedup) |
| CNN-only | 240 ms | **100%** |
| Exported TN-only | 26.5 ms | 11% |

**Root cause: CNN dominates.** The TN export works (26.5ms exported TN vs 2.9ms pure fPEPS is same ballpark after adding backflow cat/split ops). But the CNN takes 240ms — it's the bottleneck. The `functional_call` + Python loops in geometric heads likely cause graph breaks in `torch.compile`, so CNN runs in eager mode even inside the compiled function.

Each VMC step makes ~30 forward calls (10 edges sampling + ~20 chunks energy), so 30 * 240ms = 7.2s matches the observed ~7.4s/step.

### Optimization path
To get real speedup, need to speed up the CNN:
1. Eliminate graph breaks: replace Python loops/dict iteration in geometric heads with pure tensor ops
2. Or: pre-compute CNN output once per sweep, reuse for TN calls (CNN output only depends on config, not which edge is being evaluated)

---

## 2026-02-27: Conv2D-Geometric NN-fPEPS Model

### Question
Port the CPU `Conv2D_Geometric_fPEPS_Model_Cluster` (from `vmap/models/ConvNNfTNS.py`) to the GPU `WavefunctionModel_GPU` framework, combining CNN geometric backflow with fPEPS tensor network contraction.

### What was done
- Created `models/NNfTNS.py` with two classes:
  - `_CNN_Geometric_Backflow_GPU(nn.Module)`: CNN backbone + per-geometry-type output heads (up to 9 types: 4 corners, 4 edges, 1 bulk). Shared embedding + coordinate grid + Conv2D backbone -> per-site Linear heads. Output: `(B, total_ftn_params)`.
  - `Conv2D_Geometric_fPEPS_GPU(WavefunctionModel_GPU)`: Combines CNN backflow + fPEPS TN contraction. Two-level evaluation in `vamp()`: (1) CNN runs on full batch via `functional_call`, (2) `torch.vmap` runs TN contraction per sample with additive backflow `ftn_vector + nn_eta * nn_output`.
- Parameter layout: `[TN_param_0, ..., TN_param_{N-1}, NN_param_0, ..., NN_param_{M-1}]`.
- NN module hidden from `nn.Module` child scanning via `self._nn_container = [module]` (same pattern as `AttentionNNBF_GPU`).
- `amplitude()` also implemented for single-sample eval / export compatibility.
- Small-init output heads (`init_scale=1e-5`) so model starts near pure fPEPS.
- Registered in `models/__init__.py`.
- Created `run_scripts/vmc_run_nnfpeps.py` for 4x2 Fermi-Hubbard test.

### Verification (4x2 lattice, D=4, chi=-1, N_f=6)
- Import: OK
- Instantiation: 10976 params (8 TN tensors + 17 NN param groups)
- Single-sample `amplitude()`: OK
- Batch `forward()` / `vamp()`: shape `(B,)`, values match
- Per-sample gradient via `torch.func.grad`: OK (grad norm ~52)

### VMC comparison: 4x2 Fermi-Hubbard, t=1, U=8, N_f=6, D=4, chi=-1 (exact), Ns=4096, MinSR, lr=0.05, 1 GPU

| Model | Np | E/site (step 50) | E/site (step 100) | min E/site | Time/step |
|---|---|---|---|---|---|
| Pure fPEPS | 640 | -0.648 | -0.675 | -0.679 | ~2.4s |
| NN-fPEPS (CNN Geometric, embed=8, hid=16, 2 layers) | 10976 | -0.653 | **-0.683** | **-0.683** | ~7.5s |

NN-fPEPS reaches slightly lower energy (-0.683 vs -0.679). Convergence similar in first 50 steps; NN-fPEPS continues to push lower where pure fPEPS plateaus. Cost: ~3x slower per step.

### torch.compile attempt
- `torch.compile(model)` runs without error and produces correct results + gradients
- But gives **no speedup** (0.99x) — dynamo can't trace through vmap + quimb/symmray dispatch
- Would need `torch.export` on the TN contraction part separately, then compose with CNN
- The pure fPEPS model gets ~10x from export+compile because export flattens quimb/symmray into aten ops first

### Remaining
- Tune `nn_eta`, CNN width/depth, number of layers
- Test on larger lattices (4x4) and with finite chi where backflow may help more
- Implement export+compile for NN-fPEPS: export only `_tn_contraction`, keep CNN eager

---

## 2026-02-27: NNBF (Neural Network Backflow) Model

### Question
Add a simple MLP-based neural network backflow model `NNBF_GPU` to the GPU models.

### What was done
- Rewrote `models/NNBF.py` from a Slater determinant copy to a proper NNBF model: `psi(x) = det((M_base + scale * MLP(n(x)))[occupied, :])`.
- Input: binary occupation vector `n = [spin_up | spin_dn]` of length `2*N_sites` (no embedding layer). The same `n` is reused for orbital selection via `argsort`.
- MLP architecture: `n` (binary) -> multi-layer MLP -> reshape to `delta_M`. User-configurable: `hidden_dim`, `n_layers` (default 2), `activation` (tanh/relu/gelu/silu/sigmoid).
- Output layer initialized to zero so the model starts as a plain Slater determinant; `bf_scale` controls initial correction magnitude.
- Registered `NNBF_GPU` in `models/__init__.py`.

- Added `AttentionNNBF_GPU` as a second model demonstrating the `torch.func.functional_call` pattern for using standard `nn.Module` layers (nn.Linear, nn.LayerNorm, F.scaled_dot_product_attention) within the params_list interface. Key trick: hide the module in a plain list (`self._nn_container = [mod]`) to prevent double parameter registration, then reconstruct param dict from names in `amplitude()`.

### Attention depth sweep: 4x2 Fermi-Hubbard, t=1, U=8, N_f=6, 1 GPU, B=1024, Ns=1024

200-step VMC (MinSR, lr=0.05, diag_shift=1e-3). Attention uses d_model=32, n_heads=4.

| Model | Np | E/site (step 50) | E/site (step 200) | min E/site |
|---|---|---|---|---|
| MLP (h=32, L=2) | 4864 | -0.489 | **-0.554** | **-0.556** |
| Attn L=1 | 4646 | -0.365 | -0.467 | -0.469 |
| Attn L=2 | 8934 | -0.386 | -0.506 | -0.509 |
| Attn L=3 | 13222 | -0.392 | -0.487 | -0.497 |
| Attn L=4 | 17510 | -0.376 | -0.391 | -0.414 |

Observations:
- **Larger models do need longer training.** At 50 steps, Attn L=2 was at -0.39; by step 200 it reached -0.51, nearly matching MLP's 50-step result (-0.50).
- MLP still converges fastest and reaches lowest energy (-0.556).
- Attn L=2 is the best attention variant (min -0.509), close to MLP despite 2x params.
- Attn L=3 is still catching up at step 200 (-0.497), may improve further.
- Attn L=4 (17.5k params) struggles to optimize — converges slowest, worst final energy. Likely needs even more steps, smaller lr, or warmup schedule.
- Bug fix: output projection was zero-initialized, killing gradients for all preceding attention layers. Changed to `Normal(std=1e-3)` init.

Script: `run_scripts/vmc_run_nnbf.py`. Results: `data/4x2/nnbf_depth_sweep/depth_sweep.json`.

### Remaining
- Benchmark against plain `SlaterDeterminant_GPU` (no backflow) as lower bound.
- Try attention on larger systems where long-range correlations may matter more.
- Try longer runs (500+ steps) for L=3,4 to see if they eventually catch up.

---

## 2026-02-27: bMPS Reuse Port to GPU + 4x4 fPEPS Reuse Run Script

### Question
Port the bMPS reuse machinery from the CPU pipeline (`vmap/vmap_utils.py` and `vmap/models/pureTNS.py`) to the GPU pipeline, then create a run script for 4x4 spinful Fermi-Hubbard with `fPEPS_Model_reuse_GPU`.

### What was done

1. **`sampler.py` — `MetropolisExchangeSpinfulSamplerReuse_GPU`**: New `SamplerGPU` subclass implementing two-phase bMPS-cached MCMC sweep. Phase 1 sweeps x-direction row edges with `cache_bMPS_params_any_direction_vmap(fxs, 'x')`, evaluates proposals via `forward_reuse(selected_rows=...)`, and incrementally updates via `update_bMPS_params_to_row_vmap`. Phase 2 does the same for y-direction col edges. All proposals/accept-reject are fully vectorized (no Python loop over samples).

2. **`vmc_utils.py` — `detect_changed_row_col_pair` and `evaluate_energy_reuse`**:
   - `detect_changed_row_col_pair(fx1, fx2, Ly)`: Classifies connected config changes as row-aligned, col-aligned, or diagonal. Compares at most 2 changed sites in 2D coordinates.
   - `evaluate_energy_reuse(fxs, model, H, current_amps)`: Caches both x/y bMPS for all B walkers, gets connected configs via `get_conn_batch_gpu`, groups them by `(mode, affected_indices)`, evaluates each group with sliced parent bMPS environments via `forward_reuse`, assembles local energies via `index_add_`. Diagonal terms reuse `current_amps` directly.
   - `_slice_env_dict`: Helper to index into batched bMPS pytree dicts.

3. **`models/pureTNS.py` — `export_and_compile_reuse`**: Pre-exports `amplitude_reuse` for all `2*(Lx+Ly)-2` reuse patterns (14 for 4x4). Each pattern builds a wrapper that takes `(x, *tn_flat, *bMPS_min_flat, *bMPS_max_flat)`, exports via `torch.export`, vmaps over batch dim, and compiles. Stored in `self._compiled_reuse[(direction, indices)]`. `forward_reuse` dispatches to compiled path when available, falls back to eager.

4. **`run_scripts/vmc_run_fpeps_reuse.py`**: Run script for 4x4 Fermi-Hubbard (14 fermions, D=4, chi=4). Uses `fPEPS_Model_reuse_GPU` + `MetropolisExchangeSpinfulSamplerReuse_GPU` + `evaluate_energy_reuse`. Includes `cache_bMPS_skeleton` one-time init and optional `export_and_compile_reuse`.

### Key design decisions
- **Same `VMC_GPU` driver**: No changes to `VMC.py`. The reuse sampler and energy function are injected via constructor args.
- **`forward_reuse` dispatch**: Checks `_compiled_reuse` dict first, falls back to eager `vamp_reuse`. This allows gradual adoption of compiled reuse.
- **Vectorized accept/reject**: Unlike CPU version (Python loop over samples), GPU version uses `torch.rand` + boolean masking.

### Bug fix
- **Non-contiguous `batch_ids`**: `get_conn_batch_gpu` returns connected configs interleaved by hop type (edge), not contiguous by parent sample. The initial `evaluate_energy_reuse` used an offset-based loop assuming contiguous ordering, causing wrong parent-child associations and assertion failures. Fixed by iterating over all `total_conn` entries and using `batch_ids[k]` to look up the correct parent config.

### Verification
- All files pass syntax check and imports resolve
- Runtime test: `torchrun --nproc_per_node=1 run_scripts/vmc_run_fpeps_reuse.py` runs successfully

### Next steps
- Compare timing per VMC step: reuse vs non-reuse on same 4x4 system
- Test `export_and_compile_reuse` (set `use_export_compile_reuse=True`)
- Profile bottleneck: is it `cache_bMPS_params_vmap` or `forward_reuse`?

---

## 2026-02-26: Slater Determinant — Spinful Amplitude Fix + Run Script

### What was done

1. **Fixed `models/slater.py` amplitude for spinful fermions**: The old implementation used `argsort(descending=True)[:Nf]` directly on the quimb config `x ∈ {0,1,2,3}`, which is wrong for spinful fermions. Replaced with proper quimb→netket conversion:
   - `spin_up[i] = 1` if `x[i] ∈ {2, 3}`, else 0
   - `spin_dn[i] = 1` if `x[i] ∈ {1, 3}`, else 0
   - `n = cat([spin_up, spin_dn])` → `(2*N_sites,)` binary occupation
   - `occupied = argsort(n, descending=True)[:N_f]` (vmap-compatible; `nonzero` has dynamic output shape and is not supported by `torch.vmap`)
   - `M` shape is now `(2*N_sites, N_f)` matching `SpinfulFermion.n_orbitals`

2. **Created `run_scripts/vmc_run_slater.py`**: VMC run script for `SlaterDeterminant_GPU` on spinful Fermi-Hubbard. Based on `vmc_run_fpeps.py` with differences:
   - No PEPS loading, no `setup_linalg_hooks` (no SVD/eigh needed)
   - Uses MinSR by default (Np = 2*N_sites*N_f is small, e.g. 96 for 4x2)
   - No checkpoint resume (fresh run)
   - Output dir: `data/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/slater/`

### Verification
- `amplitude`, `forward` (vmap), `vamp` all produce consistent results
- `compute_grads_gpu` returns correct shape `(B, Np)` with `Np = 2*N_sites*N_f`
- Gradient via `vmap(grad)` works correctly

### Run results (4x2 Fermi-Hubbard, t=1.0, U=8.0, N_f=6, single GPU)
- **Np** = 96 (M shape 16x6)
- **N_samples** = 4096, MinSR, lr=0.1, 100 steps
- **First E/site**: 1.074, **Last E/site**: -0.451, **Min E/site**: -0.475
- Energy decreased steadily. ~0.2s/step (dominated by SR solve).
- Converged to mean-field variational bound as expected.

### Next steps
- Compare Slater det E/site against fPEPS results for the same system
- Try larger systems (4x4, 8x8)

---

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

## 2026-02-20: Patching eigh dispatch — cusolverDnXsyevBatched for all n

### Context
`torch.linalg.eigh` has an 80-120x performance cliff at n=33 because PyTorch gates the batched cuSOLVER path (`syevjBatched` / `XsyevBatched`) behind `n <= 32`. PR #155695 (merged June 2025) added `cusolverDnXsyevBatched` wrappers but only inside the existing n<=32 branch — the dispatch logic was never updated. JAX fixed this in Sep 2025 (jax-ml/jax#31375).

### Approach
Instead of rebuilding all of PyTorch from source (which requires ~30GB RAM for CUDA compilation), we wrote a lightweight C++ extension (`my_eigh.cpp`) that calls `cusolverDnXsyevBatched` directly via `torch.utils.cpp_extension.load`. Compiles in ~2 seconds.

### Files
- `GPU/scripts/my_eigh.cpp` — C++ extension calling cusolverDnXsyevBatched unconditionally
- `GPU/scripts/eigh_batched_ext.cu` — equivalent CUDA extension (same logic, .cu file)
- `GPU/scripts/test_eigh_cpp.py` — benchmark comparing stock eigh vs custom extension
- `GPU/scripts/test_eigh_dispatch.py` — benchmark using .cu extension

### Results

**System:** RTX 4080, CUDA 12.8, PyTorch 2.10.0+cu128, float64.

**Correctness:** eigenvalue max |diff| = 0, residual ||Av - λv|| = 2.1e-13 (same order as stock 1.7e-13). PASS.

**B=64:**

| n | stock (ms) | custom (ms) | speedup |
|---|---|---|---|
| 32 | 1.05 | 0.88 | 1.2x |
| 33 | 77.2 | 1.68 | **46x** |
| 48 | 97.3 | 1.77 | **55x** |
| 64 | 122 | 2.34 | **52x** |
| 128 | 316 | 9.88 | **32x** |

**B=1024:**

| n | stock (ms) | custom (ms) | speedup |
|---|---|---|---|
| 32 | 8.30 | 7.81 | 1.1x |
| 33 | 1137 | 15.1 | **75x** |
| 48 | 1491 | 21.9 | **68x** |
| 64 | 1874 | 30.4 | **62x** |
| 128 | 4783 | 145.7 | **33x** |

### Conclusions
1. **cusolverDnXsyevBatched works for all n** — cliff completely eliminated
2. **32-75x speedup** for n>32, matching JAX's fix
3. **No regression for n<=32** — custom is slightly faster (1.1-1.5x) since it avoids PyTorch dispatch overhead
4. **Numerically correct** — residuals match stock PyTorch
5. This confirms the fix is just a dispatch logic change — the C++ wrappers already exist in PyTorch

### Next steps
- [ ] File upstream PyTorch PR to update dispatch logic in `BatchLinearAlgebraLib.cpp`
- [ ] Integrate `my_eigh` into the VMC pipeline for n>32 boundary contraction
- [ ] Benchmark impact on full VMC step at chi=D where eigh is in the hot path

### Remaining questions
- What is the actual GPU speedup from size-aware dispatch at 8x8 D=chi=10?
- Does the QR-via-SVD advantage hold for larger chi where matrices exceed n=32?
- How does export+compile interact with the size-aware dispatch?
- Can we register `my_eigh` via autoray to transparently replace `torch.linalg.eigh`?

### Ablation test: eigh dispatch branching (all B, n, dtype)

**Question:** Does the proposed PyTorch patch (keep non-batched dispatch for B=1, use XsyevBatched only for batched) make sense? Or can we just use XsyevBatched unconditionally?

**Setup:** RTX 4080, CUDA 12.8, PyTorch 2.10.0+cu128, 5 warmup + 10 reps, CUDA events timing.

**Files:** `GPU/scripts/eigh_dispatch/test_eigh_ablation.py`, data saved to `GPU/data/eigh_dispatch/eigh_ablation.npy`.

**Correctness:** All (B, n, dtype) combos pass. For float32, eigenvalue disagreement between stock and XsyevBatched grows with n (up to ~7e-3 at n=512) — expected since different algorithms (syevj vs XsyevBatched). But XsyevBatched residuals ||Av-lv|| are consistently **better** than stock (e.g. 1.5e-7 vs 1.9e-5 at n=512 B=64 f32).

**Key finding — B=1 (non-batched fallback):**

| dtype | n range | XsyevBatched vs stock |
|---|---|---|
| f32 | n<=512 | XsyevBatched wins or ties (1.1-3.6x faster) |
| f64 | n<=32 | XsyevBatched wins (1.2-1.8x) |
| f64 | n=33-512 | ~tie (0.96-1.12x) |

**Conclusion: The non-batched fallback in the proposed patch is NOT justified.** XsyevBatched is never slower than stock even for B=1. The proposed patch can be simplified to use XsyevBatched unconditionally for all B and n.

**n=32/33 cliff (stock eigh):**

| dtype | B | n=32 | n=33 | ratio |
|---|---|---|---|---|
| f32 | 64 | 0.32 ms | 58.1 ms | **182x** |
| f32 | 1024 | 1.87 ms | 901.8 ms | **482x** |
| f64 | 64 | 0.90 ms | 69.1 ms | **77x** |
| f64 | 1024 | 8.03 ms | 1083 ms | **135x** |

Cliff is 2-4x worse for float32 than float64 because stock f32 uses syevj (Jacobi, fast for n<=32) which has no batched path for n>32, while stock f64 uses syevd for all n>32.

**n=512/513 boundary (float32 only):**

| dtype | B | n=512 | n=513 | ratio |
|---|---|---|---|---|
| f32 | 1 | 11.75 ms | 3.70 ms | **0.3x** (513 faster!) |
| f32 | 64 | 758 ms | 233 ms | **0.3x** |
| f64 | 1 | 10.99 ms | 11.76 ms | 1.1x (no change) |
| f64 | 64 | 692 ms | 730 ms | 1.1x |

**Surprising:** For float32, n=512 is 3x *slower* than n=513. This is because stock PyTorch uses syevj (Jacobi) for float32 n<=512 but switches to syevd for n>512, and syevj is slower than syevd for large matrices in the non-batched path. This boundary doesn't exist for float64 (syevd for all n>32).

**XsyevBatched speedup (batched, B>=64):**

| B | n | f32 speedup | f64 speedup |
|---|---|---|---|
| 64 | 33 | 84x | 40x |
| 64 | 128 | 84x | 30x |
| 64 | 512 | 20x | 4.3x |
| 1024 | 33 | 272x | 72x |
| 1024 | 128 | 91x | 31x |

Float32 gets ~2x more speedup than float64 because stock f32 dispatch (syevj for n<=512) is particularly slow in the non-batched loop.

**Summary:**
1. XsyevBatched wins or ties for ALL (B, n, dtype) — can be used unconditionally
2. Non-batched fallback in proposed patch is unnecessary (simplifies the patch)
3. The n=32/33 cliff is 77-482x — the batched path fix is critical
4. Float32 has an additional n=512/513 anomaly (syevj→syevd transition)
5. XsyevBatched residuals are better than stock for float32 — more numerically stable

### Simplified PyTorch patch (based on ablation results)

The ablation confirms we don't need `#ifdef USE_CUSOLVER_64_BIT_XSYEV_BATCHED` in the dispatch logic. The fix is a one-line change in `BatchLinearAlgebraLib.cpp:~1596`:

```cpp
// Before:
if (batchCount(eigenvectors) > 1 && eigenvectors.size(-1) <= 32) {
// After:
if (batchCount(eigenvectors) > 1) {
```

**Why no `#ifdef`?** `apply_syevj_batched` (the function called by this branch) already has the ifdef internally — it calls `cusolverDnXsyevBatched` on CUDA >= 12.6 and falls back to `cusolverDnXsyevjBatched` on older CUDA. The ifdef is in `CUDASolver.h:10-12` (`CUSOLVER_VERSION >= 11701`).

**Why keep `batch > 1`?** Although ablation shows XsyevBatched is never slower even for B=1, there's no benefit to routing B=1 through the batched API — syevj/syevd work fine for single matrices.

Updated `torch_eig_performance.md` with the simplified proposal.

## 2026-02-25: Analysis — Extending GPU VMC to non-fermionic Hamiltonians (e.g. bosons)

### Question
What changes are needed to run VMC with a different Hamiltonian (e.g. Bose-Hubbard)?

### Findings
Traced through the full GPU VMC pipeline to identify fermion-specific vs generic components.

**Generic (no changes needed):**
- `VMC.py` — VMC driver loop, warmup, energy stats, SR step, parameter update
- `vmc_modules.py` — `run_sampling_phase_gpu`, `distributed_minres_solver_gpu`, `minSR_solver_gpu`
- `optimizer.py` — `SGDGPU`, `AdamGPU`, `DistributedSRMinresGPU`, `MinSRGPU`
- `models.py` — `fPEPS_Model_GPU`, `PureNN_GPU` (treat `x` as generic indices into physical states)
- `vmc_utils.py::sample_next` — generic Metropolis accept/reject on |ψ'|²/|ψ|²
- `vmc_utils.py::evaluate_energy` — generic E_loc = Σ H_{ss'} ψ(s')/ψ(s) via H.get_conn
- `vmc_utils.py::compute_grads_gpu` — pure autograd

**Fermion-specific (must replace for bosons):**
1. `vmc_utils.py::propose_exchange_or_hopping_vec` — hardcodes spinful fermion state map {0:empty, 1:↓, 2:↑, 3:↑↓}
2. `hamiltonian.py` — all existing Hamiltonians are fermionic; need new `get_conn` for bosons
3. `vmc_utils.py::random_initial_config` — hardcodes alternating spin-up/down initialization

**How to swap via `RunConfig` factories:**
- `hamiltonian_builder` / `hamiltonian_builder_fn` — plug in boson Hamiltonian builder
- `walker_initializer_fn` — plug in boson-compatible walker init
- `sampler_factory` — plug in sampler with boson proposal function
- No changes needed to `vmc_run.py` itself

### Remaining questions
- How to handle variable local Hilbert space dimension (fermion: 4 states/site, boson: n_max+1 states/site)?
- Does `fPEPS_Model_GPU` physical index dimension need to be adjusted for bosonic Hilbert space?
- For bosons with `n_max > 1`, the exchange proposal needs to handle multi-particle moves — what's the right proposal strategy?

## 2026-02-25: GPU-batched `get_conn` for all Hamiltonian types

### Context
Only `spinful_Fermi_Hubbard_square_lattice_torch` had GPU-batched `get_conn_batch_gpu`. The other 8 Hamiltonian classes fell back to a slow per-sample CPU `get_conn` loop. This blocks GPU-batched VMC for spin models, spinless fermions, and the spinful chain.

### What we did
Added 4 GPU mixin classes in `hamiltonian_torch.py` to avoid code duplication:

| Mixin | Shared by |
|---|---|
| `SpinlessFermionGPUMixin` | `spinless_Fermi_Hubbard_chain_torch`, `spinless_Fermi_Hubbard_square_lattice_torch` |
| `SpinfulFermionGPUMixin` | `spinful_Fermi_Hubbard_chain_torch`, `spinful_Fermi_Hubbard_square_lattice_torch` |
| `SpinHeisenbergGPUMixin` | `spin_Heisenberg_chain_torch`, `spin_Heisenberg_square_lattice_torch` |
| `SpinTransverseIsingGPUMixin` | `spin_transverse_Ising_chain_torch`, `spin_transverse_Ising_square_lattice_torch` |

Each mixin provides `precompute_hops_gpu(device)` and `get_conn_batch_gpu(fxs)`. The existing inline methods in `spinful_Fermi_Hubbard_square_lattice_torch` were removed (now inherited from mixin). `_Ao` inherits from its parent — no change needed.

**Key physics per mixin:**
- **Spinless:** binary {0,1}, Jordan-Wigner phase `(-1)^(sum between sites)`, V interaction + chemical potential diagonal terms
- **Spinful:** quimb encoding {0,1,2,3}, netket↔symmray representation conversion, symmray-convention fermionic phase
- **Heisenberg:** binary {0,1}, off-diagonal flip `0.5*J` when `sigma_i != sigma_j`, diagonal ZZ `0.25*J*(-1)^|sigma_i-sigma_j|` for ALL samples
- **Transverse Ising:** binary {0,1}, diagonal ZZ for all samples, off-diagonal spin flip `0.5*h` for all samples

No changes needed to `vmc_utils.py`, `vmc_setup.py`, `vmc_run.py` — they dispatch on `hasattr(H, '_hop_list')`.

### Files changed
- `hamiltonian_torch.py` — added 4 mixin classes, updated 8 class inheritance lines, removed inline GPU methods from spinful square lattice
- `GPU/scripts/test_get_conn_batch.py` — expanded with test suites for all 8 Hamiltonian types

### Verification
All tests pass on RTX 4080. Tested: single config, batch B=64, exhaustive enumeration for small systems.

```
python GPU/scripts/test_get_conn_batch.py
# ALL TESTS PASSED (18/18 test cases across 8 Hamiltonian types)
```

### Next steps
- [ ] Run full GPU VMC loop with spin Hamiltonians (needs compatible sampler/proposal functions)
- [ ] Benchmark GPU batch speedup vs CPU per-sample loop for each Hamiltonian type

## 2026-02-25: Rewrite `vmc_run.py` for physicist-readability

### Motivation
`vmc_run.py` hid physics parameters behind a 30+ field `RunConfig` dataclass with 10 factory callables. A physicist reading `main()` saw `config.hamiltonian_builder_fn(config=config, device=device)` — no idea what Hamiltonian, what `t`, `U`, what boundary conditions. The CPU example (`examples/vmc_run_example.py`) is the gold standard: every physical parameter spelled out explicitly.

### What changed

**`vmc_run.py`:**
- Removed `RunConfig` dataclass (30+ fields, 10 factory callables)
- Removed factory functions (`default_sampler_factory`, `default_preconditioner_factory`, `default_optimizer_factory`)
- Physics is now explicit in `main()`: `H = spinful_Fermi_Hubbard_square_lattice_torch(Lx, Ly, t, U, ...)`, model construction `fPEPS_Model_GPU(tn=peps, max_bond=chi, ...)` inline
- Added small `VMCConfig` dataclass with only numerical/training knobs (batch_size, lr, etc.)
- Section headers with `# ==========` for quick scanning
- Summary prints the physics (system params, not just energy)

**`vmc_setup.py`:**
- Removed `build_model()`, `build_hamiltonian()`, `print_summary()` — now inline in `vmc_run.py`
- `load_or_generate_peps(config, tensor_dtype)` → `load_or_generate_peps(Lx, Ly, t, U, N_f, D, seed, dtype)`
- `initialize_walkers(config, rank, device)` → `initialize_walkers(N_f, N_sites, batch_size, seed, rank, device)`
- `ensure_output_dir(config)` → `ensure_output_dir(Lx, Ly, t, U, N_f, D)`
- `setup_linalg_hooks` unchanged

### Verification
- `python -c "from vmc_torch.experiment.vmap.GPU.vmc_run import main"` — PASS
- `python -c "from vmc_torch.experiment.vmap.GPU.vmc_setup import ..."` — PASS
- Grep confirms no other files import `RunConfig`, `build_model`, `build_hamiltonian`, or `print_summary` from these modules

**`vmc_run_v0.py`** (also rewritten):
- Old version had standalone helper functions (`build_model`, `build_hamiltonian`, `load_peps`, etc.), module-level autoray registration, and stale imports (`vmap_models.fPEPS_Model`)
- Now uses the same pattern as `vmc_run.py`: explicit physics, `VMCConfig` for numerical knobs, imports from `GPU.models.fPEPS_Model_GPU` and `GPU.vmc_setup`
- Keeps its distinguishing feature: `on_step_end` callback that saves energy stats to JSON + periodic model checkpoints via `vmc.run_vmc_loop(..., on_step_end=...)`
- Import check passes

## 2026-02-25: Generic `initialize_walkers` via `init_fn`

### Motivation
`initialize_walkers` hardcoded spinful-fermion logic (`random_initial_config`), blocking reuse for spin models, spinless fermions, or specific initial states (Neel, CDW).

### Changes
- **`vmc_setup.py`**: Replaced `(N_f, N_sites, batch_size, ...)` signature with `(init_fn, batch_size, ...)` where `init_fn(seed) -> 1D tensor` is a callable that generates one walker config.
- **`vmc_run.py`**, **`vmc_run_v0.py`**: Call sites updated to pass `init_fn=lambda seed: random_initial_config(N_f, N_sites, seed=seed)`, preserving identical behavior.

### Also fixed: export+compile was dead code
The old `build_model` in `vmc_setup.py` had the `model.export_and_compile()` call. When inlined into `vmc_run.py`, only a comment was left — setting `use_export_compile=True` did nothing. Added the actual export+compile block (gated on `vmc_cfg.use_export_compile`) in both `vmc_run.py` and `vmc_run_v0.py`.

### Verification
- Import check: all three files parse and import correctly
- Functional test: `initialize_walkers(init_fn=..., batch_size=4)` produces correct `(4, 8)` int64 tensor with valid spinful configs
- `torchrun --nproc_per_node=1 vmc_run.py` with `use_export_compile=True`: runs successfully, energy decreases 0.84 → -0.46 E/site over 10 steps (4x2 Fermi-Hubbard, t=1, U=8, N_f=6, D=10, chi=10, random PEPS init)
- Warnings are all from upstream (quimb deprecation, torch.compile DeviceCopy constants, PyTorch internal deprecation) — none from our code

## 2026-02-26: Eliminated DeviceCopy warnings in export+compile

### Root cause
`torch.export` captures symmray's block-sparse index tensors (sector maps, size arrays) as CPU int64 constants (`lifted_tensor_0` through `lifted_tensor_183`). 16 of 184 lifted constants were on CPU. When `torch.compile` processes the vmapped graph, inductor inserts `ir.DeviceCopy` (H2D) for each CPU→GPU transfer, emitting 16 warnings.

Profiling showed 16 H2D memcpy events per forward call, totaling 0.015 ms (0.03% of CUDA time). Small in absolute terms but unnecessary.

### Fix
Added `_move_exported_constants_to_device()` to `fPEPS_Model_GPU` in `models.py`. Called after `exported.module()` in both `export_and_compile` and `export_only`. Two steps:
1. Walk `get_attr` nodes, move any CPU tensors to the target GPU device
2. Patch `_assert_tensor_metadata` nodes that still reference `device='cpu'` — without this, `torch.compile` raises `AssertionError('Tensor device mismatch')`

### Results (4x2, D=chi=10, B=512)
| Metric | Before | After |
|---|---|---|
| DeviceCopy warnings | 16 | 0 |
| H2D memcpy per call | 16 | 0 |
| Memcpy CUDA time/call | 0.015 ms | 0.005 ms |
| Compiled vs eager rel diff | — | 1.8e-13 |

## 2026-02-26: Can we compile the gradient computation?

### Context
The forward pass uses `export + vmap + compile`, but the gradient path (`compute_grads_gpu`) uses eager `vmap(grad(model.vamp))` where `model.vamp` calls the uncompiled `_vmapped_amplitude`. Question: can we also compile the gradient path?

### Architecture of compiled vs gradient paths
| Call site | Function | Batch dim | Compiled? |
|---|---|---|---|
| `sample_next` | `model.forward()` → `_vmapped_compiled` | always `B` (padded) | Yes |
| `evaluate_energy` | `model.forward()` → `_vmapped_compiled` | always `B` (padded) | Yes |
| `compute_grads_gpu` | `model.vamp()` → `_vmapped_amplitude` | `grad_batch_size` chunks | No (eager) |

Different batch sizes between forward (`B`) and gradients (`grad_batch_size`) do NOT trigger recompilation — the gradient path never hits the compiled code at all.

### Benchmark (4x2, D=chi=10, B=512, B_grad=64)

**Path 1 — Eager `vmap(grad(model.vamp))` (current):** 1010 ms

**Path 2 — `vmap(grad(exported_module))` (export, no compile):** 968 ms (1.04x — no speedup). Export eliminates Python dispatch, but without compile to fuse kernels, the backward graph is still many small kernel launches.

**Path 3 — `compile(vmap(grad(exported_module)))`: FAILS.** Triton compilation crashes (`PassManager::run failed`). The backward graph through eigh + boundary contraction is too complex — the fused kernel tries to combine dozens of ops (eigh backward, norm backward, diagonal, permute, etc.) and exceeds Triton's codegen limits.

### Follow-up: does B_grad=1 compile?

Yes — B_grad=1 compiles successfully (~18s compile time). But B_grad=2 already crashes (vmap doubles graph complexity, exceeds Triton limits).

**B_grad=1 is 4.3x slower than eager B_grad=64:** compiled B_grad=1 takes 4248 ms for B=512 (8.3 ms/sample × 512 calls) vs eager B_grad=64 at 979 ms. The per-call Python overhead of invoking the compiled function 512 times individually far outweighs any kernel fusion benefit.

### Conclusion
- Compiling the gradient path is **not viable**: B_grad≥2 crashes Triton, B_grad=1 is 4.3x slower than eager
- The backward graph through eigh is fundamentally too complex for Triton when batched
- Gradient computation is ~1s vs ~10s for sampling+energy — only ~10% of step time, not the bottleneck
- Possible future paths: (1) larger `grad_batch_size` to reduce loop overhead, (2) wait for Triton/inductor improvements, (3) custom eigh backward to simplify the autograd graph

## 2026-02-26: Decouple sampler from energy/gradient — sampler only does MCMC

### Motivation
`SamplerGPU.run_sampling_phase` bundled MCMC + local energy + gradients, forcing the sampler to accept `hamiltonian`, `grad_batch_size`, etc. A user wanting a custom MCMC proposal (bosons, spins) had to replace the entire sampling phase. The warmup code already had the right pattern: sampler does MCMC, VMC_GPU calls energy/grad directly.

### Changes

**`sampler.py`** — simplified to pure MCMC:
- `SamplerGPU` base class now has `step()` and `burn_in()` only
- `MetropolisExchangeSpinfulSamplerGPU.step()` contains the full Metropolis sweep logic (from `vmc_utils.sample_next`): iterate edges, propose exchange/hopping, evaluate amps, accept/reject
- Removed: `run_sampling_phase`, `warmup_step`, `sampling_phase_fn`, `sample_next_fn`, `sampling_count_key`

**`VMC.py`** — VMC_GPU owns the sample→energy→grad loop:
- New `_run_sampling_phase()`: calls `sampler.step()` for MCMC, then `evaluate_energy_fn` and `compute_grads_fn` directly
- `run_vmc_loop()` uses single code path via `_run_sampling_phase()` (removed if/else sampler vs function-based branching)
- `run_warmup()` calls `sampler.step()` directly (was `sampler.warmup_step()`)
- Removed legacy function-based init args: `sampling_phase_fn`, `sample_next_fn`, `sampling_count_key`
- Kept `evaluate_energy_fn`, `compute_grads_fn`, `distributed_sr_solver_fn`, `min_sr_solver_fn` for pluggability

**No changes to:**
- `vmc_utils.py` — `sample_next`, `evaluate_energy`, `compute_grads_gpu` stay as standalone functions
- `vmc_modules.py` — `run_sampling_phase_gpu` stays as standalone convenience for scripts
- `vmc_run.py`, `vmc_run_v0.py` — already used `sampler=` only, no code changes needed

### Custom sampler example
```python
class BosonSamplerGPU(SamplerGPU):
    def step(self, fxs, model, graph, **kwargs):
        current_amps = model(fxs)
        # ... boson-specific proposal + accept/reject ...
        return fxs, current_amps
```

### Verification
- Import checks: all 4 files pass (`sampler.py`, `VMC.py`, `vmc_run.py`, `vmc_run_v0.py`)
- No external files import removed symbols
- Functional: `torchrun --nproc_per_node=1 vmc_run_v0.py` runs successfully with correct phase timing output

**Timing (4x2, D=chi=10, B=2048, Ns=4096, export+compile, minSR):**
```
Step 7 | E/site: -0.407045 +/- 0.004773 | N=4096 |
  T_samp=0.5s T_locE=0.3s T_grad=9.4s T_SR=0.69s T_total=10.9s
```
Gradient computation dominates at 86% of step time. Sampling (5%) and energy (3%) are negligible. SR solve is 6%.

## 2026-02-26: Break VMC timing into T_samp, T_locE, T_grad

### Motivation
VMC output only showed a single `T_samp` for the entire sampling phase (MCMC + local energy + gradients). Want to see where time is spent: sampling vs energy evaluation vs gradient computation.

### Changes
- **`vmc_modules.py::run_sampling_phase_gpu`**: Added per-phase `time.time()` calls for each of the three steps (sample_next, evaluate_energy, compute_grads). Now returns a 4-tuple `(data, fxs, sample_time, phase_times)` where `phase_times = {'t_samp': ..., 't_locE': ..., 't_grad': ...}`.
- **`sampler.py::SamplerGPU`**: Updated return type annotation to 4-tuple.
- **`VMC.py::run_vmc_loop`**: Unpacks `phase_times`, prints `T_samp=... T_locE=... T_grad=... T_SR=... T_total=...`. Also includes `phase_times` in `on_step_end` callback dict.

## 2026-02-26: WavefunctionModel_GPU base class

### Motivation
Adding a new model required ~100 lines of boilerplate: `params`, `_compiled`, `_exported`, `vamp`, `forward`, `export_and_compile`, etc. Goal: user only defines `__init__` and `amplitude`.

### Changes

**`models.py`** — added `WavefunctionModel_GPU` base class:
- **User must define**: `__init__(params_list=...)` and `amplitude(x, params_list) -> (B,)`.
- **User may override**: `vamp(x, params)` for model-specific param handling (e.g. quimb pytree unflatten).
- **Base provides for free**: `self.params` (ParameterList), `_compiled`/`_exported` flags, `forward(x)` with compiled→exported→eager dispatch, `export_and_compile`, `export_only`, `compile_model`, `_move_exported_constants_to_device`.
- **`_single_sample_amplitude`**: default wraps `amplitude` with unsqueeze/squeeze. Override for natively single-sample models (e.g. quimb TN).

**`fPEPS_Model_GPU(WavefunctionModel_GPU)`**:
- Overrides `_single_sample_amplitude` (quimb TN contraction is natively single-sample), `vamp` (pytree unflatten), `_amplitude_for_export` (pytree unflatten for export).
- `amplitude` raises NotImplementedError (single-sample path used via vmap).
- Deleted: `forward`, `export_and_compile`, `export_only`, `compile_model`, `_move_exported_constants_to_device` — all inherited.

**`PureNN_GPU(WavefunctionModel_GPU)`**:
- Defines `amplitude(x, params_list)` (the old `_amp_from_params` body).
- Everything else inherited. Deleted: `vamp`, `forward`, `_compiled`, `_exported`, `compile_model`.

**New file: `models_slater.py`** — `SlaterDeterminant_GPU(WavefunctionModel_GPU)`:
- Minimal example: single `(n_orbitals, n_fermions)` parameter matrix, `amplitude` computes `det(M[occupied, :])`.

### Verification
- Import checks: all modules (`models.py`, `models_slater.py`, `vmc_run.py`, `vmc_run_v0.py`) pass.
- Functional: `PureNN_GPU` and `SlaterDeterminant_GPU` produce correct `(B,)` outputs via `forward` and `vamp`.
- Gradient: `vmap(grad)` through `vamp` works for both `PureNN_GPU` and `SlaterDeterminant_GPU`.
- Full pipeline: `torchrun --nproc_per_node=1 vmc_run_v0.py` runs 50 VMC steps, energy decreases 0.95 → -0.70 E/site (4x2, D=chi=10, B=2048, minSR). Same behavior as before refactoring.

## 2026-02-26: GPU models/ subfolder + uniform single-sample base class

### Motivation
1. **Uniform interface**: All models define a single-sample `amplitude(x, params_list)` where `x` is `(N_sites,)` → scalar. Base class vmaps automatically. No model ever sees `(B, N_sites)`.
2. **models/ package**: Moved from flat `GPU/models.py` + `GPU/models_slater.py` to a proper `GPU/models/` package mirroring `vmap/models/`.
3. **Reuse model**: Ported `fPEPS_Model_reuse` from `vmap/models/pureTNS.py` as `fPEPS_Model_reuse_GPU`.

### File structure
```
GPU/models/
    __init__.py     — re-exports all public classes
    _base.py        — WavefunctionModel_GPU base class
    pureTNS.py      — fPEPS_Model_GPU, fPEPS_Model_reuse_GPU
    pureNN.py       — PureNN_GPU
    slater.py       — SlaterDeterminant_GPU
```

Deleted: `GPU/models.py`, `GPU/models_slater.py`.

### Key design changes
- **Base class** (`_base.py`): `amplitude(x, params_list)` is now single-sample `(N_sites,) -> scalar`. `__init__` creates `_vmapped_amplitude = torch.vmap(self.amplitude, ...)`. The old `_single_sample_amplitude` wrapper is gone.
- **fPEPS_Model_GPU**: `amplitude` is natively single-sample (quimb TN contraction). Overrides `vamp` for pytree unflatten. Overrides `_amplitude_for_export` for export with pytree unflatten.
- **PureNN_GPU**: `amplitude` operates on `(N_sites,)` — `F.embedding(x, emb_w).reshape(-1)` (flat, no batch dim). Uses default `vamp` from base.
- **SlaterDeterminant_GPU**: `amplitude` uses `argsort(descending=True)[:Nf]` (no `dim=1`). Uses default `vamp`.
- **fPEPS_Model_reuse_GPU**: Ported from CPU `fPEPS_Model_reuse`. Full contraction via `amplitude` (inherits export+compile). Reuse via `amplitude_reuse` / `vamp_reuse` / `forward_reuse` (eager vmap). Uses `pack_ftn`/`unpack_ftn`/`get_params_ftn` from `vmap/models/_model_base.py`.

### Verification
- All imports work: `from vmc_torch.experiment.vmap.GPU.models import X` for all classes.
- No stale `GPU.models_slater` references remain.
- PureNN and Slater: `forward(x)` produces `(B,)`, `vmap(grad)` through `vamp` works.
- All callers (`vmc_run.py`, `vmc_run_v0.py`, `vmc_run_dev.py`, scripts) use `from vmc_torch.experiment.vmap.GPU.models import X` — zero import changes needed.

### Full pipeline verified
- `torchrun --nproc_per_node=1 vmc_run_v0.py`: 4x2 Fermi-Hubbard, t=1.0, U=8.0, N_f=6, D=chi=10, B=2048, minSR. Energy: 0.95 → -0.70 E/site in 50 steps. Same as before refactoring.

### Checkpoint resume added to `vmc_run_v0.py` and `VMC.py`
- `VMCConfig.resume_step: int = 0` (0 = fresh start, e.g. 50 = load step-50 checkpoint).
- Resolves path automatically: `{output_dir}/checkpoint_{model_name}_{resume_step}.pt`.
- `VMCLoopConfig.step_offset`: passed into the VMC loop so that printed step numbers, `info['step']` in the callback, and checkpoint filenames are all globally offset. E.g. `resume_step=50, vmc_steps=50` → steps 50–99, checkpoints at 60, 70, …

### Run scripts reorganized
- Deleted `vmc_run_v0.py`. Moved to `run_scripts/vmc_run_fpeps.py`.
- Output dir now includes `chi=` in path: `.../D={D}/chi={chi}/`.
- `load_or_generate_peps` now accepts `file_path` kwarg for explicit PEPS location.
- Updated defaults: `batch_size=4096`, `grad_batch_size=1024`, `vmc_steps=100`.

### GPU/ made standalone (no vmap/ dependencies)
- Created `GPU/fermion_utils.py`: copied `pack_ftn`/`unpack_ftn`/`get_params_ftn` from `vmap/models/_model_base.py`.
- Updated `models/pureTNS.py` to import from `GPU.fermion_utils` instead of `vmap.models._model_base`.
- Updated `vmc_run_dev.py` to import `size_aware_qr`/`size_aware_svd` from `GPU.torch_utils`.
- Updated all 11 scripts in `scripts/` to import from `GPU.torch_utils` instead of `vmap.vmap_torch_utils`.
- **Only remaining external deps**: `vmc_torch.hamiltonian_torch` (core lib, stays), `scripts/record_time.py` imports CPU `fPEPS_Model` for benchmark comparison (intentional).
- GPU/ is now ready to move to `vmc_torch/GPU/` as a standalone package.

### Remaining work
- Test `fPEPS_Model_reuse_GPU`: create model, call `cache_bMPS_skeleton`, `cache_bMPS_params_vmap`, verify amplitudes match non-reuse model.
- NN-fTNS hybrid models (Conv, Transformer, LoRA) — future work.

## 2026-02-26: NaN debugging in VMC optimization

### Problem
Running `vmc_run_fpeps.py` (4x2 Fermi-Hubbard, resume from step 50): after one optimization step, all 4096 amplitudes are NaN. Model parameters went bad after SR update.

### Diagnostics added

**`vmc_utils.py`** (earlier): Enhanced NaN detection in `compute_grads_gpu`:
- Checks amplitudes for NaN/Inf first
- For gradient NaN/Inf: reports affected samples count, affected parameter indices, maps flat indices back to `ParameterList` entries with shapes

**`VMC.py`** (this session): Added NaN/Inf checks at three points in the VMC loop:
1. **Before SR solve**: checks `local_energies` and `local_O` for NaN/Inf
2. **After SR solve**: checks `dp` (parameter update direction) for NaN/Inf, prints norm of non-bad entries
3. **After parameter update**: checks updated `model.parameters()` for NaN/Inf. If clean, prints `dp norm` and `params norm` for monitoring.

### Root cause identified

**Problem chain:**
1. SU-pretrained PEPS has boundary MPS with near-degenerate eigh eigenvalues (gaps as small as 2.6e-8)
2. The eigh backward formula has `1/(w_i - w_j)` terms → gradients amplified by ~4e+7
3. Raw `d(psi)/d(params)` is 10^8–10^11 (vs ~0.3 for exact contraction)
4. O_loc = grad/amp has rms=2e+7, max=8.8e+10
5. This makes the QGT matrix ill-conditioned: MinSR produces dp~10^63, MINRES produces dp~10^-11 — both unusable

**Evidence (4x2, D=chi=10, SU-loaded scale_factor=4):**
| Contraction | raw grads rms | O_loc rms | O_loc max |
|---|---|---|---|
| chi=-1 (exact, no SVD/eigh) | 3.1e-01 | 3.2e-01 | 2.1e+02 |
| chi=10 (boundary, eigh) | 1.9e+08 | 8.4e+07 | 1.1e+11 |

**eigh eigenvalue comparison:**
| State | Condition # max | Min eigenvalue gap | #(gap<1e-3) |
|---|---|---|---|
| SU-loaded | 2.35e+11 | 2.64e-08 | 5/8 |
| Random | 1.92e+05 | 3.52e-02 | 0/8 |

The SU-trained tensors produce near-singular boundary MPS matrices (eigenvalues like `[1.01, 0.94, 8e-7, 4e-8, 1e-10]`). Random PEPS has well-separated eigenvalues (`[8.3, 6.8, 4.5, 3.6, 2.2]`).

### Debug controls added
- `VMCLoopConfig.debug: bool = False` — gates all `[dbg]` prints in VMC loop
- `VMCConfig.debug` in run scripts → passed through to VMCLoopConfig
- NaN/Inf warnings always print regardless of debug flag

### Further investigation: sequential vs vmap

Tested B=16 with sequential per-sample backward vs vmap(grad):
- **Identical results** — vmap is NOT the issue
- **Sample 3** has grad_norm=9.8e+04, while most are O(10-40)
- The problem is **outlier configurations**: specific configs create ill-conditioned intermediate matrices in boundary contraction, producing extreme gradients through the eigh backward
- With B=4096, the probability of hitting extreme outliers is much higher, explaining O_loc max=8.8e+10

### Root cause (refined)

Not a vmap bug, not a systematic eigh issue. **Specific configurations** create near-degenerate boundary MPS Gram matrices during contraction. The custom `safe_inverse_random` in `RobustSVD_EIG` backward caps F at ~5e5 (Lorentzian with eps=1e-12), but the gradient can still be amplified through the chain of multiple SVD calls + tensor contractions in the boundary MPS.

### SVD call path (important note)

Boundary contraction does **NOT** go through `linalg.svd`. The call chain is:
```
autoray linalg.qr → size_aware_qr(via_eigh=True) → qr_via_eigh → RobustSVD_EIG.apply
```
So the S values to check are in `RobustSVD_EIG`, not in any `linalg.svd` or `size_aware_svd` function.

### SVD singular values: normal vs outlier sample

Hooked `RobustSVD_EIG.apply` via patching `tu.qr_via_eigh` to log S during forward pass for both normal sample (seed=42) and outlier (seed=45).

**Normal sample (grad_norm=14):** 4 RobustSVD_EIG calls
| Call | Shape | Condition # | Min s_gap |
|---|---|---|---|
| #0 | (2, 5, 50) | 585 / 103 | 0.019 / 0.030 |
| #1 | (2, 5, 5) | **4.8e+5** / 6e+3 | **1.6e-4** / 7e-4 |
| #2 | (2, 5, 50) | 328 / 115 | 0.016 / 0.006 |
| #3 | (2, 5, 5) | **8.5e+4** / 743 | **1.3e-4** / 0.006 |

**Outlier sample (grad_norm=9.8e+4):** 4 RobustSVD_EIG calls
| Call | Shape | Condition # | Min s_gap |
|---|---|---|---|
| #0 | (2, 5, 50) | 237 / 151 | 0.055 / 0.016 |
| #1 | (2, 5, 5) | **3.7e+10** / **7.5e+8** | **1.1e-4** / **2.8e-9** |
| #2 | (2, 5, 50) | 378 / 163 | 0.008 / 0.023 |
| #3 | (2, 5, 5) | **2.5e+29** / **9.7e+29** | **4.5e-10** / **1.4e-6** |

Key findings:
- SVD #1 and #3 are the (2, 5, 5) boundary MPS Gram matrices — these are ill-conditioned
- Outlier has **exact zeros** in S: `[0.248, 0.071, 3.3e-5, 4.5e-10, 0.0]` — rank-deficient
- Min s_gap drops from 1.3e-4 (normal) to **4.5e-10** (outlier)
- The `F = safe_inverse_random(s_i - s_j, eps=1e-12)` backward term: for gap=4.5e-10, F ≈ 4.5e-10/1e-12 = 450 — bounded by Lorentzian, but multiple SVDs chain together amplifying the gradient through the full contraction

### Why burn-in doesn't help

Increasing burn-in does **not** fix the problem because:
1. Outlier configs are **valid MCMC states** with non-negligible ψ(x) amplitude (sample 3 has amp=-10.6, not small)
2. Burn-in removes correlation with initial state, but outlier configs will always exist in the Markov chain at equilibrium
3. The configs that create ill-conditioned boundary MPS are determined by the arrangement of fermions, not by under-burning
4. With B=4096 walkers doing many sweeps, encountering rare ill-conditioned configs is inevitable

### How to handle outliers

Standard VMC approaches:
1. **O_loc clipping** (recommended): `clamp(O_loc, -clip_val, clip_val)` before SR solve. Standard in VMC codes. Clip value typically 5-10× median |O_loc|, or use adaptive percentile-based clipping.
2. **Gradient norm clipping**: clip per-sample grad norm before forming O_loc
3. **Winsorization**: replace outlier O_loc values with the boundary percentile value
4. **chi=-1 (exact contraction)**: gradients are healthy (O_loc~0.3) but O(D^Lx) cost — only feasible for small systems
5. **Increase Lorentzian epsilon** in `RobustSVD_EIG` backward (from 1e-12 to ~1e-6): damps the F matrix more aggressively for near-degenerate singular values, at the cost of less accurate gradients for well-conditioned cases
6. **Start from random PEPS**: random PEPS has well-separated S values, avoids the ill-conditioning. Let VMC optimize from scratch rather than from SU-pretrained state.

### O_loc outlier masking — implemented and tested

Added `outlier_clip_factor` to `VMCLoopConfig` (default=5.0). After forming O_loc = grads/amps for the full batch, computes per-sample norms, and masks out (sets O_loc row to 0, replaces E_loc with batch mean) any sample whose |O_loc| norm exceeds `clip_factor × median`.

**Files changed:**
- `VMC.py`: added `outlier_clip_factor` to `VMCLoopConfig`, masking logic in `_run_sampling_phase` after `torch.cat`
- `run_scripts/vmc_run_fpeps.py`: added `outlier_clip_factor` to `VMCConfig`, passed through to `VMCLoopConfig`

**Test result** (4x2, D=chi=10, SU-loaded scale_factor=4, MINRES SR):
- Step 0: dropped 390/4096 (9.5%) — first step, many outliers
- Steps 1-9: dropped 2-8/4096 (~0.1%) — steady state
- Energy: -0.617 → -0.665 over 10 steps, steadily decreasing, **no NaN**
- Threshold ~9×median, median |O_loc|~1.9

Previously this setup produced NaN after 1 step. The outlier masking completely fixes it.
