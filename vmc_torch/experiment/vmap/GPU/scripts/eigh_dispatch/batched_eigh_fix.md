# `torch.linalg.eigh`: catastrophic performance cliff at matrix size n=32 for batched inputs on CUDA

## Summary

`torch.linalg.eigh` suffers a catastrophic performance cliff when the matrix size crosses n=32 → 33 for batched inputs on CUDA. The root cause is an outdated size gate in PyTorch's dispatch logic that only routes to the batched cuSOLVER API (`cusolverDnXsyevBatched`) for n ≤ 32, falling back to a sequential for-loop over individual matrices for n > 32. The fix is a **one-line change** to remove the `<= 32` condition—using an API PyTorch already wraps. This matches the approach JAX took in [jax-ml/jax#31375](https://github.com/jax-ml/jax/pull/31375) (merged September 2025).

## Minimal reproducer

```python
import torch
torch.manual_seed(42)
device = "cuda"
B = 128
dtype = torch.float64
print(f'torch version: {torch.__version__}')
print(f'B={B}, dtype={dtype}, matrix size n')
w = 8
print(f"  {'n':>{w}} | {'eigh (ms)':>{w}} ")
print("-" * (2 * w + 9))
for n in [1,2,4,8,16,32,33,34,64,96,128]:
    x = torch.randn(B, n, n, device=device, dtype=dtype)
    a = (x + x.mT) / 2
    # warmup
    for _ in range(1):
        torch.linalg.eigh(a)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        eigvals, eigvecs = torch.linalg.eigh(a)
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / 5
    marker = " <-- n=32 threshold" if n == 32 else ""
    print(f"  {n:{w}d} | {t:{w}.2f} ms{marker}")
```

**Output** (PyTorch 2.10.0+cu126, RTX 4080):

```
torch version: 2.10.0+cu126
B=128, dtype=torch.float64, matrix size n
         n | eigh (ms)
-------------------------
         1 |     0.12 ms
         2 |     0.33 ms
         4 |     0.20 ms
         8 |     0.30 ms
        16 |     0.62 ms
        32 |     1.65 ms <-- n=32 threshold
        33 |   137.13 ms
        34 |   139.70 ms
        64 |   228.77 ms
        96 |   457.44 ms
       128 |   580.13 ms
```

n=32 → n=33 jumps from **1.65 ms to 137 ms** — an **~83x cliff**.

## Root cause

The dispatch lives in `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp`, function `linalg_eigh_cusolver` (~line 1614):

```cpp
void linalg_eigh_cusolver(const Tensor& eigenvalues, const Tensor& eigenvectors,
                          const Tensor& infos, bool upper, bool compute_eigenvectors) {
#if defined(USE_ROCM)
  ...
#else
  if (batchCount(eigenvectors) > 1 && eigenvectors.size(-1) <= 32) {
    // Use syevjBatched for batched matrix operation when matrix size <= 32
    linalg_eigh_cusolver_syevj_batched(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  } else if (eigenvectors.scalar_type() == at::kFloat &&
             eigenvectors.size(-1) >= 32 && eigenvectors.size(-1) <= 512) {
    linalg_eigh_cusolver_syevj(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  } else {
    linalg_eigh_cusolver_syevd(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  }
#endif
}
```

- **n ≤ 32, batch > 1**: routes to `linalg_eigh_cusolver_syevj_batched()`, which (since PR #155695) calls `cusolverDnXsyevBatched` — a batched API that works efficiently for **all** matrix sizes.
- **n > 32, batch > 1**: falls through to `syevj` or `syevd` branches, which solve **one matrix at a time in a loop** with separate kernel launches.

The `<= 32` gate is a leftover from when the old `syevjBatched` Jacobi API performed poorly for larger matrices (see PR #53040). PR #155695 (June 2025) replaced the underlying call with the newer `cusolverDnXsyevBatched`, but the top-level dispatch condition was never updated.

## Proposed fix

Remove the `<= 32` size gate so all batched `eigh` calls route through `cusolverDnXsyevBatched`:

```cpp
  if (batchCount(eigenvectors) > 1) {
    // cusolverDnXsyevBatched works efficiently for all n when batch > 1
    linalg_eigh_cusolver_syevj_batched(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  } else if ...
```

This is the same approach JAX took in [jax-ml/jax#31375](https://github.com/jax-ml/jax/pull/31375).

## Benchmark results

I verified the fix by building PyTorch from source with only this one-line change. All benchmarks on NVIDIA GeForce RTX 4080, CUDA 12.8, Ubuntu 22.04.5 LTS (WSL2). Timing via CUDA events (3 warmup, 5 timed iterations).

### Main result: B=128, float64

| n | default (ms) | fixed (ms) | speedup |
|---:|---:|---:|---:|
| 8 | 0.32 | 0.31 | 1.0x |
| 16 | 0.66 | 0.63 | 1.1x |
| 24 | 1.22 | 1.19 | 1.0x |
| 28 | 1.44 | 1.49 | 1.0x |
| 30 | 1.56 | 1.66 | 0.9x |
| 31 | 1.77 | 1.64 | 1.1x |
| 32 | 1.54 | 1.59 | 1.0x |
| **33** | **137.7** | **3.04** | **45x** |
| **34** | **145.8** | **2.63** | **55x** |
| **40** | **160.9** | **3.19** | **50x** |
| **48** | **180.6** | **3.47** | **52x** |
| **56** | **204.4** | **4.12** | **50x** |
| **64** | **241.4** | **4.67** | **52x** |
| **80** | **387.9** | **10.5** | **37x** |
| **96** | **458.0** | **12.9** | **36x** |
| **128** | **605.1** | **18.9** | **32x** |
| **256** | **714.2** | **68.3** | **10x** |
| **400** | **1113.7** | **198.9** | **6x** |
| **512** | **1382.3** | **322.5** | **4x** |
| **513** | **1455.7** | **391.0** | **4x** |
| **600** | **1793.8** | **547.6** | **3x** |
| **1024** | **3555.8** | **1962.0** | **1.8x** |

**Up to 55x speedup** for n > 32, **no regression** for n ≤ 32.

### Additional configurations

<details>
<summary>B=1, float64 (no batching — confirms no regression)</summary>

| n | default (ms) | fixed (ms) | speedup |
|---:|---:|---:|---:|
| 8 | 0.31 | 0.29 | 1.1x |
| 16 | 0.54 | 0.44 | 1.2x |
| 24 | 0.59 | 0.51 | 1.2x |
| 28 | 0.63 | 0.74 | 0.9x |
| 30 | 0.65 | 0.58 | 1.1x |
| 31 | 0.79 | 0.60 | 1.3x |
| 32 | 0.72 | 0.63 | 1.1x |
| 33 | 1.31 | 1.18 | 1.1x |
| 34 | 1.28 | 1.22 | 1.1x |
| 40 | 1.43 | 1.43 | 1.0x |
| 48 | 1.88 | 1.57 | 1.2x |
| 56 | 1.95 | 1.73 | 1.1x |
| 64 | 2.00 | 1.94 | 1.0x |
| 80 | 3.48 | 3.17 | 1.1x |
| 96 | 3.73 | 3.66 | 1.0x |
| 128 | 4.87 | 4.84 | 1.0x |
| 256 | 5.78 | 5.66 | 1.0x |
| 400 | 8.98 | 8.76 | 1.0x |
| 512 | 11.3 | 11.2 | 1.0x |
| 513 | 11.8 | 11.7 | 1.0x |
| 600 | 14.3 | 14.0 | 1.0x |
| 1024 | 28.3 | 28.2 | 1.0x |

No cliff for B=1 (single matrix takes the `syevj`/`syevd` path directly, not a loop). No regression from the fix.

</details>

<details>
<summary>B=32, float64 (up to 35x speedup)</summary>

| n | default (ms) | fixed (ms) | speedup |
|---:|---:|---:|---:|
| 8 | 0.26 | 0.29 | 0.9x |
| 16 | 0.38 | 0.42 | 0.9x |
| 24 | 0.59 | 0.59 | 1.0x |
| 28 | 0.62 | 0.67 | 0.9x |
| 30 | 0.62 | 0.63 | 1.0x |
| 31 | 0.67 | 0.98 | 0.7x |
| 32 | 0.65 | 0.73 | 0.9x |
| **33** | **37.0** | **1.43** | **26x** |
| **34** | **36.8** | **1.24** | **30x** |
| **40** | **42.1** | **1.39** | **30x** |
| **48** | **47.6** | **1.51** | **32x** |
| **56** | **54.7** | **1.75** | **31x** |
| **64** | **64.1** | **1.83** | **35x** |
| **80** | **100.6** | **4.38** | **23x** |
| **96** | **116.0** | **5.04** | **23x** |
| **128** | **147.4** | **7.09** | **21x** |
| **256** | **178.7** | **26.9** | **7x** |
| **400** | **278.7** | **66.5** | **4x** |
| **512** | **348.9** | **90.7** | **4x** |
| **513** | **367.8** | **108.3** | **3x** |
| **600** | **457.2** | **149.9** | **3x** |
| **1024** | **892.1** | **505.5** | **1.8x** |

</details>

<details>
<summary>B=64, float64 (up to 49x speedup)</summary>

| n | default (ms) | fixed (ms) | speedup |
|---:|---:|---:|---:|
| 8 | 0.27 | 0.27 | 1.0x |
| 16 | 0.41 | 0.48 | 0.8x |
| 24 | 0.67 | 0.68 | 1.0x |
| 28 | 0.78 | 0.88 | 0.9x |
| 30 | 0.88 | 0.81 | 1.1x |
| 31 | 0.99 | 0.83 | 1.2x |
| 32 | 0.89 | 0.82 | 1.1x |
| **33** | **71.1** | **1.66** | **43x** |
| **34** | **73.7** | **1.50** | **49x** |
| **40** | **80.5** | **1.63** | **49x** |
| **48** | **91.0** | **1.89** | **48x** |
| **56** | **105.3** | **2.18** | **48x** |
| **64** | **117.5** | **2.61** | **45x** |
| **80** | **194.9** | **5.75** | **34x** |
| **96** | **228.0** | **6.84** | **33x** |
| **128** | **295.8** | **9.85** | **30x** |
| **256** | **356.2** | **44.7** | **8x** |
| **400** | **555.3** | **109.2** | **5x** |
| **512** | **698.0** | **159.8** | **4x** |
| **513** | **737.4** | **189.5** | **4x** |
| **600** | **916.1** | **275.1** | **3x** |
| **1024** | **1785.2** | **977.7** | **1.8x** |

</details>

<details>
<summary>B=64, float32 (up to 115x speedup)</summary>

| n | default (ms) | fixed (ms) | speedup |
|---:|---:|---:|---:|
| 8 | 0.20 | 0.18 | 1.1x |
| 16 | 0.27 | 0.24 | 1.1x |
| 24 | 0.29 | 0.28 | 1.1x |
| 28 | 0.31 | 0.54 | 0.6x |
| 30 | 0.31 | 0.32 | 1.0x |
| 31 | 0.30 | 0.33 | 0.9x |
| 32 | 0.30 | 0.32 | 1.0x |
| **33** | **59.3** | **0.83** | **71x** |
| **34** | **58.1** | **0.52** | **111x** |
| **40** | **59.9** | **0.52** | **115x** |
| **48** | **62.7** | **0.57** | **110x** |
| **56** | **64.6** | **0.63** | **103x** |
| **64** | **67.6** | **0.67** | **101x** |
| **80** | **93.1** | **1.12** | **83x** |
| **96** | **101.9** | **1.30** | **78x** |
| **128** | **137.5** | **1.83** | **75x** |
| **256** | **307.2** | **6.36** | **48x** |
| **400** | **587.1** | **31.3** | **19x** |
| **512** | **754.4** | **38.9** | **19x** |
| **513** | **237.8** | **60.0** | **4x** |
| **600** | **282.3** | **76.9** | **4x** |
| **1024** | **523.4** | **258.4** | **2x** |

</details>

<details>
<summary>B=256, float32 (up to 249x speedup)</summary>

| n | default (ms) | fixed (ms) | speedup |
|---:|---:|---:|---:|
| 8 | 0.20 | 0.20 | 1.0x |
| 16 | 0.29 | 0.30 | 0.9x |
| 24 | 0.47 | 0.50 | 0.9x |
| 28 | 0.57 | 0.62 | 0.9x |
| 30 | 0.61 | 0.59 | 1.0x |
| 31 | 0.60 | 0.60 | 1.0x |
| 32 | 0.61 | 0.64 | 1.0x |
| **33** | **232.6** | **1.37** | **170x** |
| **34** | **255.5** | **1.02** | **249x** |
| **40** | **253.5** | **1.16** | **218x** |
| **48** | **272.8** | **1.32** | **206x** |
| **56** | **280.4** | **1.62** | **173x** |
| **64** | **329.6** | **1.85** | **178x** |
| **80** | **399.0** | **3.28** | **122x** |
| **96** | **399.0** | **4.26** | **94x** |
| **128** | **565.0** | **6.11** | **93x** |
| **256** | **1249.6** | **25.6** | **49x** |
| **400** | **2472.4** | **135.2** | **18x** |
| **512** | **3108.4** | **214.2** | **15x** |
| **513** | **956.5** | **257.5** | **4x** |
| **600** | **1130.2** | **371.8** | **3x** |
| **1024** | **2045.9** | **1158.7** | **1.8x** |

</details>

## Key observations

1. The n=32 → 33 performance cliff is **eliminated** — timing scales smoothly across all n.
2. **No regression** for n ≤ 32 (the batched path is already used there).
3. **No regression** for B=1 (unbatched case takes a different code path, unaffected by the change).
4. Speedup scales with batch size: larger B → larger speedup for n > 32.
5. The fix is **numerically identical** — same cuSOLVER API, just called unconditionally for batched inputs.
6. This is a **one-line change**, using an API PyTorch already wraps.
7. This matches what [JAX shipped in September 2025](https://github.com/jax-ml/jax/pull/31375).

## Environment

- GPU: NVIDIA GeForce RTX 4080
- CUDA: 12.8
- PyTorch: 2.10.0+cu126 (default) / 2.12.0a0+git67c428c (with fix)
- OS: Ubuntu 22.04.5 LTS (WSL2)
- Driver: 572.83

## Related

- PR #155695 — replaced `syevjBatched` with `cusolverDnXsyevBatched` (but kept the `<= 32` gate)
- PR #53040 — original introduction of the `<= 32` heuristic
- [jax-ml/jax#31375](https://github.com/jax-ml/jax/pull/31375) — JAX's equivalent fix

cc @lezcano @nikitaved
