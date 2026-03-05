# `torch.linalg.eigh`: Analysis of the Performance Cliff at Matrix Size n=32 (and cuSOLVER Integration Fix)

**Sijing Du**, California Institute of Technology

February 2026

## Abstract

`torch.linalg.eigh` suffers a catastrophic 80-120x performance cliff when the matrix size crosses n=32 to 33 for batched inputs on GPU. The root cause is PyTorch's size-dependent dispatch logic, which only calls the genuinely batched cuSOLVER API (`cusolverDnXsyevBatched`) for n<=32, but returns to a sequential *for-loop* for n>32. We demonstrate the problem, propose a high-level MAGMA workaround and a low-level fix that calls cuSOLVER `cusolverDnXsyevBatched` for all matrix size n --- matching JAX's fix (PR #31375) --- which eliminates the cliff entirely. The fix requires changing a **single line** of dispatch condition in PyTorch's C++ source code.

**Test environment.** All benchmarks: NVIDIA GeForce RTX 4080, CUDA 12.8, PyTorch 2.10.0+cu128, `float64`, Ubuntu 22.04.5 LTS (WSL2). Timing via CUDA events (3 warmup, 5 timed repetitions).

---

## 1. Observation in default PyTorch `eigh` performance

`torch.linalg.eigh` applied to a batch of (B, n, n) symmetric matrices exhibits smooth scaling for n<=32, then an abrupt 80-140x slowdown at n=33.

**Table 1: Default `torch.linalg.eigh` timing (ms) vs. matrix size n.**

|  n  | B=64 time (ms) | B=64 ratio | B=1024 time (ms) | B=1024 ratio |
|----:|---------------:|-----------:|------------------:|-------------:|
|   8 |           0.29 |            |              0.91 |              |
|  16 |           0.52 |            |              2.86 |              |
|  24 |           0.85 |            |              5.49 |              |
|  28 |           0.93 |            |              7.22 |              |
|  32 |           1.05 |       1.0x |              8.30 |         1.0x |
|  33 |          77.20 |       73x  |              1137 |        137x  |
|  34 |          78.87 |       75x  |              1153 |        139x  |
|  40 |          87.37 |       83x  |              1296 |        156x  |
|  48 |          97.31 |       93x  |              1491 |        180x  |
|  64 |          122.1 |      116x  |              1874 |        226x  |
|  96 |          246.2 |      234x  |              3685 |        444x  |
| 128 |          316.0 |      301x  |              4783 |        576x  |

### 1.1 Root cause: PyTorch's dispatch heuristics

The dispatch lives in `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp`, function `linalg_eigh_cusolver` (line ~1614):

```cpp
if (use_cusolver_syevj_batched_
    && batchCount(eigenvectors) > 1
    && eigenvectors.size(-1) <= 32) {
    // genuinely batched: calls cusolverDnXsyevBatched
    linalg_eigh_cusolver_syevj_batched();
} else if (scalar_type == kFloat
           && size >= 32 && size <= 512) {
    // single-matrix Jacobi, LOOPED over batch
    linalg_eigh_cusolver_syevj();
} else {
    // single-matrix divide-and-conquer, LOOPED
    linalg_eigh_cusolver_syevd();
}
```

The branching heuristics:

- **n<=32, batch>1**: correctly routes to `linalg_eigh_cusolver_syevj_batched()`, which (since PR #155695, cuSOLVER >= 11701) calls `cusolverDnXsyevBatched` --- a new batched API alternative to the legacy `cusolverDn<t>syevjBatched`. (We find that this is fast even for batched matrices of size n>32.)

- **Otherwise**: falls through to the `else if` or `else` branches, which call single-matrix wrappers (`syevj` or `syevd`) in a *loop*. Each matrix is solved sequentially with a separate kernel launch. This is where the 80-120x cliff comes from.

The n<=32 gate for batched input is a leftover from the poor performance of the old `syevjBatched` Jacobi API (see PR #53040), which was replaced by the newer/faster `cusolverDnXsyevBatched` in PR #155695 (June 2025). The top-level dispatch condition was never updated.

---

## 2. High-level MAGMA workaround (no C++ level change)

### 2.1 MAGMA vs. cuSOLVER for batched linear algebra

See Figures 1-2 for direct comparison of batched linear algebra performance between cuSOLVER and MAGMA. Default cuSOLVER backend exhibits the n=33 performance cliff, while MAGMA backend shows smooth scaling for `svd, eigh`. Note that `QR` is in general slow for both backends for small matrices compared to `svd, eigh`.

For `eigh`, we observe that for batched input matrices of median size (32 < n < 256), MAGMA is faster than PyTorch's current default cuSOLVER call.

### 2.2 MAGMA + cuSOLVER dynamic dispatch for `eigh`

PyTorch exposes MAGMA as an alternative backend:

```python
torch.backends.cuda.preferred_linalg_library('magma')
```

MAGMA's batched eigensolver has no n=32 cliff and scales smoothly. However, it is significantly slower than cuSOLVER's fused kernel for n<=32, as shown in Table 2.

**Table 2: `torch.linalg.eigh` performance: cuSOLVER (default) vs. MAGMA backend (B=64, `float64`).**

|  n  | default (ms) | MAGMA (ms) | MAGMA speedup      |
|----:|-------------:|-----------:|:-------------------|
|  32 |         0.31 |       2.86 | 0.1x (slower)      |
|  33 |        54.52 |       3.15 | 17.3x              |
|  48 |        67.00 |       5.43 | 12.3x              |
|  64 |        64.45 |       8.37 | 7.7x               |
|  96 |        98.29 |      17.00 | 5.8x               |
| 128 |       142.69 |      29.81 | 4.8x               |
| 256 |       302.38 |     480.18 | 0.6x (slower)      |

MAGMA is a valid workaround for 32 < n <= 256 (5-17x faster than default), but default cuSOLVER wins back at n=256 (1.7x faster than MAGMA). Thus, we propose the workaround on the PyTorch dispatching level: **For batched `eigh`, for n<=32 use cuSOLVER (default) and for 32 < n <= 256 use MAGMA.**

Pseudocode:

```python
def size_aware_eigh(A):
    """
    Heuristic to bypass the n=33 performance cliff in torch.linalg.eigh.
    """
    n = A.size(-1)

    # Use MAGMA for the "cliff zone" on GPU
    if A.is_cuda and A.dim() >= 3 and 32 < n <= 512:
        with torch.backends.cuda.preferred_linalg_library('magma'):
            return torch.linalg.eigh(A)

    # Default cuSOLVER path (fast for n <= 32, but slow loop for n > 32)
    return torch.linalg.eigh(A)
```

Note that this MAGMA workaround:

1. **Requires size-aware dispatch in user code**: `preferred_linalg_library` is process-wide, but it can be toggled locally with save/restore around each call in a matrix-size-dependent manner.
2. **(Caveat) Suboptimal for n > 32**: as we show in Section 3, `cusolverDnXsyevBatched` is 2-4x faster than MAGMA for the same sizes.

---

## 3. Ultimate solution: dispatch updated cuSOLVER batched `eigh` (`cusolverDnXsyevBatched`), as in JAX PR #31375

### 3.1 The fix

`cusolverDnXsyevBatched` (available since cuSOLVER 11.7.1 / CUDA 12.6) is a newer batched eigensolver (divide-and-conquer, or Jacobi for n<=32) alternative to older `cusolverDn<t>syevjBatched`. PyTorch already wraps it and uses it for n<=32. The fix is to remove the n<=32 gate:

```cpp
// Proposed change in linalg_eigh_cusolver:
  // cusolverDnXsyevBatched works for all n when batch size > 1.
  // See jax-ml/jax#31375 for precedent.
if (batchCount(eigenvectors) > 1) {
    linalg_eigh_cusolver_syevj_batched(...);
} else if (scalar_type == kFloat
            && size >= 32 && size <= 512) {
// Non-batched: preserve original dispatch.
    linalg_eigh_cusolver_syevj(...);
} else {
    linalg_eigh_cusolver_syevd(...);
}
```

This is to route all batched `eigh` through `cusolverDnXsyevBatched`, replacing the per-matrix `syevd` loop for n>32. This fix is probably similar to the approach JAX took in [jax-ml/jax#31375](https://github.com/jax-ml/jax/pull/31375) (merged September 2025).

### 3.2 Verification via C++ extension

To verify without rebuilding PyTorch, we wrote a minimal C++ extension (`my_eigh.cpp`) that calls `cusolverDnXsyevBatched` directly for all matrix sizes, loaded via `torch.utils.cpp_extension.load`.

**Correctness.** In our test, the maximum eigenvalue difference |delta| w.r.t. results from default `torch.linalg.eigh` and eigenvector residual ||Av - lambda v|| are required to be smaller than 1e-10 to ensure the correctness of the eigen decomposition. For an example case at (B=16, n=48), we have verified |delta|=0 and ||Av - lambda v||=2.13e-13, which pass the correctness test.

**Performance.**

**Table 3: Default `eigh` vs. `XsyevBatched` extension (B=64, `float64`).**

|  n  | default (ms) | XsyevBatched (ms) | speedup  |
|----:|-----------:|-------------------:|:---------|
|   8 |       0.29 |               0.20 | 1.5x     |
|  16 |       0.52 |               0.41 | 1.3x     |
|  24 |       0.85 |               0.74 | 1.1x     |
|  28 |       0.93 |               0.78 | 1.2x     |
|  32 |       1.05 |               0.88 | 1.2x     |
|  33 |      77.20 |               1.68 | **46x**  |
|  34 |      78.87 |               1.43 | **55x**  |
|  40 |      87.37 |               1.53 | **57x**  |
|  48 |      97.31 |               1.77 | **55x**  |
|  64 |      122.1 |               2.34 | **52x**  |
|  96 |      246.2 |               6.74 | **37x**  |
| 128 |      316.0 |               9.88 | **32x**  |

**Table 4: Default `eigh` vs. `XsyevBatched` extension (B=1024, `float64`).**

|  n  | default (ms) | XsyevBatched (ms) | speedup  |
|----:|-----------:|-------------------:|:---------|
|   8 |       0.91 |               0.75 | 1.2x     |
|  16 |       2.86 |               2.66 | 1.1x     |
|  24 |       5.49 |               5.47 | 1.0x     |
|  28 |       7.22 |               6.92 | 1.0x     |
|  32 |       8.30 |               7.81 | 1.1x     |
|  33 |       1137 |              15.06 | **75x**  |
|  34 |       1153 |              15.01 | **77x**  |
|  40 |       1296 |              17.36 | **75x**  |
|  48 |       1491 |              21.89 | **68x**  |
|  64 |       1874 |              30.44 | **62x**  |
|  96 |       3685 |              96.10 | **38x**  |
| 128 |       4783 |              145.7 | **33x**  |

### 3.3 Comparison: all three methods

Table 5 summarizes the three dispatch strategies for the representative case B=64:

**Table 5: Three-way comparison (B=64, `float64`). Best time per row in bold.**

|  n  | default (ms) | MAGMA (ms) | XsyevBatched (ms) |
|----:|-------------:|-----------:|-------------------:|
|  32 |     **0.90** |       4.24 |           **0.88** |
|  33 |        69.88 |       4.44 |           **1.68** |
|  48 |        93.12 |       7.76 |           **1.77** |
|  64 |        116.7 |      10.01 |           **2.34** |

`XsyevBatched` is the best of both worlds: fast for n<=32 (matching current cuSOLVER) *and* fast for n>32 (2-4x faster than MAGMA).

### 3.4 Summary

1. The n=32 to 33 performance cliff is **completely eliminated** --- timing scales smoothly across all n.
2. **32-77x speedup** for n>32, with no regression for n<=32.
3. **Numerically identical** to default PyTorch (same cuSOLVER API, just called unconditionally).
4. The fix is a **one-line dispatch heuristics change**, using an API PyTorch already wraps.
5. This matches what [JAX shipped in September 2025](https://github.com/jax-ml/jax/pull/31375).

---

## Appendix A: Reproducer

```python
import torch

device = "cuda"
B = 1024
dtype = torch.float64

for n in [31, 32, 33, 34, 48, 64]:
    x = torch.randn(B, n, n, device=device, dtype=dtype)
    a = (x + x.mT) / 2

    # warmup
    for _ in range(5):
        torch.linalg.eigh(a)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        torch.linalg.eigh(a)
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / 10
    print(f"eigh n={n}: {t:.2f} ms")
```

## Appendix B: C++ extension source (`my_eigh.cpp`)

```cpp
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cusolverDn.h>

static cusolverDnHandle_t get_handle() {
    static thread_local cusolverDnHandle_t h = nullptr;
    if (!h) cusolverDnCreate(&h);
    cusolverDnSetStream(h,
        c10::cuda::getCurrentCUDAStream().stream());
    return h;
}

std::tuple<torch::Tensor, torch::Tensor>
custom_batched_eigh(torch::Tensor input, bool upper) {
    TORCH_CHECK(input.is_cuda(),
                "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2
                && input.size(-1) == input.size(-2),
                "input must be square");

    const auto dtype = input.scalar_type();
    const int64_t n = input.size(-1);
    const int64_t batch = input.numel() / (n * n);

    // Clone: cuSOLVER overwrites input with eigenvectors.
    // Symmetric => row-major == column-major, no transpose needed.
    auto vectors = input.contiguous().clone();
    auto values = torch::empty({batch, n}, input.options());
    auto info = torch::zeros({batch},
        input.options().dtype(torch::kInt32));

    auto handle = get_handle();
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);

    auto jobz = CUSOLVER_EIG_MODE_VECTOR;
    auto uplo = upper ? CUBLAS_FILL_MODE_UPPER
                      : CUBLAS_FILL_MODE_LOWER;
    auto cuda_dtype = (dtype == torch::kFloat64)
                      ? CUDA_R_64F : CUDA_R_32F;

    // Query workspace, allocate, compute
    size_t work_device_sz = 0, work_host_sz = 0;
    cusolverDnXsyevBatched_bufferSize(
        handle, params, jobz, uplo, n, cuda_dtype,
        vectors.data_ptr(), n, cuda_dtype,
        values.data_ptr(), cuda_dtype,
        &work_device_sz, &work_host_sz, batch);

    auto work_device = torch::empty(
        {static_cast<int64_t>(work_device_sz)},
        input.options().dtype(torch::kUInt8));
    std::vector<uint8_t> work_host(work_host_sz);

    cusolverDnXsyevBatched(
        handle, params, jobz, uplo, n, cuda_dtype,
        vectors.data_ptr(), n, cuda_dtype,
        values.data_ptr(), cuda_dtype,
        work_device.data_ptr(), work_device_sz,
        work_host.data(), work_host_sz,
        info.data_ptr<int>(), batch);

    cusolverDnDestroyParams(params);

    // cuSOLVER writes column-major eigenvectors into our
    // row-major buffer; transpose back for PyTorch.
    vectors = vectors.view({batch, n, n})
                     .mT().contiguous();
    auto out_shape = input.sizes().vec();
    vectors = vectors.reshape(out_shape);
    auto val_shape = std::vector<int64_t>(
        out_shape.begin(), out_shape.end() - 1);
    values = values.reshape(val_shape);

    return {values, vectors};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eigh", &custom_batched_eigh,
          "Batched eigh via cusolverDnXsyevBatched",
          py::arg("input"), py::arg("upper") = false);
}
```

Load and benchmark:

```python
from torch.utils.cpp_extension import load
ext = load("my_eigh", sources=["my_eigh.cpp"],
           extra_cflags=["-O3"],
           extra_include_paths=[f"{cuda_home}/include"],
           extra_ldflags=[f"-L{cuda_home}/lib64",
                          "-lcusolver", "-lcudart"])
vals, vecs = ext.eigh(A)  # (B,n,n) -> (B,n), (B,n,n)
```
