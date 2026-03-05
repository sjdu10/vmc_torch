/*
 * my_eigh.cpp — Batched symmetric eigendecomposition via
 * cusolverDnXsyevBatched, bypassing PyTorch's n<=32 dispatch gate.
 *
 * This is a plain C++ file (no CUDA kernels). cuSOLVER functions are
 * host-callable C APIs — we just need the header and the library.
 *
 * Load from Python:
 *   from torch.utils.cpp_extension import load
 *   ext = load("my_eigh", sources=["my_eigh.cpp"], ...)
 *   vals, vecs = ext.eigh(A)  # A is (B, n, n) symmetric
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cusolverDn.h>  // NOLINT — IDE may not find this; JIT passes -I$(CUDA_HOME)/include

// One cuSOLVER handle per thread, synced to current PyTorch stream.
static cusolverDnHandle_t get_handle() {
    static thread_local cusolverDnHandle_t h = nullptr;
    if (!h) cusolverDnCreate(&h);
    cusolverDnSetStream(h, c10::cuda::getCurrentCUDAStream().stream());
    return h;
}

// ── Main function ──────────────────────────────────────────────────
//
// Calls cusolverDnXsyevBatched for ANY (B, n, n) input.
// This is the same API that PyTorch already uses internally for
// n<=32, but here we call it unconditionally.
//
// cuSOLVER expects column-major (Fortran) layout. Since a symmetric
// matrix satisfies A == A^T, row-major == column-major for the input.
// But the OUTPUT eigenvectors are written column-major, so we need to
// transpose them back to row-major for PyTorch.

std::tuple<torch::Tensor, torch::Tensor>
custom_batched_eigh(torch::Tensor input, bool upper) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "input must be >= 2D");
    TORCH_CHECK(input.size(-1) == input.size(-2), "input must be square");

    const auto dtype = input.scalar_type();
    TORCH_CHECK(dtype == torch::kFloat64 || dtype == torch::kFloat32,
                "only float32 / float64 supported");

    const int64_t n = input.size(-1);
    const int64_t batch = input.numel() / (n * n);

    // cuSOLVER overwrites the input with eigenvectors (column-major).
    // For a symmetric matrix, row-major data == column-major data,
    // so we can just clone and pass the contiguous buffer directly.
    auto vectors = input.contiguous().clone();

    // Eigenvalues: real-valued, shape (batch, n)
    auto values = torch::empty({batch, n}, input.options());

    // Info per matrix (0 = success)
    auto info = torch::zeros({batch}, input.options().dtype(torch::kInt32));

    // ── cuSOLVER setup ─────────────────────────────────────────────
    auto handle = get_handle();

    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);

    auto jobz = CUSOLVER_EIG_MODE_VECTOR;   // compute eigenvectors
    auto uplo = upper ? CUBLAS_FILL_MODE_UPPER
                      : CUBLAS_FILL_MODE_LOWER;

    // Pick the right CUDA datatype tag
    auto cuda_dtype = (dtype == torch::kFloat64) ? CUDA_R_64F : CUDA_R_32F;

    // ── Query workspace size ───────────────────────────────────────
    size_t work_device_sz = 0, work_host_sz = 0;

    cusolverDnXsyevBatched_bufferSize( // NOLINT
        handle, params, jobz, uplo,
        n,                          // matrix dimension
        cuda_dtype,                 // datatype of A
        vectors.data_ptr(),         // A (will be overwritten)
        n,                          // leading dimension
        cuda_dtype,                 // datatype of eigenvalues
        values.data_ptr(),          // eigenvalues output
        cuda_dtype,                 // compute type
        &work_device_sz,
        &work_host_sz,
        batch);

    // ── Allocate workspace ─────────────────────────────────────────
    auto work_device = torch::empty(
        {static_cast<int64_t>(work_device_sz)},
        input.options().dtype(torch::kUInt8));
    // Host workspace (pinned not required, regular CPU is fine)
    std::vector<uint8_t> work_host(work_host_sz);

    // ── Compute ────────────────────────────────────────────────────
    cusolverDnXsyevBatched(
        handle, params, jobz, uplo,
        n,
        cuda_dtype,
        vectors.data_ptr(),
        n,
        cuda_dtype,
        values.data_ptr(),
        cuda_dtype,
        work_device.data_ptr(),
        work_device_sz,
        work_host.data(),
        work_host_sz,
        info.data_ptr<int>(),
        batch);

    cusolverDnDestroyParams(params);

    // ── Fix layout ─────────────────────────────────────────────────
    // cuSOLVER wrote eigenvectors in column-major order into our
    // row-major buffer. Reinterpret as (batch, n, n) and transpose.
    vectors = vectors.view({batch, n, n}).mT().contiguous();

    // Reshape to match input's batch dimensions
    auto out_shape = input.sizes().vec();
    vectors = vectors.reshape(out_shape);

    auto val_shape = std::vector<int64_t>(
        out_shape.begin(), out_shape.end() - 1);
    values = values.reshape(val_shape);

    return {values, vectors};
}

// ── Python binding ─────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eigh", &custom_batched_eigh,
          "Batched eigh via cusolverDnXsyevBatched (no n<=32 gate)",
          py::arg("input"), py::arg("upper") = false);
}
