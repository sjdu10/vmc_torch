/*
 * =================================================================
 * eigh_batched_ext.cu
 * =================================================================
 *
 * A CUDA extension for PyTorch that performs batched symmetric
 * eigendecomposition using NVIDIA's cusolverDnXsyevBatched API.
 *
 * WHY THIS EXISTS:
 *   Stock PyTorch's torch.linalg.eigh has a hard-coded gate:
 *       if (batch > 1 && n <= 32)  -->  fast batched kernel
 *       else                        -->  slow looped-per-matrix kernel
 *   This causes an 80-120x performance cliff at n=33.
 *   cusolverDnXsyevBatched (added in cuSOLVER 11.7.1 / CUDA 12.6)
 *   is genuinely batched for ALL matrix sizes. This extension calls
 *   it directly, bypassing the n<=32 gate.
 *
 * WHAT IS A .cu FILE?
 *   A .cu file is compiled by NVIDIA's nvcc compiler (not gcc/clang).
 *   nvcc understands both regular C++ code AND special CUDA syntax
 *   for writing GPU kernels (functions that run on the GPU).
 *
 *   However, THIS file doesn't actually contain any custom GPU
 *   kernels — we're just calling cuSOLVER library functions from the
 *   CPU side. We use .cu instead of .cpp so that torch's build
 *   system automatically finds CUDA headers and links CUDA libraries.
 *   (The .cpp version in my_eigh.cpp achieves the same thing with
 *   manual include/link flags.)
 *
 * HOW TO USE FROM PYTHON:
 *   from torch.utils.cpp_extension import load
 *   ext = load("eigh_batched_ext",
 *              sources=["eigh_batched_ext.cu"],
 *              extra_cuda_cflags=["-O3"],
 *              extra_ldflags=["-lcusolver"],
 *              verbose=True)
 *   eigenvalues, eigenvectors = ext.eigh_xsyev_batched(A)
 *
 * PREREQUISITES:
 *   - PyTorch with CUDA support (pip install torch)
 *   - CUDA toolkit >= 12.6 (for cusolverDnXsyevBatched)
 *   - ninja build system (pip install ninja)
 */


/* ================================================================
 * SECTION 1: INCLUDES
 * ================================================================
 *
 * In C/C++, #include copies the contents of a header file into
 * this file at compile time. Think of it as "import" in Python.
 */

// torch/extension.h — PyTorch's C++ API.
// Gives us torch::Tensor (the C++ equivalent of torch.Tensor in
// Python), plus TORCH_CHECK (like Python's assert), and the
// PYBIND11_MODULE macro to expose C++ functions to Python.
#include <torch/extension.h>

// c10/cuda/CUDAStream.h — Access to PyTorch's current CUDA stream.
// A "stream" is a queue of GPU operations. PyTorch uses streams to
// order GPU work. We need this to tell cuSOLVER "run on the same
// stream PyTorch is using" so operations don't run out of order.
#include <c10/cuda/CUDAStream.h>

// cusolverDn.h — NVIDIA's cuSOLVER library for dense linear algebra.
// "Dn" = Dense. This header declares functions like
// cusolverDnXsyevBatched (batched symmetric eigensolver).
// These are LIBRARY CALLS that run on the GPU — we call them from
// CPU code, and they launch GPU kernels internally.
#include <cusolverDn.h>

// cuda_runtime.h — CUDA runtime API (memory management, streams, etc.)
// We don't use it directly here, but it's included for completeness
// since cuSOLVER depends on types defined here.
#include <cuda_runtime.h>


/* ================================================================
 * SECTION 2: cuSOLVER HANDLE MANAGEMENT
 * ================================================================
 *
 * cuSOLVER requires a "handle" — an opaque object that holds internal
 * state (GPU memory allocators, selected algorithms, etc.).
 *
 * Analogy: a handle is like opening a database connection. You create
 * it once, reuse it for many operations, then destroy it at the end.
 *
 * KEY CONCEPTS:
 *
 *   static thread_local:
 *     - "static" means the variable persists across function calls
 *       (like a global, but scoped to this function).
 *     - "thread_local" means each CPU thread gets its own copy.
 *       This matters because cuSOLVER handles are NOT thread-safe.
 *
 *   CUDA stream synchronization:
 *     PyTorch schedules GPU work on "streams". We must tell cuSOLVER
 *     to use the SAME stream, otherwise our cuSOLVER call might run
 *     before PyTorch's tensor operations finish.
 */
static cusolverDnHandle_t get_cusolver_handle() {
    // One handle per thread, created on first call, reused after.
    static thread_local cusolverDnHandle_t handle = nullptr;

    if (handle == nullptr) {
        // First call on this thread: create the handle.
        cusolverDnCreate(&handle);
    }

    // Every call: sync to PyTorch's current CUDA stream.
    // c10::cuda::getCurrentCUDAStream() returns PyTorch's stream,
    // .stream() extracts the raw cudaStream_t that cuSOLVER needs.
    cusolverDnSetStream(
        handle,
        c10::cuda::getCurrentCUDAStream().stream()
    );

    return handle;
}


/* ================================================================
 * SECTION 3: THE MAIN FUNCTION
 * ================================================================
 *
 * This function takes a batch of symmetric matrices and returns
 * their eigenvalues and eigenvectors using cusolverDnXsyevBatched.
 *
 * INPUT:  (B, n, n) symmetric matrix on GPU
 * OUTPUT: eigenvalues (B, n), eigenvectors (B, n, n)
 *
 * MEMORY LAYOUT ISSUE (the trickiest part!):
 *
 *   PyTorch stores matrices in ROW-MAJOR order (C convention):
 *     [[a, b],    stored as [a, b, c, d] in memory
 *      [c, d]]
 *
 *   cuSOLVER expects COLUMN-MAJOR order (Fortran convention):
 *     [[a, b],    stored as [a, c, b, d] in memory
 *      [c, d]]
 *
 *   For SYMMETRIC matrices (A == A^T), row-major and column-major
 *   layouts are IDENTICAL in memory! So we can pass the input
 *   directly without any transpose.
 *
 *   But the OUTPUT eigenvectors are written in column-major order,
 *   so we need to transpose them back to row-major for PyTorch.
 */

std::tuple<torch::Tensor, torch::Tensor>
eigh_xsyev_batched(torch::Tensor input, bool upper) {

    /* ── Step 1: Input validation ──────────────────────────────────
     *
     * TORCH_CHECK is like Python's "assert" but throws a C++
     * exception with a nice error message. Always validate inputs
     * in library code!
     */
    TORCH_CHECK(input.is_cuda(),
                "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2,
                "Input must be at least 2D");
    TORCH_CHECK(input.size(-1) == input.size(-2),
                "Input must be square (last two dims must match)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat64 ||
                input.scalar_type() == torch::kFloat32,
                "Only float32/float64 supported");

    /* ── Step 2: Extract dimensions ────────────────────────────────
     *
     * For input shape (B, n, n):
     *   n = matrix size (last dimension)
     *   batch_size = total number of matrices = B
     *
     * We use numel()/(n*n) instead of size(0) to handle inputs
     * with more batch dims, e.g. (B1, B2, n, n).
     */
    const int64_t n = input.size(-1);
    const int64_t batch_size = input.numel() / (n * n);

    /* ── Step 3: Allocate output tensors ───────────────────────────
     *
     * cuSOLVER works IN-PLACE: it overwrites the input matrix with
     * eigenvectors. So we clone the input to avoid destroying it.
     *
     * .contiguous() ensures the data is laid out sequentially in
     * memory (no strides/gaps). cuSOLVER requires this.
     * .clone() makes a deep copy so the original input is safe.
     *
     * For a symmetric matrix, row-major == column-major, so we
     * pass the data buffer directly to cuSOLVER.
     */
    auto eigenvectors = input.contiguous().clone();

    // Eigenvalues: one real number per eigenvector, shape (B, n).
    // input.options() copies device and dtype from input tensor.
    auto eigenvalues = torch::empty({batch_size, n}, input.options());

    // Info: one integer per matrix. 0 = success, >0 = failed.
    auto infos = torch::zeros(
        {batch_size},
        input.options().dtype(torch::kInt32)
    );

    /* ── Step 4: cuSOLVER setup ────────────────────────────────────
     *
     * cuSOLVER functions take many configuration arguments. Let's
     * set them up.
     */
    auto handle = get_cusolver_handle();

    // "params" is an opaque config object. We create it, use it,
    // then destroy it. (Could set algorithm hints here, but
    // defaults are fine.)
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);

    // jobz: do we want eigenvectors?
    //   CUSOLVER_EIG_MODE_VECTOR  = yes, compute eigenvalues + vectors
    //   CUSOLVER_EIG_MODE_NOVECTOR = eigenvalues only (faster)
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    // uplo: which triangle of the symmetric matrix to read?
    //   UPPER = use upper triangle, ignore lower
    //   LOWER = use lower triangle, ignore upper
    // For truly symmetric input, both give the same result.
    cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER
                                  : CUBLAS_FILL_MODE_LOWER;

    // CUDA datatype tag — tells cuSOLVER the precision.
    //   CUDA_R_64F = real float64 (double)
    //   CUDA_R_32F = real float32 (float)
    // cuSOLVER uses this instead of C++ templates because it's a C API.
    cudaDataType cuda_dtype = (input.scalar_type() == torch::kFloat64)
                              ? CUDA_R_64F
                              : CUDA_R_32F;

    /* ── Step 5: Query workspace size ──────────────────────────────
     *
     * cuSOLVER needs temporary scratch memory ("workspace") to run.
     * The pattern is always:
     *   1. Call _bufferSize() to ask "how much workspace do you need?"
     *   2. Allocate that much memory
     *   3. Call the actual function, passing the workspace
     *
     * This is a common pattern in Fortran-era linear algebra libs
     * (LAPACK, cuSOLVER, MAGMA) because they don't allocate memory
     * internally — the caller manages all memory.
     *
     * There are TWO workspaces:
     *   - Device workspace: scratch memory on the GPU
     *   - Host workspace: scratch memory on the CPU
     * cuSOLVER may need both for its internal algorithms.
     */
    size_t worksize_device = 0;
    size_t worksize_host = 0;

    cusolverDnXsyevBatched_bufferSize(
        handle,                        // cuSOLVER handle
        params,                        // config params
        jobz,                          // compute vectors?
        uplo,                          // upper/lower triangle
        n,                             // matrix dimension
        cuda_dtype,                    // datatype of input matrix
        eigenvectors.data_ptr(),       // pointer to matrix data (GPU)
        n,                             // leading dimension (= n for contiguous)
        cuda_dtype,                    // datatype of eigenvalues
        eigenvalues.data_ptr(),        // pointer to eigenvalue data (GPU)
        cuda_dtype,                    // compute precision
        &worksize_device,              // OUTPUT: bytes needed on GPU
        &worksize_host,                // OUTPUT: bytes needed on CPU
        batch_size                     // number of matrices
    );

    /* ── Step 6: Allocate workspace ────────────────────────────────
     *
     * GPU workspace: allocate as a raw byte tensor on the same GPU.
     * Host workspace: allocate as a plain C++ vector (CPU memory).
     */
    auto work_device = torch::empty(
        {static_cast<int64_t>(worksize_device)},
        input.options().dtype(torch::kUInt8)  // raw bytes on GPU
    );
    std::vector<uint8_t> work_host(worksize_host);  // raw bytes on CPU

    /* ── Step 7: COMPUTE! ──────────────────────────────────────────
     *
     * This is the actual eigendecomposition. cuSOLVER launches GPU
     * kernels internally to solve all B eigenproblems in parallel.
     *
     * After this call:
     *   - eigenvalues contains the eigenvalues (sorted ascending)
     *   - eigenvectors is OVERWRITTEN with eigenvectors
     *     (in column-major layout!)
     *
     * The call is asynchronous — it returns immediately and the GPU
     * does the work in the background. The results are only
     * guaranteed ready after a cudaStreamSynchronize (which PyTorch
     * handles when you access the tensor data from Python).
     */
    cusolverDnXsyevBatched(
        handle,
        params,
        jobz,
        uplo,
        n,
        cuda_dtype,
        eigenvectors.data_ptr(),       // INPUT: symmetric matrices
                                       // OUTPUT: eigenvectors (overwritten!)
        n,                             // leading dimension
        cuda_dtype,
        eigenvalues.data_ptr(),        // OUTPUT: eigenvalues
        cuda_dtype,
        work_device.data_ptr(),        // scratch space (GPU)
        worksize_device,
        work_host.data(),              // scratch space (CPU)
        worksize_host,
        infos.data_ptr<int>(),         // OUTPUT: error codes per matrix
        batch_size
    );

    // Clean up the params object (handle persists for reuse).
    cusolverDnDestroyParams(params);

    /* ── Step 8: Fix memory layout ─────────────────────────────────
     *
     * cuSOLVER wrote eigenvectors in COLUMN-MAJOR order into our
     * row-major buffer. To fix this:
     *
     *   .view({batch, n, n})  — interpret flat buffer as 3D
     *   .mT()                 — transpose last two dims (logical,
     *                           just changes strides, no data copy)
     *   .contiguous()         — actually rearrange data in memory
     *                           to match the new logical layout
     *
     * After this, eigenvectors[i] is a proper row-major matrix
     * where column j is the j-th eigenvector.
     */
    eigenvectors = eigenvectors.view({batch_size, n, n})
                               .mT()
                               .contiguous();

    // Reshape outputs to match the original input's batch shape.
    // E.g., if input was (B1, B2, n, n), eigenvalues → (B1, B2, n).
    auto out_shape = input.sizes().vec();
    eigenvectors = eigenvectors.reshape(out_shape);

    auto val_shape = std::vector<int64_t>(
        out_shape.begin(), out_shape.end() - 1);
    eigenvalues = eigenvalues.reshape(val_shape);

    /* ── Step 9: Return ────────────────────────────────────────────
     *
     * Return (eigenvalues, eigenvectors) as a tuple, matching the
     * signature of torch.linalg.eigh.
     */
    return std::make_tuple(eigenvalues, eigenvectors);
}


/* ================================================================
 * SECTION 4: PYTHON BINDING
 * ================================================================
 *
 * PYBIND11_MODULE creates the bridge between C++ and Python.
 *
 * When Python does `ext = load(...)`, pybind11 creates a Python
 * module. m.def() registers our C++ function so Python can call it:
 *
 *   ext.eigh_xsyev_batched(tensor, upper=False)
 *
 * TORCH_EXTENSION_NAME is a macro set by PyTorch's build system
 * to the name we passed to load() (e.g., "eigh_batched_ext").
 *
 * py::arg("input") etc. define keyword argument names for Python.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "eigh_xsyev_batched",         // Python function name
        &eigh_xsyev_batched,          // pointer to the C++ function
        "Batched symmetric eigendecomposition via "
        "cusolverDnXsyevBatched (no n<=32 gate). "
        "Input: (B, n, n) symmetric CUDA tensor. "
        "Returns: (eigenvalues (B, n), eigenvectors (B, n, n)).",
        py::arg("input"),              // first argument name
        py::arg("upper") = false       // second arg with default
    );
}
