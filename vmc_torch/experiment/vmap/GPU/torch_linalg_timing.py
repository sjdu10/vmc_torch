import os
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL to 1 thread for better timing accuracy
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP to 1 thread
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Limit OpenBLAS to 1 thread
import torch
import time
import pandas as pd
w_n = 12
w_val = 12
def benchmark_pytorch_linalg(device='cuda', batch_size=1024, n_range=range(16, 129, 8)):
    """
    Benchmarks torch.linalg.eigh and torch.linalg.svd for batched matrices (B, n, n).
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        device = 'cpu'
    
    results = []
    print(f"Benchmarking torch linalg on {device} with Batch Size B={batch_size}...")
    print(f"{'Matrix dim n':>{w_n}} | {'eigh (ms)':>{w_val}} | {'svd (ms)':>{w_val}}")
    print("-" * (w_n + 3 + w_val + 3 + w_val))

    for n in n_range:
        # 1. Prepare batched matrices
        # For eigh, we need Hermitian/Symmetric matrices
        x = torch.randn(batch_size, n, n, device=device)
        a_eigh = (x + x.transpose(-1, -2)) / 2.0
        # For svd, any matrix works
        a_svd = torch.randn(batch_size, n, n, device=device)

        # 2. Warmup
        for _ in range(5):
            _ = torch.linalg.eigh(a_eigh)
            _ = torch.linalg.svd(a_svd)
        
        if device == 'cuda':
            torch.cuda.synchronize()

        # 3. Timing eigh
        start_event = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
        end_event = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
        
        n_iters = 10
        
        if device == 'cuda':
            start_event.record()
            for _ in range(n_iters):
                _ = torch.linalg.eigh(a_eigh)
            end_event.record()
            torch.cuda.synchronize()
            eigh_time = start_event.elapsed_time(end_event) / n_iters
        else:
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = torch.linalg.eigh(a_eigh)
            eigh_time = (time.perf_counter() - t0) * 1000 / n_iters

        # 4. Timing svd
        if device == 'cuda':
            start_event.record()
            for _ in range(n_iters):
                # Only compute singular values for speed, or full SVD if needed
                _ = torch.linalg.svd(a_svd)
            end_event.record()
            torch.cuda.synchronize()
            svd_time = start_event.elapsed_time(end_event) / n_iters
        else:
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = torch.linalg.svd(a_svd)
            svd_time = (time.perf_counter() - t0) * 1000 / n_iters

        print(f"{n:{w_n}d} | {eigh_time:{w_val}.3f} | {svd_time:{w_val}.3f}")
        results.append({"n": n, "eigh_ms": eigh_time, "svd_ms": svd_time})

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Test around the 32 threshold mentioned in JAX issue
    # and extend to larger n to see the scaling
    n_list = list(range(28, 36))
    df = benchmark_pytorch_linalg(device='cpu', batch_size=64, n_range=n_list)
    
    # You can plot this using df.plot(x='n', y=['eigh_ms', 'svd_ms'], logy=True)