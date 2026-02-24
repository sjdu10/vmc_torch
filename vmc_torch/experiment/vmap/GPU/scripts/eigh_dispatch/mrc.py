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