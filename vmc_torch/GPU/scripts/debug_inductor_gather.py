"""Test if vmap+inductor gather is buggy for dim=5.

vmap converts index_select → gather. D=10 Z2 symmetry
gives block dimension 5. If gather's Triton kernel has
a bug for non-power-of-2 dims, this would explain why
D=10 fails but D=8 (dim=4) and D=12 (dim=6) pass.

Run:
    python debug_inductor_gather.py
"""
import torch

torch.set_default_dtype(torch.float64)
device = torch.device('cuda:0')
torch.set_default_device(device)
torch.manual_seed(42)

B = 4


def test_gather(name, src_shape, idx_shape, dim,
                n_trials=5):
    """Test vmap(index_select) → gather through inductor."""

    def fn(src, idx):
        return torch.index_select(src, dim, idx)

    vmapped = torch.vmap(fn)

    results = []
    for trial in range(n_trials):
        torch.manual_seed(trial)
        src = torch.randn(
            B, *src_shape, dtype=torch.float64,
            device=device,
        )
        idx = torch.randint(
            0, src_shape[dim], (B, idx_shape),
            device=device,
        )

        with torch.inference_mode():
            eager = vmapped(src, idx)

        torch._dynamo.reset()
        compiled = torch.compile(vmapped, backend='inductor')
        with torch.inference_mode():
            ind = compiled(src, idx)

        diff = (eager - ind).abs().max().item()
        results.append(diff)

    max_diff = max(results)
    ok = max_diff < 1e-10
    print(f"  {name:30s} src={src_shape} idx=({idx_shape},) "
          f"dim={dim}: max_diff={max_diff:.2e} "
          f"{'PASS' if ok else '*** FAIL ***'}")
    return ok


print("=== Testing gather via vmap(index_select) ===\n")

# Test various dimension sizes
all_pass = True
for dim_size in [3, 4, 5, 6, 7, 8, 10, 16, 32]:
    ok = test_gather(
        f"1D dim={dim_size}",
        src_shape=(dim_size,),
        idx_shape=2,
        dim=0,
    )
    all_pass &= ok

print()
# Test 2D with various sizes (simulates symmray blocks)
for d in [3, 4, 5, 6, 7, 8]:
    ok = test_gather(
        f"2D ({d},{d})",
        src_shape=(d, d),
        idx_shape=2,
        dim=0,
    )
    all_pass &= ok

print()
# Test with shapes matching D=10 Z2 block structure
# (5, 5, 2) is a corner tensor block
for shape in [(5, 5, 2), (5, 5, 5, 2), (5, 5, 5, 5, 2)]:
    for dim in range(len(shape)):
        ok = test_gather(
            f"D10-like {shape} dim={dim}",
            src_shape=shape,
            idx_shape=1,
            dim=dim,
        )
        all_pass &= ok

print()
# Test with actual block counts from D=10
# (4 blocks of shape (5,5,2), 8 blocks of (5,5,5,2))
for n_blocks, block_shape in [
    (4, (5, 5, 2)),
    (8, (5, 5, 5, 2)),
    (16, (5, 5, 5, 5, 2)),
]:
    ok = test_gather(
        f"blocks={n_blocks} shape={block_shape}",
        src_shape=(n_blocks, *block_shape),
        idx_shape=2,
        dim=0,
    )
    all_pass &= ok

print(f"\n{'=' * 50}")
print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
print(f"{'=' * 50}")
