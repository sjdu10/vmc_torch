"""
Test script to inspect the tag structure of quimb's
compute_x_environments output for a 4x4 PEPS.
"""

import numpy as np
import quimb.tensor as qtn

# 1. Create a random 4x4 PEPS with bond dimension 4, physical dimension 2
Lx, Ly = 4, 4
bond_dim = 4
phys_dim = 2

peps = qtn.PEPS.rand(Lx, Ly, bond_dim=bond_dim, phys_dim=phys_dim,
                      dtype='float64', seed=42)

print("=" * 70)
print("PEPS info:")
print(f"  Lx={Lx}, Ly={Ly}, bond_dim={bond_dim}, phys_dim={phys_dim}")
print(f"  Number of tensors: {peps.num_tensors}")
print(f"  Sites: {peps.sites}")
print()

# 2. Select physical indices (isel) with random config
config = np.random.randint(0, phys_dim, size=Lx * Ly)
isel_map = {
    peps.site_ind(site): int(config[i])
    for i, site in enumerate(peps.sites)
}
peps_selected = peps.isel(isel_map)

print("After isel (physical index selection):")
print(f"  Number of tensors: {peps_selected.num_tensors}")
print()

# Print tags and shapes of all tensors in the selected PEPS
print("Selected PEPS tensor info:")
for i, t in enumerate(peps_selected.tensors):
    print(f"  Tensor {i}: tags={t.tags}, shape={t.shape}, inds={t.inds}")
print()

# 3. Compute x environments
print("=" * 70)
print("Computing x environments with max_bond=8, mode='mps', canonize=True")
x_envs = peps_selected.compute_x_environments(
    max_bond=8, mode='mps', canonize=True
)

print(f"\nNumber of keys in x_envs: {len(x_envs)}")
print(f"Keys: {sorted(x_envs.keys())}")
print()

# 4. For each key, print details
for key in sorted(x_envs.keys()):
    env = x_envs[key]
    print("-" * 70)
    print(f"Key: {key}")
    print(f"  Type: {type(env).__name__}")
    print(f"  Number of tensors: {env.num_tensors}")

    # Print info for each tensor (first 2 only for brevity)
    for j, t in enumerate(env.tensors):
        if j < 2:
            print(f"  Tensor {j}:")
            print(f"    tags   = {t.tags}")
            print(f"    shape  = {t.shape}")
            print(f"    inds   = {t.inds}")
            print(f"    dtype  = {t.dtype}")
        elif j == 2:
            print(f"  ... ({env.num_tensors - 2} more tensors)")
            break

    # Also print ALL tensor tags as a summary
    all_tags = [t.tags for t in env.tensors]
    print(f"  All tensor tags: {all_tags}")
    print()
