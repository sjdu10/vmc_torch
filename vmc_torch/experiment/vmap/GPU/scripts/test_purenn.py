"""
test_purenn.py — Smoke test for PureNN_GPU on 2x2 Hubbard with 4 fermions.

Checks that the VMC loop runs end-to-end and energy decreases.

Run (single GPU, no torchrun needed):
    python test_purenn.py

Run (multi-GPU):
    torchrun --nproc_per_node=2 test_purenn.py
"""
import os
import time
import numpy as np
import torch
import torch.distributed as dist

from vmc_torch.experiment.vmap.GPU.models import PureNN_GPU
from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    random_initial_config,
    sample_next,
    evaluate_energy,
    compute_grads_gpu,
)
from vmc_torch.experiment.vmap.GPU.vmc_modules import (
    run_sampling_phase_gpu,
)
from vmc_torch.hamiltonian_torch import (
    spinful_Fermi_Hubbard_square_lattice_torch,
)

# ==========================================
# Distributed setup (works with or without torchrun)
# ==========================================
def setup_distributed():
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12399"
        os.environ["LOCAL_RANK"] = "0"

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


RANK, WORLD_SIZE, device = setup_distributed()
torch.set_default_device(device)
torch.manual_seed(1234 + RANK)

# ==========================================
# Physics: 2x2 Hubbard, half-filling (N_f=4)
# ==========================================
Lx, Ly = 2, 2
nsites = Lx * Ly   # 4
N_f = nsites        # 4 fermions (2 up, 2 down)
t_hop, U = 1.0, 8.0
n_fermions_per_spin = (N_f // 2, N_f // 2)  # (2, 2)

H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, t_hop, U, N_f,
    pbc=False,
    n_fermions_per_spin=n_fermions_per_spin,
    no_u1_symmetry=False,
    gpu=True,
)
graph = H.graph
H.precompute_hops_gpu(device)

# ==========================================
# Model: PureNN_GPU (no PEPS file needed)
# ==========================================
model = PureNN_GPU(
    n_sites=nsites,
    phys_dim=4,     # 0=empty, 1=up, 2=down, 3=doubly-occupied
    embed_dim=8,
    hidden_dim=64,
    n_layers=2,
    dtype=torch.float64,
)
model.to(device)
n_params = sum(p.numel() for p in model.parameters())

if RANK == 0:
    print(f"\n{'='*50}")
    print(f"PureNN_GPU test: 2x2 Hubbard t={t_hop} U={U} N_f={N_f}")
    print(f"n_params = {n_params}")
    print(f"World size: {WORLD_SIZE}, device: {device}")
    print(f"{'='*50}")

# ==========================================
# VMC settings
# ==========================================
B = 64                   # walkers per rank
Ns_per_rank = 512        # samples per rank per step
grad_batch_size = 32     # chunk size for grad computation
vmc_steps = 20
learning_rate = 1e-3     # SGD lr (no metric rescaling)
burn_in_steps = 10

# Initial configs: (B, n_sites) int64
fxs = torch.stack([
    random_initial_config(N_f, nsites, seed=42 + RANK * B + i)
    for i in range(B)
]).to(device)

# ==========================================
# Sanity check: forward pass shapes
# ==========================================
with torch.no_grad():
    amps = model(fxs)
    assert amps.shape == (B,), f"Expected ({B},), got {amps.shape}"
    assert not torch.isnan(amps).any(), "NaN in amplitudes!"
if RANK == 0:
    print(f"\nForward pass OK: amps.shape={amps.shape}, dtype={amps.dtype}")
    print(f"  amp range: [{amps.min().item():.4f}, {amps.max().item():.4f}]")

# ==========================================
# Sanity check: gradient shapes
# ==========================================
with torch.enable_grad():
    grads_test, amps_test = compute_grads_gpu(
        fxs, model, vectorize=True,
        batch_size=grad_batch_size, vmap_grad=True,
    )
assert grads_test.shape == (B, n_params), \
    f"Expected ({B}, {n_params}), got {grads_test.shape}"
if RANK == 0:
    g_norm = grads_test.norm().item()
    g_max  = grads_test.abs().max().item()
    print(f"Gradient pass OK: grads.shape={grads_test.shape}"
          f"  ||G||={g_norm:.3e}  max|G|={g_max:.3e}")
del grads_test, amps_test

# ==========================================
# VMC loop
# ==========================================
if RANK == 0:
    print(
        f"\n--- VMC SGD ({vmc_steps} steps, B={B}, "
        f"Ns/rank={Ns_per_rank}, lr={learning_rate}) ---"
    )

energy_history = []

for step in range(vmc_steps):
    t0 = time.time()

    # Step 1: Sample + energy + grads (all ranks)
    (local_energies, local_O), fxs, t_samp = run_sampling_phase_gpu(
        fxs=fxs,
        model=model,
        hamiltonian=H,
        graph=graph,
        Ns=Ns_per_rank,
        grad_batch_size=grad_batch_size,
        burn_in=(step == 0),
        burn_in_steps=burn_in_steps,
        verbose=False,
    )

    # Step 2: Global energy (local_energies is now a GPU tensor)
    Total_Ns = local_energies.shape[0] * WORLD_SIZE
    local_E_sum = local_energies.sum()   # already on device
    dist.all_reduce(local_E_sum, op=dist.ReduceOp.SUM)
    energy_mean = local_E_sum.item() / Total_Ns

    local_E_sq = (local_energies ** 2).sum()   # already on device
    dist.all_reduce(local_E_sq, op=dist.ReduceOp.SUM)
    energy_var = local_E_sq.item() / Total_Ns - energy_mean ** 2

    # Step 3: Raw energy gradient  F = <E_loc * O_loc> - <E_loc> * <O_loc>
    t_grad = time.time()
    # local_O is (Ns, Np) GPU tensor; local_energies is (Ns,) GPU tensor
    EO_t = (local_energies.unsqueeze(1) * local_O).sum(dim=0)  # (Np,)
    O_sum_t = local_O.sum(dim=0)                               # (Np,)
    dist.all_reduce(EO_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(O_sum_t, op=dist.ReduceOp.SUM)

    mean_EO = EO_t / Total_Ns
    mean_O = O_sum_t / Total_Ns
    energy_grad = mean_EO - energy_mean * mean_O  # (Np,) on device
    t_grad = time.time() - t_grad

    # Step 4: SGD parameter update  θ ← θ - lr * F
    with torch.no_grad():
        cur = torch.nn.utils.parameters_to_vector(model.parameters())
        new = cur - learning_rate * energy_grad
        torch.nn.utils.vector_to_parameters(new, model.parameters())

    t1 = time.time()

    e_per_site = energy_mean / nsites
    energy_history.append(e_per_site)

    if RANK == 0:
        err = np.sqrt(max(energy_var, 0) / Total_Ns) / nsites
        grad_norm = energy_grad.norm().item()
        print(
            f"Step {step:3d} | E/site={e_per_site:+.6f} +/- {err:.6f}"
            f" | |grad|={grad_norm:.3e}"
            f" | T_samp={t_samp:.2f}s T_grad={t_grad:.3f}s"
        )

# ==========================================
# Summary
# ==========================================
if RANK == 0:
    print(f"\n{'='*50}")
    print(f"First E/site: {energy_history[0]:+.6f}")
    print(f"Last  E/site: {energy_history[-1]:+.6f}")
    print(f"Min   E/site: {min(energy_history):+.6f}")
    delta = energy_history[-1] - energy_history[0]
    print(f"Delta E/site: {delta:+.6f}")
    # Reference: exact GS for 2x2 Hubbard t=1 U=8 (half-filling, OBC)
    # E_exact/site ≈ -0.9 (rough estimate, varies with boundary conditions)
    if delta < 0:
        print("\nPASSED: Energy decreased.")
    else:
        print("\nWARNING: Energy did NOT decrease.")
    print(f"{'='*50}\n")

dist.destroy_process_group()
