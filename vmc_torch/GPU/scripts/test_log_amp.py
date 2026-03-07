"""
test_log_amp.py — Verification tests for the log-amplitude refactor.

Tests:
  1. sign * exp(log_abs) == amplitude(x) for PureNN_GPU
  2. grad(log|psi|) == grad(psi)/psi (log-derivative equivalence)
  3. Metropolis accept/reject equivalence (same seed, same decisions)
  4. E_loc equivalence (same local energies)
  5. Full VMC step (2 steps, energy is finite and sensible)

Run (single GPU):
    python scripts/test_log_amp.py
"""
import os
import time
import torch
import torch.distributed as dist
import numpy as np

from vmc_torch.GPU.models import PureNN_GPU
from vmc_torch.GPU.vmc_utils import (
    random_initial_config,
    evaluate_energy,
    compute_grads_gpu,
)
from vmc_torch.GPU.sampler import MetropolisExchangeSpinfulSamplerGPU
from vmc_torch.hamiltonian_torch import (
    spinful_Fermi_Hubbard_square_lattice_torch,
)


# ==========================================
# Setup
# ==========================================
def setup_distributed():
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
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
torch.set_default_dtype(torch.float64)
torch.manual_seed(42 + RANK)

# Physics: 2x2 Hubbard
Lx, Ly = 4, 2
nsites = Lx * Ly
N_f = nsites
t_hop, U = 1.0, 8.0
n_fermions_per_spin = (N_f // 2, N_f // 2)

H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, t_hop, U, N_f, pbc=False,
    n_fermions_per_spin=n_fermions_per_spin,
    no_u1_symmetry=False, gpu=True,
)
graph = H.graph
H.precompute_hops_gpu(device)

# Model
model = PureNN_GPU(
    n_sites=nsites, phys_dim=4,
    embed_dim=8, hidden_dim=32, n_layers=2,
    dtype=torch.float64,
)
model.to(device)
n_params = sum(p.numel() for p in model.parameters())

# Walkers
B = 512
fxs = torch.stack([
    random_initial_config(N_f, nsites, seed=42 + RANK * B + i)
    for i in range(B)
]).to(device)


def PASS(name):
    print(f"  [PASS] {name}")


def FAIL(name, msg=""):
    print(f"  [FAIL] {name}: {msg}")


n_pass, n_fail = 0, 0
results = []

print(f"\n{'='*60}")
print(f"Log-Amplitude Refactor Verification")
print(f"  System: {Lx}x{Ly} Hubbard t={t_hop} U={U}")
print(f"  Model: PureNN_GPU, {n_params} params")
print(f"  B={B} walkers, device={device}")
print(f"{'='*60}\n")


# ==========================================
# Test 1: sign * exp(log_abs) == forward(x)
# ==========================================
print("Test 1: forward_log consistency")
with torch.no_grad():
    amps = model(fxs)
    signs, log_abs = model.forward_log(fxs)
    reconstructed = signs * torch.exp(log_abs)
    err = (amps - reconstructed).abs().max().item()

if err < 1e-12:
    PASS(f"forward_log consistency: max_err={err:.2e}")
    n_pass += 1
else:
    FAIL(f"forward_log consistency", f"max_err={err:.2e}")
    n_fail += 1

# Also test vamp_log
with torch.no_grad():
    params = list(model.params)
    signs2, log_abs2 = model.vamp_log(fxs, params)
    reconstructed2 = signs2 * torch.exp(log_abs2)
    err2 = (amps - reconstructed2).abs().max().item()

if err2 < 1e-12:
    PASS(f"vamp_log consistency: max_err={err2:.2e}")
    n_pass += 1
else:
    FAIL(f"vamp_log consistency", f"max_err={err2:.2e}")
    n_fail += 1


# ==========================================
# Test 2: grad(log|psi|) == grad(psi)/psi
# ==========================================
print("\nTest 2: Log-derivative equivalence")
grad_batch = min(B, 16)

# Standard: grad(psi) / psi
with torch.enable_grad():
    grads_std, amps_std = compute_grads_gpu(
        fxs, model, vectorize=True,
        batch_size=grad_batch, vmap_grad=True,
        use_log_amp=False,
    )
# O_loc = grads / amps
oloc_std = grads_std / amps_std.unsqueeze(1)

# Log-amp: grad(log|psi|) directly
with torch.enable_grad():
    grads_log, (signs_log, log_abs_log) = compute_grads_gpu(
        fxs, model, vectorize=True,
        batch_size=grad_batch, vmap_grad=True,
        use_log_amp=True,
    )

# grads_log is already the log-derivative
diff = (oloc_std - grads_log).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()
rel_diff = max_diff / (oloc_std.abs().max().item() + 1e-30)

if rel_diff < 1e-8:
    PASS(f"log-derivative: max_abs_diff={max_diff:.2e}, "
         f"rel_diff={rel_diff:.2e}")
    n_pass += 1
else:
    FAIL(f"log-derivative", f"rel_diff={rel_diff:.2e}")
    n_fail += 1


# ==========================================
# Test 3: Metropolis accept/reject equivalence
# ==========================================
print("\nTest 3: Metropolis equivalence")
sampler = MetropolisExchangeSpinfulSamplerGPU()

# Both paths start from the same fxs, same model state
fxs_a = fxs.clone()
fxs_b = fxs.clone()

# Standard path
torch.manual_seed(123 + RANK)
with torch.no_grad():
    fxs_a_out, amps_a = sampler.step(
        fxs_a, model, graph, use_log_amp=False,
    )

# Log-amp path
torch.manual_seed(123 + RANK)
with torch.no_grad():
    fxs_b_out, (signs_b, log_abs_b) = sampler.step(
        fxs_b, model, graph, use_log_amp=True,
    )

# Compare: configs should be identical
configs_match = torch.all(fxs_a_out == fxs_b_out).item()
if configs_match:
    PASS("Metropolis configs match exactly")
    n_pass += 1
else:
    n_mismatch = (fxs_a_out != fxs_b_out).any(dim=1).sum().item()
    FAIL("Metropolis configs", f"{n_mismatch}/{B} walkers differ")
    n_fail += 1

# Verify reconstructed amps match
amps_b_reconstructed = signs_b * torch.exp(log_abs_b)
amp_err = (amps_a - amps_b_reconstructed).abs().max().item()
if amp_err < 1e-10:
    PASS(f"Metropolis amps match: max_err={amp_err:.2e}")
    n_pass += 1
else:
    FAIL(f"Metropolis amps", f"max_err={amp_err:.2e}")
    n_fail += 1


# ==========================================
# Test 4: E_loc equivalence
# ==========================================
print("\nTest 4: E_loc equivalence")
# Use the same configs from standard sampler output
with torch.no_grad():
    amps_for_eloc = model(fxs_a_out)
    signs_for_eloc, log_abs_for_eloc = model.forward_log(fxs_a_out)

# Standard
with torch.no_grad():
    e_std, eloc_std = evaluate_energy(
        fxs_a_out, model, H, amps_for_eloc,
        use_log_amp=False,
    )

# Log-amp
with torch.no_grad():
    e_log, eloc_log = evaluate_energy(
        fxs_a_out, model, H,
        (signs_for_eloc, log_abs_for_eloc),
        use_log_amp=True,
    )

eloc_diff = (eloc_std - eloc_log).abs()
max_eloc_diff = eloc_diff.max().item()
rel_eloc_diff = max_eloc_diff / (eloc_std.abs().max().item() + 1e-30)

if rel_eloc_diff < 1e-10:
    PASS(f"E_loc match: max_abs_diff={max_eloc_diff:.2e}, "
         f"rel={rel_eloc_diff:.2e}")
    n_pass += 1
else:
    FAIL(f"E_loc", f"rel_diff={rel_eloc_diff:.2e}")
    n_fail += 1

e_diff = abs(e_std.item() - e_log.item())
if e_diff < 1e-10:
    PASS(f"Mean energy match: diff={e_diff:.2e}")
    n_pass += 1
else:
    FAIL(f"Mean energy", f"diff={e_diff:.2e}")
    n_fail += 1


# ==========================================
# Test 5: Full VMC step with log-amp
# ==========================================
print("\nTest 5: Full VMC step (2 steps)")
from vmc_torch.GPU.vmc_modules import run_sampling_phase_gpu
from vmc_torch.GPU.optimizer import (
    distributed_minres_solver_gpu,
)

fxs_vmc = fxs.clone()
energy_history = []

for step in range(2):
    # Sampling phase with log-amp
    (local_energies, local_lpg), fxs_vmc, sample_time, _ = (
        run_sampling_phase_gpu(
            fxs=fxs_vmc,
            model=model,
            hamiltonian=H,
            graph=graph,
            Ns=B,
            grad_batch_size=16,
            burn_in=False,
            verbose=False,
            use_log_amp=True,
        )
    )

    # Global energy
    Total_Ns = local_energies.shape[0] * WORLD_SIZE
    local_E_sum = local_energies.sum()
    dist.all_reduce(local_E_sum, op=dist.ReduceOp.SUM)
    energy_mean = local_E_sum.item() / Total_Ns

    energy_history.append(energy_mean)

    # SR solve
    dp, t_sr, info = distributed_minres_solver_gpu(
        local_lpg=local_lpg,
        local_energies=local_energies,
        energy_mean=energy_mean,
        total_samples=Total_Ns,
        n_params=n_params,
        diag_shift=1e-4,
    )

    # Check dp is finite
    if not torch.isfinite(dp).all():
        FAIL(f"VMC step {step}", "dp contains NaN/Inf")
        n_fail += 1
        break

    # Update params
    lr = 0.1
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.add_(dp[offset:offset + numel].reshape(p.shape), alpha=-lr)
            offset += numel

    print(f"  Step {step}: E/site={energy_mean / nsites:.6f}")

# Check no NaN
has_nan = any(np.isnan(e) for e in energy_history)
has_inf = any(np.isinf(e) for e in energy_history)
all_finite = not has_nan and not has_inf

if all_finite and len(energy_history) == 2:
    PASS(f"VMC loop: 2 steps, energies finite "
         f"({energy_history[0]/nsites:.4f}, "
         f"{energy_history[1]/nsites:.4f})")
    n_pass += 1
else:
    FAIL(f"VMC loop", f"nan={has_nan}, inf={has_inf}")
    n_fail += 1

# Check local_lpg is finite and has reasonable norms
lpg_finite = torch.isfinite(local_lpg).all().item()
lpg_norm = local_lpg.norm().item()
if lpg_finite and lpg_norm < 1e10:
    PASS(f"log_psi_grad finite, norm={lpg_norm:.2e}")
    n_pass += 1
else:
    FAIL(f"log_psi_grad", f"finite={lpg_finite}, norm={lpg_norm:.2e}")
    n_fail += 1


# ==========================================
# Summary
# ==========================================
print(f"\n{'='*60}")
print(f"Results: {n_pass} passed, {n_fail} failed "
      f"out of {n_pass + n_fail} tests")
print(f"{'='*60}")

dist.destroy_process_group()

if n_fail > 0:
    exit(1)
