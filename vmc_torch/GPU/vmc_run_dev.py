import os
import time
import json
import pickle
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

# Third-party libraries
import autoray as ar
import quimb.tensor as qtn

# User-defined modules
from vmc_torch.GPU.vmc_utils import (
    random_initial_config
)
from vmc_torch.GPU.vmc_modules import (
    run_sampling_phase_gpu,
    distributed_minres_solver_gpu,
    minSR_solver_gpu,
)
from vmc_torch.GPU.models import (
    fPEPS_Model_GPU,
)
from vmc_torch.hamiltonian_torch import (
    spinful_Fermi_Hubbard_square_lattice_torch
)
from vmc_torch.GPU.torch_utils import (
    size_aware_qr,
    size_aware_svd,
)

from vmc_torch.GPU.vmc_utils import (
    sample_next, evaluate_energy, compute_grads_gpu,
)

# --- Global Configurations ---
JITTER = 1e-16
driver = None
ar.register_function(
    'torch', 'linalg.svd',
    lambda x: size_aware_svd(x, jitter=JITTER, driver=driver),
)
ar.register_function('torch', 'linalg.qr', size_aware_qr)
dtype = torch.float64
torch.set_default_dtype(dtype)

# ==========================================
# 1. Distributed Environment Setup
# ==========================================
def setup_distributed():
    if "RANK" not in os.environ:
        print("Warning: Not using torchrun. Single device.")
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
torch.manual_seed(42 + RANK)

# ==========================================
# 2. Physics & Model Configuration
# ==========================================
Lx, Ly = 4, 2
nsites = Lx * Ly
N_f = nsites - 2
D = 4
chi = -1
try:
    pwd = (
        '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
        'vmc_torch/experiment/vmap/data'
    )
    u1z2 = True
    appendix = '_U1SU' if u1z2 else ''

    params_path = (
        f"{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/"
        f"peps_su_params{appendix}.pkl"
    )
    skeleton_path = (
        f"{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/"
        f"peps_skeleton{appendix}.pkl"
    )

    with open(params_path, 'rb') as f:
        params_pkl = pickle.load(f)
    with open(skeleton_path, 'rb') as f:
        skeleton = pickle.load(f)

    peps = qtn.unpack(params_pkl, skeleton)

    for ts in peps.tensors:
        ts.modify(data=ts.data.to_flat() * 4)
    for site in peps.sites:
        peps[site].data._label = site
        peps[site].data.indices[-1]._linearmap = (
            (0, 0), (1, 0), (1, 1), (0, 1)
        )
except Exception as e:
    import symmray as sr
    peps = sr.networks.PEPS_fermionic_rand(
        "Z2",
        Lx,
        Ly,
        D,
        phys_dim=[
            (0, 0),
            (1, 1),
            (1, 0),
            (0, 1),
        ],
        subsizes="equal",
        flat=True,
        seed=42,
        dtype=str(dtype).split(".")[-1],
    )

# Use GPU-specific model
fpeps_model = fPEPS_Model_GPU(
    tn=peps, max_bond=chi, dtype=dtype,
    contract_boundary_opts={
        'mode': 'mps',
        'equalize_norms': 1.0,
        'canonize': True,
    }
)
fpeps_model.to(device)

# Optional: export + compile for GPU speedup
# torch.export captures quimb/symmray ops as pure aten graph,
# then vmap + compile fuses into CUDA kernels (~10x for chi=-1)
USE_EXPORT_COMPILE = True
if USE_EXPORT_COMPILE:
    example_x = random_initial_config(N_f, nsites, seed=0).to(device)
    if RANK == 0:
        print("Running torch.export + compile...")
    t_ec = time.time()
    fpeps_model.export_and_compile(example_x, mode='default')
    if RANK == 0:
        print(f"  export_and_compile: {time.time()-t_ec:.1f}s")

n_params = sum(p.numel() for p in fpeps_model.parameters())
if RANK == 0:
    print(
        f'Model parameters: {n_params} | '
        f'World Size: {WORLD_SIZE} | Device: {device}'
    )

# Hamiltonian
t, U = 1.0, 8.0
n_fermions_per_spin = (N_f // 2, N_f // 2)
H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, t, U, N_f,
    pbc=False,
    n_fermions_per_spin=n_fermions_per_spin,
    no_u1_symmetry=False,
    gpu=True,
)
graph = H.graph
H.precompute_hops_gpu(device)

# ==========================================
# 3. Sampling & VMC Settings
# ==========================================
B = 1024
Ns_per_rank = 2048
grad_batch_size = max(1, B // 2)

if RANK == 0:
    Total_Ns_expected = Ns_per_rank * WORLD_SIZE
    n_sweeps = int(np.ceil(Ns_per_rank / B))
    print(
        f"B={B}, Ns_per_rank={Ns_per_rank}, "
        f"sweeps/rank={n_sweeps}, "
        f"Total_Ns~{Total_Ns_expected}, "
        f"grad_batch={grad_batch_size}"
    )

# Initialize walkers on GPU (B walkers)
fxs = torch.stack([
    random_initial_config(N_f, nsites, seed=42 + RANK * B + i)
    for i in range(B)
]).to(device)

# Training Hyperparameters
vmc_steps = 10
learning_rate = 0.1
diag_shift = 1e-4
save_state_every = 10
burn_in_steps = 0
run_SR = True
use_minSR = False  # False: distributed MINRES

# Output paths
output_dir = (
    f"{pwd}/GPU/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}"
)
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 4. Warmup run (torch.compile warmup)
# ==========================================
if RANK == 0:
    print("\n--- Warmup (1 sweep) ---")
t_warm = time.time()

with torch.inference_mode():
    fxs, amps = sample_next(fxs, fpeps_model, graph, compile=USE_EXPORT_COMPILE)
    if RANK == 0:
        print(f"  sample_next:     {time.time()-t_warm:.2f}s")
    t1 = time.time()
    _, evals = evaluate_energy(fxs, fpeps_model, H, amps)
    if RANK == 0:
        print(f"  evaluate_energy: {time.time()-t1:.2f}s")
t2 = time.time()
with torch.enable_grad():
    grads, amps2 = compute_grads_gpu(
        fxs, fpeps_model, vectorize=True,
        batch_size=grad_batch_size, vmap_grad=True,
    )
if RANK == 0:
    print(f"  compute_grads:   {time.time()-t2:.2f}s")
    print(f"  Warmup total:    {time.time()-t_warm:.2f}s")
del grads, amps2, evals

# ==========================================
# 5. Main VMC Loop
# ==========================================
if RANK == 0:
    print(f"\n--- VMC ({vmc_steps} steps) ---")
    vmc_pbar = tqdm(total=vmc_steps, desc="VMC Steps")

energy_history = []

for step in range(vmc_steps):
    t0 = time.time()

    # --- Step 1: Sampling Phase ---
    (local_energies, local_O), fxs, sample_time = (
        run_sampling_phase_gpu(
            fxs=fxs,
            model=fpeps_model,
            hamiltonian=H,
            graph=graph,
            Ns=Ns_per_rank,
            grad_batch_size=grad_batch_size,
            burn_in=(step == 0),
            burn_in_steps=burn_in_steps,
            verbose=False,
            compile=USE_EXPORT_COMPILE
        )
    )
    # print(f'local O norm: {np.linalg.norm(local_O):.3e}') BUG: local_O is 0, grad not passing through??
    t_sample_end = time.time()

    # --- Step 2: Global energy statistics ---
    n_local = local_energies.shape[0]
    Total_Ns = n_local * WORLD_SIZE

    local_E_sum = local_energies.sum()   # already on device
    dist.all_reduce(local_E_sum, op=dist.ReduceOp.SUM)
    energy_mean = local_E_sum.item() / Total_Ns

    local_E_sq_sum = (local_energies ** 2).sum()   # already on device
    dist.all_reduce(local_E_sq_sum, op=dist.ReduceOp.SUM)
    energy_var = (
        local_E_sq_sum.item() / Total_Ns - energy_mean ** 2
    )

    # --- Step 3: SR Solve ---
    t_sr_start = time.time()
    if use_minSR:
        dp, t_sr, info = minSR_solver_gpu(
            local_lpg=local_O,
            local_energies=local_energies,
            energy_mean=energy_mean,
            total_samples=Total_Ns,
            n_params=n_params,
            diag_shift=diag_shift,
            device=device,
            run_SR=run_SR
        )
    else:
        dp, t_sr, info = distributed_minres_solver_gpu(
            local_lpg=local_O,
            local_energies=local_energies,
            energy_mean=energy_mean,
            total_samples=Total_Ns,
            n_params=n_params,
            diag_shift=diag_shift,
            rtol=5e-5,
            run_SR=run_SR
        )

    # --- Step 4: Parameter Update ---
    with torch.no_grad():
        dp_tensor = torch.tensor(
            dp, device=device, dtype=torch.float64
        )
        current_params_vec = (
            torch.nn.utils.parameters_to_vector(
                fpeps_model.parameters()
            )
        )
        new_params_vec = (
            current_params_vec - learning_rate * dp_tensor
        )
        torch.nn.utils.vector_to_parameters(
            new_params_vec, fpeps_model.parameters()
        )

    t1 = time.time()

    # --- Step 5: Logging ---
    e_per_site = energy_mean / nsites
    energy_history.append(e_per_site)

    if RANK == 0:
        err = np.sqrt(max(energy_var, 0) / Total_Ns) / nsites
        print(
            f"Step {step:3d} | E/site: {e_per_site:.6f} "
            f"+/- {err:.6f} | N={Total_Ns} | "
            f"T_samp={sample_time:.1f}s T_SR={t_sr:.2f}s "
            f"T_total={t1-t0:.1f}s"
        )
        vmc_pbar.update(1)

# Summary
if RANK == 0:
    vmc_pbar.close()
    print(f"\n{'='*50}")
    print(f"VERIFICATION ({Lx}x{Ly}, GPU)")
    print(f"{'='*50}")
    print(f"First E/site: {energy_history[0]:.6f}")
    print(f"Last E/site:  {energy_history[-1]:.6f}")
    print(f"Min E/site:   {min(energy_history):.6f}")
    if energy_history[-1] < energy_history[0]:
        print("\nPASSED: Energy decreased.")
    else:
        print("\nWARNING: Energy did NOT decrease.")

dist.destroy_process_group()
