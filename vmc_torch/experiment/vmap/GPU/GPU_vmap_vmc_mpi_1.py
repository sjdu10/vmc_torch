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

# User-defined modules (Placeholder for future definitions)
from vmc_torch.experiment.vmap.GPU.GPU_vmap_utils import (
    sample_next, evaluate_energy, compute_grads_gpu, random_initial_config
)
from vmc_torch.experiment.vmap.vmap_models import PEPS_Model, fPEPS_Model
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.experiment.vmap.vmap_torch_utils import robust_svd_err_catcher_wrapper

# --- Global Configurations ---
JITTER = 1e-16
driver = None
# Register robust SVD for stability
ar.register_function('torch', 'linalg.svd', lambda x: robust_svd_err_catcher_wrapper(x, jitter=JITTER, driver=driver))

# Set default precision to double (FP64) for scientific accuracy
torch.set_default_dtype(torch.float64)

# ==========================================
# 1. Distributed Environment Setup
# ==========================================
def setup_distributed():
    """
    Initializes the distributed process group and sets the device.
    """
    if "RANK" not in os.environ:
        print("Warning: Not using torchrun. Defaulting to single device (Rank 0).")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["LOCAL_RANK"] = "0"

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Critical: Set the device for the current process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    return rank, world_size, device

RANK, WORLD_SIZE, device = setup_distributed()
torch.set_default_device(device)

# Set different seeds for different ranks to ensure independent sampling
torch.manual_seed(42 + RANK)

# ==========================================
# 2. Physics & Model Configuration
# ==========================================
Lx, Ly = 4, 2
nsites = Lx * Ly
N_f = nsites - 2
D = 4
chi = 8

# Path configurations
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/data'
u1z2 = True
appendix = '_U1SU' if u1z2 else ''

# Load Skeleton and Parameters
# Note: In a large-scale cluster, it is better to load on Rank 0 and broadcast, 
# but for local NVLink setups, concurrent read is acceptable.
params_path = f"{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/peps_su_params{appendix}.pkl"
skeleton_path = f"{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/peps_skeleton{appendix}.pkl"

with open(params_path, 'rb') as f:
    params_pkl = pickle.load(f)
with open(skeleton_path, 'rb') as f:
    skeleton = pickle.load(f)

peps = qtn.unpack(params_pkl, skeleton)

# Preprocessing PEPS tensors (CPU side)
for ts in peps.tensors:
    ts.modify(data=ts.data.to_flat() * 4)
for site in peps.sites:
    peps[site].data._label = site
    peps[site].data.indices[-1]._linearmap = ((0, 0), (1, 0), (1, 1), (0, 1))

# Initialize fPEPS Model and move to GPU
fpeps_model = fPEPS_Model(
    tn=peps, max_bond=chi, dtype=torch.float64,
    compile=False, # Set True if PyTorch 2.0+ compile is supported for TNs
    contract_boundary_opts={
        'mode': 'mps',
        'equalize_norms': 1.0,
        'canonize': True,
    }
)
fpeps_model.to(device)

# Parameter info
n_params = sum(p.numel() for p in fpeps_model.parameters())
if RANK == 0:
    print(f'Model parameters: {n_params} | World Size: {WORLD_SIZE} | Device: {device}')

# Hamiltonian Setup
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

# ==========================================
# 3. Sampling & VMC Settings
# ==========================================
Total_Ns = int(4096)  # Total number of samples across all ranks
assert Total_Ns % WORLD_SIZE == 0, f"Total samples {Total_Ns} must be divisible by World Size {WORLD_SIZE}"

samples_per_rank = Total_Ns // WORLD_SIZE
batch_size_per_rank = samples_per_rank # Adjust this if OOM occurs
grad_batch_size = 512

if RANK == 0:
    print(f"Sampling: {samples_per_rank} samples/rank | Batch size: {batch_size_per_rank}")

# Initialize walkers on GPU
fxs_list = [random_initial_config(N_f, nsites, seed=42 + i) for i in range(batch_size_per_rank)]
fxs = torch.stack(fxs_list).to(device)

# # Burn-in (Warmup) phase
# for _ in range(4):
#     fxs, current_amps = sample_next(fxs, fpeps_model, graph, seed=42)

# Training Hyperparameters
vmc_steps = 50
minSR = True
learning_rate = 0.1
save_state_every = 10

# Output paths
output_dir = f"{pwd}/GPU/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}"
os.makedirs(output_dir, exist_ok=True)
stats_file = f"{output_dir}/stats_{fpeps_model._get_name()}.json"
stats = {'Np': n_params, 'sample size': Total_Ns, 'mean': [], 'error': [], 'variance': []}

# ==========================================
# 4. Main VMC Loop (Fully on GPU)
# ==========================================
if RANK == 0:
    vmc_pbar = tqdm(total=vmc_steps, desc="VMC Steps")

# Pre-allocate buffers for gathering to avoid frequent memory re-allocation
# Note: 'total_grads' might be large. If OOM, we must switch to iterative solvers without gathering.
total_energies_buffer = torch.zeros(Total_Ns, device=device)
total_amps_buffer = torch.zeros(Total_Ns, device=device)
total_grads_buffer = torch.zeros((Total_Ns, n_params), device=device)

for step in range(vmc_steps):
    t0 = time.time()
    
    # --- A. Local Sampling & Evaluation ---
    # Pre-allocate local tensors
    local_energies = torch.zeros(samples_per_rank, device=device)
    local_amps = torch.zeros(samples_per_rank, device=device)
    local_grads = torch.zeros((samples_per_rank, n_params), device=device)

    current_count = 0
    
    while current_count < samples_per_rank:
        # Determine current batch size
        needed = min(batch_size_per_rank, samples_per_rank - current_count)
        
        # 1. Sample
        fxs, current_amps_batch = sample_next(fxs, fpeps_model, graph, seed=42 + step)
        
        # 2. Energy Evaluation
        # Ensure evaluate_energy returns detached tensors on GPU
        _, local_E_batch = evaluate_energy(fxs, fpeps_model, H, current_amps_batch)
        
        # 3. Gradient Computation
        # Note: local_grads_batch usually contains gradients of psi (not log_psi) if amps are returned
        local_grads_batch, local_amps_batch = compute_grads_gpu(
            fxs, fpeps_model, vectorize=True, batch_size=grad_batch_size
        )
        
        # 4. Fill buffers
        # Slice the batch in case batch_size > needed (unlikely here but good practice)
        idx_start = current_count
        idx_end = current_count + needed
        
        local_energies[idx_start:idx_end] = local_E_batch[:needed]
        local_grads[idx_start:idx_end] = local_grads_batch[:needed]
        local_amps[idx_start:idx_end] = local_amps_batch[:needed]
        
        current_count += needed

    # --- B. Global Aggregation (All-Gather) ---
    # We use all_gather_into_tensor for better efficiency (PyTorch > 1.7)
    dist.all_gather_into_tensor(total_energies_buffer, local_energies)
    dist.all_gather_into_tensor(total_amps_buffer, local_amps)
    dist.all_gather_into_tensor(total_grads_buffer, local_grads)

    # --- C. Stochastic Reconfiguration (SR) ---
    E_mean = torch.mean(total_energies_buffer)
    E_var = torch.var(total_energies_buffer)
    
    dp = torch.zeros(n_params, device=device, dtype=torch.float64)

    # Perform Linear Algebra on Rank 0
    if RANK == 0:
        # 1. Compute log-gradients: \nabla log(psi) = \nabla psi / psi
        # Warning: This can be numerically unstable if amplitudes are very small.
        # Ideally, compute_grads_gpu should return \nabla log(psi) directly.
        log_grads = total_grads_buffer / total_amps_buffer.unsqueeze(1) # (Total_Ns, Np)
        
        # 2. Centering: O_k = \nabla log(psi_k) - <\nabla log(psi)>
        log_grads_mean = torch.mean(log_grads, dim=0)
        O_centered = (log_grads - log_grads_mean.unsqueeze(0))
        
        # Rescale by 1/sqrt(Ns) for matrix construction
        O_sk = O_centered / np.sqrt(Total_Ns)
        E_s = (total_energies_buffer - E_mean) / np.sqrt(Total_Ns)
        
        # 3. Solve the SR equation: S * dp = g
        # Where S = O_sk^T @ O_sk (covariance matrix) and g = O_sk^T @ E_s
        
        if minSR:
            # MinSR approach: Solve in sample space (Ns x Ns)
            # Efficient when Ns < Np (common in Neural Network VMC)
            
            # Construct Gram matrix T = O_sk @ O_sk^dagger (Ns x Ns)
            T = O_sk @ O_sk.T  # For real parameters. Use .conj() if complex.
            
            # Regularization (Diagonal Shift)
            diag_shift = 1e-4
            T += diag_shift * torch.eye(Total_Ns, device=device)
            
            # Solve T * x = E_s
            # 'torch.linalg.solve' is generally faster and more stable than 'pinv' for positive definite matrices
            try:
                x = torch.linalg.solve(T, E_s)
            except RuntimeError:
                print("Warning: Singular matrix in MinSR, falling back to lstsq")
                x = torch.linalg.lstsq(T, E_s).solution

            # Project back to parameter space: dp = O_sk^dagger * x
            dp = O_sk.T @ x
            
        else:
            # Standard SR: Solve in parameter space (Np x Np)
            # Only use this if Np is small.
            S = O_sk.T @ O_sk
            S += 1e-4 * torch.eye(n_params, device=device)
            force_vector = O_sk.T @ E_s
            dp = torch.linalg.solve(S, force_vector)

        # Logging
        print(f"Step {step}: E = {E_mean.item()/nsites:.6f}, "
              f"Var = {E_var.item()/nsites**2:.2e}, "
              f"StdErr = {(E_var.item()/(Total_Ns*nsites**2))**0.5:.2e}")
        print(f'SR Update norm: {dp.norm().item():.4e}')

    # --- D. Broadcast Update & Apply ---
    dist.broadcast(dp, src=0)
    
    # Update parameters
    current_params_vec = torch.nn.utils.parameters_to_vector(fpeps_model.parameters())
    new_params_vec = current_params_vec - learning_rate * dp
    torch.nn.utils.vector_to_parameters(new_params_vec, fpeps_model.parameters())

    # --- E. Checkpointing & Stats ---
    if RANK == 0:
        stats['mean'].append(E_mean.item()/nsites)
        stats['error'].append(torch.sqrt(E_var).item()/nsites)
        stats['variance'].append(E_var.item())
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
            
        if (step + 1) % save_state_every == 0:
            ckpt_path = f"{output_dir}/checkpoint_{fpeps_model._get_name()}_{step+1}.pt"
            torch.save(fpeps_model.state_dict(), ckpt_path)
        
        vmc_pbar.update(1)

# Cleanup
if RANK == 0:
    vmc_pbar.close()
dist.destroy_process_group()