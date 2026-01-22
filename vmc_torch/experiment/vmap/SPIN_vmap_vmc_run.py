import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
from mpi4py import MPI
import numpy as np
import quimb.tensor as qtn
import pickle
from functools import partial
import torch
import json
# ==============================================================================
from vmc_torch.experiment.vmap.vmap_utils import sample_next_reuse, evaluate_energy_reuse, compute_grads
from vmc_torch.experiment.vmap.vmap_models import (
    PEPS_Model_reuse,
    PEPS_Model,
)
from vmc_torch.experiment.vmap.vmap_modules import run_sampling_phase, run_sampling_phase_reuse, distributed_minres_solver
from vmc_torch.hamiltonian_torch import spin_Heisenberg_square_lattice_torch
# ==============================================================================
import warnings
warnings.filterwarnings("ignore")
# ==============================================================================

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/data'
torch.set_default_device("cpu")
torch.random.manual_seed(42 + RANK)
# ==============================================================================
# 1. Initialization & Configuration
# ==============================================================================
Lx, Ly = 4, 6
nsites = Lx * Ly
D, chi = 4, 16

# Load PEPS
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/data'
params_path = f'{pwd}/{Lx}x{Ly}/heis/D={D}/'
skeleton = pickle.load(open(params_path + 'peps_skeleton.pkl', 'rb'))
params = pickle.load(open(params_path + 'peps_su_params.pkl', 'rb'))
peps = qtn.unpack(params, skeleton)
peps.apply_to_arrays(lambda x: x*4)

# ==============================================================================
# Model Configuration (Define this FIRST)
# ==============================================================================

# Model
model_config = {
    'max_bond': chi,
    'dtype_str': 'float64' 
}
dtype_map = {'float64': torch.float64, 'float32': torch.float32}
model_dtype = dtype_map[model_config['dtype_str']]
init_kwargs = model_config.copy()
init_kwargs.pop('dtype_str')
model_reuse = PEPS_Model_reuse(tn=peps, dtype=model_dtype, **init_kwargs)
# model_reuse =  PEPS_Model(tn=peps, dtype=model_dtype, **init_kwargs)

n_params = sum(p.numel() for p in model_reuse.parameters())
if RANK == 0: 
    print(f'Model Params: {n_params}')

# Hamiltonian
H = spin_Heisenberg_square_lattice_torch(Lx, Ly, J=1.0, total_sz=0)

# VMC Hyperparams
Ns = int(2e3) 
B = 200
B_grad = 200
vmc_steps = 100
init_step = 0
burn_in_steps = 5
learning_rate = 0.1
diag_shift = 1e-4
save_state_every = 1


# Grad Function
get_grads = partial(compute_grads, vectorize=True, vmap_grad=True, batch_size=B_grad, verbose=False)

# Init State
init_config = torch.cat([torch.ones(nsites//2, dtype=torch.int32), torch.zeros(nsites//2, dtype=torch.int32)]).to(torch.int32)
# random shuffles to get B different initial states
random_permutations = torch.stack([init_config[torch.randperm(nsites)] for _ in range(B)])
fxs = random_permutations.to(torch.long)
model_reuse.cache_bMPS_skeleton(fxs[0]) if isinstance(model_reuse, PEPS_Model_reuse) else None
stats = {
    "Np": n_params,
    "sample size": Ns,
    "model_config": model_config,
    "mean": [],
    "error": [],
    "variance": [],
}

# ==============================================================================
# 2. Main VMC Loop
# ==============================================================================
for svmc in range(init_step, vmc_steps + init_step):
    t_start = MPI.Wtime()
    
    # --- Step 1: Sampling Phase (Modularized) ---
    # fxs is updated and returned for the next step (Markov Chain)
    (local_energies, local_grads, local_amps), fxs, sample_stats, total_sample_time = (
        run_sampling_phase_reuse(
            svmc,
            Ns,
            B,
            fxs,
            model_reuse,
            H,
            H.graph,
            get_grads,
            COMM,
            RANK,
            SIZE,
            should_burn_in=svmc == init_step,
            burn_in_steps=burn_in_steps,
            sampling_hopping_rate=0.0
        )
    )
    
    # --- Step 2: Aggregation Phase ---
    # Gather statistics for logging
    all_energies_list = COMM.allgather(local_energies)
    # Filter out empty arrays (from Master)
    valid_energies = [e for e in all_energies_list if e.size > 0]
    all_energies_global = np.concatenate(valid_energies)
    
    energy_mean = np.mean(all_energies_global)
    energy_var = np.var(all_energies_global)
    total_samples = all_energies_global.shape[0]
    
    if RANK == 1:
        print(f' Rank {RANK} | N={sample_stats["n_local"]} | T_samp={sample_stats["t_sample"]:.2f} T_E={sample_stats["t_energy"]:.2f} T_G={sample_stats["t_grad"]:.2f}, MKL={os.environ["MKL_NUM_THREADS"]}')

    COMM.Barrier()
    
    # --- Step 3: Optimization Phase (SR) ---
    # Now this is just a single function call!
    # Master (Rank 0) participates in Allreduce but contributes 0 data
    
    dp, t_sr = distributed_minres_solver(
        local_grads, local_amps, local_energies,
        energy_mean, total_samples, n_params, diag_shift, COMM
    )

    # check if dp contains NaN or Inf
    if torch.isnan(torch.tensor(dp)).any() or torch.isinf(torch.tensor(dp)).any():
        print(f'detect NaN or Inf in dp at step {svmc}, aborting...')
        COMM.Abort(1)
    
    if RANK == 0:
        print(f" Step {svmc} Energy: {energy_mean/nsites:.6f} +/- {np.sqrt(energy_var/total_samples)/nsites:.6f}")
        print(f" Total Samples: {total_samples}, per Rank: {total_samples//SIZE}")
        print(f" Batch Size: {B}, Grad Batch Size: {B_grad}, MKL: {os.environ['MKL_NUM_THREADS']}")
        print(f" Total Time: {MPI.Wtime() - t_start:.4f}s")
        print(f" Sample Time: {total_sample_time:.4f}s")
        print(f" SR Time: {t_sr:.4f}s")
        

    # --- Step 4: Parameter Update ---
    # dp is already available on all ranks due to distributed solver
    with torch.no_grad():
        dp_tensor = torch.tensor(dp, device='cpu', dtype=torch.float64)
        curr_params = torch.nn.utils.parameters_to_vector(model_reuse.parameters())
        new_params = curr_params - learning_rate * dp_tensor
        torch.nn.utils.vector_to_parameters(new_params, model_reuse.parameters())

    # --- Step 5: Logging (Master Only) ---
    t_end = MPI.Wtime()
    file_path = f'{params_path}/{model_reuse._get_name()}/chi={chi}/'
    if RANK == 0:
        if not os.path.exists(file_path): 
            os.makedirs(file_path)
        
        # Text Log
        log_file = file_path + f'vmc_mpi_log_{init_step}.txt'
        if svmc == 0 and os.path.exists(log_file): os.remove(log_file)
        with open(log_file, 'a') as f:
            f.write(f'STEP {svmc}:\nEnergy: {energy_mean/nsites}\
                    \nErr: {np.sqrt(energy_var/total_samples)/nsites}\
                    \nN: {total_samples}\
                    \nTotal Time: {t_end - t_start}\
                    \nTotal Sample Time: {total_sample_time}\
                    \nSR Time: {t_sr}\n\n')
        
        # JSON Stats
        stats['mean'].append(energy_mean/nsites)
        stats['error'].append(np.sqrt(energy_var/total_samples)/nsites)
        stats['variance'].append(energy_var/nsites**2)
        stats['sample size'] = total_samples
        with open(file_path + f'vmc_mpi_stats_{init_step}.json', 'w') as f:
            json.dump(stats, f, indent=4)
            
        # Checkpoint
        if (svmc + 1) % save_state_every == 0:
            torch.save(model_reuse.state_dict(), file_path + f'checkpoint_step_{svmc+1}.pt')