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
import autoray as ar
# ==============================================================================
from vmc_torch.experiment.vmap.vmap_utils import sample_next_reuse, evaluate_energy_reuse, compute_grads
from vmc_torch.experiment.vmap.SPIN.SPIN_models import (
    PEPS_Model,
    circuit_TNF_2d_Model
)
from vmc_torch.experiment.vmap.vmap_modules import run_sampling_phase, run_sampling_phase_reuse, distributed_minres_solver
from vmc_torch.hamiltonian_torch import spin_Heisenberg_square_lattice_torch
from vmc_torch.experiment.vmap.vmap_torch_utils import robust_svd_err_catcher_wrapper
from vmc_torch.optimizer import DecayScheduler
# ==============================================================================
import warnings
warnings.filterwarnings("ignore")
# ==============================================================================
SVD_JITTER = 1e-12
driver = None
# ar.register_function('torch','linalg.svd', lambda x: robust_svd_wrapper(x, jitter=SVD_JITTER, driver=driver))
# ar.register_function('torch','linalg.svd', lambda x: robust_svd_eig_wrapper(x, jitter=SVD_JITTER, driver=driver))
ar.register_function('torch','linalg.svd', lambda x: robust_svd_err_catcher_wrapper(x, jitter=SVD_JITTER, driver=driver))

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/data'
torch.set_default_device("cpu")
torch.random.manual_seed(42 + RANK)
# ==============================================================================
# 1. Initialization & Configuration
# ==============================================================================
Lx, Ly = 4, 4
nsites = Lx * Ly
D, chi = 4, 1

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
    'max_bond_final': 8,
    'dtype_str': 'float64',
    'from_which': 'zmax',
    'mode': 'SVDU',
    'trotter_tau': 0.1,
    'depth': 2,
}
dtype_map = {'float64': torch.float64, 'float32': torch.float32}
model_dtype = dtype_map[model_config['dtype_str']]
init_kwargs = model_config.copy()
init_kwargs.pop('dtype_str')

# model_reuse = PEPS_Model_reuse(tn=peps, dtype=model_dtype, **init_kwargs)
# model =  PEPS_Model(tn=peps, dtype=model_dtype, **init_kwargs)
model = circuit_TNF_2d_Model(
    tns=peps,
    ham=qtn.ham_2d_heis(Lx, Ly, j=1),
    **init_kwargs
)

n_params = sum(p.numel() for p in model.parameters())
if RANK == 0: 
    print(f'Model Params: {n_params}')
file_path = f'{params_path}/{model._get_name()}/chi={chi}/'
# Broadcast RANK 0 parameters to all ranks (Important to ensure the training makes sense -- train the same model!!)
curr_params = torch.nn.utils.parameters_to_vector(model.parameters())
COMM.Bcast(curr_params.detach().numpy(), root=0)
curr_params = curr_params.to(model_dtype)
torch.nn.utils.vector_to_parameters(curr_params, model.parameters())
COMM.Barrier()

# Hamiltonian
H = spin_Heisenberg_square_lattice_torch(Lx, Ly, J=1.0, total_sz=0)
if RANK == 0:
    if Lx*Ly <=16:
        H_dense = H.to_dense()
        import scipy.sparse.linalg as la
        # get ground state energy
        gs_e = la.eigsh(H_dense, k=1, which='SA', tol=1e-8)[0][0]
        print(f"Exact Diagonalization Ground State Energy: {gs_e/Lx/Ly:.8f}")
    else:
        print("Exact diagonalization skipped due to large system size.")

# VMC Hyperparams
Ns = int(2e3) 
B = 10
B_grad = 10
vmc_steps = 100
init_step = 0
burn_in_steps = 5
learning_rate = 0.1
diag_shift = 1e-4
save_state_every = 5
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-2)
# Grad Function
get_grads = partial(compute_grads, vectorize=True, vmap_grad=True, batch_size=B_grad, verbose=False)

# Init State
init_config = torch.cat([torch.ones(nsites//2, dtype=torch.int32), torch.zeros(nsites//2, dtype=torch.int32)]).to(torch.int32)
# random shuffles to get B different initial states
random_permutations = torch.stack([init_config[torch.randperm(nsites)] for _ in range(B)])
fxs = random_permutations.to(torch.long)

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
    (local_energies, local_O), fxs, sample_stats, total_sample_time = (
        run_sampling_phase(
            svmc,
            Ns,
            B,
            fxs,
            model,
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
    
    dp, t_sr, info = distributed_minres_solver(
        local_O=local_O,
        local_energies=local_energies,
        energy_mean=energy_mean,
        total_samples=total_samples,
        n_params=n_params,
        diag_shift=diag_shift,
        comm=COMM,
        rtol=5e-5
    )
    
    if RANK == 0:
        print(f" Step {svmc} Energy: {energy_mean/peps.nsites:.6f} +/- {np.sqrt(energy_var/total_samples)/peps.nsites:.6f}")
        print(f" Total Samples: {total_samples}, per Rank: {total_samples//SIZE}")
        print(f" Batch Size: {B}, Grad Batch Size: {B_grad}, MKL: {os.environ['MKL_NUM_THREADS']}")
        print(f" Total Time: {MPI.Wtime() - t_start:.4f}s")
        print(f" Sample Time: {total_sample_time:.4f}s")
        print(f" SR Time: {t_sr:.4f}s, Info: {info}\n")
        

    # --- Step 4: Parameter Update ---
    # dp is already available on all ranks due to distributed solver
    with torch.no_grad():
        dp_tensor = torch.tensor(dp, device='cpu', dtype=model_dtype)
        curr_params = torch.nn.utils.parameters_to_vector(model.parameters())
        lr = scheduler(svmc)
        new_params = curr_params - lr * dp_tensor
        COMM.Bcast(new_params.detach().numpy(), root=0)
        new_params = new_params.to(model_dtype)
        # torch.utils._pytree.tree_map(lambda x: x.requires_grad_(True), new_params)
        torch.nn.utils.vector_to_parameters(new_params, model.parameters())

    # --- Step 5: Logging (Master Only) ---
    t_end = MPI.Wtime()
    if RANK == 0:
        if not os.path.exists(file_path): 
            os.makedirs(file_path)
        
        # Text Log
        log_file = file_path + f'vmc_mpi_log_{init_step}.txt'
        if svmc == 0 and os.path.exists(log_file): os.remove(log_file)
        with open(log_file, 'a') as f:
            f.write(f'STEP {svmc}:\nEnergy: {energy_mean/peps.nsites}\
                    \nErr: {np.sqrt(energy_var/total_samples)/peps.nsites}\
                    \nN: {total_samples}\
                    \nTotal Time: {t_end - t_start}\
                    \nTotal Sample Time: {total_sample_time}\
                    \nSR Time: {t_sr}\n\n')
        
        # JSON Stats
        stats['mean'].append(energy_mean/peps.nsites)
        stats['error'].append(np.sqrt(energy_var/total_samples)/peps.nsites)
        stats['variance'].append(energy_var/peps.nsites**2)
        stats['sample size'] = total_samples
        with open(file_path + f'vmc_mpi_stats_{init_step}.json', 'w') as f:
            json.dump(stats, f, indent=4)
            
        # Checkpoint
        if (svmc + 1) % save_state_every == 0:
            torch.save(model.state_dict(), file_path + f'checkpoint_step_{svmc+1}.pt')