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
# 导入你的辅助函数
from vmap_utils import compute_grads, random_initial_config
from vmap_models import Transformer_fPEPS_Model_batchedAttn
from vmap_modules import run_sampling_phase, distributed_minres_solver
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
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
Lx, Ly = 4, 2
N_f = Lx * Ly - 2
D, chi = 4, -1
t, U = 1.0, 8.0

# Load PEPS
u1z2 = True
appendix = '_U1SU' if u1z2 else ''
params_path = f'{pwd}/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/'
params = pickle.load(open(params_path + f'peps_su_params{appendix}.pkl', 'rb'))
skeleton = pickle.load(open(params_path + f'peps_skeleton{appendix}.pkl', 'rb'))
peps = qtn.unpack(params, skeleton)
for ts in peps.tensors: 
    ts.modify(data=ts.data.to_flat()*10)
for site in peps.sites: 
    peps[site].data._label = site
    peps[site].data.indices[-1]._linearmap = ((0, 0), (1, 0), (1, 1), (0, 1)) # Important for U1->Z2 fPEPS

# Model
fpeps_model = Transformer_fPEPS_Model_batchedAttn(
    tn=peps,
    max_bond=chi,
    embed_dim=16,
    attn_heads=4,
    attn_depth=1,
    nn_hidden_dim=4 * peps.nsites,
    nn_eta=1,
    init_perturbation_scale=1e-3,
    dtype=torch.float64,
)
n_params = sum(p.numel() for p in fpeps_model.parameters())
if RANK == 0: print(f'Model Params: {n_params}')

# Hamiltonian
H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=(N_f//2, N_f//2), no_u1_symmetry=False,
)

# VMC Hyperparams
Ns = int(1e4) 
B = 1024
B_grad = 128
vmc_steps = 500
init_step = 0
learning_rate = 0.1
diag_shift = 1e-5
save_state_every = 10

# Load Checkpoint
file_path = f'{params_path}/{fpeps_model._get_name()}/chi={chi}/'
if init_step > 0:
    ckpt_path = file_path + f'checkpoint_step_{init_step}.pt'
    fpeps_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    if RANK == 0: 
        print(f'Loaded step {init_step}')

# Grad Function
get_grads = partial(compute_grads, vectorize=True, vmap_grad=True, batch_size=B_grad, verbose=False)

# Init State
fxs = torch.stack([random_initial_config(N_f, peps.nsites) for _ in range(B)]).to(torch.long)
stats = {'Np': n_params, 'sample size': Ns, 'mean': [], 'error': [], 'variance': []}

# ==============================================================================
# 2. Main VMC Loop
# ==============================================================================
for svmc in range(init_step, vmc_steps + init_step):
    t_start = MPI.Wtime()
    
    # --- Step 1: Sampling Phase (Modularized) ---
    # fxs is updated and returned for the next step (Markov Chain)
    (local_energies, local_grads, local_amps), fxs, timing = run_sampling_phase(
        svmc, Ns, B, fxs, fpeps_model, H, H.graph, get_grads, COMM, RANK, SIZE
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
    
    if RANK != 0:
        print(f' Rank {RANK} | N={timing["n_local"]} | T_samp={timing["t_sample"]:.2f} T_E={timing["t_energy"]:.2f} T_G={timing["t_grad"]:.2f}')

    # --- Step 3: Optimization Phase (SR) ---
    # Now this is just a single function call!
    # Master (Rank 0) participates in Allreduce but contributes 0 data
    
    dp, t_sr = distributed_minres_solver(
        local_grads, local_amps, local_energies,
        energy_mean, total_samples, n_params, diag_shift, COMM
    )
    
    if RANK == 0:
        print(f" SR Time: {t_sr:.4f}s")
        print(f" Step {svmc} Energy: {energy_mean/peps.nsites:.6f} +/- {np.sqrt(energy_var/total_samples)/peps.nsites:.6f}")

    # --- Step 4: Parameter Update ---
    # dp is already available on all ranks due to distributed solver
    with torch.no_grad():
        dp_tensor = torch.tensor(dp, device='cpu', dtype=torch.float64)
        curr_params = torch.nn.utils.parameters_to_vector(fpeps_model.parameters())
        new_params = curr_params - learning_rate * dp_tensor
        torch.nn.utils.vector_to_parameters(new_params, fpeps_model.parameters())

    # --- Step 5: Logging (Master Only) ---
    t_end = MPI.Wtime()
    if RANK == 0:
        if not os.path.exists(file_path): 
            os.makedirs(file_path)
        
        # Text Log
        log_file = file_path + f'vmc_mpi_log_{init_step}.txt'
        if svmc == 0 and os.path.exists(log_file): os.remove(log_file)
        with open(log_file, 'a') as f:
            f.write(f'STEP {svmc}:\nEnergy: {energy_mean/peps.nsites}\nErr: {np.sqrt(energy_var/total_samples)/peps.nsites}\nN: {total_samples}\nTime: {t_end - t_start}\n\n')
        
        # JSON Stats
        stats['mean'].append(energy_mean/peps.nsites)
        stats['error'].append(np.sqrt(energy_var/total_samples)/peps.nsites)
        stats['variance'].append(energy_var/peps.nsites**2)
        stats['sample size'] = total_samples
        with open(file_path + f'vmc_mpi_stats_{init_step}.json', 'w') as f:
            json.dump(stats, f)
            
        # Checkpoint
        if (svmc + 1) % save_state_every == 0:
            torch.save(fpeps_model.state_dict(), file_path + f'checkpoint_step_{svmc+1}.pt')