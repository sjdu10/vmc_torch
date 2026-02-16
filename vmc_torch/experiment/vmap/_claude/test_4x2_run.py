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
from vmc_torch.experiment.vmap.vmap_utils import compute_grads, random_initial_config
from vmc_torch.experiment.vmap.models import (
    Conv2D_Geometric_fPEPS_Model_Cluster,
)
from vmc_torch.experiment.vmap.vmap_modules import run_sampling_phase, distributed_minres_solver
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.experiment.tn_model import init_weights_to_zero
from vmc_torch.experiment.vmap.vmap_torch_utils import robust_svd_err_catcher_wrapper
from vmc_torch.optimizer import DecayScheduler
# ==============================================================================
import warnings
warnings.filterwarnings("ignore")
# ==============================================================================
SVD_JITTER = 1e-12
driver = None
ar.register_function('torch','linalg.svd', lambda x: robust_svd_err_catcher_wrapper(x, jitter=SVD_JITTER, driver=driver))
# ==============================================================================
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/data'
torch.set_default_device("cpu")
torch.random.manual_seed(42 + RANK)
torch.set_num_threads(1)
# ==============================================================================
# 1. Initialization & Configuration
# ==============================================================================
Lx, Ly = 4, 2
N_f = 6
D, chi = 4, -2
t, U = 1.0, 8.0

# Load PEPS
u1z2 = True
appendix = '_U1SU' if u1z2 else ''
params_path = f'{pwd}/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/'
params = pickle.load(open(params_path + f'peps_su_params{appendix}.pkl', 'rb'))
skeleton = pickle.load(open(params_path + f'peps_skeleton{appendix}.pkl', 'rb'))
peps = qtn.unpack(params, skeleton)
for ts in peps.tensors:
    ts.modify(data=ts.data.to_flat()*4)
for site in peps.sites:
    peps[site].data._label = site
    peps[site].data.indices[-1]._linearmap = ((0, 0), (1, 0), (1, 1), (0, 1))
# ==============================================================================
# Model Configuration
# ==============================================================================
model_config = {
    'max_bond': chi,
    'embed_dim': 16,
    'attn_depth': 1,
    'attn_heads': 4,
    'nn_hidden_dim': peps.nsites,
    'init_perturbation_scale': 1e-3,
    'nn_eta': 1,
    'dtype_str': 'float64',
    'uniform_kernel': 0,
}
dtype_map = {'float64': torch.float64, 'float32': torch.float32}
model_dtype = dtype_map[model_config['dtype_str']]
init_kwargs = model_config.copy()
init_kwargs.pop('dtype_str')

fpeps_model = Conv2D_Geometric_fPEPS_Model_Cluster(
    tn=peps,
    dtype=model_dtype,
    layers=1,
    kernel_size=5,
    contract_boundary_opts={
        'mode': 'mps',
        'equalize_norms': 1.0,
        'canonize': True,
    },
    **init_kwargs
)

n_params = sum(p.numel() for p in fpeps_model.parameters())
if RANK == 0:
    print(f'Model Params: {n_params}')
    print(f'Lattice: {Lx}x{Ly}, N_f={N_f}, D={D}, chi={chi}')
    print(f'MPI Size: {SIZE} (1 master + {SIZE-1} workers)')

# Hamiltonian
H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=(N_f//2, N_f//2), no_u1_symmetry=False,
)

# VMC Hyperparams
Ns = 4000
B = 500
B_grad = max(1, B//2)
vmc_steps = 50
init_step = 0
burn_in_steps = 10
learning_rate = 0.1
diag_shift = 1e-5
save_state_every = 50
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-2)

if RANK == 0:
    print(f'Ns={Ns}, B={B}, B_grad={B_grad}')
    print(f'Workers={SIZE-1}, samples per worker per sweep={B}')
    print(f'Expected sweeps per worker: ~{Ns // ((SIZE-1)*B)}')

# Broadcast RANK 0 parameters to all ranks
curr_params = torch.nn.utils.parameters_to_vector(fpeps_model.parameters())
COMM.Bcast(curr_params.detach().numpy(), root=0)
curr_params = curr_params.to(model_dtype)
torch.nn.utils.vector_to_parameters(curr_params, fpeps_model.parameters())
COMM.Barrier()

# Grad Function
get_grads = partial(compute_grads, vectorize=True, vmap_grad=True, batch_size=B_grad, verbose=False)

# Init State
fxs = torch.stack([random_initial_config(N_f, peps.nsites) for _ in range(B)]).to(torch.long)
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
step_times = []
for svmc in range(init_step, vmc_steps + init_step):
    t_start = MPI.Wtime()

    # --- Step 1: Sampling Phase ---
    (local_energies, local_grads, local_amps), fxs, sample_stats, total_sample_time = (
        run_sampling_phase(
            svmc=svmc,
            Ns=Ns,
            B=B,
            fxs=fxs,
            model=fpeps_model,
            hamiltonian=H,
            graph=H.graph,
            get_grads_func=get_grads,
            comm=COMM,
            rank=RANK,
            size=SIZE,
            should_burn_in=svmc==init_step,
            burn_in_steps=burn_in_steps,
            verbose=False
        )
    )

    # --- Step 2: Aggregation Phase ---
    all_energies_list = COMM.allgather(local_energies)
    valid_energies = [e for e in all_energies_list if e.size > 0]
    all_energies_global = np.concatenate(valid_energies)

    energy_mean = np.mean(all_energies_global)
    energy_var = np.var(all_energies_global)
    total_samples = all_energies_global.shape[0]

    COMM.Barrier()

    # --- Step 3: Optimization Phase (SR) ---
    dp, t_sr, info = distributed_minres_solver(
        local_grads=local_grads,
        local_amps=local_amps,
        local_energies=local_energies,
        energy_mean=energy_mean,
        total_samples=total_samples,
        n_params=n_params,
        diag_shift=diag_shift,
        comm=COMM,
        rtol=5e-5
    )

    # --- Step 4: Parameter Update ---
    with torch.no_grad():
        dp_tensor = torch.tensor(dp, device='cpu', dtype=model_dtype)
        curr_params = torch.nn.utils.parameters_to_vector(fpeps_model.parameters())
        lr = scheduler(svmc)
        new_params = curr_params - lr * dp_tensor
        COMM.Bcast(new_params.detach().numpy(), root=0)
        new_params = new_params.to(model_dtype)
        torch.nn.utils.vector_to_parameters(new_params, fpeps_model.parameters())

    # --- Step 5: Logging ---
    t_end = MPI.Wtime()
    step_time = t_end - t_start
    step_times.append(step_time)

    if RANK == 0:
        print(f"Step {svmc:3d} | E/site: {energy_mean/peps.nsites:.6f} +/- {np.sqrt(energy_var/total_samples)/peps.nsites:.6f} | N={total_samples} | T_total={step_time:.2f}s T_samp={total_sample_time:.2f}s T_SR={t_sr:.2f}s")

# ==============================================================================
# 3. Summary
# ==============================================================================
if RANK == 0:
    print(f"\n{'='*60}")
    print(f"SUMMARY ({Lx}x{Ly}, N_f={N_f}, Conv2D_Geometric)")
    print(f"{'='*60}")
    print(f"Total steps: {vmc_steps}")
    print(f"Avg time/step: {np.mean(step_times):.2f}s")
    print(f"Avg time/step (excl. first): {np.mean(step_times[1:]):.2f}s")
    print(f"Min time/step: {np.min(step_times):.2f}s")
    print(f"Max time/step: {np.max(step_times):.2f}s")
    print(f"Final E/site: {stats['mean'][-1] if stats['mean'] else energy_mean/peps.nsites:.6f}")

    stats['mean'].append(energy_mean/peps.nsites)
    stats['error'].append(np.sqrt(energy_var/total_samples)/peps.nsites)
    stats['variance'].append(energy_var/peps.nsites**2)
