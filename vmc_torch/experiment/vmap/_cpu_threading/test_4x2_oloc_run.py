"""
Test script: verify run_sampling_phase + distributed_minres_solver
produce correct VMC optimization (energy should decrease over 10 steps).

Run with:
    mpirun -np 4 /home/sijingdu/TNVMC/VMC_code/clean_symmray/bin/python test_4x2_oloc_run.py
"""
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
import autoray as ar
# ==============================================================================
from vmc_torch.experiment.vmap.vmap_utils import compute_grads, random_initial_config
from vmc_torch.experiment.vmap.models import (
    Conv2D_Geometric_fPEPS_Model_Cluster,
)
from vmc_torch.experiment.vmap.vmap_modules import (
    run_sampling_phase,
    distributed_minres_solver,
)
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.experiment.vmap.vmap_torch_utils import robust_svd_err_catcher_wrapper
from vmc_torch.optimizer import DecayScheduler
# ==============================================================================
import warnings
warnings.filterwarnings("ignore")
# ==============================================================================
SVD_JITTER = 1e-12
ar.register_function('torch', 'linalg.svd', lambda x: robust_svd_err_catcher_wrapper(x, jitter=SVD_JITTER, driver=None))
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
params_path = f'{pwd}/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/'
params = pickle.load(open(params_path + 'peps_su_params_U1SU.pkl', 'rb'))
skeleton = pickle.load(open(params_path + 'peps_skeleton_U1SU.pkl', 'rb'))
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
model_dtype = torch.float64
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
vmc_steps = 10
init_step = 0
burn_in_steps = 10
learning_rate = 0.1
diag_shift = 1e-5
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-2)

if RANK == 0:
    print(f'Ns={Ns}, B={B}, B_grad={B_grad}')
    print(f'Workers={SIZE-1}, samples per worker per sweep={B}')

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

# ==============================================================================
# 2. Main VMC Loop (using oloc variants)
# ==============================================================================
step_times = []
energy_history = []

for svmc in range(init_step, vmc_steps + init_step):
    t_start = MPI.Wtime()

    # --- Step 1: Sampling Phase (oloc variant) ---
    (local_energies, local_O), fxs, sample_stats, total_sample_time = (
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

    # --- Step 3: Optimization Phase (oloc SR solver) ---
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

    e_per_site = energy_mean / peps.nsites
    energy_history.append(e_per_site)

    if RANK == 0:
        print(f"Step {svmc:3d} | E/site: {e_per_site:.6f} +/- {np.sqrt(energy_var/total_samples)/peps.nsites:.6f} | N={total_samples} | T_total={step_time:.2f}s T_samp={total_sample_time:.2f}s T_SR={t_sr:.2f}s")

# ==============================================================================
# 3. Summary & Verification
# ==============================================================================
if RANK == 0:
    print(f"\n{'='*60}")
    print(f"VERIFICATION ({Lx}x{Ly}, N_f={N_f}, oloc variants)")
    print(f"{'='*60}")
    print(f"Total steps: {vmc_steps}")
    print(f"Avg time/step: {np.mean(step_times):.2f}s")
    print(f"Energy history (E/site):")
    for i, e in enumerate(energy_history):
        print(f"  Step {i}: {e:.6f}")

    e_first = energy_history[0]
    e_last = energy_history[-1]
    e_min = min(energy_history)
    print(f"\nFirst E/site:  {e_first:.6f}")
    print(f"Last E/site:   {e_last:.6f}")
    print(f"Min E/site:    {e_min:.6f}")
    print(f"Delta (last-first): {e_last - e_first:.6f}")

    if e_last < e_first:
        print("\nPASSED: Energy decreased over VMC steps.")
    else:
        print("\nWARNING: Energy did NOT decrease. Check implementation.")
