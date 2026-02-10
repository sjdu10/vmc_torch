# Define the number of cores per MPI rank
num_threads = 1
import os
# Set environment variables BEFORE importing torch or numpy
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)

from mpi4py import MPI
import numpy as np
import quimb.tensor as qtn
import pickle
from functools import partial
import torch
import json
import autoray as ar
# ==============================================================================
from vmc_torch.experiment.vmap.vmap_utils import random_initial_config, sample_next_reuse, sample_next, evaluate_energy, evaluate_energy_reuse, compute_grads
from vmc_torch.experiment.vmap.vmap_models import (
    Transformer_fPEPS_Model_Cluster_reuse,
    Transformer_fPEPS_Model_Cluster,
)
from vmc_torch.experiment.vmap.vmap_modules import distributed_minres_solver, run_sampling_phase_reuse
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.experiment.tn_model import init_weights_to_zero
from vmc_torch.experiment.vmap.vmap_torch_utils import robust_svd_err_catcher_wrapper
from vmc_torch.optimizer import DecayScheduler
import symmray as sr
import warnings

# ==============================================================================
warnings.filterwarnings("ignore")
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)
# ==============================================================================
SVD_JITTER = 1e-16
driver = None
ar.register_function('torch','linalg.svd', lambda x: robust_svd_err_catcher_wrapper(x, jitter=SVD_JITTER, driver=driver))
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
Lx, Ly = 8, 8
N_f = Lx * Ly - 8
D, chi = 10, 10
t, U = 1.0, 8.0
dtype = torch.float64
rpeps = False
if rpeps:
    # Load PEPS
    peps = sr.networks.PEPS_fermionic_rand(
        "Z2",
        Lx,
        Ly,
        D,
        phys_dim=[
            (0, 0),  # linear index 0 -> charge 0, offset 0
            (1, 1),  # linear index 1 -> charge 1, offset 1
            (1, 0),  # linear index 2 -> charge 1, offset 0
            (0, 1),  # linear index 3 -> charge 0, offset 1
        ],
        subsizes="equal",
        flat=True,
        seed=1,
        dtype=str(dtype).split(".")[-1],
    )
else:
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
        peps[site].data.indices[-1]._linearmap = ((0, 0), (1, 0), (1, 1), (0, 1)) # Important for U1->Z2 fPEPS

# ==============================================================================
# Model Configuration (Define this FIRST)
# ==============================================================================
# Put all hyperparameters for initialization here
# Note: ftn (peps) is usually too large or an object, not suitable for json storage, only record the parameters used to generate peps (Lx, Ly, etc.)
model_config = {
    'max_bond': chi,
    'embed_dim': 16,
    'attn_depth': 1,
    'attn_heads': 4,
    'nn_hidden_dim': D, #peps.nsites,
    'init_perturbation_scale': 1e-3,
    'nn_eta': 1,
    'dtype_str': 'float64',
    'uniform_kernel': 0,
}
dtype_map = {'float64': torch.float64, 'float32': torch.float32}
model_dtype = dtype_map[model_config['dtype_str']]
init_kwargs = model_config.copy()
init_kwargs.pop('dtype_str')
# Model
# fpeps_model = Transformer_fPEPS_Model_Conv2d(
#     tn=peps,
#     dtype=model_dtype,
#     **init_kwargs
# )
fpeps_model = Transformer_fPEPS_Model_Cluster_reuse(
    tn=peps,
    dtype=model_dtype,
    contract_boundary_opts={
        'mode': 'mps',
        # 'equalize_norms': 1.0,
        'canonize': True,
    },
    **init_kwargs
)
fpeps_model1 = Transformer_fPEPS_Model_Cluster(
    tn=peps,
    dtype=model_dtype,
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

# Hamiltonian
H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=(N_f//2, N_f//2), no_u1_symmetry=False,
)

# VMC Hyperparams
Ns = int(6e3) 
B = 28#*num_threads
B_grad = 2#*num_threads
vmc_steps = 50
init_step = 0
burn_in_steps = 5
learning_rate = 0.1
diag_shift = 1e-4
save_state_every = 10
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-2)

# # Load Checkpoint
# file_path = f'{params_path}/{fpeps_model._get_name()}/chi={chi}/'
# fpeps_model.debug_file = file_path
# if init_step > 0:
#     ckpt_path = file_path + f'checkpoint_step_{init_step}.pt'
#     fpeps_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
#     fpeps_model1.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
#     if RANK == 0: 
#         print(f'Loaded step {init_step}')
fpeps_model_params = fpeps_model.state_dict()
fpeps_model1.load_state_dict(fpeps_model_params)

fxs0 = torch.stack([random_initial_config(N_f, peps.nsites) for _ in range(B)]).to(torch.long)
# fpeps_model.cache_bMPS_skeleton(fxs0[0])
print(f'Batch size: {B}, Grad Batch Size: {B_grad}')

import time
fxs = fxs0.clone()
# with torch.no_grad():
#     # t0 = time.time()
#     # # reuse_amps = fpeps_model(fxs)
#     # reuse_fxs, reuse_amps = sample_next_reuse(fxs0.clone(), fpeps_model, H.graph, show_pbar=True)
    
#     t1 = time.time()
#     # amps1 = fpeps_model1(fxs)
#     fxs1, amps1 = sample_next(fxs0.clone(), fpeps_model1, H.graph, show_pbar=True)
#     t2 = time.time()
#     E, loc_Es = evaluate_energy(fxs1, fpeps_model1, H, amps1, show_pbar=True)
#     t3 = time.time()
#     # fxs1 = fxs0.clone()
#     amps1 = fpeps_model1(fxs1)
    
fxs1 = fxs0.clone()
a, b = compute_grads(fxs1, fpeps_model1, vectorize=True, vmap_grad=True, batch_size=B_grad, show_pbar=True)
time.sleep(2)
t4 = time.time()

# amps1, t2-t1, t3-t2, t4-t3
