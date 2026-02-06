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
from vmc_torch.experiment.vmap.vmap_models import (
    fPEPS_Model_reuse,
    Transformer_fPEPS_Model_Cluster_reuse
)
from vmc_torch.experiment.vmap.vmap_modules import sample_next_reuse, evaluate_energy_reuse
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.experiment.tn_model import init_weights_to_zero
from vmc_torch.experiment.vmap.vmap_torch_utils import robust_svd_err_catcher_wrapper
from vmc_torch.optimizer import DecayScheduler
# ==============================================================================
import warnings
warnings.filterwarnings("ignore")
# ==============================================================================
SVD_JITTER = 1e-10
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
D, chi = 8, 24
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
    'jitter_svd': 0,
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
# fpeps_model = fPEPS_Model_reuse(
#     tn=peps,
#     max_bond=chi,
#     dtype=model_dtype,
#     contract_boundary_opts={
#         'mode': 'mps',
#         'equalize_norms': 1.0,
#         'canonize': True,
#     },
# )
# fpeps_model.radius = 0  # Set the radius for local updates
n_params = sum(p.numel() for p in fpeps_model.parameters())
if RANK == 0: 
    print(f'Model Params: {n_params}')

# Hamiltonian
H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=(N_f//2, N_f//2), no_u1_symmetry=False,
)

# VMC Hyperparams
Ns = int(10) 
B = 10
B_grad = 2
vmc_steps = 50
init_step = 0
burn_in_steps = 0
learning_rate = 0.1
diag_shift = 1e-4
save_state_every = 10
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-2)

# Load Checkpoint
file_path = f'{params_path}/{fpeps_model._get_name()}/chi={chi}/'
fpeps_model.debug_file = file_path
if init_step > 0:
    ckpt_path = file_path + f'checkpoint_step_{init_step}.pt'
    fpeps_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    if RANK == 0: 
        print(f'Loaded step {init_step}')
# Broadcast RANK 0 parameters to all ranks (Important to ensure the training makes sense -- train the same model!!)
curr_params = torch.nn.utils.parameters_to_vector(fpeps_model.parameters())
COMM.Bcast(curr_params.detach().numpy(), root=0)
curr_params = curr_params.to(model_dtype)
# require gradient
torch.utils._pytree.tree_map(lambda x: x.requires_grad_(True), curr_params)
torch.nn.utils.vector_to_parameters(curr_params, fpeps_model.parameters())
COMM.Barrier()

# Grad Function
get_grads = partial(compute_grads, vectorize=True, vmap_grad=True, batch_size=B_grad, verbose=False)

# Init State
fxs = torch.stack([random_initial_config(N_f, peps.nsites) for _ in range(B)]).to(torch.long)
fpeps_model.cache_bMPS_skeleton(fxs[0])
stats = {
    "Np": n_params,
    "sample size": Ns,
    "model_config": model_config,
    "mean": [],
    "error": [],
    "variance": [],
}
model = fpeps_model
hamiltonian = H
graph = H.graph
sampling_hopping_rate = 0.25
get_grads_func = get_grads
# ==============================================================================
# 2. Main VMC Loop
# ==============================================================================
for svmc in range(init_step, vmc_steps + init_step):
    t00 = MPI.Wtime()
    print(f"Rank {RANK} starts sampling batch with {fxs.shape[0]} configs.", flush=True)
    fxs, current_amps = sample_next_reuse(fxs, model, graph, hopping_rate=sampling_hopping_rate, verbose=False)
    t11 = MPI.Wtime()
    # with torch.no_grad():
    #     current_amps = model(fxs)
    print(f"Rank {RANK} finished sampling batch. Starts energy evaluation.", flush=True)
    energy_batch, local_energies_batch = evaluate_energy_reuse(fxs, model, hamiltonian, current_amps, verbose=False)
    t22 = MPI.Wtime()
    print(f"Rank {RANK} finished energy evaluation. Starts grad evaluation.", flush=True)
    grads_vec_batch, amps_batch = get_grads_func(fxs, model)
    t33 = MPI.Wtime()