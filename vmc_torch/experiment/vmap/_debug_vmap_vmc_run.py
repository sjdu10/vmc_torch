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
from vmc_torch.experiment.vmap.vmap_utils import compute_grads, random_initial_config
from vmc_torch.experiment.vmap.vmap_models import (
    Transformer_fPEPS_Model_Conv2d,
    Transformer_fPEPS_Model_GlobalMLP,
    Transformer_fPEPS_Model_UNet,
    Transformer_fPEPS_Model_DConv2d,
    fTN_backflow_attn_Tensorwise_Model_vmap
)
from vmc_torch.experiment.vmap.vmap_utils import sample_next, evaluate_energy
from vmc_torch.experiment.vmap.vmap_modules import run_sampling_phase, distributed_minres_solver
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.experiment.tn_model import init_weights_to_zero
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
Lx, Ly = 8, 8
N_f = Lx * Ly - 8
D, chi = 8, 8
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

# ==============================================================================
# Model Configuration (Define this FIRST)
# ==============================================================================
# 将所有用于初始化的超参数放在这里
# 注意：ftn (peps) 通常太大或是对象，不适合存json，只要记录生成peps的参数(Lx, Ly等)即可
model_config = {
    'max_bond': chi,
    'embed_dim': 16,
    'attn_depth': 1,
    'attn_heads': 4,
    'nn_hidden_dim': peps.nsites,
    'init_perturbation_scale': 1e-3,
    'nn_eta': 1,
    'dtype_str': 'float64' 
}
dtype_map = {'float64': torch.float64, 'float32': torch.float32}
model_dtype = dtype_map[model_config['dtype_str']]
init_kwargs = model_config.copy()
init_kwargs.pop('dtype_str')
# Model
fpeps_model = Transformer_fPEPS_Model_Conv2d(
    tn=peps,
    dtype=model_dtype,
    **init_kwargs
)
# fpeps_model = Transformer_fPEPS_Model_GlobalMLP(
#     tn=peps,
#     max_bond=chi,
#     embed_dim=16,
#     attn_heads=4,
#     attn_depth=1,
#     nn_hidden_dim=peps.nsites,
#     nn_eta=1,
#     init_perturbation_scale=1e-3,
#     dtype=torch.float64,
# )
# fpeps_model = Transformer_fPEPS_Model_DConv2d(
#     tn=peps,
#     max_bond=chi,
#     embed_dim=16,
#     attn_heads=4,
#     attn_depth=1,
#     nn_hidden_dim=peps.nsites,
#     nn_eta=1,
#     init_perturbation_scale=1e-3,
#     dtype=torch.float64,
# )
# fpeps_model = Transformer_fPEPS_Model_UNet(
#     tn=peps,
#     dtype=model_dtype,
#     **init_kwargs
# )

# fpeps_model = fTN_backflow_attn_Tensorwise_Model_vmap(
#     ftn=peps,
#     dtype=model_dtype,
#     **init_kwargs
# )
# fpeps_model.apply(partial(init_weights_to_zero, std=1e-3))

n_params = sum(p.numel() for p in fpeps_model.parameters())
if RANK == 0: 
    print(f'Model Params: {n_params}')

# Hamiltonian
H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=(N_f//2, N_f//2), no_u1_symmetry=False,
)

# VMC Hyperparams
Ns = int(30) 
B = 30
B_grad = 4
vmc_steps = 500
init_step = 0
burn_in_steps = 0
learning_rate = 0.1
diag_shift = 1e-4
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
for _ in range(10):
    t0 = MPI.Wtime()
    fxs, current_amps = sample_next(fxs, fpeps_model, H.graph, verbose=False)
    energy_batch, local_energies_batch = evaluate_energy(fxs, fpeps_model, H, current_amps, verbose=False)
    t1 = MPI.Wtime()
    print(f"Rank {RANK} Sampling + Energy Evaluation Time: {t1 - t0} seconds")
    print(energy_batch)
# ==============================================================================