import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from quimb.utils import progbar as Progbar
from mpi4py import MPI
import pickle
import pyinstrument

# torch
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(False)

# quimb
import quimb as qu
import quimb.tensor as qtn
import autoray as ar
from autoray import do

from vmc_torch.experiment.tn_model import fTNModel, fTN_backflow_attn_Tensorwise_Model_v1
from vmc_torch.experiment.tn_model import init_weights_to_zero
from vmc_torch.sampler import MetropolisExchangeSamplerSpinful
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, TrivialPreconditioner, SGD_momentum, DecayScheduler, Adam
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.torch_utils import SVD,QR

# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)

from vmc_torch.global_var import DEBUG
from vmc_torch.utils import closest_divisible


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Hamiltonian parameters
Lx = int(4)
Ly = int(2)
symmetry = 'Z2'
t = 1.0
U = 8.0
N_f = int(Lx*Ly-2)
# N_f = int(Lx*Ly)
n_fermions_per_spin = (N_f//2, N_f//2)
H = spinful_Fermi_Hubbard_square_lattice_torch(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph

# TN parameters
D1 = 4
D2 = 6
chi1 = -1
chi2 = -1
dtype=torch.float64

pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'
# Load peps1
skeleton = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D1}/peps_skeleton.pkl", "rb"))
peps_params = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D1}/peps_su_params.pkl", "rb"))
peps1 = qtn.unpack(peps_params, skeleton)
peps1.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype))
# Load peps2
skeleton = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D2}/peps_skeleton_U1.pkl", "rb"))
peps_params = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D2}/peps_su_params_U1.pkl", "rb"))
peps2 = qtn.unpack(peps_params, skeleton)
peps2.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype))

model_names = {
    fTNModel: 'fTN',
    fTN_backflow_attn_Tensorwise_Model_v1: 'fTN_backflow_attn_Tensorwise_v1',
}

# Learning model
model1 = fTN_backflow_attn_Tensorwise_Model_v1(
    peps1,
    max_bond=chi1,
    embedding_dim=16,
    attention_heads=4,
    nn_final_dim=4,
    nn_eta=1.0,
    dtype=dtype,
)
# model1 = fTNModel(peps1, max_bond=chi1, dtype=dtype)
init_std = 5e-2
model1.apply(lambda x: init_weights_to_zero(x, std=init_std))
# model1 = fpepsModel(peps1, dtype=dtype)
model_name = model_names.get(type(model1), 'UnknownModel')

# Set learning steps
init_step = 0
final_step = 20
total_steps = final_step - init_step

# Target model
model2 = fTNModel(peps2, max_bond=chi2, dtype=dtype)
target_model_name = model_names.get(type(model2), 'UnknownModel')
target_step = 57

# SWO sample size
N_samples = int(1e4)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

# Load learning model parameters
optimizer_state = None
if init_step != 0:
    saved_model_params = torch.load(pwd+f'/SWO_fit/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/target_{target_model_name}_D={D2}_chi={chi2}/{model_name}/D={D1}/chi={chi1}/model_params_step{init_step}.pth', weights_only=False)
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model1.load_state_dict(saved_model_state_dict)
    except:
        model1.load_params(saved_model_params_vec)
    optimizer_state = saved_model_params.get('optimizer_state', None)

# Load target model parameters
if target_step != 0:
    saved_model_params = torch.load(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D2}/{target_model_name}/chi={chi2}/model_params_step{target_step}.pth', weights_only=False)
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model2.load_state_dict(saved_model_state_dict)
    except:
        model2.load_params(saved_model_params_vec)

# Set up optimizer and scheduler
learning_rate = 5e-2
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-3)
use_prev_opt = True
if optimizer_state is not None and use_prev_opt:
    optimizer_name = optimizer_state['optimizer']
    if optimizer_name == 'SGD_momentum':
        optimizer = SGD_momentum(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=learning_rate, weight_decay=1e-5)
    print('Loading optimizer: ', optimizer)
    optimizer.lr = learning_rate
    if isinstance(optimizer, SGD_momentum):
        optimizer.velocity = optimizer_state['velocity']
    if isinstance(optimizer, Adam):
        optimizer.m = optimizer_state['m']
        optimizer.v = optimizer_state['v']
        optimizer.t = optimizer_state['t']
else:
    # optimizer = SignedSGD(learning_rate=learning_rate)
    # optimizer = SignedRandomSGD(learning_rate=learning_rate)
    optimizer = SGD(learning_rate=learning_rate)
    # optimizer = SGD_momentum(learning_rate=learning_rate, momentum=0.9)
    # optimizer = Adam(learning_rate=learning_rate, t_step=init_step, weight_decay=0.0)

# Set up sampler
sampler = MetropolisExchangeSamplerSpinful(H.hilbert, graph, N_samples=N_samples, burn_in_steps=20, reset_chain=True, random_edge=False, equal_partition=False, dtype=dtype)
# Set up variational state
variational_state = Variational_State(model1, hi=H.hilbert, sampler=sampler, dtype=dtype)
target_state = Variational_State(model2, hi=H.hilbert, sampler=sampler, dtype=dtype)
# Set up preconditioner (Trivial for SWO)
preconditioner = TrivialPreconditioner()
# Set up VMC
vmc = VMC(hamiltonian=H, variational_state=variational_state, optimizer=optimizer, preconditioner=preconditioner, scheduler=scheduler, SWO=True, beta=0.1)

if __name__ == "__main__":
    os.makedirs(pwd+f'/SWO_fit/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/target_{target_model_name}_D={D2}_chi={chi2}/{model_name}/D={D1}/chi={chi1}/', exist_ok=True)
    vmc.run_SWO_state_fitting(target_state=target_state, sample_times=total_steps, compute_energy=True, SWO_max_iter=100, log_fidelity_tol=0.0, tmpdir=pwd+f'/SWO_fit/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/target_{target_model_name}_D={D2}_chi={chi2}/{model_name}/D={D1}/chi={chi1}/')

