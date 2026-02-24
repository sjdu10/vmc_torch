import os
import json
from mpi4py import MPI
import pickle
from tqdm import trange

# torch
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

# quimb
import quimb.tensor as qtn
import autoray as ar

from vmc_torch.experiment.tn_model import fTNModel, fTN_backflow_attn_Tensorwise_Model_v1
from vmc_torch.experiment.tn_model import init_weights_to_zero
from vmc_torch.sampler import MetropolisExchangeSamplerSpinful
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, TrivialPreconditioner, SGD_momentum, DecayScheduler, Adam
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.torch_utils import SVD,QR
from vmc_torch.fermion_utils import from_netket_config_to_quimb_config

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
D2 = 8
chi1 = -1
chi2 = -1
dtype=torch.float32

pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'
scale = 2.0
# Load peps1
skeleton = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D1}/peps_skeleton.pkl", "rb"))
peps_params = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D1}/peps_su_params.pkl", "rb"))
peps1 = qtn.unpack(peps_params, skeleton)
peps1.apply_to_arrays(lambda x: torch.tensor(scale*x, dtype=dtype))
# Load peps2
skeleton = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D2}/peps_skeleton_U1.pkl", "rb"))
peps_params = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D2}/peps_su_params_U1.pkl", "rb"))
peps2 = qtn.unpack(peps_params, skeleton)
peps2.apply_to_arrays(lambda x: torch.tensor(scale*x, dtype=dtype))

model_names = {
    fTNModel: 'fTN',
    fTN_backflow_attn_Tensorwise_Model_v1: 'fTN_backflow_attn_Tensorwise_v1',
}

# Learning model
modelA = fTN_backflow_attn_Tensorwise_Model_v1(
    peps1,
    max_bond=chi1,
    embedding_dim=16,
    attention_heads=4,
    nn_final_dim=4,
    nn_eta=1.0,
    dtype=dtype,
)
modelA = fTNModel(peps1, max_bond=chi1, dtype=dtype)
init_std = 5e-2
modelA.apply(lambda x: init_weights_to_zero(x, std=init_std))

model_name = model_names.get(type(modelA), 'UnknownModel')

# Set learning steps
init_step = 0
final_step = 20
total_steps = final_step - init_step

# Target model
modelB = fTNModel(peps2, max_bond=chi2, dtype=dtype)
target_model_name = model_names.get(type(modelB), 'UnknownModel')
target_step = 199

# SWO sample size
N_samples = int(1)
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
        modelA.load_state_dict(saved_model_state_dict)
    except:
        modelA.load_params(saved_model_params_vec)
    optimizer_state = saved_model_params.get('optimizer_state', None)

# Load target model parameters
if target_step != 0:
    saved_model_params = torch.load(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D2}/{target_model_name}/chi={chi2}/model_params_step{target_step}.pth', weights_only=False)
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        modelB.load_state_dict(saved_model_state_dict)
    except:
        modelB.load_params(saved_model_params_vec)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

modelA = modelA.to(device)
modelB = modelB.to(device)

# assume modelA, sampler and variational_state are already defined
# e.g. modelA = MyModel(); sampler = MySampler(); variational_state = ...
num_epochs = 1000

# 1. loss and optimizer

def fidelity(amp_pred, amp_target):
    norm_pred = torch.norm(amp_pred)
    norm_target = torch.norm(amp_target)
    return torch.norm(torch.matmul(amp_pred.conj(), amp_target))/(norm_pred*norm_target)

def fidelity_loss(amp_pred, amp_target):
    return 1-fidelity(amp_pred, amp_target)**2

# Define the loss function
criterion = fidelity_loss

# Define the optimizer over modelA parameters
optimizer = optim.Adam(modelA.parameters(), lr=1e-3)

log_dir = f"runs/{target_model_name}_Dt={D2}_chit={chi2}/{model_name}_D={D1}_chi={chi1}_exact/"
if criterion == fidelity_loss:
    log_dir += "fidelity_loss/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# 2. training loop
pbar = trange(1, num_epochs + 1, desc="Training")
X = torch.tensor(from_netket_config_to_quimb_config(H.hilbert.all_states()), dtype=dtype, device=device)

with torch.no_grad():
    Y_target = modelB(X)

for epoch in range(num_epochs):
    # forward pass
    Y_pred = modelA(X)           # shape: [10]
    loss = criterion(Y_pred, Y_target)
    
    # backward + step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # update bar with latest loss
    pbar.set_postfix({"loss": f"{loss.item():.3e}"})

    fidelity_value = 1 - loss
    writer.add_scalar("train/fidelity", fidelity_value.item(), epoch)
    print(f"Epoch {epoch}: Fidelity = {fidelity_value.item():.3e}")

    # log scalar
    writer.add_scalar("train/loss", loss.item(), epoch)
    pbar.update(1)

writer.close()