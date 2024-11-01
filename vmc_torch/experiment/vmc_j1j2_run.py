import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from quimb.utils import progbar as Progbar
from mpi4py import MPI
import pickle


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

from vmc_torch.experiment.tn_model import PEPS_model, PEPS_NN_Model, init_weights_to_zero, PEPS_NNproj_Model, PEPS_delocalized_Model
from vmc_torch.sampler import MetropolisExchangeSamplerSpinless, MetropolisExchangeSamplerSpinful
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import TrivialPreconditioner, SignedSGD, SGD, SR
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian import spin_J1J2_square_lattice
from vmc_torch.torch_utils import SVD,QR
from vmc_torch.utils import closest_divisible

# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)

from vmc_torch.global_var import DEBUG


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Hamiltonian parameters
Lx = int(4)
Ly = int(6)
J1 = 1.0
J2 = 0.5
H = spin_J1J2_square_lattice(Lx, Ly, J1, J2, total_sz=0.0) 
graph = H.graph
# TN parameters
D = 3
chi = -1
chi_nn = 2
dtype=torch.float64

# Load PEPS
skeleton = pickle.load(open(f"../../data/{Lx}x{Ly}/J1={J1}_J2={J2}/D={D}/peps_skeleton.pkl", "rb"))
peps_params = pickle.load(open(f"../../data/{Lx}x{Ly}/J1={J1}_J2={J2}/D={D}/peps_su_params.pkl", "rb"))
peps = qtn.unpack(peps_params, skeleton)
peps.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype))

# VMC sample size
N_samples = int(1e3)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

# model = PEPS_model(peps, max_bond=chi)
model = PEPS_delocalized_Model(peps, max_bond=chi, diag=False)
# model = PEPS_NN_Model(peps, max_bond=chi_nn, nn_eta=1.0, nn_hidden_dim=L**2)
# model = PEPS_NNproj_Model(peps, max_bond=chi_nn, nn_eta=1.0, nn_hidden_dim=L**2)
# model.apply(init_weights_to_zero)
model_names = {
    PEPS_model: 'PEPS',
    PEPS_delocalized_Model: 'PEPS_delocalized_diag='+str(model.diag) if type (model)==PEPS_delocalized_Model else None,
    PEPS_NN_Model: 'PEPS_NN',
    PEPS_NNproj_Model: 'PEPS_NNproj'
}
model_name = model_names.get(type(model), 'UnknownModel')

init_step = 0
final_step = 200
total_steps = final_step - init_step
if init_step != 0:
    saved_model_params = torch.load(f'../../data/{Lx}x{Ly}/J1={J1}_J2={J2}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except:
        model.load_params(saved_model_params_vec)

# optimizer = SignedSGD(learning_rate=0.05)
optimizer = SGD(learning_rate=5e-2)
sampler = MetropolisExchangeSamplerSpinless(H.hilbert, graph, N_samples=N_samples, burn_in_steps=20, reset_chain=False, random_edge=True, equal_partition=False, dtype=dtype)
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, diag_eta=5e-2, iter_step=1e5, dtype=dtype)
vmc = VMC(H, variational_state, optimizer, preconditioner)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(False)
    os.makedirs(f'../../data/{Lx}x{Ly}/J1={J1}_J2={J2}/D={D}/{model_name}/chi={chi}/', exist_ok=True)
    vmc.run(init_step, init_step+total_steps, tmpdir=f'../../data/{Lx}x{Ly}/J1={J1}_J2={J2}/D={D}/{model_name}/chi={chi}/')

