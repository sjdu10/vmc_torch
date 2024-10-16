import numpy as np
from quimb.utils import progbar as Progbar
from mpi4py import MPI
import pickle
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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

from experiment.tn_model import PEPS_model, PEPS_NN_Model
from vmc_torch.sampler import MetropolisExchangeSamplerSpinless, MetropolisExchangeSamplerSpinful
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import TrivialPreconditioner, SignedSGD, SGD, SR
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian import square_lattice_spin_Heisenberg, spin_Heisenberg_square_lattice
from vmc_torch.torch_utils import SVD,QR

# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)

from vmc_torch.global_var import DEBUG


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Hamiltonian parameters
L = int(4)
J = 1.0
# H, hi, graph = square_lattice_spinful_Fermi_Hubbard(Lx, Ly, t, U, N_f)
H = spin_Heisenberg_square_lattice(L, J, total_sz=0.0)
graph = H.graph
# TN parameters
D = 2
chi = -1
dtype=torch.float64

# Load PEPS
skeleton = pickle.load(open(f"../data/{L}x{L}/J={J}/D={D}/peps_skeleton.pkl", "rb"))
peps_params = pickle.load(open(f"../data/{L}x{L}/J={J}/D={D}/peps_su_params.pkl", "rb"))
peps = qtn.unpack(peps_params, skeleton)
peps.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype))

# VMC sample size
N_samples = 2**9
N_samples = N_samples - N_samples % SIZE + SIZE
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

# model = PEPS_model(peps, max_bond=None)
model = PEPS_NN_Model(peps, max_bond=chi)
model_names = {
    PEPS_model: 'PEPS',
    PEPS_NN_Model: 'PEPS_NN'
}
model_name = model_names.get(type(model), 'UnknownModel')

init_step = 0
total_steps = 200
if init_step != 0:
    saved_model_params = torch.load(f'../data/{L}x{L}/J={J}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except:
        model.load_params(saved_model_params_vec)

# optimizer = SignedSGD(learning_rate=0.05)
optimizer = SGD(learning_rate=0.05)
sampler = MetropolisExchangeSamplerSpinless(H.hilbert, graph, N_samples=N_samples, burn_in_steps=50, reset_chain=False, random_edge=True, dtype=dtype)
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, diag_eta=0.05, iter_step=1e5, dtype=dtype)
vmc = VMC(H, variational_state, optimizer, preconditioner)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(False)
    os.makedirs(f'../data/{L}x{L}/J={J}/D={D}/{model_name}/chi={chi}/', exist_ok=True)
    vmc.run(init_step, init_step+total_steps, tmpdir=f'../data/{L}x{L}/J={J}/D={D}/{model_name}/chi={chi}/')

