import numpy as np
from quimb.utils import progbar as Progbar
from mpi4py import MPI
import pickle

# torch
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn

# quimb
import quimb as qu
import quimb.tensor as qtn
import symmray as sr
import autoray as ar
from autoray import do

from experiment.tn_model import fTNModel, fTN_NNiso_Model
from sampler import MetropolisExchangeSampler
from variational_state import Variational_State
from optimizer import TrivialPreconditioner, SignedSGD, SGD, SR
from VMC import VMC
from hamiltonian import square_lattice_spinless_Fermi_Hubbard

import global_var
global_var.set_debug(True)
from global_var import DEBUG


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()



Lx = int(4)
Ly = int(4)
symmetry = 'Z2'
t = 1.0
V = 4.0
N_f = int(Lx*Ly/2)-2

H, hi, graph = square_lattice_spinless_Fermi_Hubbard(Lx, Ly, t, V, N_f)

skeleton = pickle.load(open(f"./data/{Lx}x{Ly}/{symmetry}/peps_skeleton.pkl", "rb"))
peps_params = pickle.load(open(f"./data/{Lx}x{Ly}/{symmetry}/peps_su_params.pkl", "rb"))
peps = qtn.unpack(peps_params, skeleton)
peps.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

N_samples = 512
N_samples = N_samples - N_samples % SIZE
model = fTNModel(peps)
# model = fTN_NNiso_Model(peps, max_bond=8, nn_hidden_dim=2, nn_eta=1e-3)
optimizer = SignedSGD(learning_rate=1e-3)
sampler = MetropolisExchangeSampler(hi, graph, N_samples=N_samples, burn_in_steps=1)
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler)
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True)
# preconditioner = TrivialPreconditioner()
vmc = VMC(H, variational_state, optimizer, preconditioner)
vmc.run(0, 20, tmpdir=f'./data/{Lx}x{Ly}/{symmetry}/test{N_samples}.txt')

