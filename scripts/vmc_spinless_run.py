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

from vmc_torch.experiment.tn_model import fTNModel, fTN_Transformer_Model, SlaterDeterminant, NeuralBackflow, FFNN, NeuralJastrow
from vmc_torch.experiment.tn_model import init_weights_xavier, init_weights_kaiming, init_weights_to_zero
from vmc_torch.sampler import MetropolisExchangeSamplerSpinless
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import TrivialPreconditioner, SignedSGD, SGD, SR, DecayScheduler
from vmc_torch.VMC import VMC
from vmc_torch.experiment.tests.dev.hamiltonian_old import square_lattice_spinless_Fermi_Hubbard, spinful_Fermi_Hubbard_square_lattice, spinless_Fermi_Hubbard_square_lattice
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
Ly = int(4)
symmetry = 'U1'
t = 1.0
V = 1.0
N_f = int(Lx*Ly/2)-2
# H, hi, graph = square_lattice_spinless_Fermi_Hubbard(Lx, Ly, t, V, N_f)
H = spinless_Fermi_Hubbard_square_lattice(Lx, Ly, t, V, N_f)

# TN parameters
D = 4
chi = 5
dtype=torch.float64

# Load PEPS
skeleton = pickle.load(open(f"../../data/{Lx}x{Ly}/t={t}_V={V}/N={N_f}/{symmetry}/D={D}/peps_skeleton.pkl", "rb"))
peps_params = pickle.load(open(f"../../data/{Lx}x{Ly}/t={t}_V={V}/N={N_f}/{symmetry}/D={D}/peps_su_params.pkl", "rb"))
peps = qtn.unpack(peps_params, skeleton)
peps.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype))

# VMC sample size
N_samples = 2**12
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

# model = fTNModel(peps, max_bond=chi)
# model = SlaterDeterminant(H.hi)
# model = NeuralBackflow(hi, param_dtype=dtype, hidden_dim=hi.size)
# model = NeuralJastrow(hi, param_dtype=dtype, hidden_dim=hi.size)
model = FFNN(H.hi, hidden_dim=2*H.hi.size, param_dtype=dtype)

# model.apply(init_weights_to_zero)
# model.apply(init_weights_xavier)

model_names = {
    fTNModel: 'fTN',
    fTN_Transformer_Model: 'fTN_Transformer',
    SlaterDeterminant: 'SlaterDeterminant',
    NeuralBackflow: 'NeuralBackflow',
    FFNN: 'FFNN',
    NeuralJastrow: 'NeuralJastrow',
}
model_name = model_names.get(type(model), 'UnknownModel')

init_step = 0
total_steps = 200
if init_step != 0:
    saved_model_params = torch.load(f'../../data/{Lx}x{Ly}/t={t}_V={V}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except:
        model.load_params(saved_model_params_vec)

# Set up optimizer and scheduler
learning_rate = 1e-1
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.5, patience=50, min_lr=5e-3)
optimizer = SGD(learning_rate=0.05)
sampler = MetropolisExchangeSamplerSpinless(H.hi, H.graph, N_samples=N_samples, burn_in_steps=16, reset_chain=False, random_edge=True, dtype=dtype)
# sampler = None
variational_state = Variational_State(model, hi=H.hi, sampler=sampler, dtype=dtype)
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, solver='minres', diag_eta=1e-3, iter_step=1e5, dtype=dtype, rtol=1e-4)
# preconditioner = TrivialPreconditioner()
vmc = VMC(H, variational_state=variational_state, optimizer=optimizer, preconditioner=preconditioner, scheduler=scheduler)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(False)
    os.makedirs(f'../../data/{Lx}x{Ly}/t={t}_V={V}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/', exist_ok=True)
    vmc.run(init_step, init_step+total_steps, tmpdir=f'../../data/{Lx}x{Ly}/t={t}_V={V}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/')

