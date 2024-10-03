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
import symmray as sr
import autoray as ar
from autoray import do

from experiment.tn_model import fTNModel, fTN_NNiso_Model, fTN_NN_Model, fTN_Transformer_Model
from experiment.tn_model import init_weights_xavier, init_weights_kaiming, init_weights_to_zero
from vmc_torch.sampler import MetropolisExchangeSampler
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import TrivialPreconditioner, SignedSGD, SGD, SR
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian import square_lattice_spinless_Fermi_Hubbard
from vmc_torch.torch_utils import SVD,QR

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
symmetry = 'Z2'
t = 1.0
V = 1.0
N_f = int(Lx*Ly/2)-2
H, hi, graph = square_lattice_spinless_Fermi_Hubbard(Lx, Ly, t, V, N_f)

# TN parameters
D = 4
chi = 32

# Load PEPS
skeleton = pickle.load(open(f"../data/{Lx}x{Ly}/t={t}_V={V}/N={N_f}/{symmetry}/D={D}/peps_skeleton.pkl", "rb"))
peps_params = pickle.load(open(f"../data/{Lx}x{Ly}/t={t}_V={V}/N={N_f}/{symmetry}/D={D}/peps_su_params.pkl", "rb"))
peps = qtn.unpack(peps_params, skeleton)
peps.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

# VMC parameters
N_samples = 256
N_samples = N_samples - N_samples % SIZE + SIZE - 1

model = fTNModel(peps, max_bond=chi)
# model = fTN_NNiso_Model(peps, max_bond=chi, nn_hidden_dim=8, nn_eta=1e-3)
# model = fTN_NN_Model(peps, max_bond=chi, nn_hidden_dim=8, nn_eta=1e-3)
# model = fTN_Transformer_Model(
#     peps, 
#     max_bond=chi, 
#     nn_eta=1e-1, 
#     d_model=8, 
#     nhead=2, 
#     num_encoder_layers=2, 
#     num_decoder_layers=2,
#     dim_feedforward=32,
#     dropout=0.0,
# )

model.apply(init_weights_to_zero)


model_name = 'fTN' if isinstance(model, fTNModel) else 'fTN_NNiso' if isinstance(model, fTN_NNiso_Model) else 'fTN_NN' if isinstance(model, fTN_NN_Model) else 'fTN_Transformer'
init_step = 0
total_steps = 50
if init_step != 0:
    saved_model_params = torch.load(f'../data/{Lx}x{Ly}/t={t}_V={V}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except:
        model.load_params(saved_model_params_vec)

optimizer = SignedSGD(learning_rate=1e-3)
# optimizer = SGD(learning_rate=0.05)
sampler = MetropolisExchangeSampler(hi, graph, N_samples=N_samples, burn_in_steps=1)
# sampler = None
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler)
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, diag_eta=1e-3, iter_step=1e4)
# preconditioner = TrivialPreconditioner()
vmc = VMC(H, variational_state, optimizer, preconditioner)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(False)
    os.makedirs(f'../data/{Lx}x{Ly}/t={t}_V={V}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/', exist_ok=True)
    vmc.run(init_step, init_step+total_steps, tmpdir=f'../data/{Lx}x{Ly}/t={t}_V={V}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/')

