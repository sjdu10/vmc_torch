import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
import sys
import warnings
warnings.filterwarnings("ignore")
from mpi4py import MPI
import numpy as np
import pickle
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'
# torch
import torch
torch.autograd.set_detect_anomaly(False)

# quimb
import quimb.tensor as qtn
import autoray as ar

from vmc_torch.experiment.tn_model import *
from vmc_torch.sampler import MetropolisExchangeSamplerSpinful, MetropolisMPSSamplerSpinful
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SR,Adam, SGD_momentum, DecayScheduler, TrivialPreconditioner
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
Ly = int(4)
symmetry = 'Z2'
t = 1.0
U = 8.0
N_f = int(Lx*Ly-2)
# N_f = int(Lx*Ly)
n_fermions_per_spin = (N_f//2, N_f//2)
H = spinful_Fermi_Hubbard_square_lattice_torch(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph
# TN parameters
D = 4
chi = -1
dtype=torch.float64
torch.random.manual_seed(RANK)
np.random.seed(RANK)

# Load PEPS
skeleton = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_skeleton.pkl", "rb"))
peps_params = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_su_params.pkl", "rb"))
peps = qtn.unpack(peps_params, skeleton)
device = torch.device("cpu")
peps.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype, device=device))
peps.exponent = torch.tensor(peps.exponent, dtype=dtype, device=device)

# # randomize the PEPS tensors
# peps.apply_to_arrays(lambda x: torch.randn_like(torch.tensor(x, dtype=dtype), dtype=dtype))

# VMC sample size
N_samples = int(20)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE
        
# nn_hidden_dim = Lx*Ly
model = fTNModel(peps, max_bond=chi, dtype=dtype, functional=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = fTN_backflow_attn_Tensorwise_Model_v1(
#     peps,
#     max_bond=chi,
#     embedding_dim=16,
#     attention_heads=4,
#     nn_final_dim=4,
#     nn_eta=1.0,
#     dtype=dtype,
# )

# Set up sampler
sampler = MetropolisExchangeSamplerSpinful(H.hilbert, graph, N_samples=N_samples, burn_in_steps=2, reset_chain=False, random_edge=False, equal_partition=True, dtype=dtype, subchain_length=10)
mps_dir = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/tmp'
# sampler = MetropolisMPSSamplerSpinful(H.hilbert, graph, mps_dir=mps_dir, mps_n_sample=1, N_samples=N_samples, burn_in_steps=20, reset_chain=True, equal_partition=True, dtype=dtype)

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import functional_call
from torch.autograd.functional import jacobian
import pyinstrument

param_dict = dict(model.named_parameters())
# 1) Suppose we've already built a param_dict, but we also want to store
#    the shape/size info for each param in a fixed order:
names, params = zip(*param_dict.items())  # separate keys, values
shapes = [p.shape for p in params]
numels = [p.numel() for p in params]

def vector_to_param_dict(vec):
    """
    vec: 1D Tensor containing *all* parameters in the correct order.
    returns a dict { name_i : param_tensor_i }, with shapes matching the original.
    """
    out = {}
    start = 0
    for name, shape, length in zip(names, shapes, numels):
        end = start + length
        out[name] = vec[start:end].reshape(shape)
        start = end
    return out

device = torch.device("cuda")
model.to(device)
model.skeleton.exponent = model.skeleton.exponent.to(device)

# Example usage
new_vec = model.from_params_to_vec()
new_param_dict = vector_to_param_dict(new_vec)  # {"linear1.weight": tensor(...), ...}

# 2) Now define a "functional" forward using functional_call:
def fmodel(vec, x):
    # Overwrite the model's original parameters with the new ones from vec
    pdict = vector_to_param_dict(vec,)
    return functional_call(model, pdict, (x,))

# 3) Finally, we can compute the Jacobian:
np.random.seed(0)
rand_number = np.random.randint(0, 1000)
X = [H.hilbert.random_state(i) for i in range(4)]
X = torch.tensor(X, dtype=torch.int, device=device)

# use vmap to compute the Jacobian
from functorch import jacrev, vmap
# Single-sample jacobian wrt param vector
def single_sample_jac(vec, x):
    return jacrev(lambda v: fmodel(v, x))(vec)

# Suppose X has shape [batch_size, ...], we can vmap over the first axis of X:
batched_jac = vmap(single_sample_jac, in_dims=(None, 0))
# Now compute per-example Jacobian:
J = batched_jac(new_vec, X)

# # Set up variational state
# variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
# with pyinstrument.Profiler() as prof:
#     J = jacobian(lambda v: fmodel(v, X), new_vec, vectorize=True)
# prof.print()
# with pyinstrument.Profiler() as prof:
#     amp_list = []
#     for x in X:
#         amp, _ = variational_state.amplitude_grad(x)
#         amp_list.append(amp)
# prof.print()
# print("amp_list", amp_list)