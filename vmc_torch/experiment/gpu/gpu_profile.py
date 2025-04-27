import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
import time
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
Lx = int(6)
Ly = int(6)
symmetry = 'Z2'
t = 1.0
U = 8.0
N_f = int(Lx*Ly-4)
# N_f = int(Lx*Ly)
n_fermions_per_spin = (N_f//2, N_f//2)
H = spinful_Fermi_Hubbard_square_lattice_torch(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph
# TN parameters
D = 8
chi = 32
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

# VMC sample size
N_samples = int(20)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE
        
# nn_hidden_dim = Lx*Ly
model = fTNModel(peps, max_bond=chi, dtype=dtype, functional=False)
model.to(device)
# model = fTNModel_vec(peps, max_bond=chi, dtype=dtype, functional=True, device=device)


np.random.seed(0)
rand_number = np.random.randint(0, 1000)
X = [H.hilbert.random_state(i) for i in range(10)]
X = torch.tensor(X, dtype=dtype, device=device)

# # compile the model
# model_c = torch.compile(model)
# model_c(X[1])
t0 = time.time()


# # for loop
# for x in X:
#     x = x.to(device)
#     model(x)

model(X)


# vmap the model
# vmap(model)(X)

# # choose number of streams to interleave launches (tunable)
# num_streams = 6
# streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]

# # prepare output list
# B = X.shape[0]
# outputs = [None] * B

# # launch each sample on a stream in round-robin
# for i in range(B):
#     stream = streams[i % num_streams]
#     x_i = X[i]  # shape (...)
#     with torch.cuda.stream(stream):
#         # model_c only handles single-sample input
#         outputs[i] = model(x_i)

# # wait for all streams to finish
# torch.cuda.synchronize()

# # stack results back into (B, ...)
# Y = torch.stack(outputs, dim=0)

t1 = time.time()
print(f"Time taken: {t1-t0:.4f} seconds")
