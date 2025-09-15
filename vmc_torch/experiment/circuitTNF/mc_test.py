import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

from vmc_torch.sampler import MetropolisExchangeSamplerSpinless
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SR, DecayScheduler
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian_torch import spin_Heisenberg_chain_torch
from vmc_torch.torch_utils import SVD,QR
from vmc_torch.experiment.tn_model import circuit_TNF

import torch
import quimb.tensor as qtn

import pickle

L = 10
H = spin_Heisenberg_chain_torch(L, J=1.0, pbc=False, total_sz=0)
ham_quimb = qtn.ham_1d_heis(L, j=1.0, cyclic=False)
ham_quimb.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float64))
graph = H.graph
hilbert = H.hilbert
dtype = torch.float64


su_params, su_skeleton = pickle.load(open("circuitTNF_heis_L10_D6_su_state.pkl", "rb"))
su_mps = qtn.unpack(su_params, su_skeleton)

model = circuit_TNF(
    su_mps,
    ham_quimb,
    trotter_tau=0.01,
    depth=4,
    max_bond=-1,
    dtype=torch.float64
)

# VMC sample size
N_samples = int(1e4)
N_samples = N_samples - N_samples % SIZE + SIZE
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

# Set up optimizer and scheduler
learning_rate = 1e-1
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-4)
optimizer = SGD(learning_rate=learning_rate)
sampler = MetropolisExchangeSamplerSpinless(H.hilbert, graph, N_samples=N_samples, burn_in_steps=1, reset_chain=False, random_edge=True, dtype=dtype, equal_partition=True)
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, diag_eta=0.05, iter_step=1e5, dtype=dtype)
# Set up VMC
vmc = VMC(hamiltonian=H, variational_state=variational_state, optimizer=optimizer, preconditioner=preconditioner, scheduler=scheduler)

if __name__ == "__main__":
    energy_dict = variational_state.expect(H)
    pickle.dump(energy_dict, open("circuitTNF_heis_L10_D6_MC_energy_dict.pkl", "wb"))