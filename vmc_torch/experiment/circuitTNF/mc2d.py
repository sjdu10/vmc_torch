import json
import os
import pickle

import quimb.tensor as qtn
import torch
from models import circuit_TNF_2d
from vmc_torch.hamiltonian_torch import spin_Heisenberg_square_lattice_torch
from vmc_torch.sampler import MetropolisExchangeSamplerSpinless
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SR, DecayScheduler
from vmc_torch.VMC import VMC

# os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# from quimb.core import isreal, issparse
from mpi4py import MPI

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


Lx = 4
Ly = 4
D = 4
H = spin_Heisenberg_square_lattice_torch(Lx, Ly, J=1.0, pbc=False, total_sz=0)
ham_quimb = qtn.ham_2d_heis(Lx, Ly, j=1.0, cyclic=False)
ham_quimb.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float64))
graph = H.graph
hilbert = H.hilbert
dtype = torch.float64


su_params, su_skeleton = pickle.load(
    open(f"./2D/circuitTNF_heis2D_Lx{Lx}_Ly{Ly}_D{D}_su_state.pkl", "rb")
)
su_peps = qtn.unpack(su_params, su_skeleton)
# su_energy = su_peps.compute_local_expectation_exact(ham_quimb.terms, normalized=True)
# su_energy=0.0
# print(f"SU state energy: {su_energy}")

tau = 0.1
depth = 2
max_bond = 8

model = circuit_TNF_2d(
    su_peps,
    ham_quimb,
    trotter_tau=tau,
    depth=depth,
    max_bond=max_bond,
    dtype=torch.float64,
    from_which="zmax",
    mode='projector3d',
)

# VMC sample size
N_samples = int(2e3)
N_samples = N_samples - N_samples % SIZE + SIZE
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

# Set up optimizer and scheduler
learning_rate = 1e-1
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-4)
optimizer = SGD(learning_rate=learning_rate)
sampler = MetropolisExchangeSamplerSpinless(H.hilbert, graph, N_samples=N_samples, burn_in_steps=2, reset_chain=False, random_edge=False, dtype=dtype, equal_partition=False)
sampler.current_config = torch.tensor([0,1]*(Lx*Ly//2), dtype=torch.int8) # Neel initial state
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, diag_eta=0.05, iter_step=1e5, dtype=dtype)

if __name__ == "__main__":
    data_dict = variational_state.expect(H)
    if RANK == 0:
        data_dict['max_bond'] = model.max_bond
        data_dict['trotter_tau'] = model.trotter_tau
        data_dict['depth'] = model.depth
        data_dict['Lx'] = Lx
        data_dict['Ly'] = Ly
        data_dict['from_which'] = model.from_which
        print(data_dict)
    # pickle.dump(energy_dict, open("circuitTNF_heis_L10_D6_MC_energy_dict.pkl", "wb"))