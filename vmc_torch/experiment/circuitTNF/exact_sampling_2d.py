import json
import os
import pickle

import quimb.tensor as qtn
import torch
from funcs import enumerate_bitstrings, state_vector_from_amps
from models import circuit_TNF_2d
from vmc_torch.hamiltonian_torch import spin_Heisenberg_square_lattice_torch

import quimb as qu

# os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "10"
# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
# from quimb.core import isreal, issparse
from scipy.sparse import csr_matrix
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

exp_all_states = list(enumerate_bitstrings(Lx*Ly))  # noqa: F405
projected_states = torch.tensor(hilbert.all_states())
projected_states_tuple = [tuple(state.tolist()) for state in projected_states]


su_params, su_skeleton = pickle.load(
    open(f"./2D/circuitTNF_heis2D_Lx{Lx}_Ly{Ly}_D{D}_su_state.pkl", "rb")
)
su_peps = qtn.unpack(su_params, su_skeleton)
# su_energy = su_peps.compute_local_expectation_exact(ham_quimb.terms, normalized=True)
su_energy=0.0
print(f"SU state energy: {su_energy}")


ham_ed = qu.ham_heis_2D(Lx, Ly, j=1.0, cyclic=False)

exact_energy = qu.groundenergy(ham_ed)
print(f"Exact ground state energy: {exact_energy}")
ham_matrix = csr_matrix(ham_ed.toarray())

max_bonds = [10]
depths = [1,]

with torch.no_grad():
    for max_bond in max_bonds:
        for tau in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
            for depth in depths:
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

                amps = model(projected_states)
                amp_dict = dict(zip(projected_states_tuple, amps.detach().numpy()))
                state_vec = state_vector_from_amps(amp_dict, Lx*Ly)

                E = (
                    (state_vec @ (ham_matrix @ state_vec)) / (state_vec @ state_vec)
                ).real
                print(f"Variational state energy: {E}, tau={tau}, max_bond={max_bond}")
                # collect_data_dict = {(depth, tau, max_bond): E}
                collect_data_dict = {
                    f"Lx{Lx}_Ly{Ly}_D{D}_depth={depth}_tau={tau}_maxbond={max_bond}": (
                        E,
                        np.float64(su_energy),
                        np.float64(exact_energy),
                    )
                }
                # add to existing data file or create a new one
                data_file = f"./data/circuitTNF2d_heis_Lx{Lx}_Ly{Ly}_D{D}_exact_sampling_results_{model.from_which}.json"

                if os.path.exists(data_file):
                    with open(data_file, "r") as f:
                        existing_data = json.load(f)
                    existing_data.update(collect_data_dict)
                    with open(data_file, "w") as f:
                        json.dump(existing_data, f, indent=4)
                else:
                    with open(data_file, "w") as f:
                        json.dump(collect_data_dict, f, indent=4)
