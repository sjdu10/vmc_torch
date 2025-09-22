import json
import os
import pickle

import quimb.tensor as qtn
import torch
from funcs import enumerate_bitstrings, state_vector_from_amps
from vmc_torch.experiment.tn_model import circuit_TNF
from vmc_torch.hamiltonian_torch import spin_Heisenberg_chain_torch

import quimb as qu

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from mpi4py import MPI

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


L = 10
D = 4
H = spin_Heisenberg_chain_torch(L, J=1.0, pbc=False, total_sz=0)
ham_quimb = qtn.ham_1d_heis(L, j=1.0, cyclic=False)
ham_quimb.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float64))
graph = H.graph
hilbert = H.hilbert
dtype = torch.float64

exp_all_states = list(enumerate_bitstrings(L))  # noqa: F405
projected_states = torch.tensor(hilbert.all_states())
projected_states_tuple = [tuple(state.tolist()) for state in projected_states]


su_params, su_skeleton = pickle.load(
    open(f"circuitTNF_heis_L{L}_D{D}_su_state.pkl", "rb")
)
su_mps = qtn.unpack(su_params, su_skeleton)
su_energy = su_mps.compute_local_expectation_exact(ham_quimb.terms, normalized=True)

print(f"SU state energy: {su_energy}")


ham_ed = qu.ham_heis(L, j=1.0)
exact_energy = qu.groundenergy(ham_ed)
print(f"Exact ground state energy: {exact_energy}")
ham_matrix = ham_ed.toarray()

max_bonds = [2, 4, 6, 8]
depths = [1]


for max_bond in max_bonds:
    for tau in [0.0]:
        for depth in depths:
            model = circuit_TNF(
                su_mps,
                ham_quimb,
                trotter_tau=tau,
                depth=depth,
                max_bond=max_bond,
                dtype=torch.float64,
                from_which="xmax",
            )

            amps = model(projected_states)
            amp_dict = dict(zip(projected_states_tuple, amps.detach().numpy()))
            state_vec = state_vector_from_amps(amp_dict, L)

            E = (
                (state_vec @ np.matmul(ham_matrix, state_vec)) / (state_vec @ state_vec)
            ).real
            print(f"Variational state energy: {E}, tau={tau}, max_bond={max_bond}")
            # collect_data_dict = {(depth, tau, max_bond): E}
            collect_data_dict = {
                f"L{L}_D{D}_depth={depth}_tau={tau}_maxbond={max_bond}": (
                    E,
                    np.float64(su_energy),
                    np.float64(exact_energy),
                )
            }
            # add to existing data file or create a new one
            data_file = f"./data/circuitTNF_heis_L{L}_D{D}_exact_sampling_results_{model.from_which}.json"

            if os.path.exists(data_file):
                with open(data_file, "r") as f:
                    existing_data = json.load(f)
                existing_data.update(collect_data_dict)
                with open(data_file, "w") as f:
                    json.dump(existing_data, f, indent=4)
            else:
                with open(data_file, "w") as f:
                    json.dump(collect_data_dict, f, indent=4)
