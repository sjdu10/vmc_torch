import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"
import json
import pickle
import tqdm

import quimb.tensor as qtn
import torch
from funcs import enumerate_bitstrings, state_vector_from_amps
from models import circuit_TNF_2d_SU
from vmc_torch.hamiltonian_torch import spin_Heisenberg_square_lattice_torch

import quimb as qu


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
# distribute the work to different ranks, take care of the last few states if SIZE does not divide the total number of states
projected_states_local = projected_states[RANK::SIZE]
print(f"Rank {RANK} has {len(projected_states_local)} states to process.")
projected_states_local_tuple = [tuple(state.tolist()) for state in projected_states_local]


su_params, su_skeleton = pickle.load(
    open(f"./2D/circuitTNF_heis2D_Lx{Lx}_Ly{Ly}_D{D}_su_state.pkl", "rb")
)
su_peps = qtn.unpack(su_params, su_skeleton)
if RANK == 0:
    su_energy = su_peps.compute_local_expectation_exact(ham_quimb.terms, normalized=True)
    print(f"SU state energy: {su_energy}")


ham_ed = qu.ham_heis_2D(Lx, Ly, j=1.0, cyclic=False)
if RANK == 0:
    exact_energy = -9.189207065192965 if (Lx, Ly) == (4, 4) else qu.groundenergy(ham_ed)
    print(f"Exact ground state energy: {exact_energy}")
ham_matrix = csr_matrix(ham_ed.toarray())

max_bonds = [10]
depths = [4,6,8]

with torch.no_grad():
    for max_bond in max_bonds:
        for tau in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
            for depth in depths:
                # collect_data_dict = {(depth, tau, max_bond): E}
                collect_data_dict = {
                    f"Lx{Lx}_Ly{Ly}_D{D}_depth={depth}_tau={tau}_maxbond={max_bond}": (
                        None,
                    )
                }
                # check if this piece of data already exists
                data_file = f"./data/circuitTNF2d_su_heis_Lx{Lx}_Ly{Ly}_D{D}_exact_sampling_results_inverse_su.json"

                if os.path.exists(data_file):
                    with open(data_file, "r") as f:
                        existing_data = json.load(f)
                    if list(collect_data_dict.keys())[0] in existing_data.keys():
                        if RANK == 0:
                            print(f"Data for depth={depth}, tau={tau}, max_bond={max_bond} already exists, skipping...")
                        continue

                model = circuit_TNF_2d_SU(
                    su_peps,
                    ham_quimb,
                    trotter_tau=tau,
                    depth=depth,
                    max_bond=max_bond,
                    dtype=torch.float64,
                    second_order_reflect=False,
                )
                if RANK == 0:
                    pbar = tqdm.tqdm(total=len(projected_states_local))
                # equally distribute the work to different ranks
                amps = []
                for state in projected_states_local:
                    amp = model(state.unsqueeze(0))
                    amps.append(amp)
                    if RANK == 0:
                        pbar.update(1)
                if RANK == 0:
                    pbar.close()
                amps = torch.cat(amps, dim=0)
                # amps = model(projected_states)
                amp_dict = dict(zip(projected_states_local_tuple, amps.detach().numpy()))
                COMM.Barrier()

                all_amp_dict = COMM.gather(amp_dict, root=0)
                if RANK != 0:
                    continue
                # combine the list of dicts to a single dict
                all_amp_dict = {k: v for d in all_amp_dict for k, v in d.items()}
                state_vec = state_vector_from_amps(all_amp_dict, Lx*Ly)

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
                data_file = f"./data/circuitTNF2d_su_heis_Lx{Lx}_Ly{Ly}_D{D}_exact_sampling_results_inverse_su.json"

                if os.path.exists(data_file):
                    with open(data_file, "r") as f:
                        existing_data = json.load(f)
                    existing_data.update(collect_data_dict)
                    with open(data_file, "w") as f:
                        json.dump(existing_data, f, indent=4)
                else:
                    with open(data_file, "w") as f:
                        json.dump(collect_data_dict, f, indent=4)
