import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import json
import time

import quimb.tensor as qtn
import torch
from funcs import enumerate_bitstrings, state_vector_from_amps
from models import circuit_TNF_2d, circuit_TNF_2d_SU
from vmc_torch.hamiltonian_torch import spin_Heisenberg_square_lattice_torch
import tqdm
import quimb as qu
from vmc_torch.torch_utils import SVD,QR_tao
import autoray as ar

import numpy as np
# from quimb.core import isreal, issparse
from scipy.sparse import csr_matrix
from mpi4py import MPI

# # Register safe SVD and QR functions to torch
# ar.register_function('torch','linalg.svd',SVD.apply)
# ar.register_function('torch','linalg.qr',QR_tao.apply)

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
# if RANK == 0:
#     su_energy = su_peps.compute_local_expectation_exact(ham_quimb.terms, normalized=True)
#     print(f"SU state energy: {su_energy}")


# ham_ed = qu.ham_heis_2D(Lx, Ly, j=1.0, cyclic=False)
# if RANK == 0:
#     exact_energy = -9.189207065192965 if (Lx, Ly) == (4, 4) else qu.groundenergy(ham_ed)
#     print(f"Exact ground state energy: {exact_energy}")
# ham_matrix = csr_matrix(ham_ed.toarray())

max_bonds = [2, 4, 6, 8, 10]
depths = [4, 6, 8, 10, 12, 14, 16, 18]
from_which = "zmax"
with torch.no_grad():
    for max_bond in max_bonds:
        for tau in [0.04]:
            for depth in depths:
                # collect_data_dict = {(depth, tau, max_bond): E}
                collect_data_dict = {
                    f"Lx{Lx}_Ly{Ly}_D{D}_depth={depth}_tau={tau}_maxbond={max_bond}_su": (
                        None,
                    )
                }
                # check if this piece of data already exists
                data_file = f"./data/circuitTNF2d_heis_Lx{Lx}_Ly{Ly}_D{D}_exact_sampling_time_cost.json"

                if os.path.exists(data_file):
                    with open(data_file, "r") as f:
                        existing_data = json.load(f)
                    if list(collect_data_dict.keys())[0] in existing_data.keys():
                        if RANK == 0:
                            print(f"Data for depth={depth}, tau={tau}, max_bond={max_bond} already exists, skipping...")
                        continue
                COMM.Barrier()

                model = circuit_TNF_2d_SU(
                    su_peps,
                    ham_quimb,
                    trotter_tau=tau,
                    depth=depth,
                    max_bond=max_bond,
                    dtype=torch.float64,
                    profile_time=True,
                )
                if RANK == 0:
                    pbar = tqdm.tqdm(total=len(projected_states_local))
                # equally distribute the work to different ranks
                amps = []
                t0 = time.time()
                N_sample = 5
                for state in projected_states_local[:N_sample]:  # for recording time cost only
                    amp = model(state.unsqueeze(0))
                    amps.append(amp)
                    if RANK == 0:
                        pbar.update(1)
                if RANK == 0:
                    pbar.close()
                t1 = time.time()

                average_time = (t1 - t0) / N_sample
                if RANK == 0:
                    print(f"Depth={depth}, tau={tau}, max_bond={max_bond}, average time per state: {average_time:.4f} seconds")

                collect_data_dict = {
                    f"Lx{Lx}_Ly{Ly}_D{D}_depth={depth}_tau={tau}_maxbond={max_bond}_su": average_time
                }
                
                
                # add to existing data file or create a new one
                data_file = f"./data/circuitTNF2d_heis_Lx{Lx}_Ly{Ly}_D{D}_exact_sampling_time_cost.json"

                if os.path.exists(data_file):
                    with open(data_file, "r") as f:
                        existing_data = json.load(f)
                    existing_data.update(collect_data_dict)
                    with open(data_file, "w") as f:
                        json.dump(existing_data, f, indent=4)
                else:
                    with open(data_file, "w") as f:
                        json.dump(collect_data_dict, f, indent=4)
