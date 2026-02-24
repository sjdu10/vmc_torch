import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import sys
import warnings

import autoray as ar
import numpy as np

# quimb
import quimb.tensor as qtn

# torch
import torch
from mpi4py import MPI

from vmc_torch.experiment.tn_model import *

# from vmc_torch.hamiltonian import spinful_Fermi_Hubbard_square_lattice
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.optimizer import (
    SGD,
    SR,
    Adam,
    DecayScheduler,
    SGD_momentum,
    TrivialPreconditioner,
)
from vmc_torch.sampler import (
    MetropolisExchangeSamplerSpinful,
    MetropolisExchangeSamplerSpinful_hopping,
    MetropolisExchangeSamplerSpinful_2D_reusable,
    MetropolisMPSSamplerSpinful,
)
from vmc_torch.torch_utils import QR, SVD
from vmc_torch.utils import closest_divisible
from vmc_torch.variational_state import Variational_State
from vmc_torch.VMC import VMC
from vmc_torch.fermion_utils import unpack_ftns

# Register safe SVD and QR functions to torch
ar.register_function("torch", "linalg.svd", SVD.apply)
ar.register_function("torch", "linalg.qr", QR.apply)
pwd = "/home/sijingdu/TNVMC/VMC_code/vmc_torch/scripts/4x4/test_data"
warnings.filterwarnings("ignore")

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Hamiltonian parameters
Lx = int(4)
Ly = int(4)
symmetry = "U1SU"
t = 1.0
U = 8.0
N_f = int(Lx * Ly - 2)
# N_f = int(Lx*Ly)
n_fermions_per_spin = (N_f // 2, N_f // 2)
H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin
)
# H1 = spinful_Fermi_Hubbard_square_lattice(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph
# TN parameters
D = 6
chi = 4*D
# chi = -1
dtype = torch.float64
torch.random.manual_seed(RANK)
np.random.seed(RANK)

# Load PEPS
appendix = ''
if symmetry == 'U1_Z2':
    symmetry = 'Z2'
    appendix = '_U1'
elif symmetry == 'U1SU':
    symmetry = 'Z2'
    appendix = '_U1SU'
    
# Load PEPS
peps = unpack_ftns(
    params_path=pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_su_params{appendix}.pkl",
    skeleton_path=pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_skeleton{appendix}.pkl",
    scale=2.0,
    new_symmray_format=True,
)

# VMC sample size
N_samples = int(2e3)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples / SIZE) % 2 != 0:
    N_samples += SIZE

# nn_hidden_dim = Lx*Ly
# model = fTNModel(peps, max_bond=chi, dtype=dtype)
model = fTNModel_reuse(peps, max_bond=chi, dtype=dtype, debug=False)
init_std = 5e-3

model_names = {
    fTNModel: f"fTN{appendix}",
    fTNModel_reuse: f"fTN_reuse{appendix}",
}
model_name = model_names.get(type(model), "UnknownModel")


init_step = 0
final_step = 1
total_steps = final_step - init_step

# Load model parameters
optimizer_state = None
if init_step != 0:
    saved_model_params = torch.load(
        pwd
        + f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth",
        weights_only=False,
    )
    saved_model_state_dict = saved_model_params["model_state_dict"]
    saved_model_params_vec = torch.tensor(saved_model_params["model_params_vec"])
    try:
        model.load_state_dict(saved_model_state_dict)
        print("Loading model parameters")
    except Exception:
        model.load_params(saved_model_params_vec)
        print(
            "Loading model parameters failed, loading model parameters vector instead"
        )
    optimizer_state = saved_model_params.get("optimizer_state", None)


# Set up optimizer and scheduler
learning_rate = 5e-2
scheduler = DecayScheduler(
    init_lr=learning_rate, decay_rate=0.99, patience=50, min_lr=1e-3
)
use_prev_opt = False
if optimizer_state is not None and use_prev_opt:
    optimizer_name = optimizer_state["optimizer"]
    if optimizer_name == "SGD_momentum":
        optimizer = SGD_momentum(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = Adam(learning_rate=learning_rate, weight_decay=1e-5)
    print("Loading optimizer: ", optimizer)
    optimizer.lr = learning_rate
    if isinstance(optimizer, SGD_momentum):
        optimizer.velocity = optimizer_state["velocity"]
    if isinstance(optimizer, Adam):
        optimizer.m = optimizer_state["m"]
        optimizer.v = optimizer_state["v"]
        optimizer.t = optimizer_state["t"]
else:
    # optimizer = SignedSGD(learning_rate=learning_rate)
    # optimizer = SignedRandomSGD(learning_rate=learning_rate)
    optimizer = SGD(learning_rate=learning_rate)
    # optimizer = SGD_momentum(learning_rate=learning_rate, momentum=0.9)
    # optimizer = Adam(learning_rate=learning_rate, t_step=init_step, weight_decay=1e-5)

# Set up sampler
sampler = MetropolisExchangeSamplerSpinful_2D_reusable(
    H.hilbert,
    graph,
    N_samples=N_samples,
    burn_in_steps=10,
    reset_chain=False,
    random_edge=False,
    equal_partition=False,
    dtype=dtype,
    subchain_length=10,
    hopping_rate=0.25,
)
# sampler = MetropolisExchangeSamplerSpinful_hopping(
#     H.hilbert,
#     graph,
#     N_samples=N_samples,
#     burn_in_steps=10,
#     reset_chain=False,
#     random_edge=False,
#     equal_partition=False,
#     dtype=dtype,
#     subchain_length=10,
#     hopping_rate=0.25,
# )

# Set up variational state
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
# Set up SR preconditioner
preconditioner = SR(
    dense=False,
    exact=True if sampler is None else False,
    use_MPI4Solver=True,
    solver="minres",
    diag_eta=1e-4,
    iter_step=5e2,
    dtype=dtype,
    rtol=1e-5,
)
# preconditioner = TrivialPreconditioner()
# Set up VMC
vmc = VMC(
    hamiltonian=H,
    variational_state=variational_state,
    optimizer=optimizer,
    preconditioner=preconditioner,
    scheduler=scheduler,
)


if __name__ == "__main__":
    os.makedirs(
        pwd
        + f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/",
        exist_ok=True,
    )
    record_file = open(
        pwd
        + f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/record{init_step}.txt",
        "w",
    )
    record_file = open(
        pwd
        + f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/record{init_step}.txt",
        "a",
    )

    if RANK == 0:
        # print training information
        print(f"Running VMC for {model_name}")
        print(f"model params: {variational_state.num_params}")
        print(f"Optimizer: {optimizer}")
        print(f"Preconditioner: {preconditioner}")
        print(f"Scheduler: {scheduler}")
        print(f"Sampler: {sampler}")
        print(
            f"2D Fermi-Hubbard model on {Lx}x{Ly} lattice with {N_f} fermions, Sz=0, t={t}, U={U}"
        )
        print(f"Running {total_steps} steps from {init_step} to {final_step}")
        print(f"Model initialized with mean=0, std={init_std}")
        print(f"Learning rate: {learning_rate}")
        print(f"Sample size: {N_samples}")
        print(f"fPEPS bond dimension: {D}, max bond: {chi}")
        print(f"fPEPS symmetry: {symmetry}\n")
        try:
            print(model.model_structure)
        except Exception:
            pass

    sys.stdout = record_file

    if RANK == 0:
        # print training information
        print(f"Running VMC for {model_name}")
        print(f"model params: {variational_state.num_params}")
        print(f"Optimizer: {optimizer}")
        print(f"Preconditioner: {preconditioner}")
        print(f"Scheduler: {scheduler}")
        print(f"Sampler: {sampler}")
        print(
            f"2D Fermi-Hubbard model on {Lx}x{Ly} lattice with {N_f} fermions, Sz=0, t={t}, U={U}"
        )
        print(f"Running {total_steps} steps from {init_step} to {final_step}")
        print(f"Model initialized with mean=0, std={init_std}")
        print(f"Learning rate: {learning_rate}")
        print(f"Sample size: {N_samples}")
        print(f"fPEPS bond dimension: {D}, max bond: {chi}")
        print(f"fPEPS symmetry: {symmetry}\n")
        try:
            print(model.model_structure)
        except Exception:
            pass

    vmc.run(
        init_step,
        init_step + total_steps,
        tmpdir=pwd
        + f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/",
    )
