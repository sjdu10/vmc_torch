import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
from mpi4py import MPI
import pickle
# pwd = os.getcwd()
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'

# torch
import torch
torch.autograd.set_detect_anomaly(False)

# quimb
import quimb.tensor as qtn
import autoray as ar

from vmc_torch.experiment.tn_model import *
from vmc_torch.experiment.tn_model import init_weights_to_zero, init_weights_uniform
from vmc_torch.sampler import MetropolisExchangeSamplerSpinful, MetropolisExchangeSamplerSpinful_2D_reusable
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SR, DecayScheduler
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.torch_utils import SVD,QR
from vmc_torch.fermion_utils import get_psi_from_fTN

# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)

from vmc_torch.global_var import DEBUG
from vmc_torch.utils import closest_divisible


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Hamiltonian parameters
Lx = int(8)
Ly = int(8)
symmetry = 'Z2'
t = 1.0
U = 8.0
N_f = int(Lx*Ly)
n_fermions_per_spin = (N_f//2, N_f//2)
H = spinful_Fermi_Hubbard_square_lattice_torch(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph
# TN parameters
D = 4
chi = 64
dtype=torch.float64

if symmetry == 'U1_Z2':
    su_skeleton = 'peps_skeleton_U1.pkl'
    su_params = 'peps_su_params_U1.pkl'
    symmetry = 'Z2'
else:
    su_skeleton = 'peps_skeleton.pkl'
    su_params = 'peps_su_params.pkl'
    
# Load PEPS
skeleton = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{su_skeleton}", "rb"))
peps_params = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{su_params}", "rb"))
peps = qtn.unpack(peps_params, skeleton)
scale = 1.0
peps.apply_to_arrays(lambda x: torch.tensor(scale*x, dtype=dtype))

# VMC sample size
N_samples = int(20)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

# model = fTNModel_reuse(peps, max_bond=chi, dtype=dtype, debug=False)
model = fTNModel(peps, max_bond=chi, dtype=dtype)
# model = fTN_backflow_attn_Tensorwise_Model(
#     peps,
#     max_bond=chi,
#     embedding_dim=16,
#     attention_heads=4,
#     nn_final_dim=D,
#     nn_eta=1.0,
#     dtype=dtype,
# )
init_std = 5e-3
# model.apply(lambda x: init_weights_to_zero(x, std=init_std))
# model.apply(lambda x: init_weights_uniform(x, a=-1*init_std, b=init_std))

model_names = {
    fTNModel: 'fTN',
    fTNModel_reuse: 'fTN_reuse',
    fTN_backflow_Model: 'fTN_backflow',
    fTN_backflow_attn_Model: 'fTN_backflow_attn',
    fTN_backflow_attn_Tensorwise_Model: 'fTN_backflow_attn_Tensorwise',
    fTN_backflow_attn_Tensorwise_Model_v1: 'fTN_backflow_attn_Tensorwise_v1',
}

model_name = model_names.get(type(model), 'UnknownModel')

init_step = 0
final_step = 1000
total_steps = final_step - init_step
if init_step != 0:
    saved_model_params = torch.load(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except:
        model.load_params(saved_model_params_vec)

# optimizer = SignedSGD(learning_rate=0.05)
# Set up optimizer and scheduler
learning_rate = 0.1
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-4)
optimizer = SGD(learning_rate=learning_rate)
# sampler = MetropolisExchangeSamplerSpinful(H.hilbert, graph, N_samples=N_samples, burn_in_steps=20, reset_chain=False, random_edge=False, equal_partition=False, dtype=dtype)
sampler = MetropolisExchangeSamplerSpinful_2D_reusable(H.hilbert, graph, N_samples=N_samples, burn_in_steps=1, reset_chain=False, random_edge=False, equal_partition=False, dtype=dtype)
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, solver='minres', diag_eta=1e-3, iter_step=5e2, dtype=dtype, rtol=1e-4)
# preconditioner = TrivialPreconditioner()
# Set up VMC
vmc = VMC(hamiltonian=H, variational_state=variational_state, optimizer=optimizer, preconditioner=preconditioner, scheduler=scheduler)

# BUG: 1. reusable model reuse algorithms (?) 2. sampler sequence (?)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(False)
    os.makedirs(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/', exist_ok=True)
    record_file = open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/record{init_step}.txt', 'w')
    if RANK == 0:
        # print training information
        print(f"Running VMC for {model_name}")
        print(f'model params: {variational_state.num_params}')
        print(f"Optimizer: {optimizer}")
        print(f"Preconditioner: {preconditioner}")
        print(f"Scheduler: {scheduler}")
        print(f"Sampler: {sampler}")
        print(f'2D Fermi-Hubbard model on {Lx}x{Ly} lattice with {N_f} fermions, Sz=0, t={t}, U={U}')
        print(f"Running {total_steps} steps from {init_step} to {final_step}")
        print(f'Model initialized with mean=0, std={init_std}')
        print(f'Learning rate: {learning_rate}')
        print(f'Sample size: {N_samples}')
        print(f'fPEPS bond dimension: {D}, max bond: {chi}')
        print(f'fPEPS symmetry: {symmetry}\n')
        try:
            print(f'model structure: {model.model_structure}')
        except:
            pass
        sys.stdout = record_file
    COMM.Barrier()  # Ensure all processes are synchronized before starting
    if RANK == 0:
        # print training information
        print(f"Running VMC for {model_name}")
        print(f'model params: {variational_state.num_params}')
        print(f"Optimizer: {optimizer}")
        print(f"Preconditioner: {preconditioner}")
        print(f"Scheduler: {scheduler}")
        print(f"Sampler: {sampler}")
        print(f'2D Fermi-Hubbard model on {Lx}x{Ly} lattice with {N_f} fermions, Sz=0, t={t}, U={U}')
        print(f"Running {total_steps} steps from {init_step} to {final_step}")
        print(f'Model initialized with mean=0, std={init_std}')
        print(f'Learning rate: {learning_rate}')
        print(f'Sample size: {N_samples}')
        print(f'fPEPS bond dimension: {D}, max bond: {chi}')
        print(f'fPEPS symmetry: {symmetry}\n')
        try:
            print(f'model structure: {model.model_structure}')
        except:
            pass

    vmc.run(init_step, init_step+total_steps, tmpdir=pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/')

