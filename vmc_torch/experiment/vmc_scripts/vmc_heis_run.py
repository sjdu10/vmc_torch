import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
from mpi4py import MPI
import pickle

# torch
import torch
torch.autograd.set_detect_anomaly(False)

# quimb
import quimb.tensor as qtn
import autoray as ar

from vmc_torch.experiment.tn_model import PEPS_model, TN_backflow_attn_Tensorwise_Model_v1, init_weights_uniform
from vmc_torch.sampler import MetropolisExchangeSamplerSpinless
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SR, DecayScheduler
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian_torch import spin_Heisenberg_square_lattice_torch
from vmc_torch.torch_utils import SVD,QR

# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)

from vmc_torch.global_var import DEBUG

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Hamiltonian parameters
Lx = int(4)
Ly = int(4)
J = 1.0
# H, hi, graph = square_lattice_spinful_Fermi_Hubbard(Lx, Ly, t, U, N_f)
H = spin_Heisenberg_square_lattice_torch(Lx,Ly, J, total_sz=0)
graph = H.graph
# TN parameters
D = 4
chi = -1
dtype=torch.float64

# Load PEPS
pwd = '/pscratch/sd/s/sijingdu/VMC/fermion/data'
skeleton = pickle.load(open(pwd+f"/{Lx}x{Ly}/J={J}/D={D}/peps_skeleton.pkl", "rb"))
peps_params = pickle.load(open(pwd+f"/{Lx}x{Ly}/J={J}/D={D}/peps_su_params.pkl", "rb"))
peps = qtn.unpack(peps_params, skeleton)
peps.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype))

# VMC sample size
N_samples = int(1e4)
N_samples = N_samples - N_samples % SIZE + SIZE
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

model = PEPS_model(peps, max_bond=chi)
model = TN_backflow_attn_Tensorwise_Model_v1(
    peps,
    max_bond=chi,
    embedding_dim=16,
    attention_heads=4,
    nn_final_dim=Lx,
    nn_eta=1.0,
    dtype=dtype,
)
model.apply(init_weights_uniform)
model_names = {
    PEPS_model: 'PEPS',
    TN_backflow_attn_Tensorwise_Model_v1: 'TN_backflow_attn_Tensorwise',
}
model_name = model_names.get(type(model), 'UnknownModel')

init_step = 0
total_steps = 200

if init_step != 0:
    saved_model_params = torch.load(pwd+f'/{Lx}x{Ly}/J={J}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except:
        model.load_params(saved_model_params_vec)

# Set up optimizer and scheduler
learning_rate = 1e-1
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-4)
optimizer = SGD(learning_rate=learning_rate)
sampler = MetropolisExchangeSamplerSpinless(H.hilbert, graph, N_samples=N_samples, burn_in_steps=20, reset_chain=False, random_edge=True, dtype=dtype, equal_partition=False)
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, diag_eta=0.05, iter_step=1e5, dtype=dtype)
# Set up VMC
vmc = VMC(hamiltonian=H, variational_state=variational_state, optimizer=optimizer, preconditioner=preconditioner, scheduler=scheduler)

if __name__ == "__main__":
    os.makedirs(pwd+f'/{Lx}x{Ly}/J={J}/D={D}/{model_name}/chi={chi}/', exist_ok=True)
    vmc.run(init_step, init_step+total_steps, tmpdir=pwd+f'/{Lx}x{Ly}/J={J}/D={D}/{model_name}/chi={chi}/')

