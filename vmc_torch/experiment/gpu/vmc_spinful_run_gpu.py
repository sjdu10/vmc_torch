import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
import sys
from mpi4py import MPI
import pickle
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'
# torch
import torch
torch.autograd.set_detect_anomaly(False)

# quimb
import quimb.tensor as qtn
import autoray as ar

from vmc_torch.experiment.tn_model import fTNModel, fTNModel_test, fTN_backflow_attn_Model, fTN_backflow_attn_Jastrow_Model
from vmc_torch.experiment.tn_model import fTN_backflow_Model, fTN_backflow_Model_Blockwise, fTN_backflow_attn_Model_Stacked, fTN_backflow_attn_Model_boundary, fTN_backflow_Model_embedding
from vmc_torch.experiment.tn_model import fTN_Transformer_Model, fTN_Transformer_Proj_Model, fTN_Transformer_Proj_lazy_Model, fTN_NN_proj_Model, fTN_NN_proj_variable_Model
from vmc_torch.experiment.tn_model import PureAttention_Model, NeuralBackflow_spinful, SlaterDeterminant, NeuralBackflow, FFNN, NeuralJastrow, HFDS
from vmc_torch.experiment.tn_model import init_weights_to_zero, init_weights_uniform
from vmc_torch.sampler import MetropolisExchangeSamplerSpinful
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SignedSGD, SignedRandomSGD, SR, TrivialPreconditioner, Adam, SGD_momentum, DecayScheduler
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian import spinful_Fermi_Hubbard_square_lattice
from vmc_torch.torch_utils import SVD,QR
from vmc_torch.fermion_utils import generate_random_fpeps

# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)

from vmc_torch.global_var import DEBUG
from vmc_torch.utils import closest_divisible


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Hamiltonian parameters
Lx = int(4)
Ly = int(16)
symmetry = 'Z2'
t = 1.0
U = 8.0
N_f = int(Lx*Ly-8)
n_fermions_per_spin = (N_f//2, N_f//2)
H = spinful_Fermi_Hubbard_square_lattice(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph
# TN parameters
D = 4
chi = -1
dtype=torch.float64

# # Load PEPS
skeleton = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_skeleton.pkl", "rb"))
peps_params = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_su_params.pkl", "rb"))
peps = qtn.unpack(peps_params, skeleton)
peps.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype))

# # randomize the PEPS tensors
# peps.apply_to_arrays(lambda x: torch.randn_like(torch.tensor(x, dtype=dtype), dtype=dtype))

# VMC sample size
N_samples = int(2e3)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

model = fTNModel(peps, max_bond=chi, dtype=dtype)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
init_std = 5e-3
# seed = 2
# torch.manual_seed(seed)
# model.apply(lambda x: init_weights_to_zero(x, std=init_std))
# model.apply(lambda x: init_weights_uniform(x, a=-5e-3, b=5e-3))

model_names = {
    fTNModel: 'fTN',
}
model_name = model_names.get(type(model), 'UnknownModel')


init_step = 0
final_step = 400
total_steps = final_step - init_step
# Load model parameters
optimizer_state = None
if init_step != 0:
    saved_model_params = torch.load(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}_gpu/chi={chi}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
        print('Loading model parameters')
    except:
        model.load_params(saved_model_params_vec)
        print('Loading model parameters failed, loading model parameters vector instead')
    optimizer_state = saved_model_params.get('optimizer_state', None)

# Set up optimizer and scheduler
learning_rate = 1e-1
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=5e-3)
use_prev_opt = True
if optimizer_state is not None and use_prev_opt:
    optimizer_name = optimizer_state['optimizer']
    if optimizer_name == 'SGD_momentum':
        optimizer = SGD_momentum(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=learning_rate, weight_decay=1e-5)
    print('Loading optimizer: ', optimizer)
    optimizer.lr = learning_rate
    if isinstance(optimizer, SGD_momentum):
        optimizer.velocity = optimizer_state['velocity']
    if isinstance(optimizer, Adam):
        optimizer.m = optimizer_state['m']
        optimizer.v = optimizer_state['v']
        optimizer.t = optimizer_state['t']
else:
    # optimizer = SignedSGD(learning_rate=learning_rate)
    # optimizer = SignedRandomSGD(learning_rate=learning_rate)
    optimizer = SGD(learning_rate=learning_rate)
    # optimizer = SGD_momentum(learning_rate=learning_rate, momentum=0.9)
    # optimizer = Adam(learning_rate=learning_rate, t_step=init_step, weight_decay=1e-5)

# Set up sampler
sampler = MetropolisExchangeSamplerSpinful(H.hilbert, graph, N_samples=N_samples, burn_in_steps=1, reset_chain=False, random_edge=False, equal_partition=True, dtype=dtype)
# Set up variational state
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
# Set up SR preconditioner
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, solver='minres', diag_eta=1e-3, iter_step=1e3, dtype=dtype, rtol=1e-4)
# preconditioner = TrivialPreconditioner()
# Set up VMC
vmc = VMC(hamiltonian=H, variational_state=variational_state, optimizer=optimizer, preconditioner=preconditioner, scheduler=scheduler)
# if __name__ == "__main__":
    
torch.autograd.set_detect_anomaly(False)
os.makedirs(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}_gpu/chi={chi}/', exist_ok=True)
record_file = open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}_gpu/chi={chi}/record{init_step}.txt', 'w')
if RANK == 0:
    # print training information
    print(f"Running VMC for {model_name}_gpu")
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
        print(model.model_structure)
    except:
        pass
    sys.stdout = record_file

if RANK == 0:
    # print training information
    print(f"Running VMC for {model_name}_gpu")
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
        print(model.model_structure)
    except:
        pass

vmc.run(init_step, init_step+total_steps, tmpdir=pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}_gpu/chi={chi}/')


