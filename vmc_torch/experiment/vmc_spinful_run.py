import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
import sys
from mpi4py import MPI
import pickle

# torch
import torch
torch.autograd.set_detect_anomaly(False)

# quimb
import quimb.tensor as qtn
import autoray as ar

from vmc_torch.experiment.tn_model import fTNModel, fTNModel_test, fTN_NN_proj_Model, fTN_NN_proj_variable_Model, SlaterDeterminant, NeuralBackflow, FFNN, NeuralJastrow
from vmc_torch.experiment.tn_model import fTN_backflow_Model, fTN_backflow_attn_Model, fTN_backflow_Model_Blockwise, fTN_backflow_attn_Model_Stacked, fTN_backflow_attn_Model_boundary, fTN_backflow_Model_embedding
from vmc_torch.experiment.tn_model import fTN_Transformer_Model, fTN_Transformer_Proj_Model, fTN_Transformer_Proj_lazy_Model
from vmc_torch.experiment.tn_model import PureAttention_Model
from vmc_torch.experiment.tn_model import init_weights_to_zero
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
Lx = int(3)
Ly = int(2)
symmetry = 'U1'
t = 1.0
U = 8.0
N_f = int(Lx*Ly-2)
n_fermions_per_spin = (N_f//2, N_f//2)
H = spinful_Fermi_Hubbard_square_lattice(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph
# TN parameters
D = 4
chi = -3
dtype=torch.float64

# # Load PEPS
skeleton = pickle.load(open(f"../../data/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_skeleton.pkl", "rb"))
peps_params = pickle.load(open(f"../../data/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_su_params.pkl", "rb"))
peps = qtn.unpack(peps_params, skeleton)
peps.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype))

# # randomize the PEPS tensors
# peps.apply_to_arrays(lambda x: torch.randn_like(torch.tensor(x, dtype=dtype), dtype=dtype))

# VMC sample size
N_samples = int(1e4)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

# model = fTNModel(peps, max_bond=chi, dtype=dtype)
# model = fTNModel_test(peps, max_bond=chi, dtype=dtype)
# model = fTN_backflow_Model(peps, max_bond=chi, nn_eta=1.0, num_hidden_layer=2, nn_hidden_dim=2*Lx*Ly, dtype=dtype)
# model = fTN_backflow_Model_embedding(peps, max_bond=chi, nn_eta=1.0, embedding_dim=8, num_hidden_layer=1, nn_hidden_dim=2*Lx*Ly, dtype=dtype)
# model = PureAttention_Model(phys_dim=4, n_site=Lx*Ly, num_attention_blocks=1, embedding_dim=8, attention_heads=4, nn_hidden_dim=2*Lx*Ly, dtype=dtype)
model = fTN_backflow_attn_Model(peps, max_bond=chi, embedding_dim=4, attention_heads=2, nn_eta=1.0, nn_hidden_dim=2*Lx*Ly, dtype=dtype)
# model = fTN_backflow_attn_Model_boundary(peps, max_bond=chi, embedding_dim=8, attention_heads=4, nn_eta=1.0, nn_hidden_dim=2*Lx*Ly, dtype=dtype)
# model = fTN_backflow_Model_Blockwise(peps, max_bond=chi, nn_eta=1.0, num_hidden_layer=2, nn_hidden_dim=2*Lx*Ly, dtype=dtype)
# model = fTN_backflow_attn_Model_Stacked(
#     peps,
#     max_bond=chi, 
#     num_attention_blocks=2,
#     embedding_dim=8, 
#     d_inner=2*Lx*Ly,
#     attention_heads=2, 
#     nn_eta=1.0, 
#     nn_hidden_dim=2*Lx*Ly, 
#     dtype=dtype
# )
# model = fTN_Transformer_Model(
#     peps, 
#     max_bond=chi, 
#     nn_eta=1.0, 
#     d_model=2**2, 
#     nhead=2, 
#     num_encoder_layers=2, 
#     num_decoder_layers=2,
#     dim_feedforward=32,
#     dropout=0.0,
#     dtype=dtype,
# )
# model = fTN_Transformer_Proj_Model(
#     peps,
#     max_bond=chi,
#     nn_eta=1e-2,
#     d_model=2**3,
#     nhead=2,
#     num_encoder_layers=1,
#     num_decoder_layers=1,
#     dim_feedforward=2**5,
#     dropout=0.0,
#     dtype=dtype,
# )
# model = fTN_Transformer_Proj_lazy_Model(
#     peps,
#     max_bond=chi,
#     nn_eta=1.0,
#     d_model=8,
#     nhead=2,
#     num_encoder_layers=1,
#     num_decoder_layers=1,
#     dim_feedforward=16,
#     dropout=0.0,
#     dtype=dtype,
# )
init_std = 5e-2
seed = 2
torch.manual_seed(seed)
model.apply(lambda x: init_weights_to_zero(x, std=init_std))
# model.apply(lambda x: init_weights_kaiming(x))

model_names = {
    fTNModel: 'fTN',
    fTNModel_test: 'fTN_test',
    fTN_backflow_Model: 'fTN_backflow',
    fTN_backflow_Model_embedding: 'fTN_backflow_embedding',
    fTN_backflow_attn_Model: 'fTN_backflow_attn',
    fTN_backflow_Model_Blockwise: 'fTN_backflow_Blockwise',
    fTN_backflow_attn_Model_Stacked: 'fTN_backflow_attn_Stacked',
    fTN_backflow_attn_Model_boundary: 'fTN_backflow_attn_boundary',
    fTN_NN_proj_Model: 'fTN_NN_proj',
    fTN_NN_proj_variable_Model: 'fTN_NN_proj_variable',
    fTN_Transformer_Model: 'fTN_Transformer',
    fTN_Transformer_Proj_Model:'fTN_Transformer_Proj',
    fTN_Transformer_Proj_lazy_Model:'fTN_Transformer_Proj_lazy',
    PureAttention_Model: 'PureAttention',
    SlaterDeterminant: 'SlaterDeterminant',
    NeuralBackflow: 'NeuralBackflow',
    FFNN: 'FFNN',
    NeuralJastrow: 'NeuralJastrow',
}
model_name = model_names.get(type(model), 'UnknownModel')


init_step = 10
final_step = 450
total_steps = final_step - init_step
# Load model parameters
if init_step != 0:
    saved_model_params = torch.load(f'../../data/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except:
        model.load_params(saved_model_params_vec)
    optimizer_state = saved_model_params.get('optimizer_state', None)

# Set up optimizer and scheduler
learning_rate = 1e-1
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=10, min_lr=5e-2)
optimizer_state = None
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
sampler = MetropolisExchangeSamplerSpinful(H.hilbert, graph, N_samples=N_samples, burn_in_steps=16, reset_chain=False, random_edge=False, equal_partition=False, dtype=dtype)
# Set up variational state
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
# Set up SR preconditioner
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, diag_eta=1e-3, iter_step=1e5, dtype=dtype)
# preconditioner = TrivialPreconditioner()
# Set up VMC
vmc = VMC(hamiltonian=H, variational_state=variational_state, optimizer=optimizer, preconditioner=preconditioner, scheduler=scheduler)
# if __name__ == "__main__":
    
torch.autograd.set_detect_anomaly(False)
os.makedirs(f'../../data/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/', exist_ok=True)
record_file = open(f'../../data/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/record{init_step}.txt', 'w')
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
        print(model.model_structure)
    except:
        pass
    sys.stdout = record_file

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
        print(model.model_structure)
    except:
        pass
# with pyinstrument.Profiler() as prof:
vmc.run(init_step, init_step+total_steps, tmpdir=f'../../data/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/')
# if RANK == 0:
#     prof.print()

