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
from vmc_torch.experiment.tests.dev.hamiltonian_old import spinful_Fermi_Hubbard_square_lattice
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
Ly = int(2)
symmetry = 'Z2'
t = 1.0
U = 8.0
N_f = int(Lx*Ly-2)
n_fermions_per_spin = (N_f//2, N_f//2)
H = spinful_Fermi_Hubbard_square_lattice(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph
# TN parameters
D = 4
chi = -200
dtype=torch.float64

# Load PEPS
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
# model = fTNModel_test(peps, max_bond=chi, dtype=dtype)
# model = fTN_backflow_Model(peps, max_bond=chi, nn_eta=1.0, num_hidden_layer=2, nn_hidden_dim=2*Lx*Ly, dtype=dtype)
# model = fTN_backflow_Model_embedding(peps, max_bond=chi, nn_eta=1.0, embedding_dim=8, num_hidden_layer=1, nn_hidden_dim=2*Lx*Ly, dtype=dtype)
# model = PureAttention_Model(phys_dim=4, n_site=Lx*Ly, num_attention_blocks=1, embedding_dim=8, attention_heads=4, nn_hidden_dim=2*Lx*Ly, dtype=dtype)
# model = fTN_backflow_attn_Model(peps, max_bond=chi, embedding_dim=8, attention_heads=4, nn_eta=1.0, nn_hidden_dim=2*Lx*Ly, dtype=dtype)
# model = fTN_backflow_attn_Model(peps, max_bond=chi, embedding_dim=16, attention_heads=4, nn_eta=1.0, nn_hidden_dim=2*Lx*Ly, dtype=dtype)
# model = fTN_backflow_attn_Jastrow_Model(peps, max_bond=chi, embedding_dim=8, attention_heads=4, nn_eta=1.0, nn_hidden_dim=2*Lx*Ly, dtype=dtype)
# model = fTN_backflow_attn_Model_boundary(peps, max_bond=chi, embedding_dim=8, attention_heads=4, nn_eta=1.0, nn_hidden_dim=2*Lx*Ly, dtype=dtype)
# model = NeuralBackflow_spinful(H.hi, param_dtype=dtype, hidden_dim=4*Lx*Ly)
# model = HFDS(H.hi, param_dtype=dtype, hidden_dim=4*Lx*Ly, num_hidden_fermions=int(abs(chi))*N_f, jastrow=False)
init_std = 5e-3
# seed = 2
# torch.manual_seed(seed)
# model.apply(lambda x: init_weights_to_zero(x, std=init_std))
# model.apply(lambda x: init_weights_uniform(x, a=-5e-3, b=5e-3))

model_names = {
    fTNModel: 'fTN',
    fTN_backflow_Model: 'fTN_backflow',
    fTN_backflow_Model_embedding: 'fTN_backflow_embedding',
    fTN_backflow_attn_Model: 'fTN_backflow_attn',
    fTN_backflow_attn_Jastrow_Model: 'fTN_backflow_attn_Jastrow',
    fTN_backflow_attn_Model_boundary: 'fTN_backflow_attn_boundary',
    PureAttention_Model: 'PureAttention',
    SlaterDeterminant: 'SlaterDeterminant',
    NeuralBackflow: 'NeuralBackflow',
    NeuralBackflow_spinful: 'NeuralBackflow_spinful',
    HFDS: 'HFDS',
    FFNN: 'FFNN',
    NeuralJastrow: 'NeuralJastrow',
}
model_name = model_names.get(type(model), 'UnknownModel')


init_step = 0
final_step = 400
# Load model parameters
if init_step != 0:
    saved_model_params = torch.load(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
        print('Loading model parameters')
    except:
        model.load_params(saved_model_params_vec)
        print('Loading model parameters failed, loading model parameters vector instead')
    optimizer_state = saved_model_params.get('optimizer_state', None)


# Set up sampler
equal_partition = False # Whether to equally partition the samples to each MPI process, on HPC recommended to set to False
sampler = MetropolisExchangeSamplerSpinful(H.hilbert, graph, N_samples=N_samples, burn_in_steps=16, reset_chain=False, random_edge=False, equal_partition=equal_partition, dtype=dtype)
# Set up variational state
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)



if __name__ == "__main__":
    MC_dict = variational_state.expect(op=H)
    if RANK == 0:
        print(f"MC_dict: {MC_dict}")



