import os
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
# os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"
from mpi4py import MPI
import pickle


# torch
import torch
torch.autograd.set_detect_anomaly(False)

# quimb
import quimb.tensor as qtn
import autoray as ar

from vmc_torch.experiment.tn_model import fTNModel, fTN_backflow_attn_Tensorwise_Model, fTN_backflow_attn_Tensorwise_Model_v1
from vmc_torch.experiment.tn_model import init_weights_to_zero
from vmc_torch.sampler import MetropolisExchangeSamplerSpinful
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SR
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian import spinful_Fermi_Hubbard_square_lattice
from vmc_torch.torch_utils import SVD,QR
from vmc_torch.utils import closest_divisible
from vmc_torch.fermion_utils import generate_random_fpeps

# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Hamiltonian parameters
Lx = int(4)
Ly = int(4)
symmetry = 'Z2'
t = 1.0
U = 8.0
N_f = int(Lx*Ly-2)
# N_f=12
n_fermions_per_spin = (N_f//2, N_f//2)
H = spinful_Fermi_Hubbard_square_lattice(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph
# TN parameters
D = 10
chi = 8
dtype=torch.float64

# Load PEPS
try:
    skeleton = pickle.load(open(f"../../data/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_skeleton.pkl", "rb"))
    peps_params = pickle.load(open(f"../../data/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_su_params.pkl", "rb"))
    peps = qtn.unpack(peps_params, skeleton)
except:
    peps = generate_random_fpeps(Lx, Ly, D=D, seed=2, symmetry=symmetry, Nf=N_f, spinless=False)[0]
peps_np = peps.copy()
peps.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype))

# VMC sample size
N_samples = 2
N_samples = closest_divisible(N_samples, SIZE)

# model = fTNModel(peps, max_bond=chi)
# model = fTN_backflow_attn_Tensorwise_Model(peps, max_bond=chi, embedding_dim=16, attention_heads=4, nn_final_dim=4, nn_eta=1.0, dtype=dtype)
model = fTN_backflow_attn_Tensorwise_Model_v1(peps, max_bond=chi, embedding_dim=16, attention_heads=4, nn_final_dim=4, nn_eta=1.0, dtype=dtype)
model.apply(lambda x: init_weights_to_zero(x, std=2e-2))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

model_names = {
    fTNModel: 'fTN',
    fTN_backflow_attn_Tensorwise_Model: 'fTN_backflow_attn_Tensorwise',
    fTN_backflow_attn_Tensorwise_Model_v1: 'fTN_backflow_attn_Tensorwise_v1',
    
}
model_name = model_names.get(type(model), 'UnknownModel')

init_step = 0
total_steps = 200
if init_step != 0:
    saved_model_params = torch.load(f'../../data/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except:
        model.load_params(saved_model_params_vec)

# optimizer = SignedSGD(learning_rate=0.05)
optimizer = SGD(learning_rate=0.05)
sampler = MetropolisExchangeSamplerSpinful(H.hilbert, graph, N_samples=N_samples, burn_in_steps=2, reset_chain=False, random_edge=False, equal_partition=True, dtype=dtype)
# sampler = None
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, diag_eta=0.05, iter_step=1e5, dtype=dtype)
# preconditioner = TrivialPreconditioner()
vmc = VMC(H, variational_state=variational_state, optimizer=optimizer, preconditioner=preconditioner)


import jax
import pyinstrument
random_config = H.hilbert.random_state(key=jax.random.PRNGKey(1))
random_config_np = random_config.copy()
random_config = torch.tensor(random_config, dtype=dtype)
amp = peps.get_amp(random_config)
amp_np = peps_np.get_amp(random_config_np)
# model(random_config), amp.contract(), amp_np.contract()

amp_np = peps_np.get_amp(random_config_np, contract=True)
# with pyinstrument.Profiler() as prof:
#     # amp.contract()
#     # amp_np.contract()
#     # amp_np = amp_np.contract_boundary_from_ymin(max_bond=chi, cutoff=0.0, yrange=[0, amp_np.Ly//2-1])
#     # amp_np = amp_np.contract_boundary_from_ymax(max_bond=chi, cutoff=0.0, yrange=[amp_np.Ly//2, amp_np.Ly-1])
#     amp_val = amp_np.contract()
# prof.print()
with pyinstrument.Profiler() as prof:
    model(random_config)
prof.print()