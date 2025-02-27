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

from vmc_torch.experiment.tn_model import fMPSModel, fMPS_backflow_Model, fMPS_backflow_attn_Tensorwise_Model_v1
from vmc_torch.experiment.tn_model import init_weights_to_zero
from vmc_torch.sampler import MetropolisExchangeSamplerSpinful
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SR, Adam, SGD_momentum, DecayScheduler
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian import spinful_Fermi_Hubbard_chain, spinful_Fermi_Hubbard_chain_quimb
from vmc_torch.torch_utils import SVD,QR
from vmc_torch.fermion_utils import generate_random_fmps, form_gated_fmps_tnf

# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)

from vmc_torch.global_var import DEBUG
from vmc_torch.utils import closest_divisible
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Hamiltonian parameters
L = int(10)
symmetry = 'Z2'
t = 1.0
U = 8.0
N_f = int(L-2)
n_fermions_per_spin = (N_f//2, N_f//2)
H = spinful_Fermi_Hubbard_chain(L, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
quimb_ham = spinful_Fermi_Hubbard_chain_quimb(L, t, U, mu=0.0, pbc=False, symmetry=symmetry)
graph = H.graph
# TN parameters
D = 4
chi = -1
dtype=torch.float64

# Load mps
skeleton = pickle.load(open(pwd+f"/L={L}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/mps_skeleton.pkl", "rb"))
mps_params = pickle.load(open(pwd+f"/L={L}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/mps_su_params.pkl", "rb"))
mps = qtn.unpack(mps_params, skeleton)
# fmps_tnf = form_gated_fmps_tnf(fmps=mps, ham=quimb_ham, depth=2)
mps.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype))

# # randomize the mps tensors
# mps.apply_to_arrays(lambda x: torch.randn_like(torch.tensor(x, dtype=dtype), dtype=dtype))

# VMC sample size
N_samples = int(15000)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

# model = fMPSModel(mps, dtype=dtype)
model = fMPS_backflow_attn_Tensorwise_Model_v1(mps, embedding_dim=16, attention_heads=4, nn_final_dim=int(L/2), nn_eta=1.0, dtype=dtype)
# model = fMPS_backflow_Model(mps, nn_eta=1.0, num_hidden_layer=2, nn_hidden_dim=2*L, dtype=dtype)
init_std = 5e-2
seed = 2
torch.manual_seed(seed)
model.apply(lambda x: init_weights_to_zero(x, std=init_std))
# model.apply(lambda x: init_weights_kaiming(x))

model_names = {
    fMPSModel: 'fMPS',
    fMPS_backflow_Model: 'fMPS_backflow',
    fMPS_backflow_attn_Tensorwise_Model_v1: 'fMPS_backflow_attn_Tensorwise_v1'
}
model_name = model_names.get(type(model), 'UnknownModel')


init_step = 63
final_step = 250
total_steps = final_step - init_step
# Load model parameters
if init_step != 0:
    saved_model_params = torch.load(pwd+f'/L={L}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except:
        model.load_params(saved_model_params_vec)
    optimizer_state = saved_model_params.get('optimizer_state', None)

# Set up optimizer and scheduler
learning_rate = 1e-1
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-2)
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
sampler = MetropolisExchangeSamplerSpinful(H.hilbert, graph, N_samples=N_samples, burn_in_steps=40, reset_chain=False, random_edge=False, equal_partition=False, dtype=dtype)
# Set up variational state
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
# Set up SR preconditioner
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, diag_eta=1e-3, iter_step=1e5, dtype=dtype)
# preconditioner = TrivialPreconditioner()
# Set up VMC
vmc = VMC(hamiltonian=H, variational_state=variational_state, optimizer=optimizer, preconditioner=preconditioner, scheduler=scheduler, SWO=False, beta=0.01)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(False)
    os.makedirs(pwd+f'/L={L}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/', exist_ok=True)
    # with pyinstrument.Profiler() as prof:
    vmc.run(init_step, init_step+total_steps, tmpdir=pwd+f'/L={L}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/')
    # if RANK == 0:
    #     prof.print()

