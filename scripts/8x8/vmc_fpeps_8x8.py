from mpi4py import MPI

# torch
import torch

# quimb
import autoray as ar

from vmc_torch.experiment.tn_model import fTNModel_reuse, fTNModel
from vmc_torch.experiment.tn_model import fTN_BFA_cluster_Model_reuse
from vmc_torch.experiment.tn_model import init_weights_to_zero, init_weights_uniform
from vmc_torch.sampler import MetropolisExchangeSamplerSpinful_hopping, MetropolisExchangeSamplerSpinful_2D_reusable
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SR, DecayScheduler
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.torch_utils import SVD,QR_tao,QR_tao_direct,QR
from vmc_torch.fermion_utils import unpack_ftns
from vmc_torch.utils import closest_divisible
import time
# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR_tao_direct.apply)
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'



COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Hamiltonian parameters
Lx = int(8)
Ly = int(8)
symmetry = 'U1SU'
t = 1.0
U = 8.0
N_f = int(Lx*Ly-8)
n_fermions_per_spin = (N_f//2, N_f//2)
H = spinful_Fermi_Hubbard_square_lattice_torch(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph
# TN parameters
D = int(6)
chi = int(4*D)
dtype=torch.float64

# Load PEPS
appendix = ''
if symmetry == 'U1_Z2':
    symmetry = 'Z2'
    appendix = '_U1'
elif symmetry == 'U1SU':
    symmetry = 'Z2'
    appendix = '_U1SU'

gopeps_step = 0
gopeps_appendix = '' if gopeps_step<=0 else f'_gopeps{gopeps_step}'
if gopeps_step > 0:
    skeleton_path = pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/GOfPEPS{appendix}_step{gopeps_step}.pth"
    class CustomPickleModule:
        from vmc_torch.fermion_utils import CustomUnpickler
        Unpickler = CustomUnpickler
    saved_gopeps = torch.load(skeleton_path, weights_only=False, pickle_module=CustomPickleModule)
    goskeleton = saved_gopeps['model_skeleton']
    goparams = saved_gopeps['model_params']
    peps = unpack_ftns(params=goparams, skeleton=goskeleton, scale=1.0, dtype=dtype, new_symmray_format=True)
else:
    skeleton_path = pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_skeleton{appendix}.pkl"
    params_path = pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_su_params{appendix}.pkl"
    peps = unpack_ftns(params_path, skeleton_path, scale=4.0, dtype=dtype, new_symmray_format=True)

# VMC sample size
N_samples = int(5e3)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

radius = 1
grad_contraction_kwargs = {
    'mode': 'fit',
    'tol': 1e-5,
    'tn_fit': "zipup",
    'bsz': 2,
    'max_iterations': 5,
}
contraction_kwargs = {
    'mode': 'mps'
}

model = fTNModel(
    peps,
    max_bond=chi,
    contraction_kwargs=contraction_kwargs,
    # grad_contraction_kwargs=grad_contraction_kwargs,
)

# model = fTN_BFA_cluster_Model_reuse(
#         peps,
#         max_bond=chi,
#         embedding_dim=16,
#         attention_heads=4,
#         nn_final_dim=D,
#         nn_eta=1.0,
#         radius=radius,
#         jastrow=False,
#         dtype=dtype,
#         # contraction_kwargs=grad_contraction_kwargs,
#         contraction_kwargs=contraction_kwargs,
#         # grad_contraction_kwargs=grad_contraction_kwargs,
# )

# load benchmark model
model1 = fTNModel_reuse(
        peps,
        max_bond=chi,
)


init_std = 5e-3
if RANK == 0:
    print(f'fPEPS paramter mean: {model.get_tn_params_vec().abs().mean().item()}')
    print(f'fPEPS parameter median: {model.get_tn_params_vec().abs().median().item()}')
    print(f'fPEPS parameter std: {model.get_tn_params_vec().abs().std().item()}')
    print(f'fPEPS parameter min: {model.get_tn_params_vec().abs().min().item()}')
    print(f'fPEPS parameter max: {model.get_tn_params_vec().abs().max().item()}')

    print(f'NN params init std: {init_std}\n')
# init_std = 5e-3
model.apply(lambda x: init_weights_to_zero(x, std=init_std))
# model.apply(lambda x: init_weights_uniform(x, a=-1*init_std, b=init_std))

model_names = {
    fTNModel_reuse: f'fTN_reuse{appendix}',
    fTN_BFA_cluster_Model_reuse: f'fTNNN_r={radius}{appendix}',
}
model_name = model_names.get(type(model), 'UnknownModel')

# Set up optimizer and scheduler
learning_rate = float(5e-3)
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-4)
optimizer = SGD(learning_rate=learning_rate)
# optimizer = SignedRandomSGD(learning_rate=learning_rate)
preconditioner = SR(dense=False, exact=False, use_MPI4Solver=True, solver='minres', diag_eta=1e-3, iter_step=5e2, dtype=dtype, rtol=5e-5)
# preconditioner = TrivialPreconditioner()
sampler = MetropolisExchangeSamplerSpinful_hopping(H.hilbert, graph, N_samples=N_samples, burn_in_steps=1, reset_chain=False, random_edge=False, equal_partition=True, dtype=dtype, hopping_rate=0.25)
sampler_reuse = MetropolisExchangeSamplerSpinful_2D_reusable(H.hilbert, graph, N_samples=N_samples, burn_in_steps=1, reset_chain=False, random_edge=False, equal_partition=True, dtype=dtype, subchain_length=10)

# Set up VMC
random_x = torch.tensor([2., 1., 2., 1., 2., 1., 2., 1., 1., 1., 1., 2., 1., 2., 2., 1., 2.,
       2., 0., 2., 0., 1., 0., 0., 2., 0., 2., 1., 2., 0., 2., 1., 1., 0.,
       1., 2., 1., 1., 1., 2., 2., 1., 2., 1., 2., 0., 2., 1., 1., 3., 2.,
       3., 1., 2., 1., 2., 2., 1., 0., 2., 0., 1., 2., 1.])
# random_x = torch.tensor(H.hilbert.random_state(), dtype=dtype)

variational_state1 = Variational_State(model1, hi=H.hilbert, sampler=sampler_reuse, dtype=dtype)
model1.cache_env_mode = True
print('\nBenchmark grad max and amp:')
print('Max grad element and corresponding param:')
print(variational_state1.amplitude_grad(random_x)[1].abs().max())
t0 = time.time()
print(f'Amp:{model1(random_x)}')
t1 = time.time()
print('delta T=', t1-t0)

print('\nTest model grad max and amp:')
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
print('Max grad element and corresponding param:')
print(variational_state.amplitude_grad(random_x)[1].abs().max())
t0 = time.time()
print(f'Amp:{model(random_x)}')
t1 = time.time()
print('delta T=', t1-t0)

# get the index of the maximum gradient, and print the corresponding model parameter
# max_grad_idx = torch.argmax(variational_state.amplitude_grad(random_x)[1].abs())
# print(model.from_params_to_vec()[max_grad_idx], 1/model.from_params_to_vec()[max_grad_idx])

# get the index of the maximum gradient, and print the corresponding model parameter
# max_grad_idx = torch.argmax(variational_state1.amplitude_grad(random_x)[1].abs())
# print(model1.from_params_to_vec()[max_grad_idx], 1/model1.from_params_to_vec()[max_grad_idx])


# print('\nTest GO-fPEPS grad max and amp:')
# variational_state3 = Variational_State(model3, hi=H.hilbert, sampler=sampler, dtype=dtype)
# model3.cache_env_mode = True
# print('Max grad element and corresponding param:')
# print(variational_state3.amplitude_grad(random_x)[1].abs().max())
# print(f'Amp:{model3(random_x)}')
# # get the index of the maximum gradient, and print the corresponding model parameter
# # max_grad_idx = torch.argmax(variational_state3.amplitude_grad(random_x)[1].abs())
# # print(model3.from_params_to_vec()[max_grad_idx], 1/model3.from_params_to_vec()[max_grad_idx])

