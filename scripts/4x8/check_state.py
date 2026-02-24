import os
import sys
from mpi4py import MPI
import pickle
# torch
import torch

# quimb
import quimb.tensor as qtn
import autoray as ar

from vmc_torch.experiment.tn_model import fTNModel_reuse, fTNModel
from vmc_torch.experiment.tn_model import fTN_BFA_cluster_Model_reuse
from vmc_torch.experiment.tn_model import init_weights_to_zero, init_weights_uniform
from vmc_torch.fermion_utils import generate_random_fpeps_symmray, generate_random_fpeps
from vmc_torch.sampler import MetropolisExchangeSamplerSpinful_hopping, MetropolisExchangeSamplerSpinful_2D_reusable
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SR, DecayScheduler, SignedRandomSGD, TrivialPreconditioner
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.torch_utils import SVD,QR_tao
from vmc_torch.global_var import DEBUG
from vmc_torch.utils import closest_divisible
from symmray.fermionic_local_operators import FermionicOperator
# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR_tao.apply)
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'



COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Hamiltonian parameters
Lx = int(4)
Ly = int(8)
symmetry = 'U1SU'
t = 1.0
U = 8.0
N_f = int(Lx*Ly-4)
n_fermions_per_spin = (N_f//2, N_f//2)
H = spinful_Fermi_Hubbard_square_lattice_torch(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph
# TN parameters
D = int(4)
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
skeleton = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_skeleton{appendix}.pkl", "rb"))
peps_params = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_su_params{appendix}.pkl", "rb"))
peps = qtn.unpack(peps_params, skeleton)
# peps = generate_random_fpeps_symmray(Lx, Ly, D, symmetry='Z2', Nf=N_f, subsizes='equal', seed=42)
# Precondition the fPEPS
## 1. Sync the stored fermionic phases
for ts in peps.tensors:
    ts.data.phase_sync(inplace=True)
## 2. Scale the tensor elements
scale = 4.0
peps.apply_to_arrays(lambda x: torch.tensor(scale*x, dtype=dtype))
## 3. Set the exponent to 0.0
peps.exponent = 0.0
# correct the format of oddpos
for ts in peps.tensors:
    ts.data.phase_sync(inplace=True)
    # for Z2 fPEPS converted from U1 fPEPS, need to correct the format of oddpos
    if ts.data.oddpos:
        oddpos = ts.data.oddpos
        if isinstance(oddpos[0].label, tuple):
            nested_oddpos = oddpos[0].label[0]
            if isinstance(nested_oddpos, FermionicOperator):
                ts.data._oddpos = (nested_oddpos,)


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
#         contraction_kwargs=contraction_kwargs,
#         grad_contraction_kwargs=grad_contraction_kwargs,
# )


# model = fTNModel_reuse(
#         peps,
#         max_bond=chi,
# )
model = fTNModel(
    peps,
    max_bond=chi,
    contraction_kwargs=contraction_kwargs,
    grad_contraction_kwargs=grad_contraction_kwargs,
)

model1 = fTNModel(
        peps,
        max_bond=-1,
)

init_std = float(model.get_tn_params_vec().abs().std().item()) * 0.1
if RANK == 0:
    print(init_std)
# init_std = 5e-3
# model.apply(lambda x: init_weights_to_zero(x, std=init_std))
model.apply(lambda x: init_weights_uniform(x, a=-1*init_std, b=init_std))

model_names = {
    fTNModel_reuse: f'fTN_reuse{appendix}',
    fTN_BFA_cluster_Model_reuse: f'fTNNN_r={radius}{appendix}',
}
model_name = model_names.get(type(model), 'UnknownModel')

init_step = int(0)
final_step = int(1e2)
total_steps = final_step - init_step
if init_step != 0:
    saved_model_params = torch.load(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth', weights_only=False)
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except Exception:
        model.load_params(saved_model_params_vec)

# Set up optimizer and scheduler
learning_rate = float(5e-3)
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-4)
optimizer = SGD(learning_rate=learning_rate)
# optimizer = SignedRandomSGD(learning_rate=learning_rate)
preconditioner = SR(dense=False, exact=False, use_MPI4Solver=True, solver='minres', diag_eta=1e-3, iter_step=5e2, dtype=dtype, rtol=5e-5)
# preconditioner = TrivialPreconditioner()
# sampler = MetropolisExchangeSamplerSpinful_hopping(H.hilbert, graph, N_samples=N_samples, burn_in_steps=1, reset_chain=False, random_edge=False, equal_partition=True, dtype=dtype, hopping_rate=0.25)
sampler = MetropolisExchangeSamplerSpinful_2D_reusable(H.hilbert, graph, N_samples=N_samples, burn_in_steps=1, reset_chain=False, random_edge=False, equal_partition=True, dtype=dtype, subchain_length=10)
if N_f == int(Lx*Ly):
    sampler.current_config = torch.tensor([1,2,1,2,1,2,1,2,
                                        2,1,2,1,2,1,2,1,
                                        1,2,1,2,1,2,1,2,
                                        2,1,2,1,2,1,2,1,
                                        1,2,1,2,1,2,1,2,
                                        2,1,2,1,2,1,2,1,
                                        1,2,1,2,1,2,1,2,
                                        2,1,2,1,2,1,2,1])[:Lx*Ly]
else:
    half_filled_config = torch.tensor([1,2,1,2,1,2,1,2,
                                      2,1,2,1,2,1,2,1,
                                      1,2,1,2,1,2,1,2,
                                      2,1,2,1,2,1,2,1,
                                      1,2,1,2,1,2,1,2,
                                      2,1,2,1,2,1,2,1,
                                      1,2,1,2,1,2,1,2,
                                      2,1,2,1,2,1,2,1])[:Lx*Ly]
    # set first (Lx*Ly - N_f) sites to be empty (0)
    empty_sites = list(range(Lx*Ly-N_f))
    doped_config = half_filled_config.clone()
    doped_config[empty_sites] = 0
    # randomly permute the doped_config
    perm = torch.randperm(Lx*Ly)
    doped_config = doped_config[perm]
    sampler.current_config = doped_config
    num_1 = torch.sum(sampler.current_config == 1).item()
    num_2 = torch.sum(sampler.current_config == 2).item()
    assert num_1 == N_f//2 and num_2 == N_f//2, f"Number of spin up and spin down fermions should be {N_f//2}, but got {num_1} and {num_2}"

variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)

# Set up VMC
vmc = VMC(hamiltonian=H, variational_state=variational_state, optimizer=optimizer, preconditioner=preconditioner, scheduler=scheduler)
random_x = sampler.current_config

variational_state1 = Variational_State(model1, hi=H.hilbert, sampler=sampler, dtype=dtype)
print('Benchmark grad max and amp:')
print(variational_state1.amplitude_grad(random_x)[1].abs().max())
print(model1(random_x))

print('Test grad max and amp:')
model.cache_env_mode = True
print(variational_state.amplitude_grad(random_x)[1].abs().max())
model.cache_env_mode = False
print(model(random_x))

# if __name__ == "__main__":
#     torch.autograd.set_detect_anomaly(False)
#     os.makedirs(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/', exist_ok=True)
#     record_file = open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/record{init_step}.txt', 'w')
#     if RANK == 0:
#         # print training information
#         print(f"Running VMC for {model_name}")
#         print(f'model params: {variational_state.num_params}')
#         print(f"Optimizer: {optimizer}")
#         print(f"Preconditioner: {preconditioner}")
#         print(f"Scheduler: {scheduler}")
#         print(f"Sampler: {sampler}")
#         print(f'2D Fermi-Hubbard model on {Lx}x{Ly} lattice with {N_f} fermions, Sz=0, t={t}, U={U}')
#         print(f"Running {total_steps} steps from {init_step} to {final_step}")
#         print(f'Model initialized with mean=0, std={init_std}')
#         print(f'Learning rate: {learning_rate}')
#         print(f'Sample size: {N_samples}')
#         print(f'fPEPS bond dimension: {D}, max bond: {chi}')
#         print(f'fPEPS symmetry: {symmetry}\n')
#         try:
#             print(f'model structure: {model.model_structure}')
#         except:
#             pass
#         # sys.stdout = record_file
    
#     if RANK == 0:
#         # print training information
#         print(f"Running VMC for {model_name}")
#         print(f'model params: {variational_state.num_params}')
#         print(f"Optimizer: {optimizer}")
#         print(f"Preconditioner: {preconditioner}")
#         print(f"Scheduler: {scheduler}")
#         print(f"Sampler: {sampler}")
#         print(f'2D Fermi-Hubbard model on {Lx}x{Ly} lattice with {N_f} fermions, Sz=0, t={t}, U={U}')
#         print(f"Running {total_steps} steps from {init_step} to {final_step}")
#         print(f'Model initialized with mean=0, std={init_std}')
#         print(f'Learning rate: {learning_rate}')
#         print(f'Sample size: {N_samples}')
#         print(f'fPEPS bond dimension: {D}, max bond: {chi}')
#         print(f'fPEPS symmetry: {symmetry}\n')
#         try:
#             print(f'model structure: {model.model_structure}')
#         except:
#             pass
#     COMM.barrier()
#     vmc.run(init_step, init_step+total_steps, tmpdir=pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/', save=False)

