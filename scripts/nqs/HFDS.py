import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
from mpi4py import MPI
import pickle

# torch
import torch
import numpy as np

# quimb
import quimb.tensor as qtn
import autoray as ar

from vmc_torch.experiment.tn_model import NNBF,NNBF_attention, HFDS, SlaterDeterminant
from vmc_torch.experiment.tn_model import init_weights_to_zero, init_weights_uniform
from vmc_torch.sampler import MetropolisExchangeSamplerSpinful, MetropolisMPSSamplerSpinful
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SR, DecayScheduler
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.torch_utils import SVD,QR

from vmc_torch.utils import closest_divisible

# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'

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
H = spinful_Fermi_Hubbard_square_lattice_torch(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph
# TN parameters
D = 4
chi = -1
dtype=torch.float64

orbital_matrix_numpy = np.load(f"Hubbard{Lx}x{Ly}_{N_f}.npy")
orbital_matrix = torch.tensor(orbital_matrix_numpy)
init_std = 5e-3
def kernel_init_from_matrix(tensor):
    if tensor.shape != orbital_matrix.shape:
        # slice part of the tensor to match with orbital_matrix shape, the rest is randomly initialized
        assert tensor.shape[0] == orbital_matrix.shape[0]
        tensor_slice = tensor[:,:orbital_matrix.shape[1]]
        tensor_rest = tensor[:,orbital_matrix.shape[1]:]
        tensor_slice.copy_(orbital_matrix) # In-place copy
        tensor_rest.normal_(mean=0.0, std=init_std)
        return tensor
        
    # In-place copy
    tensor.copy_(orbital_matrix) 
    return tensor

# VMC sample size
N_samples = int(5e3)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

# model = NNBF(H.hilbert)
# model = NNBF_attention(
#     nsite=Lx*Ly,
#     hilbert=H.hilbert,
#     param_dtype=dtype,
#     hidden_dim=16,
#     attention_heads=4,
#     nn_eta=0.0,
#     phys_dim=4
# )
model = HFDS(
    hilbert=H.hilbert,
    kernel_init=kernel_init_from_matrix,
    param_dtype=dtype,
    num_hidden_fermions=8,
    hidden_dim=32,
)
# model = SlaterDeterminant(
#     hilbert=H.hilbert,
#     kernel_init=kernel_init_from_matrix,
#     param_dtype=dtype,
# )

# model.apply(lambda x: init_weights_to_zero(x, std=init_std))
# model.apply(lambda x: init_weights_uniform(x, a=-1*init_std, b=init_std))

model_names = {
    NNBF: 'NNBF',
    NNBF_attention: 'NNBF_attention',
    HFDS: 'HFDS',
    SlaterDeterminant: 'SlaterDeterminant'
}
model_name = model_names.get(type(model), 'UnknownModel')

init_step = 0
final_step = 1000
total_steps = final_step - init_step
if init_step != 0:
    saved_model_params = torch.load(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth', weights_only=False)
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except Exception:
        model.load_params(saved_model_params_vec)

# optimizer = SignedSGD(learning_rate=0.05)
# Set up optimizer and scheduler
learning_rate = float(0.05)
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-4)
optimizer = SGD(learning_rate=learning_rate)
mps_dir = '/pscratch/sd/s/sijingdu/VMC/fermion/data'+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/tmp'
sampler = MetropolisExchangeSamplerSpinful(H.hilbert, graph, N_samples=N_samples, burn_in_steps=1, reset_chain=False, random_edge=False, equal_partition=True, dtype=dtype)
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
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, solver='minres', diag_eta=1e-4, iter_step=5e2, dtype=dtype, rtol=1e-4)
# preconditioner = TrivialPreconditioner()
# Set up VMC
vmc = VMC(hamiltonian=H, variational_state=variational_state, optimizer=optimizer, preconditioner=preconditioner, scheduler=scheduler)

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
        except Exception:
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
            print(f'model structure: {model.model_structure}')
        except Exception:
            pass
    COMM.barrier()
    vmc.run(init_step, init_step+total_steps, tmpdir=pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/')

