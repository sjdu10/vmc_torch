import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
from mpi4py import MPI
import numpy as np

# torch
import torch

# quimb
import autoray as ar

from vmc_torch.experiment.tn_model import NNBF,NNBF_attention, NNBF_attention_Nsd, HFDS, init_weights_to_zero, SlaterDeterminant, init_weights_uniform, NNBF_attention_Nsd_v1
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.fermion_utils import calc_phase_netket, from_quimb_config_to_netket_config, from_netket_config_to_quimb_config
from vmc_torch.sampler import MetropolisExchangeSamplerSpinful, MetropolisExchangeSamplerSpinful_hopping
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SR, DecayScheduler
from vmc_torch.VMC import VMC
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

# Customized Hamiltonian elements to match with Ao's initial SD's definition
class spinful_Fermi_Hubbard_square_lattice_torch_Ao(spinful_Fermi_Hubbard_square_lattice_torch):
    def __init__(self, Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=None):
        super().__init__(Lx, Ly, t, U, N_f, pbc=pbc, n_fermions_per_spin=n_fermions_per_spin)

    def get_conn(self, sigma_quimb):
        """
        Return the connected configurations <eta| by the Hamiltonian to the state |sigma>,
        and their corresponding coefficients <eta|H|sigma>.
        """
        sigma = from_quimb_config_to_netket_config(sigma_quimb)
        connected_config_coeff = dict()
        for key, value in self._H.items():
            if len(key) == 3:
                # hopping term
                i0, j0, spin = key
                i = i0 if spin == 1 else i0 + self.hilbert.n_orbitals // 2
                j = j0 if spin == 1 else j0 + self.hilbert.n_orbitals // 2
                # Check if the two sites are different
                if sigma[i] != sigma[j]:
                    # H|sigma> = -t * |eta>
                    eta = sigma.copy()
                    eta[i], eta[j] = sigma[j], sigma[i]
                    eta_quimb0 = from_netket_config_to_quimb_config(eta)
                    eta_quimb = tuple(eta_quimb0)
                    # Calculate the phase correction
                    phase = calc_phase_netket(from_netket_config_to_quimb_config(sigma), eta_quimb0)
                    if eta_quimb not in connected_config_coeff:
                        connected_config_coeff[eta_quimb] = value*phase
                    else:
                        connected_config_coeff[eta_quimb] += value*phase
            elif len(key) == 1:
                # on-site term
                i = key[0]
                if sigma_quimb[i] == 3:
                    eta_quimb = tuple(sigma_quimb)
                    if eta_quimb not in connected_config_coeff:
                        connected_config_coeff[eta_quimb] = value
                    else:
                        connected_config_coeff[eta_quimb] += value
        
        return ar.do('array', list(connected_config_coeff.keys())), ar.do('array', list(connected_config_coeff.values()))
    
H = spinful_Fermi_Hubbard_square_lattice_torch_Ao(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)

graph = H.graph
# TN parameters
D = 4
chi = -2
dtype=torch.float64


# VMC sample size
N_samples = int(2e3)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

orbital_matrix_numpy = np.load(f"Hubbard4x2_{N_f}.npy")
orbital_matrix = torch.tensor(orbital_matrix_numpy)

def kernel_init_from_matrix(tensor):
    if tensor.shape != orbital_matrix.shape:
        raise ValueError(f"Expected shape {orbital_matrix.shape}, got {tensor.shape}")
    # In-place copy
    tensor.copy_(orbital_matrix) 
    return tensor

# model = NNBF(
#     hilbert=H.hilbert,
#     kernel_init=kernel_init_from_matrix,
#     param_dtype=dtype,
#     hidden_dim=32,
#     nn_eta=1.0,
# )
# embed_dim=8
# attention_heads=2
# hidden_dim=embed_dim*Lx*Ly
# model = NNBF_attention(
#     nsite=Lx*Ly,
#     hilbert=H.hilbert,
#     kernel_init=kernel_init_from_matrix,
#     param_dtype=dtype,
#     hidden_dim=hidden_dim,
#     embed_dim=embed_dim,
#     attention_heads=attention_heads,
#     nn_eta=1.0,
#     phys_dim=4,
# )

embed_dim=6
attention_heads=2
hidden_dim=embed_dim*Lx*Ly
model = NNBF_attention_Nsd(
    nsite=Lx*Ly,
    hilbert=H.hilbert,
    kernel_init=kernel_init_from_matrix,
    param_dtype=dtype,
    hidden_dim=hidden_dim,
    embed_dim=embed_dim,
    attention_heads=attention_heads,
    attention_layers=2,
    position_wise_mlp_hidden_dim=embed_dim,
    nn_eta=1.0,
    phys_dim=4,
    Nsd=1
)

# embed_dim=16
# attention_heads=2
# hidden_dim=16
# model = NNBF_attention_Nsd_v1(
#     nsite=Lx*Ly,
#     hilbert=H.hilbert,
#     kernel_init=kernel_init_from_matrix,
#     param_dtype=dtype,
#     hidden_dim=hidden_dim,
#     embed_dim=embed_dim,
#     attention_heads=attention_heads,
#     attention_layers=2,
#     position_wise_mlp_hidden_dim=embed_dim,
#     nn_eta=1.0,
#     phys_dim=4,
#     Nsd=1,
#     eps=1e-3,
# )

init_std = 1e-3
# model.apply(lambda x: init_weights_to_zero(x, std=init_std))


model_names = {
    NNBF: 'NNBF',
    NNBF_attention: 'NNBF_attention',
    NNBF_attention_Nsd: f'NNBF_attention_Nsd={model.Nsd if hasattr(model, "Nsd") else ""}',
    NNBF_attention_Nsd_v1: f'NNBF_attention_Nsd_v1={model.Nsd if hasattr(model, "Nsd") else ""}',
    HFDS: 'HFDS',
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
learning_rate = float(5e-2)
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-4)
optimizer = SGD(learning_rate=learning_rate)
# sampler = MetropolisExchangeSamplerSpinful(H.hilbert, graph, N_samples=N_samples, burn_in_steps=20, reset_chain=False, random_edge=False, equal_partition=False, dtype=dtype)
sampler = MetropolisExchangeSamplerSpinful_hopping(H.hilbert, graph, N_samples=N_samples, burn_in_steps=20, reset_chain=False, random_edge=False, equal_partition=False, dtype=dtype, hopping_rate=0.25)
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
preconditioner = SR(
    dense=False,
    exact=True if sampler is None else False,
    use_MPI4Solver=True,
    solver="minres",
    diag_eta=1e-4,
    iter_step=5e2,
    dtype=dtype,
    rtol=1e-5,
)  # rtol=1e-5, diag_eta=1e-4 NOTE: the rtol effects the optimization accuracy significantly, diageta=1e-3 is usually fine

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
        print(f'SR diag_shift: {preconditioner.diag_eta}, rtol: {preconditioner.rtol}')

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
        print(f'SR diag_shift: {preconditioner.diag_eta}, rtol: {preconditioner.rtol}')

        try:
            print(f'model structure: {model.model_structure}')
        except Exception:
            pass
    COMM.barrier()
    vmc.run(init_step, init_step+total_steps, tmpdir=pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/')
