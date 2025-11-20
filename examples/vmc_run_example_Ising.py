import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
from mpi4py import MPI
import pickle


# torch
import torch

# quimb
import quimb.tensor as qtn
import autoray as ar

from vmc_torch.experiment.tn_model import PEPS_model, PEPS_NN_Model, init_weights_to_zero, PEPS_NNproj_Model, PEPS_delocalized_Model
from vmc_torch.sampler import MetropolisSamplerSpinless
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SR
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian_torch import spin_transverse_Ising_square_lattice_torch
from vmc_torch.torch_utils import SVD,QR

# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

"""Define the Hamiltonian"""
Lx = int(4)
Ly = int(2)
J = 1.0
h = 0.5
H = spin_transverse_Ising_square_lattice_torch(Lx, Ly, J, -h, pbc=False, total_sz=None) 
graph = H.graph

"""TNS parameters"""
D = 2
chi = 4*D
dtype=torch.float64

"""Load Simple Update PEPS as the initial state"""
skeleton = pickle.load(open(f"./example_data/{Lx}x{Ly}/J={J}_h={h}/D={D}/peps_skeleton.pkl", "rb"))
peps_params = pickle.load(open(f"./example_data/{Lx}x{Ly}/J={J}_h={h}/D={D}/peps_su_params.pkl", "rb"))
peps = qtn.unpack(peps_params, skeleton)
peps.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype))


"""Choose the torch model (you can create your own model of course)"""
model = PEPS_model(peps, max_bond=chi)
# model = PEPS_delocalized_Model(peps, max_bond=chi, diag=False)
# model = PEPS_NN_Model(peps, max_bond=chi_nn, nn_eta=1.0, nn_hidden_dim=Lx*Ly)
# model = PEPS_NNproj_Model(peps, max_bond=chi_nn, nn_eta=1.0, nn_hidden_dim=Lx*Ly)
model_names = {
    PEPS_model: 'PEPS',
    PEPS_delocalized_Model: 'PEPS_delocalized_diag='+str(model.diag) if isinstance(model, PEPS_delocalized_Model) else None,
    PEPS_NN_Model: 'PEPS_NN',
    PEPS_NNproj_Model: 'PEPS_NNproj'
}
model_name = model_names.get(type(model), 'UnknownModel')


"""Choose the initialization method for NN parameters (optional)"""
model.apply(init_weights_to_zero)

"""Potentially load a pre-trained model"""
init_step = 0
final_step = 50
total_steps = final_step - init_step
if init_step != 0:
    saved_model_params = torch.load(f'./example_data/{Lx}x{Ly}/J={J}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except Exception:
        model.load_params(saved_model_params_vec)

"""Set VMC sample size"""
N_samples = int(5e3)
N_samples = N_samples - N_samples % SIZE + SIZE - 1

"""Choose the sampler""" 
sampler = MetropolisSamplerSpinless(H.hilbert, graph, N_samples=N_samples, burn_in_steps=20, reset_chain=True, random_site=False, equal_partition=False, dtype=dtype)

"""Choose the optimizer and preconditioner"""
optimizer = SGD(learning_rate=1e-2)
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, diag_eta=1e-4, rtol=1e-5, iter_step=1e3, dtype=dtype)

"""Create the variational state object"""
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)

"""Create the VMC object"""
vmc = VMC(hamiltonian=H, variational_state=variational_state, optimizer=optimizer, preconditioner=preconditioner)

if __name__ == "__main__":
    # psi = []
    # for c in H.hilbert.all_states():
    #     amp = variational_state.amplitude(torch.tensor(c, dtype=dtype).unsqueeze(0))
    #     psi.append(amp.item())
    # H_dense = torch.tensor(H.to_dense(), dtype=dtype)
    # print(H_dense)
    # psi = torch.tensor(psi, dtype=dtype)
    # print(psi)
    # E_var = (psi @ (H_dense @ psi)) / (psi @ psi)
    # print(f"Initial variational energy: {E_var.item()/(Lx*Ly)}")
    # import scipy
    # E_gs = scipy.linalg.eigh(H.to_dense())[0][0]
    # print(f"Exact ground state energy per site: {E_gs/(Lx*Ly)}")
    os.makedirs(f'./example_data/{Lx}x{Ly}/J={J}/D={D}/{model_name}/chi={chi}/', exist_ok=True)
    vmc.run(init_step, init_step+total_steps, tmpdir=f'./example_data/{Lx}x{Ly}/J={J}/D={D}/{model_name}/chi={chi}/', save_every=25)