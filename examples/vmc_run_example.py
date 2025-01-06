from mpi4py import MPI
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# torch
import torch
torch.autograd.set_detect_anomaly(False)
dtype=torch.float64

# autoray
import autoray as ar

from vmc_torch.model import SlaterDeterminant, NeuralBackflow, FFNN, NeuralJastrow
from vmc_torch.model import init_weights_xavier, init_weights_kaiming, init_weights_to_zero
from vmc_torch.sampler import MetropolisExchangeSampler
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import TrivialPreconditioner, SignedSGD, SGD, SR
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian import spinless_Fermi_Hubbard_square_lattice
from vmc_torch.torch_utils import SVD,QR
from vmc_torch.global_var import DEBUG

# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

"""Define the Hamiltonian"""
Lx = int(4)
Ly = int(4)
symmetry = 'Z2'
t = 1.0
V = 1.0
N_f = int(Lx*Ly/2)-2
H= spinless_Fermi_Hubbard_square_lattice(Lx, Ly, t, V, N_f)
hi = H.hi
graph = H.graph

"""Choose the torch model (you can create your own model of course)"""
model = SlaterDeterminant(hi)
# model = NeuralBackflow(hi, param_dtype=dtype, hidden_dim=hi.size)
# model = NeuralJastrow(hi, param_dtype=dtype, hidden_dim=hi.size)
# model = FFNN(hi, hidden_dim=2*hi.size)
model_names = {
    SlaterDeterminant: 'SlaterDeterminant',
    NeuralBackflow: 'NeuralBackflow',
    FFNN: 'FFNN',
    NeuralJastrow: 'NeuralJastrow',
}
model_name = model_names.get(type(model), 'UnknownModel')


"""Choose the initialization method for NN parameters"""
# model.apply(init_weights_to_zero)
model.apply(init_weights_xavier)


"""Potentially load a pre-trained model"""
init_step = 0
total_steps = 200
if init_step != 0:
    saved_model_params = torch.load(f'../data/{Lx}x{Ly}/t={t}_V={V}/N={N_f}/{symmetry}/{model_name}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except:
        model.load_params(saved_model_params_vec)


"""Set VMC sample size"""
N_samples = 2**10
N_samples = N_samples - N_samples % SIZE + SIZE - 1


"""Choose the sampler""" 
sampler = MetropolisExchangeSampler(H.hilbert, H.graph, N_samples=N_samples, burn_in_steps=16, reset_chain=False, random_edge=True, dtype=dtype)
# sampler = None


"""Choose the optimizer and preconditioner"""
# optimizer = SignedSGD(learning_rate=0.05)
optimizer = SGD(learning_rate=0.05)
preconditioner = SR(dense=False, exact=True if sampler is None else False, use_MPI4Solver=True, diag_eta=0.05, iter_step=1e5, dtype=dtype)
# preconditioner = TrivialPreconditioner()


"""Create the variational state object"""
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)


"""Create the VMC object"""
vmc = VMC(hamiltonian=H, variational_state=variational_state, optimizer=optimizer, preconditioner=preconditioner)


if __name__ == "__main__":

    """Run the code by command: `mpirun -np 10 python vmc_run_example.py`
    you can change the number 10 to the number of MPI processes you want to use"""
    
    os.makedirs(f'../data/{Lx}x{Ly}/t={t}_V={V}/N={N_f}/{symmetry}/{model_name}/', exist_ok=True)
    vmc.run(init_step, init_step+total_steps, tmpdir=f'../data/{Lx}x{Ly}/t={t}_V={V}/N={N_f}/{symmetry}/{model_name}/')

