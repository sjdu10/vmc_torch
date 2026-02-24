import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
from mpi4py import MPI
import pickle
# torch
import torch

# quimb
import quimb.tensor as qtn
import autoray as ar

from vmc_torch.experiment.tn_model import *
from vmc_torch.experiment.tn_model import init_weights_to_zero, init_weights_uniform
from vmc_torch.sampler import MetropolisExchangeSamplerSpinful, MetropolisExchangeSamplerSpinful_2D_reusable
from vmc_torch.variational_state import Variational_State
from vmc_torch.optimizer import SGD, SR, DecayScheduler
from vmc_torch.VMC import VMC
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.torch_utils import SVD,QR
from vmc_torch.fermion_utils import get_psi_from_fTN
from vmc_torch.global_var import DEBUG
from vmc_torch.utils import closest_divisible

pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'
torch.autograd.set_detect_anomaly(False)
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
N_f = int(Lx*Ly)
n_fermions_per_spin = (N_f//2, N_f//2)
H = spinful_Fermi_Hubbard_square_lattice_torch(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph
# TN parameters
D = 16
chi = 256
dtype=torch.float64

if symmetry == 'U1_Z2':
    su_skeleton = 'peps_skeleton_U1.pkl'
    su_params = 'peps_su_params_U1.pkl'
    symmetry = 'Z2'
else:
    su_skeleton = 'peps_skeleton.pkl'
    su_params = 'peps_su_params.pkl'
    
# Load PEPS
skeleton = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{su_skeleton}", "rb"))
peps_params = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{su_params}", "rb"))
peps = qtn.unpack(peps_params, skeleton)
# Precondition the fPEPS !!! Important for proper gradient calculation of fermionic TNS
## 1. Sync the stored fermionic phases
for ts in peps.tensors:
    ts.data.phase_sync(inplace=True)
## 2. Scale the tensor elements
scale = 1.0
peps.apply_to_arrays(lambda x: torch.tensor(scale*x, dtype=dtype))
## 3. Set the exponent to 0.0
peps.exponent = 0.0

# VMC sample size
N_samples = int(5000)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

model = fTNModel_reuse(peps, max_bond=chi, dtype=dtype, debug=True)
init_std = 5e-3

model_names = {
    fTNModel: 'fTN',
    fTNModel_reuse: 'fTN_reuse',
    fTN_backflow_Model: 'fTN_backflow',
    fTN_backflow_attn_Model: 'fTN_backflow_attn',
    fTN_backflow_attn_Tensorwise_Model: 'fTN_backflow_attn_Tensorwise',
    fTN_backflow_attn_Tensorwise_Model_v1: 'fTN_backflow_attn_Tensorwise_v1',
}

model_name = model_names.get(type(model), 'UnknownModel')

init_step = 0
final_step = 1000
total_steps = final_step - init_step
if init_step != 0:
    saved_model_params = torch.load(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/model_params_step{init_step}.pth')
    saved_model_state_dict = saved_model_params['model_state_dict']
    saved_model_params_vec = torch.tensor(saved_model_params['model_params_vec'])
    try:
        model.load_state_dict(saved_model_state_dict)
    except:
        model.load_params(saved_model_params_vec)

# optimizer = SignedSGD(learning_rate=0.05)
# Set up optimizer and scheduler
learning_rate = 0.1
scheduler = DecayScheduler(init_lr=learning_rate, decay_rate=0.9, patience=50, min_lr=1e-4)
optimizer = SGD(learning_rate=learning_rate)
# sampler = MetropolisExchangeSamplerSpinful(H.hilbert, graph, N_samples=N_samples, burn_in_steps=20, reset_chain=False, random_edge=False, equal_partition=False, dtype=dtype)
sampler = MetropolisExchangeSamplerSpinful_2D_reusable(H.hilbert, graph, N_samples=N_samples, burn_in_steps=1, reset_chain=False, random_edge=False, equal_partition=False, dtype=dtype, hopping_rate=0.25)
sampler.debug = False
if N_f == int(Lx*Ly):
    sampler.current_config = torch.tensor([1,2,1,2,
                                        2,1,2,1,
                                        1,2,1,2,
                                        2,1,2,1,])
variational_state = Variational_State(model, hi=H.hilbert, sampler=sampler, dtype=dtype)
if __name__ == "__main__":
    import numpy as np
    N_config_per_rank = N_samples // SIZE

    configs_list = []
    amp_list = []
    # add a progress bar
    from tqdm import tqdm
    pbar = tqdm(total=N_config_per_rank, desc=f"Rank {RANK} sampling configurations")
    for _ in range(N_config_per_rank):
        # Sample a configuration
        config, amp = sampler._sample_next(variational_state, burn_in=True)
        configs_list.append(config)
        amp_list.append(amp)
        pbar.update(1)
    
    # convert to numpy arrays
    configs_list = [config.numpy() for config in configs_list]
    amp_list = [amp.numpy() for amp in amp_list]
    
    # MPI gather all configurations
    all_configs = COMM.gather(configs_list, root=0)
    if RANK == 0:
        # Save the configurations and amplitudes in one file
        target_dir = pwd + f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}'
        os.makedirs(target_dir, exist_ok=True)
        np.savez(target_dir + f'/configs_and_amps.npz',
                 configs=np.concatenate(all_configs, axis=0),
                 amps=np.concatenate(amp_list, axis=0))

