import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from mpi4py import MPI
import pickle
# pwd = os.getcwd()
# torch
import torch

# quimb
import quimb.tensor as qtn
import autoray as ar

from vmc_torch.experiment.tn_model import fTNModel_reuse, init_weights_uniform, fTN_backflow_attn_Tensorwise_Model_v1
from vmc_torch.sampler import MetropolisExchangeSamplerSpinful_2D_reusable, MetropolisExchangeSamplerSpinful_hopping
from vmc_torch.variational_state import Variational_State
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.torch_utils import SVD,QR_tao
from vmc_torch.utils import closest_divisible
from symmray.fermionic_local_operators import FermionicOperator


# Register safe SVD and QR functions to torch
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR_tao.apply)
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/scripts/4x4/test_data'
# pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
torch.manual_seed(1234 + RANK)
# Hamiltonian parameters
Lx = int(4)
Ly = int(4)
symmetry = 'U1SU'
t = 1.0
U = 8.0
N_f = int(Lx*Ly-2)
n_fermions_per_spin = (N_f//2, N_f//2)
H = spinful_Fermi_Hubbard_square_lattice_torch(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph
# TN parameters
D = 6
chi = 4*D
dtype=torch.float64

# Load PEPS
appendix = ''
if symmetry == 'U1_Z2':
    symmetry = 'Z2'
    appendix = '_U1'
elif symmetry == 'U1SU':
    symmetry = 'Z2'
    appendix = '_U1SU'
    
# Load PEPS
skeleton = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_skeleton{appendix}.pkl", "rb"))
peps_params = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_su_params{appendix}.pkl", "rb"))
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
# correct the format of oddpos
for ts in peps.tensors:
    ts.data.phase_sync(inplace=True)
    # for Z2 fPEPS converted from U1 fPEPS, need to correct the format of oddpos
    if ts.data.oddpos:
        oddpos = ts.data.oddpos
        nested_oddpos = oddpos[0].label[0]
        if isinstance(nested_oddpos, FermionicOperator):
            ts.data._oddpos = (nested_oddpos,)

# VMC sample size
N_samples = int(4e2)
N_samples = closest_divisible(N_samples, SIZE)
if (N_samples/SIZE)%2 != 0:
    N_samples += SIZE

# Set up variational model
contraction_kwargs = {
    "mode": "fit",
    "bsz": 2,
    "max_iterations": 50,
    "tn_fit": "zipup",
    "progbar": False,
    "tol": 1e-5,
}
contraction_kwargs = {"mode": "mps"}
model = fTNModel_reuse(
    peps, max_bond=chi, dtype=dtype, debug=False, contraction_kwargs=contraction_kwargs
)
# model = fTN_backflow_attn_Tensorwise_Model_v1(
#     peps,
#     max_bond=chi,
#     embedding_dim=16,
#     attention_heads=4,
#     nn_final_dim=4,
#     nn_eta=1.0,
#     dtype=dtype,
# )

init_std = 1e-3
# model.apply(lambda x: init_weights_to_zero(x, std=init_std))
model.apply(lambda x: init_weights_uniform(x, a=-1*init_std, b=init_std))

model_names = {
    fTNModel_reuse: f'fTN_reuse{appendix}',
    fTN_backflow_attn_Tensorwise_Model_v1: f'fTN_backflow_attn_Tensorwise_v1{appendix}',
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

sampler = (
    MetropolisExchangeSamplerSpinful_2D_reusable(
        H.hilbert,
        graph,
        N_samples=N_samples,
        burn_in_steps=10,
        reset_chain=False,
        random_edge=False,
        equal_partition=False,
        dtype=dtype,
        hopping_rate=0.25,
    )
    if isinstance(model, fTNModel_reuse)
    else MetropolisExchangeSamplerSpinful_hopping(
        H.hilbert,
        graph,
        N_samples=N_samples,
        burn_in_steps=10,
        reset_chain=False,
        random_edge=False,
        equal_partition=False,
        dtype=dtype,
        hopping_rate=0.25,
    )
)
# sampler = MetropolisExchangeSamplerSpinful_hopping(
#     H.hilbert,
#     graph,
#     N_samples=N_samples,
#     burn_in_steps=10,
#     reset_chain=False,
#     random_edge=True,
#     equal_partition=False,
#     dtype=dtype,
#     hopping_rate=0.5,
# )

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

if __name__ == "__main__":
    
    # graph = H.graph
    # print(graph.col_edges)
    local_configs, local_amps = sampler.sample_configs_eager(
        variational_state,
        message_tag=42
    )
    # total number of double occupied sites in local_configs
    db_occ = torch.sum((local_configs == 3).int())
    print(f'RANK {RANK}: double occupied sites: {db_occ}, total samples: {local_configs.shape[0]}')

    # gather all local configs to rank 0
    all_local_configs = COMM.gather(local_configs, root=0)

    if RANK == 0:
        local_configs = torch.cat(all_local_configs, dim=0)
        os.makedirs(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/', exist_ok=True)
        # save to local npy file
        import numpy as np
        np.save(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/{model_name}/chi={chi}/sampled_configs.npy', local_configs.numpy())

