import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import quimb as qu
import quimb.tensor as qtn
import symmray as sr
import torch
import pickle
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch

Lx = 4
Ly = 4
nsites = Lx * Ly
D = 4
chi = 2*D
seed = 42
# only the flat backend is compatible with jax.jit
flat = True
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap'

params = pickle.load(open(pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N=16/Z2/D={D}/peps_su_params.pkl', 'rb'))
skeleton = pickle.load(open(pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N=16/Z2/D={D}/peps_skeleton.pkl', 'rb'))
peps = qtn.unpack(params, skeleton)
peps = sr.networks.PEPS_fermionic_rand(
    "Z2",
    Lx,
    Ly,
    D,
    phys_dim=[
        (0, 0),  # linear index 0 -> charge 0, offset 0
        (1, 1),  # linear index 1 -> charge 1, offset 1
        (1, 0),  # linear index 2 -> charge 1, offset 0
        (0, 1),  # linear index 3 -> charge 0, offset 1
    ],
    subsizes="equal",
    flat=flat,
    seed=seed,
)
peps.set_params(params)

# get pytree of initial parameters, and reference tn structure
params, skeleton = qtn.pack(peps)


def amplitude(x, params):
    tn = qtn.unpack(params, skeleton)

    fx = 2 * x[::2] + x[1::2]

    # might need to specify the right site ordering here
    amp = tn.isel({tn.site_ind(site): fx[i] for i, site in enumerate(tn.sites)})

    amp.contract_boundary_from_ymin_(max_bond=chi, cutoff=0.0, yrange=[0, amp.Ly//2-1])
    amp.contract_boundary_from_ymax_(max_bond=chi, cutoff=0.0, yrange=[amp.Ly//2, amp.Ly-1])

    return amp.contract()

# generate half-filling configs
# batchsize
B = 1024
rng = np.random.default_rng(seed)
xs_u = np.concatenate(
    [
        np.zeros((B, nsites // 2), dtype=np.int32),
        np.ones((B, nsites // 2), dtype=np.int32),
    ],
    axis=1,
)
xs_d = xs_u.copy()
xs_u = rng.permuted(xs_u, axis=1)
xs_d = rng.permuted(xs_d, axis=1)
xs = np.concatenate([xs_u[:, :, None], xs_d[:, :, None]], axis=2).reshape(B, -1)

# torch.set_default_device("cuda:0") # GPU
torch.set_default_device("cpu") # CPU

# convert bitstrings and arrays to torch
xs = torch.tensor(xs)
params = qu.tree_map(
    lambda x: torch.tensor(x, dtype=torch.float32),
    params,
)

vamp = torch.vmap(
    amplitude,
    # batch on configs, not parameters
    in_dims=(0, None),
)


# generate Hamiltonian
t=1.0
U=8.0
N_f = int(Lx*Ly) # half-filling
n_fermions_per_spin = (N_f // 2, N_f // 2)

H = spinful_Fermi_Hubbard_square_lattice_torch(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=n_fermions_per_spin)
graph = H.graph

import random
def from_coo_to_index_2D(coo, Lx, Ly):
    x, y = coo
    return x * Ly + y

def propose_exchange_or_hopping(i, j, current_config, hopping_rate=0.25):
    ind_n_map = {0: 0, 1: 1, 2: 1, 3: 2}
    if current_config[i] == current_config[j]:
        return current_config
    proposed_config = current_config.clone()
    config_i = current_config[i].item()
    config_j = current_config[j].item()
    if random.random() < hopping_rate:
        # exchange
        proposed_config[i] = config_j
        proposed_config[j] = config_i
    else:
        # hopping
        n_i = ind_n_map[current_config[i].item()]
        n_j = ind_n_map[current_config[j].item()]
        delta_n = abs(n_i - n_j)
        if delta_n == 1:
            # consider only valid hopping: (0, u) -> (u, 0); (d, ud) -> (ud, d)
            proposed_config[i] = config_j
            proposed_config[j] = config_i
        elif delta_n == 0:
            # consider only valid hopping: (u, d) -> (0, ud) or (ud, 0)
            choices = [(0, 3), (3, 0)]
            choice = random.choice(choices)
            proposed_config[i] = choice[0]
            proposed_config[j] = choice[1]
        elif delta_n == 2:
            # consider only valid hopping: (0, ud) -> (u, d) or (d, u)
            choices = [(1, 2), (2, 1)]
            choice = random.choice(choices)
            proposed_config[i] = choice[0]
            proposed_config[j] = choice[1]
        else:
            raise ValueError("Invalid configuration")
    return proposed_config
