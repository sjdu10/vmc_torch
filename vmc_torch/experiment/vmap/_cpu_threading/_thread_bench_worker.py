
import os, sys, json, time

num_threads = int(sys.argv[1])
batch_size = int(sys.argv[2])
n_repeats = int(sys.argv[3])
config_json = sys.argv[4]
config = json.loads(config_json)

# Set ALL thread env vars BEFORE any imports
for var in ['MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[var] = str(num_threads)

import numpy as np
import quimb.tensor as qtn
import pickle
import torch
import autoray as ar
import warnings
warnings.filterwarnings('ignore')

from vmc_torch.experiment.vmap.vmap_utils import (
    random_initial_config, sample_next_reuse, evaluate_energy_reuse, compute_grads,
)
from vmc_torch.experiment.vmap.models import fPEPS_Model_reuse
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.experiment.vmap.vmap_torch_utils import robust_svd_err_catcher_wrapper
from functools import partial

torch.set_num_threads(num_threads)
torch.set_default_device('cpu')
torch.random.manual_seed(42)

ar.register_function('torch', 'linalg.svd',
    lambda x: robust_svd_err_catcher_wrapper(x, jitter=1e-16, driver=None))

# --- Setup ---
Lx, Ly = config['Lx'], config['Ly']
N_f = config['N_f']
D, chi = config['D'], config['chi']
t_hop, U_int = config['t'], config['U']

pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/data'
params_path = f'{pwd}/{Lx}x{Ly}/t={t_hop}_U={U_int}/N={N_f}/Z2/D={D}/'
params = pickle.load(open(params_path + 'peps_su_params_U1SU.pkl', 'rb'))
skeleton = pickle.load(open(params_path + 'peps_skeleton_U1SU.pkl', 'rb'))
peps = qtn.unpack(params, skeleton)
for ts in peps.tensors:
    ts.modify(data=ts.data.to_flat() * 4)
for site in peps.sites:
    peps[site].data._label = site
    peps[site].data.indices[-1]._linearmap = ((0, 0), (1, 0), (1, 1), (0, 1))

# Build reusable fPEPS model (chi = 4D)
model_dtype = torch.float64
fpeps_model = fPEPS_Model_reuse(
    tn=peps,
    max_bond=chi,
    dtype=model_dtype,
    contract_boundary_opts={
        'mode': 'mps',
        'canonize': True,
    },
)

H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, t_hop, U_int, N_f, pbc=False,
    n_fermions_per_spin=(N_f // 2, N_f // 2), no_u1_symmetry=False,
)

B = batch_size
B_grad = max(1, B // 2)
get_grads = partial(compute_grads, vectorize=True, vmap_grad=True, batch_size=B_grad, verbose=False)
fxs0 = torch.stack([random_initial_config(N_f, peps.nsites) for _ in range(B)]).to(torch.long)

# Cache bMPS skeletons (needed for reuse model)
fpeps_model.cache_bMPS_skeleton(fxs0[0])

# --- Warmup (1 run, discarded) ---
with torch.no_grad():
    fxs_w, amps_w = sample_next_reuse(fxs0.clone(), fpeps_model, H.graph)
    evaluate_energy_reuse(fxs_w, fpeps_model, H, amps_w)
get_grads(fxs0.clone(), fpeps_model)

# --- Timed runs ---
sample_times = []
energy_times = []
grad_times = []

for _ in range(n_repeats):
    fxs = fxs0.clone()

    with torch.no_grad():
        t0 = time.perf_counter()
        fxs, amps = sample_next_reuse(fxs, fpeps_model, H.graph)
        t1 = time.perf_counter()
        evaluate_energy_reuse(fxs, fpeps_model, H, amps)
        t2 = time.perf_counter()

    get_grads(fxs0.clone(), fpeps_model)
    t3 = time.perf_counter()

    sample_times.append(t1 - t0)
    energy_times.append(t2 - t1)
    grad_times.append(t3 - t2)

result = {
    'num_threads': num_threads,
    'batch_size': batch_size,
    'sample_times': sample_times,
    'energy_times': energy_times,
    'grad_times': grad_times,
}
print(json.dumps(result))
