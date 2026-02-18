import os
import torch
import torch.distributed as dist
import numpy as np
import pickle
import json
import time
from tqdm import tqdm

from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    random_initial_config
)
from vmc_torch.experiment.vmap.GPU.vmc_modules import (
    run_sampling_phase_gpu,
    distributed_minres_solver_gpu,
    minSR_solver_gpu,
)
from vmc_torch.experiment.vmap.vmap_models import PEPS_Model, fPEPS_Model
from vmc_torch.hamiltonian_torch import (
    spinful_Fermi_Hubbard_square_lattice_torch,
    spin_Heisenberg_square_lattice_torch,
)
from vmc_torch.experiment.tn_model import init_weights_to_zero
from vmc_torch.experiment.vmap.vmap_torch_utils import (
    robust_svd_err_catcher_wrapper
)
import autoray as ar
import quimb.tensor as qtn

JITTER = 1e-16
driver = None
ar.register_function(
    'torch', 'linalg.svd',
    lambda x: robust_svd_err_catcher_wrapper(
        x, jitter=JITTER, driver=driver
    ),
)

# ==========================================
# 1. Distributed Environment Setup (GPU)
# ==========================================
def setup_distributed():
    if "RANK" not in os.environ:
        print("Warning: Not using torchrun. Single device.")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["LOCAL_RANK"] = "0"

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device

RANK, WORLD_SIZE, device = setup_distributed()

torch.set_default_dtype(torch.float64)
torch.set_default_device(device)
torch.manual_seed(42 + RANK)

# ==========================================
# 2. Model & Hamiltonian Setup
# ==========================================
Lx, Ly = 2, 2
nsites = Lx * Ly
N_f = nsites
D = 4
chi = -1

pwd = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/'
    'vmc_torch/experiment/vmap/data'
)
u1z2 = True
appendix = '_U1SU' if u1z2 else ''

params_pkl = pickle.load(open(
    pwd + f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/'
    f'peps_su_params{appendix}.pkl', 'rb'
))
skeleton = pickle.load(open(
    pwd + f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/'
    f'peps_skeleton{appendix}.pkl', 'rb'
))
peps = qtn.unpack(params_pkl, skeleton)

for ts in peps.tensors:
    ts.modify(data=ts.data.to_flat() * 4)
for site in peps.sites:
    peps[site].data._label = site
    peps[site].data.indices[-1]._linearmap = (
        (0, 0), (1, 0), (1, 1), (0, 1)
    )

fpeps_model = fPEPS_Model(
    tn=peps, max_bond=chi, dtype=torch.float64,
    compile=False,
    contract_boundary_opts={
        'mode': 'mps',
        'equalize_norms': 1.0,
        'canonize': True,
    }
)
fpeps_model.to(device)

n_params = sum(p.numel() for p in fpeps_model.parameters())
if RANK == 0:
    print(
        f'Model parameters: {n_params} | '
        f'World Size: {WORLD_SIZE} | Device: {device}'
    )

# Hamiltonian
t, U = 1.0, 8.0
n_fermions_per_spin = (N_f // 2, N_f // 2)
H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx, Ly, t, U, N_f,
    pbc=False,
    n_fermions_per_spin=n_fermions_per_spin,
    no_u1_symmetry=False,
    gpu=True,
)
graph = H.graph

# ==========================================
# 3. VMC Settings
# ==========================================
Total_Ns = int(4096)
assert Total_Ns % WORLD_SIZE == 0
samples_per_rank = Total_Ns // WORLD_SIZE

grad_batch_size = 512
fxs = torch.stack([
    random_initial_config(N_f, nsites, seed=42 + _)
    for _ in range(samples_per_rank)
]).to(device)

# VMC Settings
vmc_steps = 50
use_minSR = True
learning_rate = 0.1
diag_shift = 1e-4
save_state_every = 10
burn_in_steps = 4

os.makedirs(
    pwd + f'/GPU/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}',
    exist_ok=True,
)
stats_file = (
    pwd + f'/GPU/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/'
    f'stats_{fpeps_model._get_name()}.json'
)
stats = {
    'Np': n_params, 'sample size': Total_Ns,
    'mean': [], 'error': [], 'variance': [],
}

# ==========================================
# 4. VMC Loop
# ==========================================
if RANK == 0:
    vmc_pbar = tqdm(total=vmc_steps, desc="VMC Steps")

for step in range(vmc_steps):
    t0 = time.time()

    # --- Step 1: Sampling (inline O_loc) ---
    (local_energies, local_O), fxs, sample_time = (
        run_sampling_phase_gpu(
            fxs=fxs,
            model=fpeps_model,
            hamiltonian=H,
            graph=graph,
            samples_per_rank=samples_per_rank,
            grad_batch_size=grad_batch_size,
            burn_in=(step == 0),
            burn_in_steps=burn_in_steps,
        )
    )

    # --- Step 2: Global energy stats ---
    local_E_sum = torch.tensor(
        local_energies.sum(), device=device
    )
    dist.all_reduce(local_E_sum, op=dist.ReduceOp.SUM)
    energy_mean = local_E_sum.item() / Total_Ns

    local_E_sq_sum = torch.tensor(
        (local_energies ** 2).sum(), device=device
    )
    dist.all_reduce(local_E_sq_sum, op=dist.ReduceOp.SUM)
    energy_var = (
        local_E_sq_sum.item() / Total_Ns - energy_mean ** 2
    )

    # --- Step 3: SR Solve ---
    if use_minSR:
        dp, t_sr, info = minSR_solver_gpu(
            local_O=local_O,
            local_energies=local_energies,
            energy_mean=energy_mean,
            total_samples=Total_Ns,
            n_params=n_params,
            diag_shift=diag_shift,
            device=device,
        )
    else:
        dp, t_sr, info = distributed_minres_solver_gpu(
            local_O=local_O,
            local_energies=local_energies,
            energy_mean=energy_mean,
            total_samples=Total_Ns,
            n_params=n_params,
            diag_shift=diag_shift,
            rtol=5e-5,
        )

    # --- Step 4: Parameter Update ---
    with torch.no_grad():
        dp_tensor = torch.tensor(
            dp, device=device, dtype=torch.float64
        )
        current_params_vec = (
            torch.nn.utils.parameters_to_vector(
                fpeps_model.parameters()
            )
        )
        new_params_vec = (
            current_params_vec - learning_rate * dp_tensor
        )
        torch.nn.utils.vector_to_parameters(
            new_params_vec, fpeps_model.parameters()
        )

    t1 = time.time()

    # --- Step 5: Logging ---
    if RANK == 0:
        e_per_site = energy_mean / nsites
        err = np.sqrt(energy_var / Total_Ns) / nsites
        print(
            f"Step {step}: E/site = {e_per_site:.6f} "
            f"+/- {err:.6f}, "
            f"T_samp={sample_time:.2f}s T_SR={t_sr:.2f}s"
        )

        stats['mean'].append(e_per_site)
        stats['error'].append(err)
        stats['variance'].append(energy_var)

        with open(stats_file, 'w') as f:
            json.dump(stats, f)

        if (step + 1) % save_state_every == 0:
            ckpt_path = (
                pwd + f'/GPU/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/'
                f'Z2/D={D}/checkpoint_'
                f'{fpeps_model._get_name()}_{step+1}.pt'
            )
            torch.save(fpeps_model.state_dict(), ckpt_path)

        vmc_pbar.update(1)

if RANK == 0:
    vmc_pbar.close()
dist.destroy_process_group()
