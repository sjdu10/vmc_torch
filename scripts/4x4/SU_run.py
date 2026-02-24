import os
os.environ["NUMBA_NUM_THREADS"] = "20"
from vmc_torch.experiment.su_scripts.SU_func import (
    run_u1SU_w_pinning_field,
    run_u1SU,
    run_z2SU_from_u1SU,
)
from vmc_torch.fermion_utils import format_fpeps_keys
import numpy as np

Lx, Ly = 4, 4

N_f = Lx*Ly-2
t = 1.0
U = 8.0
mu = 0.0
pwd = './test_data'
# seed = np.random.randint(1, 100000)
seed = 42

beta=1
su_kwargs = {
    "compute_energy_per_site": True,
    'compute_energy_opts':{"max_distance":1}, 
    'compute_energy_every':50,
    'gate_opts':{'cutoff':1e-12},
    'second_order_reflect': False,
    # 'update': 'parallel',
    'ordering': 'smallest_last',
    'tol':1e-6
}

D1 = 4
su_evolve_schedule = [
    (int(beta/0.01), 0.01),
    (int(beta/0.005), 0.005),
]
u1peps = run_u1SU(
    Lx=Lx,
    Ly=Ly,
    D=D1,
    N_f=N_f,
    t=t,U=U,mu=mu,
    pwd=pwd,
    seed=seed,
    initial_peps=None,
    save_file=True,
    run_su=True,
    su_evolve_schedule=su_evolve_schedule,
    **su_kwargs,
)
u1peps = format_fpeps_keys(u1peps)

D2 = 4
su_evolve_schedule = [
    (50, 0.005),
]
z2peps = run_z2SU_from_u1SU(
    Lx=Lx,
    Ly=Ly,
    D=D2,
    N_f=N_f,
    t=t,U=U,mu=mu,
    pwd=pwd,
    u1peps=u1peps,
    save_file=True,
    su_evolve_schedule=su_evolve_schedule,
    **su_kwargs,
)

