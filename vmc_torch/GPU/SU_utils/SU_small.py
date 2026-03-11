# import os
# os.environ["NUMBA_NUM_THREADS"] = "20"
from SU_func import (
    run_u1SU_w_pinning_field,
    run_u1SU,
    run_z2SU_from_u1SU,
)

Lx, Ly = 3, 2
D0 = 4
N_f = Lx*Ly-2
t = 1.0
U = 8.0
mu = 0.0
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/GPU/data'
seed = 42


su_kwargs = {
    "compute_energy_per_site": True,
    'compute_energy_final':False,
    'compute_energy_opts':{"max_distance":1}, 
    'compute_energy_every':None,
    'gate_opts':{'cutoff':0.0},
    'ordering': 'smallest_last',
    'tol':1e-6,
}
D1 = 4
beta = 5
su_evolve_schedule = [
    (50, 0.01),
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

# su_kwargs = {
#     "compute_energy_per_site": True,
#     'compute_energy_opts':{"max_distance":1}, 
#     'compute_energy_every':50,
#     'gate_opts':{'cutoff':0.0},
#     'ordering': 'smallest_last',
#     'tol':2.5e-5,
# }
# D2 = 4
# beta = 5
# su_evolve_schedule = [
#     (500, 0.001),
# ]
# z2peps = run_z2SU_from_u1SU(
#     Lx=Lx,
#     Ly=Ly,
#     D=D2,
#     N_f=N_f,
#     t=t,U=U,mu=mu,
#     pwd=pwd,
#     u1peps=u1peps,
#     save_file=True,
#     run_su=True,
#     save_every=500,
#     su_evolve_schedule=su_evolve_schedule,
#     **su_kwargs,
# )

# su_kwargs = {
#     "compute_energy_per_site": True,
#     'compute_energy_opts':{"max_distance":1}, 
#     'compute_energy_every':50,
#     'gate_opts':{'cutoff':0.0},
#     'ordering': 'smallest_last',
#     'tol':1.5e-5,
# }
# D2 = 6
# beta = 5
# su_evolve_schedule = [
#     (500, 0.001),
# ]
# z2peps = run_z2SU_from_u1SU(
#     Lx=Lx,
#     Ly=Ly,
#     D=D2,
#     N_f=N_f,
#     t=t,U=U,mu=mu,
#     pwd=pwd,
#     u1peps=u1peps,
#     save_file=True,
#     run_su=True,
#     save_every=500,
#     su_evolve_schedule=su_evolve_schedule,
#     **su_kwargs,
# )

su_kwargs = {
    "compute_energy_per_site": True,
    'compute_energy_final':False,
    'compute_energy_opts':{"max_distance":1}, 
    'compute_energy_every':None,
    'gate_opts':{'cutoff':0.0},
    'ordering': 'smallest_last',
    'tol':1e-6,
}
D2 = 4
beta = 5
su_evolve_schedule = [
    (100, 0.001),
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
    run_su=True,
    su_evolve_schedule=su_evolve_schedule,
    **su_kwargs,
)