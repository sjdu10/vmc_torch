import os
os.environ["NUMBA_NUM_THREADS"] = "20"
from SU_func import (
    run_u1SU_w_pinning_field,
    run_u1SU,
    run_z2SU_from_u1SU,
)

Lx, Ly = 4, 2
D0 = 4
N_f = Lx*Ly - 2
t = 1.0
U = 8.0
mu = 0.0
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/data'
seed = 42

# cpf = 0.0
# cpf_target_sites = [(x,x) for x in range(Ly)]
# cpf_target_sites = {
#     site: 1 for site in cpf_target_sites
# }
# spf = 0.0
# spf_target_sites = [(0,x) for x in range(Ly)]
# spf_target_sites = {
#     site: 1 for site in spf_target_sites
# }
# su_kwargs = {
#     "compute_energy_per_site": True,
#     'compute_energy_opts':{"max_distance":1}, 
#     'gate_opts':{'cutoff':0}
# }
# su_evolve_schedule = [
#     (2, 0.05),
#     (2, 0.01),
# ]
# u1peps_w_pf = run_u1SU_w_pinning_field(
#     Lx=Lx,
#     Ly=Ly,
#     D=D0,
#     N_f=N_f,
#     t=t,U=U,mu=mu,
#     cpf=cpf,
#     cpf_target_sites=cpf_target_sites,
#     spf=spf,
#     spf_target_sites=spf_target_sites,
#     pwd=pwd,
#     seed=seed,
#     save_file=True,
#     run_su=True,
#     su_evolve_schedule=su_evolve_schedule,
#     rfpeps_kwargs={"subsizes": 'equal'},
#     **su_kwargs,
# )

D1 = 4
su_kwargs = {
    "compute_energy_per_site": True,
    'compute_energy_opts':{"max_distance":1}, 
    'gate_opts':{'cutoff':0.0}
}
su_evolve_schedule = [
    (200, 0.05),
    (200, 0.01),
]
u1peps = run_u1SU(
    Lx=Lx,
    Ly=Ly,
    D=D1,
    N_f=N_f,
    t=t,U=U,mu=mu,
    pwd=pwd,
    seed=seed,
    # initial_peps=u1peps_w_pf,
    initial_peps=None,
    save_file=True,
    run_su=True,
    su_evolve_schedule=su_evolve_schedule,
    **su_kwargs,
)

su_evolve_schedule = [
    # (1, 0.0),
    (100, 0.001),
]
z2peps = run_z2SU_from_u1SU(
    Lx=Lx,
    Ly=Ly,
    D=D1,
    N_f=N_f,
    t=t,U=U,mu=mu,
    pwd=pwd,
    u1peps=u1peps,
    save_file=True,
    su_evolve_schedule=su_evolve_schedule,
    **su_kwargs,
)

