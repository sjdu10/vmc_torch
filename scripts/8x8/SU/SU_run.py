import os
os.environ["NUMBA_NUM_THREADS"] = "20"
from vmc_torch.experiment.su_scripts.SU_func import (
    run_u1SU_w_pinning_field,
    run_u1SU,
    run_z2SU_from_u1SU,
)

Lx, Ly = 8, 8
D0 = 8
N_f = Lx*Ly-8
t = 1.0
U = 8.0
mu = 0.0
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'
seed = 42

cpf = 1
cpf_target_sites_po = [(x,y) if (x<=1 or x>=6) else None for x in range(Lx) for y in range(Ly)]
cpf_target_sites_ne = [(x,y) if (x==3 or x==4) else None for x in range(Lx) for y in range(Ly)]
cpf_target_sites = {
    site: 1 for site in cpf_target_sites_po
}
cpf_target_sites.update({
    site: -1 for site in cpf_target_sites_ne
})
spf = 1
spf_target_sites_po = [(x,y) if ((x+y)%2==0 and x<=1) or ((x+y)%2==1 and x>=6) else None for x in range(Lx) for y in range(Ly)]
spf_target_sites_ne = [(x,y) if ((x+y)%2==1 and x<=1) or ((x+y)%2==0 and x>=6) else None for x in range(Lx) for y in range(Ly)]

spf_target_sites = {
    site: 1 for site in spf_target_sites_po
}
spf_target_sites.update({
    site: -1 for site in spf_target_sites_ne
})

su_kwargs = {
    "compute_energy_per_site": True,
    'compute_energy_opts':{"max_distance":1}, 
    'compute_energy_every':50,
    'gate_opts':{'cutoff':1e-12},
    'ordering': 'smallest_last',
    'tol':1e-6,
}
beta = 5
su_evolve_schedule = [
    (100, 0.3),
    (100, 0.1),
    (100, 0.05),
    (200, 0.01),
    # (int(beta/0.01), 0.01),
]
u1peps_w_pf = run_u1SU_w_pinning_field(
    Lx=Lx,
    Ly=Ly,
    D=D0,
    N_f=N_f,
    t=t,U=U,mu=mu,
    cpf=cpf,
    cpf_target_sites=cpf_target_sites,
    spf=spf,
    spf_target_sites=spf_target_sites,
    pwd=pwd,
    seed=seed,
    save_file=True,
    run_su=False,
    su_evolve_schedule=su_evolve_schedule,
    rfpeps_kwargs={'subsizes':'equal'},
    **su_kwargs,
)

su_kwargs = {
    # "compute_energy_per_site": True,
    # 'compute_energy_opts':{"max_distance":1}, 
    # 'compute_energy_every':50,
    'gate_opts':{'cutoff':1e-12},
    'ordering': 'smallest_last',
    'tol':1e-6,
}
D1 = 8
beta = 5
su_evolve_schedule = [
    # (int(beta/0.01), 0.01),
    (50, 0.3),
    (50, 0.1),
    (50, 0.05),
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
    initial_peps=u1peps_w_pf,
    save_file=True,
    run_su=True,
    su_evolve_schedule=su_evolve_schedule,
    **su_kwargs,
)

su_kwargs = {
    "compute_energy_per_site": True,
    'compute_energy_opts':{"max_distance":1}, 
    'compute_energy_every':50,
    'gate_opts':{'cutoff':1e-12},
    'ordering': 'smallest_last',
    'tol':2.5e-5,
}
D2 = 4
beta = 5
su_evolve_schedule = [
    # (int(beta/0.001), 0.001),
    # (50, 0.1),
    (200, 0.005)
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
    save_every=500,
    su_evolve_schedule=su_evolve_schedule,
    **su_kwargs,
)

su_kwargs = {
    "compute_energy_per_site": True,
    'compute_energy_opts':{"max_distance":1}, 
    'compute_energy_every':50,
    'gate_opts':{'cutoff':1e-12},
    'ordering': 'smallest_last',
    'tol':1.5e-5,
}
D2 = 6
beta = 5
su_evolve_schedule = [
    # (int(beta/0.001), 0.001),
    (200, 0.005),
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
    save_every=500,
    su_evolve_schedule=su_evolve_schedule,
    **su_kwargs,
)

D2 = 8
beta = 5
su_evolve_schedule = [
    # (int(beta/0.001), 0.001),
    (200, 0.005),
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

