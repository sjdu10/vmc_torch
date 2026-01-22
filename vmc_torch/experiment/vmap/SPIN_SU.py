import os
import pickle
import quimb.tensor as qtn

Lx = 4
Ly = 6
D = 4
seed = 42

# note fully random initialization will not be a very
# good initial state, used as a demonstration only here
peps = qtn.PEPS.rand(Lx, Ly, D, seed=seed)
_, skeleton = qtn.pack(peps)

ham = qtn.ham_2d_heis(Lx, Ly, j=1.0)

su = qtn.SimpleUpdateGen(
    peps,
    ham,
    # setting a cutoff is important to turn on dynamic charge sectors
    # cutoff=1e-12,
    cutoff=0.0,
    second_order_reflect=True,
    # SimpleUpdateGen computes cluster energies by default
    # which might not be accurate
    compute_energy_every=10,
    compute_energy_opts=dict(max_distance=1),
    compute_energy_per_site=True,
    # use a fixed trotterization order
    ordering="sort",
    # if the gauge difference drops below this, we consider the PEPS converged
    tol=1e-9,
)

# run the evolution, these are reasonable defaults
su.evolve(5, tau=0.3)
# su.evolve(100, tau=0.1)
# su.evolve(100, tau=0.05)
# su.evolve(100, tau=0.01)

su_peps = su.get_state()
# save the state
params, _ = qtn.pack(su_peps)


pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/data'
os.makedirs(pwd+f'/{Lx}x{Ly}/heis/D={D}', exist_ok=True)
with open(pwd+f'/{Lx}x{Ly}/heis/D={D}/peps_skeleton.pkl', 'wb') as f:
    pickle.dump(skeleton, f)
with open(pwd+f'/{Lx}x{Ly}/heis/D={D}/peps_su_params.pkl', 'wb') as f:
    pickle.dump(params, f)
