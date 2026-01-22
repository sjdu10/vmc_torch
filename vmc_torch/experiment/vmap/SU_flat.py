import os
import pickle
import symmray as sr
import quimb.tensor as qtn

symmetry = "Z2"

Lx = 2
Ly = 2
D = 4
seed = 42

# note fully random initialization will not be a very
# good initial state, used as a demonstration only here
peps = sr.networks.PEPS_fermionic_rand(
    "Z2",
    Lx,
    Ly,
    D,
    phys_dim=[
        (0, 0),  # linear index 0 -> charge 0, offset 0
        (1, 0),  # linear index 1 -> charge 1, offset 0
        (1, 1),  # linear index 2 -> charge 1, offset 1
        (0, 1),  # linear index 3 -> charge 0, offset 1
    ],
    subsizes="equal",
    flat=False,
    seed=seed,
)
_, skeleton = qtn.pack(peps)

t=1.0
U=8.0
N_f = int(Lx*Ly)
mu=U/2
terms = sr.ham_fermi_hubbard_from_edges(
    symmetry=symmetry,
    edges=tuple(peps.gen_bond_coos()),
    t=t,
    U=U,
    mu=mu,
)
ham = qtn.LocalHam2D(Lx, Ly, terms)

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
for ts in su_peps.tensors:
    ts.data.phase_sync(inplace=True)
# for ts in su_peps.tensors:
#     ts.modify(data=ts.data.to_flat())
for site in su_peps.sites:
    su_peps[site].data._label = site
# save the state
params, _ = qtn.pack(su_peps)


pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/data'
os.makedirs(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}', exist_ok=True)
with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_skeleton.pkl', 'wb') as f:
    pickle.dump(skeleton, f)
with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_su_params.pkl', 'wb') as f:
    pickle.dump(params, f)
