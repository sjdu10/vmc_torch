from math import pi

from vmc_torch.fermion_utils import generate_random_fpeps
import quimb.tensor as qtn
import symmray as sr
import pickle

# Define the lattice shape
Lx = 4
Ly = 4
spinless = False
N = int(Lx * Ly)
# Define the fermion filling and the Hilbert space
N_f = int(Lx*Ly-2)

# SU in quimb
D = 6
seed = 2
symmetry = 'U1'
spinless = False
peps = generate_random_fpeps(Lx, Ly, D=D, seed=2, symmetry=symmetry, Nf=N_f, spinless=spinless)[0]
edges = qtn.edges_2d_square(Lx, Ly, cyclic=False)
try:
    parse_edges_to_site_info = sr.utils.parse_edges_to_site_info
except AttributeError:
    parse_edges_to_site_info = sr.parse_edges_to_site_info
site_info = parse_edges_to_site_info(
    edges,
    D,
    phys_dim=4,
    site_ind_id="k{},{}",
    site_tag_id="I{},{}",
)

t = 1.0
U = 8.0
if N_f == int(Lx*Ly-2) or N_f == int(Lx*Ly-8):
    mu = 0.0 if symmetry == 'U1' else (U*N_f/(2*N)-2.42)#(U*N_f/(2*N)-2.3)
elif N_f == int(Lx*Ly):
    mu = 0.0 if symmetry == 'U1' else (U/2)
elif N_f == int(Lx*Ly-4):
    mu = 0.0 if symmetry == 'U1' else (U*N_f/(2*N)-2.46)
else:
    mu = 0.0

print(mu)

terms = {
    (sitea, siteb): sr.fermi_hubbard_local_array(
        t=t, U=U, mu=mu,
        symmetry=symmetry,
        coordinations=(
            site_info[sitea]['coordination'],
            site_info[siteb]['coordination'],
        ),
    )
    for (sitea, siteb) in peps.gen_bond_coos()
}
N_terms = {
    site: sr.fermi_number_operator_spinful_local_array(
        symmetry=symmetry
    )
    for site in peps.gen_site_coos()
}
def occ_fn(su):
    if su.n%100==0:
        occ_num = su.get_state().compute_local_expectation(N_terms, normalized=True, max_bond=64)/N#, mode='fit', tn_fit='zipup', bsz=2)/N
        print(f'N per site:{occ_num}')

density = True
ham = qtn.LocalHam2D(Lx, Ly, terms)

su = qtn.SimpleUpdateGen(peps, ham, compute_energy_per_site=True, D=D, compute_energy_opts={"max_distance":1}, gate_opts={'cutoff':1e-12}, callback=occ_fn if density else None)

# cluster energies may not be accuracte yet
su.evolve(50, tau=0.3)
# su.evolve(50, tau=0.1)
# su.evolve(50, tau=0.03)
# su.evolve(50, tau=0.01)
# su.evolve(50, tau=0.003)

peps = su.get_state()
peps.equalize_norms_(value=1)

# save the state
params, skeleton = qtn.pack(peps)
skeleton.exponent = 0

import os
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'
os.makedirs(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}', exist_ok=True)

with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_skeleton.pkl', 'wb') as f:
    pickle.dump(skeleton, f)
with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_su_params.pkl', 'wb') as f:
    pickle.dump(params, f)
    