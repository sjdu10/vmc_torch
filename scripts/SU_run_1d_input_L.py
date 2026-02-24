import os
import sys
os.environ["NUMBA_NUM_THREADS"] = "20"

from vmc_torch.fermion_utils import generate_random_fmps, u1peps_to_z2peps
import quimb.tensor as qtn
import symmray as sr
import pickle

# Define the lattice shape
L = int(sys.argv[1]) # length of the chain
spinless = False
cyclic = False
N = L

# Define the fermion filling and the Hilbert space
N_f = int(L)
n_fermions_per_spin = (N_f//2, N_f//2)

# Define the Hubbard Hamiltonian
t = 1.0
U = 8.0
mu = 0.0

# fMPS spinful
D = 10
symmetry = 'U1'
seed = 2
# SU in quimb
fmps, charge_config = generate_random_fmps(L, D, seed, symmetry, Nf=N_f, cyclic=cyclic, spinless=spinless)
edges = qtn.edges_1d_chain(L, cyclic=False)

try:
    parse_edges_to_site_info = sr.utils.parse_edges_to_site_info
except AttributeError:
    parse_edges_to_site_info = sr.parse_edges_to_site_info

site_info = parse_edges_to_site_info(
    edges,
    D,
    phys_dim=2 if spinless else 4,
    site_ind_id="k{}",
    site_tag_id="I{}",
)
print(site_info)

terms = {
    (sitea, siteb): sr.fermi_hubbard_local_array(
        t=t, U=U, mu=mu,
        symmetry=symmetry,
        coordinations=(
            site_info[sitea]['coordination'],
            site_info[siteb]['coordination'],
        ),
    )
    for (sitea, siteb) in fmps.gen_bond_coos()
}

N_terms = {
    site: sr.fermi_number_operator_spinful_local_array(
        symmetry=symmetry,
    )
    for site in fmps.gen_site_coos()
}

ham = qtn.LocalHam1D(L, terms, cyclic=False)
occ_fn = lambda su: print(f'N per site:{su.get_state().compute_local_expectation(N_terms)/N}') if su.n%5==0 else None
density = False

su = qtn.SimpleUpdateGen(
    fmps, 
    ham, 
    compute_energy_per_site=True,
    compute_energy_every=10, 
    D=D, 
    gate_opts={'cutoff':1e-10},
    callback=occ_fn if density else None
)

# cluster energies may not be accuracte yet
su.evolve(50, tau=0.3)
su.evolve(50, tau=0.1)
# su.evolve(50, tau=0.03)
# su.evolve(50, tau=0.01)
# su.evolve(100, tau=0.003)

mps = su.get_state()
mps.equalize_norms_(value=1)
z2mps = u1peps_to_z2peps(mps)

import os
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'
# os.makedirs(pwd+f'/L={L}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}', exist_ok=True)

# with open(pwd+f'/L={L}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/mps_skeleton.pkl', 'wb') as f:
#     pickle.dump(skeleton, f)
# with open(pwd+f'/L={L}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/mps_su_params.pkl', 'wb') as f:
#     pickle.dump(params, f)
params, skeleton = qtn.pack(z2mps)

os.makedirs(pwd+f'/L={L}/t={t}_U={U}/N={N_f}/Z2/D={D}', exist_ok=True)

with open(pwd+f'/L={L}/t={t}_U={U}/N={N_f}/Z2/D={D}/mps_skeleton.pkl', 'wb') as f:
    pickle.dump(skeleton, f)
with open(pwd+f'/L={L}/t={t}_U={U}/N={N_f}/Z2/D={D}/mps_su_params.pkl', 'wb') as f:
    pickle.dump(params, f)