# from vmc_torch.fermion_utils import generate_random_fpeps
import quimb.tensor as qtn
import symmray as sr
import pickle

# Define the lattice shape
Lx = 8
Ly = 8
spinless = False
N = int(Lx * Ly)
# Define the fermion filling and the Hilbert space
N_f = int(Lx*Ly-8)

# SU in quimb
D = 6
symmetry = 'Z2'
spinless = False
# Define model parameters
t=1.0
U=8.0

# Load U1 to Z2 PEPS
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'
skeleton = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/peps_skeleton_U1.pkl", "rb"))
peps_params = pickle.load(open(pwd+f"/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/peps_su_params_U1.pkl", "rb"))
peps = qtn.unpack(peps_params, skeleton)
# print(peps.tensors[0].data.blocks)
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

# if N_f == int(Lx*Ly-2) or N_f == int(Lx*Ly-8):
#     mu = 0.0 if symmetry == 'U1' else (U*N_f/(2*N)-2.42)#(U*N_f/(2*N)-2.3)
# elif N_f == int(Lx*Ly):
#     mu = 0.0 if symmetry == 'U1' else (U/2)
# elif N_f == int(Lx*Ly-4):
#     mu = 0.0 if symmetry == 'U1' else (U*N_f/(2*N)-2.46)
# else:
#     mu = 0.0

# print(mu)
mu = 0.0

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
    if su.n%50==0:
        occ_num = su.get_state().compute_local_expectation(N_terms, normalized=True, max_bond=64)/N
        print(f'N per site:{occ_num}')

density = False
ham = qtn.LocalHam2D(Lx, Ly, terms)
# print(peps.compute_local_expectation(ham.terms, normalized=True, max_bond=64)/N)
su = qtn.SimpleUpdateGen(
    peps,
    ham,
    compute_energy_per_site=True,
    D=D,
    compute_energy_opts={"max_distance": 1},
    gate_opts={"cutoff": 1e-12},
    callback=occ_fn if density else None,
    compute_energy_every=None,
)

# cluster energies may not be accuracte yet
# su.evolve(50, tau=0.3)
su.evolve(50, tau=0.1)
# su.evolve(50, tau=0.03)
# su.evolve(50, tau=0.01)
# su.evolve(50, tau=0.005)

peps = su.get_state()
peps.equalize_norms_(value=1)

# print(peps.tensors[0].data.blocks)

# save the state
params, skeleton = qtn.pack(peps)
skeleton.exponent = 0

import os
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'
os.makedirs(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}', exist_ok=True)

with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_skeleton_U1SU.pkl', 'wb') as f:
    pickle.dump(skeleton, f)
with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_su_params_U1SU.pkl', 'wb') as f:
    pickle.dump(params, f)
    