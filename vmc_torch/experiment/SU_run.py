# import os
# os.environ["NUMBA_NUM_THREADS"] = "20"

import netket as nk
import netket.experimental as nkx

from math import pi

from vmc_torch.fermion_utils import generate_random_fpeps
import quimb.tensor as qtn
import symmray as sr
import pickle

# Define the lattice shape
Lx = 8
Ly = 8
spinless = False
# graph = nk.graph.Square(L)
graph = nk.graph.Grid([Lx,Ly], pbc=False)
N = graph.n_nodes

# Define the fermion filling and the Hilbert space
N_f = int(Lx*Ly)
n_fermions_per_spin = (N_f//2, N_f//2)
hi = nkx.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermions_per_spin=n_fermions_per_spin)


# # Define the Hubbard Hamiltonian
# t = 1.0
# U = 8.0
# mu = 0.0

# H = 0.0
# for (i, j) in graph.edges(): # Definition of the Hubbard Hamiltonian
#     for spin in (1,-1):
#         H -= t * (cdag(hi,i,spin) * c(hi,j,spin) + cdag(hi,j,spin) * c(hi,i,spin))
# for i in graph.nodes():
#     H += U * nc(hi,i,+1) * nc(hi,i,-1)


# # Exact diagonalization of the Hamiltonian for benchmark
# sp_h = H.to_sparse() # Convert the Hamiltonian to a sparse matrix
# from scipy.sparse.linalg import eigsh
# eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
# E_gs = eig_vals[0]
# print("Exact ground state energy per site:", E_gs/N)


# SU in quimb
D = 4
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
occ_fn = lambda su: print(f'N per site:{su.get_state().compute_local_expectation(N_terms, normalized=True, max_bond=64,)/N}') if su.n%50==0 else None
density = False
ham = qtn.LocalHam2D(Lx, Ly, terms)

su = qtn.SimpleUpdateGen(peps, ham, compute_energy_per_site=True, D=D, compute_energy_opts={"max_distance":1}, gate_opts={'cutoff':1e-12}, callback=occ_fn if density else None)

# cluster energies may not be accuracte yet
su.evolve(50, tau=0.3)
su.evolve(50, tau=0.1)
# su.evolve(50, tau=0.03)
# # su.evolve(50, tau=0.01)
# # su.evolve(50, tau=0.003)

peps = su.get_state()
peps.equalize_norms_(value=1)

# save the state
params, skeleton = qtn.pack(peps)
skeleton.exponent = 1  # Set the exponent to 1 for SU(2) symmetry

import os
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'
os.makedirs(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}', exist_ok=True)

with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_skeleton.pkl', 'wb') as f:
    pickle.dump(skeleton, f)
with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/{symmetry}/D={D}/peps_su_params.pkl', 'wb') as f:
    pickle.dump(params, f)
    