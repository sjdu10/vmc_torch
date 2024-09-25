from fermion_utils import *
# import flax.linen as nn
# from flax.core import FrozenDict
# from flax import traverse_util
import netket as nk
import netket.experimental as nkx
import netket.nn as nknn

from math import pi

from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc

# Define the lattice shape
L = 4  # Side of the square
Lx = int(L)
Ly = int(L)
# graph = nk.graph.Square(L)
graph = nk.graph.Grid([Lx,Ly], pbc=False)
N = graph.n_nodes


# Define the fermion filling and the Hilbert space
N_f = int(Lx*Ly/2)
hi = nkx.hilbert.SpinOrbitalFermions(N, s=None, n_fermions=N_f)


# Define the Hubbard Hamiltonian
t = 1.0
V = 4.0
H = 0.0
for (i, j) in graph.edges(): # Definition of the Hubbard Hamiltonian
    H -= t * (cdag(hi,i) * c(hi,j) + cdag(hi,j) * c(hi,i))
    H += V * nc(hi,i) * nc(hi,j)


# Exact diagonalization of the Hamiltonian for benchmark
sp_h = H.to_sparse() # Convert the Hamiltonian to a sparse matrix
from scipy.sparse.linalg import eigsh
eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
E_gs = eig_vals[0]
print("Exact ground state energy per site:", E_gs/N)


# SU in quimb
D = 4
seed = 2
symmetry = 'Z2'
peps, parity_config = generate_random_fpeps(Lx, Ly, D, seed, symmetry, Nf=N_f)

edges = qtn.edges_2d_square(Lx, Ly)
site_info = sr.utils.parse_edges_to_site_info(
    edges,
    D,
    phys_dim=2,
    site_ind_id="k{},{}",
    site_tag_id="I{},{}",
)

t = 1.0
V = 4.0
mu = 0.0

terms = {
    (sitea, siteb): sr.fermi_hubbard_spinless_local_array(
        t=t, V=V, mu=mu,
        symmetry=symmetry,
        coordinations=(
            site_info[sitea]['coordination'],
            site_info[siteb]['coordination'],
        ),
    ).fuse((0, 1), (2, 3))
    for (sitea, siteb) in peps.gen_bond_coos()
}
ham = qtn.LocalHam2D(Lx, Ly, terms)
su = qtn.SimpleUpdateGen(peps, ham, compute_energy_per_site=True,D=D, compute_energy_opts={"max_distance":2}, gate_opts={'cutoff':1e-12})

# cluster energies may not be accuracte yet
su.evolve(50, tau=0.3)
# su.evolve(50, tau=0.1)
# su.evolve(100, tau=0.03)
# su.evolve(100, tau=0.01)
# su.evolve(100, tau=0.003)

peps = su.get_state()
peps.equalize_norms_(value=1)

# save the state
params, skeleton = qtn.pack(peps)
import pickle
with open(f'./data/{Lx}x{Ly}/{symmetry}/peps_skeleton.pkl', 'wb') as f:
    pickle.dump(skeleton, f)
with open(f'./data/{Lx}x{Ly}/{symmetry}/peps_su_params.pkl', 'wb') as f:
    pickle.dump(params, f)
    


