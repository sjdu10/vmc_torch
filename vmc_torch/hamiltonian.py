import netket as nk
import netket.experimental as nkx
import netket.nn as nknn

from math import pi

from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc

# Currently we borrow the Hamiltonian objects from the netket library.

def square_lattice_spinless_Fermi_Hubbard(Lx, Ly, t, V, N_f, pbc=False):
    graph = nk.graph.Grid([Lx,Ly], pbc=pbc)
    N = graph.n_nodes
    hi = nkx.hilbert.SpinOrbitalFermions(N, s=None, n_fermions=N_f)
    H = 0.0
    for (i, j) in graph.edges(): # Definition of the spinless Hubbard Hamiltonian
        H -= t * (cdag(hi,i) * c(hi,j) + cdag(hi,j) * c(hi,i))
        H += V * nc(hi,i) * nc(hi,j)
    return H, hi, graph

def square_lattice_spinful_Fermi_Hubbard(Lx, Ly, t, U, N_f, pbc=False, n_fermion_per_spin=None):
    graph = nk.graph.Grid([Lx,Ly], pbc=pbc)
    N = graph.n_nodes
    if n_fermion_per_spin is None:
        hi = nkx.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermions=N_f)
    else:
        hi = nkx.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermion_per_spin=n_fermion_per_spin)
    H = 0.0
    for (i, j) in graph.edges():  # Definition of the spinful Hubbard Hamiltonian
        for spin in (1,-1):
            H -= t * (cdag(hi,i,spin) * c(hi,j,spin) + cdag(hi,j,spin) * c(hi,i,spin))
    for i in graph.nodes():
        H += U * nc(hi,i,+1) * nc(hi,i,-1)
    return H, hi, graph

