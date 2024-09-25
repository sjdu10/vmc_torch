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


def square_lattice_spinless_Fermi_Hubbard(Lx, Ly, t, V, N_f, pbc=False):
    graph = nk.graph.Grid([Lx,Ly], pbc=pbc)
    N = graph.n_nodes
    hi = nkx.hilbert.SpinOrbitalFermions(N, s=None, n_fermions=N_f)
    H = 0.0
    for (i, j) in graph.edges(): # Definition of the Hubbard Hamiltonian
        H -= t * (cdag(hi,i) * c(hi,j) + cdag(hi,j) * c(hi,i))
        H += V * nc(hi,i) * nc(hi,j)
    return H, hi, graph

