import netket as nk
import netket.experimental as nkx
import netket.nn as nknn

from math import pi
from autoray import do

from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc

from .fermion_utils import from_netket_config_to_quimb_config, from_quimb_config_to_netket_config, calc_phase_correction_netket_symmray


class Hamiltonian:
    def __init__(self, H, hi, graph):
        self._H = H
        self._hi = hi
        self._graph = graph
    
    @property
    def hi(self):
        return self._hi
    
    @property
    def hilbert(self):
        return self._hi
    
    @property
    def H(self):
        return self._H
    
    @property
    def graph(self):
        return self._graph


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


#----- self-customized Hamiltonian class -----#

class spinless_Fermi_Hubbard_square_lattice(Hamiltonian):
    def __init__(self, Lx, Ly, t, V, N_f, pbc=False):
        H, hi, graph = square_lattice_spinless_Fermi_Hubbard(Lx, Ly, t, V, N_f, pbc)
        super().__init__(H, hi, graph)
    def get_conn(self, sigma):
        assert self.hi.spin is None
        return self.H.get_conn(sigma)

class spinful_Fermi_Hubbard_square_lattice(Hamiltonian):
    def __init__(self, Lx, Ly, t, U, N_f, pbc=False, n_fermion_per_spin=None):
        H, hi, graph = square_lattice_spinful_Fermi_Hubbard(Lx, Ly, t, U, N_f, pbc, n_fermion_per_spin)
        super().__init__(H, hi, graph)

    def get_conn(self, sigma):
        assert self.hi.spin == 1/2
        if len(sigma) == self.graph.n_nodes:
            config_calc = from_quimb_config_to_netket_config(sigma)
        else:
            raise ValueError("The input configuration is not compatible with the Hamiltonian Hilbert space.")
        eta_calc_netket, H_etasigma = self.H.get_conn(config_calc)
        eta_calc = from_netket_config_to_quimb_config(eta_calc_netket)
        # Calculate the phase correction
        correction_phase_vec = do('array', ([calc_phase_correction_netket_symmray(sigma, eta) for eta in eta_calc]))
        H_etasigma *= correction_phase_vec

        return eta_calc, H_etasigma