import netket as nk
import netket.experimental as nkx
import netket.nn as nknn

from math import pi
from autoray import do

from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc

from .fermion_utils import from_netket_config_to_quimb_config, from_quimb_config_to_netket_config, calc_phase_correction_netket_symmray



class Hilbert:
    def __init__(self, hi):
        self._hi = hi
    
    @property
    def hi(self):
        return self._hi
    
    @property
    def size(self):
        return self.hi.size
    
    def to_quimb_config(self, config):
        raise NotImplementedError
    
    def all_states(self):
        return do('array',self.to_quimb_config(self.hi.all_states()))
    
    def random_state(self, key):
        """key is a random key for jax"""
        return self.to_quimb_config(self.hi.random_state(key))


class SpinlessFermion(Hilbert):
    def __init__(self, n_orbitals, n_fermions):
        hi = nkx.hilbert.SpinOrbitalFermions(n_orbitals, s=None, n_fermions=n_fermions)
        super().__init__(hi)
    
    def to_quimb_config(self, config):
        def func(x):
            return x
        if len(config.shape) == 1:
            return func(config)
        else:
            return do('array',([func(c) for c in config]), dtype=int)
    
    def all_states(self):
        return self.hi.all_states()
    
    def random_state(self, key):
        return self.hi.random_state(key)


class SpinfulFermion(Hilbert):
    def __init__(self, n_orbitals, n_fermions, n_fermions_per_spin=None):
        if n_fermions_per_spin == (None, None):
            n_fermions_per_spin = None
        hi = nkx.hilbert.SpinOrbitalFermions(n_orbitals, s=1/2, n_fermions=n_fermions, n_fermions_per_spin=n_fermions_per_spin)
        super().__init__(hi)
    
    def to_quimb_config(self, config):
        return from_netket_config_to_quimb_config(config)


class Spin(Hilbert):
    def __init__(self, s, N, total_sz=None):
        hi = nk.hilbert.Spin(s, N, total_sz=total_sz)
        super().__init__(hi)
    
    def from_netket_to_quimb_spin_config(self, config):
        """From (-1,1) to (0,1) basis"""
        def func(x):
            return (x + 1) / 2
        if len(config.shape) == 1:
            return func(config)
        else:
            return do('array',([func(c) for c in config]))
    
    def from_quimb_to_netket_spin_config(self, config):
        """From (0,1) to (-1,1) basis"""
        def func(x):
            return 2*x - 1
        if len(config.shape) == 1:
            return func(config)
        else:
            return do('array',([func(c) for c in config]))
    
    def to_quimb_config(self, config):
        return self.from_netket_to_quimb_spin_config(config)




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
        raise NotImplementedError
    
    @property
    def H(self):
        return self._H
    
    @property
    def graph(self):
        return self._graph
    
    def get_conn(self, sigma):
        raise NotImplementedError


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

def square_lattice_spinful_Fermi_Hubbard(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=None):
    graph = nk.graph.Grid([Lx,Ly], pbc=pbc)
    N = graph.n_nodes
    if n_fermions_per_spin is None:
        hi = nkx.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermions=N_f)
    else:
        hi = nkx.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermions_per_spin=n_fermions_per_spin)
    H = 0.0
    for (i, j) in graph.edges():  # Definition of the spinful Hubbard Hamiltonian
        for spin in (1,-1):
            H -= t * (cdag(hi,i,spin) * c(hi,j,spin) + cdag(hi,j,spin) * c(hi,i,spin))
    for i in graph.nodes():
        H += U * nc(hi,i,+1) * nc(hi,i,-1)
    return H, hi, graph

def square_lattice_spin_Heisenberg(L, J, pbc=False, total_sz=None):
    # Build square lattice with nearest and next-nearest neighbor edges
    # lattice = nk.graph.Square(L, max_neighbor_order=1, pbc=False)
    lattice = nk.graph.Hypercube(L, n_dim=2, pbc=pbc)
    # g = lattice = nk.graph.Pyrochlore([L, L, L], pbc=pbc)
    n = lattice.n_nodes
    hi = nk.hilbert.Spin(s=1/2, total_sz=total_sz, N=n)
    # Heisenberg with coupling J=1.0 for nearest neighbors
    # and J=0.5 for next-nearest neighbors
    # H = nk.operator.Ising(hilbert=hi, graph=lattice, J=1.0, h=1.0)
    H = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=J, sign_rule=False) # In Netket, the spin operators are Pauli matrices, while in Quimb they are 1/2*Pauli matrices
    return H, hi, lattice



#----- self-customized Hamiltonian class -----#

class spinless_Fermi_Hubbard_square_lattice(Hamiltonian):
    def __init__(self, Lx, Ly, t, V, N_f, pbc=False):
        H, hi, graph = square_lattice_spinless_Fermi_Hubbard(Lx, Ly, t, V, N_f, pbc)
        super().__init__(H, hi, graph)
        self._hilbert = SpinlessFermion(self.hi.size, self.hi.n_fermions)
    
    @property
    def hilbert(self):
        return self._hilbert
    
    def get_conn(self, sigma):
        # assert self.hi.spin is None
        return self.H.get_conn(sigma)

class spinful_Fermi_Hubbard_square_lattice(Hamiltonian):
    def __init__(self, Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=None):
        H, hi, graph = square_lattice_spinful_Fermi_Hubbard(Lx, Ly, t, U, N_f, pbc, n_fermions_per_spin)
        super().__init__(H, hi, graph)
        self._hilbert = SpinfulFermion(self.hi.size, self.hi.n_fermions, self.hi.n_fermions_per_spin)
    
    @property
    def hilbert(self):
        return self._hilbert

    def get_conn(self, sigma):
        # Quimb2Netket
        if len(sigma) == self.graph.n_nodes:
            config_calc = from_quimb_config_to_netket_config(sigma)
        else:
            raise ValueError("The input configuration is not compatible with the Netket Hamiltonian Hilbert space.")
        eta_calc_netket, H_etasigma = self.H.get_conn(config_calc)
        # Netket2Quimb
        eta_calc = from_netket_config_to_quimb_config(eta_calc_netket)
        # Calculate the phase correction
        correction_phase_vec = do('array', ([calc_phase_correction_netket_symmray(sigma, eta) for eta in eta_calc]))
        H_etasigma *= correction_phase_vec

        return eta_calc, H_etasigma


class spin_Heisenberg_square_lattice(Hamiltonian):
    def __init__(self, L, J, pbc=False, total_sz=None):
        H, hi, graph = square_lattice_spin_Heisenberg(L, J, pbc, total_sz)
        super().__init__(H, hi, graph)
        self._hilbert = Spin(self.hi._s, self.hi.size, total_sz)
    
    @property
    def hilbert(self):
        return self._hilbert
    
    def get_conn(self, sigma):
        # Quimb2Netket
        config_calc = self.hilbert.from_quimb_to_netket_spin_config(sigma)
        eta_calc_netket, H_etasigma = self.H.get_conn(config_calc)
        # Netket2Quimb
        eta_calc = self.hilbert.to_quimb_config(eta_calc_netket)
        return eta_calc, H_etasigma
    