import netket as nk
import netket.experimental as nkx
import jax
from netket.graph import Lattice

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
        """Netket Hilbert space"""
        return self._hi
    
    @property
    def size(self):
        return self.hi.size
    
    def to_quimb_config(self, config):
        raise NotImplementedError
    
    def all_states(self):
        return do('array',self.to_quimb_config(self.hi.all_states()))
    
    def random_state(self, key):
        if type(key) is int:
            key = jax.random.PRNGKey(key)
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


class SpinfulFermion(Hilbert):
    def __init__(self, n_orbitals, n_fermions, n_fermions_per_spin=None):
        if n_fermions_per_spin is None:
            hi = nkx.hilbert.SpinOrbitalFermions(n_orbitals, s=1/2, n_fermions=n_fermions)
        else:
            hi = nkx.hilbert.SpinOrbitalFermions(n_orbitals, s=1/2, n_fermions_per_spin=n_fermions_per_spin)
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
        self._hilbert = None # Customized Hilbert space
    
    @property
    def hi(self):
        """Netket Hilbert space"""
        return self._hi
    
    @property
    def hilbert(self):
        """Return the customized Hilbert space"""
        if self._hilbert is None:
            raise NotImplementedError("The customized Hilbert space is not defined.")
        return self._hilbert
    
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
    """Netket implementation of spinful Fermi-Hubbard model on a square lattice"""
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

def square_lattice_spin_Heisenberg(Lx, Ly, J, pbc=False, total_sz=None):
    # Build square lattice with nearest neighbor edges
    graph = nk.graph.Grid([Lx, Ly], pbc=pbc)
    n = graph.n_nodes
    hi = nk.hilbert.Spin(s=1/2, total_sz=total_sz, N=n)
    # Heisenberg with coupling J=1.0 for nearest neighbors
    H = nk.operator.Heisenberg(hilbert=hi, graph=graph, J=J, sign_rule=False) # In Netket, the spin operators are Pauli matrices, while in Quimb they are 1/2*Pauli matrices
    return H, hi, graph

def square_lattice_spin_J1J2(Lx, Ly, J1, J2, pbc=False, total_sz=None):
    """J1-J2 model on a square lattice"""
    basis = do('array', [
     [1.0,0.0],
     [0.0,1.0],
    ])
    custom_edges = [
        (0, 0, [1.0,0.0], 0),
        (0, 0, [0.0,1.0], 0),
        (0, 0, [1.0, 1.0], 1),
        (0, 0, [1.0, -1.0], 1),
    ]
    graph = Lattice(basis_vectors=basis, pbc=pbc, extent=[Lx, Ly],
        custom_edges=custom_edges)
    n = graph.n_nodes
    hi = nk.hilbert.Spin(s=1/2, total_sz=total_sz, N=n)
    H = nk.operator.Heisenberg(hilbert=hi, graph=graph, J=(J1,J2))

    return H, hi, graph

def chain_spinful_Fermi_Hubbard(L, t, U, N_f, pbc=False, n_fermions_per_spin=None):
    """Netket implementation of spinful Fermi-Hubbard model on a 1D chain"""
    graph = nk.graph.Chain(L, pbc=pbc)
    N = graph.n_nodes
    if n_fermions_per_spin is None:
        hi = nkx.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermions=N_f)
    else:
        hi = nkx.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermions_per_spin=n_fermions_per_spin)
    H = 0.0
    for (i, j) in graph.edges(): # Definition of the Hubbard Hamiltonian
        for spin in (1,-1):
            H -= t * (cdag(hi,i,spin) * c(hi,j,spin) + cdag(hi,j,spin) * c(hi,i,spin))
    for i in graph.nodes():
        H += U * nc(hi,i,+1) * nc(hi,i,-1)
    return H, hi, graph

def chain_spinful_random_Hubbard(L, t_mean, t_std, U, N_f, n_fermions_per_spin=None, seed=42):
    import numpy as np
    np.random.seed(seed)
    """Netket implementation of fully-connected random hopping Hubbard model"""
    # Generate all possible edges for a fully connected graph
    edges = [(i, j) for i in range(L) for j in range(i + 1, L)]

    # Create the fully connected graph
    graph = nk.graph.Graph(edges=edges)
    N = graph.n_nodes

    # Generate edge-hopping dictionary
    edge_to_hopping = {}
    for (i,j) in graph.edges():
        t = np.random.normal(t_mean, t_std)
        edge_to_hopping[(i,j)] = t

    if n_fermions_per_spin is None:
        hi = nkx.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermions=N_f)
    else:
        hi = nkx.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermions_per_spin=n_fermions_per_spin)
    H = 0.0
    for (i,j) in graph.edges():
        t = edge_to_hopping[(i,j)]
        for spin in (1,-1):
            H -= t * (cdag(hi,i,spin) * c(hi,j,spin) + cdag(hi,j,spin) * c(hi,i,spin))
    for i in graph.nodes():
        H += U * nc(hi,i,+1) * nc(hi,i,-1)
    return H, hi, graph

#----- self-customized Hamiltonian class -----#

# Fermionic Hamiltonians

class spinless_Fermi_Hubbard_square_lattice(Hamiltonian):
    def __init__(self, Lx, Ly, t, V, N_f, pbc=False):
        H, hi, graph = square_lattice_spinless_Fermi_Hubbard(Lx, Ly, t, V, N_f, pbc)
        super().__init__(H, hi, graph)
        self._hilbert = SpinlessFermion(self.hi.n_orbitals, self.hi.n_fermions)
    
    def get_conn(self, sigma):
        # assert self.hi.spin is None
        return self.H.get_conn(sigma)

class spinful_Fermi_Hubbard_square_lattice(Hamiltonian):
    def __init__(self, Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=None):
        H, hi, graph = square_lattice_spinful_Fermi_Hubbard(Lx, Ly, t, U, N_f, pbc, n_fermions_per_spin)
        super().__init__(H, hi, graph)
        self._hilbert = SpinfulFermion(hi.n_orbitals, hi.n_fermions, hi.n_fermions_per_spin)

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

class spinful_Fermi_Hubbard_chain(Hamiltonian):
    def __init__(self, L, t, U, N_f, pbc=False, n_fermions_per_spin=None):
        H, hi, graph = chain_spinful_Fermi_Hubbard(L, t, U, N_f, pbc, n_fermions_per_spin)
        super().__init__(H, hi, graph)
        self._hilbert = SpinfulFermion(hi.n_orbitals, hi.n_fermions, hi.n_fermions_per_spin)

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

class spinful_random_Hubbard_chain(Hamiltonian):
    def __init__(self, L, t_mean, t_std, U, N_f, n_fermions_per_spin=None, seed=42):
        H, hi, graph = chain_spinful_random_Hubbard(L, t_mean, t_std, U, N_f, n_fermions_per_spin,seed)
        super().__init__(H, hi, graph)
        self._hilbert = SpinfulFermion(hi.n_orbitals, hi.n_fermions, hi.n_fermions_per_spin)

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


# Spin Hamiltonians

class spin_Heisenberg_square_lattice(Hamiltonian):
    def __init__(self, Lx, Ly, J, pbc=False, total_sz=None):
        H, hi, graph = square_lattice_spin_Heisenberg(Lx, Ly, J, pbc, total_sz)
        super().__init__(H, hi, graph)
        self._hilbert = Spin(self.hi._s, self.hi.size, total_sz)
    
    def get_conn(self, sigma):
        # Quimb2Netket
        config_calc = self.hilbert.from_quimb_to_netket_spin_config(sigma)
        eta_calc_netket, H_etasigma = self.H.get_conn(config_calc)
        # Netket2Quimb
        eta_calc = self.hilbert.to_quimb_config(eta_calc_netket)
        return eta_calc, H_etasigma


class spin_J1J2_square_lattice(Hamiltonian):
    def __init__(self, Lx, Ly, J1, J2, pbc=False, total_sz=None):
        H, hi, graph = square_lattice_spin_J1J2(Lx, Ly, J1, J2, pbc, total_sz)
        super().__init__(H, hi, graph)
        self._hilbert = Spin(self.hi._s, self.hi.size, total_sz)
    
    def get_conn(self, sigma):
        # Quimb2Netket
        config_calc = self.hilbert.from_quimb_to_netket_spin_config(sigma)
        eta_calc_netket, H_etasigma = self.H.get_conn(config_calc)
        # Netket2Quimb
        eta_calc = self.hilbert.to_quimb_config(eta_calc_netket)
        return eta_calc, H_etasigma
    