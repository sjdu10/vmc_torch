import os
import itertools
import random
import quimb.tensor as qtn
import numpy as np
from autoray import do

from vmc_torch.fermion_utils import from_netket_config_to_quimb_config, from_quimb_config_to_netket_config, calc_phase_symmray


def generate_binary_vectors(N, m):
    # Generate all combinations of m positions out of N indices
    combinations = itertools.combinations(range(N), m)
    
    # Create a vector for each combination of positions
    vectors = []
    for positions in combinations:
        vec = [0] * N
        for idx in positions:
            vec[idx] = 1
        vectors.append(vec)
    return vectors

class Hilbert:

    @property
    def size(self):
        """Number of states in the Hilbert space"""
        raise NotImplementedError
    
    def to_quimb_config(self, config):
        raise NotImplementedError
    
    def all_states(self):
        """Generate all states in the Hilbert space"""
        raise NotImplementedError
    
    def random_state(self, key):
        """Generate a random state in the Hilbert space"""
        raise NotImplementedError

class SpinfulFermion(Hilbert):
    def __init__(self, n_orbitals, n_fermions=None, n_fermions_per_spin=None):
        """
        Spinful fermionic Hilbert space with n_orbitals orbitals and n_fermions fermions.
        n_fermions_per_spin is the number of fermions per spin, if not provided, it will be set to n_fermions // 2.

        Configuration structure: [nu1,...,nuN, nd1,...,ndN]
        where nu1,...,nuN are the spin-up fermions and nd1,...,ndN are the spin-down fermions.
        The total number of fermions is n_fermions = n_fermions_up + n_fermions_down.
        The total number of orbitals is n_orbitals = n_up + n_down.
        
        """
        self.n_orbitals = n_orbitals*2  # Total number of orbitals (spin up + spin down)
        self.n_fermions = n_fermions
        self.n_fermions_spin_up = n_fermions_per_spin[0] if n_fermions_per_spin is not None else n_fermions // 2
        self.n_fermions_spin_down = n_fermions_per_spin[1] if n_fermions_per_spin is not None else n_fermions // 2

    
    def _all_states(self):
        spin_up_states = generate_binary_vectors(self.n_orbitals // 2, self.n_fermions_spin_up)
        spin_down_states = generate_binary_vectors(self.n_orbitals // 2, self.n_fermions_spin_down)
        return (up + down for up, down in itertools.product(spin_up_states, spin_down_states))
    
    def all_states(self):
        """Generate all states in the Hilbert space.
    
        Returns:
            list: All possible configurations of the Hilbert space.
        """
        return do('array', list(self._all_states()))
    
    def random_state(self, key=None):
        """Generate a random state in the Hilbert space.
    
        Args:
            key (int, optional): Random seed for reproducibility.
        
        Returns:
            np.ndarray: A random binary state of shape (n_orbitals,).
        """
        rng = np.random.default_rng(key)
    
        n_up = self.n_fermions_spin_up
        n_down = self.n_fermions_spin_down
        n_half = self.n_orbitals // 2

        # Randomly select positions for spin-up fermions
        up_positions = rng.choice(n_half, size=n_up, replace=False)
        up_state = np.zeros(n_half, dtype=np.int32)
        up_state[up_positions] = 1

        # Randomly select positions for spin-down fermions
        down_positions = rng.choice(n_half, size=n_down, replace=False)
        down_state = np.zeros(n_half, dtype=np.int32)
        down_state[down_positions] = 1

        # Concatenate spin-up and spin-down states
        return from_netket_config_to_quimb_config(np.concatenate([up_state, down_state]))
    
    # def to_quimb_config(self, config):
    #     return from_netket_config_to_quimb_config(config)

class Spin(Hilbert):
    def __init__(self, s, N, total_sz=None):
        """Hilbert space obtained as tensor product of local spin states.

        Args:
           s: Spin at each site. Must be integer or half-integer.
           N: Number of sites (default=1)
           total_sz: If given, constrains the total spin of system to a particular
                value.
        """
        self.s = s
        self.N = N
        self.total_sz = total_sz
        self._size = int(2 * s + 1) ** N
    
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

class Graph:
    def __init__(self):
        self._edges = None
        self._site_index_map = None
    
    def edges(self):
        return self._edges
    
    @property
    def n_edges(self):
        return len(self._edges)
    
    @property
    def site_index_map(self):
        return self._site_index_map

class SquareLatticeGraph(Graph):
    def __init__(self, Lx, Ly, pbc=False, site_index_map=lambda i, j, Lx, Ly: i * Ly + j):
        """Zig-zag ordering"""
        self.Lx = Lx
        self.Ly = Ly
        self.pbc = pbc
        edges = qtn.edges_2d_square(self.Lx, self.Ly, cyclic=self.pbc)
        self._edges = [(site_index_map(*site_i, Lx, Ly), site_index_map(*site_j, Lx, Ly)) for site_i, site_j in edges]
        self._site_index_map = site_index_map


class Hamiltonian:
    def __init__(self, H, hi, graph):
        self._H = H
        self._graph = graph
        self._hilbert = hi
    
    @property
    def graph(self):
        """Graph"""
        return self._graph
    
    @property
    def hilbert(self):
        """Customized Hilbert space"""
        return self._hilbert
    
    @property
    def H(self):
        return self._H
    
    def get_conn(self, sigma):
        raise NotImplementedError


def square_lattice_spinful_Fermi_Hubbard(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=None):
    """Implementation of spinful Fermi-Hubbard model on a square lattice"""
    if pbc:
        raise NotImplementedError("PBC not implemented yet")
    N = Lx * Ly
    if n_fermions_per_spin is None:
        hi = SpinfulFermion(n_orbitals=N, n_fermions=N_f)
    else:
        hi = SpinfulFermion(n_orbitals=N, n_fermions_per_spin=n_fermions_per_spin)
    
    graph = SquareLatticeGraph(Lx, Ly, pbc)

    H = dict()
    for i, j in graph.edges():
        for spin in (1,-1):
            H[(i, j, spin)] = -t

    for i in range(N):
        H[(i,)] = U
        
    return H, hi, graph

class spinful_Fermi_Hubbard_square_lattice_torch(Hamiltonian):
    def __init__(self, Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=None):
        """
        Implementation of spinful Fermi-Hubbard model on a square lattice using torch.
        Args:
            N_f is used to restrict the Hilbert space.
        """
        H, hi, graph = square_lattice_spinful_Fermi_Hubbard(Lx, Ly, t, U, N_f, pbc=pbc, n_fermions_per_spin=n_fermions_per_spin)
        super().__init__(H, hi, graph)

    def get_conn(self, sigma_quimb):
        """
        Return the connected configurations <eta| by the Hamiltonian to the state |sigma>,
        and their corresponding coefficients <eta|H|sigma>.
        """
        sigma = from_quimb_config_to_netket_config(sigma_quimb)
        connected_config_coeff = dict()
        for key, value in self._H.items():
            if len(key) == 3:
                # hopping term
                i0, j0, spin = key
                i = i0 if spin == 1 else i0 + self.hilbert.n_orbitals // 2
                j = j0 if spin == 1 else j0 + self.hilbert.n_orbitals // 2
                # Check if the two sites are different
                if sigma[i] != sigma[j]:
                    # H|sigma> = -t * |eta>
                    eta = sigma.copy()
                    eta[i], eta[j] = sigma[j], sigma[i]
                    eta_quimb0 = from_netket_config_to_quimb_config(eta)
                    eta_quimb = tuple(eta_quimb0)
                    # Calculate the phase correction
                    phase = calc_phase_symmray(from_netket_config_to_quimb_config(sigma), eta_quimb0)
                    if eta_quimb not in connected_config_coeff:
                        connected_config_coeff[eta_quimb] = value*phase
                    else:
                        connected_config_coeff[eta_quimb] += value*phase
            elif len(key) == 1:
                # on-site term
                i = key[0]
                if sigma_quimb[i] == 3:
                    eta_quimb = tuple(sigma_quimb)
                    if eta_quimb not in connected_config_coeff:
                        connected_config_coeff[eta_quimb] = value
                    else:
                        connected_config_coeff[eta_quimb] += value
        
        return do('array', list(connected_config_coeff.keys())), do('array', list(connected_config_coeff.values()))