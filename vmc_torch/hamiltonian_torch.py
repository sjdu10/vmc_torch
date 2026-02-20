import itertools
import quimb.tensor as qtn
import numpy as np
from autoray import do

from vmc_torch.fermion_utils import (
    from_netket_config_to_quimb_config,
    from_quimb_config_to_netket_config,
    calc_phase_symmray,
    calc_phase_netket,
)


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

# ==== Hilbert Space Definitions ====
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

class SpinlessFermion(Hilbert):
    def __init__(self, n_orbitals, n_fermions=None):
        """Hilbert space of spinless fermions.

        Args:
           n_orbitals: Number of orbitals (sites).
           n_fermions: If given, constrains the number of fermions to a particular value.

        """
        self.n_orbitals = n_orbitals
        self.n_fermions = n_fermions
        if self.n_fermions is not None:
            assert 0 <= self.n_fermions <= self.n_orbitals, "n_fermions must be between 0 and n_orbitals"
    
    @property
    def size(self):
        """Number of states in the Hilbert space"""
        if self.n_fermions is not None:
            from math import comb
            return comb(self.n_orbitals, self.n_fermions)
        else:
            return 2 ** self.n_orbitals
    
    def _all_states(self):
        if self.n_fermions is not None:
            return generate_binary_vectors(self.n_orbitals, self.n_fermions)
        else:
            all_states = []
            for n in range(self.n_orbitals + 1):
                all_states.extend(generate_binary_vectors(self.n_orbitals, n))
            return all_states
    
    def all_states(self):
        """Generate all states in the Hilbert space.

        Returns:
            list: All possible configurations of the Hilbert space.
        """
        all_states = do('array', list(self._all_states()))
        return all_states
    
    def random_state(self, key=None):
        """Generate a random state in the Hilbert space.

        Args:
            key (int, optional): Random seed for reproducibility.
        Returns:
            np.ndarray: A random binary state of shape (n_orbitals,).
        """
        rng = np.random.default_rng(key)
        n = self.n_orbitals
        if self.n_fermions is not None:
            m = self.n_fermions
        else:
            # Randomly choose number of fermions
            m = rng.integers(0, n + 1)
        
        # Randomly select positions for fermions
        positions = rng.choice(n, size=m, replace=False)
        state = np.zeros(n, dtype=np.int32)
        state[positions] = 1
        return state

class SpinfulFermion(Hilbert):
    def __init__(self, n_orbitals, n_fermions=None, n_fermions_per_spin=None, no_u1_symmetry=False):
        """
        Spin-1/2 fermionic Hilbert space with n_orbitals orbitals per spin.

        Args:
            n_orbitals (int): Number of orbitals (sites) for each spin species.
            n_fermions (int, optional): Total number of fermions (default: None).
            n_fermions_per_spin (tuple, optional): Number of fermions per spin (n_up, n_down) (default: None).
            no_u1_symmetry (bool, optional): If True, no U1 symmetry on the Hilbert space (default: False).

        Configuration structure: [nu1,...,nuN, nd1,...,ndN]
        where nu1,...,nuN are the spin-up fermions and nd1,...,ndN are the spin-down fermions.
        The total number of orbitals is n_up + n_down.
        
        """
        self.n_orbitals = n_orbitals*2  # Total number of orbitals (spin up + spin down)
        self.no_u1_symmetry = no_u1_symmetry # By default the Hilbert space has even Z2 symmetry
        # If fermion numbers are provided, set them.
        self.n_fermions = n_fermions if n_fermions is not None else n_fermions_per_spin[0] + n_fermions_per_spin[1]
        self.n_fermions_spin_up = n_fermions_per_spin[0] if n_fermions_per_spin is not None else None
        self.n_fermions_spin_down = n_fermions_per_spin[1] if n_fermions_per_spin is not None else None
        self.n_fermions_per_spin = n_fermions_per_spin if n_fermions_per_spin is not None else None
    
    @property
    def size(self):
        """Number of states in the Hilbert space"""
        # Use combinatorial formula to calculate size
        from math import comb
        if self.no_u1_symmetry:
            size = 0
            for n in range(0, self.n_orbitals+1, 2):
                size += comb(self.n_orbitals, n)
            return size
        
        if self.n_fermions_per_spin is not None:
            size_up = comb(self.n_orbitals // 2, self.n_fermions_spin_up)
            size_down = comb(self.n_orbitals // 2, self.n_fermions_spin_down)
            return size_up * size_down
        else:
            return comb(self.n_orbitals, self.n_fermions)
    
    def _all_states(self):
        if self.no_u1_symmetry:
            all_states = []
            for n in range(0, self.n_orbitals+1, 2):
                all_states.extend(generate_binary_vectors(self.n_orbitals, n))
            return all_states

        if self.n_fermions_per_spin is not None:
            spin_up_states = generate_binary_vectors(self.n_orbitals // 2, self.n_fermions_spin_up)
            spin_down_states = generate_binary_vectors(self.n_orbitals // 2, self.n_fermions_spin_down)
            return (up + down for up, down in itertools.product(spin_up_states, spin_down_states))
        else:
            return generate_binary_vectors(self.n_orbitals, self.n_fermions)
    
    def all_states(self):
        """Generate all states in the Hilbert space.
    
        Returns:
            list: All possible configurations of the Hilbert space.
        """
        all_states_netket = do('array', list(self._all_states()))
        all_states = from_netket_config_to_quimb_config(all_states_netket)
        return all_states
    
    def random_state(self, key=None):
        """Generate a random state in the Hilbert space.
    
        Args:
            key (int, optional): Random seed for reproducibility.
        
        Returns:
            np.ndarray: A random binary state of shape (n_orbitals,).
        """
        rng = np.random.default_rng(key)
        if self.no_u1_symmetry:
            n = self.n_orbitals
            # Randomly choose an even number of fermions
            m = rng.choice(np.arange(0, n+1, 2))
            # Randomly select positions for fermions
            positions = rng.choice(n, size=m, replace=False)
            state = np.zeros(n, dtype=np.int32)
            state[positions] = 1
            return from_netket_config_to_quimb_config(state)

        if self.n_fermions_per_spin is not None:
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

        else:
            n = self.n_orbitals
            m = self.n_fermions

            # Randomly select positions for fermions
            positions = rng.choice(n, size=m, replace=False)
            state = np.zeros(n, dtype=np.int32)
            state[positions] = 1

            return from_netket_config_to_quimb_config(state)


class Spin(Hilbert):
    def __init__(self, s, N, total_sz=None):
        """Hilbert space obtained as tensor product of local spin states.

        Args:
           s: Spin at each site. Must be integer or half-integer. Currently only supports s=1/2.
           N: Number of sites (default=1)
           total_sz: If given, constrains the total spin of system to a particular
                value.
        """
        self.s = s
        assert float(s)==0.5, "Currently only supports s=1/2 for Spin Hilbert space"
        self.N = N
        self.total_sz = total_sz
        if self.total_sz is not None:
            assert isinstance(self.total_sz, int), "total_sz must be an integer for spin-1/2 sites"
        self._size = int(2 * s + 1) ** N # total number of states in the Hilbert space without total_sz constraint
    
    @property
    def size(self):
        """Number of states in the Hilbert space"""
        if self.total_sz is not None:
            su = int(self.N/2 + self.total_sz/self.s/2)
            from math import comb
            return comb(self.N, su)
        else:
            return self._size
    
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
        
    def all_states(self):
        """
        Generate all states in the Hilbert space.
        Each state is a configuration of spins at each site.
        """
        # Generate all possible configurations for a single site
        single_site_states = np.linspace(0, 1, int(2 * self.s + 1))
        assert len(single_site_states) == 2, 'Currently only supports s=1/2 for Spin Hilbert space, len(single_site_states) should be 2'
        
        if self.total_sz is not None:
            su = int(self.N/2 + self.total_sz/self.s/2)
            all_states = generate_binary_vectors(self.N, su)
            return do('array', list(all_states))
            

        else:
            # Generate all combinations of single-site states for N sites
            all_states = itertools.product(single_site_states, repeat=self.N)            
            # Convert to numpy array
            return do('array', list(all_states))
        
    def random_state(self, key=None):
        """
        Generate a random state in the Hilbert space.
        
        Args:
            key (int, optional): Random seed for reproducibility.
        
        Returns:
            np.ndarray: A random configuration of spins in the Hilbert space.
        """
        rng = np.random.default_rng(key)
        # Generate a random state by sampling from the possible spin values
        single_site_states = np.linspace(-self.s, self.s, int(2 * self.s + 1))
        assert len(single_site_states) == 2, 'Currently only supports s=1/2 for Spin Hilbert space, len(single_site_states) should be 2'

        if self.total_sz is not None:
            # If total_sz is specified, we need to sample a state with the correct total spin
            su = int(self.N/2 + self.total_sz/self.s/2)
            su_positions = rng.choice(self.N, size=su, replace=False)
            random_state = np.zeros(self.N, dtype=np.int32)
            random_state[su_positions] = 1

        else:
            # Otherwise, just generate a random state from the available single-site states
            random_state = rng.choice(np.linspace(0, 1, int(2 * self.s + 1)), size=self.N)
            
        return do('array', random_state)
    

# ==== Graph Definitions ====
class Graph:
    def __init__(self, edges=None):
        self._edges = edges
        self._site_index_map = None
        self.n_nodes = None
    
    def edges(self):
        return self._edges
    
    @property
    def N(self):
        """Number of nodes in the graph"""
        return self.n_nodes
    
    @property
    def n_edges(self):
        return len(self._edges) if self._edges is not None else 0
    
    @property
    def site_index_map(self):
        return self._site_index_map

class SquareLattice(Graph):
    def __init__(self, Lx, Ly, pbc=False, site_index_map=lambda i, j, Lx, Ly: i * Ly + j):
        """Zig-zag ordering, nearest neighbor edges"""
        self.Lx = Lx
        self.Ly = Ly
        self.n_nodes = Lx * Ly
        self.pbc = pbc
        edges = qtn.edges_2d_square(self.Lx, self.Ly, cyclic=self.pbc)
        self._edges = [(site_index_map(*site_i, Lx, Ly), site_index_map(*site_j, Lx, Ly)) for site_i, site_j in edges]
        self._site_index_map = site_index_map

        # used for reusable samplers
        self.row_edges = {} # edges in the same row
        self.col_edges = {} # edges in the same column
        for (i, j) in self.edges():
            i_coo = self.from_index_to_2dcoo(i)
            j_coo = self.from_index_to_2dcoo(j)
            if i_coo[0] == j_coo[0]:
                if i_coo[0] not in self.row_edges:
                    self.row_edges[i_coo[0]] = []
                self.row_edges[i_coo[0]].append((i, j))
            elif i_coo[1] == j_coo[1]:
                if i_coo[1] not in self.col_edges:
                    self.col_edges[i_coo[1]] = []
                self.col_edges[i_coo[1]].append((i, j))
                
    def from_index_to_2dcoo(self, index):
        """Convert a 1D zig-zag ordering index to 2D coordinates"""
        return index // self.Ly, index % self.Ly

class Chain(Graph):
    def __init__(self, L, pbc=False):
        self.L = L
        self.pbc = pbc
        edges = qtn.edges_1d_chain(L, cyclic=pbc)
        self._edges = edges
        self.n_nodes = L


# ==== Hamiltonian Definitions ====
class Operator:
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
    
    def get_conn(self, sigma):
        raise NotImplementedError

class Hamiltonian(Operator):
    def __init__(self, H, hi, graph):
        super().__init__(H, hi, graph)
    
    def to_dense(self):
        """Convert the Hamiltonian to a dense matrix representation."""
        size = self.hilbert.size
        H_matrix = np.zeros((size, size), dtype=np.float64)
        all_states = self.hilbert.all_states()
        state_index_map = {tuple(state): idx for idx, state in enumerate(all_states)}
        
        for idx, sigma in enumerate(all_states):
            connected_configs, coeffs = self.get_conn(sigma)
            for eta, coeff in zip(connected_configs, coeffs):
                eta_tuple = tuple(eta)
                jdx = state_index_map[eta_tuple]
                H_matrix[jdx, idx] += coeff
                
        return H_matrix
    
    @property
    def H(self):
        return self._H

def chain_spinless_Fermi_Hubbard(L, t, V, N_f, pbc=False):
    """Implementation of spinless free fermion model on a 1D chain"""
    if pbc:
        raise NotImplementedError("PBC not implemented yet")
    N = L
    hi = SpinlessFermion(n_orbitals=N, n_fermions=N_f)
    
    graph = Chain(L, pbc)

    H = dict()
    for i, j in graph.edges():
        H[(i, j, 't')] = -t
        H[(i, j, 'V')] = V

    return H, hi, graph

class spinless_Fermi_Hubbard_chain_torch(Hamiltonian):
    def __init__(self, L, t, V, N_f, pbc=False):
        """
        Implementation of spinless free fermion model on a chain using torch.
        Args:
            N_f is used to restrict the Hilbert space.
        """
        H, hi, graph = chain_spinless_Fermi_Hubbard(L, t, V, N_f, pbc)
        super().__init__(H, hi, graph)
    def get_conn(self, sigma_quimb):
        """
        Return the connected configurations <eta| by the Hamiltonian to the state |sigma>,
        and their corresponding coefficients <eta|H|sigma>.
        """
        sigma = np.array(sigma_quimb)
        connected_config_coeff = dict()
        for key, value in self._H.items():
            i, j, term_type = key
            if term_type == 't':
                # hopping term
                if sigma[i] != sigma[j]:
                    # H|sigma> = -t * |eta>
                    eta = sigma.copy()
                    eta[i], eta[j] = sigma[j], sigma[i]
                    eta_quimb = tuple(eta)
                    # Calculate the phase correction
                    phase = (-1)**(sum(sigma[min(i,j)+1:max(i,j)]))  # Jordan-Wigner phase
                    if eta_quimb not in connected_config_coeff:
                        connected_config_coeff[eta_quimb] = value*phase
                    else:
                        connected_config_coeff[eta_quimb] += value*phase
            elif term_type == 'V':
                # interaction term
                if sigma[i] == 1 and sigma[j] == 1:
                    eta_quimb = tuple(sigma)
                    if eta_quimb not in connected_config_coeff:
                        connected_config_coeff[eta_quimb] = value
                    else:
                        connected_config_coeff[eta_quimb] += value
        
        return do('array', list(connected_config_coeff.keys())), do('array', list(connected_config_coeff.values()))

def square_lattice_spinless_Fermi_Hubbard(Lx, Ly, t, V, mu, N_f, pbc=False):
    """Implementation of spinless Fermi-Hubbard model on a square lattice"""
    if pbc:
        raise NotImplementedError("PBC not implemented yet")
    N = Lx * Ly
    hi = SpinlessFermion(n_orbitals=N, n_fermions=N_f)
    
    graph = SquareLattice(Lx, Ly, pbc)

    H = dict()
    for i, j in graph.edges():
        H[(i, j, 't')] = -t
        H[(i, j, 'V')] = V
    for i in range(N):
        H[(i,)] = -mu
    
    return H, hi, graph

class spinless_Fermi_Hubbard_square_lattice_torch(Hamiltonian):
    def __init__(self, Lx, Ly, t=1.0, V=0.0, mu=0.0, N_f=None, pbc=False):
        """
        Implementation of spinless Fermi-Hubbard model on a square lattice using torch.
        Args:
            N_f is used to restrict the Hilbert space.
        """
        H, hi, graph = square_lattice_spinless_Fermi_Hubbard(Lx, Ly, t, V, mu, N_f, pbc)
        super().__init__(H, hi, graph)
    def get_conn(self, sigma_quimb):
        """
        Return the connected configurations <eta| by the Hamiltonian to the state |sigma>,
        and their corresponding coefficients <eta|H|sigma>.
        """
        sigma = np.array(sigma_quimb)
        connected_config_coeff = dict()
        for key, value in self._H.items():
            if len(key) == 3:
                i, j, term_type = key
                if term_type == 't':
                    # hopping term
                    if sigma[i] != sigma[j]:
                        # H|sigma> = -t * |eta>
                        eta = sigma.copy()
                        eta[i], eta[j] = sigma[j], sigma[i]
                        eta_quimb = tuple(eta)
                        # Calculate the phase correction
                        phase = (-1)**(sum(sigma[min(i,j)+1:max(i,j)]))  # Jordan-Wigner phase
                        if eta_quimb not in connected_config_coeff:
                            connected_config_coeff[eta_quimb] = value*phase
                        else:
                            connected_config_coeff[eta_quimb] += value*phase
                elif term_type == 'V':
                    # interaction term
                    if sigma[i] == 1 and sigma[j] == 1:
                        eta_quimb = tuple(sigma)
                        if eta_quimb not in connected_config_coeff:
                            connected_config_coeff[eta_quimb] = value
                        else:
                            connected_config_coeff[eta_quimb] += value
            elif len(key) == 1:
                # on-site term
                i = key[0]
                if sigma_quimb[i] == 1:
                    eta_quimb = tuple(sigma)
                    if eta_quimb not in connected_config_coeff:
                        connected_config_coeff[eta_quimb] = value * sigma[i]
                    else:
                        connected_config_coeff[eta_quimb] += value * sigma[i]
        
        return do('array', list(connected_config_coeff.keys())), do('array', list(connected_config_coeff.values()))



def chain_spinful_Fermi_Hubbard(L, t, U, N_f, pbc=False, n_fermions_per_spin=None, no_u1_symmetry=False):
    """Implementation of spinful Fermi-Hubbard model on a 1D chain"""
    if pbc:
        raise NotImplementedError("PBC not implemented yet")
    N = L
    if n_fermions_per_spin is None:
        hi = SpinfulFermion(n_orbitals=N, n_fermions=N_f, no_u1_symmetry=no_u1_symmetry)
    else:
        hi = SpinfulFermion(n_orbitals=N, n_fermions_per_spin=n_fermions_per_spin, no_u1_symmetry=no_u1_symmetry)
    
    graph = Chain(L, pbc)

    H = dict()
    for i, j in graph.edges():
        for spin in (1,-1):
            H[(i, j, spin)] = -t

    for i in range(N):
        H[(i,)] = U
    
    return H, hi, graph

class spinful_Fermi_Hubbard_chain_torch(Hamiltonian):
    def __init__(self, L, t, U, N_f, pbc=False, n_fermions_per_spin=None, no_u1_symmetry=False):
        """
        Implementation of spinful Fermi-Hubbard model on a square lattice using torch.
        Args:
            N_f is used to restrict the Hilbert space.
        """
        H, hi, graph = chain_spinful_Fermi_Hubbard(L, t, U, N_f, pbc, n_fermions_per_spin, no_u1_symmetry=no_u1_symmetry)
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



def square_lattice_spinful_Fermi_Hubbard(Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=None, no_u1_symmetry=False):
    """Implementation of spinful Fermi-Hubbard model on a square lattice"""
    if pbc:
        raise NotImplementedError("PBC not implemented yet")
    N = Lx * Ly
    if n_fermions_per_spin is None:
        hi = SpinfulFermion(n_orbitals=N, n_fermions=N_f, no_u1_symmetry=no_u1_symmetry)
    else:
        hi = SpinfulFermion(n_orbitals=N, n_fermions_per_spin=n_fermions_per_spin, no_u1_symmetry=no_u1_symmetry)
    
    graph = SquareLattice(Lx, Ly, pbc)

    H = dict()
    for i, j in graph.edges():
        for spin in (1,-1):
            H[(i, j, spin)] = -t

    for i in range(N):
        H[(i,)] = U
        
    return H, hi, graph

class spinful_Fermi_Hubbard_square_lattice_torch(Hamiltonian):
    def __init__(self, Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=None, no_u1_symmetry=False, gpu=False):
        """
        Implementation of spinful Fermi-Hubbard model on a square lattice using torch.
        Args:
            N_f is used to restrict the Hilbert space.
        """
        H, hi, graph = square_lattice_spinful_Fermi_Hubbard(
            Lx,
            Ly,
            t,
            U,
            N_f,
            pbc=pbc,
            n_fermions_per_spin=n_fermions_per_spin,
            no_u1_symmetry=no_u1_symmetry,
        )
        super().__init__(H, hi, graph)
        self.gpu = gpu

    def precompute_hops_gpu(self, device):
        """Call once after init to precompute hop metadata for GPU-batched get_conn."""
        import torch
        N = self.hilbert.n_orbitals // 2  # N_sites
        hop_list = []   # (i_netket, j_netket, coeff, p_symmray, q_symmray)
        diag_list = []  # (site_quimb, coeff)
        for key, value in self._H.items():
            if len(key) == 3:
                i0, j0, spin = key
                # netket orbital indices
                i_net = i0 if spin == 1 else i0 + N
                j_net = j0 if spin == 1 else j0 + N
                # symmray positions: up↔2*site+1, down↔2*site+0
                p_sym = 2 * i0 + (1 if spin == 1 else 0)
                q_sym = 2 * j0 + (1 if spin == 1 else 0)
                if p_sym > q_sym:
                    p_sym, q_sym = q_sym, p_sym
                hop_list.append((i_net, j_net, value, p_sym, q_sym))
            elif len(key) == 1:
                diag_list.append((key[0], value))
        self._hop_list = hop_list
        self._diag_list = diag_list
        self._gpu_device = device

    def get_conn_batch_gpu(self, fxs):
        """
        Batched get_conn on GPU. No Python loop over samples.

        Args:
            fxs: (B, N_sites) int64 GPU tensor (quimb encoding)
        Returns:
            conn_etas:   (total_conn, N_sites) int64 GPU tensor
            conn_coeffs: (total_conn,) float64 GPU tensor
            batch_ids:   (total_conn,) int64 GPU tensor
        """
        import torch
        B, N = fxs.shape
        device = fxs.device

        # Build netket representation: (B, 2*N)
        spin_up = ((fxs == 2) | (fxs == 3)).long()    # (B, N)
        spin_down = ((fxs == 1) | (fxs == 3)).long()  # (B, N)
        netket = torch.cat([spin_up, spin_down], dim=1)  # (B, 2*N)

        # Build symmray representation: [d0, u0, d1, u1, ...]  (B, 2*N)
        symmray = torch.stack(
            [spin_down, spin_up], dim=-1
        ).reshape(B, 2 * N)  # (B, 2*N)

        all_etas, all_coeffs, all_bids = [], [], []

        # --- Hopping terms ---
        for i_net, j_net, coeff, p_sym, q_sym in self._hop_list:
            valid = netket[:, i_net] != netket[:, j_net]  # (B,) bool
            if not valid.any():
                continue
            idx = valid.nonzero(as_tuple=True)[0]  # (k,)

            # Swap i_net and j_net in the valid netket configs
            new_netket = netket[idx].clone()       # (k, 2*N)
            tmp = new_netket[:, i_net].clone()
            new_netket[:, i_net] = new_netket[:, j_net]
            new_netket[:, j_net] = tmp

            # Convert back: netket → quimb
            su = new_netket[:, :N]
            sd = new_netket[:, N:]
            new_fxs = su * 2 + sd  # (k, N)

            # Phase: count symmray occupancies in (p_sym, q_sym) open interval
            between = symmray[idx, p_sym + 1:q_sym].sum(dim=-1) % 2  # (k,)
            phases = 1.0 - 2.0 * between.double()  # +1 or -1

            all_etas.append(new_fxs)
            all_coeffs.append(phases * coeff)
            all_bids.append(idx)

        # --- Diagonal (on-site U) terms ---
        for site, coeff in self._diag_list:
            valid = fxs[:, site] == 3  # (B,) bool
            if not valid.any():
                continue
            idx = valid.nonzero(as_tuple=True)[0]
            all_etas.append(fxs[idx])
            all_coeffs.append(
                torch.full(
                    (len(idx),), coeff,
                    device=device, dtype=torch.float64,
                )
            )
            all_bids.append(idx)

        conn_etas = torch.cat(all_etas, dim=0)
        conn_coeffs = torch.cat(all_coeffs, dim=0)
        batch_ids = torch.cat(all_bids, dim=0)
        return conn_etas, conn_coeffs, batch_ids

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
                    eta_quimb = tuple(np.array(eta_quimb0))
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
                    if self.gpu:
                        eta_quimb = tuple(sigma_quimb.cpu().numpy())
                    else:
                        eta_quimb = tuple(np.array(sigma_quimb))
                    if eta_quimb not in connected_config_coeff:
                        connected_config_coeff[eta_quimb] = value
                    else:
                        connected_config_coeff[eta_quimb] += value
        
        return do('array', list(connected_config_coeff.keys())), do('array', list(connected_config_coeff.values()))

# Customized Hamiltonian elements to match with Ao's initial SD's definition
class spinful_Fermi_Hubbard_square_lattice_torch_Ao(spinful_Fermi_Hubbard_square_lattice_torch):
    def __init__(self, Lx, Ly, t, U, N_f, pbc=False, n_fermions_per_spin=None, no_u1_symmetry=False):
        super().__init__(Lx, Ly, t, U, N_f, pbc=pbc, n_fermions_per_spin=n_fermions_per_spin, no_u1_symmetry=no_u1_symmetry)

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
                    phase = calc_phase_netket(from_netket_config_to_quimb_config(sigma), eta_quimb0)
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
    
# ---- Spin Heisenberg model ----

def square_lattice_spin_Heisenberg(Lx, Ly, J, pbc=False, total_sz=None):
    # Build square lattice with nearest neighbor edges
    N = Lx * Ly
    hi = Spin(s=1/2, N=N, total_sz=total_sz)  # Spin-1/2 Hilbert space
    graph = SquareLattice(Lx, Ly, pbc)
    # Heisenberg with coupling J for nearest neighbors
    H = dict()
    for i, j in graph.edges():
        # Add the Heisenberg term for the edge (i, j)
        # The Heisenberg Hamiltonian is J * (S_i . S_j) = J * (S_i^x S_j^x + S_i^y S_j^y + S_i^z S_j^z)
        # H = \sum_<i,j> 0.5J * (S_i^+ S_j^- + S_i^- S_j^+) + J * S_i^z S_j^z
        # Note S = 1/2\sigma

        if type(J) is dict:
            # If J is a dictionary, use the specific coupling for the edge (i,j)
            J_value = J.get((i, j), 0)
            H[(i, j)] = J_value
        else:
            H[(i, j)] = J
    
    return H, hi, graph

class spin_Heisenberg_square_lattice_torch(Hamiltonian):
    def __init__(self, Lx, Ly, J, pbc=False, total_sz=None):
        """
        Implementation of spin-1/2 Heisenberg model on a square lattice using torch.
        Args:
            J: Coupling constant (can be a dict for edge-specific couplings)
            total_sz: If given, constrains the total spin of system to a particular value.
        """
        H, hi, graph = square_lattice_spin_Heisenberg(Lx, Ly, J, pbc=pbc, total_sz=total_sz)
        super().__init__(H, hi, graph)
    
    def to_dense(self):
        """Convert the Hamiltonian to a dense matrix representation."""
        size = self.hilbert.size
        H_matrix = np.zeros((size, size), dtype=np.float64)
        all_states = self.hilbert.all_states()
        state_index_map = {tuple(state): idx for idx, state in enumerate(all_states)}
        
        for idx, sigma in enumerate(all_states):
            connected_configs, coeffs = self.get_conn(sigma)
            for eta, coeff in zip(connected_configs, coeffs):
                eta_tuple = tuple(eta)
                jdx = state_index_map[eta_tuple]
                H_matrix[jdx, idx] += coeff

        return H_matrix
    
    def get_conn(self, sigma_quimb):
        """
        Return the connected configurations <eta| by the Hamiltonian to the state |sigma>,
        and their corresponding coefficients <eta|H|sigma>.
        """
        connected_config_coeff = dict()
        sigma = np.array(sigma_quimb)
        for key, value in self._H.items():
            i, j = key
            J = value
            if sigma[i] != sigma[j]:
                # Hopping term

                # H|sigma> = 0.5J * |eta>
                eta = sigma.copy()
                eta[i], eta[j] = sigma[j], sigma[i]
                if tuple(eta) not in connected_config_coeff:
                    # Calculate the phase correction (not needed for Heisenberg)
                    connected_config_coeff[tuple(eta)] = 0.5 * J
                else:
                    # Accumulate the coefficients for degenerate states
                    connected_config_coeff[tuple(eta)] += 0.5 * J
            
            eta0 = sigma.copy()
            if tuple(eta0) not in connected_config_coeff:
                # Handle the case of on-site term, which is J * S_i^z S_j^z
                # For Heisenberg, this is already included in the coupling above
                connected_config_coeff[tuple(eta0)] = 0.25*J*(-1)**(abs(sigma[i]-sigma[j]))
            else:
                # Accumulate the coefficients for degenerate states
                connected_config_coeff[tuple(eta0)] += 0.25*J*(-1)**(abs(sigma[i]-sigma[j]))

        return do('array', list(connected_config_coeff.keys())), do('array', list(connected_config_coeff.values()))


def chain_spin_Heisenberg(L, J, pbc=False, total_sz=None):
    # Build chain with nearest neighbor edges
    N = L
    hi = Spin(s=1/2, N=N, total_sz=total_sz)  # Spin-1/2 Hilbert space
    graph = Chain(L, pbc)
    # Heisenberg with coupling J for nearest neighbors
    H = dict()
    for i, j in graph.edges():
        # Add the Heisenberg term for the edge (i, j)
        # The Heisenberg Hamiltonian is J * (S_i . S_j) = J * (S_i^x S_j^x + S_i^y S_j^y + S_i^z S_j^z)
        # H = \sum_<i,j> 0.5J * (S_i^+ S_j^- + S_i^- S_j^+) + J * S_i^z S_j^z
        # Note S = 1/2\sigma
        if type(J) is dict:
            # If J is a dictionary, use the specific coupling for the edge (i,j)
            J_value = J.get((i, j), 0)
            H[(i, j)] = J_value
        else:
            H[(i, j)] = J
    
    return H, hi, graph

class spin_Heisenberg_chain_torch(Hamiltonian):
    def __init__(self, L, J, pbc=False, total_sz=None):
        """
        Implementation of spin-1/2 Heisenberg model on a chain using torch.
        Args:
            J: Coupling constant (can be a dict for edge-specific couplings)
            total_sz: If given, constrains the total spin of system to a particular value.
        """
        H, hi, graph = chain_spin_Heisenberg(L, J, pbc=pbc, total_sz=total_sz)
        super().__init__(H, hi, graph)
    
    def to_dense(self):
        """Convert the Hamiltonian to a dense matrix representation."""
        size = self.hilbert.size
        H_matrix = np.zeros((size, size), dtype=np.float64)
        all_states = self.hilbert.all_states()
        state_index_map = {tuple(state): idx for idx, state in enumerate(all_states)}
        
        for idx, sigma in enumerate(all_states):
            connected_configs, coeffs = self.get_conn(sigma)
            for eta, coeff in zip(connected_configs, coeffs):
                eta_tuple = tuple(eta)
                jdx = state_index_map[eta_tuple]
                H_matrix[jdx, idx] += coeff
                
        return H_matrix
    
    def get_conn(self, sigma_quimb):
        """
        Return the connected configurations <eta| by the Hamiltonian to the state |sigma>,
        and their corresponding coefficients <eta|H|sigma>.
        """
        connected_config_coeff = dict()
        sigma = np.array(sigma_quimb)
        for key, value in self._H.items():
            i, j = key
            J = value
            if sigma[i] != sigma[j]:
                # Hopping term

                # H|sigma> = 0.5J * |eta>
                eta = sigma.copy()
                eta[i], eta[j] = sigma[j], sigma[i]
                if tuple(eta) not in connected_config_coeff:
                    # Calculate the phase correction (not needed for Heisenberg)
                    connected_config_coeff[tuple(eta)] = 0.5 * J
                else:
                    # Accumulate the coefficients for degenerate states
                    connected_config_coeff[tuple(eta)] += 0.5 * J
            
            eta0 = sigma.copy()
            if tuple(eta0) not in connected_config_coeff:
                # Handle the case of on-site term, which is J * S_i^z S_j^z
                # For Heisenberg, this is already included in the coupling above
                connected_config_coeff[tuple(eta0)] = 0.25*J*(-1)**(abs(sigma[i]-sigma[j]))
            else:
                # Accumulate the coefficients for degenerate states
                connected_config_coeff[tuple(eta0)] += 0.25*J*(-1)**(abs(sigma[i]-sigma[j]))

        return do('array', list(connected_config_coeff.keys())), do('array', list(connected_config_coeff.values()))


def chain_spin_transverse_Ising(L, J, h, pbc=False, total_sz=None):
    # Build chain with nearest neighbor edges
    N = L
    hi = Spin(s=1/2, N=N, total_sz=total_sz)  # Spin-1/2 Hilbert space
    graph = Chain(L, pbc)
    # Ising with coupling J for nearest neighbors
    H = dict()
    for i, j in graph.edges():
        # Add the Ising term for the edge (i, j)
        # The transverse field Ising Hamiltonian is J * S_i^z S_j^z + h * S_i^x
        # H = \sum_<i,j> J * S_i^z S_j^z + h * S_i^x
        # Note S = 1/2\sigma
        if type(J) is dict:
            # If J is a dictionary, use the specific coupling for the edge (i,j)
            J_value = J.get((i, j), 0)
            H[(i, j), 'zz'] = J_value
        else:
            H[(i, j), 'zz'] = J
    for i in range(N):
        H[(i, 'x')] = h
    
    return H, hi, graph

class spin_transverse_Ising_chain_torch(Hamiltonian):
    def __init__(self, L, J, h, pbc=False, total_sz=None):
        """
        Implementation of spin-1/2 transverse field Ising model on a chain using torch.
        Args:
            J: Coupling constant (can be a dict for edge-specific couplings)
            h: Transverse field strength
            pbc: Whether to use periodic boundary conditions
            total_sz: If given, constrains the total spin of system to a particular value.
        """
        H, hi, graph = chain_spin_transverse_Ising(L, J, h, pbc=pbc, total_sz=total_sz)
        super().__init__(H, hi, graph)

    def get_conn(self, sigma_quimb):
        """
        Return the connected configurations <eta| by the Hamiltonian to the state |sigma>,
        and their corresponding coefficients <eta|H|sigma>.
        """
        connected_config_coeff = dict()
        sigma = np.array(sigma_quimb)
        for key, value in self._H.items():
            if len(key) == 2 and key[1] == 'zz':
                # ZZ interaction term
                i, j = key[0]
                J = value
                eta0 = sigma.copy()
                if tuple(eta0) not in connected_config_coeff:
                    connected_config_coeff[tuple(eta0)] = 0.25*J*(-1)**(abs(sigma[i]-sigma[j]))
                else:
                    connected_config_coeff[tuple(eta0)] += 0.25*J*(-1)**(abs(sigma[i]-sigma[j]))
            elif len(key) == 2 and key[1] == 'x':
                # Transverse field term
                i = key[0]
                h = value
                # H|sigma> = h * |eta>
                eta = sigma.copy()
                eta[i] = 1 - sigma[i]  # Flip the spin at site i
                if tuple(eta) not in connected_config_coeff:
                    connected_config_coeff[tuple(eta)] = 0.5*h
                else:
                    connected_config_coeff[tuple(eta)] += 0.5*h

        return do('array', list(connected_config_coeff.keys())), do('array', list(connected_config_coeff.values()))
    

def square_lattice_spin_transverse_Ising(Lx, Ly, J, h, pbc=False, total_sz=None):
    # Build square lattice with nearest neighbor edges
    N = Lx * Ly
    hi = Spin(s=1/2, N=N, total_sz=total_sz)  # Spin-1/2 Hilbert space
    graph = SquareLattice(Lx, Ly, pbc)
    # Ising with coupling J for nearest neighbors
    H = dict()
    for i, j in graph.edges():
        # Add the Ising term for the edge (i, j)
        # The transverse field Ising Hamiltonian is J * S_i^z S_j^z + h * S_i^x
        # H = \sum_<i,j> J * S_i^z S_j^z + h * S_i^x
        # Note S = 1/2\sigma
        if type(J) is dict:
            # If J is a dictionary, use the specific coupling for the edge (i,j)
            J_value = J.get((i, j), 0)
            H[(i, j), 'zz'] = J_value
        else:
            H[(i, j), 'zz'] = J
    for i in range(N):
        H[(i, 'x')] = h
    
    return H, hi, graph

class spin_transverse_Ising_square_lattice_torch(Hamiltonian):
    def __init__(self, Lx, Ly, J, h, pbc=False, total_sz=None):
        """
        Implementation of spin-1/2 transverse field Ising model on a square lattice using torch.
        Args:
            J: Coupling constant (can be a dict for edge-specific couplings)
            h: Transverse field strength
            pbc: Whether to use periodic boundary conditions
            total_sz: If given, constrains the total spin of system to a particular value.
        """
        H, hi, graph = square_lattice_spin_transverse_Ising(Lx, Ly, J, h, pbc=pbc, total_sz=total_sz)
        super().__init__(H, hi, graph)
    
    def to_dense(self):
        """Convert the Hamiltonian to a dense matrix representation."""
        size = self.hilbert.size
        H_matrix = np.zeros((size, size), dtype=np.float64)
        all_states = self.hilbert.all_states()
        state_index_map = {tuple(state): idx for idx, state in enumerate(all_states)}
        
        for idx, sigma in enumerate(all_states):
            connected_configs, coeffs = self.get_conn(sigma)
            for eta, coeff in zip(connected_configs, coeffs):
                eta_tuple = tuple(eta)
                jdx = state_index_map[eta_tuple]
                H_matrix[jdx, idx] += coeff
                
        return H_matrix

    def get_conn(self, sigma_quimb):
        """
        Return the connected configurations <eta| by the Hamiltonian to the state |sigma>,
        and their corresponding coefficients <eta|H|sigma>.
        """
        connected_config_coeff = dict()
        sigma = np.array(sigma_quimb)
        for key, value in self._H.items():
            if len(key) == 2 and key[1] == 'zz':
                # ZZ interaction term
                i, j = key[0]
                J = value
                eta0 = sigma.copy()
                if tuple(eta0) not in connected_config_coeff:
                    connected_config_coeff[tuple(eta0)] = 0.25*J*(-1)**(abs(sigma[i]-sigma[j]))
                else:
                    connected_config_coeff[tuple(eta0)] += 0.25*J*(-1)**(abs(sigma[i]-sigma[j]))
            elif len(key) == 2 and key[1] == 'x':
                # Transverse field term
                i = key[0]
                h = value
                # H|sigma> = h * |eta>
                eta = sigma.copy()
                eta[i] = 1 - sigma[i]  # Flip the spin at site i
                if tuple(eta) not in connected_config_coeff:
                    connected_config_coeff[tuple(eta)] = 0.5*h
                else:
                    connected_config_coeff[tuple(eta)] += 0.5*h

        return do('array', list(connected_config_coeff.keys())), do('array', list(connected_config_coeff.values()))