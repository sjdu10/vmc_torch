from .hamiltonian_torch import Operator, SquareLattice, SpinfulFermion
from autoray import do


class charge_density_square_lattice(Operator):
    """Charge density operator n_i = n_{i,up} + n_{i,down} on site i."""

    def __init__(self, Lx, Ly, N_f, n_fermions_per_spin=None):

        graph = SquareLattice(Lx, Ly)
        N = Lx * Ly

        if n_fermions_per_spin is None:
            hi = SpinfulFermion(n_orbitals=N, n_fermions=N_f)
        else:
            hi = SpinfulFermion(n_orbitals=N, n_fermions_per_spin=n_fermions_per_spin)
        
        O = dict()
        super().__init__(O, hi, graph)
    
    def get_conn(self, sigma_quimb):

        ind_charge_map = {0:0, 1:1, 2:1, 3:2} # quimb to netket mapping

        charge_config = do('array',[ind_charge_map[int(s)] for s in sigma_quimb])

        return sigma_quimb, charge_config



class spin_density_square_lattice(Operator):
    """Spin density operator on site i."""

    def __init__(self, Lx, Ly, N_f, n_fermions_per_spin=None):

        graph = SquareLattice(Lx, Ly)
        N = Lx * Ly

        if n_fermions_per_spin is None:
            hi = SpinfulFermion(n_orbitals=N, n_fermions=N_f)
        else:
            hi = SpinfulFermion(n_orbitals=N, n_fermions_per_spin=n_fermions_per_spin)
        
        O = dict()
        super().__init__(O, hi, graph)
    
    def get_conn(self, sigma_quimb):

        ind_charge_map = {0:0, 1:1, 2:-1, 3:0} # quimb to netket mapping

        charge_config = do('array',[ind_charge_map[int(s)] for s in sigma_quimb])

        return sigma_quimb, charge_config


