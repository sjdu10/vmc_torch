import os
import pickle
import quimb.tensor as qtn
import quimb as qu
import netket as nk
from netket.graph import Lattice
from vmc_torch.hamiltonian import spin_Heisenberg_square_lattice
import numpy as np
from math import pi
from autoray import do

ndim = 2
Lx = 10
Ly = 10
pbc = False
total_sz = 0.0
print(f"Total Sz = {total_sz}")
D = 2

basis = np.array([
     [1.0,0.0],
     [0.0,1.0],
 ])
custom_edges = [
     (0, 0, [1.0,0.0], 0),
     (0, 0, [0.0,1.0], 0),
     (0, 0, [1.0, 1.0], 1),
     (0, 0, [1.0, -1.0], 1),
 ]

g = Lattice(basis_vectors=basis, pbc=False, extent=[Lx, Ly],
     custom_edges=custom_edges)

n = g.n_nodes
hi = nk.hilbert.Spin(s=1/2, total_sz=0.0, N=n)
# Heisenberg with coupling J=1.0 for nearest neighbors
# and J=0.5 for next-nearest neighbors
H = nk.operator.Heisenberg(hilbert=hi, graph=g, J=(1.0,0.5)) # In Netket, the spin operators are Pauli matrices, while in Quimb they are 1/2*Pauli matrices

# PEPS tensor network
psi = qtn.PEPS.rand(Lx=Lx, Ly=Ly, bond_dim=D, phys_dim=2) # initialization from PEPS
J1=1.0
J2=0.5

ham = qtn.ham_2d_j1j2(Lx, Ly, j1=J1, j2=J2)

su = qtn.tensor_2d_tebd.SimpleUpdate(
    psi, 
    ham,
    D=D,
    compute_energy_every=100,
    compute_energy_per_site=True,
)

for tau in [1.0, 0.3, 0.1, 0.03, 0.01]:
    su.evolve(100, tau=tau)
psi_su = su.state

peps = su.get_state()
peps.equalize_norms_(value=1)

# save the state
params, skeleton = qtn.pack(peps)


os.makedirs(f'../data/{Lx}x{Ly}/J1={J1}_J2={J2}/D={D}', exist_ok=True)

with open(f'../data/{Lx}x{Ly}/J1={J1}_J2={J2}/D={D}/peps_skeleton.pkl', 'wb') as f:
    pickle.dump(skeleton, f)
with open(f'../data/{Lx}x{Ly}/J1={J1}_J2={J2}/D={D}/peps_su_params.pkl', 'wb') as f:
    pickle.dump(params, f)