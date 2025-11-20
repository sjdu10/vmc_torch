import quimb.tensor as qtn
import os
import pickle
import scipy
import quimb as qu

ndim = 2
Lx = 4
Ly = 2
pbc = False
total_sz = 0.0
print(f"Total Sz = {total_sz}")
D = 2

# PEPS tensor network
psi = qtn.PEPS.rand(Lx=Lx, Ly=Ly, bond_dim=D, phys_dim=2) # initialization from random PEPS

# Ising Hamiltonian parameters
J=1.0
h=0.5

# ED
ham_dense =  qu.ham_heis_2D(Lx, Ly, j=(J,0,0), bz=h, cyclic=pbc)
evals, evecs = scipy.linalg.eigh(ham_dense)
exact_energy = evals[0]/(Lx*Ly)
print(f"Exact ground state energy per site: {exact_energy}")

ham = qtn.ham_2d_ising(Lx=Lx, Ly=Ly, j=J, bx=h)
su = qtn.tensor_arbgeom_tebd.SimpleUpdateGen(
    psi, 
    ham,
    compute_energy_every=10,
    compute_energy_per_site=True,
)
for tau in [1.0, 0.3, 0.1, 0.01]:#, 0.1, 0.03, 0.01]:
    su.evolve(100, tau=tau)
psi_su = su.state
peps = su.get_state()
peps.equalize_norms_(value=1)

# save the state
params, skeleton = qtn.pack(peps)

pwd = './example_data'
os.makedirs(pwd+f'/{Lx}x{Ly}/J={J}_h={h}/D={D}', exist_ok=True)

with open(pwd+f'/{Lx}x{Ly}/J={J}_h={h}/D={D}/peps_skeleton.pkl', 'wb') as f:
    pickle.dump(skeleton, f)
with open(pwd+f'/{Lx}x{Ly}/J={J}_h={h}/D={D}/peps_su_params.pkl', 'wb') as f:
    pickle.dump(params, f)