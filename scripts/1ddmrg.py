import os
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'

L = 24
N = L-4
TWOSZ = 0
scratch_dir = pwd+"/tmp"
os.makedirs(scratch_dir, exist_ok=True)
driver = DMRGDriver(scratch=scratch_dir, symm_type=SymmetryTypes.SZ, n_threads=4, mpi=False)
driver.initialize_system(n_sites=L, n_elec=N, spin=TWOSZ)

t = 1
U = 8

b = driver.expr_builder()

# hopping term
b.add_term("cd", np.array([[[i, i + 1], [i + 1, i]] for i in range(L - 1)]).flatten(), -t)
b.add_term("CD", np.array([[[i, i + 1], [i + 1, i]] for i in range(L - 1)]).flatten(), -t)

# onsite term
b.add_term("cdCD", np.array([[i, ] * 4 for i in range(L)]).flatten(), U)

mpo = driver.get_mpo(b.finalize(), iprint=2)

bond_dim = 500
n_sweeps = 32
def run_dmrg(driver, mpo):
    ket = driver.get_random_mps(tag="KET", bond_dim=bond_dim, nroots=1)
    bond_dims = [bond_dim] * n_sweeps
    # bond_dims = [int(bond_dim/2)] * int(n_sweeps/2) + [int(bond_dim)] * int(n_sweeps/2)
    noises = [1e-4] * int(n_sweeps/2) + [0] * int(n_sweeps/2)
    thrds = [1e-10] * n_sweeps
    return driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises,
        thrds=thrds, cutoff=0, iprint=1)

energies = run_dmrg(driver, mpo)
print('DMRG energy = %20.15f' % energies)
os.makedirs(pwd+f'/dmrg_data', exist_ok=True)
np.savetxt(pwd+f'/dmrg_data/L={L}_t={t}_U={U}_N={N}_D={bond_dim}_DMRG_energy.txt', np.array([energies]))