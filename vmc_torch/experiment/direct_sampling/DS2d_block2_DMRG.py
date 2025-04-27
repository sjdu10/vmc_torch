from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
import os

bond_dims = [100] * 12 + [200] * 18
noises = [1E-7] * 12 + [1E-7] * 12 + [0] * 6
thrds = [1E-6] * 30
n_sweeps = 30

nx, ny = 4, 4
n = nx * ny
u = 8.0
nelec = int(nx*ny)
scratch_dir = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/data'+f'/{nx}x{ny}/t=1.0_U={u}/N={nelec}/tmp'
os.makedirs(scratch_dir, exist_ok=True)
driver = DMRGDriver(scratch=scratch_dir, symm_type=SymmetryTypes.SZ, stack_mem=24 << 30, n_threads=12, mpi=True)

print("NELEC = %d" % nelec)
print("U = %d" % u)

driver.initialize_system(n_sites=n, n_elec=nelec, spin=0, orb_sym=None)

b = driver.expr_builder()

f = lambda i, j: i * ny + j #if i % 2 == 0 else i * ny + ny - 1 - j

for i in range(0, nx):
    for j in range(0, ny):
        if i + 1 < nx:
            b.add_term("cd", [f(i, j), f(i + 1, j), f(i + 1, j), f(i, j)], -1)
            b.add_term("CD", [f(i, j), f(i + 1, j), f(i + 1, j), f(i, j)], -1)
        if j + 1 < ny:
            b.add_term("cd", [f(i, j), f(i, j + 1), f(i, j + 1), f(i, j)], -1)
            b.add_term("CD", [f(i, j), f(i, j + 1), f(i, j + 1), f(i, j)], -1)
        b.add_term("cdCD", [f(i, j), f(i, j), f(i, j), f(i, j)], u)

mpo = driver.get_mpo(b.finalize(adjust_order=True), algo_type=MPOAlgorithmTypes.FastBipartite, iprint=2)
ket = driver.get_random_mps(tag='KET', bond_dim=250)
energy = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises, thrds=thrds,
    lowmem_noise=True, twosite_to_onesite=None, tol=1E-12, cutoff=1E-24, iprint=1,
    dav_max_iter=50, dav_def_max_size=20)
mps_energy = driver._dmrg.sweep_energies[-1][-1]
print('MPS  energy = %20.15f' % mps_energy)
print('DMRG energy = %20.15f' % energy)
# print('MPS center = %d' % ket.center)


if ket.center != 0:
    print('Aligning MPS center to 0')
    ket = driver.copy_mps(ket, tag="CSF-TMP")
    driver.align_mps_center(ket, ref=0)


samples = [4]
vmc_energies = []
import numpy as np

driver = DMRGDriver(scratch=scratch_dir, symm_type=SymmetryTypes.SZ, stack_mem=24 << 30, n_threads=12, mpi=True)
ket = driver.load_mps(tag='KET')

for i in range(2):
    for n_sample in samples:
        print('MPS center = %d' % ket.center)
        # sample determinants on |MPS> and compute <DET|MPS>
        dets, coeffs = driver.sample_csf_coefficients(ket, n_sample=n_sample, max_print=20, rand_seed=np.random.randint(0, 2**31-1))

    # # compute <DET|H|MPS> on sampled determinants
    # hdets, hcoeffs = driver.get_csf_coefficients(hket, cutoff=0.0, given_dets=dets, max_print=20)
    # assert np.linalg.norm(hdets - dets) == 0.0
    # print(dets, coeffs)

    # # compute E[VMC] = E[ <DET|H|MPS> / <DET|<MPS> ]
    # vmc_energy = np.sum(hcoeffs / coeffs) / n_sample
    # vmc_energies.append(vmc_energy)
    # print('N = %15d VMC  energy = %20.15f DIFF = %20.15f' % (n_sample, vmc_energy, vmc_energy - mps_energy)