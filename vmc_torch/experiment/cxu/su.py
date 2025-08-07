import quimb as qu
import quimb.tensor as qtn

Lx = 4
Ly = 4
D = 4

ham = qtn.ham_2d_heis(
    Lx, Ly, j=1
)

psi0 = qtn.PEPS.rand(
    Lx, Ly, bond_dim=D, phys_dim=2
)

su = qtn.SimpleUpdate(
    psi0,
    ham,
    chi=32,  # boundary contraction bond dim for computing energy
    compute_energy_every=10,
    compute_energy_fn = lambda x: x.get_state().compute_local_expectation(
        ham.terms,
        cutoff=0.0,
        max_bond=256,
        normalized=True
    )/(Lx * Ly),
    keep_best=True,
)

su.evolve(100, tau=0.1)

psi = su.get_state()

qu.save_to_disk(
    psi,
    f'psi_heis_SU_4x4_D{D}.dmp'
)