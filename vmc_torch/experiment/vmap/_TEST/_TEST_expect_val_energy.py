import quimb.tensor as qtn

import symmray as sr
import quimb.experimental.operatorbuilder as qop

# System parameters
Lx = 2
Ly = 2
nsites = Lx * Ly
D = 4
seed = 42
flat = True

# random fPEPS
peps = sr.networks.PEPS_fermionic_rand(
    "Z2",
    Lx,
    Ly,
    D,
    phys_dim=[
        (0, 0),  # linear index 0 -> charge 0, offset 0
        (1, 1),  # linear index 1 -> charge 1, offset 1
        (1, 0),  # linear index 2 -> charge 1, offset 0
        (0, 1),  # linear index 3 -> charge 0, offset 1
    ],  # -> (0, 3), (2, 1)
    # put an odd number of odd sites in, for testing
    site_charge=lambda site: int(site in [(0, 0), (0, 1), (1, 0)]),
    subsizes="equal",
    flat=flat,
    seed=seed,
)

# generate Hamiltonian graph
edges = qtn.edges_2d_square(Lx, Ly)
sites = [(i, j) for i in range(Lx) for j in range(Ly)]
H_quimb = qop.fermi_hubbard_from_edges(
    edges,
    U=8,
    mu=0,
    # this ordering pairs spins together, as with the fermionic TN
    order=lambda site: (site[1], site[0]),
    sector=int(sum(ary.charge for ary in peps.arrays) % 2),
    symmetry="Z2",
)
hs = H_quimb.hilbert_space

# BUG: pack or not pack, affects the result???
params, skeleton = qtn.pack(peps)
peps = qtn.unpack(params, skeleton)

# prepare amplitude function
def flat_amplitude(fx):
    # convert neighboring pairs (up, down) to single index 0..3
    # these should match up with the phys_dim ordering above
    fx = 2 * fx[::2] + fx[1::2] # grouped by sites turned into tn indices
    selector = {peps.site_ind(site): val for site, val in zip(peps.sites, fx)}
    tnb = peps.isel(selector)
    return tnb.contract()


# compute the full exact energy at the amplitude level
E_single = 0.0
p = 0.0
fcs = []
for i in range(hs.size):
    fx = hs.rank_to_flatconfig(i)

    xpsi = flat_amplitude(fx)
    if not xpsi:
        continue

    pi = abs(xpsi) ** 2
    p += pi

    Oloc = 0.0
    for fy, hxy in zip(*H_quimb.flatconfig_coupling(fx)):
        ypsi = flat_amplitude(fy)
        Oloc = Oloc + hxy * ypsi / xpsi
    E_single += Oloc * pi
print(f'Dense VMC energy: {E_single / p / nsites}')

# exact energy via local expectation contraction
terms = sr.hamiltonians.ham_fermi_hubbard_from_edges(
    "Z2",
    edges=edges,
    U=8,
    mu=0.0,
)
if flat:
    terms = {k: v.to_flat() for k, v in terms.items()}
E_double = peps.compute_local_expectation_exact(terms, normalized=True)
print(f'Exact double layer energy: {E_double.item() / nsites}')
