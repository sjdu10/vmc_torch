import quimb.experimental.operatorbuilder as qop
import quimb.tensor as qtn
import symmray as sr
import numpy as np

Lx = 3
Ly = 2
D = 4
seed = 42
flat = False

peps = sr.networks.PEPS_fermionic_rand(
    "Z2",
    Lx,
    Ly,
    D,
    phys_dim=[
        (0, 0),  # linear index 0 -> charge 0, offset 0
        (1, 0),  # linear index 1 -> charge 1, offset 0
        (1, 1),  # linear index 2 -> charge 1, offset 1
        (0, 1),  # linear index 3 -> charge 0, offset 1
    ],  # -> (0, 3), (2, 1)
    subsizes="equal",
    flat=flat,
    seed=seed,
    dtype="float64"
)
fpeps_random = peps.copy()

# SU
terms = sr.ham_fermi_hubbard_from_edges(
    symmetry='Z2',
    edges=tuple(peps.gen_bond_coos()),
    t=1.0,
    U=8,
    mu=4,
)
ham = qtn.LocalHam2D(Lx, Ly, terms)
if flat:
    ham.terms = {k: v.to_flat() for k, v in ham.terms.items()}
# ham.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float64))
su = qtn.SimpleUpdateGen(
    fpeps_random,
    ham,
    # setting a cutoff is important to turn on dynamic charge sectors
    # cutoff=1e-12,
    cutoff=0.0,
    second_order_reflect=True,
    # SimpleUpdateGen computes cluster energies by default
    # which might not be accurate
    compute_energy_every=10,
    compute_energy_opts=dict(max_distance=1),
    compute_energy_per_site=False,
    # use a fixed trotterization order
    ordering="sort",
    # if the gauge difference drops below this, we consider the PEPS converged
    tol=1e-9,
)

# run the evolution, these are reasonable defaults
tau = 0.1
steps = 25
su.evolve(steps, tau=tau)
su_peps = su.get_state()
for site in su_peps.sites:
    su_peps[site].data._label = site


edges = qtn.edges_2d_square(Lx, Ly)
sites = [(i, j) for i in range(Lx) for j in range(Ly)]
# get a symbolic representation of the Hamiltonian
H = qop.fermi_hubbard_from_edges(
    edges,
    t=1,
    U=8,
    mu=0,
    # this ordering pairs spins together, as with the fermionic TN
    order=lambda site: (site[1], site[0]),
    sector=int(sum(ary.charge for ary in peps.arrays) % 2),
    symmetry="Z2",
)
hs = H.hilbert_space

# symmray Hamiltonian terms for double layer exact energy computation
terms = sr.hamiltonians.ham_fermi_hubbard_from_edges(
    "Z2",
    edges=edges,
    U=8,
    mu=0,
)
if flat:
    terms = {k: v.to_flat() for k, v in terms.items()}

from symmray import FermionicOperator
from autoray import do

# Define the symmray amplitude function
def flat_amplitude(fx, peps):
    # convert neighboring pairs (up, down) to single index 0..3
    # these should match up with the phys_dim ordering above
    fx = fx[::2] + 2*fx[1::2] # grouped by sites turned into tn indices

    selector = {peps.site_ind(site): val for site, val in zip(peps.sites, fx)}
    tnb = peps.isel(selector)
    return tnb.contract()


# Benchmark amplitude function from Sijing
def get_amp(psi0, config):
    psi = psi0.copy()
    # psi = copy.deepcopy(psi0)
    if psi.arrays[0].symmetry == 'Z2':
        index_map = {0: 0, 1: 1, 2: 1, 3: 0}
        array_map = {
            0: do('array', [1.0, 0.0]),
            1: do('array', [1.0, 0.0]),
            2: do('array', [0.0, 1.0]),
            3: do('array', [0.0, 1.0])
        }

    for n, site in zip(config, psi.sites):
        p_ind = psi.site_ind_id.format(*site)
        site_id = psi.sites.index(site)
        site_tag = psi.site_tag_id.format(*site)
        # fts = psi.tensors[site_id]
        fts = psi[site_tag]
        ftsdata = fts.data # this is the on-site fermionic tensor (f-tensor) to be contracted
        ftsdata.phase_sync(inplace=True) # explicitly apply all lazy phases that are stored and not yet applied
        phys_ind_order = fts.inds.index(p_ind)
        charge = index_map[int(n)] # charge of the on-site fermion configuration
        input_vec = array_map[int(n)] # input vector of the on-site fermion configuration
        charge_sec_data_dict = ftsdata.blocks # the dictionary of the f-tensor data

        new_fts_inds = fts.inds[:phys_ind_order] + fts.inds[phys_ind_order + 1:] # calculate indices of the contracted f-tensor
        new_charge_sec_data_dict = {} # new dictionary to store the data of the contracted f-tensor
        for charge_blk, data in charge_sec_data_dict.items():
            if charge_blk[phys_ind_order] == charge:
                # 1. Determine which index to select (0 or 1) from the input vector.
                #    `argmax` finds the position of the '1.0'.
                # select_index = torch.argmax(input_vec).item()
                select_index = do('argmax', input_vec)

                # 2. Build the slicer tuple dynamically.
                #    This creates a list of `slice(None)` (which is equivalent to `:`)
                #    and inserts the `select_index` at the correct position.
                slicer = [slice(None)] * data.ndim
                slicer[phys_ind_order] = select_index

                # 3. Apply the slice to get the new data.
                new_data = data[tuple(slicer)]

                # 4. Fermionic sign correction due to potential permutation of odd indices.
                #     (In our convention the physical ind should be the last ind during contraction)
                if charge % 2 != 0 and phys_ind_order != len(charge_blk) - 1:
                    # Count how many odd indices are to the right of the physical index.
                    # Check if odd physical ind permutes through odd number of odd indices.
                    num_odd_right_blk = sum(1 for i in charge_blk[phys_ind_order + 1:] if i % 2 == 1)
                    if num_odd_right_blk % 2 == 1:
                        # local fermionic sign change due to odd parity inds + odd permutation
                        new_data = -new_data

                new_charge_blk = charge_blk[:phys_ind_order] + charge_blk[phys_ind_order + 1:] # new charge block
                new_charge_sec_data_dict[new_charge_blk] = new_data # new charge block and its corresponding data in a dictionary

        new_duals = ftsdata.duals[:phys_ind_order] + ftsdata.duals[phys_ind_order + 1:]

        if int(n) == 1:
            new_dummy_modes = (3 * site_id + 1) * (-1)
        elif int(n) == 2:
            new_dummy_modes = (3 * site_id + 2) * (-1)
        elif int(n) == 3 or int(n) == 0:
            new_dummy_modes = ()
        
        new_dummy_modes1 = FermionicOperator(new_dummy_modes, dual=True) if new_dummy_modes else ()
        new_dummy_modes = ftsdata.dummy_modes + (new_dummy_modes1,) if isinstance(new_dummy_modes1, FermionicOperator) else ftsdata.dummy_modes
        dummy_modes = list(new_dummy_modes)[::-1]
        try:
            if psi.arrays[0].symmetry == 'Z2':
                new_charge = (charge + ftsdata.charge) % 2 # Z2 symmetry, charge should be 0 or 1
            new_fts_data = sr.FermionicArray.from_blocks(new_charge_sec_data_dict, duals=new_duals, charge=new_charge, symmetry=psi.arrays[0].symmetry, dummy_modes=dummy_modes)
        except Exception:
            raise ValueError("Error when constructing the new f-tensor after contraction.")
        
        fts.modify(data=new_fts_data, inds=new_fts_inds, left_inds=None)

    amp = qtn.PEPS(psi)

    return amp

# construct the dense Hamiltonian matrix and compute the state vectors
psi_vec_su = np.zeros(hs.size, dtype=np.float64)
psi_vec_su_benchmark = np.zeros(hs.size, dtype=np.float64)
H_dense = np.zeros((hs.size, hs.size), dtype=np.float64)

# for i in range(1):
fx = hs.rank_to_flatconfig(0)
fx0 = fx[::2]+2*fx[1::2]
for ts in su_peps.tensors:
    ts.data.phase_sync(inplace=True)

# print("Are the original SU PEPS and the SU PEPS identical after SU evolution?", are_pytrees_equal(original_su_peps.get_params(), su_peps.get_params()))


# somehow fix the bugs??
_, skeleton = qtn.pack(peps)
params, _ = qtn.pack(su_peps)
new_peps = qtn.unpack(params, skeleton)

psi_vec_su = np.zeros(hs.size, dtype=np.float64)
psi_vec_su_benchmark_new = np.zeros(hs.size, dtype=np.float64)
H_dense = np.zeros((hs.size, hs.size), dtype=np.float64)

for i in range(hs.size):
    fx = hs.rank_to_flatconfig(i)
    fx0 = fx[::2]+2*fx[1::2]
    xpsi = flat_amplitude(fx, new_peps)
    psi_vec_su[i] = xpsi
    psi_vec_su_benchmark_new[i] = get_amp(new_peps, np.array(fx0)).contract()
    
    for fy, hxy in zip(*H.flatconfig_coupling(fx)):
        fy_idx = hs.flatconfig_to_rank(fy)
        H_dense[i, fy_idx] += hxy
        if not xpsi:
            continue

E = (psi_vec_su.conj().T @ H_dense @ psi_vec_su) / (psi_vec_su.conj().T @ psi_vec_su)
print(f'PEPS state vector (symmray slicing) energy from amps after SU: {E}')
E_benchmark = (psi_vec_su_benchmark_new.conj().T @ H_dense @ psi_vec_su_benchmark_new) / (psi_vec_su_benchmark_new.conj().T @ psi_vec_su_benchmark_new)
print(f'PEPS state vector (benchmark amp) energy from amps after SU: {E_benchmark}')
eref = su_peps.compute_local_expectation_exact(terms, normalized=True)
print(f'Reference double layer contraction energy after SU: {eref}')