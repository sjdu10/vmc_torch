import os
os.environ["NUMBA_NUM_THREADS"] = "20"
from vmc_torch.fermion_utils import generate_random_fpeps, u1peps_to_z2peps
import quimb.tensor as qtn
import symmray as sr
import pickle
from symmray.fermionic_local_operators import (
    build_local_fermionic_array,
    FermionicOperator,
    get_spinful_charge_indexmap,
)

# Define the lattice shape
Lx = int(4)
Ly = int(4)
spinless = False
N = int(Lx * Ly)

# Define the fermion filling and the Hilbert space
N_f = int(Lx*Ly-2)

# SU in quimb
D = 4
seed = 42
peps = generate_random_fpeps(Lx, Ly, D=D, seed=seed, symmetry='U1', Nf=N_f, spinless=spinless)[0]
edges = qtn.edges_2d_square(Lx, Ly)
site_info = sr.parse_edges_to_site_info(
    edges,
    D,
    phys_dim=4,
    site_ind_id="k{},{}",
    site_tag_id="I{},{}",
)

# FH params
t = 1.0
U = 8.0
mu = 0.0

def fermi_hubbard_local_array_w_spf(
    symmetry,
    t=1.0,
    U=8.0,
    mu=0.0,
    spf=0.0,
    coordinations=(1, 1),
    like="numpy",
):
    """Construct the fermionic local tensor for the Fermi-Hubbard model. The
    indices are ordered as (a, b, a', b'), with the local basis like
    (|00>, ad+|00>, au+|00>, au+ad+|00>) for site a with up (au) and down (ad)
    spin respectively and similar for site b.

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Either "Z2", "U1", "Z2Z2", or "U1U1".
    t : float, optional
        The hopping parameter, by default 1.0.
    U : float or (float, float), optional
        The interaction parameter, by default 8.0. If a tuple, then the
        interaction parameter is different for each site.
    mu : float or (float, float), optional
        The chemical potential, by default 0.0. If a tuple, then the chemical
        potential is different for each site.
    spf : float or (float, float), optional
        The magnetic field, by default 0.0. If a tuple, then the magnetic field is
        different for each site.
    coordinations : tuple[int, int], optional
        The coordinations of the sites, by default (1, 1). If applying this
        local operator to every edge in a graph, then the single site
        contributions can be properly accounted for if the coordinations are
        provided.
    like : str, optional
        The backend to use, by default "numpy".

    Returns
    -------
    array : FermionicArray
        The local operator in fermionic array form.
    """
    au = FermionicOperator("au")
    ad = FermionicOperator("ad")
    bu = FermionicOperator("bu")
    bd = FermionicOperator("bd")

    try:
        Ua, Ub = U
    except TypeError:
        Ua = Ub = U

    try:
        mua, mub = mu
    except TypeError:
        mua = mub = mu
    
    try:
        spfa, spfb = spf
    except TypeError:
        spfa = spfb = spf

    terms = [
        (-t, (au.dag, bu)),
        (-t, (bu.dag, au)),
        (-t, (ad.dag, bd)),
        (-t, (bd.dag, ad)),
        # U, mu are single site and will be overcounted without coordinations
        (Ua / coordinations[0], (au.dag, au, ad.dag, ad)),
        (Ub / coordinations[1], (bu.dag, bu, bd.dag, bd)),
        (-mua / coordinations[0], (au.dag, au)),
        (-mua / coordinations[0], (ad.dag, ad)),
        (-mub / coordinations[1], (bu.dag, bu)),
        (-mub / coordinations[1], (bd.dag, bd)),
        (-spfa/2/coordinations[0], (au.dag, au)),
        ( spfa/2/coordinations[0], (ad.dag, ad)),
        (-spfb/2/coordinations[1], (bu.dag, bu)),
        ( spfb/2/coordinations[1], (bd.dag, bd)),
    ]

    basis_a = ((), (ad.dag,), (au.dag,), (au.dag, ad.dag))
    basis_b = ((), (bd.dag,), (bu.dag,), (bu.dag, bd.dag))
    bases = [basis_a, basis_b]
    indexmap = get_spinful_charge_indexmap(symmetry)

    return build_local_fermionic_array(
        terms,
        bases,
        symmetry,
        index_maps=[indexmap, indexmap],
        like=like,
    )

# Define the Hubbard Hamiltonian
t = 1.0
U = 8.0
mu = 0.0

cpf_target_sites = [(x,x) for x in range(Ly)]
cpf = 5.0

spf_target_sites = [(0,x) for x in range(Ly)]
spf = 1.0

print(f'Chemical potential: {mu}')
print(f'Charge pinning field {cpf} on sites: {cpf_target_sites}')
print(f'Spin pinning field {spf} on sites: {spf_target_sites}')

def mu_f(sitea, siteb, target_sites, cpf=0, mu=0):
    return (cpf+mu if sitea in target_sites else mu, cpf+mu if siteb in target_sites else mu)
def spf_f(sitea, siteb, target_sites, spf=0):
    return (spf if sitea in target_sites else 0, spf if siteb in target_sites else 0)

u1_terms = {
    (sitea, siteb): fermi_hubbard_local_array_w_spf(
        t=t,
        U=U,
        mu=mu_f(sitea, siteb, cpf_target_sites, cpf=cpf, mu=mu),
        spf=spf_f(sitea, siteb, spf_target_sites, spf=spf),
        symmetry="U1",
        coordinations=(
            site_info[sitea]["coordination"],
            site_info[siteb]["coordination"],
        ),
    )
    for (sitea, siteb) in edges
}

u1ham = qtn.LocalHam2D(Lx, Ly, H2=u1_terms)
u1su = qtn.SimpleUpdateGen(
    peps, 
    u1ham, 
    compute_energy_per_site=True,
    D=D, 
    compute_energy_opts={"max_distance":1}, 
    gate_opts={'cutoff':1e-12}, 
)
# Evolve the U1-fPEPS
u1su.evolve(100, tau=0.1)
u1su.evolve(100, tau=0.01)

u1peps = u1su.get_state()
u1peps.equalize_norms_(value=1)
u1peps.exponent = 0.0

# Convert U1-fPEPS to Z2-fPEPS
z2peps = u1peps_to_z2peps(u1peps)

# Define Z2 Fermi-Hubbard Hamiltonian terms
# Fermi-Hubbard terms U1 symmetry
z2_terms = {
    (sitea, siteb): sr.fermi_hubbard_local_array(
        t=t, U=U, mu=mu,
        symmetry='Z2',
        coordinations=(
            site_info[sitea]['coordination'],
            site_info[siteb]['coordination'],
        ),
    )
    for (sitea, siteb) in edges
}

z2ham = qtn.LocalHam2D(Lx, Ly, H2=z2_terms)
z2su = qtn.SimpleUpdateGen(
    z2peps, 
    z2ham, 
    compute_energy_per_site=True,
    D=D, 
    compute_energy_opts={"max_distance":1}, 
    gate_opts={'cutoff':1e-12}, 
)
# Evolve the Z2-fPEPS
z2su.evolve(2, tau=0.01)

peps = z2su.get_state()
peps.equalize_norms_(value=1)
peps.exponent = 0.0


# save the state
params, skeleton = qtn.pack(peps)
pwd = './test_data'
import os
os.makedirs(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}', exist_ok=True)

with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/peps_skeleton_U1SU.pkl', 'wb') as f:
    pickle.dump(skeleton, f)
with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/peps_su_params_U1SU.pkl', 'wb') as f:
    pickle.dump(params, f)