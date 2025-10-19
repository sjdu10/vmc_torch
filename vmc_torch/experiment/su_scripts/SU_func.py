import os
from vmc_torch.fermion_utils import generate_random_fpeps, u1peps_to_z2peps, generate_random_fpeps_symmray
import quimb.tensor as qtn
import symmray as sr
import pickle
from symmray.fermionic_local_operators import (
    build_local_fermionic_array,
    FermionicOperator,
    get_spinful_charge_indexmap,
)
from vmc_torch.fermion_utils import format_fpeps_keys

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
    ]
    if spfa != 0.0:
        terms += [
        (-spfa/2/coordinations[0], (au.dag, au)),
        ( spfa/2/coordinations[0], (ad.dag, ad)),
    ]
    if spfb != 0.0:
        terms += [
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

def mu_f(sitea, siteb, target_sites, cpf=0, mu=0):
    if cpf == 0:
        return mu
    return (cpf+mu if sitea in target_sites else mu, cpf+mu if siteb in target_sites else mu)
def spf_f(sitea, siteb, target_sites: dict, spf=0):
    if spf == 0:
        return 0
    return (spf*target_sites.get(sitea, 0), spf*target_sites.get(siteb, 0))

def run_u1SU_w_pinning_field(
    Lx, Ly, D, N_f, t, U, mu, cpf, cpf_target_sites, spf, spf_target_sites, pwd, seed, save_file=True, run_su=True, su_evolve_schedule=[(100, 0.01)],
    rfpeps_kwargs={'subsizes':'maximal'}, **su_kwargs
):
    """
    Lx : int
        Lattice size in x direction.
    Ly : int
        Lattice size in y direction.
    D : int
        Bond dimension of the fPEPS.
    N_f : int
        Number of fermions.
    t, U, mu : float
        Fermi-Hubbard model parameters.
    cpf : float
        Charge pinning field strength.
    cpf_target_sites : list of tuple
        List of sites to apply the charge pinning field.
    spf : float
        Spin pinning field (magnetic field) strength.
    spf_target_sites : dict: tuple -> int
        Dictionary of sites to apply the magnetic field with target field direction (+1 or -1).
    pwd : str
        Path to save the data.
    seed : int
        Random seed for fPEPS initialization.
    save_file : bool, optional
        Whether to save the resulting fPEPS state, by default True.
    run_su : bool, optional
        Whether to run the simple update or just load existing state, by default True.
    su_evolve_schedule : list of tuple, optional
        List of (n_steps, tau) for SU evolution, by default [(100, 0.01)].
    rfpeps_kwargs : dict, optional
        Additional kwargs for random fPEPS generation, by default {}.
    su_kwargs : dict, optional
        Additional kwargs for SimpleUpdateGen, by default {}.
    """
    print('\n ===================================')
    print('Running U1 SU with pinning fields:')
    print(f'Hamiltonian parameters: t={t}, U={U}, mu={mu}')
    print(f'fPEPS parameters: Lx={Lx}, Ly={Ly}, D={D}, N_f={N_f}, seed={seed}')
    print(f'Chemical potential: {mu}')
    print(f'Charge pinning field {cpf} on sites: {cpf_target_sites}')
    print(f'Spin pinning field {spf} on sites: {spf_target_sites}\n')

    if not run_su:
        print('Just loading potentially existing SU fPEPS state...')
        # check if the file exists
        params_file = pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}/peps_su_params_cpf={cpf}_spf={spf}.pkl'
        skeleton_file = pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}/peps_skeleton_cpf={cpf}_spf={spf}.pkl'
        if os.path.exists(params_file) and os.path.exists(skeleton_file):
            print('Found existing files, loading...')
            skeleton = pickle.load(open(skeleton_file, 'rb'))
            peps_params = pickle.load(open(params_file, 'rb'))
            peps = qtn.unpack(peps_params, skeleton)
            return peps
        else:
            print('No existing files found')
            raise FileNotFoundError('No existing SU fPEPS state found with the specified pinning fields.')

    # Define the lattice shape
    spinless = False

    # SU in quimb
    # rpeps = generate_random_fpeps(Lx, Ly, D=D, seed=seed, symmetry='U1', Nf=N_f, spinless=spinless)[0]
    rpeps = generate_random_fpeps_symmray(Lx, Ly, D=D, seed=seed, symmetry='U1', Nf=N_f, spinless=spinless, **rfpeps_kwargs)
    edges = qtn.edges_2d_square(Lx, Ly)
    site_info = sr.parse_edges_to_site_info(
        edges,
        D,
        phys_dim=4,
        site_ind_id="k{},{}",
        site_tag_id="I{},{}",
    )
    
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
        rpeps, 
        u1ham, 
        D=D, 
        **su_kwargs
    )
    print('Starting SU evolution with pinning fields...')
    # Evolve the U1-fPEPS
    for n_steps, tau in su_evolve_schedule:
        u1su.evolve(n_steps, tau=tau)
    
    fig, _ = u1su.plot()
    fig.savefig(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}/su_evolution_cpf={cpf}_spf={spf}.png')

    u1peps = u1su.get_state()
    u1peps.equalize_norms_(value=1)
    u1peps.exponent = 0.0
    u1peps = format_fpeps_keys(u1peps)
    # save the state
    params, skeleton = qtn.pack(u1peps)
    if save_file:
        os.makedirs(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}', exist_ok=True)
        with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}/peps_skeleton_cpf={cpf}_spf={spf}.pkl', 'wb') as f:
            pickle.dump(skeleton, f)
        with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}/peps_su_params_cpf={cpf}_spf={spf}.pkl', 'wb') as f:
            pickle.dump(params, f)
    print('===================================')
    return u1peps


def run_u1SU(
    Lx, Ly, D, N_f, t, U, mu, pwd, seed=42, initial_peps=None, save_file=True, run_su=True, su_evolve_schedule=[(100, 0.01)],
    **su_kwargs
):
    """
    Lx : int
        Lattice size in x direction.
    Ly : int
        Lattice size in y direction.
    D : int
        Bond dimension of the fPEPS.
    N_f : int
        Number of fermions.
    t, U : float
        Fermi-Hubbard model parameters.
    pwd : str
        Path to save the data.
    seed : int
        Random seed for fPEPS initialization.
    initial_peps : fPEPS, optional
        Initial fPEPS state to start SU from, by default None.
    save_file : bool, optional
        Whether to save the resulting fPEPS state, by default True.
    run_su : bool, optional
        Whether to run the simple update or just load existing state, by default True.
    su_evolve_schedule : list of tuple, optional
        List of (n_steps, tau) for SU evolution, by default [(100, 0.01)].
    su_kwargs : dict, optional
        Additional kwargs for SimpleUpdateGen, by default {}.
    """
    print('\n ===================================')
    print('Running U1 SU:')
    print(f'Hamiltonian parameters: t={t}, U={U}, mu={mu}')
    print(f'fPEPS parameters: Lx={Lx}, Ly={Ly}, D={D}, N_f={N_f}, seed={seed}')
    print(f'Chemical potential: {mu}\n')
    
    if not run_su:
        print('Just loading potentially existing SU fPEPS state...')
        # check if the file exists
        params_file = pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}/peps_su_params.pkl'
        skeleton_file = pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}/peps_skeleton.pkl'
        if os.path.exists(params_file) and os.path.exists(skeleton_file):
            print('Found existing files, loading...')
            skeleton = pickle.load(open(skeleton_file, 'rb'))
            peps_params = pickle.load(open(params_file, 'rb'))
            peps = qtn.unpack(peps_params, skeleton)
            return peps
        else:
            print('No existing files found')
            return None
    if initial_peps is not None:
        peps = initial_peps
    else:
        print('Generating random U1-fPEPS...')
        # Define the lattice shape
        spinless = False
        # SU in quimb
        # peps = generate_random_fpeps(Lx, Ly, D=D, seed=seed, symmetry='U1', Nf=N_f, spinless=spinless)[0]
        peps = generate_random_fpeps_symmray(Lx, Ly, D=D, seed=seed, symmetry='U1', Nf=N_f, spinless=spinless)
    
    edges = qtn.edges_2d_square(Lx, Ly)
    site_info = sr.parse_edges_to_site_info(
        edges,
        D,
        phys_dim=4,
        site_ind_id="k{},{}",
        site_tag_id="I{},{}",
    )
    u1_terms = {
        (sitea, siteb): sr.fermi_hubbard_local_array(
            t=t,
            U=U,
            mu=mu,
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
        D=D, 
        **su_kwargs
    )
    # Evolve the U1-fPEPS
    for n_steps, tau in su_evolve_schedule:
        u1su.evolve(n_steps, tau=tau)
    
    fig, _ = u1su.plot()
    fig.savefig(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}/su_evolution.png')

    u1peps = u1su.get_state()
    u1peps.equalize_norms_(value=1)
    u1peps.exponent = 0.0
    u1peps = format_fpeps_keys(u1peps)

    # save the state
    params, skeleton = qtn.pack(u1peps)
    if save_file:
        os.makedirs(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}', exist_ok=True)
        with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}/peps_skeleton.pkl', 'wb') as f:
            pickle.dump(skeleton, f)
        with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}/peps_su_params.pkl', 'wb') as f:
            pickle.dump(params, f)
    print('===================================')
    return u1peps

def run_z2SU(
    Lx, Ly, D, N_f, t, U, mu, pwd, seed=42, initial_peps=None, save_file=True, run_su=True, su_evolve_schedule=[(100, 0.01)],
    **su_kwargs
):
    """
    Run SU on a Z2-fPEPS.

    Lx : int
        Lattice size in x direction.
    Ly : int
        Lattice size in y direction.
    D : int
        Bond dimension of the fPEPS.
    N_f : int
        Number of fermions.
    t, U : float
        Fermi-Hubbard model parameters.
    pwd : str
        Path to save the data.
    initial_peps : fPEPS, optional
        Initial fPEPS state to start SU from, by default None.
    save_file : bool, optional
        Whether to save the resulting fPEPS state, by default True.
    run_su : bool, optional
        Whether to run the simple update or just load existing state, by default True.
    su_evolve_schedule : list of tuple, optional
        List of (n_steps, tau) for SU evolution, by default [(100, 0.01)].
    su_kwargs : dict, optional
        Additional kwargs for SimpleUpdateGen, by default {}.
    """
    print('\n ===================================')
    print('Running Z2 SU:')
    print(f'Hamiltonian parameters: t={t}, U={U}, mu={mu}')
    print(f'fPEPS parameters: Lx={Lx}, Ly={Ly}, D={D}, N_f={N_f}, seed={seed}')
    print(f'Chemical potential: {mu}\n')

    if not run_su:
        print('Just loading potentially existing SU fPEPS state...')
        # check if the file exists
        params_file = pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/peps_su_params.pkl'
        skeleton_file = pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/peps_skeleton.pkl'
        if os.path.exists(params_file) and os.path.exists(skeleton_file):
            print('Found existing files, loading...')
            skeleton = pickle.load(open(skeleton_file, 'rb'))
            peps_params = pickle.load(open(params_file, 'rb'))
            peps = qtn.unpack(peps_params, skeleton)
            return peps
        else:
            print('No existing files found')
            return None
    if initial_peps is not None:
        peps = initial_peps
    else:
        print('Generating random Z2-fPEPS...')
        # Define the lattice shape
        spinless = False
        # SU in quimb
        peps = generate_random_fpeps(Lx, Ly, D=D, seed=seed, symmetry='Z2', Nf=N_f, spinless=spinless)[0]

    edges = qtn.edges_2d_square(Lx, Ly)
    site_info = sr.parse_edges_to_site_info(
        edges,
        D,
        phys_dim=4,
        site_ind_id="k{},{}",
        site_tag_id="I{},{}",
    )
    z2_terms = {
        (sitea, siteb): sr.fermi_hubbard_local_array(
            t=t,
            U=U,
            mu=mu,
            symmetry="Z2",
            coordinations=(
                site_info[sitea]["coordination"],
                site_info[siteb]["coordination"],
            ),
        )
        for (sitea, siteb) in edges
    }
    z2ham = qtn.LocalHam2D(Lx, Ly, H2=z2_terms)
    z2su = qtn.SimpleUpdateGen(
        peps, 
        z2ham, 
        D=D, 
        **su_kwargs
    )
    # Evolve the Z2-fPEPS
    for n_steps, tau in su_evolve_schedule:
        z2su.evolve(n_steps, tau=tau)
    
    fig, _ = z2su.plot()
    fig.savefig(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/su_evolution.png')

    z2peps = z2su.get_state()
    z2peps.equalize_norms_(value=1)
    z2peps.exponent = 0.0
    # save the state
    params, skeleton = qtn.pack(z2peps)
    if save_file:
        os.makedirs(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}', exist_ok=True)
        with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/peps_skeleton.pkl', 'wb') as f:
            pickle.dump(skeleton, f)
        with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/peps_su_params.pkl', 'wb') as f:
            pickle.dump(params, f)
    print('===================================')
    return z2peps

def run_z2SU_from_u1SU(
    Lx, Ly, D, N_f, t, U, mu, pwd, u1peps=None, save_file=True, run_su=True, su_evolve_schedule=[(100, 0.01)],
    **su_kwargs
):
    """
    Convert a U1-fPEPS to Z2-fPEPS and run SU on the Z2-fPEPS.

    u1peps : fPEPS
        The input U1-fPEPS state.
    Lx : int
        Lattice size in x direction.
    Ly : int
        Lattice size in y direction.
    D : int
        Bond dimension of the fPEPS.
    N_f : int
        Number of fermions.
    t, U : float
        Fermi-Hubbard model parameters.
    pwd : str
        Path to save the data.
    save_file : bool, optional
        Whether to save the resulting fPEPS state, by default True.
    run_su : bool, optional
        Whether to run the simple update or just load existing state, by default True.
    su_evolve_schedule : list of tuple, optional
        List of (n_steps, tau) for SU evolution, by default [(100, 0.01)].
    su_kwargs : dict, optional
        Additional kwargs for SimpleUpdateGen, by default {}.
    """
    print('\n ===================================')
    print('Running Z2 SU from U1-fPEPS:')
    print(f'Hamiltonian parameters: t={t}, U={U}, mu={mu}')
    print(f'fPEPS parameters: Lx={Lx}, Ly={Ly}, D={D}, N_f={N_f}')
    print(f'Chemical potential: {mu}\n')

    if not run_su:
        print('Just loading potentially existing SU fPEPS state...')
        # check if the file exists
        params_file = pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/peps_su_params.pkl'
        skeleton_file = pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/peps_skeleton.pkl'
        if os.path.exists(params_file) and os.path.exists(skeleton_file):
            print('Found existing files, loading...')
            skeleton = pickle.load(open(skeleton_file, 'rb'))
            peps_params = pickle.load(open(params_file, 'rb'))
            peps = qtn.unpack(peps_params, skeleton)
            return peps
    
    if u1peps is not None:
        peps = u1peps_to_z2peps(u1peps)
    else:
        # try loading existing U1-fPEPS
        print('Loading existing U1-fPEPS state...')
        params_file = pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}/peps_su_params.pkl'
        skeleton_file = pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}/peps_skeleton.pkl'
        if os.path.exists(params_file) and os.path.exists(skeleton_file):
            print('Found existing U1-fPEPS files, loading...')
            skeleton = pickle.load(open(skeleton_file, 'rb'))
            peps_params = pickle.load(open(params_file, 'rb'))
            u1peps = qtn.unpack(peps_params, skeleton)
            peps = u1peps_to_z2peps(u1peps)
        else:
            print('No existing U1-fPEPS files found, cannot proceed.')
            return None
    
    edges = qtn.edges_2d_square(Lx, Ly)
    site_info = sr.parse_edges_to_site_info(
        edges,
        D,
        phys_dim=4,
        site_ind_id="k{},{}",
        site_tag_id="I{},{}",
    )
    z2_terms = {
        (sitea, siteb): sr.fermi_hubbard_local_array(
            t=t,
            U=U,
            mu=mu,
            symmetry="Z2",
            coordinations=(
                site_info[sitea]["coordination"],
                site_info[siteb]["coordination"],
            ),
        )
        for (sitea, siteb) in edges
    }
    z2ham = qtn.LocalHam2D(Lx, Ly, H2=z2_terms)
    z2su = qtn.SimpleUpdateGen(
        peps, 
        z2ham, 
        D=D, 
        **su_kwargs
    )
    # Evolve the Z2-fPEPS
    for n_steps, tau in su_evolve_schedule:
        z2su.evolve(n_steps, tau=tau)
    
    fig, _ = z2su.plot()
    fig.savefig(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/su_evolution_U1SU.png')

    z2peps = z2su.get_state()
    z2peps.equalize_norms_(value=1)
    z2peps.exponent = 0.0
    # save the state
    params, skeleton = qtn.pack(z2peps)
    if save_file:
        os.makedirs(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}', exist_ok=True)
        with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/peps_skeleton_U1SU.pkl', 'wb') as f:
            pickle.dump(skeleton, f)
        with open(pwd+f'/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/peps_su_params_U1SU.pkl', 'wb') as f:
            pickle.dump(params, f)
    
    print('===================================')
    return z2peps


