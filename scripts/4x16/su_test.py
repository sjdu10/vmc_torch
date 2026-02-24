import symmray as sr
import quimb.tensor as qtn
from vmc_torch.fermion_utils import generate_random_fpeps_symmray

Lx, Ly = 4, 16
edges = qtn.edges_2d_square(Lx, Ly)
D=8
site_info = sr.parse_edges_to_site_info(
    edges,
    D,
    phys_dim=4,
    site_ind_id="k{},{}",
    site_tag_id="I{},{}",
)
u1_terms = {
    (sitea, siteb): sr.fermi_hubbard_local_array(
        t=1,
        U=8,
        mu=0.0,
        symmetry="U1",
        coordinations=(
            site_info[sitea]["coordination"],
            site_info[siteb]["coordination"],
        ),
    )
    for (sitea, siteb) in edges
}

u1ham = qtn.LocalHam2D(Lx, Ly, H2=u1_terms)
# fpeps = sr.PEPS_fermionic_rand(
#     "U1",
#     Lx,
#     Ly,
#     bond_dim=D,
#     phys_dim=4
# )
fpeps = generate_random_fpeps_symmray(
    Lx,
    Ly,
    D=D,
    seed=42,
    symmetry="U1",
    Nf=Lx*Ly-8,
    spinless=False,
)

su_kwargs = {
    # "compute_energy_per_site": True,
    # 'compute_energy_opts':{"max_distance":1}, 
    # 'compute_energy_every':50,
    'gate_opts':{'cutoff':1e-12},
    'equilibrate_start': False,
}

u1su = qtn.SimpleUpdateGen(
        fpeps, 
        u1ham, 
        D=D, 
        **su_kwargs
)
u1su.evolve(10, tau=0.01)