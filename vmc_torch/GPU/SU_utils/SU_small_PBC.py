"""PBC SU pipeline: U1 SU on torus, then convert to Z2 + short Z2 SU.

Mirror of ``SU_small.py`` with ``pbc=True``. Output files land in a
``PBC/`` subdirectory under the standard layout, e.g.

    {pwd}/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/U1/D={D}/PBC/peps_su_params.pkl
    {pwd}/{Lx}x{Ly}/t={t}_U={U}/N={N_f}/Z2/D={D}/PBC/peps_su_params_U1SU.pkl

so OBC SU outputs are not overwritten. To consume the resulting
torus PEPS from a VMC run script, point
``load_or_generate_peps(..., file_path=..., pbc=True)`` at the PBC
subdirectory.
"""
from SU_func import (
    run_u1SU,
    run_z2SU_from_u1SU,
)

Lx, Ly = 4, 4   # use Lx, Ly >= 3 for genuine torus (no degenerate wrap)
D0 = 4
N_f = Lx * Ly
t = 1.0
U = 8.0
mu = 0.0
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/GPU/data'
seed = 42
pbc = True

# # --- U1 SU on torus ---
# su_kwargs = {
#     "compute_energy_per_site": True,
#     'compute_energy_final': False,
#     'compute_energy_opts': {"max_distance": 1},
#     'compute_energy_every': None,
#     'gate_opts': {'cutoff': 0.0},
#     'ordering': 'smallest_last',
#     'tol': 1e-6,
# }
# D1 = 4
# su_evolve_schedule = [
#     (100, 0.05),
#     # (100, 0.01),
#     # (100, 0.001),
# ]
# u1peps = run_u1SU(
#     Lx=Lx,
#     Ly=Ly,
#     D=D1,
#     N_f=N_f,
#     t=t, U=U, mu=mu,
#     pwd=pwd,
#     seed=seed,
#     initial_peps=None,
#     rfpeps_kwargs={'subsizes': 'equal', 'u1_all_even': True},
#     save_file=True,
#     run_su=True,
#     su_evolve_schedule=su_evolve_schedule,
#     pbc=pbc,
#     **su_kwargs,
# )

# --- Convert to Z2 and do a short Z2 SU ---
su_kwargs = {
    "compute_energy_per_site": True,
    'compute_energy_final': False,
    'compute_energy_opts': {"max_distance": 1},
    'compute_energy_every': 25,
    'gate_opts': {'cutoff': 0.0},
    'ordering': 'smallest_last',
    'tol': 1e-6,
}
D2 = 4
su_evolve_schedule = [
    (50, 0.1),
    (50, 0.01),
]
z2peps = run_z2SU_from_u1SU(
    Lx=Lx,
    Ly=Ly,
    D=D2,
    N_f=N_f,
    t=t, U=U, mu=U/2,
    pwd=pwd,
    # u1peps=u1peps,
    save_file=True,
    run_su=True,
    su_evolve_schedule=su_evolve_schedule,
    pbc=pbc,
    **su_kwargs,
)
