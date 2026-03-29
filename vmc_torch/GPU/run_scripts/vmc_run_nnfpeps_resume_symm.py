"""Resume from a pre-trained NN-fPEPS and add symmetry projection.

Strategy: train without symmetry until convergence, then wrap with
fermionic C4v/C2v projection and continue training. This avoids the
optimization difficulty of training symmetry-projected models from
scratch.

Run:
    torchrun --nproc_per_node=1 run_scripts/vmc_run_nnfpeps_resume_symm.py
    torchrun --nproc_per_node=2 run_scripts/vmc_run_nnfpeps_resume_symm.py
"""
import os
import time

import torch
import torch.distributed as dist

from vmc_torch.GPU.VMC import (
    VMC_GPU,
    VMCLoopConfig,
    VMCWarmupConfig,
    print_sampling_settings,
    setup_distributed,
)
from vmc_torch.GPU.hamiltonian import (
    spinful_Fermi_Hubbard_square_lattice_torch,
)
from vmc_torch.GPU.models import Conv2D_Geometric_fPEPS_GPU
from vmc_torch.GPU.models.symmetry import (
    FermionSymmetryProjectedModel,
)
from vmc_torch.GPU.optimizer import DecayScheduler, SGDGPU
from vmc_torch.GPU.sampler import (
    MetropolisExchangeSpinfulSamplerGPU,
)
from vmc_torch.GPU.vmc_setup import (
    initialize_walkers,
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.GPU.vmc_utils import random_initial_config
from vmc_torch.GPU.run_scripts.vmcconfig import (
    VMCConfig,
    make_on_step_end,
    make_preconditioner,
    make_stats,
    make_stats_file,
    print_summary,
)


# =============================================================
#  Paths
# =============================================================

DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/GPU/data'
)


# =============================================================
#  System parameters
# =============================================================

Lx, Ly = 4, 4
N_sites = Lx * Ly
t = 1.0
U = 8.0
N_f = N_sites - 2
n_fermions_per_spin = (N_f // 2, N_f // 2)
D = 4
chi = -1
irrep = 'B1'

# NN backflow hyperparameters
nn_eta = 1.0
embed_dim = 16
hidden_dim = 4 * N_sites
kernel_size = 3
cnn_layers = 1

dtype = torch.float64
nnbackbone_dtype = torch.float64


# =============================================================
#  Pre-trained checkpoint to load (without symmetry)
# =============================================================

# Directory and step of the non-symmetry checkpoint
PRETRAINED_DIR = (
    f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/"
    f"t={t}_U={U}/N={N_f}/Z2/D={D}/"
    f"Conv2D_Geometric_fPEPS_GPU/chi={chi}/"
)
PRETRAINED_STEP = 610  # which step to load from


# =============================================================
#  VMC training config (for the symmetry-projected phase)
# =============================================================

vmc_cfg = VMCConfig(
    batch_size=4096,
    ns_per_rank=4096,
    grad_batch_size=512,
    vmc_steps=1000,
    burn_in_steps=1,
    learning_rate=0.01,       # smaller LR for fine-tuning
    sr_diag_shift=5e-4,
    use_distributed_sr_minres=True,
    sr_rtol=1e-4,
    offload_grad_to_cpu=False,
    use_log_amp=True,
    use_export_compile=True,
    save_every=10,
    resume_step=0,            # step within THIS run (0 = fresh)
    verbose=False,
)
vmc_cfg.lr_scheduler = DecayScheduler(
    init_lr=vmc_cfg.learning_rate,
    decay_rate=0.9, patience=100,
)

warmup_cfg = VMCWarmupConfig(
    use_export_compile=vmc_cfg.use_export_compile,
    grad_batch_size=vmc_cfg.grad_batch_size,
    use_log_amp=vmc_cfg.use_log_amp,
    offload_grad_to_cpu=vmc_cfg.offload_grad_to_cpu,
    run_sampling=False,
    run_locE=False,
    run_grad=True,
)


# =============================================================
#  Helpers
# =============================================================


def load_pretrained_checkpoint(model, ckpt_dir, step, device,
                               rank):
    """Load a non-symmetry checkpoint into the base model.

    Args:
        model: the base model (Conv2D_Geometric_fPEPS_GPU),
            NOT the symmetry wrapper.
        ckpt_dir: directory containing the checkpoint.
        step: training step to load.
        device: torch device.
        rank: distributed rank.
    """
    # Match the naming convention from vmc_run_nnfpeps_4x4.py
    model_name = model._get_name()
    ckpt_path = os.path.join(
        ckpt_dir,
        f'checkpoint_{model_name}_{step}.pt',
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}"
        )
    ckpt = torch.load(
        ckpt_path, map_location=device, weights_only=True,
    )
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    if rank == 0:
        print(f"Loaded pre-trained checkpoint: {ckpt_path}")


def load_symm_checkpoint(model, ckpt_dir, model_name, step,
                         device, rank):
    """Load a symmetry-run checkpoint to resume training.

    Args:
        model: the FermionSymmetryProjectedModel wrapper.
        ckpt_dir: output directory for this symm run.
        model_name: name used in checkpoint filenames.
        step: step to resume from (within this symm run).
        device: torch device.
        rank: distributed rank.
    """
    if step <= 0:
        return
    ckpt_path = os.path.join(
        ckpt_dir,
        f'checkpoint_{model_name}_{step}.pt',
    )
    ckpt = torch.load(
        ckpt_path, map_location=device, weights_only=True,
    )
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    if rank == 0:
        print(f"Resumed symm checkpoint: {ckpt_path}")


# =============================================================
#  Main
# =============================================================


def main():
    # --- Runtime setup ---
    setup_linalg_hooks(
        jitter=1e-8, qr_via_eigh=True,
        cholesky_qr=False,
        cholesky_qr_adaptive_jitter=False,
    )
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # --- Hamiltonian ---
        H = spinful_Fermi_Hubbard_square_lattice_torch(
            Lx, Ly, t, U, N_f,
            pbc=False,
            n_fermions_per_spin=n_fermions_per_spin,
            no_u1_symmetry=False,
            gpu=True,
        )
        H.precompute_hops_gpu(device)
        graph = H.graph

        # --- Build base model (same arch as pre-trained) ---
        fpeps_base = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/t={t}_U={U}"
            f"/N={N_f}/Z2/D={D}/"
        )
        peps = load_or_generate_peps(
            Lx, Ly, t, U, N_f, D,
            seed=42, dtype=dtype,
            file_path=fpeps_base,
            scale_factor=4,
        )
        import quimb.tensor as qtn
        import quimb as qu
        _params, _ = qtn.pack(peps)
        _flat, _ = qu.utils.tree_flatten(
            _params, get_ref=True,
        )
        ftn_params_mean = torch.mean(torch.stack([
            torch.as_tensor(p, dtype=dtype).abs().mean()
            for p in _flat
        ])).item()
        init_scale = 1e-2 * ftn_params_mean

        base_model = Conv2D_Geometric_fPEPS_GPU(
            tn=peps,
            max_bond=chi,
            nn_eta=nn_eta,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            layers=cnn_layers,
            init_scale=init_scale,
            dtype=dtype,
            backbone_dtype=nnbackbone_dtype,
            contract_boundary_opts={
                'mode': 'mps',
                'equalize_norms': 1.0,
                'canonize': True,
            },
        )
        base_model.to(device)

        # --- Load pre-trained weights (no symmetry) ---
        load_pretrained_checkpoint(
            base_model, PRETRAINED_DIR,
            PRETRAINED_STEP, device, rank,
        )

        # --- Wrap with fermionic symmetry projection ---
        group_name = 'C4v' if Lx == Ly else 'C2v'
        model = FermionSymmetryProjectedModel(
            base_model, Lx, Ly, irrep=irrep,
        )

        N_params = sum(
            p.numel() for p in model.parameters()
        )

        # --- Output directory ---
        output_dir = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/"
            f"t={t}_U={U}/N={N_f}/Z2/D={D}/"
            f"{base_model._get_name()}/chi={chi}/"
            f"symm_{irrep}_from{PRETRAINED_STEP}/"
        )
        os.makedirs(output_dir, exist_ok=True)
        model_name = (
            f"FermSymm_{irrep}_"
            f"{base_model._get_name()}"
        )

        # Resume within this symm run (if restarting)
        load_symm_checkpoint(
            model, output_dir, model_name,
            vmc_cfg.resume_step, device, rank,
        )

        # --- Print info ---
        if rank == 0:
            print(
                f"System: {Lx}x{Ly} Fermi-Hubbard, "
                f"t={t}, U={U}, N_f={N_f} "
                f"({n_fermions_per_spin[0]}up+"
                f"{n_fermions_per_spin[1]}dn)"
            )
            print(
                f"Model: NN-fPEPS D={D}, chi={chi}, "
                f"{group_name} {irrep} projection, "
                f"{N_params} params | "
                f"{world_size} GPUs | {device}"
            )
            print(
                f"Pre-trained from step {PRETRAINED_STEP} "
                f"(no symmetry)"
            )
            print(
                f"nn_eta={nn_eta}, embed={embed_dim}, "
                f"hidden={hidden_dim}, "
                f"kernel={kernel_size}, layers={cnn_layers}"
            )

        # --- Export + compile ---
        example_x = random_initial_config(
            N_f, N_sites, seed=0,
        ).to(device)
        if vmc_cfg.use_export_compile:
            if rank == 0:
                print("Running torch.export + compile...")
            _t0 = time.time()
            model.export_and_compile(
                example_x,
                use_log_amp=vmc_cfg.use_log_amp,
            )
            if rank == 0:
                print(
                    f"Export + compile done in "
                    f"{time.time() - _t0:.1f}s"
                )

        print_sampling_settings(
            rank, world_size, vmc_cfg.batch_size,
            vmc_cfg.ns_per_rank, vmc_cfg.grad_batch_size,
        )

        # --- Initialize walkers ---
        fxs = initialize_walkers(
            init_fn=lambda seed: random_initial_config(
                N_f, N_sites, seed=seed,
            ),
            batch_size=vmc_cfg.batch_size,
            seed=42, rank=rank, device=device,
        )

        # --- Stats + callback ---
        system_str = (
            f'{Lx}x{Ly} Fermi-Hubbard, t={t}, U={U}, '
            f'N_f={N_f}, D={D}, chi={chi}, '
            f'{group_name} {irrep} '
            f'(from pretrained step {PRETRAINED_STEP}), '
            f'nn_eta={nn_eta}, embed={embed_dim}, '
            f'hidden={hidden_dim}'
        )
        stats_file = make_stats_file(
            output_dir, model_name,
            vmc_cfg.resume_step,
        )
        stats = make_stats(
            system_str, N_params,
            vmc_cfg.ns_per_rank, world_size,
        )
        on_step_end = make_on_step_end(
            rank, stats, stats_file, output_dir,
            model_name, model, vmc_cfg.save_every,
        )

        # --- VMC driver ---
        vmc = VMC_GPU(
            sampler=MetropolisExchangeSpinfulSamplerGPU(),
            preconditioner=make_preconditioner(vmc_cfg),
            optimizer=SGDGPU(
                learning_rate=vmc_cfg.learning_rate,
            ),
        )

        fxs = vmc.run_warmup(
            fxs=fxs, model=model, graph=graph,
            hamiltonian=H, rank=rank,
            config=warmup_cfg,
        )
        energy_history, _ = vmc.run_vmc_loop(
            fxs=fxs, model=model, hamiltonian=H,
            graph=graph, rank=rank,
            world_size=world_size,
            config=VMCLoopConfig.from_vmc_config(
                vmc_cfg,
                n_params=N_params,
                nsites=N_sites,
            ),
            on_step_end=on_step_end,
        )

        print_summary(
            rank, energy_history,
            system_str, stats_file,
        )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
