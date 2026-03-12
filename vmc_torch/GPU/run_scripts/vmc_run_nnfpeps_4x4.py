"""GPU VMC with Conv2D-Geometric NN-fPEPS backflow model — 4x4 system.

Run:
    torchrun --nproc_per_node=1 run_scripts/vmc_run_nnfpeps_4x4.py
    torchrun --nproc_per_node=2 run_scripts/vmc_run_nnfpeps_4x4.py
"""
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
from vmc_torch.GPU.models import (
    Conv2D_Geometric_fPEPS_GPU,
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
    load_checkpoint,
    make_on_step_end,
    make_preconditioner,
    make_stats,
    make_stats_file,
    print_summary,
)

dtype = torch.float64
nnbackbone_dtype = torch.float64

# Data paths
DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/GPU/data'
)
# SU-initialized PEPS from CPU vmap pipeline
CPU_DATA_ROOT = DEFAULT_DATA_ROOT

vmc_cfg = VMCConfig(
    batch_size=2048,
    ns_per_rank=2048,
    grad_batch_size=1024,
    vmc_steps=1000,
    burn_in_steps=1,
    learning_rate=0.1,
    sr_diag_shift=5e-4,
    use_distributed_sr_minres=True,
    sr_rtol=1e-4,
    offload_grad_to_cpu=True,
    use_log_amp=True,
    use_export_compile=True,
    save_every=1,
    resume_step=0,
    verbose=False,
)
vmc_cfg.lr_scheduler = DecayScheduler(
    init_lr=vmc_cfg.learning_rate,
    decay_rate=0.9, patience=50,
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


def main():
    setup_linalg_hooks(
        jitter=1e-12, qr_via_eigh=True,
        cholesky_qr=False, cholesky_qr_adaptive_jitter=False,
    )
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # ========== System parameters ==========
        Lx, Ly = 4, 4
        N_sites = Lx * Ly
        t = 1.0
        U = 8.0
        N_f = N_sites - 2  # 2 holes -> 14 fermions
        n_fermions_per_spin = (N_f // 2, N_f // 2)
        D = 4   # PEPS bond dimension
        chi = -1  # exact contraction

        # NN backflow hyperparameters
        nn_eta = 1.0
        embed_dim = 16
        hidden_dim = N_sites
        kernel_size = 3
        cnn_layers = 1
        # ========== Hamiltonian ==========
        H = spinful_Fermi_Hubbard_square_lattice_torch(
            Lx,
            Ly,
            t,
            U,
            N_f,
            pbc=False,
            n_fermions_per_spin=n_fermions_per_spin,
            no_u1_symmetry=False,
            gpu=True,
        )
        H.precompute_hops_gpu(device)
        graph = H.graph

        # ========== Variational state (NN-fPEPS model) ==========
        fpeps_base = (
            f"{CPU_DATA_ROOT}/{Lx}x{Ly}/t={t}_U={U}"
            f"/N={N_f}/Z2/D={D}/"
        )
        peps = load_or_generate_peps(
            Lx, Ly, t, U, N_f, D,
            seed=42, dtype=dtype,
            file_path=fpeps_base,
            scale_factor=4,
        )
        # Set init_scale relative to fTN param magnitudes
        import quimb.tensor as qtn
        import quimb as qu
        _params, _ = qtn.pack(peps)
        _flat, _ = qu.utils.tree_flatten(_params, get_ref=True)
        ftn_params_mean = torch.mean(torch.stack([
            torch.as_tensor(p, dtype=dtype).abs().mean()
            for p in _flat
        ])).item()
        init_scale = 1e-2 * ftn_params_mean
        if rank == 0:
            print(f"ftn_params_mean={ftn_params_mean:.6e}, "
                  f"init_scale={init_scale:.6e}")
        model = Conv2D_Geometric_fPEPS_GPU(
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
        model.to(device)

        # Export + compile (optional, ~10-40s one-time cost)
        if vmc_cfg.use_export_compile:
            example_x = random_initial_config(
                N_f, N_sites, seed=0,
            ).to(device)
            if rank == 0:
                print("Running torch.export + compile...")
            import time as _time
            _t0 = _time.time()
            model.export_and_compile(
                example_x,
                use_log_amp=vmc_cfg.use_log_amp,
            )
            if rank == 0:
                print(
                    f"Export + compile done in "
                    f"{_time.time() - _t0:.1f}s"
                )

        # ========== Setup ==========
        output_dir = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/"
            f"t={t}_U={U}/N={N_f}/Z2/D={D}/{model._get_name()}/chi={chi}/"
        )
        import os
        os.makedirs(output_dir, exist_ok=True)
        model_name = model._get_name()
        N_params = sum(p.numel() for p in model.parameters())

        load_checkpoint(
            model, output_dir, model_name,
            vmc_cfg.resume_step, device, rank,
        )

        if rank == 0:
            print(
                f"Model: {model_name} | {N_params} params | "
                f"TN: {model.n_ftn} tensors, "
                f"NN: {sum(p.numel() for p in list(model.parameters())[-len(model._nn_param_names):])} params"
            )
            print(
                f"nn_eta={nn_eta}, embed={embed_dim}, "
                f"hidden={hidden_dim}, "
                f"kernel={kernel_size}, layers={cnn_layers}"
            )
            print(
                f"backbone_dtype={nnbackbone_dtype}, TN dtype={dtype}, "
            )
            print(
                f"{world_size} GPUs | {device}"
            )

        print_sampling_settings(
            rank,
            world_size,
            vmc_cfg.batch_size,
            vmc_cfg.ns_per_rank,
            vmc_cfg.grad_batch_size,
        )

        # ========== Initialize walkers ==========
        fxs = initialize_walkers(
            init_fn=lambda seed: random_initial_config(
                N_f, N_sites, seed=seed,
            ),
            batch_size=vmc_cfg.batch_size,
            seed=42, rank=rank, device=device,
        )

        # ========== Stats + callback ==========
        system_str = (
            f'{Lx}x{Ly} Fermi-Hubbard, t={t}, U={U}, '
            f'N_f={N_f}, D={D}, chi={chi}, '
            f'nn_eta={nn_eta}, embed={embed_dim}, '
            f'hidden={hidden_dim}, backbone_dtype={nnbackbone_dtype}, TN dtype={dtype}'
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

        # ========== VMC driver ==========
        vmc = VMC_GPU(
            sampler=MetropolisExchangeSpinfulSamplerGPU(),
            preconditioner=make_preconditioner(vmc_cfg),
            optimizer=SGDGPU(
                learning_rate=vmc_cfg.learning_rate,
            ),
        )

        fxs = vmc.run_warmup(
            fxs=fxs,
            model=model,
            graph=graph,
            hamiltonian=H,
            rank=rank,
            config=warmup_cfg,
        )

        energy_history, _ = vmc.run_vmc_loop(
            fxs=fxs,
            model=model,
            hamiltonian=H,
            graph=graph,
            rank=rank,
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
