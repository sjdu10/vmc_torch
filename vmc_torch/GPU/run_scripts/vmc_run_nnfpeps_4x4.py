"""GPU VMC with Conv2D-Geometric NN-fPEPS backflow model — 4x4 system.

Run:
    torchrun --nproc_per_node=1 run_scripts/vmc_run_nnfpeps_4x4.py
    torchrun --nproc_per_node=2 run_scripts/vmc_run_nnfpeps_4x4.py
"""
import json
import os

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
from vmc_torch.GPU.optimizer import (
    DecayScheduler,
    DistributedMinSRGPU,
    DistributedSRMinresGPU,
    MinSRGPU,
    SGDGPU,
)
from vmc_torch.GPU.sampler import (
    MetropolisExchangeSpinfulSamplerGPU,
)
from vmc_torch.GPU.vmc_setup import (
    initialize_walkers,
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.GPU.vmc_utils import random_initial_config
from vmcconfig import VMCConfig

dtype = torch.float64
nnbackbone_dtype = torch.float64

# Data paths
DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/GPU/data'
)
# SU-initialized PEPS from CPU vmap pipeline
CPU_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/experiment/vmap/data'
)

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
    save_every=10,
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
)


def main():
    setup_linalg_hooks(
        jitter=1e-12, qr_via_eigh=False,
        cholesky_qr=True, cholesky_qr_adaptive_jitter=False,
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
        init_scale = 1e-5

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

        # ========== Config ==========
        output_dir = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/"
            f"t={t}_U={U}/N={N_f}/Z2/D={D}/chi={chi}/"
            f"nnfpeps_eta={nn_eta}_emb={embed_dim}"
            f"_hid={hidden_dim}/"
        )
        os.makedirs(output_dir, exist_ok=True)
        model_name = model._get_name()
        if vmc_cfg.resume_step > 0:
            ckpt_path = os.path.join(
                output_dir,
                f'checkpoint_{model_name}_{vmc_cfg.resume_step}.pt',
            )
            ckpt = torch.load(
                ckpt_path,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(ckpt)
            if rank == 0:
                print(f"Loaded checkpoint: {ckpt_path}")

        N_params = sum(p.numel() for p in model.parameters())
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

        # ========== Stats tracking ==========
        step_tag = (
            f'_from{vmc_cfg.resume_step}'
            if vmc_cfg.resume_step > 0 else ''
        )
        stats_file = os.path.join(
            output_dir,
            f'stats_{model_name}{step_tag}.json',
        )
        total_ns = vmc_cfg.ns_per_rank * world_size
        stats = {
            'system': (
                f'{Lx}x{Ly} Fermi-Hubbard, t={t}, U={U}, '
                f'N_f={N_f}, D={D}, chi={chi}, '
                f'nn_eta={nn_eta}, embed={embed_dim}, '
                f'hidden={hidden_dim}, backbone_dtype={nnbackbone_dtype}, TN dtype={dtype}'
            ),
            'Np': N_params,
            'sample size': total_ns,
            'mean': [],
            'error': [],
            'variance': [],
        }

        # ========== VMC driver ==========
        if vmc_cfg.use_distributed_min_sr:
            preconditioner = DistributedMinSRGPU(
                param_chunk_size=vmc_cfg.param_chunk_size,
            )
        elif vmc_cfg.use_min_sr:
            preconditioner = MinSRGPU()
        elif vmc_cfg.use_distributed_sr_minres:
            preconditioner = DistributedSRMinresGPU(
                rtol=vmc_cfg.sr_rtol,
                maxiter=vmc_cfg.sr_maxiter,
                use_scipy=vmc_cfg.minres_sr_use_scipy,
            )
        else:
            preconditioner = None
            
        vmc = VMC_GPU(
            sampler=MetropolisExchangeSpinfulSamplerGPU(),
            preconditioner=preconditioner,
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

        # ========== Data-saving callback ==========
        def on_step_end(info):
            if rank != 0:
                return
            stats['mean'].append(info['energy_per_site'])
            stats['error'].append(info['error_per_site'])
            stats['variance'].append(info['energy_var'])
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)

            step = info['step']
            if (step + 1) % vmc_cfg.save_every == 0:
                ckpt_path = os.path.join(
                    output_dir,
                    f'checkpoint_{model_name}_{step + 1}.pt',
                )
                torch.save(model.state_dict(), ckpt_path)

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

        # ========== Summary ==========
        if rank == 0 and energy_history:
            print(f"\n{'=' * 50}")
            print(
                f"Result: {Lx}x{Ly} Fermi-Hubbard, t={t}, U={U}, "
                f"N_f={N_f}, D={D}, chi={chi}"
            )
            print(
                f"NN-fPEPS: nn_eta={nn_eta}, embed={embed_dim}, "
                f"hidden={hidden_dim}, backbone_dtype=float32"
            )
            print(f"{'=' * 50}")
            print(f"First E/site: {energy_history[0]:.6f}")
            print(f"Last  E/site: {energy_history[-1]:.6f}")
            print(f"Min   E/site: {min(energy_history):.6f}")
            print(f"Stats saved to: {stats_file}")
            if energy_history[-1] < energy_history[0]:
                print("\nEnergy decreased.")
            else:
                print("\nWARNING: Energy did NOT decrease.")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
