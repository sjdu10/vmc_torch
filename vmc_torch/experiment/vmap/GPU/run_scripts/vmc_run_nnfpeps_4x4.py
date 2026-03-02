"""GPU VMC with Conv2D-Geometric NN-fPEPS backflow model — 4x4 system.

Run:
    torchrun --nproc_per_node=1 run_scripts/vmc_run_nnfpeps_4x4.py
    torchrun --nproc_per_node=2 run_scripts/vmc_run_nnfpeps_4x4.py
"""
import json
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from vmc_torch.experiment.vmap.GPU.VMC import (
    VMC_GPU,
    VMCLoopConfig,
    VMCWarmupConfig,
    print_sampling_settings,
    setup_distributed,
)
from vmc_torch.experiment.vmap.GPU.hamiltonian import (
    spinful_Fermi_Hubbard_square_lattice_torch,
)
from vmc_torch.experiment.vmap.GPU.models import (
    Conv2D_Geometric_fPEPS_GPU,
)
from vmc_torch.experiment.vmap.GPU.optimizer import (
    DecayScheduler,
    DistributedSRMinresGPU,
    MinSRGPU,
    SGDGPU,
)
from vmc_torch.experiment.vmap.GPU.sampler import (
    MetropolisExchangeSpinfulSamplerGPU,
)
from vmc_torch.experiment.vmap.GPU.vmc_setup import (
    initialize_walkers,
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.experiment.vmap.GPU.vmc_utils import random_initial_config

dtype = torch.float64
USE_EXPORT_COMPILE = False

# Data paths
DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/experiment/vmap/GPU/data'
)
# SU-initialized PEPS from CPU vmap pipeline
CPU_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/experiment/vmap/data'
)


@dataclass
class VMCConfig:
    """VMC numerical / training settings."""

    batch_size: int = 4096*2
    ns_per_rank: int = 4096*2
    grad_batch_size: int = 1024
    vmc_steps: int = 100
    learning_rate: float = 0.1
    diag_shift: float = 1e-4
    burn_in_steps: int = 0
    use_min_sr: bool = False
    sr_rtol: float = 1e-4
    sr_maxiter: int = 100
    save_every: int = 50
    resume_step: int = 0
    debug: bool = False
    outlier_clip_factor: float = 100.0 # drop O_loc outliers > factor * median
    run_sr: bool = True
    lr_scheduler: object = None  # set after construction


def main():
    setup_linalg_hooks(jitter=1e-12)
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
            backbone_dtype=torch.float32,
            contract_boundary_opts={
                'mode': 'mps',
                'equalize_norms': 1.0,
                'canonize': True,
            },
        )
        model.to(device)

        # Export + compile (optional, ~10-40s one-time cost)
        if USE_EXPORT_COMPILE:
            example_x = random_initial_config(
                N_f, N_sites, seed=0,
            ).to(device)
            if rank == 0:
                print("Running torch.export + compile...")
            import time as _time
            _t0 = _time.time()
            model.export_and_compile(example_x)
            if rank == 0:
                print(
                    f"Export + compile done in "
                    f"{_time.time() - _t0:.1f}s"
                )

        # ========== Config ==========
        vmc_cfg = VMCConfig()
        vmc_cfg.lr_scheduler = DecayScheduler(
            init_lr=vmc_cfg.learning_rate,
            decay_rate=0.9, patience=50,
        )
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
                f"NN: {len(model._nn_param_names)} params"
            )
            print(
                f"nn_eta={nn_eta}, embed={embed_dim}, "
                f"hidden={hidden_dim}, "
                f"kernel={kernel_size}, layers={cnn_layers}"
            )
            print(
                f"backbone_dtype=float32"
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
                f'hidden={hidden_dim}, backbone_dtype=float32'
            ),
            'Np': N_params,
            'sample size': total_ns,
            'mean': [],
            'error': [],
            'variance': [],
        }

        # ========== VMC driver ==========
        preconditioner = (
            MinSRGPU()
            if vmc_cfg.use_min_sr
            else DistributedSRMinresGPU(
                rtol=vmc_cfg.sr_rtol,
                maxiter=vmc_cfg.sr_maxiter,
            )
        )
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
            config=VMCWarmupConfig(
                use_export_compile=USE_EXPORT_COMPILE,
                grad_batch_size=vmc_cfg.grad_batch_size,
            ),
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
            config=VMCLoopConfig(
                vmc_steps=vmc_cfg.vmc_steps,
                ns_per_rank=vmc_cfg.ns_per_rank,
                grad_batch_size=vmc_cfg.grad_batch_size,
                n_params=N_params,
                nsites=N_sites,
                learning_rate=vmc_cfg.learning_rate,
                diag_shift=vmc_cfg.diag_shift,
                burn_in_steps=vmc_cfg.burn_in_steps,
                run_sr=vmc_cfg.run_sr,
                use_min_sr=vmc_cfg.use_min_sr,
                use_export_compile=USE_EXPORT_COMPILE,
                step_offset=vmc_cfg.resume_step,
                debug=vmc_cfg.debug,
                outlier_clip_factor=vmc_cfg.outlier_clip_factor,
                lr_scheduler=vmc_cfg.lr_scheduler,
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
