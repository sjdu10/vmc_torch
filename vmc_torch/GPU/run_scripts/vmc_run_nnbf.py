"""GPU VMC with neural network backflow (MLP or Attention).

System: spinful Fermi-Hubbard on a square lattice.

Run:
    torchrun --nproc_per_node=1 run_scripts/vmc_run_nnbf.py
    torchrun --nproc_per_node=<N> run_scripts/vmc_run_nnbf.py
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
from vmc_torch.GPU.models import NNBF_GPU, AttentionNNBF_GPU
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
from vmc_torch.GPU.vmc_setup import initialize_walkers
from vmc_torch.GPU.vmc_utils import random_initial_config
from vmcconfig import VMCConfig

dtype = torch.float64
DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/GPU/data'
)

vmc_cfg = VMCConfig(
    vmc_steps=200,
    burn_in_steps=4,
    sr_diag_shift=1e-3,
    outlier_clip_factor=100.0,
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
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # ========== System parameters ==========
        Lx, Ly = 4, 2
        N_sites = Lx * Ly
        t = 1.0
        U = 8.0
        N_f = N_sites - 2  # 2 holes
        n_fermions_per_spin = (N_f // 2, N_f // 2)

        # ========== Hamiltonian ==========
        H = spinful_Fermi_Hubbard_square_lattice_torch(
            Lx, Ly, t, U, N_f,
            pbc=False,
            n_fermions_per_spin=n_fermions_per_spin,
            no_u1_symmetry=False,
            gpu=True,
        )
        H.precompute_hops_gpu(device)
        graph = H.graph

        # ========== Model ==========
        # Option A: MLP backflow
        model = NNBF_GPU(
            n_sites=N_sites, n_fermions=N_f,
            hidden_dim=32, n_layers=2,
            activation='tanh', bf_scale=0.01,
            dtype=dtype,
        )
        # Option B: Attention backflow (uncomment):
        # model = AttentionNNBF_GPU(
        #     n_sites=N_sites, n_fermions=N_f,
        #     d_model=32, n_heads=4, n_layers=2,
        #     bf_scale=0.01, dtype=dtype,
        # )
        model.to(device)

        # ========== Config ==========
        output_dir = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/"
            f"t={t}_U={U}/N={N_f}/nnbf/"
        )
        os.makedirs(output_dir, exist_ok=True)
        model_name = model._get_name()
        if vmc_cfg.resume_step > 0:
            ckpt_path = os.path.join(
                output_dir,
                f'checkpoint_{model_name}'
                f'_{vmc_cfg.resume_step}.pt',
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
                f"{world_size} GPUs | {device}"
            )
            print(
                f"System: {Lx}x{Ly} Fermi-Hubbard, "
                f"t={t}, U={U}, N_f={N_f}"
            )

        # ========== Export + compile (optional) ==========
        if vmc_cfg.use_export_compile:
            example_x = random_initial_config(
                N_f, N_sites, seed=0,
            ).to(device)
            if rank == 0:
                print("Running torch.export + compile...")
            model.export_and_compile(
                example_x,
                use_log_amp=vmc_cfg.use_log_amp,
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
                f'N_f={N_f}'
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
                f"Result: {Lx}x{Ly} Fermi-Hubbard, "
                f"t={t}, U={U}, N_f={N_f}"
            )
            print(f"Model: {model_name}")
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
