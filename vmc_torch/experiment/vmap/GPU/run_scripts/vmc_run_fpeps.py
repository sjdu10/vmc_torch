"""GPU VMC for spinful Fermi-Hubbard on a square lattice (with data saving).

Same as vmc_run.py but saves energy stats to JSON and model checkpoints.

Run:
    torchrun --nproc_per_node=<N> vmc_run_fpeps.py
    torchrun --nproc_per_node=1 vmc_run_fpeps.py   # single GPU
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
from vmc_torch.experiment.vmap.GPU.models import fPEPS_Model_GPU
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
    ensure_output_dir,
    initialize_walkers,
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.experiment.vmap.GPU.vmc_utils import random_initial_config

dtype = torch.float64
DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/GPU/data'
)

@dataclass
class VMCConfig:
    """VMC numerical / training settings."""

    batch_size: int = 4096
    ns_per_rank: int = 4096
    grad_batch_size: int = 1024
    vmc_steps: int = 100
    learning_rate: float = 0.1
    diag_shift: float = 1e-4
    burn_in_steps: int = 4
    use_export_compile: bool = False
    use_min_sr: bool = False
    sr_rtol: float = 1e-4
    sr_maxiter: int = 100
    save_every: int = 10
    resume_step: int = 0  # checkpoint step to resume from, 0 = fresh
    debug: bool = False  # print [dbg] diagnostics each step
    outlier_clip_factor: float = 100.0  # drop O_loc outliers > factor * median
    run_sr: bool = True
    lr_scheduler: object = None  # set after construction


def main():
    setup_linalg_hooks(jitter=1e-12, qr_via_eigh=True)
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
        D = 4  # PEPS bond dimension
        chi = -1  # boundary bond dim

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

        # ========== Variational state (fPEPS model) ==========
        fpeps_base = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/t={t}_U={U}"
            f"/N={N_f}/Z2/D={D}/"
        )
        peps = load_or_generate_peps(
            Lx, Ly, t, U, N_f, D, seed=42, dtype=dtype, file_path=fpeps_base, scale_factor=4
        )
        model = fPEPS_Model_GPU(
            tn=peps,
            max_bond=chi,
            dtype=dtype,
            contract_boundary_opts={
                'mode': 'mps',
                'equalize_norms': 1.0,
                'canonize': True,
            },
        )
        model.to(device)

        # ========== Load checkpoint (optional) ==========
        vmc_cfg = VMCConfig()
        vmc_cfg.lr_scheduler = DecayScheduler(
            init_lr=vmc_cfg.learning_rate,
            decay_rate=0.9, patience=50,
        )
        output_dir = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/"
            f"t={t}_U={U}/N={N_f}/Z2/D={D}/chi={chi}/"
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
                f"Model: {N_params} params | "
                f"{world_size} GPUs | {device}"
            )

        # Export + compile (optional, ~10-40s one-time cost)
        if vmc_cfg.use_export_compile:
            example_x = random_initial_config(
                N_f, N_sites, seed=0,
            ).to(device)
            if rank == 0:
                print("Running torch.export + compile...")
            model.export_and_compile(example_x, mode='default')

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
            'system': f'{Lx}x{Ly} Fermi-Hubbard, t={t}, U={U}, '
                      f'N_f={N_f}, D={D}, chi={chi}',
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
                use_export_compile=vmc_cfg.use_export_compile,
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

            step = info['step']  # already global (offset applied)
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
                use_export_compile=vmc_cfg.use_export_compile,
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
