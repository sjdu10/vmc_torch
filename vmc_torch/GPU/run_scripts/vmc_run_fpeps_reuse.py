"""GPU VMC for spinful Fermi-Hubbard with bMPS reuse.

Uses fPEPS_Model_reuse_GPU with cached boundary MPS environments
for incremental updates during sampling and energy evaluation.

Run:
    torchrun --nproc_per_node=<N> vmc_run_fpeps_reuse.py
    torchrun --nproc_per_node=1 vmc_run_fpeps_reuse.py
"""
import json
import os
from dataclasses import dataclass

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
    fPEPS_Model_reuse_GPU,
)
from vmc_torch.GPU.optimizer import (
    DecayScheduler,
    DistributedMinSRGPU,
    DistributedSRMinresGPU,
    MinSRGPU,
    SGDGPU,
)
from vmc_torch.GPU.sampler import (
    MetropolisExchangeSpinfulSamplerReuse_GPU,
)
from vmc_torch.GPU.vmc_setup import (
    initialize_walkers,
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.GPU.vmc_utils import (
    compute_grads_cheap_gpu,
    evaluate_energy_reuse,
    random_initial_config,
)

dtype = torch.float64
DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/GPU/data'
)


@dataclass
class VMCConfig:
    """VMC numerical / training settings."""

    batch_size: int = 256
    ns_per_rank: int = 256
    grad_batch_size: int = 128
    vmc_steps: int = 100
    learning_rate: float = 0.1
    diag_shift: float = 1e-4
    burn_in_steps: int = 4
    use_export_compile: bool = False  # full-contraction export
    use_export_compile_reuse: bool = False  # reuse export
    use_min_sr: bool = False
    use_distributed_min_sr: bool = False
    use_distributed_sr_minres: bool = True
    minres_sr_use_scipy: bool = False
    sr_rtol: float = 1e-4
    sr_maxiter: int = 100
    param_chunk_size: int = 1024
    save_every: int = 10
    resume_step: int = 0
    debug: bool = False
    outlier_clip_factor: float = 1e4
    run_sr: bool = True
    use_log_amp: bool = True
    use_cheap_grad: bool = True
    lr_scheduler: object = None  # set after construction
    verbose: bool = True

vmc_cfg = VMCConfig()
vmc_cfg.lr_scheduler = DecayScheduler(
    init_lr=vmc_cfg.learning_rate,
    decay_rate=0.9, patience=50,
)
warmup_cfg = VMCWarmupConfig(
    use_export_compile=VMCConfig.use_export_compile,
    grad_batch_size=VMCConfig.grad_batch_size,
    use_log_amp=vmc_cfg.use_log_amp,
    run_sampling=True,
    run_locE=True,
    run_grad=True,
)


def main():
    setup_linalg_hooks(jitter=1e-12)
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # ========== System parameters ==========
        Lx, Ly = 8, 8
        N_sites = Lx * Ly
        t = 1.0
        U = 8.0
        N_f = N_sites - 2  # 2 holes
        n_fermions_per_spin = (N_f // 2, N_f // 2)
        D = 4  # PEPS bond dimension
        chi = 16  # boundary bond dim

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

        # ========== Variational state (fPEPS reuse model) ==========
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
        model = fPEPS_Model_reuse_GPU(
            tn=peps,
            max_bond=chi,
            dtype=dtype,
            contract_boundary_opts={
                'mode': 'mps',
                # 'equalize_norms': 1.0,
                'canonize': True,
            },
        )
        model.to(device)

        # ========== Load checkpoint (optional) ==========
        output_dir = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/"
            f"t={t}_U={U}/N={N_f}/Z2/D={D}/chi={chi}/"
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
                f"Model: {N_params} params | "
                f"{world_size} GPUs | {device}"
            )
            print(
                f"System: {Lx}x{Ly} Fermi-Hubbard, "
                f"t={t}, U={U}, N_f={N_f}, D={D}, chi={chi}"
            )

        # ========== bMPS skeleton init (one-time) ==========
        example_x = random_initial_config(
            N_f, N_sites, seed=0,
        ).to(device)

        if rank == 0:
            print("Initializing bMPS skeleton...")
        model.cache_bMPS_skeleton(example_x)

        # ========== Export + compile reuse (optional) ==========
        if vmc_cfg.use_export_compile_reuse:
            if rank == 0:
                print("Exporting reuse patterns...")
            model.export_and_compile_reuse(
                example_x, mode='default',
                verbose=(rank == 0),
            )

        # ========== Export + compile full contraction ==========
        if vmc_cfg.use_export_compile:
            if rank == 0:
                print("Running torch.export + compile...")
            model.export_and_compile(
                example_x, mode='default',
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
            f'stats_{model_name}_reuse{step_tag}.json',
        )
        total_ns = vmc_cfg.ns_per_rank * world_size
        stats = {
            'system': (
                f'{Lx}x{Ly} Fermi-Hubbard, t={t}, U={U}, '
                f'N_f={N_f}, D={D}, chi={chi}'
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
            sampler=MetropolisExchangeSpinfulSamplerReuse_GPU(),
            preconditioner=preconditioner,
            optimizer=SGDGPU(
                learning_rate=vmc_cfg.learning_rate,
            ),
            evaluate_energy_fn=evaluate_energy_reuse,
            **(
                {'compute_grads_fn': compute_grads_cheap_gpu}
                if vmc_cfg.use_cheap_grad
                else {}
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
                f"t={t}, U={U}, N_f={N_f}, D={D}, chi={chi}"
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
