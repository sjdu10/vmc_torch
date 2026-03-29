"""GPU VMC for spinful Fermi-Hubbard with bMPS reuse.

Uses fPEPS_Model_reuse_GPU with cached boundary MPS environments
for incremental updates during sampling and energy evaluation.

Run:
    torchrun --nproc_per_node=<N> vmc_run_fpeps_reuse.py
    torchrun --nproc_per_node=1 vmc_run_fpeps_reuse.py
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'  # for better CPU multi-processing performance
os.environ['MKL_NUM_THREADS'] = '1'  # for better CPU multi-processing performance
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # for better CPU multi-processing performance
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
from vmc_torch.GPU.optimizer import DecayScheduler, SGDGPU
from vmc_torch.GPU.sampler import (
    MetropolisExchangeSpinfulSamplerReuse_GPU,
    MetropolisExchangeSpinfulSamplerXReuse_GPU,
)
from vmc_torch.GPU.vmc_setup import (
    initialize_walkers,
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.GPU.vmc_utils import (
    compute_grads_cheap_gpu,
    evaluate_energy_reuse,
    evaluate_energy_reuse_x,
    random_initial_config,
)
from vmcconfig import (
    VMCConfig,
    load_checkpoint,
    make_on_step_end,
    make_preconditioner,
    make_stats,
    make_stats_file,
    print_summary,
)

dtype = torch.float64
DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/GPU/data'
)


@dataclass
class ReuseCfg(VMCConfig):
    """Reuse-model specific settings."""
    use_export_compile_reuse: bool = False
    use_export_compile_cache: bool = False
    use_cheap_grad: bool = True
    use_x_only: bool = True

vmc_cfg = ReuseCfg(
    batch_size=1,
    ns_per_rank=1,
    grad_batch_size=1,
    vmc_steps=1,
    burn_in_steps=0,
    learning_rate=0.1,
    sr_diag_shift=5e-4,
    use_distributed_sr_minres=True,
    sr_rtol=1e-4,
    offload_grad_to_cpu=True,
    use_log_amp=True,
    use_export_compile=False,
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
    run_sampling=True,
    run_locE=True,
    run_grad=True,
)


def main():
    setup_linalg_hooks(jitter=1e-8)
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed(cpu=True)
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # ========== System parameters ==========
        Lx, Ly = 16, 8
        N_sites = Lx * Ly
        t = 1.0
        U = 8.0
        N_f = N_sites - 24 # 24 holes
        n_fermions_per_spin = (N_f // 2, N_f // 2)
        D = 8  # PEPS bond dimension
        chi = 4*D  # boundary bond dim

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
                # 'mode': 'fit',
                # 'tn_fit': 'zipup',
                # 'bsz':2,
                # 'max_iterations':5,
                # 'tol':1e-5,
                'mode': 'mps',
                'equalize_norms': 1.0,
                'canonize': True,
            },
            bold=False,
        )
        model.to(device)

        # ========== Setup ==========
        output_dir = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/"
            f"t={t}_U={U}/N={N_f}/Z2/D={D}/"
            f"{model._get_name()}/chi={chi}/"
        )
        import os
        os.makedirs(output_dir, exist_ok=True)
        model_name = model._get_name()
        N_params = sum(
            p.numel() for p in model.parameters()
        )

        load_checkpoint(
            model, output_dir, model_name,
            vmc_cfg.resume_step, device, rank,
        )

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

        # ========== Export + compile cache (optional) ==========
        if vmc_cfg.use_export_compile_cache:
            if rank == 0:
                print("Exporting cache functions...")
            cache_dirs = (
                ('x',) if vmc_cfg.use_x_only
                else ('x', 'y')
            )
            model.export_and_compile_cache(
                example_x, mode='default',
                verbose=(rank == 0),
                directions=cache_dirs,
            )

        # ========== Export + compile reuse (optional) ==========
        if vmc_cfg.use_export_compile_reuse:
            if rank == 0:
                print("Exporting reuse patterns...")
            cache_dirs = (
                ('x',) if vmc_cfg.use_x_only
                else ('x', 'y')
            )
            model.export_and_compile_reuse(
                example_x, mode='default',
                verbose=(rank == 0),
                use_log_amp=vmc_cfg.use_log_amp,
                directions=cache_dirs,
            )

        # ========== Export + compile full contraction ==========
        if vmc_cfg.use_export_compile:
            if rank == 0:
                print("Running torch.export + compile...")
            model.export_and_compile(
                example_x, mode='default',
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

        # ========== Stats + callback ==========
        system_str = (
            f'{Lx}x{Ly} Fermi-Hubbard, t={t}, U={U}, '
            f'N_f={N_f}, D={D}, chi={chi}'
        )
        stats_file = make_stats_file(
            output_dir, model_name,
            vmc_cfg.resume_step, suffix='_reuse',
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
        if vmc_cfg.use_x_only:
            sampler = (
                MetropolisExchangeSpinfulSamplerXReuse_GPU()
            )
            energy_fn = evaluate_energy_reuse_x
        else:
            sampler = (
                MetropolisExchangeSpinfulSamplerReuse_GPU()
            )
            energy_fn = evaluate_energy_reuse

        vmc = VMC_GPU(
            sampler=sampler,
            preconditioner=make_preconditioner(vmc_cfg),
            optimizer=SGDGPU(
                learning_rate=vmc_cfg.learning_rate,
            ),
            evaluate_energy_fn=energy_fn,
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
        print_summary(
            rank, energy_history,
            system_str, stats_file,
        )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
