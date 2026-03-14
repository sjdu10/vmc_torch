"""GPU VMC for spin-1/2 Heisenberg with bMPS reuse.

Uses PEPS_Model_reuse_GPU with cached boundary MPS environments
for incremental updates during sampling.

Run:
    torchrun --nproc_per_node=<N> vmc_run_spin_reuse.py
    torchrun --nproc_per_node=1 vmc_run_spin_reuse.py
"""
import os
import time
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
from vmc_torch.GPU.vmc_utils import (
    compute_grads_cheap_gpu,
    evaluate_energy_reuse,
    evaluate_energy_reuse_x,
)
from vmc_torch.hamiltonian_torch import (
    spin_Heisenberg_square_lattice_torch,
)
from vmc_torch.GPU.models import (
    PEPS_Model_reuse_GPU,
)
from vmc_torch.GPU.optimizer import DecayScheduler, SGDGPU
from vmc_torch.GPU.sampler import (
    MetropolisExchangeSpinSamplerReuse_GPU,
    MetropolisExchangeSpinSamplerXReuse_GPU,
)
from vmc_torch.GPU.vmc_setup import (
    generate_random_spin_peps,
    initialize_walkers,
    random_spin_config_sz0,
    setup_linalg_hooks,
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
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/GPU/data'
)


@dataclass
class ReuseCfg(VMCConfig):
    """Reuse-model specific settings."""
    use_export_compile_reuse: bool = False
    use_export_compile_cache: bool = True
    use_cheap_grad: bool = True
    use_x_only: bool = True


vmc_cfg = ReuseCfg(
    batch_size=2048,
    ns_per_rank=2048,
    grad_batch_size=512,
    vmc_steps=10,
    burn_in_steps=2,
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
    run_locE=False,
    run_grad=False,
)


def main():
    setup_linalg_hooks(jitter=1e-16)
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # ========== System parameters ==========
        Lx, Ly = 4, 4
        N_sites = Lx * Ly
        J = 1.0
        D = 4
        chi = 4*D

        # ========== Hamiltonian ==========
        H = spin_Heisenberg_square_lattice_torch(
            Lx, Ly, J=J, total_sz=0,
        )
        H.precompute_hops_gpu(device)

        # ED reference for small systems
        if rank == 0 and N_sites <= 16:
            H_dense = H.to_dense()
            import scipy.sparse.linalg as la
            gs_e = la.eigsh(
                H_dense, k=1, which='SA', tol=1e-8,
            )[0][0]
            print(
                f"ED ground state E/site: "
                f"{gs_e / N_sites:.8f}"
            )

        # ========== Model ==========
        peps = generate_random_spin_peps(
            Lx, Ly, D, seed=42, dtype=dtype,
        )
        model = PEPS_Model_reuse_GPU(
            tn=peps,
            max_bond=chi,
            dtype=dtype,
            contract_boundary_opts={
                'mode': 'mps',
                'canonize': True,
                'equalize_norms': 1.0,
            },
            bold=True,
        )
        model.to(device)

        # ========== Setup ==========
        output_dir = (
            f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/"
            f"J={J}/D={D}/{model._get_name()}/chi={chi}/"
        )
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
                f"System: {Lx}x{Ly} Heisenberg, "
                f"J={J}, Sz=0"
            )
            print(
                f"Model: {model_name} D={D}, "
                f"chi={chi}, {N_params} params | "
                f"{world_size} GPUs | {device}"
            )

        # ========== bMPS skeleton + reuse compile ==========
        example_x = random_spin_config_sz0(
            N_sites, seed=0,
        ).to(device)
        t0 = time.time()
        model.cache_bMPS_skeleton(example_x)
        print(f'Cached skeleton time={time.time()-t0}')

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
        if vmc_cfg.use_export_compile:
            if rank == 0:
                print("Running torch.export + compile...")
            model.export_and_compile(
                example_x, mode='default',
                use_log_amp=vmc_cfg.use_log_amp,
            )

        print_sampling_settings(
            rank, world_size, vmc_cfg.batch_size,
            vmc_cfg.ns_per_rank, vmc_cfg.grad_batch_size,
        )

        # ========== Initialize walkers ==========
        fxs = initialize_walkers(
            init_fn=lambda seed: random_spin_config_sz0(
                N_sites, seed=seed,
            ),
            batch_size=vmc_cfg.batch_size,
            seed=42, rank=rank, device=device,
        )

        # ========== Stats + callback ==========
        system_str = (
            f'{Lx}x{Ly} Heisenberg, J={J}, '
            f'D={D}, chi={chi}'
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
                MetropolisExchangeSpinSamplerXReuse_GPU()
            )
            energy_fn = evaluate_energy_reuse_x
        else:
            sampler = (
                MetropolisExchangeSpinSamplerReuse_GPU()
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
            fxs=fxs, model=model, graph=H.graph,
            hamiltonian=H, rank=rank, config=warmup_cfg,
        )
        energy_history, fxs = vmc.run_vmc_loop(
            fxs=fxs, model=model, hamiltonian=H,
            graph=H.graph, rank=rank,
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
