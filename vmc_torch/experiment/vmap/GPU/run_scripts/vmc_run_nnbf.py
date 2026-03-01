"""GPU VMC comparison: MLP backflow vs Attention backflow (depth sweep).

Runs NNBF_GPU (MLP, 2 layers) as baseline, then AttentionNNBF_GPU with
n_layers = 1, 2, 3, 4 on 4x2 spinful Fermi-Hubbard for 50 VMC steps.
Prints a side-by-side energy comparison table at the end.

System: 4x2 Fermi-Hubbard, t=1, U=8, N_f=6 (2 holes), OBC.

Run:
    torchrun --nproc_per_node=1 run_scripts/vmc_run_nnbf.py
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
from vmc_torch.experiment.vmap.GPU.models import NNBF_GPU, AttentionNNBF_GPU
from vmc_torch.experiment.vmap.GPU.optimizer import (
    DecayScheduler, MinSRGPU, SGDGPU,
)
from vmc_torch.experiment.vmap.GPU.sampler import (
    MetropolisExchangeSpinfulSamplerGPU,
)
from vmc_torch.experiment.vmap.GPU.vmc_setup import initialize_walkers
from vmc_torch.experiment.vmap.GPU.vmc_utils import random_initial_config

dtype = torch.float64
DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/experiment/vmap/GPU/data'
)

# ========== System parameters ==========
Lx, Ly = 4, 2
N_sites = Lx * Ly
t = 1.0
U = 8.0
N_f = N_sites - 2  # 2 holes
n_fermions_per_spin = (N_f // 2, N_f // 2)


@dataclass
class VMCConfig:
    """VMC numerical / training settings."""
    batch_size: int = 1024
    ns_per_rank: int = 1024
    grad_batch_size: int = 1024
    vmc_steps: int = 200
    learning_rate: float = 0.1
    diag_shift: float = 1e-3
    burn_in_steps: int = 4
    use_export_compile: bool = False
    use_min_sr: bool = False
    outlier_clip_factor: float = 100.0 # drop O_loc outliers > factor * median
    run_sr: bool = True
    lr_scheduler: object = None  # set after construction


def run_one_model(
    model, model_label, H, graph, device, rank, world_size,
):
    """Run VMC for one model and return energy history + stats."""
    cfg = VMCConfig()
    cfg.lr_scheduler = DecayScheduler(
        init_lr=cfg.learning_rate,
        decay_rate=0.9, patience=50,
    )
    N_params = sum(p.numel() for p in model.parameters())

    if rank == 0:
        print(f"\n{'=' * 55}")
        print(f"  {model_label}: {N_params} parameters")
        print(f"{'=' * 55}")

    print_sampling_settings(
        rank, world_size,
        cfg.batch_size, cfg.ns_per_rank, cfg.grad_batch_size,
    )

    # Initialize walkers (fresh for each model, same seed)
    fxs = initialize_walkers(
        init_fn=lambda seed: random_initial_config(
            N_f, N_sites, seed=seed,
        ),
        batch_size=cfg.batch_size,
        seed=42, rank=rank, device=device,
    )

    # VMC driver
    vmc = VMC_GPU(
        sampler=MetropolisExchangeSpinfulSamplerGPU(),
        preconditioner=MinSRGPU(),
        optimizer=SGDGPU(learning_rate=cfg.learning_rate),
    )

    # Warmup
    fxs = vmc.run_warmup(
        fxs=fxs, model=model, graph=graph,
        hamiltonian=H, rank=rank,
        config=VMCWarmupConfig(
            use_export_compile=cfg.use_export_compile,
            grad_batch_size=cfg.grad_batch_size,
        ),
    )

    # Collect per-step stats
    stats = {'mean': [], 'error': [], 'variance': []}

    def on_step_end(info):
        if rank != 0:
            return
        stats['mean'].append(info['energy_per_site'])
        stats['error'].append(info['error_per_site'])
        stats['variance'].append(info['energy_var'])

    energy_history, fxs = vmc.run_vmc_loop(
        fxs=fxs, model=model, hamiltonian=H,
        graph=graph, rank=rank, world_size=world_size,
        config=VMCLoopConfig(
            vmc_steps=cfg.vmc_steps,
            ns_per_rank=cfg.ns_per_rank,
            grad_batch_size=cfg.grad_batch_size,
            n_params=N_params,
            nsites=N_sites,
            learning_rate=cfg.learning_rate,
            diag_shift=cfg.diag_shift,
            burn_in_steps=cfg.burn_in_steps,
            run_sr=cfg.run_sr,
            use_min_sr=cfg.use_min_sr,
            use_export_compile=cfg.use_export_compile,
            step_offset=0,
            outlier_clip_factor=cfg.outlier_clip_factor,
            lr_scheduler=cfg.lr_scheduler,
        ),
        on_step_end=on_step_end,
    )
    return energy_history, stats


def main():
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)

        if rank == 0:
            print(
                f"System: {Lx}x{Ly} Fermi-Hubbard, t={t}, U={U}, "
                f"N_f={N_f}, {world_size} GPU(s)"
            )

        # ========== Hamiltonian (shared) ==========
        H = spinful_Fermi_Hubbard_square_lattice_torch(
            Lx, Ly, t, U, N_f,
            pbc=False,
            n_fermions_per_spin=n_fermions_per_spin,
            no_u1_symmetry=False,
            gpu=True,
        )
        H.precompute_hops_gpu(device)
        graph = H.graph

        # ========== Models to compare ==========
        # (label, model_builder)
        attn_depths = [1, 2, 3, 4]
        models_to_run = [
            (
                "MLP (h=32, L=2)",
                lambda: NNBF_GPU(
                    n_sites=N_sites, n_fermions=N_f,
                    hidden_dim=32, n_layers=2,
                    activation='tanh', bf_scale=0.01, dtype=dtype,
                ),
            ),
        ]
        for L in attn_depths:
            models_to_run.append((
                f"Attn (d=32, L={L})",
                lambda L=L: AttentionNNBF_GPU(
                    n_sites=N_sites, n_fermions=N_f,
                    d_model=32, n_heads=4, n_layers=L,
                    bf_scale=0.01, dtype=dtype,
                ),
            ))

        # ========== Run all models ==========
        results = {}  # label -> {Np, hist, stats}
        for label, builder in models_to_run:
            torch.manual_seed(42 + rank)
            model = builder()
            model.to(device)
            Np = sum(p.numel() for p in model.parameters())
            torch.manual_seed(42 + rank)
            hist, stats = run_one_model(
                model, label, H, graph, device, rank, world_size,
            )
            results[label] = {
                'Np': Np, 'hist': hist, 'stats': stats,
            }

        # ========== Summary table ==========
        if rank == 0:
            print(f"\n{'=' * 72}")
            print(
                f"  Depth sweep: {Lx}x{Ly} Fermi-Hubbard, "
                f"t={t}, U={U}, N_f={N_f}, 50 VMC steps"
            )
            print(f"{'=' * 72}")

            # Summary row per model
            print(
                f"\n{'Model':<22} {'Np':>6}  "
                f"{'E/site(1)':>10}  {'E/site(50)':>10}  "
                f"{'min E/site':>10}"
            )
            print("-" * 68)
            for label, r in results.items():
                h = r['hist']
                if h:
                    print(
                        f"{label:<22} {r['Np']:>6}  "
                        f"{h[0]:>10.6f}  {h[-1]:>10.6f}  "
                        f"{min(h):>10.6f}"
                    )

            # Step-by-step table
            labels = list(results.keys())
            hists = [results[l]['hist'] for l in labels]
            n_steps = min(len(h) for h in hists) if hists else 0

            header_parts = [f"{'Step':>5}"]
            for l in labels:
                header_parts.append(f"{l:>22}")
            header = "  ".join(header_parts)
            print(f"\n{header}")
            print("-" * len(header))

            steps_to_show = list(range(0, n_steps, 20))
            if n_steps > 0 and (n_steps - 1) not in steps_to_show:
                steps_to_show.append(n_steps - 1)
            for i in steps_to_show:
                row = [f"{i + 1:>5}"]
                for h in hists:
                    row.append(f"{h[i]:>22.6f}")
                print("  ".join(row))

            # Save results
            output_dir = (
                f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/nnbf_depth_sweep/"
            )
            os.makedirs(output_dir, exist_ok=True)
            save_data = {
                'system': (
                    f'{Lx}x{Ly} Fermi-Hubbard, t={t}, U={U}, '
                    f'N_f={N_f}'
                ),
            }
            for label, r in results.items():
                save_data[label] = {
                    'Np': r['Np'],
                    **r['stats'],
                }
            stats_file = os.path.join(
                output_dir, 'depth_sweep.json',
            )
            with open(stats_file, 'w') as f:
                json.dump(save_data, f, indent=4)
            print(f"\n  Results saved to: {stats_file}")

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
