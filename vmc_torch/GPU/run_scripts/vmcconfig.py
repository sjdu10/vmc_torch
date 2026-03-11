"""Shared VMC configuration and helpers for GPU run scripts.

Base VMCConfig contains all common hyperparameters with sensible
defaults.  Run scripts import it and override fields as needed:

    # Simple override:
    cfg = VMCConfig(batch_size=256, use_log_amp=True)

    # Extend with script-specific fields:
    @dataclass
    class ReuseCfg(VMCConfig):
        use_export_compile_reuse: bool = False
        use_cheap_grad: bool = True

Helper functions (make_preconditioner, load_checkpoint, etc.)
reduce boilerplate in run scripts while keeping full flexibility
— every helper is optional and can be replaced with inline code.
"""
import json
import os
from dataclasses import dataclass

import torch

from vmc_torch.GPU.optimizer import (
    DistributedMinSRGPU,
    DistributedSRMinresGPU,
    MinSRGPU,
)


@dataclass
class VMCConfig:
    """VMC numerical / training settings.

    Groups:
        Sampling:   batch_size, ns_per_rank, grad_batch_size,
                    burn_in_steps
        Training:   vmc_steps, learning_rate, run_sr,
                    lr_scheduler
        SR solver:  sr_diag_shift, use_min_sr, use_distributed_min_sr,
                    use_distributed_sr_minres, minres_sr_use_scipy,
                    sr_rtol, sr_maxiter, param_chunk_size
        Compile:    use_export_compile, use_log_amp
        Gradient:   offload_grad_to_cpu
        Checkpoint: save_every, resume_step
        Debug:      debug, verbose, outlier_clip_factor
    """

    # ----- Sampling -----
    batch_size: int = 1024
    ns_per_rank: int = 1024
    grad_batch_size: int = 1024
    burn_in_steps: int = 0

    # ----- Training -----
    vmc_steps: int = 100
    learning_rate: float = 0.1
    run_sr: bool = True
    lr_scheduler: object = None

    # ----- SR solver -----
    sr_diag_shift: float = 1e-4
    use_min_sr: bool = False
    use_distributed_min_sr: bool = False
    use_distributed_sr_minres: bool = True
    minres_sr_use_scipy: bool = False
    sr_rtol: float = 1e-4
    sr_maxiter: int = 100
    param_chunk_size: int = 1024

    # ----- Compile -----
    use_export_compile: bool = False
    
    # ----- Log-amp -----
    use_log_amp: bool = True

    # ----- Gradient -----
    offload_grad_to_cpu: bool = False

    # ----- Checkpoint -----
    save_every: int = 50
    resume_step: int = 0

    # ----- Debug -----
    debug: bool = False
    verbose: bool = False
    outlier_clip_factor: float = 1e3


# ============================================================
# Helper functions for run scripts
# ============================================================


def make_preconditioner(cfg):
    """Create SR preconditioner from VMCConfig flags.

    Returns None if no SR preconditioner is selected.
    Scripts can skip this and instantiate their own.
    """
    if cfg.use_distributed_min_sr:
        return DistributedMinSRGPU(
            param_chunk_size=cfg.param_chunk_size,
        )
    elif cfg.use_min_sr:
        return MinSRGPU()
    elif cfg.use_distributed_sr_minres:
        return DistributedSRMinresGPU(
            rtol=cfg.sr_rtol,
            maxiter=cfg.sr_maxiter,
            use_scipy=cfg.minres_sr_use_scipy,
        )
    return None


def load_checkpoint(
    model, output_dir, model_name,
    resume_step, device, rank,
):
    """Load model checkpoint if resume_step > 0.

    Handles both old format (bare state_dict) and new
    format (dict with 'model_state_dict' + 'fxs' keys).
    """
    if resume_step <= 0:
        return
    ckpt_path = os.path.join(
        output_dir,
        f'checkpoint_{model_name}_{resume_step}.pt',
    )
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=True,
    )
    if (
        isinstance(ckpt, dict)
        and 'model_state_dict' in ckpt
    ):
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    if rank == 0:
        print(f"Loaded checkpoint: {ckpt_path}")


def make_stats(system_str, n_params, ns_per_rank, world_size):
    """Create stats tracking dict for energy history."""
    return {
        'system': system_str,
        'Np': n_params,
        'sample size': ns_per_rank * world_size,
        'mean': [],
        'error': [],
        'variance': [],
    }


def make_stats_file(
    output_dir, model_name, resume_step, suffix='',
):
    """Build stats JSON file path.

    Args:
        suffix: optional suffix before step tag,
            e.g. '_reuse'.
    """
    step_tag = (
        f'_from{resume_step}'
        if resume_step > 0 else ''
    )
    return os.path.join(
        output_dir,
        f'stats_{model_name}{suffix}{step_tag}.json',
    )


def make_on_step_end(
    rank, stats, stats_file, output_dir,
    model_name, model, save_every,
):
    """Create on_step_end callback for the VMC loop.

    The returned callback:
    - Appends energy stats and writes JSON (rank 0 only)
    - Saves checkpoint with model state_dict + fxs
      every ``save_every`` steps (rank 0 only)
    """
    def on_step_end(info):
        if rank != 0:
            return
        stats['mean'].append(info['energy_per_site'])
        stats['error'].append(info['error_per_site'])
        stats['variance'].append(info['energy_var'])
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)

        step = info['step']
        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(
                output_dir,
                f'checkpoint_{model_name}_{step + 1}.pt',
            )
            torch.save({
                'model_state_dict': model.state_dict(),
                'fxs': info['fxs'].cpu(),
            }, ckpt_path)

    return on_step_end


def print_summary(
    rank, energy_history, system_str, stats_file=None,
):
    """Print VMC run summary (rank 0 only)."""
    if rank != 0 or not energy_history:
        return
    print(f"\n{'=' * 50}")
    print(f"Result: {system_str}")
    print(f"{'=' * 50}")
    print(f"First E/site: {energy_history[0]:.6f}")
    print(f"Last  E/site: {energy_history[-1]:.6f}")
    print(f"Min   E/site: {min(energy_history):.6f}")
    if stats_file:
        print(f"Stats saved to: {stats_file}")
    if energy_history[-1] < energy_history[0]:
        print("\nEnergy decreased.")
    else:
        print("\nWARNING: Energy did NOT decrease.")
