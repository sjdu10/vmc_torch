"""Shared VMC configuration for GPU run scripts.

Base VMCConfig contains all common hyperparameters with sensible
defaults.  Run scripts import it and override fields as needed:

    # Simple override:
    cfg = VMCConfig(batch_size=256, use_log_amp=True)

    # Extend with script-specific fields:
    @dataclass
    class ReuseCfg(VMCConfig):
        use_export_compile_reuse: bool = False
        use_cheap_grad: bool = True
"""
from dataclasses import dataclass


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
