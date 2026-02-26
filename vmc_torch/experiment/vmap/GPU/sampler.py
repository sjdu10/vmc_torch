from typing import Any, Dict, Optional, Tuple

import torch

from vmc_torch.experiment.vmap.GPU.vmc_modules import run_sampling_phase_gpu
from vmc_torch.experiment.vmap.GPU.vmc_utils import sample_next


class SamplerGPU:
    """Base sampler interface for GPU VMC drivers."""

    sampling_count_key = "Ns"

    def warmup_step(
        self,
        fxs: torch.Tensor,
        model,
        graph,
        use_export_compile: bool = False,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return sample_next(
            fxs,
            model,
            graph,
            verbose=verbose,
            compile=use_export_compile,
        )

    def run_sampling_phase(
        self,
        *,
        fxs: torch.Tensor,
        model,
        hamiltonian,
        graph,
        ns_per_rank: int,
        grad_batch_size: int,
        burn_in: bool,
        burn_in_steps: int,
        use_export_compile: bool,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
    ):
        raise NotImplementedError


class MetropolisExchangeSpinfulSamplerGPU(SamplerGPU):
    """
    Default GPU sampler backed by `run_sampling_phase_gpu` and `sample_next`.
    """

    def __init__(
        self,
        hopping_rate: float = 0.25,
        sampling_phase_fn=run_sampling_phase_gpu,
        sample_next_fn=sample_next,
        sampling_count_key: str = "Ns",
    ):
        self.hopping_rate = hopping_rate
        self.sampling_phase_fn = sampling_phase_fn
        self.sample_next_fn = sample_next_fn
        self.sampling_count_key = sampling_count_key

    def warmup_step(
        self,
        fxs: torch.Tensor,
        model,
        graph,
        use_export_compile: bool = False,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample_next_fn(
            fxs,
            model,
            graph,
            hopping_rate=self.hopping_rate,
            verbose=verbose,
            compile=use_export_compile,
        )

    def run_sampling_phase(
        self,
        *,
        fxs: torch.Tensor,
        model,
        hamiltonian,
        graph,
        ns_per_rank: int,
        grad_batch_size: int,
        burn_in: bool,
        burn_in_steps: int,
        use_export_compile: bool,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
    ):
        kwargs = {} if sampling_kwargs is None else dict(sampling_kwargs)
        phase_kwargs = dict(
            fxs=fxs,
            model=model,
            hamiltonian=hamiltonian,
            graph=graph,
            grad_batch_size=grad_batch_size,
            burn_in=burn_in,
            burn_in_steps=burn_in_steps,
            hopping_rate=self.hopping_rate,
            verbose=False,
            compile=use_export_compile,
        )
        phase_kwargs[self.sampling_count_key] = ns_per_rank
        phase_kwargs.update(kwargs)
        return self.sampling_phase_fn(**phase_kwargs)


__all__ = [
    "SamplerGPU",
    "MetropolisExchangeSpinfulSamplerGPU",
]
