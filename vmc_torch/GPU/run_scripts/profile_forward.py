"""Profile the model forward pass for vmc_run_spin.py setup.

Produces a Chrome-trace JSON file viewable in:
  - chrome://tracing  (paste the file)
  - https://ui.perfetto.dev  (drag-and-drop the file)

Run:
    torchrun --nproc_per_node=1 profile_forward.py

Output:
    forward_profile_trace.json  (in this directory)
"""
import torch
import torch.distributed as dist
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
)

from vmc_torch.GPU.VMC import setup_distributed
from vmc_torch.GPU.models import PEPS_Model_GPU
from vmc_torch.GPU.vmc_setup import (
    generate_random_spin_peps,
    initialize_walkers,
    random_spin_config_sz0,
    setup_linalg_hooks,
)

dtype = torch.float64


def main():
    setup_linalg_hooks(
        jitter=1e-8, qr_via_eigh=True,
        cholesky_qr=False,
        cholesky_qr_adaptive_jitter=False,
    )
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        # ========== Match vmc_run_spin.py params ==========
        Lx, Ly = 8, 8
        N_sites = Lx * Ly
        D = 6
        chi = 10
        B = 256  # reduced from 2048 for profiler memory

        # ========== Model ==========
        peps = generate_random_spin_peps(
            Lx, Ly, D, seed=42, dtype=dtype,
        )
        model = PEPS_Model_GPU(
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

        N_params = sum(p.numel() for p in model.parameters())
        print(
            f"System: {Lx}x{Ly}, D={D}, chi={chi}, "
            f"{N_params} params, B={B}, device={device}"
        )

        # ========== Walkers ==========
        fxs = initialize_walkers(
            init_fn=lambda seed: random_spin_config_sz0(
                N_sites, seed=seed,
            ),
            batch_size=B, seed=42,
            rank=rank, device=device,
        )

        # ========== Warmup (outside profiler) ==========
        print("Warming up (5 forward passes)...")
        for _ in range(5):
            with torch.no_grad():
                _ = model(fxs)
        torch.cuda.synchronize()
        print("Warmup done.")

        # ========== Profile ==========
        n_profile_iters = 10
        print(
            f"Profiling {n_profile_iters} forward passes..."
        )

        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for i in range(n_profile_iters):
                with record_function(f"forward_{i}"):
                    with torch.no_grad():
                        amps = model(fxs)
                    torch.cuda.synchronize()

        # ========== Export trace JSON ==========
        trace_path = "forward_profile_trace.json"
        prof.export_chrome_trace(trace_path)
        print(f"\nTrace saved to: {trace_path}")
        print(
            "Open in browser: chrome://tracing "
            "or https://ui.perfetto.dev"
        )

        # ========== Print table summary ==========
        print("\n" + "=" * 70)
        print("CPU time (top 30 ops):")
        print("=" * 70)
        print(
            prof.key_averages().table(
                sort_by="cpu_time_total",
                row_limit=30,
            )
        )
        print("\n" + "=" * 70)
        print("CUDA time (top 30 ops):")
        print("=" * 70)
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=30,
            )
        )

        # ========== Also print self times ==========
        print("\n" + "=" * 70)
        print("CUDA self time (top 30 kernels):")
        print("=" * 70)
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=30,
            )
        )

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
