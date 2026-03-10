"""Test whether export_and_compile_reuse speeds up forward_reuse.

System: 6x6 Heisenberg, D=4, chi=16 (matching vmc_run_spin_reuse.py)

Compares:
  1. Eager forward_reuse (no compile)
  2. Compiled forward_reuse (after export_and_compile_reuse)

Usage:
    torchrun --nproc_per_node=1 test_reuse_compile.py
"""
import time

import torch
import torch.distributed as dist

from vmc_torch.GPU.VMC import setup_distributed
from vmc_torch.GPU.models import PEPS_Model_reuse_GPU
from vmc_torch.GPU.vmc_setup import (
    generate_random_spin_peps,
    random_spin_config_sz0,
    setup_linalg_hooks,
)

dtype = torch.float64


def time_forward_reuse(model, batch_x, bMPS_x, n_trials=5):
    """Time forward_reuse for each x-direction reuse pattern."""
    Lx = model.Lx
    results = {}

    for width in (1, 2):
        for start in range(Lx - width + 1):
            rows = list(range(start, start + width))
            key = ('x', tuple(rows))

            # Warmup
            out = model.forward_reuse(
                batch_x,
                bMPS_params_x_batched=bMPS_x,
                selected_rows=rows,
            )
            torch.cuda.synchronize()

            # Time
            times = []
            for _ in range(n_trials):
                t0 = time.time()
                out = model.forward_reuse(
                    batch_x,
                    bMPS_params_x_batched=bMPS_x,
                    selected_rows=rows,
                )
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            avg = sum(times) / len(times)
            results[key] = avg

    return results


def main():
    setup_linalg_hooks(jitter=1e-16)
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42)

        # Same settings as vmc_run_spin_reuse.py
        Lx, Ly = 6, 6
        N_sites = Lx * Ly
        D = 4
        chi = 16
        B = 64  # batch size for testing

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
            },
        )
        model.to(device)
        N_params = sum(p.numel() for p in model.parameters())
        print(
            f"Model: {Lx}x{Ly} Heisenberg, D={D}, "
            f"chi={chi}, {N_params} params"
        )

        # Init bMPS skeleton
        example_x = random_spin_config_sz0(
            N_sites, seed=0,
        ).to(device)
        model.cache_bMPS_skeleton(example_x)

        # Build batch
        batch_x = torch.stack([
            random_spin_config_sz0(N_sites, seed=s)
            for s in range(B)
        ]).to(device)

        # Cache bMPS environments
        with torch.no_grad():
            bMPS_x, bMPS_y = model.cache_bMPS_params_vmap(
                batch_x,
            )

        # ========== Test 1: Eager (no compile) ==========
        print(f"\n=== Eager forward_reuse (B={B}) ===")
        eager_results = time_forward_reuse(
            model, batch_x, bMPS_x,
        )
        for key, t in eager_results.items():
            print(f"  {key}: {t*1000:.1f} ms")
        eager_avg = sum(eager_results.values()) / len(
            eager_results
        )
        print(f"  Average: {eager_avg*1000:.1f} ms")

        # ========== Test 2: Compiled ==========
        print(f"\n=== Export + compile reuse ===")
        t0 = time.time()
        model.export_and_compile_reuse(
            example_x,
            mode='default',
            verbose=True,
            use_log_amp=False,
        )
        print(
            f"Compile time: {time.time() - t0:.1f}s\n"
        )

        # Check which patterns got compiled
        compiled_keys = set(model._compiled_reuse.keys())
        print(
            f"Compiled {len(compiled_keys)} patterns: "
            f"{sorted(compiled_keys)[:5]}..."
        )
        print()

        print(f"=== Compiled forward_reuse (B={B}) ===")
        compiled_results = time_forward_reuse(
            model, batch_x, bMPS_x,
        )
        for key, t in compiled_results.items():
            speedup = eager_results[key] / t
            print(
                f"  {key}: {t*1000:.1f} ms "
                f"(was {eager_results[key]*1000:.1f} ms, "
                f"{speedup:.2f}x)"
            )
        compiled_avg = sum(compiled_results.values()) / len(
            compiled_results
        )
        print(f"  Average: {compiled_avg*1000:.1f} ms")
        print(
            f"  Overall speedup: "
            f"{eager_avg / compiled_avg:.2f}x"
        )

        # ========== Check dispatch path ==========
        print(f"\n=== Dispatch check ===")
        print(
            f"_compiled_reuse_log_amp = "
            f"{getattr(model, '_compiled_reuse_log_amp', 'N/A')}"
        )
        print(
            f"Number of _compiled_reuse entries: "
            f"{len(model._compiled_reuse)}"
        )

        # Test: does forward_reuse actually hit compiled?
        # Manually check with a known key
        test_rows = [2, 3]
        test_key = ('x', tuple(test_rows))
        has_compiled = (
            test_key in model._compiled_reuse
        )
        log_amp_flag = getattr(
            model, '_compiled_reuse_log_amp', False
        )
        will_use_compiled = (
            has_compiled and not log_amp_flag
        )
        print(
            f"Key {test_key}: "
            f"in _compiled_reuse={has_compiled}, "
            f"_compiled_reuse_log_amp={log_amp_flag}, "
            f"will_use_compiled={will_use_compiled}"
        )

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
