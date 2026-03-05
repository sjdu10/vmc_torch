"""Diagnose row/col slowdown in fPEPS bMPS reuse forward.

Same diagnostic as diag_row6_slowdown.py but for the
fermionic fPEPS_Model_reuse_GPU.

Run:
    torchrun --nproc_per_node=1 \
        GPU/scripts/diag_fpeps_reuse_slowdown.py
"""
import time

import torch
import torch.distributed as dist

from vmc_torch.experiment.vmap.GPU.VMC import (
    setup_distributed,
)
from vmc_torch.experiment.vmap.GPU.models import (
    fPEPS_Model_reuse_GPU,
)
from vmc_torch.experiment.vmap.GPU.vmc_setup import (
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.experiment.vmap.GPU.vmc_utils import (
    random_initial_config,
)

dtype = torch.float64
DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch'
    '/vmc_torch/experiment/vmap/GPU/data'
)


def main():
    setup_linalg_hooks(jitter=1e-12)
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42)

        Lx, Ly = 8, 8
        N_sites = Lx * Ly
        t, U = 1.0, 8.0
        N_f = N_sites - 2
        D = 4
        chi = 16
        B = 16

        peps = load_or_generate_peps(
            Lx, Ly, t, U, N_f, D,
            seed=42, dtype=dtype,
            file_path=(
                f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/"
                f"t={t}_U={U}/N={N_f}/Z2/D={D}/"
            ),
            scale_factor=4,
        )
        model = fPEPS_Model_reuse_GPU(
            tn=peps,
            max_bond=chi,
            dtype=dtype,
            contract_boundary_opts={
                'mode': 'mps',
                'canonize': True,
            },
        )
        model.to(device)
        n_fermions_per_spin = (N_f // 2, N_f // 2)

        example_x = random_initial_config(
            N_f, N_sites, seed=0,
        ).to(device)
        model.cache_bMPS_skeleton(example_x)

        print(f"System: {Lx}x{Ly} Fermi-Hubbard, "
              f"D={D}, chi={chi}")
        print(f"Raw x-env keys: {model._raw_bMPS_x_keys}")
        print(f"Raw y-env keys: {model._raw_bMPS_y_keys}")

        # Prepare batch
        fxs = torch.stack([
            random_initial_config(N_f, N_sites, seed=s)
            for s in range(B)
        ]).to(device)

        # Cache bMPS params
        with torch.inference_mode():
            bMPS_x, amps = (
                model.cache_bMPS_params_any_direction_vmap(
                    fxs, direction='x',
                )
            )

        # Warmup all rows
        print(f"\n{'=' * 60}")
        print(f"Warmup pass (B={B}):")
        print("=" * 60)
        with torch.inference_mode():
            for row in range(Lx):
                selected_rows = [row]
                torch.cuda.synchronize()
                t0 = time.time()
                _ = model.forward_reuse(
                    fxs,
                    bMPS_params_x_batched=bMPS_x,
                    selected_rows=selected_rows,
                )
                torch.cuda.synchronize()
                dt = time.time() - t0
                raw = (
                    ('xmin', row) in model._raw_bMPS_x_keys
                    or ('xmax', row) in model._raw_bMPS_x_keys
                )
                print(f"  Row {row}: {dt:.4f}s"
                      f" {'(RAW)' if raw else ''}")

        # Time each row (3 repeats)
        print(f"\n{'=' * 60}")
        print(f"Timing forward_reuse per row (B={B}, "
              f"3 repeats, min):")
        print("=" * 60)
        n_repeats = 3
        with torch.inference_mode():
            for row in range(Lx):
                selected_rows = [row]
                times = []
                for _ in range(n_repeats):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    _ = model.forward_reuse(
                        fxs,
                        bMPS_params_x_batched=bMPS_x,
                        selected_rows=selected_rows,
                    )
                    torch.cuda.synchronize()
                    dt = time.time() - t0
                    times.append(dt)
                raw = (
                    ('xmin', row) in model._raw_bMPS_x_keys
                    or ('xmax', row) in model._raw_bMPS_x_keys
                )
                print(
                    f"  Row {row}: min={min(times):.4f}s "
                    f"avg={sum(times)/len(times):.4f}s "
                    f"all={[f'{t:.4f}' for t in times]}"
                    f" {'(RAW)' if raw else ''}"
                )

        # Y-direction
        print(f"\n{'=' * 60}")
        print(f"Timing forward_reuse per col (B={B}, "
              f"3 repeats, min):")
        print("=" * 60)
        with torch.inference_mode():
            bMPS_y, amps_y = (
                model.cache_bMPS_params_any_direction_vmap(
                    fxs, direction='y',
                )
            )
            # warmup
            for col in range(Ly):
                _ = model.forward_reuse(
                    fxs,
                    bMPS_params_y_batched=bMPS_y,
                    selected_cols=[col],
                )
            # time
            for col in range(Ly):
                times = []
                for _ in range(n_repeats):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    _ = model.forward_reuse(
                        fxs,
                        bMPS_params_y_batched=bMPS_y,
                        selected_cols=[col],
                    )
                    torch.cuda.synchronize()
                    dt = time.time() - t0
                    times.append(dt)
                raw = (
                    ('ymin', col) in model._raw_bMPS_y_keys
                    or ('ymax', col) in model._raw_bMPS_y_keys
                )
                print(
                    f"  Col {col}: min={min(times):.4f}s "
                    f"avg={sum(times)/len(times):.4f}s"
                    f" {'(RAW)' if raw else ''}"
                )

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
