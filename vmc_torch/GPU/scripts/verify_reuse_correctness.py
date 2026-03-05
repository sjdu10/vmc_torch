"""Verify reuse amplitudes match full contraction.

Run:
    torchrun --nproc_per_node=1 \
        GPU/scripts/verify_reuse_correctness.py
"""
import torch
import torch.distributed as dist

from vmc_torch.GPU.VMC import (
    setup_distributed,
)
from vmc_torch.GPU.models import (
    PEPS_Model_reuse_GPU,
)
from vmc_torch.GPU.vmc_setup import (
    generate_random_spin_peps,
    random_spin_config_sz0,
    setup_linalg_hooks,
)

dtype = torch.float64


def main():
    setup_linalg_hooks(jitter=1e-16)
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42)

        Lx, Ly = 8, 8
        N_sites = Lx * Ly
        D = 4
        chi = 16
        B = 8

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

        example_x = random_spin_config_sz0(
            N_sites, seed=0,
        ).to(device)
        model.cache_bMPS_skeleton(example_x)

        print(f"Raw x-env keys: {model._raw_bMPS_x_keys}")
        print(f"Raw y-env keys: {model._raw_bMPS_y_keys}")

        fxs = torch.stack([
            random_spin_config_sz0(N_sites, seed=s)
            for s in range(B)
        ]).to(device)

        # Full contraction reference
        with torch.inference_mode():
            ref_amps = model.forward(fxs)

            # Cache bMPS
            bMPS_x, cache_amps = (
                model.cache_bMPS_params_any_direction_vmap(
                    fxs, direction='x',
                )
            )
            bMPS_y, cache_amps_y = (
                model.cache_bMPS_params_any_direction_vmap(
                    fxs, direction='y',
                )
            )

        # Check reuse vs full for each row
        print("\nX-direction reuse vs full:")
        all_pass = True
        with torch.inference_mode():
            for row in range(Lx):
                reuse_amps = model.forward_reuse(
                    fxs,
                    bMPS_params_x_batched=bMPS_x,
                    selected_rows=[row],
                )
                rel_diff = (
                    (reuse_amps - ref_amps).abs()
                    / ref_amps.abs().clamp(min=1e-30)
                ).max().item()
                status = "PASS" if rel_diff < 1e-6 else "FAIL"
                if status == "FAIL":
                    all_pass = False
                raw = (
                    ("xmin", row) in model._raw_bMPS_x_keys
                    or ("xmax", row) in model._raw_bMPS_x_keys
                )
                print(
                    f"  Row {row}: rel_diff={rel_diff:.2e} "
                    f"{status}"
                    f" {'(raw env, skip boundary)' if raw else ''}"
                )

        # Check reuse vs full for each col
        print("\nY-direction reuse vs full:")
        with torch.inference_mode():
            for col in range(Ly):
                reuse_amps = model.forward_reuse(
                    fxs,
                    bMPS_params_y_batched=bMPS_y,
                    selected_cols=[col],
                )
                rel_diff = (
                    (reuse_amps - ref_amps).abs()
                    / ref_amps.abs().clamp(min=1e-30)
                ).max().item()
                status = "PASS" if rel_diff < 1e-6 else "FAIL"
                if status == "FAIL":
                    all_pass = False
                raw = (
                    ("ymin", col) in model._raw_bMPS_y_keys
                    or ("ymax", col) in model._raw_bMPS_y_keys
                )
                print(
                    f"  Col {col}: rel_diff={rel_diff:.2e} "
                    f"{status}"
                    f" {'(raw env, skip boundary)' if raw else ''}"
                )

        if all_pass:
            print("\nAll checks PASS.")
        else:
            print("\nSome checks FAILED!")

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
