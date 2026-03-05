"""Verify skip-boundary gives same result as with-boundary
for raw-env rows.

For rows 1 and 6, the bMPS_max/bMPS_min has D bonds (< chi).
The boundary contraction SVD with max_bond=chi does NOT
truncate (actual bond dim <= chi). So skipping it and using
cotengra's direct contract() should give the exact same result.

Run:
    torchrun --nproc_per_node=1 \
        GPU/scripts/verify_skip_boundary.py
"""
import torch
import torch.distributed as dist
import quimb as qu
import quimb.tensor as qtn

from vmc_torch.experiment.vmap.GPU.VMC import (
    setup_distributed,
)
from vmc_torch.experiment.vmap.GPU.models import (
    PEPS_Model_reuse_GPU,
)
from vmc_torch.experiment.vmap.GPU.models.pureTNS_spin import (
    unpack_tn,
)
from vmc_torch.experiment.vmap.GPU.vmc_setup import (
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
        print(f"D={D}, chi={chi}, D^2={D*D}")

        # Single-sample test: for each row, compute amplitude
        # with and without boundary contraction
        x = example_x
        params_tree = qu.utils.tree_unflatten(
            list(model.params), model.params_pytree,
        )
        tns = qtn.unpack(params_tree, model.skeleton)
        amp_full = tns.isel({
            tns.site_ind(site): x[i]
            for i, site in enumerate(tns.sites)
        })

        env_x = amp_full.compute_x_environments(
            max_bond=chi, cutoff=0.0,
            mode='mps', canonize=True,
        )

        print("\nSingle-sample: boundary vs no-boundary "
              "for each row")
        for row in range(Lx):
            bMPS_min_tn = env_x[('xmin', row)]
            bMPS_max_tn = env_x[('xmax', row)]

            if (len(bMPS_min_tn.tensors) == 0
                    or len(bMPS_max_tn.tensors) == 0):
                print(f"  Row {row}: boundary env empty, skip")
                continue

            row_tn = amp_full.select(
                [tns.row_tag(row)], which='any',
            )

            # Path A: with boundary contraction
            tn_a = (bMPS_min_tn | row_tn | bMPS_max_tn).copy()
            tn_a.view_as_(
                qtn.PEPS,
                site_tag_id=tns._site_tag_id,
                x_tag_id=tns._x_tag_id,
                y_tag_id=tns._y_tag_id,
                Lx=Lx, Ly=Ly,
                site_ind_id=tns._site_ind_id,
            )
            if len(tn_a.tensors) > 2 * Ly:
                tn_a.contract_boundary_from_xmin_(
                    max_bond=chi, cutoff=0.0,
                    xrange=[row, min(row + 1, Lx - 1)],
                    mode='mps', canonize=True,
                )
            val_a = tn_a.contract()

            # Path B: without boundary (direct contract)
            tn_b = (bMPS_min_tn | row_tn | bMPS_max_tn).copy()
            val_b = tn_b.contract()

            rel_diff = abs(val_a - val_b) / max(
                abs(val_a), 1e-30
            )
            is_raw = (
                ('xmin', row) in model._raw_bMPS_x_keys
                or ('xmax', row) in model._raw_bMPS_x_keys
            )
            status = "PASS" if rel_diff < 1e-10 else "FAIL"
            print(
                f"  Row {row}: with_boundary={val_a:.8e} "
                f"no_boundary={val_b:.8e} "
                f"rel_diff={rel_diff:.2e} {status}"
                f" {'(RAW)' if is_raw else ''}"
            )

        # Same for y
        env_y = amp_full.compute_y_environments(
            max_bond=chi, cutoff=0.0,
            mode='mps', canonize=True,
        )

        print("\nY-direction:")
        for col in range(Ly):
            bMPS_min_tn = env_y[('ymin', col)]
            bMPS_max_tn = env_y[('ymax', col)]

            if (len(bMPS_min_tn.tensors) == 0
                    or len(bMPS_max_tn.tensors) == 0):
                print(f"  Col {col}: boundary env empty, skip")
                continue

            col_tn = amp_full.select(
                [tns.col_tag(col)], which='any',
            )

            tn_a = (bMPS_min_tn | col_tn | bMPS_max_tn).copy()
            tn_a.view_as_(
                qtn.PEPS,
                site_tag_id=tns._site_tag_id,
                x_tag_id=tns._x_tag_id,
                y_tag_id=tns._y_tag_id,
                Lx=Lx, Ly=Ly,
                site_ind_id=tns._site_ind_id,
            )
            if len(tn_a.tensors) > 2 * Lx:
                tn_a.contract_boundary_from_ymin_(
                    max_bond=chi, cutoff=0.0,
                    yrange=[col, min(col + 1, Ly - 1)],
                    mode='mps', canonize=True,
                )
            val_a = tn_a.contract()

            tn_b = (bMPS_min_tn | col_tn | bMPS_max_tn).copy()
            val_b = tn_b.contract()

            rel_diff = abs(val_a - val_b) / max(
                abs(val_a), 1e-30
            )
            is_raw = (
                ('ymin', col) in model._raw_bMPS_y_keys
                or ('ymax', col) in model._raw_bMPS_y_keys
            )
            status = "PASS" if rel_diff < 1e-10 else "FAIL"
            print(
                f"  Col {col}: with_boundary={val_a:.8e} "
                f"no_boundary={val_b:.8e} "
                f"rel_diff={rel_diff:.2e} {status}"
                f" {'(RAW)' if is_raw else ''}"
            )

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
