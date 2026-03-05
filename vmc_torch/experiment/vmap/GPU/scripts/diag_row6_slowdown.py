"""Diagnose row 6 slowdown in bMPS reuse forward.

Compares forward_reuse timing for each row and prints
bMPS skeleton tensor shapes to find the root cause.

Run:
    torchrun --nproc_per_node=1 \
        GPU/scripts/diag_row6_slowdown.py
"""
import time

import torch
import torch.distributed as dist

from vmc_torch.experiment.vmap.GPU.VMC import (
    setup_distributed,
)
from vmc_torch.experiment.vmap.GPU.models import (
    PEPS_Model_reuse_GPU,
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
        B = 16

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

        # Print bMPS skeleton tensor shapes for each key
        print("=" * 60)
        print("bMPS x-direction skeleton tensor shapes:")
        print("=" * 60)
        for key in sorted(model.bMPS_x_skeletons.keys()):
            skeleton, pytree_ref = model.bMPS_x_skeletons[key]
            if hasattr(skeleton, 'tensors'):
                shapes = [t.shape for t in skeleton.tensors]
                print(f"  {key}: {len(shapes)} tensors")
                for i, s in enumerate(shapes):
                    print(f"    tensor {i}: {s}")
            else:
                print(f"  {key}: skeleton type = "
                      f"{type(skeleton)}")

        print("\n" + "=" * 60)
        print("bMPS y-direction skeleton tensor shapes:")
        print("=" * 60)
        for key in sorted(model.bMPS_y_skeletons.keys()):
            skeleton, pytree_ref = model.bMPS_y_skeletons[key]
            if hasattr(skeleton, 'tensors'):
                shapes = [t.shape for t in skeleton.tensors]
                print(f"  {key}: {len(shapes)} tensors")
                for i, s in enumerate(shapes):
                    print(f"    tensor {i}: {s}")
            else:
                print(f"  {key}: skeleton type = "
                      f"{type(skeleton)}")

        # Prepare batch of configs
        fxs = torch.stack([
            random_spin_config_sz0(N_sites, seed=s)
            for s in range(B)
        ]).to(device)

        # Cache bMPS params
        with torch.inference_mode():
            bMPS_x, amps = (
                model.cache_bMPS_params_any_direction_vmap(
                    fxs, direction='x',
                )
            )

        # Warm up all rows once (first call may be slow
        # due to vmap wrapper creation)
        print("\n" + "=" * 60)
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
                print(f"  Row {row}: {dt:.4f}s")

        # Time each row (3 repeats, take min)
        print("\n" + "=" * 60)
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
                print(
                    f"  Row {row}: min={min(times):.4f}s "
                    f"avg={sum(times)/len(times):.4f}s "
                    f"all={[f'{t:.4f}' for t in times]}"
                )

        # Diagnose: time amplitude_reuse for a SINGLE
        # sample (no vmap) for each row
        print("\n" + "=" * 60)
        print("Single-sample amplitude_reuse (no vmap):")
        print("=" * 60)
        import quimb as qu
        import quimb.tensor as qtn
        from vmc_torch.experiment.vmap.GPU.models.pureTNS_spin import (
            pack_tn, unpack_tn,
        )

        x_single = fxs[0]
        params_tree = qu.utils.tree_unflatten(
            list(model.params), model.params_pytree,
        )
        tns = qtn.unpack(params_tree, model.skeleton)
        amp_full = tns.isel({
            tns.site_ind(site): x_single[i]
            for i, site in enumerate(tns.sites)
        })

        # Compute single-sample bMPS environments
        env_x = amp_full.compute_x_environments(
            max_bond=chi, cutoff=0.0,
            mode='mps', canonize=True,
        )

        for row in range(Lx):
            bMPS_min_key = ('xmin', row)
            bMPS_max_key = ('xmax', row)

            bMPS_min_tn = env_x[bMPS_min_key]
            bMPS_max_tn = env_x[bMPS_max_key]

            # Print shapes of bMPS envs for this row
            min_shapes = [t.shape for t in bMPS_min_tn.tensors] if len(bMPS_min_tn.tensors) > 0 else []
            max_shapes = [t.shape for t in bMPS_max_tn.tensors] if len(bMPS_max_tn.tensors) > 0 else []
            print(f"\n  Row {row}:")
            print(f"    bMPS_min {bMPS_min_key}: "
                  f"{len(min_shapes)} tensors")
            for i, s in enumerate(min_shapes):
                print(f"      tensor {i}: {s}")
            print(f"    bMPS_max {bMPS_max_key}: "
                  f"{len(max_shapes)} tensors")
            for i, s in enumerate(max_shapes):
                print(f"      tensor {i}: {s}")

            # Time the single-sample reuse contraction
            selected_rows = [row]
            row_tensors = amp_full.select(
                [tns.row_tag(row)], which='any',
            )
            amp_reuse = (bMPS_min_tn | row_tensors | bMPS_max_tn)
            amp_reuse.view_as_(
                qtn.PEPS,
                site_tag_id=tns._site_tag_id,
                x_tag_id=tns._x_tag_id,
                y_tag_id=tns._y_tag_id,
                Lx=tns._Lx, Ly=tns._Ly,
                site_ind_id=tns._site_ind_id,
            )

            n_tensors = len(amp_reuse.tensors)
            do_boundary = (
                chi > 0 and n_tensors > 2 * Ly
            )
            xrange = [row, min(row + 1, Lx - 1)]

            print(f"    amp_reuse: {n_tensors} tensors, "
                  f"do_boundary={do_boundary}, "
                  f"xrange={xrange}")

            # Time boundary contraction step
            import copy
            amp_copy = amp_reuse.copy()
            torch.cuda.synchronize()
            t0 = time.time()
            if do_boundary:
                amp_copy.contract_boundary_from_xmin_(
                    max_bond=chi, cutoff=0.0,
                    xrange=xrange,
                    mode='mps', canonize=True,
                )
            torch.cuda.synchronize()
            t_boundary = time.time() - t0

            # Print shapes after boundary contraction
            after_shapes = [
                t.shape for t in amp_copy.tensors
            ]
            print(f"    After boundary: "
                  f"{len(after_shapes)} tensors")
            for i, s in enumerate(after_shapes):
                print(f"      tensor {i}: {s}")

            # Time final contract()
            amp_copy2 = amp_copy.copy()
            torch.cuda.synchronize()
            t0 = time.time()
            result = amp_copy2.contract()
            torch.cuda.synchronize()
            t_contract = time.time() - t0

            print(f"    T_boundary={t_boundary:.4f}s, "
                  f"T_contract={t_contract:.4f}s, "
                  f"total={t_boundary + t_contract:.4f}s")

        # Also time y-direction
        print("\n" + "=" * 60)
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
                print(
                    f"  Col {col}: min={min(times):.4f}s "
                    f"avg={sum(times)/len(times):.4f}s"
                )

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
