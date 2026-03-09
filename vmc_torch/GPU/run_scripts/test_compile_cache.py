"""Test torch.compile + torch.export cache persistence.

Tests 3 caching strategies:
  A. TORCHINDUCTOR_CACHE_DIR only (baseline, ~2x speedup)
  B. A + torch.export.save/load (skip quimb re-tracing)
  C. A + B + torch.compiler.save/load_cache_artifacts
     (skip AOTAutograd + graph lowering)

Usage:
    # Clear cache, run from scratch
    rm -rf /tmp/test_compile_cache
    TORCHINDUCTOR_CACHE_DIR=/tmp/test_compile_cache \
      torchrun --nproc_per_node=1 test_compile_cache.py

    # Run again — should be faster
    TORCHINDUCTOR_CACHE_DIR=/tmp/test_compile_cache \
      torchrun --nproc_per_node=1 test_compile_cache.py
"""
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from vmc_torch.GPU.VMC import setup_distributed
from vmc_torch.GPU.hamiltonian import (
    spinful_Fermi_Hubbard_square_lattice_torch,
)
from vmc_torch.GPU.models import fPEPS_Model_GPU
from vmc_torch.GPU.vmc_setup import (
    load_or_generate_peps,
    setup_linalg_hooks,
)
from vmc_torch.GPU.vmc_utils import random_initial_config

dtype = torch.float64
DEFAULT_DATA_ROOT = (
    '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch'
    '/GPU/data'
)
CACHE_DIR = '/tmp/test_compile_cache'


def build_model_and_example(device, D=4):
    """Build the same 4x4 FH model as vmc_run_fpeps.py."""
    Lx, Ly = 4, 4
    N_sites = Lx * Ly
    t, U = 1.0, 8.0
    N_f = N_sites - 2
    n_fermions_per_spin = (N_f // 2, N_f // 2)
    chi = -1

    H = spinful_Fermi_Hubbard_square_lattice_torch(
        Lx, Ly, t, U, N_f,
        pbc=False,
        n_fermions_per_spin=n_fermions_per_spin,
        no_u1_symmetry=False,
        gpu=True,
    )
    H.precompute_hops_gpu(device)

    fpeps_base = (
        f"{DEFAULT_DATA_ROOT}/{Lx}x{Ly}/t={t}_U={U}"
        f"/N={N_f}/Z2/D={D}/"
    )
    peps = load_or_generate_peps(
        Lx, Ly, t, U, N_f, D,
        seed=42, dtype=dtype,
        file_path=fpeps_base,
        scale_factor=4,
    )
    model = fPEPS_Model_GPU(
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
        f"Model: {Lx}x{Ly} FH, D={D}, chi={chi}, "
        f"{N_params} params"
    )

    example_x = random_initial_config(
        N_f, N_sites, seed=0,
    ).to(device)

    batch_x = torch.stack([
        random_initial_config(N_f, N_sites, seed=s)
        for s in range(16)
    ]).to(device)

    return model, example_x, batch_x


def test_export_save_load(model, example_x, batch_x):
    """Test torch.export.save/load to skip re-tracing."""
    from torch.export import export

    export_path = os.path.join(CACHE_DIR, 'exported_log.pt2')
    params_list = list(model.params)
    n_params = len(params_list)
    use_log_amp = True

    # --- Step 1: Export (trace or load) ---
    t0 = time.time()
    if os.path.exists(export_path):
        print("  Loading cached exported graph...")
        exported = torch.export.load(export_path)
        exported_module = exported.module()
        t_export = time.time() - t0
        print(f"  [1] Load exported graph: {t_export:.2f}s")
    else:
        print("  Tracing from scratch...")
        export_fn = model._log_amplitude_for_export

        class _AmpModule(nn.Module):
            def __init__(self_, amp_fn):
                super().__init__()
                self_._fn = amp_fn

            def forward(self_, x, *flat_params):
                return self_._fn(x, *flat_params)

        with torch.no_grad():
            exported = export(
                _AmpModule(export_fn),
                (example_x, *params_list),
            )
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.export.save(exported, export_path)
        exported_module = exported.module()
        t_export = time.time() - t0
        print(
            f"  [1] Export + save: {t_export:.2f}s "
            f"(saved to {export_path})"
        )

    # --- Step 2: Move constants to GPU ---
    t0 = time.time()
    model._exported_module = exported_module
    model._move_exported_constants_to_device(
        example_x.device,
    )
    t_move = time.time() - t0
    print(f"  [2] Move constants to GPU: {t_move:.4f}s")

    # --- Step 3: vmap ---
    t0 = time.time()
    vmapped = torch.vmap(
        model._exported_module,
        in_dims=(0, *([None] * n_params)),
    )
    t_vmap = time.time() - t0
    print(f"  [3] torch.vmap: {t_vmap:.4f}s")

    # --- Step 4: compile ---
    t0 = time.time()
    compiled = torch.compile(vmapped, mode='default')
    t_compile_call = time.time() - t0
    print(
        f"  [4] torch.compile() call: "
        f"{t_compile_call:.4f}s"
    )

    # --- Step 5: first forward (triggers actual compilation) ---
    t0 = time.time()
    out = compiled(batch_x, *params_list)
    torch.cuda.synchronize()
    t_first = time.time() - t0
    print(f"  [5] First forward (warmup): {t_first:.2f}s")

    # --- Step 6: second forward ---
    t0 = time.time()
    out2 = compiled(batch_x, *params_list)
    torch.cuda.synchronize()
    t_second = time.time() - t0
    print(f"  [6] Second forward: {t_second:.4f}s")

    total = (
        t_export + t_move + t_vmap
        + t_compile_call + t_first
    )
    print(f"  TOTAL overhead: {total:.2f}s")
    return total


def test_compiler_cache_artifacts(
    model, example_x, batch_x,
):
    """Test torch.compiler.save/load_cache_artifacts."""
    from torch.export import export

    export_path = os.path.join(CACHE_DIR, 'exported_log.pt2')
    artifact_path = os.path.join(
        CACHE_DIR, 'compiler_cache.bin',
    )
    params_list = list(model.params)
    n_params = len(params_list)

    # --- Load compiler cache artifacts if available ---
    if os.path.exists(artifact_path):
        t0 = time.time()
        with open(artifact_path, 'rb') as f:
            artifact_bytes = f.read()
        torch.compiler.load_cache_artifacts(artifact_bytes)
        t_load = time.time() - t0
        print(
            f"  [0] Load compiler cache artifacts: "
            f"{t_load:.2f}s"
        )
    else:
        print("  [0] No compiler cache artifacts found")

    # --- Export (trace or load) ---
    t0 = time.time()
    if os.path.exists(export_path):
        exported = torch.export.load(export_path)
        exported_module = exported.module()
        t_export = time.time() - t0
        print(f"  [1] Load exported graph: {t_export:.2f}s")
    else:
        export_fn = model._log_amplitude_for_export

        class _AmpModule(nn.Module):
            def __init__(self_, amp_fn):
                super().__init__()
                self_._fn = amp_fn

            def forward(self_, x, *flat_params):
                return self_._fn(x, *flat_params)

        with torch.no_grad():
            exported = export(
                _AmpModule(export_fn),
                (example_x, *params_list),
            )
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.export.save(exported, export_path)
        exported_module = exported.module()
        t_export = time.time() - t0
        print(f"  [1] Export + save: {t_export:.2f}s")

    # --- Move + vmap + compile ---
    model._exported_module = exported_module
    model._move_exported_constants_to_device(
        example_x.device,
    )
    vmapped = torch.vmap(
        model._exported_module,
        in_dims=(0, *([None] * n_params)),
    )
    compiled = torch.compile(vmapped, mode='default')

    # --- First forward ---
    t0 = time.time()
    out = compiled(batch_x, *params_list)
    torch.cuda.synchronize()
    t_first = time.time() - t0
    print(f"  [2] First forward (warmup): {t_first:.2f}s")

    # --- Second forward ---
    t0 = time.time()
    out2 = compiled(batch_x, *params_list)
    torch.cuda.synchronize()
    t_second = time.time() - t0
    print(f"  [3] Second forward: {t_second:.4f}s")

    # --- Save compiler cache artifacts ---
    t0 = time.time()
    try:
        artifact_bytes, _ = (
            torch.compiler.save_cache_artifacts()
        )
        with open(artifact_path, 'wb') as f:
            f.write(artifact_bytes)
        t_save = time.time() - t0
        print(
            f"  [4] Save compiler cache artifacts: "
            f"{t_save:.2f}s "
            f"({len(artifact_bytes) / 1e6:.1f} MB)"
        )
    except Exception as e:
        print(f"  [4] Save artifacts failed: {e}")

    total = t_export + t_first
    print(f"  TOTAL overhead: {total:.2f}s")
    return total


def main():
    setup_linalg_hooks(
        jitter=1e-8, qr_via_eigh=False,
        cholesky_qr=True,
        cholesky_qr_adaptive_jitter=False,
        nonuniform_diag=True,
    )
    torch.set_default_dtype(dtype)

    try:
        rank, world_size, device = setup_distributed()
        torch.set_default_device(device)
        torch.manual_seed(42 + rank)

        cache_dir = os.environ.get(
            'TORCHINDUCTOR_CACHE_DIR', '(default /tmp)'
        )
        print(f"TORCHINDUCTOR_CACHE_DIR = {cache_dir}")

        D = int(os.environ.get('TEST_D', '4'))
        model, example_x, batch_x = (
            build_model_and_example(device, D=D)
        )

        export_path = os.path.join(
            CACHE_DIR, 'exported_log.pt2',
        )
        has_export_cache = os.path.exists(export_path)
        artifact_path = os.path.join(
            CACHE_DIR, 'compiler_cache.bin',
        )
        has_artifact_cache = os.path.exists(artifact_path)
        print(
            f"Export cache: "
            f"{'EXISTS' if has_export_cache else 'NONE'}"
        )
        print(
            f"Compiler artifacts: "
            f"{'EXISTS' if has_artifact_cache else 'NONE'}"
        )
        print()

        # --- Test B: export save/load ---
        print("=== Test B: torch.export.save/load ===")
        test_export_save_load(model, example_x, batch_x)
        print()

        # Reset model state for test C
        model._exported = False
        model._compiled = False
        model._exported_module = None

        # Need to reset torch.compile state
        torch._dynamo.reset()

        # --- Test C: compiler cache artifacts ---
        print(
            "=== Test C: export.save/load "
            "+ compiler.save/load_cache_artifacts ==="
        )
        test_compiler_cache_artifacts(
            model, example_x, batch_x,
        )

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
