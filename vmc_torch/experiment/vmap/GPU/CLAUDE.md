# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Directory Is

GPU port of the batched VMC (Variational Monte Carlo) pipeline. Replaces `mpi4py` with `torch.distributed` (NCCL backend) so the entire pipeline runs on GPUs. This is the **active development area** — code here is experimental and evolving rapidly.

Parent pipeline: `experiment/vmap/` (CPU, MPI-based). This GPU port mirrors its structure but with GPU-specific optimizations.

## Running

```bash
# Multi-GPU
torchrun --nproc_per_node=<N> GPU/vmc_run.py

# Single GPU (torchrun still required for dist.init_process_group)
torchrun --nproc_per_node=1 GPU/vmc_run.py

# Scripts (some work without torchrun, some need it)
python GPU/scripts/profile_vmc.py
torchrun --nproc_per_node=1 GPU/scripts/test_export_vmap.py
```

All commands should be run from the `experiment/vmap/GPU` directory. The scripts import via `vmc_torch.experiment.vmap.GPU.*`.

## Workflow

1. **Make Notes** After modifying files in GPU folder, ALWAYS make human-friendly and readable concise notes to notebook.md. DO IT WITHOUT ASKING ME.
2. **Verify correctness** Always verify your code changes.
3. **Commit and push** when asked to commit, write clear and concise commit messages, and push using `git push private main`. 

## Architecture

### File Layout

| File | Role |
|---|---|
| `models.py` | `fPEPS_Model_GPU` (export+compile fPEPS), `PureNN_GPU` (NN baseline) |
| `vmc_utils.py` | `sample_next`, `evaluate_energy`, `compute_grads_gpu`, proposal functions |
| `vmc_modules.py` | `run_sampling_phase_gpu`, `distributed_minres_solver_gpu`, `minSR_solver_gpu` |
| `vmc_run.py` | Main entry point — full VMC loop |

### Key Difference from CPU Pipeline

**CPU** (`vmap/`): Master-worker MPI pattern. Rank 0 dispatches, workers sample. Uses `mpi4py`.

**GPU** (`vmap/GPU/`): All ranks sample independently (no master-worker dispatch). Uses `torch.distributed.all_reduce` for aggregation. Each rank owns `B` walkers and accumulates `Ns_per_rank` samples across multiple sweeps.

### The torch.export Pipeline (Critical Concept)

`torch.compile` alone **cannot** trace through quimb/symmray because they use Python-level dispatch that `torch.dynamo` can't capture. The workaround:

1. **`torch.export`** — traces the TN contraction with concrete inputs, capturing all quimb/symmray ops as a pure aten-ops FX graph. This is the key step that "flattens" the Python dispatch.
2. **`torch.vmap`** — batches the exported graph over the sample dimension.
3. **`torch.compile`** — fuses the batched aten ops into CUDA kernels.

```python
model = fPEPS_Model_GPU(tn=peps, max_bond=chi, ...)
model.to(device)
model.export_and_compile(example_x)  # one-time, ~10-40s
out = model(batch_x)                 # fast batched forward
```

Speedup: ~13x for exact contraction (chi=-1), ~1.4x for boundary contraction (finite chi).

**Caveats:**
- Must re-export if TN structure changes (different Lx/Ly, different contraction path).
- Safe to call with different parameter VALUES — only the graph structure is baked in.
- Controlled by `USE_EXPORT_COMPILE` flag in `vmc_run.py` (currently `False` by default).
- `export_only()` skips the compile step (useful for debugging).

### Padding for torch.compile

`torch.compile` recompiles when input shapes change. Both `sample_next` and `evaluate_energy` pad variable-size batches to fixed size `B` to avoid recompilation:

- In `sample_next`: when fewer than `B` configs change at an edge, changed configs are packed into the first `n_changed` slots and padded to `B`.
- In `evaluate_energy`: connected configs are chunked into size-`B` batches; the last chunk is padded with copies of the first row.

### Gradient Computation Paths

`compute_grads_gpu` in `vmc_utils.py` supports three modes:

| Mode | Flag | Description |
|---|---|---|
| `vmap(grad)` | `vectorize=True, vmap_grad=True` | Per-sample scalar grad via `torch.vmap(torch.func.grad(...))`. **Default and recommended.** |
| `jacrev` | `vectorize=True, vmap_grad=False` | Full Jacobian via `torch.func.jacrev`. Higher memory. |
| Sequential | `vectorize=False` | One-at-a-time `backward()`. Fallback for debugging. |

All modes support memory chunking via `batch_size` (called `grad_batch_size` in `vmc_run.py`).

### SR Solver Options

| Solver | Function | When to use |
|---|---|---|
| Distributed MINRES | `distributed_minres_solver_gpu` | Default. Builds no Np x Np matrix. Local matvec + `all_reduce` of (Np,) vectors. Falls back to pure numpy for single-GPU. |
| MinSR (direct) | `minSR_solver_gpu` | When `Total_Ns < Np`. Gathers all O_loc to rank 0, solves (Ns x Ns) system on GPU. |

Controlled by `use_minSR` flag in `vmc_run.py`.

### GPU-Batched Hamiltonian

The Hamiltonian (`hamiltonian_torch.py`) supports a GPU-batched `get_conn`:

```python
H.precompute_hops_gpu(device)  # one-time, call after init
conn_etas, conn_coeffs, batch_ids = H.get_conn_batch_gpu(fxs)  # no Python loop
```

This avoids CPU-GPU round-trips. `evaluate_energy` checks for `H._hop_list` and uses the GPU path automatically.

### Model Interface

Both `fPEPS_Model_GPU` and `PureNN_GPU` expose the same interface so they're drop-in replaceable in `vmc_run.py`:

- `self.params` — `nn.ParameterList` of all learnable tensors
- `self.forward(x)` — batched `(B, N_sites) -> (B,)`, uses compiled path if available
- `self.vamp(x, params)` — batched eval with explicit params (for `torch.func.grad`)
- `self._compiled`, `self._exported` — status flags

## Shape Conventions

- Configurations: `(B, N_sites)` int64
- Amplitudes: `(B,)` float64 — **not** `(B, 1)`
- O_loc (grads/amps): `(B, Np)` float64
- Connected configs: `(total_conn, N_sites)` where `total_conn = sum(conn_eta_num)`
- `batch_ids`: `(total_conn,)` int64, maps each connected config back to its sample index

## Common Pitfalls

- **Forgetting `torchrun`**: Even single-GPU runs need `torchrun` because `dist.init_process_group("nccl")` requires env vars that `torchrun` sets.
- **Amp shape**: functions return `(B,)` not `(B, 1)`. The `jacrev` path has a `unsqueeze_` call for compatibility but `vmap_grad` path does not.
- **SVD failures**: quimb's boundary contractions use SVD internally. The `robust_svd_err_catcher_wrapper` (registered via `autoray`) adds jitter and falls back to `eigh` when LAPACK SVD doesn't converge. This is registered globally at module load in `vmc_run.py`.
- **torch.compile warm-up**: First call after `export_and_compile` triggers kernel compilation (~10-40s). The warmup block in `vmc_run.py` handles this before the main loop.
