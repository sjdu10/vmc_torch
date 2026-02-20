"""
profile_chi_approx.py — Diagnose why GPU is slower than CPU for chi=D.

When chi=D (approximate boundary-MPS contraction), GPU is slower than a
single CPU core. This script pinpoints exactly which phase causes this.

ROOT-CAUSE HYPOTHESES:
  H1. Tiny SVD matrices (chi×chi = 4×4): GPU kernel-launch overhead
      (~5–10 µs/call) >> actual compute time for 4×4 SVD.
  H2. Boundary-MPS options (equalize_norms, canonize=True) add many
      extra GPU ops on tiny tensors, multiplying kernel-launch cost.
  H3. qtn.unpack + isel Python overhead is sequential even under vmap.
  H4. vmap does NOT batch SVDs → O(B × n_svd) sequential kernel calls.

SECTIONS:
  1. SVD call count & shapes  — is vmap batching SVDs or running them
                                  once per sample?
  2. GPU vs CPU single-sample — raw overhead of one amplitude() call
  3. GPU batch-size scaling   — throughput (samples/s) vs B
  4. Phase breakdown          — isel / bnd_xmin / bnd_xmax / contract
  5. Boundary option ablation — impact of equalize_norms / canonize
  6. CUDA kernel profile      — kernel counts & dominant kernels

Run (no torchrun needed):
    python GPU/scripts/profile_chi_approx.py
"""
import os
import time
import pickle
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.profiler as tprof
import autoray as ar
import quimb as qu
import quimb.tensor as qtn

from vmc_torch.experiment.vmap.vmap_torch_utils import (
    size_aware_qr,
    size_aware_svd,
)

# ============================================================
# 0. Distributed init (single-GPU standalone)
# ============================================================
if "RANK" not in os.environ:
    os.environ.update({
        "RANK": "0", "WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost", "MASTER_PORT": "12370",
        "LOCAL_RANK": "0",
    })
dist.init_process_group(backend="nccl", init_method="env://")
GPU = torch.device("cuda:0")
CPU = torch.device("cpu")
torch.cuda.set_device(0)
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

# ============================================================
# SVD call counter (wraps the registered backend function)
# ============================================================
svd_call_log: list = []
_log_svd = False


def _counting_svd(x, jitter=1e-16, driver=None):
    if _log_svd:
        svd_call_log.append(tuple(x.shape))
    return size_aware_svd(x, jitter=jitter, driver=driver)


ar.register_function("torch", "linalg.svd", _counting_svd)
ar.register_function("torch", "linalg.qr", size_aware_qr)


def svd_start():
    global _log_svd
    svd_call_log.clear()
    _log_svd = True


def svd_stop():
    global _log_svd
    _log_svd = False
    return list(svd_call_log)


# ============================================================
# 1. Load PEPS
# ============================================================
Lx, Ly = 4, 4
nsites = Lx * Ly
N_f = nsites - 2
D = 4
chi = D  # approximate contraction

pwd = (
    "/home/sijingdu/TNVMC/VMC_code/vmc_torch/"
    "vmc_torch/experiment/vmap/data"
)
params_pkl = pickle.load(open(
    f"{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/"
    "peps_su_params_U1SU.pkl", "rb"
))
skeleton_pkl = pickle.load(open(
    f"{pwd}/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/"
    "peps_skeleton_U1SU.pkl", "rb"
))


def build_peps():
    peps = qtn.unpack(params_pkl, skeleton_pkl)
    for ts in peps.tensors:
        ts.modify(data=ts.data.to_flat() * 4)
    for site in peps.sites:
        peps[site].data._label = site
        peps[site].data.indices[-1]._linearmap = (
            (0, 0), (1, 0), (1, 1), (0, 1)
        )
    return peps


# ============================================================
# 2. Build amplitude functions (flat-params style, like test_export_chi4)
# ============================================================
peps_ref = build_peps()
packed_params, skel = qtn.pack(peps_ref)
params_flat_list, params_pytree = qu.utils.tree_flatten(
    packed_params, get_ref=True
)

# GPU param tensors
params_gpu = [
    torch.as_tensor(x, dtype=torch.float64, device=GPU)
    for x in params_flat_list
]
# CPU param tensors
params_cpu = [
    torch.as_tensor(x, dtype=torch.float64, device=CPU)
    for x in params_flat_list
]

n_params = sum(p.numel() for p in params_gpu)
print(f"\nPhysics: {Lx}x{Ly}, N_f={N_f}, D={D}, chi={chi}")
print(f"n_params={n_params}, n_tensors={len(params_gpu)}")


def _make_amp_fn(max_bond, contract_opts):
    """Return an amplitude function for given chi and boundary options."""

    def amplitude(x, *flat_params):
        p = qu.utils.tree_unflatten(list(flat_params), params_pytree)
        tn = qtn.unpack(p, skel)
        tnx = tn.isel({
            tn.site_ind(site): x[i]
            for i, site in enumerate(tn.sites)
        })
        if max_bond > 0:
            tnx.contract_boundary_from_xmin_(
                max_bond=max_bond, cutoff=0.0,
                xrange=[0, tnx.Lx // 2 - 1],
                **contract_opts,
            )
            tnx.contract_boundary_from_xmax_(
                max_bond=max_bond, cutoff=0.0,
                xrange=[tnx.Lx // 2, tnx.Lx - 1],
                **contract_opts,
            )
        return tnx.contract()

    return amplitude


# Default boundary options (matches vmc_run.py)
default_opts = {"mode": "mps", "equalize_norms": 1.0, "canonize": True}

amp_fn = _make_amp_fn(chi, default_opts)

# vmap over the batch dimension of x; broadcast params
vmap_amp_gpu = torch.vmap(
    amp_fn,
    in_dims=(0, *([None] * len(params_gpu))),
    randomness="different",
)
vmap_amp_cpu = torch.vmap(
    amp_fn,
    in_dims=(0, *([None] * len(params_cpu))),
    randomness="different",
)

# ============================================================
# Generate configs
# ============================================================
B_MAX = 512


def make_configs(B, device):
    """B random configs on device."""
    configs = []
    for i in range(B):
        torch.manual_seed(42 + i)
        half = torch.tensor([1, 2] * (nsites // 2))
        doped = half.clone()
        doped[:nsites - N_f] = 0
        perm = torch.randperm(nsites)
        configs.append(doped[perm])
    return torch.stack(configs).to(device)


fxs_gpu = make_configs(B_MAX, GPU)
fxs_cpu = make_configs(B_MAX, CPU)

# Warmup (fills compile/JIT caches)
print("\nWarmup...", flush=True)
with torch.inference_mode():
    _ = vmap_amp_gpu(fxs_gpu[:4], *params_gpu)
    torch.cuda.synchronize()
    _ = vmap_amp_cpu(fxs_cpu[:4], *params_cpu)
print("  done.")

# ============================================================
# Helper: wall-clock timer with optional CUDA sync
# ============================================================
def timed(fn, sync=True, n_rep=1):
    """Run fn() n_rep times, return list of wall times in seconds."""
    times = []
    for _ in range(n_rep):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def report(label, times_s, unit="ms"):
    arr = np.array(times_s)
    if unit == "ms":
        arr *= 1000
    mean, std, mn, mx = arr.mean(), arr.std(), arr.min(), arr.max()
    print(f"  {label:45s}: {mean:8.2f} ± {std:6.2f} {unit}"
          f"  (min {mn:.2f}  max {mx:.2f})")


# ============================================================
# SECTION 1: SVD call count — does vmap batch SVDs?
# ============================================================
print("\n" + "=" * 65)
print("SECTION 1: SVD call count — batched by vmap or per sample?")
print("=" * 65)

# 1a. Single sample (no vmap)
svd_start()
with torch.inference_mode():
    _ = amp_fn(fxs_gpu[0], *params_gpu)
    torch.cuda.synchronize()
shapes_single = svd_stop()
n_svd_single = len(shapes_single)
print(f"\n[Single sample via amp_fn()]")
print(f"  SVD calls: {n_svd_single}")
unique: dict = {}
for s in shapes_single:
    unique[s] = unique.get(s, 0) + 1
for shape, count in sorted(unique.items(), key=lambda kv: -kv[1]):
    print(f"  shape {shape}  ×{count}")

# 1b. Batch of B=32 via vmap
B_svd_test = 32
svd_start()
with torch.inference_mode():
    _ = vmap_amp_gpu(fxs_gpu[:B_svd_test], *params_gpu)
    torch.cuda.synchronize()
shapes_batch = svd_stop()
n_svd_batch = len(shapes_batch)
print(f"\n[Batch B={B_svd_test} via vmap_amp_gpu]")
print(f"  SVD calls: {n_svd_batch}")
unique_b: dict = {}
for s in shapes_batch:
    unique_b[s] = unique_b.get(s, 0) + 1
for shape, count in sorted(unique_b.items(), key=lambda kv: -kv[1]):
    print(f"  shape {shape}  ×{count}")

print(f"\n[Verdict]")
if n_svd_batch == n_svd_single:
    print(f"  vmap BATCHES SVDs: {n_svd_single} calls with leading batch dim "
          f"(GOOD — no extra overhead per sample)")
elif n_svd_batch == n_svd_single * B_svd_test:
    print(f"  vmap does NOT batch: {n_svd_batch} = {n_svd_single} × {B_svd_test} "
          f"sequential calls (BAD — scales linearly with B)")
else:
    print(f"  Unexpected: {n_svd_batch} calls "
          f"(expected {n_svd_single} batched or {n_svd_single*B_svd_test} sequential)")

# ============================================================
# SECTION 2: GPU vs CPU — single-sample and batch timing
# ============================================================
print("\n" + "=" * 65)
print("SECTION 2: GPU vs CPU timing (single-sample & B=128)")
print("=" * 65)

N_rep = 30
B_ref = 128

# Warmups
with torch.inference_mode():
    for _ in range(5):
        _ = amp_fn(fxs_gpu[0], *params_gpu)
    torch.cuda.synchronize()
    for _ in range(5):
        _ = amp_fn(fxs_cpu[0], *params_cpu)

# Single-sample GPU
ts_gpu1 = timed(
    lambda: amp_fn(fxs_gpu[0], *params_gpu),
    sync=True, n_rep=N_rep
)
# Single-sample CPU
ts_cpu1 = timed(
    lambda: amp_fn(fxs_cpu[0], *params_cpu),
    sync=False, n_rep=N_rep
)

# Warmup vmap
with torch.inference_mode():
    for _ in range(3):
        _ = vmap_amp_gpu(fxs_gpu[:B_ref], *params_gpu)
    torch.cuda.synchronize()
    for _ in range(3):
        _ = vmap_amp_cpu(fxs_cpu[:B_ref], *params_cpu)

# Batch GPU (vmap)
ts_gpu_B = timed(
    lambda: vmap_amp_gpu(fxs_gpu[:B_ref], *params_gpu),
    sync=True, n_rep=N_rep
)
# Batch CPU (vmap)
ts_cpu_B = timed(
    lambda: vmap_amp_cpu(fxs_cpu[:B_ref], *params_cpu),
    sync=False, n_rep=N_rep
)

print(f"\nN_rep={N_rep}")
report("GPU  amp_fn (1 sample)", ts_gpu1)
report("CPU  amp_fn (1 sample)", ts_cpu1)
ratio1 = np.mean(ts_gpu1) / np.mean(ts_cpu1)
print(f"  → GPU/CPU ratio (1 sample): {ratio1:.2f}x  "
      f"({'SLOWER' if ratio1 > 1 else 'faster'} on GPU)")

report(f"GPU  vmap (B={B_ref})", ts_gpu_B)
report(f"CPU  vmap (B={B_ref})", ts_cpu_B)
ratio_B = np.mean(ts_gpu_B) / np.mean(ts_cpu_B)
print(f"  → GPU/CPU ratio (B={B_ref}): {ratio_B:.2f}x  "
      f"({'SLOWER' if ratio_B > 1 else 'faster'} on GPU)")

# Per-sample throughput
gpu_per = np.mean(ts_gpu_B) / B_ref * 1000
cpu_per = np.mean(ts_cpu_B) / B_ref * 1000
print(f"  GPU ms/sample (batch): {gpu_per:.3f} ms")
print(f"  CPU ms/sample (batch): {cpu_per:.3f} ms")

# ============================================================
# SECTION 3: GPU batch-size scaling
# ============================================================
print("\n" + "=" * 65)
print("SECTION 3: GPU batch-size scaling (throughput vs B)")
print("=" * 65)

batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512]
N_rep_b = 10
print(f"\n{'B':>6}  {'time(ms)':>10}  {'ms/sample':>12}  {'samples/s':>12}")
for B in batch_sizes:
    fxs_b = fxs_gpu[:B]
    # Warmup
    with torch.inference_mode():
        for _ in range(3):
            _ = vmap_amp_gpu(fxs_b, *params_gpu)
        torch.cuda.synchronize()
    ts = timed(
        lambda: vmap_amp_gpu(fxs_b, *params_gpu),  # noqa: B023
        sync=True, n_rep=N_rep_b
    )
    t_ms = np.mean(ts) * 1000
    per = t_ms / B
    sps = B / np.mean(ts)
    print(f"{B:>6}  {t_ms:>10.2f}  {per:>12.3f}  {sps:>12.0f}")

print(f"\nCPU batch scaling:")
print(f"{'B':>6}  {'time(ms)':>10}  {'ms/sample':>12}  {'samples/s':>12}")
for B in [1, 4, 8, 16, 32, 64]:
    fxs_b = fxs_cpu[:B]
    with torch.inference_mode():
        for _ in range(2):
            _ = vmap_amp_cpu(fxs_b, *params_cpu)
    ts = timed(
        lambda: vmap_amp_cpu(fxs_b, *params_cpu),  # noqa: B023
        sync=False, n_rep=5
    )
    t_ms = np.mean(ts) * 1000
    per = t_ms / B
    sps = B / np.mean(ts)
    print(f"{B:>6}  {t_ms:>10.2f}  {per:>12.3f}  {sps:>12.0f}")

# ============================================================
# SECTION 4: Phase-by-phase breakdown inside amplitude()
# ============================================================
print("\n" + "=" * 65)
print("SECTION 4: Phase breakdown inside amp_fn() — GPU vs CPU")
print("=" * 65)

N_rep_ph = 20


def time_phases(x, flat_params, device, n_rep=N_rep_ph):
    """Time each phase of amplitude() separately on device."""
    is_gpu = device.type == "cuda"
    sync = torch.cuda.synchronize if is_gpu else lambda: None

    times: dict[str, list[float]] = {
        k: [] for k in ["unpack+isel", "bnd_xmin", "bnd_xmax", "contract"]
    }

    for trial in range(n_rep + 3):  # 3 warmup
        p = qu.utils.tree_unflatten(list(flat_params), params_pytree)

        # --- Phase 1: unpack + isel ---
        sync()
        t0 = time.perf_counter()
        tn = qtn.unpack(p, skel)
        tnx = tn.isel({
            tn.site_ind(site): x[i]
            for i, site in enumerate(tn.sites)
        })
        sync()
        t_isel = time.perf_counter() - t0

        # --- Phase 2: contract_boundary_from_xmin ---
        sync()
        t0 = time.perf_counter()
        tnx.contract_boundary_from_xmin_(
            max_bond=chi, cutoff=0.0,
            xrange=[0, tnx.Lx // 2 - 1],
            **default_opts,
        )
        sync()
        t_xmin = time.perf_counter() - t0

        # --- Phase 3: contract_boundary_from_xmax ---
        sync()
        t0 = time.perf_counter()
        tnx.contract_boundary_from_xmax_(
            max_bond=chi, cutoff=0.0,
            xrange=[tnx.Lx // 2, tnx.Lx - 1],
            **default_opts,
        )
        sync()
        t_xmax = time.perf_counter() - t0

        # --- Phase 4: final scalar contraction ---
        sync()
        t0 = time.perf_counter()
        _ = tnx.contract()
        sync()
        t_contract = time.perf_counter() - t0

        if trial >= 3:  # skip warmup
            times["unpack+isel"].append(t_isel)
            times["bnd_xmin"].append(t_xmin)
            times["bnd_xmax"].append(t_xmax)
            times["contract"].append(t_contract)

    return times


with torch.inference_mode():
    print(f"\nGPU (single sample, N={N_rep_ph} trials):")
    t_gpu_ph = time_phases(fxs_gpu[0], params_gpu, GPU)
    total_gpu_ph = sum(np.mean(v) for v in t_gpu_ph.values())
    for phase, ts in t_gpu_ph.items():
        ms = np.mean(ts) * 1000
        pct = ms / (total_gpu_ph * 1000) * 100
        print(f"  {phase:20s}: {ms:7.3f} ms  ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total_gpu_ph*1000:7.3f} ms")

    print(f"\nCPU (single sample, N={N_rep_ph} trials):")
    t_cpu_ph = time_phases(fxs_cpu[0], params_cpu, CPU)
    total_cpu_ph = sum(np.mean(v) for v in t_cpu_ph.values())
    for phase, ts in t_cpu_ph.items():
        ms = np.mean(ts) * 1000
        pct = ms / (total_cpu_ph * 1000) * 100
        print(f"  {phase:20s}: {ms:7.3f} ms  ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total_cpu_ph*1000:7.3f} ms")

print(f"\nPer-phase GPU/CPU ratio:")
for phase in t_gpu_ph:
    g_ms = np.mean(t_gpu_ph[phase]) * 1000
    c_ms = np.mean(t_cpu_ph[phase]) * 1000
    r = g_ms / max(c_ms, 1e-9)
    print(f"  {phase:20s}: GPU={g_ms:.3f}ms  CPU={c_ms:.3f}ms  "
          f"ratio={r:.2f}x {'←BOTTLENECK' if r > 2 else ''}")

# ============================================================
# SECTION 5: Boundary option ablation (equalize_norms, canonize)
# ============================================================
print("\n" + "=" * 65)
print("SECTION 5: Boundary option ablation — GPU B=128")
print("=" * 65)

ablation_configs = [
    ("Full (norm+canonize)", {"mode": "mps", "equalize_norms": 1.0, "canonize": True}),
    ("No canonize        ", {"mode": "mps", "equalize_norms": 1.0, "canonize": False}),
    ("No equalize_norms  ", {"mode": "mps", "canonize": True}),
    ("Bare MPS           ", {"mode": "mps"}),
]

B_abl = 128
N_rep_abl = 10

print(f"\n  B={B_abl}, N_rep={N_rep_abl}")
for label, opts in ablation_configs:
    fn = _make_amp_fn(chi, opts)
    vf = torch.vmap(
        fn,
        in_dims=(0, *([None] * len(params_gpu))),
        randomness="different",
    )
    # Warmup
    with torch.inference_mode():
        for _ in range(3):
            _ = vf(fxs_gpu[:B_abl], *params_gpu)
        torch.cuda.synchronize()
    ts = timed(
        lambda: vf(fxs_gpu[:B_abl], *params_gpu),  # noqa: B023
        sync=True, n_rep=N_rep_abl,
    )
    t_ms = np.mean(ts) * 1000
    print(f"  {label}: {t_ms:8.2f} ms   ({t_ms/B_abl:.3f} ms/sample)")

# ============================================================
# SECTION 6: CUDA kernel profile (torch.profiler)
# ============================================================
print("\n" + "=" * 65)
print("SECTION 6: CUDA kernel profile (chi=D, B=128)")
print("=" * 65)

B_prof = 128
fxs_prof = fxs_gpu[:B_prof]

# Warmup
with torch.inference_mode():
    for _ in range(3):
        _ = vmap_amp_gpu(fxs_prof, *params_gpu)
torch.cuda.synchronize()

script_dir = os.path.dirname(os.path.abspath(__file__))
profile_dir = os.path.join(script_dir, "..", "profiles", "chi_approx")
os.makedirs(profile_dir, exist_ok=True)

with tprof.profile(
    activities=[tprof.ProfilerActivity.CPU, tprof.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
    on_trace_ready=tprof.tensorboard_trace_handler(profile_dir),
) as prof:
    with torch.inference_mode():
        with tprof.record_function("vmap_amp_chi=D"):
            _ = vmap_amp_gpu(fxs_prof, *params_gpu)
    torch.cuda.synchronize()
    prof.step()

key_avgs = prof.key_averages()

# Total kernel invocation count
total_kernel_calls = sum(
    e.count for e in key_avgs if e.device_time_total > 0
)
total_cuda_us = sum(e.device_time_total for e in key_avgs)
total_cpu_us = sum(e.cpu_time_total for e in key_avgs)

print(f"\n  Total CUDA kernel calls    : {total_kernel_calls}")
print(f"  Calls per sample (B={B_prof}) : {total_kernel_calls/B_prof:.1f}")
print(f"  Total CUDA time (ms)       : {total_cuda_us/1000:.2f}")
print(f"  Total CPU  time (ms)       : {total_cpu_us/1000:.2f}")
print(f"  CUDA utilization           : "
      f"{total_cuda_us/(total_cuda_us+total_cpu_us)*100:.1f}%")

print(f"\nTop 20 kernels by device time:")
print(key_avgs.table(sort_by="device_time_total", row_limit=20))

print(f"\nTop 15 ops by CPU time (Python overhead):")
print(key_avgs.table(sort_by="self_cpu_time_total", row_limit=15))

print(f"\nTrace written to: {os.path.abspath(profile_dir)}")
print("  View: tensorboard --logdir", os.path.abspath(profile_dir))

# ============================================================
# SECTION 7: torch.export + vmap + compile (QR-via-SVD)
# ============================================================
print("\n" + "=" * 65)
print("SECTION 7: torch.export + vmap + compile (chi=D, QR-via-SVD)")
print("=" * 65)

from torch.export import export


class AmpModule(torch.nn.Module):
    def forward(self, x, *flat_params):
        return amp_fn(x, *flat_params)


# --- Export ---
print("\nExporting single-sample amplitude function...")
t_exp = time.time()
try:
    with torch.inference_mode():
        exported = export(AmpModule(), (fxs_gpu[0], *params_gpu))
    t_export = time.time() - t_exp
    print(f"  Export time: {t_export:.2f}s")

    # --- Vmap the exported module ---
    exported_mod = exported.module()
    vf_exp = torch.vmap(
        exported_mod,
        in_dims=(0, *([None] * len(params_gpu))),
        randomness="different",
    )

    # --- Compile ---
    vf_compiled = torch.compile(vf_exp, mode="default")

    # --- Warmup (triggers compilation) ---
    # Run without inference_mode to match benchmark conditions and
    # avoid recompilation from grad_mode changes.
    print("Compiling (first call)...", flush=True)
    t_comp = time.time()
    out_compiled = vf_compiled(fxs_gpu[:B_ref], *params_gpu)
    torch.cuda.synchronize()
    t_compile = time.time() - t_comp
    print(f"  Compile + first call: {t_compile:.2f}s")

    # --- Numerical check ---
    out_eager = vmap_amp_gpu(fxs_gpu[:B_ref], *params_gpu)
    torch.cuda.synchronize()
    maxdiff = (out_compiled - out_eager).abs().max().item()
    reldiff = maxdiff / out_eager.abs().max().item()
    print(f"  Max abs diff vs eager: {maxdiff:.2e}")
    print(f"  Max rel diff vs eager: {reldiff:.2e}")
    print(f"  Matches: {reldiff < 1e-6}")

    # --- Additional warmup (all without inference_mode to match benchmark) ---
    for _ in range(10):
        _ = vf_compiled(fxs_gpu[:B_ref], *params_gpu)
    torch.cuda.synchronize()

    # --- Benchmark: compiled GPU vs eager GPU vs CPU ---
    # All benchmarks run WITHOUT inference_mode to avoid recompilation
    # from grad_mode changes.
    N_rep_ec = 30
    print(f"\nBenchmark (N_rep={N_rep_ec}):")

    # Compiled GPU
    ts_compiled = timed(
        lambda: vf_compiled(fxs_gpu[:B_ref], *params_gpu),
        sync=True, n_rep=N_rep_ec,
    )
    report(f"GPU compiled (B={B_ref})", ts_compiled)

    # Eager GPU
    ts_eager_gpu = timed(
        lambda: vmap_amp_gpu(fxs_gpu[:B_ref], *params_gpu),
        sync=True, n_rep=N_rep_ec,
    )
    report(f"GPU eager   (B={B_ref})", ts_eager_gpu)

    # CPU eager
    ts_eager_cpu = timed(
        lambda: vmap_amp_cpu(fxs_cpu[:B_ref], *params_cpu),
        sync=False, n_rep=N_rep_ec,
    )
    report(f"CPU eager   (B={B_ref})", ts_eager_cpu)

    compiled_ms = np.mean(ts_compiled) * 1000
    eager_gpu_ms = np.mean(ts_eager_gpu) * 1000
    eager_cpu_ms = np.mean(ts_eager_cpu) * 1000

    print(f"\n  Compiled/Eager GPU speedup : {eager_gpu_ms/compiled_ms:.2f}x")
    print(f"  Compiled GPU / CPU ratio   : {compiled_ms/eager_cpu_ms:.2f}x "
          f"({'SLOWER' if compiled_ms > eager_cpu_ms else 'FASTER'} on GPU)")
    print(f"  Compiled ms/sample         : {compiled_ms/B_ref:.3f}")
    print(f"  CPU ms/sample              : {eager_cpu_ms/B_ref:.3f}")

    # --- Batch-size scaling (compiled, fixed shape to avoid recompile) ---
    # torch.compile recompiles for each new shape, so we pad to B_MAX
    # and only vary the "active" count. For a clean comparison we compile
    # once at B_MAX and benchmark at that fixed shape.
    B_compile_sizes = [128, 256, 512]
    print(f"\nCompiled GPU batch-size scaling "
          f"(separate compile per B):")
    print(f"{'B':>6}  {'time(ms)':>10}  {'ms/sample':>12}  {'samples/s':>12}")
    for B_sc in B_compile_sizes:
        fxs_sc = fxs_gpu[:B_sc]
        # Warmup (triggers compile for this shape)
        for _ in range(5):
            _ = vf_compiled(fxs_sc, *params_gpu)
        torch.cuda.synchronize()
        # Now benchmark (should be stable)
        ts_sc = timed(
            lambda: vf_compiled(fxs_sc, *params_gpu),  # noqa: B023
            sync=True, n_rep=10,
        )
        t_ms_sc = np.mean(ts_sc) * 1000
        per_sc = t_ms_sc / B_sc
        sps_sc = B_sc / np.mean(ts_sc)
        print(f"{B_sc:>6}  {t_ms_sc:>10.2f}  {per_sc:>12.3f}  {sps_sc:>12.0f}")

    # --- CUDA kernel profile (compiled) ---
    print(f"\nCUDA kernel profile (compiled, B={B_ref}):")
    # Re-warmup at B_ref to ensure compiled graph is cached
    for _ in range(5):
        _ = vf_compiled(fxs_gpu[:B_ref], *params_gpu)
    torch.cuda.synchronize()

    profile_dir_comp = os.path.join(
        script_dir, "..", "profiles", "chi_approx_compiled"
    )
    os.makedirs(profile_dir_comp, exist_ok=True)

    with tprof.profile(
        activities=[
            tprof.ProfilerActivity.CPU,
            tprof.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
        on_trace_ready=tprof.tensorboard_trace_handler(profile_dir_comp),
    ) as prof_c:
        with tprof.record_function("compiled_amp_chi=D"):
            _ = vf_compiled(fxs_gpu[:B_ref], *params_gpu)
        torch.cuda.synchronize()
        prof_c.step()

    key_avgs_c = prof_c.key_averages()
    total_kern_c = sum(
        e.count for e in key_avgs_c if e.device_time_total > 0
    )
    total_cuda_c = sum(e.device_time_total for e in key_avgs_c)

    print(f"  Total CUDA kernel calls    : {total_kern_c}")
    print(f"  Calls per sample           : {total_kern_c/B_ref:.1f}")
    print(f"  Total CUDA time (ms)       : {total_cuda_c/1000:.2f}")

    print(f"\nTop 20 kernels by device time (compiled):")
    print(key_avgs_c.table(sort_by="device_time_total", row_limit=20))

    print(f"\nTrace: {os.path.abspath(profile_dir_comp)}")

    export_success = True

except Exception as e:
    print(f"\nEXPORT FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    export_success = False

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"\n  n_SVD calls per forward (single): {n_svd_single}")
print(f"  n_SVD calls per forward (B={B_svd_test} vmap): {n_svd_batch}")
if n_svd_batch == n_svd_single:
    print(f"  → vmap batches SVDs correctly (calls with leading batch dim)")
else:
    print(f"  → vmap does NOT batch SVDs ({n_svd_batch} != {n_svd_single})")

print(f"\n  Single-sample GPU/CPU ratio    : {ratio1:.2f}x")
print(f"  Batch (B={B_ref}) GPU/CPU ratio    : {ratio_B:.2f}x")
print(f"  CUDA calls per sample          : {total_kernel_calls/B_prof:.1f}")

print(f"""
INTERPRETATION GUIDE:
  If CUDA calls/sample > 100        → kernel-launch overhead dominates
  If ratio1 > 1 (GPU slower/sample) → overhead at tiny tensor sizes
  If phase bnd_xmin/xmax >> isel    → boundary SVD is the bottleneck
  If ratio1 ≈ 1 but ratio_B > 1    → batching is inefficient (serialized)
  Ablation: if "Bare MPS" ≈ "Full"  → normalize/canonize not the issue
""")

dist.destroy_process_group()
