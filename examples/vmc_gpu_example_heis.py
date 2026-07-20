"""Self-contained VMC example: dense PEPS ground state of Heisenberg.

A dense (non-symmetric) PEPS VMC example: the on-site tensors are plain
dense arrays (phys_dim=2 spin-1/2), as opposed to a block-sparse
Z2-fermionic fPEPS.  The optimization loop
is byte-for-byte the same idea -- only the model, the PEPS generator,
the Hamiltonian, the sampler, and the config initializer differ:

    fPEPS (fermions)                 dense PEPS (spins, this file)
    ----------------------------     ----------------------------------
    fPEPS_Model_GPU                  PEPS_Model_GPU
    load_or_generate_peps (Z2)       generate_random_spin_peps (dense)
    spinful_Fermi_Hubbard...         spin_Heisenberg_square_lattice...
    MetropolisExchangeSpinful...     MetropolisExchangeSpin...
    random_initial_config            random_spin_config_sz0

Everything below the model definition -- sample -> E_loc -> O_loc ->
preconditioner.solve -> explicit theta update -- is unchanged, because
all models share the WavefunctionModel_GPU interface and all solvers
share PreconditionerGPU.solve().

Multi-GPU is real (data-parallel over walkers, reductions inside the
solve / energy stats).  Launch:

    python vmc_gpu_example_heis.py                    # single GPU
    torchrun --nproc_per_node=<N> vmc_gpu_example_heis.py

Switch the update rule with ``UPDATE_METHOD`` below.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from vmc_torch.GPU.VMC import print_sampling_settings, setup_distributed
from vmc_torch.GPU.hamiltonian import spin_Heisenberg_square_lattice_torch
# The generic base class only supplies the vmap batching, the
# forward / forward_log dispatch, and the optional torch.export+compile
# machinery -- it is NOT the variational ansatz.  The ansatz itself
# (PEPS_Model_GPU) is defined explicitly further down in this file.
from vmc_torch.GPU.models import WavefunctionModel_GPU
from vmc_torch.GPU.optimizer import (
    IterSRGPU,
    MinSRGPU,
    TrivialPreconditionerGPU,
)
from vmc_torch.GPU.sampler import MetropolisExchangeSpinSamplerGPU
from vmc_torch.GPU.vmc_setup import (
    generate_random_spin_peps,
    initialize_walkers,
    random_spin_config_sz0,
    setup_linalg_hooks,
)
from vmc_torch.GPU.vmc_utils import compute_grads_gpu, evaluate_energy
from vmc_torch.GPU.vmc_utils import apply_update, broadcast_params, global_energy_stats

# ============================================================
# Settings
# ============================================================
UPDATE_METHOD = 'sr'      # 'sr' (Np-form SR) | 'minsr' (Ns-form) | 'raw'

# System: spin-1/2 Heisenberg on an Lx x Ly square lattice, Sz=0 sector.
Lx, Ly = 4, 4
J = 1.0                                 # antiferromagnetic coupling
N_sites = Lx * Ly
D = 4                                   # PEPS bond dimension
chi = -1                                # boundary-MPS bond (-1 = exact)

# Optimization
B = 256                                 # walkers PER RANK (== Ns per rank)
BURN_IN = 20                            # burn-in sweeps before step 0
N_STEPS = 50                            # number of VMC steps
LR = 0.05                               # learning rate
SR_ASHIFT = 5e-4                        # absolute Tikhonov shift lambda
SR_RSHIFT = 0.0                         # relative shift (0 -> pure absolute)

# float64 for TN contraction + SR (see tnvmc-physics notes).  float32 +
# torch.set_float32_matmul_precision('high') is the faster production
# path and also works here; float64 is the robust default.
dtype = torch.float64
USE_LOG_AMP = True                      # amplitudes carried as (sign, log|psi|)
USE_EXPORT_COMPILE = False              # torch.export+compile for faster eval


# ============================================================
# Variational model definition (dense, non-symmetric PEPS)
# ============================================================
class PEPS_Model_GPU(WavefunctionModel_GPU):
    """Dense PEPS wavefunction psi_theta(x), defined explicitly here.

    The learnable parameters ARE the PEPS site tensors.  The amplitude
    of a configuration x is computed by:
      1. projecting every site's physical index onto x[i]  (isel),
      2. contracting the resulting 2D network by boundary-MPS from both
         ends toward the middle, truncating the boundary bond to chi
         (chi <= 0 -> exact contraction, no truncation),
      3. reading off the remaining scalar.

    quimb's pack/unpack round-trips the tensors between a flat list (what
    torch trains) and the structured TN (what we contract).  The base
    class supplies vmap batching, forward/forward_log, and export+compile.
    """

    def __init__(
        self,
        tn,
        max_bond,
        dtype=torch.float64,
        contract_boundary_opts=None,
        **kwargs,
    ):
        import quimb as qu
        import quimb.tensor as qtn

        if contract_boundary_opts is None:
            contract_boundary_opts = {}

        # Split the TN into numeric params + a structural skeleton.
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.contract_boundary_opts = contract_boundary_opts
        self.chi = max_bond

        # Flatten the param pytree to a flat list for torch; keep the
        # pytree reference so the forward pass can rebuild the TN.
        params_flat, params_pytree = qu.utils.tree_flatten(
            params, get_ref=True,
        )
        self.params_pytree = params_pytree
        params_tensors = [
            torch.as_tensor(x, dtype=self.dtype) for x in params_flat
        ]
        # Registers params_tensors as an nn.ParameterList and builds the
        # vmapped amplitude / log_amplitude.
        super().__init__(params_list=params_tensors)

    def _amp_tn(self, x, params):
        """Rebuild the TN, project onto config x, get the single-layer amplitude TN."""
        import quimb.tensor as qtn

        tn = qtn.unpack(params, self.skeleton)
        # Project each physical index onto the sampled local state x[i].
        amp = tn.isel({
            tn.site_ind(site): x[i]
            for i, site in enumerate(tn.sites)
        })
        return amp

    def amplitude(self, x, params):
        """Single config x (N_sites,) int64 -> scalar amplitude psi(x)."""
        amp = self._amp_tn(x, params)
        # Boundary-MPS contraction from both ends toward the middle.
        if self.chi > 0:
            amp.contract_boundary_from_xmin_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[0, amp.Lx // 2 - 1],
                **self.contract_boundary_opts,
            )
            amp.contract_boundary_from_xmax_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[amp.Lx // 2, amp.Lx - 1],
                **self.contract_boundary_opts,
            )
        return amp.contract()

    def log_amplitude(self, x, params):
        """Single config x -> (sign, log|psi(x)|).

        Same contraction as amplitude(), but strip_exponent keeps it in
        log-space (avoids under/overflow on large lattices).
        """
        amp = self._amp_tn(x, params)
        if self.chi > 0:
            amp.contract_boundary_from_xmin_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[0, amp.Lx // 2 - 1],
                **self.contract_boundary_opts,
            )
            amp.contract_boundary_from_xmax_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[amp.Lx // 2, amp.Lx - 1],
                **self.contract_boundary_opts,
            )
        sign, exponent_10 = amp.contract(strip_exponent=True)
        exp = exponent_10 * torch.log(torch.tensor(10.0))
        return sign, exp

    def _vamp_params_preprocess(self, params):
        """ParameterList -> quimb param pytree (inverse of __init__'s
        flatten), so amplitude()/log_amplitude() receive a TN pytree."""
        import quimb as qu

        if isinstance(params, nn.ParameterList):
            params = list(params)
        return qu.utils.tree_unflatten(params, self.params_pytree)


# ============================================================
# Main
# ============================================================
def main():
    # Robust, differentiable SVD/QR for the boundary-MPS contraction
    # (never use bare torch.linalg.svd here; see torch_utils.py).
    setup_linalg_hooks(jitter=1e-8, qr_via_eigh=True, nonuniform_diag=True)

    # Distributed setup.  Works with or without torchrun: without it,
    # this falls back to a single rank / single device.
    rank, world_size, device = setup_distributed()
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.manual_seed(42 + rank)        # per-rank -> independent chains

    try:
        # -------- Hamiltonian (spin-1/2 Heisenberg, square lattice) ---
        H = spin_Heisenberg_square_lattice_torch(Lx, Ly, J=J, total_sz=0)
        H.precompute_hops_gpu(device)   # GPU-batched get_conn
        graph = H.graph

        # Exact diagonalization reference for small lattices.
        if rank == 0 and N_sites <= 16:
            import scipy.sparse.linalg as sla
            gs_e = sla.eigsh(H.to_dense(), k=1, which='SA', tol=1e-8)[0][0]
            print(f"ED ground-state E/site: {gs_e / N_sites:.6f}")

        # -------- Model definition (normal dense PEPS) ----------------
        # generate_random_spin_peps returns a plain quimb PEPS with
        # phys_dim=2 and DENSE on-site tensors (no symmetry / block
        # sparsity).  PEPS_Model_GPU wraps it and exposes:
        #     model(x)             -> (B,) amplitudes psi(x)
        #     model.forward_log(x) -> ((B,) signs, (B,) log|psi|)
        #     model.params         -> nn.ParameterList of TN tensors
        peps = generate_random_spin_peps(Lx, Ly, D, seed=42, dtype=dtype)
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
        broadcast_params(model)         # all ranks start from same theta
        Np = sum(p.numel() for p in model.parameters())

        # -------- Optional: export + compile for faster eval ----------
        # torch.export flattens the quimb TN contraction into a pure-aten
        # FX graph (plain torch.compile can't trace quimb), then vmap +
        # torch.compile fuse it into GPU kernels.  After this,
        # forward()/forward_log()/compute_grads_gpu AUTO-dispatch to the
        # compiled path -- no change to the loop below.
        if USE_EXPORT_COMPILE:
            example_x = random_spin_config_sz0(N_sites, seed=0).to(device)
            model.export_and_compile(
                example_x, use_log_amp=USE_LOG_AMP, cache_dir=None,
            )
            model.export_grad(use_log_amp=USE_LOG_AMP, do_compile=False)
            if rank == 0:
                print("Model exported + compiled.")

        # -------- Update rule = choice of preconditioner --------------
        # Every preconditioner in optimizer.py exposes the SAME
        # PreconditionerGPU.solve(O_loc, E_loc, ...) -> (dp, t, info)
        # and is distributed-aware, so switching solver is one line.
        if UPDATE_METHOD == 'raw':
            # Returns the bare energy gradient  <O E> - <O><E>.
            preconditioner = TrivialPreconditionerGPU()
        elif UPDATE_METHOD == 'sr':
            # SR, Np-form: solve (S + lambda I) dp = grad by distributed
            # MINRES (matrix-free; the Np x Np QGT is never formed).
            preconditioner = IterSRGPU(rtol=1e-4, maxiter=100)
        elif UPDATE_METHOD == 'minsr':
            # minSR, Ns-form: solve the Ns x Ns Gram system by Cholesky
            # (all_to_all column split + all_reduce). Cheaper when Ns<Np.
            preconditioner = MinSRGPU()
        else:
            raise ValueError(f"Unknown UPDATE_METHOD {UPDATE_METHOD!r}")

        if rank == 0:
            print(f"System: {Lx}x{Ly} Heisenberg, J={J}, Sz=0")
            print(
                f"Model: {model._get_name()} (dense) | Np={Np} | "
                f"world_size={world_size} | dtype={dtype} | "
                f"update={UPDATE_METHOD}"
            )
        print_sampling_settings(rank, world_size, B, B, B)

        # -------- Sampler + initial walkers ---------------------------
        sampler = MetropolisExchangeSpinSamplerGPU()
        fxs = initialize_walkers(
            init_fn=lambda seed: random_spin_config_sz0(N_sites, seed=seed),
            batch_size=B, seed=42, rank=rank, device=device,
        )
        # Burn in the Markov chain so step 0 already samples |psi|^2.
        fxs = sampler.burn_in(
            fxs, model, graph, BURN_IN,
            compile=USE_EXPORT_COMPILE, use_log_amp=USE_LOG_AMP,
        )

        # ================= VMC / optimization loop ================
        if rank == 0:
            print(f"\n--- VMC ({N_STEPS} steps) ---")
        for step in range(N_STEPS):
            t0 = time.time()

            # 1. MCMC sweep -> new configs + amps at those configs.
            with torch.inference_mode():
                fxs, amps = sampler.step(
                    fxs, model, graph,
                    compile=USE_EXPORT_COMPILE, use_log_amp=USE_LOG_AMP,
                )

                # 2. Local energies E_loc, shape (B,).
                _, E_loc = evaluate_energy(
                    fxs, model, H, amps, use_log_amp=USE_LOG_AMP,
                )

            # 3. Per-sample log-derivatives O_loc, shape (B, Np):
            #    O_loc[b, k] = d log|psi(x_b)| / d theta_k.
            with torch.enable_grad():
                O_loc, _ = compute_grads_gpu(
                    fxs, model, batch_size=B, use_log_amp=USE_LOG_AMP,
                )

            # 4. Global energy statistics (reduce over all ranks).
            total_ns, e_mean, e_var = global_energy_stats(E_loc, world_size)

            # 5. Update DIRECTION from the production solver.  raw vs SR
            #    differ here; cross-rank all_reduce happens in solve().
            dp, t_sr, info = preconditioner.solve(
                O_loc=O_loc,
                E_loc=E_loc,
                E_mean=e_mean,
                Ns=total_ns,
                Np=Np,
                rshift=SR_RSHIFT,
                ashift=SR_ASHIFT,
                device=device,
            )
            dp = torch.as_tensor(dp, device=device)

            # 6. Explicit parameter update: theta <- theta - lr * dp.
            apply_update(model, dp, LR)
            broadcast_params(model)     # keep replicas bit-identical

            if rank == 0:
                e_site = e_mean / N_sites
                err = np.sqrt(max(e_var, 0.0) / total_ns) / N_sites
                print(
                    f"Step {step:3d} | E/site = {e_site:+.6f} "
                    f"+/- {err:.6f} | Ns={total_ns} | "
                    f"|dp|={dp.norm().item():.3e} | "
                    f"T={time.time() - t0:.2f}s"
                )

        if rank == 0:
            print("\nDone.")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
