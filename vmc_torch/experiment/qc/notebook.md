# QC-VMC Experiment Notebook

## 2026-03-01: Initial setup

**Goal:** Reproduce Liu & Clark PRB 110, 115137 (2024) — NNBF for ab initio QC.

### What was done
1. Implemented `CompleteGraph` in `hamiltonian_torch.py` — fully connected graph for molecular orbital pairs.
2. Implemented `MolecularHamiltonian` in `hamiltonian_torch.py` — handles 1e/2e integrals, diagonal/single/double excitations with fermionic phases.
3. Implemented `NNBF_QC` in `model.py` — multi-determinant NNBF with correct dimensions (n_orbitals, not hilbert.size), GELU activation, HF initialization.
4. Created `pyscf_learn/generate_table1_data.py` — generates all Table I molecular data.
5. Verified `MolecularHamiltonian.to_dense()` matches PySCF FCI exactly for H2, H4, H6, N2, CH4.
6. Created `vmc_run_qc.py` — VMC run script wiring everything together.

### Reference energies (STO-3G, our geometries)

| System | M | Hilbert dim | E_HF | E_FCI | E_CCSD |
|--------|---|-------------|------|-------|--------|
| H2 | 2 | 4 | -1.1168 | -1.1373 | -1.1373 |
| H4 | 4 | 36 | -2.0985 | -2.1664 | -2.1664 |
| H6 | 6 | 400 | -2.7502 | -2.9956 | -2.9999 |
| N2 | 10 | 14,400 | -107.4959 | -107.6528 | -107.6489 |
| CH4 | 9 | 15,876 | -39.7268 | -39.8057 | -39.8055 |
| LiF | 10 | 44,100 | -105.3625 | -105.4347 | -105.4178 |
| LiCl | 14 | 1,002,001 | -461.9906 | -462.0099 | -462.0076 |
| Li2O | 15 | 41M | -88.5775 | — | -88.6938 |
| C2H4O | 19 | 2.5B | -150.9408 | — | -151.1529 |

### Next steps
- [ ] Run VMC on H2 (phase 0 sanity check)
- [ ] Run VMC on H4, H6 (phase 1 bridge)
- [ ] Verify HF energy recovery with nn_eta=0
- [ ] Run VMC on N2, CH4 (phase 2 — match FCI)
- [ ] Larger systems: LiF, LiCl, Li2O, C2H4O

### Remaining questions
- Our FCI values differ from the paper's Table I — likely different geometries. Use our PySCF FCI as ground truth.
- FSSC (full sum over Slater configurations) not yet implemented — using MCMC-VMC first.
- Need to tune: learning rate, SR diag_eta, N_samples, burn-in for molecular systems.
