# Notebook — vmc_torch

## 2026-03-01: Extending VMC to Quantum Chemistry

**Question:** What modifications are needed to extend the current lattice-model VMC framework (NQS/TNS/fTNS) to quantum chemistry problems?

**Current status:** Codebase is entirely lattice-model focused (Hubbard, Heisenberg, Ising on chains/square lattices). A `mpi4pyscf` fork exists at `pyscf/` but is not integrated.

**Key findings:** See detailed analysis below in the conversation summary.

**Existing QC-VMC libraries surveyed:**
- **1st quantization (real space):** DeepQMC (PyTorch), FermiNet (JAX), QMCPACK (C++), CASINO (Fortran), CHAMP (Fortran/C). Not compatible with our TNS framework.
- **2nd quantization (Fock space):** NetKet (JAX) — closest in spirit. Choo et al. (2020) demonstrated NQS for molecules in 2nd quantization. No existing library combines TNS/fTNS + VMC for QC.
- **Key reference:** Choo et al., Nature Comm. 2020 — NQS ab initio QC in 2nd quantization via NetKet. Also: "Neural network backflow for ab initio QC" (2024) — similar to our NN-fTNS hybrid.

**Conclusion:** The 2nd-quantization path is natural for our framework. Main work = molecular Hamiltonian `get_conn` + excitation-based sampler. Models (fTNS + backflow) can be reused treating MOs as sites.

**TODO:**
- [ ] Decide scope: ab initio molecules vs lattice-like chemistry (e.g., Hubbard-extended models for materials)
- [ ] Implement molecular Hamiltonian class with h1e/h2e integrals from PySCF
- [ ] Adapt sampler for single/double excitation proposals
- [ ] Bridge PySCF integral output to VMC pipeline
- [ ] Decide on ansatz: multi-Slater-Jastrow, NQS, or TNS-based
- [ ] Read Choo et al. 2020 for 2nd-quantization `get_conn` design
