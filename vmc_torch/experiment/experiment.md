# Code-TODO:

1. Compare fPEPS-NN model with Neural Backflow model on 4x4 spinful Fermi-Hubbard model. U/t=8, n=0.875, E_ed = -11.868. [x] -- Optimization,initialization problems..(sample#:50000)
2. Implement variational PEPS model using BP TN contraction. [?] -- may have convergence issue?
3. Implement variational model for spinful fermions. [x]
4. Implement function to transform a U1 array to a Z2 array. [ ]
5. Or implement a NN model with dynamic output length! [x] -- used a transformer NN model
6. Supervised Wavefunction Optimization (SWO): https://arxiv.org/pdf/1811.12423 and https://journals.aps.org/prb/pdf/10.1103/PhysRevB.110.115124 [x]
    --- Better convergence?
    --- Is it better than SR? or is the imaginary-time SWO equivalent to SR? 
        A: At small step size limit IT-SWO is equivalent to SR, while at large step size it is more like power method for extreme eigenvalue problem.
    --- Seems to work well with Adam, shall we implement?
        A: Yes, start from wave-function fitting MC code -- log-fidelity minimization. [x]
    --- VMC for fidelity: $F = |1/S * \sum_c <c|\phi>/<c|\psi>|^2 / (1/S * \sum_c |<c|\phi>/<c|\psi>|^2)$, where c is sampled from unnormalized \psi
    --- Better use log fidelity?
        A: Yes, as the components are separated and easier to evaluate by sampling.
    --- Store sampled configuration in each rank [x]
7. 1D system:
    --- fMPS [x]
    --- Netket, quimb Hamiltonian [x]
    --- Fully-connected random hopping Hubbard model (arXiv:2311.05749) that demonstrate volume-law, and expensive for DMRG (MPS). Try fTNF. [ ]
    --- SYK model, hard even for HFDS (arXiv:2411.04527), what about for fMPS+NN+attention? Or for TNF? [ ]
# Ideas

# TN+NN:
1. 2D Hubbard model various bond dimension D [...]
2. Non-fully connected neural network for large D fPEPS to reduce number of parameters. [wrong]
3. Add Jastrow factor. [x] need to test effect
4. Check SWO fitting for fMPS+backflow  D=4 [x]
5. Check extended fermionc PEPS ansatz. [ ]





# Volume-law TNF: (HFDS and DMRG benchmarks: arXiv:2311.05749 & arXiv:2411.04527)
1. Check Quimb's MPO contruction routine for QSK model/SYK model using Jordan-Wigner transformation. [x]
2. Fully-connected random hopping Hubbard model (arXiv:2311.05749) that demonstrate volume-law, and expensive for DMRG (MPS). Try fTNF. [...]



# Misc

1. Try Johnnie's non-local TN model. [x] -- delocalized PEPS, indeed increase variational power but boson case too many other equally good ansatz, like deep NNQS.






