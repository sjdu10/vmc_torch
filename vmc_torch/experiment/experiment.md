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
    --- Fully-connected random hopping Hubbard model (arXiv:2311.05749) that demonstrate volume-law, and expensive for DMRG (MPS). Try fTNF. [? hard to optimize]
    --- SYK model, hard even for HFDS (arXiv:2411.04527), what about for fMPS+NN+attention? Or for TNF? [ ]

8. Implement NNBF [x]
9. Implement HFDS [ ]

# Numerics TODO:
1. check 4x2 doped calculation. Compare fPEPS and fPEPS+NN
    --- u1 [x for D=4]
    --- z2 [x for D=4]
2. check 4x4 calculation
    --- half-filling
        --- u1
        --- z2

    --- doped filling
        --- u1
        --- z2
3. VMC training details:
    --- For Z2 calculation, must make sure a proper chemical potential \mu is used in SU, so that the initial TNS has major amplitude weight
    in the target quantum number sector (e.g. particle number in our case). This is very important as we want the tensor values in the TN to
    mainly contribute to amplitude of configuration with fixed particle number, instead of other unrelavant configuration amplitudes. If we
    have large amplitude on those unrelavant configurations, the initial expressvity of our ansatz will be weakened and the optimization becomes
    harder.
    --- For NN initialization, simply initialize the weights/bias with very small random numbers around zero. E.g. uniform distribution r.v. in
    [-0.005, 0.005].
    --- SR iterative solver: use 'minres' instead of 'cg'. 'cg' works for positive definite symmetric/hermitian matrix, while 'minres' works for
    symmetric/hermitian matrix. In our calculation, the quantum geometric matrix is obtained via MC sampling, thus may not be necessarily positive
    definite due to statistical errors (theoretically it is PSD). Therefore to stabilize the SR linear equation solution, one should choose 'minres'
    over 'cg'. And note it is sufficient to set the relative tolerance in these iterative solver to 1e-4 instead of the default 1e-5 in our VMC calculation.
    If the rtol is set to 1e-5, the SR solver may take a long time to converge (potentially due to the MPI communication issue on Caltech HPC).
    --- hpc-21-37,hpc-35-[01,14,23,30-31] One of the nodes is definitely problematic. hpc-21-37,hpc-34-10,hpc-52-18,hpc-53-14,hpc-54-12 is good. hpc-35-[02,30],hpc-52-[28-30] is bad.
    hpc-35-30 is bad??

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






