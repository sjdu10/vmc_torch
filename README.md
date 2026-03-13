# ⚛️ vmc_torch: Flexible Variational Monte Carlo with PyTorch

## Overview

`vmc_torch` is a **scalable implementation of Variational Monte Carlo (VMC)** designed for studying the **ground-state properties** of quantum many-body lattice Hamiltonians with **flexible** variational wavefunction *Ansätze*.

Built upon **PyTorch**, this library offers great flexibility in choice of variational *Ansätze*, including Neural Quantum States (NQS) and Tensor Network States (TNS), and hybrid TN-NN models such as NN-fTNS [1].

### Key Features and Capabilities

* **Diverse Variational Ansätze Support:** Supports a wide spectrum of modern wavefunctions:
    * **Neural Quantum States (NQS):** Leveraging PyTorch's native capabilities for large-scale, trainable neural network models.
    * **Tensor Network States (TNS):** Integrates with TN libraries (`quimb`, `symmray`) for handling Matrix Product States (MPS), Projected Entangled Pair States (PEPS), and more, bosonic or fermionic.
    * **Neuralized fermionic TNS (NN-fTNS):** Hybrid TN-NN model that improves over both fermionic TNS and NQS, see [1] for details.
    * **Tensor Network Functions (TNF):** Function defined by tensor networks with arbitrary geometry. E.g. TNF derived from (1+1)D quantum circuit that supports volume-law entanglement structure, see [2] for details.
* **Massively Parallel VMC Sampling:** Utilizes **`mpi4py`** to distribute the Markov Chain Monte Carlo (MCMC) sampling process, suitable on high-performance computing (HPC) clusters.
* **HPC Ready:** Built for large-scale VMC calculations, suitable for deployment across **thousands of CPU cores**. **Note:** *Current parallization is optimized for CPU via MPI. GPU acceleration is actively under development.*
* **Auto-differentiation for Optimization:** Leverages PyTorch's automatic differentiation for efficient energy optimization with techniques like Stochastic Reconfiguration (SR) and other ML optimizers.

## 🚀 Installation

**Installing the latest version directly from github:**

```bash
pip install -U git+https://github.com/sjdu10/vmc_torch.git
```

**Installing a local, editable development version:**

```bash
git clone https://github.com/sjdu10/vmc_torch.git
pip install --no-deps -U -e vmc_torch/
```

## 📖 Usage Example

**Fermionic neural network quantum state example:**

In `/examples`, there's an example script for running VMC for ground state of a `4x2` Fermi-Hubbard model (OBC) on square lattice at half-filling using Slater Determinant.

To run the code: 
```bash
cd ./examples
mpirun -np 10 python vmc_run_example.py
```
Feel free to substitute the number `10` with any number of MPI ranks you want to use.

An example plot (from Ref.[1]) of the VMC training curves for various Ans\"atze on a Fermi-Hubbard model: 

![VMC training curves from Ref.[1]](./docs/pics/vmc_curves.png)


## 📚 References

### Research Publication

This code is the result of the research detailed in:

[1] **"Neuralized Fermionic Tensor Networks for Quantum Many-Body Systems"** - Si-Jing Du, Ao Chen, and Garnet Kin-Lic Chan - [Phys. Rev. B 113, 085134](https://doi.org/10.1103/x8vl-qf14)

[2] **Tensor Network Computations That Capture Strict Variationality, Volume Law Behavior, and the Efficient Representation of Neural Network States** - Wen-Yuan Liu*, Si-Jing Du*, Ruojing Peng, Johnnie Gray and Garnet Kin-Lic Chan - [Phys. Rev. Lett. 133, 260404](https://doi.org/10.1103/PhysRevLett.133.260404)



### Core Dependencies

`vmc_torch` builds on and interoperates with leading libraries in Tensor Networks and quantum many-body calculations:

[3] `symmray` - *Johnnie Gray* - https://github.com/jcmgray/symmray

[4] `quimb` - *Johnnie Gray* - https://github.com/jcmgray/quimb

