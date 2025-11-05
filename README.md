# vmc_torch

A quantum variational Monte Carlo (VMC) code based on pyTorch for solving ground-state properties of quantum many-body lattice Hamiltonians.

`vmc_torch` currently supports a wide range of variational ansatz, including bosonic/fermionic tensor network states (TNS) (e.g. MPS, PEPS) defined using `quimb` and `symmray`, and neural quantum states (NQS).

## Installation

**Installing the latest version directly from github:**

```bash
pip install -U git+https://github.com/sjdu10/vmc_torch.git
```

**Installing a local, editable development version:**

```bash
git clone https://github.com/sjdu10/vmc_torch.git
pip install --no-deps -U -e vmc_torch/
```

## Usage

**Fermionic neural network quantum state example:**

In `/examples`, there's an example script for running VMC for ground state of a `4x2` Fermi-Hubbard model (OBC) on square lattice at half-filling using Slater Determinant.

To run the code: 
```bash
cd ./examples
mpirun -np 10 python vmc_run_example.py
```
Feel free to substitute the number `10` with any number of MPI ranks you want to use.

An example plot of the VMC training curves for various Ans\"atze on a spinless Hubbard model ($N_s$ is the VMC sample size): 

(we also compare our neural backflow model updates with the Netket updates)
![VMC_energy](./docs/pics/VMC_energy.png)


## References
- "NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems" - *Filippo Vicentini and Damian Hofmann and Attila Szabó and Dian Wu and Christopher Roth and Clemens Giuliani and Gabriel Pescia and Jannes Nys and Vladimir Vargas-Calderón and Nikita Astrakhantsev and Giuseppe Carleo* - https://scipost.org/10.21468/SciPostPhysCodeb.7

- `symmray` - *Johnnie Gray* - https://github.com/jcmgray/symmray

- `quimb` - *Johnnie Gray* - https://github.com/jcmgray/quimb

- "Neuralized Fermionic Tensor Networks for Quantum Many-Body Systems" - *Si-Jing Du*, *Ao Chen* and *Garnet Kin-Lic Chan* - [arXiv: ](https://doi.org/10.48550/arXiv.2506.08329)