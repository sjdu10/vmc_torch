# vmc_torch

A quantum variational Monte Carlo (VMC) code based on pyTorch for solving ground-state properties of quantum many-body lattice Hamiltonians.

`vmc_torch` currently supports a wide range of variational ansatz, including bosonic/fermionic tensor network states (TNS) (e.g. MPS, PEPS) defined using `quimb` and `symmray`, and neural quantum states (NQS).

## Install

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

**Neural network state example:**

In `/examples`, there's an example script for running VMC of a spinless Hubbard model on a `4x4` square lattice with OBC using various neural network quantum states.
To run the code: 
```bash
cd ./examples
mpirun -np 10 python vmc_run_example.py
```
Feel free to substitute the number `10` with any number of MPI ranks you want to use.

One can generate VMC results like this ($N_s$ is the VMC sample size): 

(we also compare our neural backflow model updates with the Netket updates)
![VMC_energy](./docs/pics/VMC_energy.png)


## References
- "NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems" - *Filippo Vicentini and Damian Hofmann and Attila Szabó and Dian Wu and Christopher Roth and Clemens Giuliani and Gabriel Pescia and Jannes Nys and Vladimir Vargas-Calderón and Nikita Astrakhantsev and Giuseppe Carleo* - https://scipost.org/10.21468/SciPostPhysCodeb.7

- "symmray" - *Johnnie Gray* - https://github.com/jcmgray/symmray

- "Quimb" - *Johnnie Gray* - https://github.com/jcmgray/quimb