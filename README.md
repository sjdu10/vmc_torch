# vmc_torch

A quantum variational Monte Carlo (VMC) framework for general pytorch models. Currently is used for tensor network state VMC (bosonic and fermionic) with TNS (bosonic/fermionic) defined using Quimb and Symmray.

## Install

**Installing the latest version directly from github:**

```bash
pip install -U git+https://github.com/sjdu10/vmc_torch_experiment.git
```

**Installing a local, editable development version:**

```bash
git clone https://github.com/sjdu10/vmc_torch_experiment.git
pip install --no-deps -U -e vmc_torch_experiment/
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

**IMPORTANT:** *When runing MCMC sampling using multiple chains, one should either use **only a few long chains**, or **a large number of short chains**, so that the total samples are fully converged to the target distribution. It may cause problems and generate unreliable results if one uses a small number of short chains, as each chain does not have enough time to tour over the target distribution, plus the number of chains is not large enough to distribute these chains dispersively enough over the target distribution.*

One can generate VMC results like this ($N_s$ is the VMC sample size): 

(we also compare our neural backflow model updates with the Netket updates)
![VMC_energy](./docs/pics/VMC_energy.png)


## References
- "NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems" - *Filippo Vicentini and Damian Hofmann and Attila Szabó and Dian Wu and Christopher Roth and Clemens Giuliani and Gabriel Pescia and Jannes Nys and Vladimir Vargas-Calderón and Nikita Astrakhantsev and Giuseppe Carleo* - https://scipost.org/10.21468/SciPostPhysCodeb.7

- "symmray" - *Johnnie Gray* - https://github.com/jcmgray/symmray