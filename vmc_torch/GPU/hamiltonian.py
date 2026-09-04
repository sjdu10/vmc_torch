"""Backward-compat shim -- re-exports :mod:`vmc_torch.hamiltonian_torch`.

The GPU code shares the Hamiltonian / Hilbert-space / lattice-graph
classes with the CPU code (``Hamiltonian``, ``Graph`` / ``SquareLattice``
/ ``Chain``, ``SpinfulFermion`` / ``SpinlessFermion`` / ``Spin``
Hilbert spaces, the ``*_torch`` Hubbard and Heisenberg Hamiltonians).
This module keeps ``from vmc_torch.GPU.hamiltonian import ...`` working;
new code should import from ``vmc_torch.hamiltonian_torch`` directly.
"""
from vmc_torch.hamiltonian_torch import *  # noqa: F403
