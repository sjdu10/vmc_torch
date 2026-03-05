"""Slater determinant wavefunction model for GPU VMC.

    psi(x) = det( M[occupied, :] )

For spinful fermions with quimb encoding {0=empty, 1=↓, 2=↑, 3=↑↓}:
    1. Convert x to binary occupation: n = [spin_up | spin_dn]
    2. occupied = argsort(n, descending)[:N_f]  (vmap-friendly)
    3. psi = det(M[occupied])
"""
import torch

from ._base import WavefunctionModel_GPU


class SlaterDeterminant_GPU(WavefunctionModel_GPU):
    """Slater determinant: psi(x) = det(M[occupied, :]).

    For spinful fermions on N_sites spatial sites:
        n_orbitals = 2 * N_sites  (spin-up + spin-down orbitals)
        n_fermions = N_f          (total fermion count)
        M has shape (2*N_sites, N_f)

    Args:
        n_orbitals: total number of orbitals (2 * N_sites for spinful)
        n_fermions: number of fermions
        dtype: parameter dtype (default float64)
    """

    def __init__(
        self, n_orbitals, n_fermions, dtype=torch.float64,
    ):
        M = torch.randn(n_orbitals, n_fermions, dtype=dtype)
        super().__init__(params_list=[M])
        self.n_fermions = n_fermions

    def amplitude(self, x, params_list):
        """Single-sample Slater determinant evaluation.

        Converts quimb config x ∈ {0,1,2,3}^N_sites to binary
        occupation vector [spin_up | spin_dn] of length 2*N_sites,
        then selects occupied rows of M and computes det.

        Args:
            x:           (N_sites,) int64 — one configuration
            params_list: [M] where M is (2*N_sites, N_f)

        Returns:
            scalar determinant value
        """
        M = params_list[0]
        # quimb encoding → binary occupation
        # 0=empty, 1=↓ only, 2=↑ only, 3=↑↓
        spin_up = ((x == 2) | (x == 3)).to(M.dtype)  # (N_sites,)
        spin_dn = ((x == 1) | (x == 3)).to(M.dtype)  # (N_sites,)
        n = torch.cat([spin_up, spin_dn])  # (2*N_sites,)
        # argsort on binary vector: occupied orbitals (1s) sort first
        # This is vmap-compatible (unlike nonzero which has dynamic shape)
        Nf = self.n_fermions
        occupied = n.argsort(descending=True)[:Nf]  # (N_f,)
        A = M[occupied]  # (N_f, N_f)
        return torch.linalg.det(A)
