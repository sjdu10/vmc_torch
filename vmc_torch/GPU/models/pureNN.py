"""Pure neural-network wavefunction model for GPU VMC.

Architecture:
    x: (N_sites,) int64
    -> embedding lookup    (N_sites, embed_dim)
    -> flatten             (N_sites * embed_dim,)
    -> [Linear -> Tanh] x n_layers
    -> Linear              scalar real

Parameter layout in self.params (ParameterList indices):
    [0]        emb_w  (phys_dim, embed_dim)
    [1+2k]     w_k    (hidden_dim, in_dim)   k=0..n_layers-1
    [2+2k]     b_k    (hidden_dim,)
    [-2]       out_w  (1, hidden_dim)
    [-1]       out_b  (1,)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import WavefunctionModel_GPU


class PureNN_GPU(WavefunctionModel_GPU):
    """Pure neural-network wavefunction for GPU VMC benchmarking.

    Drop-in replacement for fPEPS_Model_GPU — inherits the same
    interface from WavefunctionModel_GPU.
    """

    def __init__(
        self,
        n_sites,
        phys_dim=4,
        embed_dim=16,
        hidden_dim=256,
        n_layers=2,
        dtype=torch.float64,
    ):
        self.n_sites = n_sites
        self.phys_dim = phys_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dtype = dtype

        all_tensors = []

        # Embedding: (phys_dim, embed_dim)
        emb_w = torch.randn(phys_dim, embed_dim, dtype=dtype) * 0.3
        all_tensors.append(emb_w)

        # Hidden layers
        in_dim = n_sites * embed_dim
        for _ in range(n_layers):
            w = torch.empty(hidden_dim, in_dim, dtype=dtype)
            nn.init.kaiming_uniform_(
                w, a=0, mode='fan_in', nonlinearity='tanh',
            )
            b = torch.zeros(hidden_dim, dtype=dtype)
            all_tensors.append(w)
            all_tensors.append(b)
            in_dim = hidden_dim

        # Output layer: (1, hidden_dim) -> scalar per sample
        out_w = torch.empty(1, hidden_dim, dtype=dtype)
        nn.init.kaiming_uniform_(
            out_w, a=0, mode='fan_in', nonlinearity='linear',
        )
        out_b = torch.zeros(1, dtype=dtype)
        all_tensors.append(out_w)
        all_tensors.append(out_b)

        super().__init__(params_list=all_tensors)

    def amplitude(self, x, params_list):
        """Single-sample NN forward pass.

        Args:
            x:           (N_sites,) int64 — one configuration
            params_list: list of parameter tensors

        Returns:
            scalar real amplitude
        """
        idx = 0
        emb_w = params_list[idx]
        idx += 1

        # F.embedding has explicit vmap/grad support;
        # plain emb_w[x] does not propagate grads through vmap.
        # x is (N_sites,) -> embedding is (N_sites, embed_dim)
        # -> flatten to (N_sites * embed_dim,)
        h = F.embedding(x, emb_w).reshape(-1)

        for _ in range(self.n_layers):
            w = params_list[idx]
            idx += 1
            b = params_list[idx]
            idx += 1
            h = torch.tanh(h @ w.T + b)

        out_w = params_list[idx]
        idx += 1
        out_b = params_list[idx]
        return (h @ out_w.T + out_b).squeeze(-1)
