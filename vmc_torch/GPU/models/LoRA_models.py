"""LoRA-parametrized fTNS models for GPU VMC.

Two models
----------
LoRA_fPEPS_GPU
    Pure LoRA: fixed base TN + CP-factored learnable corrections.
    No NN dependency on configuration x. Trainable: LoRA factors only.
    Port of experiment/vmap/models/LoRA_models.py::LoRA_fPEPS_Model.

LoRA_LocalCluster_fPEPS_GPU
    Config-dependent LoRA: N independent per-site networks generate
    LoRA factors from the local occupation, which are CP-reconstructed
    into TN parameter deltas and added to a trainable base TN.
    Port of experiment/vmap/models/LoRA_models.py::LoRA_NNfPEPS_Model_Cluster.

CP correction
-------------
For a site tensor with shape (N_b, D1, ..., d), the correction is

    delta_T = CP(factor_0, factor_1, ...)

where factor_j has shape (N_b, lora_rank, leg_dim_j) and CP is
the Khatri-Rao product:

    delta_T[x, a, b, ...] = sum_r factor_0[x, r, a] * factor_1[x, r, b] * ...

implemented as a single torch.einsum.
"""
import torch

from vmc_torch.GPU.tensor_network import strip_phys_linearmap
import torch.nn as nn

from ._base import WavefunctionModel_GPU
from .NNfTNS import (
    NNfTNS_Model_GPU,
    _LocalSiteNetwork,
    _get_receptive_field_2d,
)


# ==============================================================================
# LoRA_fPEPS_GPU — pure LoRA, no NN
# ==============================================================================


class LoRA_fPEPS_GPU(WavefunctionModel_GPU):
    """fPEPS with CP-factored learnable corrections, no NN.

    The base TN tensors are fixed non-trainable buffers.  The learnable
    parameters are CP-factored LoRA corrections per site:

        T_i_corrected = T_i_base + CP(factor_i_0, factor_i_1, ...)

    where each factor_i_j has shape (N_b, lora_rank, leg_dim_j).

    This mirrors ``LoRA_fPEPS_Model`` from the CPU vmap pipeline.

    Args:
        tn: quimb fPEPS tensor network (provides the fixed base)
        max_bond: boundary contraction bond dimension (chi)
        lora_rank: CP decomposition rank (default 8)
        dtype: parameter dtype (default float64)
        contract_boundary_opts: options forwarded to boundary contraction
    """

    def __init__(
        self,
        tn,
        max_bond,
        lora_rank=8,
        dtype=torch.float64,
        contract_boundary_opts=None,
    ):
        import quimb as qu
        import quimb.tensor as qtn

        if contract_boundary_opts is None:
            contract_boundary_opts = {}

        tn = tn.copy()  # don't modify original TN
        # Handle linearmap (matches NNfTNS_Model_GPU convention)
        self._loc_basis_perm = strip_phys_linearmap(tn)

        self.dtype = dtype
        self.chi = max_bond
        self.lora_rank = lora_rank
        self.contract_boundary_opts = contract_boundary_opts

        # Extract TN structure (skeleton for reconstruction)
        params, skeleton = qtn.pack(tn)
        self._skeleton = skeleton

        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(
            params, get_ref=True,
        )
        self._ftn_params_pytree = ftn_params_pytree

        # ---------------------------------------------------------------
        # Per-site LoRA structure (computed once, stored as Python attrs)
        # ---------------------------------------------------------------
        self._site_n_legs = []       # int: # of factorized legs per site
        self._site_fshapes = []      # list[list[(N_b, rank, dim)]]: factor shapes
        self._site_einsum_eqs = []   # str: CP einsum equation per site

        all_lora_factors = []

        for tensor_data in ftn_params_flat:
            shape = tensor_data.shape  # (N_b, D1, ..., d)
            n_blocks = int(shape[0])
            leg_dims = shape[1:]
            n_legs = len(leg_dims)

            fshapes = [
                (n_blocks, lora_rank, int(d)) for d in leg_dims
            ]

            # CP einsum (non-batched): "xra, xrb, ... -> xab..."
            # x=block, r=rank (contracted), a,b,...=leg dims
            input_subs = [f'xr{chr(97 + j)}' for j in range(n_legs)]
            output_sub = 'x' + ''.join(chr(97 + j) for j in range(n_legs))
            eq = ','.join(input_subs) + '->' + output_sub

            self._site_n_legs.append(n_legs)
            self._site_fshapes.append(fshapes)
            self._site_einsum_eqs.append(eq)

            # Create LoRA factors with small noise so delta starts near zero
            for fs in fshapes:
                all_lora_factors.append(
                    torch.randn(*fs, dtype=dtype) * 0.01
                )

        # ---------------------------------------------------------------
        # Register base TN tensors as non-trainable buffers
        # ---------------------------------------------------------------
        self._fixed_tensor_names = []
        for i, tensor_data in enumerate(ftn_params_flat):
            name = f'_fixed_{i}'
            self._fixed_tensor_names.append(name)
            self.register_buffer(
                name,
                torch.as_tensor(tensor_data, dtype=dtype).detach(),
            )

        # Register LoRA factors as learnable parameters via base class
        super().__init__(params_list=all_lora_factors)

    @property
    def _fixed_tensors(self):
        return [getattr(self, n) for n in self._fixed_tensor_names]

    def amplitude(self, x, params_list):
        """Single-sample amplitude: CP-reconstruct TN, isel, contract.

        Args:
            x: (N_sites,) int64 — one configuration
            params_list: flat list of LoRA factor tensors

        Returns:
            scalar amplitude
        """
        import quimb as qu
        import quimb.tensor as qtn

        fixed = self._fixed_tensors
        reconstructed = []
        factor_cursor = 0

        for i, (n_legs, fshapes, eq) in enumerate(
            zip(self._site_n_legs, self._site_fshapes, self._site_einsum_eqs)
        ):
            factors = params_list[factor_cursor:factor_cursor + n_legs]
            factor_cursor += n_legs

            # CP: (N_b, rank, D1) x ... -> (N_b, D1, ...)
            delta_t = torch.einsum(eq, *factors)
            reconstructed.append(fixed[i] + delta_t)

        params_pytree = qu.utils.tree_unflatten(
            reconstructed, self._ftn_params_pytree,
        )
        tn = qtn.unpack(params_pytree, self._skeleton)

        if self._loc_basis_perm is not None:
            x = self._loc_basis_perm[x]

        amp_tn = tn.isel({
            tn.site_ind(site): x[i_s]
            for i_s, site in enumerate(tn.sites)
        })

        if self.chi > 0:
            amp_tn.contract_boundary_from_xmin_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[0, amp_tn.Lx // 2 - 1],
                **self.contract_boundary_opts,
            )
            amp_tn.contract_boundary_from_xmax_(
                max_bond=self.chi, cutoff=0.0,
                xrange=[amp_tn.Lx // 2, amp_tn.Lx - 1],
                **self.contract_boundary_opts,
            )

        return amp_tn.contract()


# ==============================================================================
# _LoRA_LocalCluster_Generator_GPU — backflow module
# ==============================================================================


class _LoRA_LocalCluster_Generator_GPU(nn.Module):
    """Per-site local cluster LoRA backflow generator.

    For each site i:
      1. Gather local neighborhood (Chebyshev radius r, OBC).
      2. Run an independent _LocalSiteNetwork to generate flat LoRA
         factors for site i.
      3. Reshape to per-leg factors and CP-reconstruct:
             delta_T_i = CP(factor_i_0, factor_i_1, ...)
      4. Flatten delta_T_i to (B, tensor_i_numel).

    Concatenate across all sites: output is (B, total_ftn_params).

    This is the GPU adaptation of
    experiment/vmap/models/LoRA_models.py::LoRA_LocalClusterGenerator.

    Args:
        tn: quimb fPEPS tensor network
        radius: Chebyshev receptive-field radius (OBC)
        lora_rank: CP decomposition rank
        embed_dim: per-site embedding dimension
        n_heads: number of attention heads per site
        hidden_dim: MLP hidden dimension per site
        dtype: parameter dtype
    """

    def __init__(
        self,
        tn,
        radius,
        lora_rank,
        embed_dim,
        n_heads,
        hidden_dim,
        dtype,
    ):
        import quimb as qu
        import quimb.tensor as qtn

        super().__init__()
        Lx, Ly = tn.Lx, tn.Ly
        n_sites = Lx * Ly
        phys_dim = tn.phys_dim()

        ftn_params, _ = qtn.pack(tn)
        ftn_params_flat, _ = qu.utils.tree_flatten(
            ftn_params, get_ref=True,
        )

        # ---------------------------------------------------------------
        # Per-site LoRA structure
        # ---------------------------------------------------------------
        self._site_fshapes = []      # factor shapes per site
        self._site_einsum_eqs = []   # batched CP einsum per site
        self._site_lora_dims = []    # total LoRA param count per site (NN output dim)

        for tensor_data in ftn_params_flat:
            shape = tensor_data.shape  # (N_b, D1, ..., d)
            n_blocks = int(shape[0])
            leg_dims = shape[1:]
            n_legs = len(leg_dims)

            fshapes = [(n_blocks, lora_rank, int(d)) for d in leg_dims]
            total_lora = sum(n_blocks * lora_rank * int(d) for d in leg_dims)

            # Batched CP einsum: "BSRa, BSRb, ... -> BSab..."
            # B=batch, S=block, R=rank (contracted), a,b,...=leg dims
            input_subs = [f'BSR{chr(97 + j)}' for j in range(n_legs)]
            output_sub = 'BS' + ''.join(chr(97 + j) for j in range(n_legs))
            eq = ','.join(input_subs) + '->' + output_sub

            self._site_fshapes.append(fshapes)
            self._site_einsum_eqs.append(eq)
            self._site_lora_dims.append(total_lora)

        # ---------------------------------------------------------------
        # Receptive field buffers
        # ---------------------------------------------------------------
        rf_dict = _get_receptive_field_2d(Lx, Ly, radius)
        for i in range(n_sites):
            self.register_buffer(
                f'_rf_{i}',
                torch.tensor(rf_dict[i], dtype=torch.long),
            )

        # ---------------------------------------------------------------
        # N independent site networks (one per lattice site)
        # ---------------------------------------------------------------
        self.site_networks = nn.ModuleList()
        for i in range(n_sites):
            n_neighbors = len(rf_dict[i])
            self.site_networks.append(_LocalSiteNetwork(
                n_neighbors=n_neighbors,
                num_classes=phys_dim,
                embed_dim=embed_dim,
                n_heads=n_heads,
                hidden_dim=hidden_dim,
                output_dim=self._site_lora_dims[i],
                dtype=dtype,
            ))

    def initialize_output_scale(self, scale):
        """Small-init output layer so model starts near pure fPEPS."""
        for net in self.site_networks:
            last = net.mlp[-1]
            nn.init.normal_(last.weight, mean=0.0, std=scale)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, x):
        """
        Args:
            x: (B, N_sites) int64

        Returns:
            (B, total_ftn_params) float — CP-reconstructed TN deltas
        """
        B = x.shape[0]
        outputs = []

        for i, net in enumerate(self.site_networks):
            rf_buf = getattr(self, f'_rf_{i}')
            x_local = x[:, rf_buf]          # (B, n_neighbors)
            site_flat = net(x_local)         # (B, total_site_lora_params)

            fshapes = self._site_fshapes[i]
            eq = self._site_einsum_eqs[i]

            # Split flat output into per-leg factors
            bfactors = []
            cursor = 0
            for shape in fshapes:
                # shape = (N_b, rank, dim)
                n_el = shape[0] * shape[1] * shape[2]
                bfactor = site_flat[:, cursor:cursor + n_el].view(B, *shape)
                bfactors.append(bfactor)
                cursor += n_el

            # CP: (B, N_b, rank, D1) x ... -> (B, N_b, D1, ...)
            delta_t = torch.einsum(eq, *bfactors)
            outputs.append(delta_t.reshape(B, -1))

        return torch.cat(outputs, dim=1)  # (B, total_ftn_params)


# ==============================================================================
# LoRA_LocalCluster_fPEPS_GPU — thin NNfTNS_Model_GPU subclass
# ==============================================================================


class LoRA_LocalCluster_fPEPS_GPU(NNfTNS_Model_GPU):
    """NN-fPEPS with local cluster LoRA backflow for GPU VMC.

    Config-dependent LoRA: N independent per-site attention networks
    generate LoRA correction factors from local occupation.  The factors
    are CP-reconstructed into TN parameter deltas and added to a
    trainable base TN:

        T_i_corrected = T_i_base + nn_eta * CP(NN_factors_i(x_local))

    Port of
    experiment/vmap/models/LoRA_models.py::LoRA_NNfPEPS_Model_Cluster.

    Args:
        tn: quimb fPEPS tensor network (initializes the trainable base)
        max_bond: boundary contraction bond dimension (chi)
        nn_eta: scale factor applied to NN backflow correction
        embed_dim: per-site token embedding dimension
        n_heads: number of self-attention heads per site
        hidden_dim: MLP hidden dimension per site
        lora_rank: CP decomposition rank (default 4)
        radius: Chebyshev receptive-field radius (default 1)
        init_scale: initial scale for output layer weights (default 1e-5)
        dtype: parameter dtype (default float64)
        contract_boundary_opts: options forwarded to boundary contraction
    """

    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        embed_dim,
        n_heads,
        hidden_dim,
        lora_rank=4,
        radius=1,
        init_scale=1e-5,
        dtype=torch.float64,
        contract_boundary_opts=None,
    ):
        # 1. Build LoRA local cluster backflow module
        nn_module = _LoRA_LocalCluster_Generator_GPU(
            tn=tn,
            radius=radius,
            lora_rank=lora_rank,
            embed_dim=embed_dim,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            dtype=dtype,
        )

        # 2. Small-init output layers so model starts near pure fPEPS
        nn_module.initialize_output_scale(init_scale)

        # 3. Delegate to NNfTNS base class
        super().__init__(
            tn=tn,
            nn_module=nn_module,
            max_bond=max_bond,
            nn_eta=nn_eta,
            dtype=dtype,
            contract_boundary_opts=contract_boundary_opts,
        )

        self.lora_rank = lora_rank
