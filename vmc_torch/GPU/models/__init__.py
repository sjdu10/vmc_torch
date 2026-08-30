"""GPU wavefunction models for VMC.

All models share the same interface from WavefunctionModel_GPU:
  - amplitude(x, params_list): single-sample, (N_sites,) -> scalar
  - forward(x): batched via vmap, with compiled/exported dispatch
  - vamp(x, params): batched amplitude for torch.func.grad
"""
from ._base import WavefunctionModel_GPU
from .pureTNS import (
    fPEPS_Model_GPU,
    fPEPS_Model_HOTRG_GPU,
    fPEPS_Model_reuse_GPU,
    strip_phys_linearmap,
)
from .pureTNS_spin import (
    PEPS_Model_GPU,
    PEPS_Model_reuse_GPU,
    PEPS_Model_reuse_compiled_cache_GPU,
)
from .pureNN import PureNN_GPU
from .slater import SlaterDeterminant_GPU
from .NNBF import NNBF_GPU, AttentionNNBF_GPU
from .NNfTNS import (
    NNfTNS_Model_GPU,
    NNfTNS_Reuse_Model_GPU,
    Conv2D_Geometric_fPEPS_GPU,
    Conv2D_Geometric_fPEPS_GPU_Deep,
    Conv2D_Uniform_fPEPS_GPU_Deep,
    Conv2D_Uniform_fPEPS_GPU_Deep_PBC,
    Conv2D_Uniform_fPEPS_GPU_Deep_PBC_Direct,
    CNN_fPEPS_Ao,
    ViT_Geometric_fPEPS_GPU,
    Attention_Geometric_fPEPS_GPU,
    Attention_Uniform_fPEPS_GPU_Deep_PBC_Direct,
    LocalSite_fPEPS_GPU,
    LocalCluster_fPEPS_GPU_original,
    LocalCluster_fPEPS_GPU,
    LocalCluster_fPEPS_Reuse_GPU,
)
from .LoRA_models import (
    LoRA_fPEPS_GPU,
    LoRA_LocalCluster_fPEPS_GPU,
)
from .symmetry import (
    SymmetryProjectedModel,
    FermionSymmetryProjectedModel,
)

__all__ = [
    'strip_phys_linearmap',
    "WavefunctionModel_GPU",
    "fPEPS_Model_GPU",
    "fPEPS_Model_HOTRG_GPU",
    "fPEPS_Model_reuse_GPU",
    "PureNN_GPU",
    "SlaterDeterminant_GPU",
    "NNBF_GPU",
    "AttentionNNBF_GPU",
    "NNfTNS_Model_GPU",
    "NNfTNS_Reuse_Model_GPU",
    "Conv2D_Geometric_fPEPS_GPU",
    "Conv2D_Geometric_fPEPS_GPU_Deep",
    "Conv2D_Uniform_fPEPS_GPU_Deep",
    "Conv2D_Uniform_fPEPS_GPU_Deep_PBC",
    "Conv2D_Uniform_fPEPS_GPU_Deep_PBC_Direct",
    "CNN_fPEPS_Ao",
    "ViT_Geometric_fPEPS_GPU",
    "Attention_Geometric_fPEPS_GPU",
    "Attention_Uniform_fPEPS_GPU_Deep_PBC_Direct",
    "LocalSite_fPEPS_GPU",
    "LocalCluster_fPEPS_GPU_original",
    "LocalCluster_fPEPS_GPU",
    "LocalCluster_fPEPS_Reuse_GPU",
    "LoRA_fPEPS_GPU",
    "LoRA_LocalCluster_fPEPS_GPU",
    "PEPS_Model_GPU",
    "PEPS_Model_reuse_GPU",
    "PEPS_Model_reuse_compiled_cache_GPU",
    "SymmetryProjectedModel",
    "FermionSymmetryProjectedModel",
]
