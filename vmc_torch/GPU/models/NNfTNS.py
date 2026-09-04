"""Neural-network fermionic tensor-network (NNfTNS) model interfaces.

Placeholders that keep the package's public API importable; these
neural-network-augmented fermionic TN ansatze are not available in this
build (instantiating them raises ``NotImplementedError``).
"""
from ._base import WavefunctionModel_GPU
from .pureTNS import fPEPS_Model_reuse_GPU

__all__ = [
    "NNfTNS_Model_GPU",
    "NNfTNS_Reuse_Model_GPU",
    "Conv2D_Geometric_fPEPS_GPU",
    "Conv2D_Geometric_fPEPS_GPU_Deep",
    "Conv2D_Attn_Geometric_fPEPS_GPU_Deep",
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
]

_UNAVAILABLE_MSG = "NNfTNS models are not available in this build."


class _NNfTNSStub:
    """Mixin: keep the class importable; block instantiation."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_UNAVAILABLE_MSG)


# Internal helpers referenced by LoRA_models.py (only called at runtime).
def _get_receptive_field_2d(*args, **kwargs):
    raise NotImplementedError(_UNAVAILABLE_MSG)


class _LocalSiteNetwork(_NNfTNSStub):
    pass


class NNfTNS_Model_GPU(_NNfTNSStub, WavefunctionModel_GPU):
    pass


class NNfTNS_Reuse_Model_GPU(_NNfTNSStub, fPEPS_Model_reuse_GPU):
    pass


class Conv2D_Geometric_fPEPS_GPU(NNfTNS_Model_GPU):
    pass


class Conv2D_Geometric_fPEPS_GPU_Deep(NNfTNS_Model_GPU):
    pass


class Conv2D_Attn_Geometric_fPEPS_GPU_Deep(NNfTNS_Model_GPU):
    pass


class Conv2D_Uniform_fPEPS_GPU_Deep(NNfTNS_Model_GPU):
    pass


class Conv2D_Uniform_fPEPS_GPU_Deep_PBC(NNfTNS_Model_GPU):
    pass


class Conv2D_Uniform_fPEPS_GPU_Deep_PBC_Direct(NNfTNS_Model_GPU):
    pass


class CNN_fPEPS_Ao(NNfTNS_Model_GPU):
    pass


class ViT_Geometric_fPEPS_GPU(NNfTNS_Model_GPU):
    pass


class Attention_Geometric_fPEPS_GPU(NNfTNS_Model_GPU):
    pass


class Attention_Uniform_fPEPS_GPU_Deep_PBC_Direct(NNfTNS_Model_GPU):
    pass


class LocalSite_fPEPS_GPU(NNfTNS_Model_GPU):
    pass


class LocalCluster_fPEPS_GPU_original(NNfTNS_Model_GPU):
    pass


class LocalCluster_fPEPS_GPU(NNfTNS_Model_GPU):
    pass


class LocalCluster_fPEPS_Reuse_GPU(NNfTNS_Reuse_Model_GPU):
    pass
