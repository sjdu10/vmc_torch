# ==============================================================================
from .models.pureTNS import *
# ==============================================================================
from .models.TransformerNNfTNS import *
# ==============================================================================
from .models.LoRA_models import *
# ==============================================================================
from .models.old_models import *
# ==============================================================================

import warnings
warnings.warn(
    "The vmap_models module is deprecated and will be removed in future versions. " \
    "Please use the updated models in .models directory instead.",
    category=FutureWarning,
    stacklevel=2,
)