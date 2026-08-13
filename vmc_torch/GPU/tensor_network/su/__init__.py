"""Simple-update (SU) utilities and driver scripts.

Moved here from ``GPU/SU_utils/``.  ``SU_func.py`` is a function
library; ``SU_small.py``, ``SU_small_PBC.py`` and
``SU_pinning_field.py`` are SCRIPTS -- their system parameters and the
SU run itself are at module level, with no ``if __name__ ==
'__main__'`` guard.

Run them from inside this directory::

    cd vmc_torch/GPU/tensor_network/su && python SU_small.py

They use a bare ``from SU_func import ...``, which resolves only when
this directory is on ``sys.path`` (i.e. exactly the above).  That was
deliberately left as-is during the move: converting to a relative
import would make direct script execution fail outright.

This ``__init__.py`` imports nothing on purpose -- importing the three
scripts as modules would run a full SU calculation as a side effect.
"""
