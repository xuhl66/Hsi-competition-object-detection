"""Registration entry point for the HOD26 V4 Co-DINO stack."""

from .compat import install_mmcv_torch21_compatibility

install_mmcv_torch21_compatibility()

from . import data as data  # noqa: E402,F401
from . import hooks as hooks  # noqa: E402,F401
from . import model as model  # noqa: E402,F401
from . import optim as optim  # noqa: E402,F401
from .constants import HOD26_CLASSES  # noqa: E402

__all__ = ["HOD26_CLASSES"]
