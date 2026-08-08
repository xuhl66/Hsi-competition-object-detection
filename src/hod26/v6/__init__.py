"""Registration entry point for the standalone CoSpec-DINO V6 stack."""

from .compat import install_runtime_compatibility

install_runtime_compatibility()

from . import data as data  # noqa: E402,F401
from . import hooks as hooks  # noqa: E402,F401
from . import model as model  # noqa: E402,F401
from . import optim as optim  # noqa: E402,F401
from .constants import HOD26_CLASSES  # noqa: E402

__all__ = ["HOD26_CLASSES"]
