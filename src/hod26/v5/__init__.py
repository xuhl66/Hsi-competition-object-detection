"""Registration entry point for the HOD26 V5 stack."""

from hod26.v4.compat import install_mmcv_torch21_compatibility

install_mmcv_torch21_compatibility()

from . import data as data  # noqa: E402,F401
from . import hooks as hooks  # noqa: E402,F401
from . import model as model  # noqa: E402,F401
from . import optim as optim  # noqa: E402,F401

__all__: list[str] = []
