"""Registration entry point for the clean HOD26 V5R lineage."""

from hod26.v4.compat import install_mmcv_torch21_compatibility

install_mmcv_torch21_compatibility()

# Import V5 first: V5R deliberately reuses the audited spectral, salience and
# data implementations while replacing the faulty localization/EMA lineage.
import hod26.v5 as v5  # noqa: E402,F401

from . import hooks as hooks  # noqa: E402,F401
from . import model as model  # noqa: E402,F401
from . import optim as optim  # noqa: E402,F401

__all__: list[str] = []
