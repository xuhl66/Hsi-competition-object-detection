from __future__ import annotations

import sys
from types import ModuleType


def install_deim_compatibility() -> None:
    """
    DEIM imports calflops eagerly even when profiling is unused. Its optional
    transformers dependency can pull binary-incompatible NLP packages into a
    vision-only environment, so expose the one symbol DEIM imports and fail
    explicitly only if somebody requests the unused FLOP profiler.
    """
    module = ModuleType("calflops")

    def calculate_flops(*args, **kwargs):
        raise RuntimeError(
            "The optional calflops profiler is disabled in HOD26 V2; "
            "it assumes a 3-channel square input and is invalid for this "
            "16-band rectangular model."
        )

    module.calculate_flops = calculate_flops
    sys.modules["calflops"] = module

