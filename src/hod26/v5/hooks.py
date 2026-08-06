"""Resume-safe V5 transition hook."""

from __future__ import annotations

import math

from mmdet.core.hook import ExpMomentumEMAHook
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class HOD26V5InheritedEMAHook(ExpMomentumEMAHook):
    """Continue the V4 EMA age while V5 uses a new local update coordinate."""

    def __init__(self, inherited_updates: int, total_iter: int = 2000, **kwargs):
        self.inherited_updates = int(inherited_updates)
        if self.inherited_updates < 0:
            raise ValueError("inherited_updates must be non-negative")
        super().__init__(total_iter=total_iter, **kwargs)
        self.momentum_fun = lambda x: (
            (1 - self.momentum)
            * math.exp(-(1 + self.inherited_updates + int(x)) / int(total_iter))
            + self.momentum
        )


@HOOKS.register_module()
class HOD26V5ProgressHook(Hook):
    def before_train_iter(self, runner) -> None:
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        model.set_v5_progress(runner.iter)
        if runner.rank == 0 and runner.iter % 100 == 0:
            runner.log_buffer.update({"v5_absolute_update": float(runner.iter)}, 1)
