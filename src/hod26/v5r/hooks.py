"""Post-optimizer, skip-aware EMA hooks for the clean V5R lineage."""

from __future__ import annotations

import math

import torch
from mmcv.runner.hooks import HOOKS, Hook
from mmdet.core.hook import ExpMomentumEMAHook


def _optimizer_max_step(optimizer) -> int:
    steps = []
    for state in optimizer.state.values():
        value = state.get("step")
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.numel() != 1 or not bool(torch.isfinite(value).all()):
                raise RuntimeError("Invalid AdamW step encountered by V5R EMA")
            value = value.item()
        number = float(value)
        rounded = round(number)
        if number < 0 or not math.isclose(number, rounded, abs_tol=1e-6):
            raise RuntimeError("Non-integral AdamW step encountered by V5R EMA")
        steps.append(int(rounded))
    if not steps:
        raise RuntimeError("V5R EMA requires inherited optimizer state")
    return max(steps)


@HOOKS.register_module()
class HOD26V5RInheritedEMAHook(ExpMomentumEMAHook):
    """Inherit V4 EMA, then update only after a successful optimizer step."""

    def __init__(
        self,
        inherited_updates: int,
        parent_successful_steps: int,
        total_iter: int = 2000,
        control_suffixes=("v5_transition",),
        **kwargs,
    ) -> None:
        self.inherited_updates = int(inherited_updates)
        self.parent_successful_steps = int(parent_successful_steps)
        self.control_suffixes = tuple(str(value) for value in control_suffixes)
        if self.inherited_updates < 0 or self.parent_successful_steps < 0:
            raise ValueError("Inherited EMA/optimizer ages must be non-negative")
        self.total_iter = int(total_iter)
        super().__init__(total_iter=total_iter, **kwargs)
        self._last_optimizer_step = None

    def before_run(self, runner) -> None:
        super().before_run(runner)
        self._last_optimizer_step = _optimizer_max_step(runner.optimizer)
        if self._last_optimizer_step < self.parent_successful_steps:
            raise RuntimeError("V5R optimizer step predates the declared V4 parent")

    def after_train_iter(self, runner) -> None:
        # The HIGHEST-priority EMA hook must register/swap state before
        # checkpoint/evaluation, but must not run before the ABOVE_NORMAL
        # FP16 optimizer hook.  The companion hook calls the method below.
        return None

    def update_after_optimizer(self, runner) -> None:
        current_step = _optimizer_max_step(runner.optimizer)
        if current_step < self._last_optimizer_step:
            raise RuntimeError("V5R optimizer step moved backwards")
        delta = current_step - self._last_optimizer_step
        if delta not in (0, 1):
            raise RuntimeError(f"Unexpected optimizer-step delta for V5R EMA: {delta}")

        local_successful = current_step - self.parent_successful_steps
        local_attempted = int(runner.iter) + 1
        skipped = local_attempted - local_successful
        if local_successful < 0 or skipped < 0:
            raise RuntimeError("Invalid V5R attempted/successful update accounting")

        if delta:
            momentum = (1 - self.momentum) * math.exp(
                -(1 + self.inherited_updates + local_successful) / self.total_iter
            ) + self.momentum
            for name, parameter in self.model_parameters.items():
                buffer = self.model_buffers[self.param_ema_buffer[name]]
                if any(name.endswith(suffix) for suffix in self.control_suffixes):
                    # Control coordinates are state, not learned statistics.
                    buffer.copy_(parameter.data)
                elif parameter.dtype.is_floating_point:
                    buffer.mul_(1 - momentum).add_(parameter.data, alpha=momentum)
        else:
            # Even on AMP overflow, validation must see the raw transition
            # coordinate rather than an obsolete smoothed copy.
            for name, parameter in self.model_parameters.items():
                if any(name.endswith(suffix) for suffix in self.control_suffixes):
                    self.model_buffers[self.param_ema_buffer[name]].copy_(
                        parameter.data
                    )

        self._last_optimizer_step = current_step
        runner.meta["hod26_successful_local_optimizer_steps"] = local_successful
        runner.meta["hod26_amp_skipped_local_steps"] = skipped
        runner.meta["hod26_ema_update_order"] = "post_successful_optimizer_step"
        runner.meta["hod26_ema_control_buffers"] = "copied_not_smoothed"
        if runner.rank == 0 and local_attempted % 100 == 0:
            runner.log_buffer.update(
                {
                    "v5r_successful_optimizer_steps": float(local_successful),
                    "v5r_amp_skipped_steps": float(skipped),
                },
                1,
            )


@HOOKS.register_module()
class HOD26V5RPostOptimizerEMAHook(Hook):
    """Dispatch the EMA update after MMCV's FP16 optimizer hook."""

    def before_run(self, runner) -> None:
        hooks = [
            hook for hook in runner.hooks if isinstance(hook, HOD26V5RInheritedEMAHook)
        ]
        if len(hooks) != 1:
            raise RuntimeError("V5R requires exactly one inherited EMA hook")
        self.ema_hook = hooks[0]

    def after_train_iter(self, runner) -> None:
        self.ema_hook.update_after_optimizer(runner)
