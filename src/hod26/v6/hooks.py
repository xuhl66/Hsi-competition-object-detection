"""V6 successful-step controls, EMA and checkpoint lineage metadata."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from mmcv.runner.hooks import HOOKS, Hook
from mmdet.core.hook import ExpMomentumEMAHook

from .optim import optimizer_step_range


def _set_updates_recursive(obj, updates: int, seen: set[int]) -> int:
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    changed = 0
    setter = getattr(obj, "set_successful_updates", None)
    if callable(setter):
        setter(updates)
        changed += 1
    epoch_setter = getattr(obj, "set_epoch", None)
    if callable(epoch_setter) and not callable(setter):
        # V6's Fold0 data boundaries are exact multiples of 1,200 updates.
        epoch_setter(updates // 1200)
        changed += 1
    for attribute in ("dataset", "datasets", "pipeline", "transforms"):
        child = getattr(obj, attribute, None)
        if child is None:
            continue
        if isinstance(child, (list, tuple)):
            for item in child:
                changed += _set_updates_recursive(item, updates, seen)
        else:
            changed += _set_updates_recursive(child, updates, seen)
    return changed


@HOOKS.register_module()
class HOD26V6ProgressHook(Hook):
    """Drive fusion/data stages from successful optimizer steps."""

    def before_train_epoch(self, runner) -> None:
        _, successful, _ = optimizer_step_range(runner.optimizer)
        changed = _set_updates_recursive(runner.data_loader.dataset, successful, set())
        if runner.rank == 0 and changed == 0:
            raise RuntimeError("V6 staged augmentation was not found")

    def before_train_iter(self, runner) -> None:
        _, successful, _ = optimizer_step_range(runner.optimizer)
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        # Open by one prospective update so every new parameter receives a
        # gradient on update zero; an AMP skip repeats the same coordinate.
        model.set_v6_successful_updates(successful + 1)
        if runner.rank == 0 and runner.iter % 100 == 0:
            runner.log_buffer.update(
                {
                    "v6_successful_before_iter": float(successful),
                    "v6_fusion_progress": float(model.neck.fusion_progress.item()),
                },
                1,
            )


@HOOKS.register_module()
class HOD26V6EMAHook(ExpMomentumEMAHook):
    """Fresh-lineage EMA updated only after a successful AdamW step."""

    def __init__(
        self,
        *args,
        total_iter: int = 2000,
        control_suffixes=("fusion_progress",),
        **kwargs,
    ) -> None:
        self.total_iter = int(total_iter)
        if self.total_iter <= 0:
            raise ValueError("V6 EMA total_iter must be positive")
        self.control_suffixes = tuple(str(value) for value in control_suffixes)
        super().__init__(*args, total_iter=self.total_iter, **kwargs)
        self._last_successful = 0

    def before_run(self, runner) -> None:
        super().before_run(runner)
        _, self._last_successful, _ = optimizer_step_range(runner.optimizer)
        runner.meta = runner.meta or {}
        prior = int(runner.meta.get("hod26_v6_successful_updates", self._last_successful))
        if prior != self._last_successful:
            raise RuntimeError(
                "V6 resume metadata and AdamW successful-step state disagree"
            )

    def after_train_iter(self, runner) -> None:
        # The BELOW_NORMAL dispatcher runs after MMCV's FP16 optimizer hook.
        return None

    def update_after_optimizer(self, runner) -> None:
        minimum, current, state_count = optimizer_step_range(runner.optimizer)
        if current < self._last_successful or current - self._last_successful not in (0, 1):
            raise RuntimeError("V6 optimizer successful-step continuity failed")
        delta = current - self._last_successful
        attempted = int(runner.iter) + 1
        skipped = attempted - current
        if skipped < 0:
            raise RuntimeError("V6 successful steps exceed runner attempts")
        if delta:
            momentum = (1 - self.momentum) * math.exp(
                -(1 + current) / self.total_iter
            ) + self.momentum
            for name, parameter in self.model_parameters.items():
                ema = self.model_buffers[self.param_ema_buffer[name]]
                if any(name.endswith(suffix) for suffix in self.control_suffixes):
                    ema.copy_(parameter.data)
                elif parameter.dtype.is_floating_point:
                    ema.mul_(1 - momentum).add_(parameter.data, alpha=momentum)
        else:
            for name, parameter in self.model_parameters.items():
                if any(name.endswith(suffix) for suffix in self.control_suffixes):
                    self.model_buffers[self.param_ema_buffer[name]].copy_(parameter.data)
        self._last_successful = current
        runner.meta.update(
            hod26_version="v6",
            hod26_v6_independent_lineage=True,
            hod26_v6_successful_updates=current,
            hod26_v6_amp_skipped_updates=skipped,
            hod26_v6_optimizer_min_step=minimum,
            hod26_v6_optimizer_max_step=current,
            hod26_v6_optimizer_state_count=state_count,
            hod26_v6_ema_age=current,
            hod26_v6_ema_update_order="post_successful_optimizer_step",
            hod26_v6_ema_control_buffers="copied_not_smoothed",
        )
        if runner.rank == 0 and attempted % 100 == 0:
            runner.log_buffer.update(
                {
                    "v6_successful_updates": float(current),
                    "v6_amp_skipped_updates": float(skipped),
                    "v6_optimizer_min_step": float(minimum),
                },
                1,
            )


@HOOKS.register_module()
class HOD26V6PostOptimizerEMAHook(Hook):
    def before_run(self, runner) -> None:
        hooks = [hook for hook in runner.hooks if isinstance(hook, HOD26V6EMAHook)]
        if len(hooks) != 1:
            raise RuntimeError("V6 requires exactly one EMA owner")
        self.ema_hook = hooks[0]

    def after_train_iter(self, runner) -> None:
        self.ema_hook.update_after_optimizer(runner)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@HOOKS.register_module()
class HOD26V6CheckpointContractHook(Hook):
    """Embed source/config/optimizer ordering into every V6 checkpoint."""

    def __init__(self, source_lock: str, initialization_provenance: str) -> None:
        self.source_lock = str(source_lock)
        self.initialization_provenance = str(initialization_provenance)

    def before_run(self, runner) -> None:
        source = Path(self.source_lock)
        provenance = Path(self.initialization_provenance)
        if not source.is_file() or not provenance.is_file():
            raise RuntimeError("V6 source lock or initialization provenance is missing")
        self.source_sha = _sha256(source)
        self.provenance_sha = _sha256(provenance)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("version") != "v6":
            raise RuntimeError("V6 runtime source lock schema is invalid")
        runner.meta = runner.meta or {}
        previous = runner.meta.get("hod26_v6_runtime_source_lock_sha256")
        if previous not in (None, self.source_sha):
            raise RuntimeError("V6 runner resumed across a changed source lock")
        self.before_save_checkpoint(runner, {"meta": runner.meta})

    def before_save_checkpoint(self, runner, checkpoint) -> None:
        minimum, maximum, state_count = optimizer_step_range(runner.optimizer)
        groups = []
        for group in runner.optimizer.param_groups:
            names = group.get("param_names")
            if not isinstance(names, list) or len(names) != len(group["params"]):
                raise RuntimeError("V6 optimizer group lost its explicit parameter names")
            groups.append(
                {
                    "group_name": group["group_name"],
                    "param_names": names,
                    "lr_scale": group["lr_scale"],
                    "initial_lr": group["initial_lr"],
                    "weight_decay": group["weight_decay"],
                    "betas": list(group["betas"]),
                    "eps": group["eps"],
                }
            )
        checkpoint["meta"].update(
            hod26_version="v6",
            hod26_v6_public_initialization_only=True,
            hod26_v6_runtime_source_lock=self.source_lock,
            hod26_v6_runtime_source_lock_sha256=self.source_sha,
            hod26_v6_initialization_provenance=self.initialization_provenance,
            hod26_v6_initialization_provenance_sha256=self.provenance_sha,
            hod26_v6_optimizer_min_step=minimum,
            hod26_v6_optimizer_max_step=maximum,
            hod26_v6_optimizer_state_count=state_count,
            hod26_v6_optimizer_groups=groups,
        )
        if os.environ.get("V6_FULL_END_UPDATE"):
            checkpoint["meta"].update(
                hod26_v6_full_data_stage=True,
                hod26_v6_full_start_update=int(os.environ["V6_FULL_START_UPDATE"]),
                hod26_v6_full_target_update=int(os.environ["V6_FULL_END_UPDATE"]),
                hod26_v6_full_start_lr_ratio=float(os.environ["V6_FULL_START_RATIO"]),
            )
