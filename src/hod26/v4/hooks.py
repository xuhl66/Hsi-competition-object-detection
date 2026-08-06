"""Training hooks for staged data policy and V4 provenance."""

from __future__ import annotations

from mmcv.runner.hooks import HOOKS, Hook


def _set_epoch_recursive(obj, epoch: int, seen: set[int]) -> int:
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    updated = 0
    setter = getattr(obj, "set_epoch", None)
    if callable(setter):
        setter(epoch)
        updated += 1
    for attribute in ("dataset", "datasets", "pipeline", "transforms"):
        child = getattr(obj, attribute, None)
        if child is None:
            continue
        if isinstance(child, (list, tuple)):
            for item in child:
                updated += _set_epoch_recursive(item, epoch, seen)
        else:
            updated += _set_epoch_recursive(child, epoch, seen)
    return updated


@HOOKS.register_module()
class HOD26DataStageHook(Hook):
    """Propagate the runner epoch before non-persistent workers are spawned."""

    def before_train_epoch(self, runner) -> None:
        updated = _set_epoch_recursive(runner.data_loader.dataset, runner.epoch, set())
        if runner.rank == 0:
            runner.log_buffer.update({"hod26_data_stage_epoch": float(runner.epoch)}, 1)
            if runner.epoch == 0 and updated == 0:
                raise RuntimeError("HOD26StagedAugment was not found in the loader")
