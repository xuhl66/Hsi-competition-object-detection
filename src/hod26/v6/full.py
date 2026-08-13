"""All-3,000 V6 retraining controls kept outside the frozen Fold0 runtime."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from mmcv.runner.hooks import HOOKS, Hook

from .data import HOD26V6StagedAugment
from .model import effective_number_class_weights


# Counts from storage/v2/dataset/full.json.  The Full launch gate independently
# recomputes these values from the immutable COCO annotations before training.
FULL_CLASS_COUNTS = (
    447,
    493,
    1375,
    621,
    369,
    565,
    355,
    451,
    461,
    475,
    658,
    258,
    184,
    310,
    1095,
    977,
    272,
    700,
)

FULL_IMAGES = 3000
FULL_UPDATES_PER_EPOCH = 1500
FULL_SOFT_HORIZON_UPDATES = 120000
FULL_STAGE_UPDATES = (7500, 70500, 100500)
FULL_LR_POINTS = (
    (0, 0.02),
    (1875, 1.00),
    (7500, 1.00),
    (70500, 0.30),
    (100500, 0.06),
    (115000, 0.015),
    (120000, 0.004),
)
FULL_CANDIDATE_EPOCHS = (34, 40, 46, 52, 60)


def _set_exposure_updates(obj: Any, updates: int, seen: set[int]) -> int:
    """Set only V6 augmentation pipelines to a completed-exposure coordinate."""

    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    changed = 0
    if isinstance(obj, HOD26V6StagedAugment):
        obj.set_successful_updates(updates)
        changed += 1
    for attribute in ("dataset", "datasets", "pipeline", "transforms"):
        child = getattr(obj, attribute, None)
        if child is None:
            continue
        if isinstance(child, (list, tuple)):
            for item in child:
                changed += _set_exposure_updates(item, updates, seen)
        else:
            changed += _set_exposure_updates(child, updates, seen)
    return changed


def _model(runner):
    return runner.model.module if hasattr(runner.model, "module") else runner.model


@HOOKS.register_module()
class HOD26V6FullDataContractHook(Hook):
    """Enforce balanced Full data, class weights and exposure-stage timing.

    LR, fusion progress and EMA remain indexed by successful optimizer steps.
    Augmentation stages instead follow completed passes through all 3,000
    scenes, because worker-side transforms can only change safely at an epoch
    boundary and an AMP skip must not delay a whole augmentation epoch.
    """

    def __init__(
        self,
        *,
        expected_images: int = FULL_IMAGES,
        updates_per_epoch: int = FULL_UPDATES_PER_EPOCH,
        class_counts: Sequence[int] = FULL_CLASS_COUNTS,
        stage_updates: Sequence[int] = FULL_STAGE_UPDATES,
        soft_horizon_updates: int = FULL_SOFT_HORIZON_UPDATES,
        candidate_epochs: Sequence[int] = FULL_CANDIDATE_EPOCHS,
    ) -> None:
        self.expected_images = int(expected_images)
        self.updates_per_epoch = int(updates_per_epoch)
        self.class_counts = tuple(int(value) for value in class_counts)
        self.stage_updates = tuple(int(value) for value in stage_updates)
        self.soft_horizon_updates = int(soft_horizon_updates)
        self.candidate_epochs = tuple(int(value) for value in candidate_epochs)
        if self.expected_images <= 0 or self.updates_per_epoch <= 0:
            raise ValueError("V6 Full image/update counts must be positive")
        if len(self.class_counts) != 18 or min(self.class_counts) <= 0:
            raise ValueError("V6 Full requires 18 positive class counts")
        if self.stage_updates != FULL_STAGE_UPDATES:
            raise ValueError("V6 Full exposure-stage coordinates changed")
        if self.soft_horizon_updates != FULL_SOFT_HORIZON_UPDATES:
            raise ValueError("V6 Full soft horizon changed")
        if self.candidate_epochs != FULL_CANDIDATE_EPOCHS:
            raise ValueError("V6 Full predeclared checkpoint window changed")

    def _expected_weights(self, *, device: torch.device) -> torch.Tensor:
        return effective_number_class_weights(
            self.class_counts, beta=0.999, cap=1.5
        ).to(device=device)

    def _enforce_class_weights(self, runner) -> None:
        model = _model(runner)
        targets = (
            model.query_head.hod26_class_weights,
            model.bbox_head[0].hod26_class_weights,
        )
        expected = self._expected_weights(device=targets[0].device)
        with torch.no_grad():
            for target in targets:
                if target.shape != expected.shape:
                    raise RuntimeError("V6 Full class-weight buffer shape changed")
                target.copy_(expected.to(dtype=target.dtype, device=target.device))

    def _verify_class_weights(self, runner) -> None:
        model = _model(runner)
        expected = self._expected_weights(
            device=model.query_head.hod26_class_weights.device
        )
        for name, actual in (
            ("query", model.query_head.hod26_class_weights),
            ("proposal", model.bbox_head[0].hod26_class_weights),
        ):
            # MMCV's FP16 model wrapper may cast persistent floating buffers.
            # Exactness is therefore defined in the buffer's runtime dtype.
            reference = expected.to(dtype=actual.dtype, device=actual.device)
            if not torch.equal(actual, reference):
                maximum = float((actual.float() - reference.float()).abs().max())
                raise RuntimeError(
                    f"V6 Full {name} class weights are not exact; "
                    f"dtype={actual.dtype}, max_abs={maximum:.9g}, "
                    f"actual={actual.detach().cpu().tolist()}, "
                    f"expected={reference.detach().cpu().tolist()}"
                )

    def before_run(self, runner) -> None:
        # This hook runs after the EMA resume hook (HIGH vs HIGHEST), so the
        # assignment is correct for both a fresh public bootstrap and resume.
        self._enforce_class_weights(runner)
        runner.meta = runner.meta or {}
        # MMCV 1.x has no runner-wide ``before_save_checkpoint`` event.  Its
        # CheckpointHook serializes ``runner.meta`` directly, so seed the Full
        # contract here (the same pattern used by V6's source-lock hook).
        self._update_checkpoint_metadata(runner.meta)

    def before_train_epoch(self, runner) -> None:
        if len(runner.data_loader.dataset) != self.expected_images:
            raise RuntimeError(
                "V6 Full dataset size changed: "
                f"{len(runner.data_loader.dataset)} != {self.expected_images}"
            )
        # MMCV 1.x may restore load_from/resume buffers after custom
        # before_run hooks.  Reapply this immutable, non-optimizer loss buffer
        # at every epoch boundary, after the EMA hook and before any forward.
        self._enforce_class_weights(runner)
        self._verify_class_weights(runner)
        exposure_updates = int(runner.epoch) * self.updates_per_epoch
        changed = _set_exposure_updates(
            runner.data_loader.dataset, exposure_updates, set()
        )
        if runner.rank == 0 and changed == 0:
            raise RuntimeError("V6 Full staged augmentation was not found")
        if runner.rank == 0:
            runner.log_buffer.update(
                {"v6_full_exposure_updates": float(exposure_updates)}, 1
            )

    def _update_checkpoint_metadata(self, metadata: dict[str, Any]) -> None:
        metadata.update(
            hod26_v6_full_data_stage=True,
            hod26_v6_full_training_mode="public_init_all3000_complete_retrain",
            hod26_v6_data_scope="all_3000_official_train_images",
            hod26_v6_full_images=self.expected_images,
            hod26_v6_full_updates_per_epoch=self.updates_per_epoch,
            hod26_v6_full_soft_horizon_updates=self.soft_horizon_updates,
            hod26_v6_full_stage_updates=list(self.stage_updates),
            hod26_v6_full_candidate_epochs=list(self.candidate_epochs),
            hod26_v6_full_candidate_updates=[
                epoch * self.updates_per_epoch for epoch in self.candidate_epochs
            ],
            hod26_v6_full_class_counts=list(self.class_counts),
            hod26_v6_full_class_weight_scope="all_3000_annotations",
            hod26_v6_full_augmentation_coordinate="completed_dataset_passes",
            hod26_v6_full_validation_policy="none_all_labeled_images_in_training",
        )

    def before_save_checkpoint(self, runner, checkpoint) -> None:
        # Kept for runners which do dispatch this event; MMCV 1.x is covered
        # by the explicit ``runner.meta`` update in ``before_run`` above.
        self._update_checkpoint_metadata(checkpoint["meta"])


__all__ = [
    "FULL_CANDIDATE_EPOCHS",
    "FULL_CLASS_COUNTS",
    "FULL_IMAGES",
    "FULL_LR_POINTS",
    "FULL_SOFT_HORIZON_UPDATES",
    "FULL_STAGE_UPDATES",
    "FULL_UPDATES_PER_EPOCH",
    "HOD26V6FullDataContractHook",
]
