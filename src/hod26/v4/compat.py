"""Small, explicit compatibility fixes for the pinned Co-DETR environment."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def install_mmcv_torch21_compatibility() -> None:
    """Adapt MMCV 1.x scatter's integer device IDs to PyTorch 2.1."""
    import mmcv.parallel._functions as scatter_functions

    current = scatter_functions._get_stream
    if getattr(current, "_hod26_v4_compatible", False):
        return

    def _get_stream(device):
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return current(device)

    _get_stream._hod26_v4_compatible = True
    scatter_functions._get_stream = _get_stream

    from mmcv.parallel.distributed import MMDistributedDataParallel

    current_ddp_forward = MMDistributedDataParallel._run_ddp_forward
    if not getattr(current_ddp_forward, "_hod26_v4_compatible", False):

        def _run_ddp_forward(self, *inputs, **kwargs):
            if self.device_ids:
                inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
                inputs, kwargs = inputs[0], kwargs[0]
            with self._inside_ddp_forward():
                return self.module(*inputs, **kwargs)

        _run_ddp_forward._hod26_v4_compatible = True
        MMDistributedDataParallel._run_ddp_forward = _run_ddp_forward

    # PyTorch 2.x autocast evaluates the positive BCE term in float32 while
    # the legacy MMDetection QFL scratch tensor remains float16. Explicitly
    # cast only at the indexed write; the loss/reduction semantics are
    # otherwise byte-for-byte the upstream implementation.
    import mmdet.models.losses.gfocal_loss as gfocal
    from mmdet.models.losses.utils import weighted_loss

    if not getattr(gfocal.quality_focal_loss, "_hod26_v4_compatible", False):

        @weighted_loss
        def quality_focal_loss(pred, target, beta=2.0):
            label, score = target
            pred_sigmoid = pred.sigmoid()
            zerolabel = pred_sigmoid.new_zeros(pred.shape)
            loss = F.binary_cross_entropy_with_logits(
                pred, zerolabel, reduction="none"
            ) * pred_sigmoid.pow(beta)
            background = pred.size(1)
            positive = ((label >= 0) & (label < background)).nonzero().squeeze(1)
            positive_label = label[positive].long()
            scale = score[positive] - pred_sigmoid[positive, positive_label]
            positive_loss = F.binary_cross_entropy_with_logits(
                pred[positive, positive_label],
                score[positive],
                reduction="none",
            ) * scale.abs().pow(beta)
            loss[positive, positive_label] = positive_loss.to(loss.dtype)
            return loss.sum(dim=1)

        quality_focal_loss._hod26_v4_compatible = True
        gfocal.quality_focal_loss = quality_focal_loss
