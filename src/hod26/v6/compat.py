"""Explicit compatibility fixes for the pinned MMCV-1/PyTorch-2 runtime."""

from __future__ import annotations

import torch


def install_runtime_compatibility() -> None:
    import mmcv.parallel._functions as scatter_functions

    current = scatter_functions._get_stream
    if not getattr(current, "_hod26_v6_compatible", False):

        def _get_stream(device):
            if isinstance(device, int):
                device = torch.device("cuda", device)
            return current(device)

        _get_stream._hod26_v6_compatible = True
        scatter_functions._get_stream = _get_stream

    from mmcv.parallel.distributed import MMDistributedDataParallel

    current_forward = MMDistributedDataParallel._run_ddp_forward
    if not getattr(current_forward, "_hod26_v6_compatible", False):

        def _run_ddp_forward(self, *inputs, **kwargs):
            if self.device_ids:
                inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
                inputs, kwargs = inputs[0], kwargs[0]
            with self._inside_ddp_forward():
                return self.module(*inputs, **kwargs)

        _run_ddp_forward._hod26_v6_compatible = True
        MMDistributedDataParallel._run_ddp_forward = _run_ddp_forward
