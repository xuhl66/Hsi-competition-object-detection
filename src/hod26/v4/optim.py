"""Layer-decayed optimizer and absolute-update cosine schedule for V4."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.lr_updater import LrUpdaterHook
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS
from mmcv.runner.optimizer.default_constructor import DefaultOptimizerConstructor


def piecewise_cosine_ratio(update: int, points: Sequence[Sequence[float]]) -> float:
    normalized = tuple((int(x), float(y)) for x, y in points)
    if len(normalized) < 2:
        raise ValueError("At least two LR points are required")
    if any(x < 0 or y <= 0 for x, y in normalized):
        raise ValueError("LR points require non-negative updates and positive ratios")
    if any(b[0] <= a[0] for a, b in zip(normalized, normalized[1:])):
        raise ValueError("LR update coordinates must be strictly increasing")
    update = int(update)
    if update <= normalized[0][0]:
        return normalized[0][1]
    for left, right in zip(normalized, normalized[1:]):
        if update <= right[0]:
            progress = (update - left[0]) / (right[0] - left[0])
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return right[1] + (left[1] - right[1]) * cosine
    return normalized[-1][1]


@HOOKS.register_module()
class HOD26UpdateCosineLrUpdaterHook(LrUpdaterHook):
    """Resume-safe schedule indexed by the absolute optimizer update."""

    def __init__(self, points: Sequence[Sequence[float]], **kwargs) -> None:
        self.points = tuple((int(x), float(y)) for x, y in points)
        piecewise_cosine_ratio(0, self.points)
        super().__init__(by_epoch=False, warmup=None, **kwargs)

    def get_lr(self, runner, base_lr: float) -> float:
        return float(base_lr) * piecewise_cosine_ratio(runner.iter, self.points)


def _vit_layer_id(name: str, depth: int) -> int:
    if name.startswith("backbone.blocks."):
        return min(depth, int(name.split(".")[2]) + 1)
    if name.startswith("backbone.patch_embed") or name == "backbone.pos_embed":
        return 0
    return depth + 1


def _is_new_hsi_parameter(name: str) -> bool:
    return name.startswith(
        (
            "backbone.proxy_",
            "backbone.spectral_",
            "neck.spectral_fusions",
            "neck.object_activation",
            "neck.object_gate",
            "neck.object_refinements",
        )
    )


def _is_class_parameter(name: str) -> bool:
    return any(
        token in name
        for token in (
            "query_head.cls_branches",
            "query_head.label_embedding",
            "roi_head.0.bbox_head.fc_cls",
            "bbox_head.0.atss_cls",
        )
    )


def _no_decay(name: str, parameter) -> bool:
    return (
        parameter.ndim == 1
        or name.endswith(".bias")
        or "pos_embed" in name
        or "relative_position" in name
        or ".gate" in name
        or "imagenet_" in name
    )


@OPTIMIZER_BUILDERS.register_module()
class HOD26LayerDecayOptimizerConstructor(DefaultOptimizerConstructor):
    """Protect early ViT blocks while rapidly fitting new spectral modules."""

    def add_params(self, params, module, **kwargs) -> None:
        depth = int(self.paramwise_cfg.get("num_layers", 24))
        decay_rate = float(self.paramwise_cfg.get("layer_decay_rate", 0.8))
        hsi_lr_mult = float(self.paramwise_cfg.get("hsi_lr_mult", 8.0))
        class_lr_mult = float(self.paramwise_cfg.get("class_lr_mult", 4.0))
        grouped = defaultdict(
            lambda: {
                "params": [],
                "param_names": [],
            }
        )
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if _is_new_hsi_parameter(name):
                scale = hsi_lr_mult
                family = "hsi"
            elif _is_class_parameter(name):
                scale = class_lr_mult
                family = "class"
            elif name.startswith("backbone."):
                layer_id = _vit_layer_id(name, depth)
                scale = decay_rate ** (depth + 1 - layer_id)
                family = f"vit_{layer_id:02d}"
            else:
                scale = 1.0
                family = "detector"
            decay = not _no_decay(name, parameter)
            key = (family, scale, decay)
            grouped[key]["params"].append(parameter)
            grouped[key]["param_names"].append(name)

        for (family, scale, decay), group in grouped.items():
            group.update(
                group_name=f"{family}_{'decay' if decay else 'no_decay'}",
                lr=self.base_lr * scale,
                initial_lr=self.base_lr * scale,
                lr_scale=scale,
                weight_decay=self.base_wd if decay else 0.0,
            )
            params.append(group)
