"""Successful-update LR schedule and audited V6 parameter families."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import torch
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.optimizer import Fp16OptimizerHook
from mmcv.runner.hooks.lr_updater import LrUpdaterHook
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS
from mmcv.runner.optimizer.default_constructor import DefaultOptimizerConstructor


@torch.no_grad()
def _stable_clip_grad_norm_details(
    parameters, max_norm: float, norm_type: float = 2.0
) -> tuple[torch.Tensor | None, list[torch.nn.Parameter], torch.Tensor | None]:
    if float(norm_type) != 2.0:
        raise ValueError("V6 stable clipping is audited only for L2 norm")
    maximum = float(max_norm)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("V6 max_norm must be positive and finite")
    gradient_parameters = [
        parameter
        for parameter in parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradient_parameters:
        return None, [], None
    tensor_norms = torch.stack([
        torch.linalg.vector_norm(
            parameter.grad.detach(), ord=2, dtype=torch.float64
        )
        for parameter in gradient_parameters
    ])
    total_norm = torch.linalg.vector_norm(tensor_norms, ord=2, dtype=torch.float64)
    if bool(torch.isfinite(total_norm).item()):
        coefficient = min(1.0, maximum / (float(total_norm.item()) + 1e-6))
        if coefficient < 1.0:
            for parameter in gradient_parameters:
                parameter.grad.mul_(coefficient)
    # A genuinely non-finite element is deliberately left untouched. GradScaler
    # has already recorded it during unscale_ and must skip that optimizer step.
    return total_norm, gradient_parameters, tensor_norms


def stable_clip_grad_norm_(
    parameters, max_norm: float, norm_type: float = 2.0
) -> torch.Tensor | None:
    """FP64-reduced L2 clipping for a 368M-parameter mixed-precision model.

    MMCV's second FP32 reduction can overflow even when every gradient element
    and every per-tensor norm is finite, silently turning all gradients to zero.
    Only the small list of per-tensor norms is promoted to FP64.
    """

    total, _, _ = _stable_clip_grad_norm_details(parameters, max_norm, norm_type)
    return total


@HOOKS.register_module()
class HOD26V6StableFp16OptimizerHook(Fp16OptimizerHook):
    """Native GradScaler, stable clipping, and sparse gradient-family audit."""

    def before_run(self, runner) -> None:
        super().before_run(runner)
        self._runner = runner
        self._parameter_names = {}
        self._parameter_families = {}
        for group in runner.optimizer.param_groups:
            names = group.get("param_names")
            if not isinstance(names, list) or len(names) != len(group["params"]):
                raise RuntimeError("V6 clipping hook lost explicit parameter names")
            group_name = str(group.get("group_name", ""))
            if group_name.startswith("visual_"):
                family = "visual"
            elif group_name.startswith("v6_new_"):
                family = "new"
            elif group_name.startswith("v6_class_"):
                family = "class"
            else:
                family = "detector"
            for name, parameter in zip(names, group["params"], strict=True):
                self._parameter_names[id(parameter)] = name
                self._parameter_families[id(parameter)] = family

    def clip_grads(self, params):
        if self.grad_clip is None:
            return None
        unknown = set(self.grad_clip) - {"max_norm", "norm_type"}
        if unknown:
            raise ValueError(f"Unsupported V6 grad_clip keys: {sorted(unknown)}")
        total, parameters, tensor_norms = _stable_clip_grad_norm_details(
            params, **self.grad_clip
        )
        attempted = int(self._runner.iter) + 1
        if (
            total is not None
            and tensor_norms is not None
            and self._runner.rank == 0
            and (attempted <= 4 or attempted % 100 == 0)
        ):
            squares = {}
            for parameter, norm in zip(parameters, tensor_norms, strict=True):
                family = self._parameter_families[id(parameter)]
                squares[family] = squares.get(family, total.new_zeros(())) + norm.square()
            family_metrics = {
                f"v6_grad_norm_{family}": float(value.sqrt().item())
                for family, value in sorted(squares.items())
            }
            self._runner.log_buffer.update(
                family_metrics, self._runner.outputs["num_samples"]
            )
            count = min(5, int(tensor_norms.numel()))
            values, indices = torch.topk(tensor_norms, count)
            values = values.detach().cpu().tolist()
            indices = indices.detach().cpu().tolist()
            top = [
                (self._parameter_names[id(parameters[index])], float(value))
                for value, index in zip(values, indices, strict=True)
            ]
            self._runner.logger.warning(
                "V6 pre-clip gradient audit update=%d total=%.6e families=%s top=%s",
                attempted, float(total.item()), family_metrics, top,
            )
        return total

def optimizer_step_range(optimizer) -> tuple[int, int, int]:
    """Return min/max/count of finite integral AdamW step states."""

    steps: list[int] = []
    for state in optimizer.state.values():
        value = state.get("step")
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.numel() != 1 or not bool(torch.isfinite(value).all()):
                raise RuntimeError("V6 found a malformed optimizer step tensor")
            value = value.item()
        number = float(value)
        rounded = round(number)
        if number < 0 or not math.isclose(number, rounded, abs_tol=1e-6):
            raise RuntimeError(f"V6 found a non-integral optimizer step: {number}")
        steps.append(int(rounded))
    if not steps:
        return 0, 0, 0
    return min(steps), max(steps), len(steps)


def piecewise_cosine_ratio(update: int, points: Sequence[Sequence[float]]) -> float:
    normalized = tuple((int(x), float(y)) for x, y in points)
    if len(normalized) < 2:
        raise ValueError("V6 LR schedule needs at least two points")
    if any(x < 0 or y <= 0 for x, y in normalized):
        raise ValueError("V6 LR points must be non-negative/positive")
    if any(right[0] <= left[0] for left, right in zip(normalized, normalized[1:])):
        raise ValueError("V6 LR update coordinates must increase")
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
class HOD26V6SuccessfulCosineLrUpdaterHook(LrUpdaterHook):
    """AMP-skip-safe cosine schedule indexed by actual AdamW updates."""

    def __init__(self, points: Sequence[Sequence[float]], **kwargs) -> None:
        self.points = tuple((int(x), float(y)) for x, y in points)
        piecewise_cosine_ratio(0, self.points)
        super().__init__(by_epoch=False, warmup=None, **kwargs)

    def get_lr(self, runner, base_lr: float) -> float:
        _, successful, _ = optimizer_step_range(runner.optimizer)
        return float(base_lr) * piecewise_cosine_ratio(successful, self.points)


def _vit_layer_id(name: str, depth: int) -> int:
    if name.startswith("backbone.blocks."):
        return min(depth, int(name.split(".")[2]) + 1)
    if name.startswith("backbone.patch_embed") or name == "backbone.pos_embed":
        return 0
    return depth + 1


def _is_new_v6_parameter(name: str) -> bool:
    return name.startswith(
        (
            "backbone.proxy_",
            "backbone.spectral_",
            "neck.fusions",
            "neck.spectral_proposal_",
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


def _no_decay(name: str, parameter: torch.Tensor) -> bool:
    return (
        parameter.ndim == 1
        or name.endswith(".bias")
        or "pos_embed" in name
        or "relative_position" in name
        or "imagenet_" in name
    )


@OPTIMIZER_BUILDERS.register_module()
class HOD26V6OptimizerConstructor(DefaultOptimizerConstructor):
    """LLRD visual, 1x detector, 8x new HSI and 4x class coordinates."""

    def add_params(self, params, module, **kwargs) -> None:
        depth = int(self.paramwise_cfg.get("num_layers", 24))
        decay_rate = float(self.paramwise_cfg.get("layer_decay_rate", 0.90))
        visual_mult = float(self.paramwise_cfg.get("visual_lr_mult", 0.10))
        new_mult = float(self.paramwise_cfg.get("new_lr_mult", 4.0))
        class_mult = float(self.paramwise_cfg.get("class_lr_mult", 4.0))
        if not 0 < decay_rate <= 1 or min(visual_mult, new_mult, class_mult) <= 0:
            raise ValueError("Invalid V6 optimizer multipliers")
        grouped = defaultdict(lambda: {"params": [], "param_names": []})
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if _is_new_v6_parameter(name):
                scale, family = new_mult, "v6_new"
            elif _is_class_parameter(name):
                scale, family = class_mult, "v6_class"
            elif name.startswith("backbone."):
                layer = _vit_layer_id(name, depth)
                scale = visual_mult * decay_rate ** (depth + 1 - layer)
                family = f"visual_{layer:02d}"
            else:
                scale, family = 1.0, "detector"
            decay = not _no_decay(name, parameter)
            group = grouped[(family, scale, decay)]
            group["params"].append(parameter)
            group["param_names"].append(name)
        for (family, scale, decay), group in grouped.items():
            group.update(
                group_name=f"{family}_{'decay' if decay else 'no_decay'}",
                lr=self.base_lr * scale,
                initial_lr=self.base_lr * scale,
                lr_scale=scale,
                weight_decay=self.base_wd if decay else 0.0,
            )
            params.append(group)
