"""Build a V4-e30 -> V5 initialization without optimizer or EMA amnesia."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Mapping

import mmcv
import torch
from mmcv import Config
from mmcv.runner import build_optimizer
from mmdet.models import build_detector

from hod26.v5.checkpoint import V4_E30_SHA256, sha256_file
from hod26.v5.model import collapse_adjacent_difference_weight


COLLAPSED_CONVOLUTIONS = frozenset(
    (
        "backbone.proxy_residual.0.weight",
        "backbone.spectral_stem.0.0.weight",
    )
)
LEGACY_MODEL_SOURCES = {
    "backbone.legacy_proxy_input.weight": "backbone.proxy_residual.0.weight",
    "backbone.legacy_proxy_input.bias": "backbone.proxy_residual.0.bias",
    "backbone.legacy_spectral_input.weight": "backbone.spectral_stem.0.0.weight",
}


def ema_buffer_name(name: str) -> str:
    """Return the buffer name used by MMDetection's ``BaseEMAHook``."""

    return f"ema_{name.replace('.', '_')}"


def salience_parent_name(name: str) -> str | None:
    prefix = "neck.salience_high."
    if not name.startswith(prefix):
        return None
    parts = name.split(".", 3)
    if len(parts) != 4:
        raise ValueError(f"Malformed salience parameter name: {name}")
    return f"neck.object_activation.{parts[3]}"


def _named_optimizer_states(
    optimizer_state: Mapping,
) -> tuple[dict[str, Mapping | None], dict[str, int]]:
    named_states: dict[str, Mapping | None] = {}
    named_ids: dict[str, int] = {}
    for group in optimizer_state["param_groups"]:
        names = group.get("param_names")
        if names is None or len(names) != len(group["params"]):
            raise ValueError("Every optimizer group must retain param_names")
        for parameter_id, name in zip(group["params"], names, strict=True):
            if name in named_states:
                raise ValueError(f"Duplicate optimizer parameter name: {name}")
            named_states[name] = optimizer_state["state"].get(parameter_id)
            named_ids[name] = parameter_id
    return named_states, named_ids


def _clone_optimizer_entry(
    state: Mapping,
    *,
    target_shape: tuple[int, ...],
    raw16_slice: bool,
) -> dict:
    cloned = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            cloned[key] = value
            continue
        if raw16_slice and key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
            if value.ndim < 2 or value.shape[1] != 31:
                raise ValueError(
                    f"Expected a 31-channel Adam tensor, got {tuple(value.shape)}"
                )
            value = value[:, :16]
        cloned[key] = value.detach().clone()
    for key in ("exp_avg", "exp_avg_sq"):
        if key not in cloned or tuple(cloned[key].shape) != target_shape:
            raise ValueError(
                f"Adam state {key} does not match target shape {target_shape}"
            )
    return cloned


def transplant_optimizer_state(
    parent_optimizer: Mapping,
    target_optimizer: Mapping,
    target_shapes: Mapping[str, tuple[int, ...]],
) -> tuple[dict, dict]:
    """Copy every compatible V4 Adam state into the V5 optimizer by name.

    The two 31->16 convolution moments use the first 16 raw-input channels.
    This is the correct gradient-coordinate mapping because the first 16
    evidence channels in V4 are the same independent sensor bands used by V5.
    The three scale-aware salience heads clone the state of the V4 head whose
    weights initialize them. Only genuinely new FDR/LQE parameters stay fresh.
    """

    parent_states, _ = _named_optimizer_states(parent_optimizer)
    _, target_ids = _named_optimizer_states(target_optimizer)
    transplanted = {
        "state": {},
        "param_groups": target_optimizer["param_groups"],
    }
    kinds: Counter[str] = Counter()
    used_parent_names: set[str] = set()
    fresh_names: list[str] = []
    preserved_numel = 0
    total_numel = sum(
        int(torch.tensor(shape).prod().item()) if shape else 1
        for shape in target_shapes.values()
    )

    for target_name, target_id in target_ids.items():
        source_name = salience_parent_name(target_name) or target_name
        source_state = parent_states.get(source_name)
        raw16_slice = target_name in COLLAPSED_CONVOLUTIONS
        if source_state is None:
            fresh_names.append(target_name)
            kinds["fresh"] += 1
            continue
        target_shape = target_shapes[target_name]
        transplanted["state"][target_id] = _clone_optimizer_entry(
            source_state,
            target_shape=target_shape,
            raw16_slice=raw16_slice,
        )
        used_parent_names.add(source_name)
        preserved_numel += (
            int(torch.tensor(target_shape).prod().item()) if target_shape else 1
        )
        if raw16_slice:
            kinds["raw16_gradient_slice"] += 1
        elif source_name != target_name:
            kinds["salience_clone"] += 1
        else:
            kinds["direct"] += 1

    unused_parent = sorted(set(parent_states) - used_parent_names)
    if unused_parent:
        raise ValueError(
            "V5 failed to preserve parent optimizer states: "
            + ", ".join(unused_parent[:10])
        )
    unexpected_fresh = [
        name
        for name in fresh_names
        if not name.startswith(("query_head.fdr_branches.", "query_head.lqe_layers."))
    ]
    if unexpected_fresh:
        raise ValueError(
            "Only genuinely new FDR/LQE parameters may have fresh Adam state: "
            + ", ".join(unexpected_fresh)
        )
    report = {
        "parent_optimizer_states": len(parent_states),
        "target_trainable_tensors": len(target_ids),
        "transplanted_optimizer_states": len(transplanted["state"]),
        "direct_states": kinds["direct"],
        "raw16_gradient_slice_states": kinds["raw16_gradient_slice"],
        "salience_cloned_states": kinds["salience_clone"],
        "fresh_optimizer_states": kinds["fresh"],
        "fresh_optimizer_names": fresh_names,
        "preserved_trainable_numel": preserved_numel,
        "total_trainable_numel": total_numel,
        "preserved_trainable_fraction": preserved_numel / total_numel,
    }
    return transplanted, report


def build_v5_raw_state(
    parent_state: Mapping[str, torch.Tensor],
    v5_ema_state: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], dict]:
    """Construct V5 raw train weights alongside its inherited EMA weights."""

    raw_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    sources: Counter[str] = Counter()
    for target_name, ema_value in v5_ema_state.items():
        source_name = None
        if target_name in COLLAPSED_CONVOLUTIONS:
            source_name = target_name
            raw_value = collapse_adjacent_difference_weight(
                parent_state[ema_buffer_name(source_name)]
            )
            sources["raw16_collapsed"] += 1
        elif target_name in LEGACY_MODEL_SOURCES:
            source_name = LEGACY_MODEL_SOURCES[target_name]
            raw_value = parent_state[ema_buffer_name(source_name)]
            sources["legacy_anchor"] += 1
        elif (salience_source := salience_parent_name(target_name)) is not None:
            source_name = salience_source
            raw_value = parent_state[ema_buffer_name(source_name)]
            sources["salience_clone"] += 1
        elif (
            target_name in parent_state
            and ema_buffer_name(target_name) in parent_state
            and parent_state[ema_buffer_name(target_name)].shape == ema_value.shape
        ):
            source_name = target_name
            raw_value = parent_state[ema_buffer_name(source_name)]
            sources["direct"] += 1
        else:
            raw_value = ema_value
            sources["new_raw_equals_ema"] += 1
        if raw_value.shape != ema_value.shape:
            raise ValueError(
                f"Raw/EMA shape mismatch for {target_name}: "
                f"{tuple(raw_value.shape)} != {tuple(ema_value.shape)}"
            )
        raw_state[target_name] = raw_value

    report = {
        "raw_state_tensors": len(raw_state),
        "raw_state_sources": dict(sources),
    }
    return raw_state, report


def build_stateful_initialization(
    config_path: Path,
    parent_path: Path,
    base_initialization_path: Path,
    output_path: Path,
) -> dict:
    parent_sha256 = sha256_file(parent_path)
    if parent_sha256 != V4_E30_SHA256:
        raise ValueError("V4 e30 SHA-256 mismatch")
    base_report_path = base_initialization_path.with_suffix(".pth.provenance.json")
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    base_sha256 = sha256_file(base_initialization_path)
    if base_sha256 != base_report["output_sha256"]:
        raise ValueError("V5 base initialization SHA-256 mismatch")

    cfg = Config.fromfile(str(config_path))
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    base_payload = torch.load(base_initialization_path, map_location="cpu")
    v5_ema_state = base_payload["state_dict"]
    model.load_state_dict(v5_ema_state, strict=True)
    optimizer = build_optimizer(model, cfg.optimizer)
    target_optimizer = optimizer.state_dict()
    target_shapes = {
        name: tuple(parameter.shape)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    parent_payload = torch.load(parent_path, map_location="cpu")
    parent_meta = parent_payload["meta"]
    if parent_meta.get("epoch") != 30 or parent_meta.get("iter") != 42180:
        raise ValueError("Unexpected V4 e30 checkpoint runner metadata")
    parent_state = parent_payload["state_dict"]
    raw_state, raw_report = build_v5_raw_state(parent_state, v5_ema_state)
    transplanted_optimizer, optimizer_report = transplant_optimizer_state(
        parent_payload["optimizer"],
        target_optimizer,
        target_shapes,
    )

    checkpoint_state: OrderedDict[str, torch.Tensor] = OrderedDict(v5_ema_state)
    for name, value in raw_state.items():
        buffer_name = ema_buffer_name(name)
        if buffer_name in checkpoint_state:
            raise ValueError(f"EMA buffer collision: {buffer_name}")
        checkpoint_state[buffer_name] = value

    meta = {
        "epoch": 0,
        "iter": 0,
        "seed": 20260730,
        "hod26_version": "v5",
        "single_model": True,
        "optimizer_state_transplant": True,
        "optimizer_parent": str(parent_path.resolve()),
        "optimizer_parent_sha256": parent_sha256,
        "optimizer_parent_epoch": parent_meta["epoch"],
        "optimizer_parent_iter": parent_meta["iter"],
        "ema_inherited_updates": parent_meta["iter"],
        "parent_primary_weights": "EMA",
        "parent_training_weights": "raw EMA-hook counterpart",
        "base_v5_initialization": str(base_initialization_path.resolve()),
        "base_v5_initialization_sha256": base_sha256,
        "fresh_optimizer_scope": "official D-FINE FDR/LQE parameters only",
        "config": cfg.pretty_text,
        "mmcv_version": mmcv.__version__,
        "created_at_unix": time.time(),
        **optimizer_report,
        **raw_report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.pth")
    torch.save(
        {
            "meta": meta,
            "state_dict": checkpoint_state,
            "optimizer": transplanted_optimizer,
        },
        temporary,
    )
    temporary.replace(output_path)
    output_sha256 = sha256_file(output_path)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    report = {
        "output": str(output_path.resolve()),
        "output_sha256": output_sha256,
        "config_sha256": config_sha256,
        "parent": str(parent_path.resolve()),
        "parent_sha256": parent_sha256,
        "parent_epoch": parent_meta["epoch"],
        "parent_iter": parent_meta["iter"],
        "base_v5_initialization": str(base_initialization_path.resolve()),
        "base_v5_initialization_sha256": base_sha256,
        "ema_inherited_updates": parent_meta["iter"],
        **optimizer_report,
        **raw_report,
    }
    output_path.with_suffix(".pth.provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/v5/co_dino_vitl_fdr_salience_fold0.py",
    )
    parser.add_argument(
        "--parent",
        default=(
            "storage/v4/runs/co_dino_vitl_hsi_fold0/" "best_bbox_mAP_epoch_30.pth"
        ),
    )
    parser.add_argument(
        "--base-initialization",
        default="storage/pretrained/v5/v4_e30_dfine_to_v5_init.pth",
    )
    parser.add_argument(
        "--output",
        default="storage/pretrained/v5/v4_e30_stateful_to_v5_init.pth",
    )
    args = parser.parse_args()
    report = build_stateful_initialization(
        Path(args.config),
        Path(args.parent),
        Path(args.base_initialization),
        Path(args.output),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
