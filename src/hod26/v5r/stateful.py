"""Build a clean, stateful V4-e30 -> V5R bootstrap checkpoint."""

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

from hod26.v5.model import collapse_adjacent_difference_weight
from hod26.v5.stateful import (
    COLLAPSED_CONVOLUTIONS,
    LEGACY_MODEL_SOURCES,
    _clone_optimizer_entry,
    _named_optimizer_states,
    ema_buffer_name,
    salience_parent_name,
)
from hod26.v5r.checkpoint import (
    FAULTY_V5_E18_SHA256,
    V4_E30_SHA256,
    sha256_file,
)


FRESH_PREFIXES = (
    "query_head.fdr_branches.",
    "query_head.lqe_layers.",
    "query_head.pre_bbox_head.",
    "neck.salience_high.",
)


def pre_bbox_parent_name(name: str) -> str | None:
    prefix = "query_head.pre_bbox_head."
    if not name.startswith(prefix):
        return None
    return "query_head.reg_branches.0." + name.removeprefix(prefix)


def transplant_optimizer_state(
    parent_optimizer: Mapping,
    target_optimizer: Mapping,
    target_shapes: Mapping[str, tuple[int, ...]],
) -> tuple[dict, dict]:
    """Preserve all exact V4 coordinates; keep semantic/new branches fresh."""

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
        # Salience/pre-head weight cloning is useful initialization, but their
        # gradients live at a new scale/role.  Unlike the faulty V5 bootstrap,
        # V5R does not pretend those moments are coordinate-equivalent.
        if target_name.startswith(("neck.salience_high.", "query_head.pre_bbox_head.")):
            fresh_names.append(target_name)
            kinds[
                (
                    "fresh_salience"
                    if target_name.startswith("neck.salience_high.")
                    else "fresh_pre_bbox"
                )
            ] += 1
            continue

        source_state = parent_states.get(target_name)
        if source_state is None:
            fresh_names.append(target_name)
            if target_name.startswith("query_head.fdr_branches."):
                kinds["fresh_fdr"] += 1
            elif target_name.startswith("query_head.lqe_layers."):
                kinds["fresh_lqe"] += 1
            else:
                kinds["unexpected_fresh"] += 1
            continue
        target_shape = target_shapes[target_name]
        raw16_slice = target_name in COLLAPSED_CONVOLUTIONS
        transplanted["state"][target_id] = _clone_optimizer_entry(
            source_state,
            target_shape=target_shape,
            raw16_slice=raw16_slice,
        )
        used_parent_names.add(target_name)
        preserved_numel += (
            int(torch.tensor(target_shape).prod().item()) if target_shape else 1
        )
        kinds["raw16_gradient_slice" if raw16_slice else "direct"] += 1

    unused_parent = sorted(set(parent_states) - used_parent_names)
    if unused_parent:
        raise ValueError(
            "V5R failed to preserve exact parent optimizer states: "
            + ", ".join(unused_parent[:10])
        )
    unexpected_fresh = [
        name for name in fresh_names if not name.startswith(FRESH_PREFIXES)
    ]
    if unexpected_fresh or kinds["unexpected_fresh"]:
        raise ValueError(
            "Unexpected fresh V5R Adam state: " + ", ".join(unexpected_fresh[:10])
        )
    report = {
        "parent_optimizer_states": len(parent_states),
        "target_trainable_tensors": len(target_ids),
        "transplanted_optimizer_states": len(transplanted["state"]),
        "direct_states": kinds["direct"],
        "raw16_gradient_slice_states": kinds["raw16_gradient_slice"],
        "fresh_optimizer_states": len(fresh_names),
        "fresh_fdr_states": kinds["fresh_fdr"],
        "fresh_lqe_states": kinds["fresh_lqe"],
        "fresh_pre_bbox_states": kinds["fresh_pre_bbox"],
        "fresh_salience_states": kinds["fresh_salience"],
        "salience_cloned_states": 0,
        "fresh_optimizer_names": fresh_names,
        "preserved_trainable_numel": preserved_numel,
        "total_trainable_numel": total_numel,
        "preserved_trainable_fraction": preserved_numel / total_numel,
    }
    return transplanted, report


def build_v5r_raw_state(
    parent_state: Mapping[str, torch.Tensor],
    v5r_ema_state: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], dict]:
    """Construct raw train weights independently of the inherited EMA side."""

    raw_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    sources: Counter[str] = Counter()
    for target_name, ema_value in v5r_ema_state.items():
        if target_name in COLLAPSED_CONVOLUTIONS:
            raw_value = collapse_adjacent_difference_weight(
                parent_state[ema_buffer_name(target_name)]
            )
            sources["raw16_collapsed"] += 1
        elif target_name in LEGACY_MODEL_SOURCES:
            source = LEGACY_MODEL_SOURCES[target_name]
            raw_value = parent_state[ema_buffer_name(source)]
            sources["legacy_anchor"] += 1
        elif (source := salience_parent_name(target_name)) is not None:
            raw_value = parent_state[ema_buffer_name(source)]
            sources["salience_weight_clone"] += 1
        elif (source := pre_bbox_parent_name(target_name)) is not None:
            raw_value = parent_state[ema_buffer_name(source)]
            sources["pre_bbox_weight_clone"] += 1
        elif (
            target_name in parent_state
            and ema_buffer_name(target_name) in parent_state
            and parent_state[ema_buffer_name(target_name)].shape == ema_value.shape
        ):
            raw_value = parent_state[ema_buffer_name(target_name)]
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
    return raw_state, {
        "raw_state_tensors": len(raw_state),
        "raw_state_sources": dict(sources),
    }


def build_stateful_initialization(
    config_path: Path,
    parent_path: Path,
    base_initialization_path: Path,
    output_path: Path,
) -> dict:
    parent_sha256 = sha256_file(parent_path)
    if parent_sha256 != V4_E30_SHA256:
        raise ValueError("V4 e30 SHA-256 mismatch")
    if parent_sha256 == FAULTY_V5_E18_SHA256:
        raise ValueError("Faulty V5 e18 is prohibited from V5R training lineage")

    base_report_path = base_initialization_path.with_suffix(".pth.provenance.json")
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    base_sha256 = sha256_file(base_initialization_path)
    if base_sha256 != base_report["output_sha256"]:
        raise ValueError("V5R base initialization SHA-256 mismatch")
    if base_report.get("parent_sha256") != V4_E30_SHA256:
        raise ValueError("V5R base initialization has the wrong parent")
    if base_report.get("prohibited_faulty_v5_e18_sha256") != FAULTY_V5_E18_SHA256:
        raise ValueError("V5R base provenance lost the faulty-lineage exclusion")

    cfg = Config.fromfile(str(config_path))
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    base_payload = torch.load(base_initialization_path, map_location="cpu")
    v5r_ema_state = base_payload["state_dict"]
    model.load_state_dict(v5r_ema_state, strict=True)
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
    raw_state, raw_report = build_v5r_raw_state(parent_state, v5r_ema_state)
    transplanted_optimizer, optimizer_report = transplant_optimizer_state(
        parent_payload["optimizer"], target_optimizer, target_shapes
    )

    checkpoint_state: OrderedDict[str, torch.Tensor] = OrderedDict(v5r_ema_state)
    for name, value in raw_state.items():
        buffer_name = ema_buffer_name(name)
        if buffer_name in checkpoint_state:
            raise ValueError(f"EMA buffer collision: {buffer_name}")
        checkpoint_state[buffer_name] = value

    meta = {
        "epoch": 0,
        "iter": 0,
        "seed": 20260731,
        "hod26_version": "v5r",
        "single_model": True,
        "clean_lineage": True,
        "optimizer_state_transplant": True,
        "optimizer_parent": str(parent_path.resolve()),
        "optimizer_parent_sha256": parent_sha256,
        "optimizer_parent_epoch": parent_meta["epoch"],
        "optimizer_parent_iter": parent_meta["iter"],
        "ema_inherited_updates": parent_meta["iter"],
        "parent_primary_weights": "EMA",
        "parent_training_weights": "raw EMA-hook counterpart",
        "base_v5r_initialization": str(base_initialization_path.resolve()),
        "base_v5r_initialization_sha256": base_sha256,
        "prohibited_faulty_v5_e18_sha256": FAULTY_V5_E18_SHA256,
        "fresh_optimizer_scope": (
            "official FDR/LQE, independent pre-bbox head and multiscale "
            "salience heads"
        ),
        "ema_update_order": "post_successful_optimizer_step",
        "ema_control_buffer_policy": "copied_not_smoothed",
        "fp16_bootstrap_policy": ("fresh_recalibration; parent serialized scale=8.0"),
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
        "prohibited_faulty_v5_e18_sha256": FAULTY_V5_E18_SHA256,
        "base_v5r_initialization": str(base_initialization_path.resolve()),
        "base_v5r_initialization_sha256": base_sha256,
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
        default="configs/v5r/co_dino_vitl_fdr_prehead_fold0.py",
    )
    parser.add_argument(
        "--parent",
        default=(
            "storage/v4/runs/co_dino_vitl_hsi_fold0/" "best_bbox_mAP_epoch_30.pth"
        ),
    )
    parser.add_argument(
        "--base-initialization",
        default="storage/pretrained/v5r/v4_e30_dfine_to_v5r_init.pth",
    )
    parser.add_argument(
        "--output",
        default="storage/pretrained/v5r/v4_e30_stateful_to_v5r_init.pth",
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
