"""Build the clean V4-e30 + official-D-FINE -> V5R initialization."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from mmcv import Config
from mmdet.models import build_detector

from hod26.v5.checkpoint import (
    DFINE_X_SHA256,
    V4_E30_SHA256,
    sha256_file,
)
from hod26.v5.model import collapse_adjacent_difference_weight


FAULTY_V5_E18_SHA256 = (
    "14b8b7a9987cfc18ac32d6139b41d7f31fe664686b38f7e951c6a992c3751ff5"
)


def _primary_state(payload) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (key.removeprefix("module."), value)
        for key, value in payload["state_dict"].items()
        if not key.startswith("ema_") and not key.startswith("module.ema_")
    )


def _copy_pre_bbox_head(
    target: OrderedDict[str, torch.Tensor],
    parent: OrderedDict[str, torch.Tensor],
) -> int:
    copied = 0
    for suffix in (
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
        "4.weight",
        "4.bias",
    ):
        source = f"query_head.reg_branches.0.{suffix}"
        destination = f"query_head.pre_bbox_head.{suffix}"
        if source not in parent or destination not in target:
            raise ValueError(f"Missing V5R pre-head mapping: {source} -> {destination}")
        if parent[source].shape != target[destination].shape:
            raise ValueError(f"Incompatible V5R pre-head tensor: {source}")
        target[destination] = parent[source].to(dtype=target[destination].dtype)
        copied += 1
    return copied


def build_initialization(
    config_path: Path,
    parent_path: Path,
    dfine_path: Path,
    output_path: Path,
) -> dict:
    parent_sha256 = sha256_file(parent_path)
    external_sha256 = sha256_file(dfine_path)
    if parent_sha256 != V4_E30_SHA256:
        raise ValueError("V4 e30 parent SHA-256 mismatch")
    if external_sha256 != DFINE_X_SHA256:
        raise ValueError("D-FINE-X source SHA-256 mismatch")
    if parent_sha256 == FAULTY_V5_E18_SHA256:
        raise ValueError("Faulty V5 e18 is prohibited as a V5R parent")

    config = Config.fromfile(str(config_path))
    model = build_detector(
        config.model,
        train_cfg=config.get("train_cfg"),
        test_cfg=config.get("test_cfg"),
    )
    target = OrderedDict(model.state_dict())
    parent_payload = torch.load(parent_path, map_location="cpu")
    parent = _primary_state(parent_payload)
    external_payload = torch.load(dfine_path, map_location="cpu")
    external = external_payload["model"]

    copied_parent = 0
    for key in list(target):
        if key in parent and parent[key].shape == target[key].shape:
            target[key] = parent[key].to(dtype=target[key].dtype)
            copied_parent += 1

    collapsed_keys = (
        "backbone.proxy_residual.0.weight",
        "backbone.spectral_stem.0.0.weight",
    )
    for key in collapsed_keys:
        target[key] = collapse_adjacent_difference_weight(parent[key]).to(
            dtype=target[key].dtype
        )
    legacy_map = {
        "backbone.legacy_proxy_input.weight": "backbone.proxy_residual.0.weight",
        "backbone.legacy_proxy_input.bias": "backbone.proxy_residual.0.bias",
        "backbone.legacy_spectral_input.weight": "backbone.spectral_stem.0.0.weight",
    }
    for destination, source in legacy_map.items():
        target[destination] = parent[source].to(dtype=target[destination].dtype)

    salience_copies = 0
    for level in range(3):
        for suffix in (
            "layers.0.weight",
            "layers.1.weight",
            "layers.1.bias",
            "layers.3.weight",
            "layers.3.bias",
        ):
            source = f"neck.object_activation.{suffix}"
            destination = f"neck.salience_high.{level}.{suffix}"
            target[destination] = parent[source].to(dtype=target[destination].dtype)
            salience_copies += 1

    pre_head_copies = _copy_pre_bbox_head(target, parent)
    external_copies = 0
    for layer in range(6):
        for mlp_layer in range(3):
            for parameter in ("weight", "bias"):
                source = (
                    f"decoder.dec_bbox_head.{layer}.layers." f"{mlp_layer}.{parameter}"
                )
                destination = (
                    f"query_head.fdr_branches.{layer}.layers."
                    f"{mlp_layer}.{parameter}"
                )
                if external[source].shape != target[destination].shape:
                    raise ValueError(f"Incompatible D-FINE tensor: {source}")
                target[destination] = external[source].to(
                    dtype=target[destination].dtype
                )
                external_copies += 1
        for mlp_layer in range(2):
            for parameter in ("weight", "bias"):
                source = (
                    f"decoder.decoder.lqe_layers.{layer}.reg_conf.layers."
                    f"{mlp_layer}.{parameter}"
                )
                destination = (
                    f"query_head.lqe_layers.{layer}.reg_conf.layers."
                    f"{mlp_layer}.{parameter}"
                )
                if external[source].shape != target[destination].shape:
                    raise ValueError(f"Incompatible D-FINE tensor: {source}")
                target[destination] = external[source].to(
                    dtype=target[destination].dtype
                )
                external_copies += 1

    model.load_state_dict(target, strict=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.pth")
    meta = {
        "hod26_version": "v5r",
        "single_model": True,
        "clean_lineage": True,
        "parent": str(parent_path.resolve()),
        "parent_sha256": parent_sha256,
        "parent_epoch": parent_payload.get("meta", {}).get("epoch"),
        "parent_iter": parent_payload.get("meta", {}).get("iter"),
        "prohibited_faulty_v5_e18_sha256": FAULTY_V5_E18_SHA256,
        "external_fdr_source": str(dfine_path.resolve()),
        "external_fdr_sha256": external_sha256,
        "external_scope": "FDR and LQE only; no external class head",
        "pre_bbox_source": "V4 e30 query_head.reg_branches.0",
        "optimizer_state_inherited": False,
        "ema_state_inherited": False,
        "transition_update": 0,
        "config": config.pretty_text,
    }
    torch.save({"meta": meta, "state_dict": target}, temporary)
    temporary.replace(output_path)
    report = {
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "config_sha256": sha256_file(config_path),
        "parent": str(parent_path.resolve()),
        "parent_sha256": parent_sha256,
        "parent_epoch": parent_payload.get("meta", {}).get("epoch"),
        "parent_iter": parent_payload.get("meta", {}).get("iter"),
        "prohibited_faulty_v5_e18_sha256": FAULTY_V5_E18_SHA256,
        "parent_tensors": copied_parent,
        "collapsed_tensors": list(collapsed_keys),
        "frozen_v4_numeric_anchor_tensors": len(legacy_map),
        "salience_tensors_from_parent": salience_copies,
        "pre_bbox_tensors_from_parent": pre_head_copies,
        "dfine_fdr_lqe_tensors": external_copies,
        "transition_zero": True,
    }
    output_path.with_suffix(".pth.provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v5r/co_dino_vitl_fdr_prehead_fold0.py"
    )
    parser.add_argument(
        "--parent",
        default=(
            "storage/v4/runs/co_dino_vitl_hsi_fold0/" "best_bbox_mAP_epoch_30.pth"
        ),
    )
    parser.add_argument(
        "--dfine", default="storage/pretrained/v2/deim_dfine_x_object365.pth"
    )
    parser.add_argument(
        "--output",
        default="storage/pretrained/v5r/v4_e30_dfine_to_v5r_init.pth",
    )
    args = parser.parse_args()
    report = build_initialization(
        Path(args.config), Path(args.parent), Path(args.dfine), Path(args.output)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
