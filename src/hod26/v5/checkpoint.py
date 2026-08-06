"""Build the auditable V4-e30 -> V5 single-model initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch
from mmcv import Config
from mmdet.models import build_detector

from hod26.v5.model import collapse_adjacent_difference_weight

V4_E30_SHA256 = "2b96f909af97c93c01c2141d032eb391428b18aac76be5e38f4f3ae5f9198932"
DFINE_X_SHA256 = "f240a1128280eb6ce0fff911e8811a8dff669e18be25f5160db7b4263452277d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _primary_state(payload) -> OrderedDict[str, torch.Tensor]:
    state = payload["state_dict"]
    return OrderedDict(
        (key.removeprefix("module."), value)
        for key, value in state.items()
        if not key.startswith("ema_") and not key.startswith("module.ema_")
    )


def build_initialization(
    config_path: Path,
    parent_path: Path,
    dfine_path: Path,
    output_path: Path,
) -> dict:
    if sha256_file(parent_path) != V4_E30_SHA256:
        raise ValueError("V4 e30 parent SHA-256 mismatch")
    if sha256_file(dfine_path) != DFINE_X_SHA256:
        raise ValueError("D-FINE-X source SHA-256 mismatch")
    config = Config.fromfile(str(config_path))
    model = build_detector(
        config.model,
        train_cfg=config.get("train_cfg"),
        test_cfg=config.get("test_cfg"),
    )
    target = model.state_dict()
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
        "backbone.legacy_proxy_input.weight":
            "backbone.proxy_residual.0.weight",
        "backbone.legacy_proxy_input.bias":
            "backbone.proxy_residual.0.bias",
        "backbone.legacy_spectral_input.weight":
            "backbone.spectral_stem.0.0.weight",
    }
    for destination, source in legacy_map.items():
        target[destination] = parent[source].to(dtype=target[destination].dtype)

    # Start all scale heads from the trained V4 object-activation function.
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

    external_copies = 0
    for layer in range(6):
        for mlp_layer in range(3):
            for parameter in ("weight", "bias"):
                source = (
                    f"decoder.dec_bbox_head.{layer}.layers."
                    f"{mlp_layer}.{parameter}"
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
        "hod26_version": "v5",
        "single_model": True,
        "parent": str(parent_path.resolve()),
        "parent_sha256": V4_E30_SHA256,
        "parent_epoch": parent_payload.get("meta", {}).get("epoch"),
        "parent_iter": parent_payload.get("meta", {}).get("iter"),
        "external_fdr_source": str(dfine_path.resolve()),
        "external_fdr_sha256": DFINE_X_SHA256,
        "external_scope": "FDR and LQE only; no external class head",
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
        "parent_tensors": copied_parent,
        "collapsed_tensors": list(collapsed_keys),
        "frozen_v4_numeric_anchor_tensors": len(legacy_map),
        "salience_tensors_from_parent": salience_copies,
        "dfine_fdr_lqe_tensors": external_copies,
        "transition_zero": True,
    }
    output_path.with_suffix(".pth.provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v5/co_dino_vitl_fdr_salience_fold0.py"
    )
    parser.add_argument(
        "--parent",
        default="storage/v4/runs/co_dino_vitl_hsi_fold0/"
        "best_bbox_mAP_epoch_30.pth",
    )
    parser.add_argument(
        "--dfine", default="storage/pretrained/v2/deim_dfine_x_object365.pth"
    )
    parser.add_argument(
        "--output", default="storage/pretrained/v5/v4_e30_dfine_to_v5_init.pth"
    )
    args = parser.parse_args()
    report = build_initialization(
        Path(args.config), Path(args.parent), Path(args.dfine), Path(args.output)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
