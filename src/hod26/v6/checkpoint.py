"""Convert the one allowed public Co-DINO ViT-L checkpoint into V6."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from hod26.config import atomic_json_dump

from .constants import (
    COCO_CLASSES,
    HOD26_CLASSES,
    HOD26_TO_COCO,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_CHECKPOINT_URL,
    OFFICIAL_CODETR_COMMIT,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git(upstream: Path, *args: str, binary: bool = False):
    result = subprocess.run(
        ["git", "-c", f"safe.directory={upstream.resolve()}", "-C", str(upstream.resolve()), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    return result.stdout


def verify_public_inputs(upstream: Path, checkpoint: Path) -> dict[str, Any]:
    commit = _git(upstream, "rev-parse", "HEAD").strip()
    if commit != OFFICIAL_CODETR_COMMIT:
        raise ValueError(f"Expected Co-DETR {OFFICIAL_CODETR_COMMIT}, got {commit}")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("The only allowed public checkpoint failed SHA-256")
    patch = _git(upstream, "diff", "--binary", "HEAD", binary=True)
    return {
        "official_checkpoint": str(checkpoint.resolve()),
        "official_checkpoint_url": OFFICIAL_CHECKPOINT_URL,
        "official_checkpoint_sha256": checkpoint_sha,
        "upstream_path": str(upstream.resolve()),
        "upstream_commit": commit,
        "upstream_working_diff_sha256": hashlib.sha256(patch).hexdigest(),
    }


def _semantic_rows(
    source: torch.Tensor, target: torch.Tensor, *, background: bool = False
) -> torch.Tensor:
    result = target.clone()
    index = {name: position for position, name in enumerate(COCO_CLASSES)}
    for target_index, name in enumerate(HOD26_CLASSES):
        rows = [index[source_name] for source_name in HOD26_TO_COCO[name]]
        result[target_index].copy_(source[rows].to(result.dtype).mean(0))
    if background:
        result[-1].copy_(source[-1].to(result.dtype))
    return result


def _convert_class_tensor(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor | None:
    if (
        "query_head.cls_branches" in key
        or key == "query_head.label_embedding.weight"
        or "bbox_head.0.atss_cls" in key
    ) and source.shape[0] == 80 and target.shape[0] == 18:
        return _semantic_rows(source, target)
    if (
        "roi_head.0.bbox_head.fc_cls" in key
        and source.shape[0] == 81 and target.shape[0] == 19
    ):
        return _semantic_rows(source, target, background=True)
    return None


def build_target(upstream: Path, config_path: Path):
    sys.path.insert(0, str(upstream.resolve()))
    import projects.models  # noqa: F401
    from mmcv import Config
    from mmdet.models import build_detector
    import hod26.v6  # noqa: F401

    config = Config.fromfile(str(config_path.resolve()))
    model = build_detector(
        config.model,
        train_cfg=config.get("train_cfg"),
        test_cfg=config.get("test_cfg"),
    )
    model.init_weights()
    return model, config


def convert_checkpoint(
    *,
    upstream_path: str | Path,
    config_path: str | Path,
    source_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    upstream, config_file = Path(upstream_path), Path(config_path)
    source_file, output_file = Path(source_path), Path(output_path)
    provenance = verify_public_inputs(upstream, source_file)
    torch.manual_seed(20260808)
    logging.disable(logging.INFO)
    model, config = build_target(upstream, config_file)
    target_state = model.state_dict()
    source_payload = torch.load(source_file, map_location="cpu")
    source_state = OrderedDict(
        (key.removeprefix("module."), value)
        for key, value in source_payload.get("state_dict", source_payload).items()
        if not key.startswith("ema_") and not key.startswith("module.ema_")
    )
    del source_payload
    gc.collect()

    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    exact = semantic = initialized = 0
    initialized_parameters = initialized_buffers = 0
    initialized_keys: list[str] = []
    parameter_names = dict(model.named_parameters())
    for key, target in target_state.items():
        source = source_state.get(key)
        if source is not None and source.shape == target.shape:
            converted[key] = source.to(target.dtype)
            exact += 1
            continue
        mapped = _convert_class_tensor(key, source, target) if source is not None else None
        if mapped is not None:
            converted[key] = mapped
            semantic += 1
        else:
            converted[key] = target.cpu()
            initialized += 1
            initialized_keys.append(key)
            if key in parameter_names:
                initialized_parameters += 1
            else:
                initialized_buffers += 1
    model.load_state_dict(converted, strict=True)
    unexpected = sorted(set(source_state) - set(target_state))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(".tmp.pth")
    metadata = {
        "hod26_version": "v6",
        "architecture": "CoSpec-DINO ViT-L",
        "independent_lineage": True,
        "single_model": True,
        "external_checkpoint_count": 1,
        "external_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "external_checkpoint_url": OFFICIAL_CHECKPOINT_URL,
        "source_upstream_commit": OFFICIAL_CODETR_COMMIT,
        "classes": HOD26_CLASSES,
        "optimizer": "fresh_not_serialized",
        "ema": "fresh_not_serialized",
        "fp16_scaler": "fresh_not_serialized",
        "epoch": 0,
        "iter": 0,
        "config": config.pretty_text,
    }
    torch.save({"meta": metadata, "state_dict": converted}, temporary)
    temporary.replace(output_file)
    report = {
        **provenance,
        "output": str(output_file.resolve()),
        "output_sha256": sha256_file(output_file),
        "config": str(config_file.resolve()),
        "config_sha256": sha256_file(config_file),
        "exact_public_tensors": exact,
        "semantic_class_tensors": semantic,
        "new_initialized_tensors": initialized,
        "new_initialized_parameter_tensors": initialized_parameters,
        "new_initialized_buffer_tensors": initialized_buffers,
        "new_initialized_key_sample": initialized_keys[:80],
        "ignored_public_tensor_count": len(unexpected),
        "ignored_public_key_sample": unexpected[:30],
        "prohibited_parent_versions": ["v4", "v5", "v5r"],
        "optimizer_state_inherited": False,
        "ema_state_inherited": False,
        "fp16_scaler_inherited": False,
        "single_model_initialization": True,
    }
    atomic_json_dump(report, output_file.with_suffix(".pth.provenance.json"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="storage/upstream/Co-DETR")
    parser.add_argument("--config", default="configs/v6/co_spec_dino_vitl_fold0.py")
    parser.add_argument("--source", default="storage/pretrained/co-detr/pytorch_model.pth")
    parser.add_argument("--output", default="storage/pretrained/v6/co_spec_dino_vitl_public_init.pth")
    args = parser.parse_args()
    report = convert_checkpoint(
        upstream_path=args.upstream,
        config_path=args.config,
        source_path=args.source,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
