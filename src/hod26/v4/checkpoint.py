"""Convert the official 80-class Co-DINO ViT-L checkpoint to HOD26 V4."""

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


def _git_output(repo: Path, *args: str) -> str:
    resolved = repo.resolve()
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={resolved}",
            "-C",
            str(resolved),
            *args,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def verify_inputs(upstream: Path, checkpoint: Path) -> dict[str, Any]:
    if not (upstream / "projects" / "models").is_dir():
        raise FileNotFoundError(f"Co-DETR checkout is incomplete: {upstream}")
    commit = _git_output(upstream, "rev-parse", "HEAD")
    if commit != OFFICIAL_CODETR_COMMIT:
        raise ValueError(f"Expected Co-DETR {OFFICIAL_CODETR_COMMIT}, found {commit}")
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError(
            "Official checkpoint SHA-256 mismatch: "
            f"expected {OFFICIAL_CHECKPOINT_SHA256}, got {checkpoint_hash}"
        )
    resolved_upstream = upstream.resolve()
    diff = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={resolved_upstream}",
            "-C",
            str(resolved_upstream),
            "diff",
            "--binary",
            "HEAD",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "upstream_path": str(upstream.resolve()),
        "upstream_commit": commit,
        "local_compatibility_patch_sha256": hashlib.sha256(diff).hexdigest(),
        "official_checkpoint": str(checkpoint.resolve()),
        "official_checkpoint_sha256": checkpoint_hash,
        "official_checkpoint_url": OFFICIAL_CHECKPOINT_URL,
    }


def _semantic_rows(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    background: bool = False,
) -> torch.Tensor:
    result = target.clone()
    coco_index = {name: index for index, name in enumerate(COCO_CLASSES)}
    for target_index, target_name in enumerate(HOD26_CLASSES):
        source_names = HOD26_TO_COCO[target_name]
        indices = [coco_index[name] for name in source_names]
        result[target_index].copy_(source[indices].to(dtype=result.dtype).mean(dim=0))
    if background:
        result[-1].copy_(source[-1].to(dtype=result.dtype))
    return result


def _convert_class_tensor(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor | None:
    if (
        "query_head.cls_branches" in key
        or key == "query_head.label_embedding.weight"
        or "bbox_head.0.atss_cls" in key
    ):
        if source.shape[0] == 80 and target.shape[0] == 18:
            return _semantic_rows(source, target)
    if "roi_head.0.bbox_head.fc_cls" in key:
        if source.shape[0] == 81 and target.shape[0] == 19:
            return _semantic_rows(source, target, background=True)
    return None


def build_target_model(upstream: Path, config_path: Path):
    sys.path.insert(0, str(upstream.resolve()))
    import projects.models  # noqa: F401
    from mmcv import Config
    from mmdet.models import build_detector

    import hod26.v4  # noqa: F401

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
    upstream = Path(upstream_path)
    config_file = Path(config_path)
    source_file = Path(source_path)
    output_file = Path(output_path)
    provenance = verify_inputs(upstream, source_file)

    torch.manual_seed(20260728)
    logging.disable(logging.INFO)
    model, config = build_target_model(upstream, config_file)
    target_state = model.state_dict()
    payload = torch.load(source_file, map_location="cpu")
    source_state = payload.get("state_dict", payload)
    source_state = OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in source_state.items()
        if not key.startswith("ema_")
    )
    del payload
    gc.collect()

    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    exact = 0
    semantic = 0
    initialized = 0
    initialized_keys: list[str] = []
    for key, target in target_state.items():
        source = source_state.get(key)
        if source is not None and source.shape == target.shape:
            converted[key] = source.to(dtype=target.dtype)
            exact += 1
            continue
        mapped = (
            _convert_class_tensor(key, source, target) if source is not None else None
        )
        if mapped is not None:
            converted[key] = mapped
            semantic += 1
        else:
            converted[key] = target.cpu()
            initialized += 1
            initialized_keys.append(key)

    unexpected = sorted(set(source_state) - set(target_state))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    torch.save(
        {
            "meta": {
                "architecture": ("HOD26 V4 object-aware Co-DINO ViT-L 16-band"),
                "classes": HOD26_CLASSES,
                "source_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
                "source_upstream_commit": OFFICIAL_CODETR_COMMIT,
                "config": config.pretty_text,
                "initialization": (
                    "Exact official Co-DINO tensors; semantically remapped "
                    "18-class rows; zero-start HSI fusion, object gating and "
                    "cross-scale refinement modules"
                ),
            },
            "state_dict": converted,
        },
        temporary,
    )
    temporary.replace(output_file)

    report = {
        **provenance,
        "output": str(output_file.resolve()),
        "output_sha256": sha256_file(output_file),
        "config": str(config_file.resolve()),
        "config_sha256": sha256_file(config_file),
        "exact_tensors": exact,
        "semantic_class_tensors": semantic,
        "new_initialized_tensors": initialized,
        "new_initialized_key_sample": initialized_keys[:40],
        "ignored_source_tensor_count": len(unexpected),
        "ignored_source_key_sample": unexpected[:20],
        "single_model_initialization": True,
        "external_pretraining_declared": True,
    }
    atomic_json_dump(
        report, output_file.with_suffix(output_file.suffix + ".provenance.json")
    )
    atomic_json_dump(
        provenance,
        source_file.with_suffix(source_file.suffix + ".provenance.json"),
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the exact HOD26 V4 initialization checkpoint"
    )
    parser.add_argument("--upstream", default="storage/upstream/Co-DETR")
    parser.add_argument("--config", default="configs/v4/co_dino_vitl_hsi_fold0.py")
    parser.add_argument(
        "--source", default="storage/pretrained/co-detr/pytorch_model.pth"
    )
    parser.add_argument(
        "--output",
        default="storage/pretrained/v4/co_dino_vitl_hod26_init.pth",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = convert_checkpoint(
        upstream_path=args.upstream,
        config_path=args.config,
        source_path=args.source,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
