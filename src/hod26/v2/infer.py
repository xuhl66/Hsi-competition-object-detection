from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hod26.config import atomic_json_dump
from hod26.submission import validate_submission, write_submission
from hod26.v2.compat import install_deim_compatibility


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _state_dict(
    checkpoint: dict[str, Any],
    raw_weights: bool,
) -> tuple[dict[str, torch.Tensor], str]:
    if not raw_weights and checkpoint.get("ema") is not None:
        ema = checkpoint["ema"]
        if isinstance(ema, dict) and "module" in ema:
            return ema["module"], "ema"
    if "model" in checkpoint:
        return checkpoint["model"], "model"
    raise ValueError("Checkpoint contains neither model nor EMA weights")


def _keep_best_class_per_query(
    outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Keep one class hypothesis per query before the official top-k step."""
    logits = outputs["pred_logits"]
    labels = logits.argmax(dim=-1, keepdim=True)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, labels, logits.gather(-1, labels))
    return {**outputs, "pred_logits": filtered}


def _enable_dynamic_spatial_size(model: torch.nn.Module) -> None:
    """Make the official DEIM encoder/decoder build spatial state on demand."""
    for component_name in ("encoder", "decoder"):
        component = getattr(model, component_name, None)
        if component is None or not hasattr(component, "eval_spatial_size"):
            raise ValueError(
                "The configured model cannot change inference resolution: "
                f"missing {component_name}.eval_spatial_size"
            )
        component.eval_spatial_size = None


@torch.inference_mode()
def infer(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    manifest_path: str | Path = "storage/cache/test_manifest.json",
    upstream_path: str | Path = "storage/upstream/DEIM",
    device_name: str = "cuda:0",
    raw_weights: bool = False,
    confidence: float | None = None,
    input_width: int | None = None,
    single_label: bool = False,
) -> tuple[Path, dict[str, Any]]:
    upstream = Path(upstream_path).resolve()
    if not (upstream / "engine" / "__init__.py").is_file():
        raise FileNotFoundError(f"Official DEIM checkout missing: {upstream}")
    sys.path.insert(0, str(upstream))
    install_deim_compatibility()

    from engine.core import YAMLConfig
    from hod26.v2 import extensions  # noqa: F401 - registers V2 modules

    config = Path(config_path).resolve()
    checkpoint_file = Path(checkpoint_path).resolve()
    cfg = YAMLConfig(str(config), device=device_name, use_amp=True)
    device = torch.device(device_name)
    model = cfg.model.to(device).eval()
    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )
    weights, weight_source = _state_dict(checkpoint, raw_weights)
    incompatible = model.load_state_dict(weights, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Checkpoint/config mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    configured_height, configured_width = (
        int(value) for value in cfg.yaml_cfg["eval_spatial_size"]
    )
    inference_width = int(input_width or configured_width)
    if inference_width <= 0 or inference_width % 64:
        raise ValueError("input_width must be a positive multiple of 64")
    inference_height = inference_width // 2
    if (inference_height, inference_width) != (
        configured_height,
        configured_width,
    ):
        _enable_dynamic_spatial_size(model)
    postprocessor = cfg.postprocessor.to(device).eval()
    loader = cfg.val_dataloader
    threshold = float(
        confidence
        if confidence is not None
        else cfg.yaml_cfg["inference"]["confidence"]
    )

    rows: list[dict[str, Any]] = []
    for samples, targets in loader:
        if samples.shape[-2:] != (inference_height, inference_width):
            samples = F.interpolate(
                samples,
                size=(inference_height, inference_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        samples = samples.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model(samples)
        if single_label:
            outputs = _keep_best_class_per_query(outputs)
        sizes = torch.stack(
            [target["orig_size"] for target in targets]
        ).to(device)
        predictions = postprocessor(outputs, sizes)
        for target, prediction in zip(targets, predictions, strict=True):
            image_id = str(int(target["image_id"].item()))
            width, height = (
                float(value) for value in target["orig_size"].tolist()
            )
            keep = prediction["scores"] >= threshold
            boxes = prediction["boxes"][keep].float().cpu()
            scores = prediction["scores"][keep].float().cpu()
            labels = prediction["labels"][keep].cpu()
            boxes[:, 0::2].clamp_(0.0, width)
            boxes[:, 1::2].clamp_(0.0, height)
            # CSV coordinates are written to four decimals; keep a margin so
            # tiny pre-training artifacts cannot collapse after formatting.
            valid = (boxes[:, 2] - boxes[:, 0] >= 1e-3) & (
                boxes[:, 3] - boxes[:, 1] >= 1e-3
            )
            for box, score, label in zip(
                boxes[valid],
                scores[valid],
                labels[valid],
                strict=True,
            ):
                x1, y1, x2, y2 = box.tolist()
                rows.append(
                    {
                        "image_id": image_id,
                        "class_id": int(label),
                        "confidence": float(score),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

    rows.sort(key=lambda row: (int(row["image_id"]), -row["confidence"]))
    destination = Path(
        output_path or cfg.yaml_cfg["inference"]["output"]
    ).resolve()
    write_submission(
        rows,
        destination,
        include_id=bool(cfg.yaml_cfg["inference"]["include_id"]),
    )
    inference_manifest = json.loads(
        Path(manifest_path).read_text(encoding="utf-8")
    )
    report = validate_submission(
        destination,
        inference_manifest["samples"],
        num_classes=int(cfg.yaml_cfg["num_classes"]),
        include_id=bool(cfg.yaml_cfg["inference"]["include_id"]),
    )
    report.update(
        {
            "checkpoint": str(checkpoint_file),
            "checkpoint_sha256": _sha256(checkpoint_file),
            "config": str(config),
            "config_sha256": _sha256(config),
            "manifest": str(Path(manifest_path).resolve()),
            "weights": weight_source,
            "confidence": threshold,
            "input_size": [inference_height, inference_width],
            "single_label_per_query": bool(single_label),
            "architecture": "HSI-DEIM-D-FINE-X",
        }
    )
    audit_path = destination.with_suffix(destination.suffix + ".audit.json")
    report["audit"] = str(audit_path)
    atomic_json_dump(report, audit_path)
    if not report["valid"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return destination, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an official HOD26 CSV with V2"
    )
    parser.add_argument(
        "--config",
        default="configs/v2/deim_dfine_x_hsi_infer.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--manifest",
        default="storage/cache/test_manifest.json",
    )
    parser.add_argument("--upstream", default="storage/upstream/DEIM")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--input-width", type=int)
    parser.add_argument("--single-label", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, report = infer(
        args.config,
        args.checkpoint,
        output_path=args.output,
        manifest_path=args.manifest,
        upstream_path=args.upstream,
        device_name=args.device,
        raw_weights=args.raw_weights,
        confidence=args.confidence,
        input_width=args.input_width,
        single_label=args.single_label,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
