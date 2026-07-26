from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from hod26.submission import validate_submission, write_submission
from hod26.v2.compat import install_deim_compatibility


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


@torch.inference_mode()
def infer(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    upstream_path: str | Path = "storage/upstream/DEIM",
    device_name: str = "cuda:0",
    raw_weights: bool = False,
    confidence: float | None = None,
) -> tuple[Path, dict[str, Any]]:
    upstream = Path(upstream_path).resolve()
    if not (upstream / "engine" / "__init__.py").is_file():
        raise FileNotFoundError(f"Official DEIM checkout missing: {upstream}")
    sys.path.insert(0, str(upstream))
    install_deim_compatibility()

    from engine.core import YAMLConfig
    from hod26.v2 import extensions  # noqa: F401 - registers V2 modules

    cfg = YAMLConfig(str(config_path), device=device_name, use_amp=True)
    device = torch.device(device_name)
    model = cfg.model.to(device).eval()
    checkpoint = torch.load(
        checkpoint_path,
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
    postprocessor = cfg.postprocessor.to(device).eval()
    loader = cfg.val_dataloader
    threshold = float(
        confidence
        if confidence is not None
        else cfg.yaml_cfg["inference"]["confidence"]
    )

    rows: list[dict[str, Any]] = []
    for samples, targets in loader:
        samples = samples.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model(samples)
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
    test_manifest = json.loads(
        Path("storage/cache/test_manifest.json").read_text(encoding="utf-8")
    )
    report = validate_submission(
        destination,
        test_manifest["samples"],
        num_classes=int(cfg.yaml_cfg["num_classes"]),
        include_id=bool(cfg.yaml_cfg["inference"]["include_id"]),
    )
    report.update(
        {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "weights": weight_source,
            "confidence": threshold,
            "architecture": "HSI-DEIM-D-FINE-X",
        }
    )
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
    parser.add_argument("--upstream", default="storage/upstream/DEIM")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--confidence", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, report = infer(
        args.config,
        args.checkpoint,
        output_path=args.output,
        upstream_path=args.upstream,
        device_name=args.device,
        raw_weights=args.raw_weights,
        confidence=args.confidence,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
