"""Single-model full-TTA inference and class-wise Soft-NMS CSV generation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from hod26.config import atomic_json_dump
from hod26.data import decode_x2cube
from hod26.submission import validate_submission, write_submission

from .constants import HOD26_CLASSES
from .data import load_band_limits, normalize_cube


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_cube(path: Path, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    mosaic = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mosaic is None:
        raise FileNotFoundError(path)
    if mosaic.dtype != np.uint16 or mosaic.ndim != 2:
        raise ValueError(f"Expected uint16 mosaic at {path}")
    return normalize_cube(decode_x2cube(mosaic), low, high)


def _resize_pad(
    cube: np.ndarray, width: int
) -> tuple[np.ndarray, float, float, int, int]:
    height = width // 2
    source_height, source_width = cube.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        cube, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.zeros((height, width, 16), dtype=np.float32)
    canvas[:resized_height, :resized_width] = resized
    return (
        canvas,
        resized_width / source_width,
        resized_height / source_height,
        resized_width,
        resized_height,
    )


def _flip_cube(cube: np.ndarray, flip: str) -> np.ndarray:
    if flip == "none":
        return cube
    if flip == "horizontal":
        return np.ascontiguousarray(cube[:, ::-1])
    if flip == "vertical":
        return np.ascontiguousarray(cube[::-1])
    if flip == "both":
        return np.ascontiguousarray(cube[::-1, ::-1])
    raise ValueError(f"Unknown flip {flip!r}")


def _invert_boxes(
    boxes: np.ndarray,
    *,
    sx: float,
    sy: float,
    original_width: int,
    original_height: int,
    flip: str,
) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, 0::2] /= sx
    boxes[:, 1::2] /= sy
    if flip in ("horizontal", "both"):
        old_x1 = boxes[:, 0].copy()
        boxes[:, 0] = original_width - boxes[:, 2]
        boxes[:, 2] = original_width - old_x1
    if flip in ("vertical", "both"):
        old_y1 = boxes[:, 1].copy()
        boxes[:, 1] = original_height - boxes[:, 3]
        boxes[:, 3] = original_height - old_y1
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, original_width)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, original_height)
    return boxes


def _state_dict(path: Path) -> tuple[OrderedDict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    state = OrderedDict()
    for key, value in payload["state_dict"].items():
        key = key[7:] if key.startswith("module.") else key
        if key.startswith("ema_"):
            continue
        state[key] = value
    return state, payload.get("meta", {})


def build_model(
    *,
    upstream: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    sys.path.insert(0, str(upstream.resolve()))
    import projects.models  # noqa: F401
    from mmcv import Config
    from mmcv.runner import wrap_fp16_model
    from mmdet.models import build_detector

    import hod26.v4  # noqa: F401

    config = Config.fromfile(str(config_path.resolve()))
    model = build_detector(
        config.model,
        train_cfg=config.get("train_cfg"),
        test_cfg=config.get("test_cfg"),
    )
    state, metadata = _state_dict(checkpoint_path)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Checkpoint/config mismatch: "
            f"missing={incompatible.missing_keys[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    del state
    gc.collect()
    model.CLASSES = HOD26_CLASSES
    wrap_fp16_model(model)
    model.to(device).eval()
    return model, config, metadata


def _soft_nms_class(
    arrays: Sequence[np.ndarray],
    *,
    iou_threshold: float,
    sigma: float,
    min_score: float,
    method: str = "linear",
) -> np.ndarray:
    from mmcv.ops import soft_nms

    if method not in {"linear", "gaussian"}:
        raise ValueError(f"Unsupported Soft-NMS method: {method!r}")
    nonempty = [array.astype(np.float32, copy=False) for array in arrays if len(array)]
    if not nonempty:
        return np.zeros((0, 5), dtype=np.float32)
    merged = np.concatenate(nonempty, axis=0)
    valid = (
        np.isfinite(merged).all(axis=1)
        & (merged[:, 2] > merged[:, 0])
        & (merged[:, 3] > merged[:, 1])
        & (merged[:, 4] >= min_score)
    )
    merged = merged[valid]
    if not len(merged):
        return np.zeros((0, 5), dtype=np.float32)
    detections, _ = soft_nms(
        merged[:, :4],
        merged[:, 4],
        iou_threshold=float(iou_threshold),
        sigma=float(sigma),
        min_score=float(min_score),
        method=method,
        offset=0,
    )
    order = np.argsort(-detections[:, 4])
    return np.ascontiguousarray(detections[order], dtype=np.float32)


@torch.inference_mode()
def infer(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path = "storage/cache/test_manifest.json",
    data_root: str | Path = "data/raw",
    stats_path: str | Path = "storage/v4/dataset/fold0_band_stats.json",
    upstream_path: str | Path = "storage/upstream/Co-DETR",
    device_name: str = "cuda:0",
    widths: Sequence[int] = (1024, 1152, 1280, 1408),
    flips: Sequence[str] = ("none", "horizontal", "vertical", "both"),
    pre_score_threshold: float = 0.0001,
    output_score_threshold: float = 0.0001,
    soft_nms_iou: float = 0.65,
    soft_nms_sigma: float = 0.5,
    soft_nms_method: str = "linear",
    max_detections: int = 300,
    limit: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not widths or any(width <= 0 or width % 32 for width in widths):
        raise ValueError("TTA widths must be positive multiples of 32")
    if not flips or any(
        flip not in {"none", "horizontal", "vertical", "both"} for flip in flips
    ):
        raise ValueError("Invalid TTA flip set")
    config_file = Path(config_path)
    checkpoint_file = Path(checkpoint_path)
    manifest_file = Path(manifest_path)
    root = Path(data_root)
    device = torch.device(device_name)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    samples = manifest["samples"][:limit] if limit else manifest["samples"]
    low, high = load_band_limits(stats_path)
    model, config, checkpoint_meta = build_model(
        upstream=Path(upstream_path),
        config_path=config_file,
        checkpoint_path=checkpoint_file,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for image_index, sample in enumerate(samples):
        cube = _load_cube(root / sample["image"], low, high)
        original_height, original_width = cube.shape[:2]
        if (original_width, original_height) != (
            int(sample["width"]),
            int(sample["height"]),
        ):
            raise ValueError(f"Manifest/image size mismatch for {sample['image_id']}")
        candidates: list[list[np.ndarray]] = [[] for _ in range(len(HOD26_CLASSES))]
        for width in widths:
            for flip in flips:
                view = _flip_cube(cube, flip)
                canvas, sx, sy, resized_width, resized_height = _resize_pad(
                    view, int(width)
                )
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1)))
                    .unsqueeze(0)
                    .to(device, non_blocking=True)
                )
                meta = {
                    "filename": str((root / sample["image"]).resolve()),
                    "ori_filename": Path(sample["image"]).name,
                    "ori_shape": (original_height, original_width, 16),
                    "img_shape": (resized_height, resized_width, 16),
                    "pad_shape": canvas.shape,
                    "scale_factor": np.asarray((sx, sy, sx, sy), dtype=np.float32),
                    "flip": False,
                    "flip_direction": None,
                    "img_norm_cfg": {
                        "mean": np.zeros(16, dtype=np.float32),
                        "std": np.ones(16, dtype=np.float32),
                        "to_rgb": False,
                    },
                }
                result = model(
                    return_loss=False,
                    rescale=False,
                    img=[tensor],
                    img_metas=[[meta]],
                )[0]
                for class_id, detections in enumerate(result):
                    detections = np.asarray(detections, dtype=np.float32)
                    if not len(detections):
                        continue
                    detections = detections[
                        detections[:, 4] >= pre_score_threshold
                    ].copy()
                    if not len(detections):
                        continue
                    detections[:, :4] = _invert_boxes(
                        detections[:, :4],
                        sx=sx,
                        sy=sy,
                        original_width=original_width,
                        original_height=original_height,
                        flip=flip,
                    )
                    candidates[class_id].append(detections)
                del tensor, result

        image_rows: list[dict[str, Any]] = []
        for class_id, class_candidates in enumerate(candidates):
            detections = _soft_nms_class(
                class_candidates,
                iou_threshold=soft_nms_iou,
                sigma=soft_nms_sigma,
                min_score=output_score_threshold,
                method=soft_nms_method,
            )
            for x1, y1, x2, y2, score in detections:
                if score < output_score_threshold or x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
                    continue
                image_rows.append(
                    {
                        "image_id": str(sample["image_id"]),
                        "class_id": class_id,
                        "confidence": float(score),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )
        image_rows.sort(key=lambda row: -row["confidence"])
        rows.extend(image_rows[:max_detections])
        if (image_index + 1) % 25 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[V4 inference] {image_index + 1}/{len(samples)} images, "
                f"{elapsed / (image_index + 1):.2f}s/image",
                flush=True,
            )

    destination = Path(output_path)
    rows.sort(key=lambda row: (int(row["image_id"]), -row["confidence"]))
    write_submission(rows, destination, include_id=True)
    validation_samples = manifest["samples"]
    report = validate_submission(
        destination,
        validation_samples,
        num_classes=len(HOD26_CLASSES),
        include_id=True,
    )
    report.update(
        {
            "architecture": {
                "HOD26V5RCoDETR": (
                    "HOD26 V5R Co-DINO ViT-L + supervised pre-anchor + "
                    "FDR/LQE + multilevel salience"
                ),
                "HOD26V5CoDETR": (
                    "HOD26 V5 Co-DINO ViT-L + FDR/LQE + multilevel salience"
                ),
            }.get(
                config.model.type,
                "HOD26 V4 object-aware Co-DINO ViT-L 16-band",
            ),
            "single_model": True,
            "prediction_ensemble": False,
            "checkpoint": str(checkpoint_file.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_file),
            "checkpoint_epoch": checkpoint_meta.get("epoch"),
            "checkpoint_iter": checkpoint_meta.get("iter"),
            "config": str(config_file.resolve()),
            "config_sha256": _sha256(config_file),
            "stats": str(Path(stats_path).resolve()),
            "stats_sha256": _sha256(Path(stats_path)),
            "tta_widths": list(widths),
            "tta_flips": list(flips),
            "tta_views_per_image": len(widths) * len(flips),
            "merge": (f"class-wise {soft_nms_method} Soft-NMS over same-model views"),
            "soft_nms_iou": soft_nms_iou,
            "soft_nms_sigma": soft_nms_sigma,
            "soft_nms_method": soft_nms_method,
            "pre_score_threshold": pre_score_threshold,
            "output_score_threshold": output_score_threshold,
            "max_detections_per_image": max_detections,
            "processed_images": len(samples),
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        }
    )
    audit_path = destination.with_suffix(destination.suffix + ".audit.json")
    report["audit"] = str(audit_path.resolve())
    atomic_json_dump(report, audit_path)
    if not report["valid"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return destination, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a V4 single-model full-TTA HOD26 CSV"
    )
    parser.add_argument("--config", default="configs/v4/co_dino_vitl_hsi_fold0.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="storage/cache/test_manifest.json")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--stats", default="storage/v4/dataset/fold0_band_stats.json")
    parser.add_argument("--upstream", default="storage/upstream/Co-DETR")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--widths", nargs="+", type=int, default=[1024, 1152, 1280, 1408]
    )
    parser.add_argument(
        "--flips",
        nargs="+",
        default=["none", "horizontal", "vertical", "both"],
    )
    parser.add_argument("--pre-score", type=float, default=0.0001)
    parser.add_argument("--output-score", type=float, default=0.0001)
    parser.add_argument("--soft-nms-iou", type=float, default=0.65)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.5)
    parser.add_argument(
        "--soft-nms-method",
        choices=("linear", "gaussian"),
        default="linear",
    )
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, report = infer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        manifest_path=args.manifest,
        data_root=args.data_root,
        stats_path=args.stats,
        upstream_path=args.upstream,
        device_name=args.device,
        widths=args.widths,
        flips=args.flips,
        pre_score_threshold=args.pre_score,
        output_score_threshold=args.output_score,
        soft_nms_iou=args.soft_nms_iou,
        soft_nms_sigma=args.soft_nms_sigma,
        soft_nms_method=args.soft_nms_method,
        max_detections=args.max_detections,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
