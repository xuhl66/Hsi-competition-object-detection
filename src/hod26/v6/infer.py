"""Frozen V4-champion inference protocol for one converged V6 EMA checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from hod26.config import atomic_json_dump
from hod26.data import decode_x2cube
from hod26.submission import validate_submission, write_submission

from .constants import HOD26_CLASSES
from .data import load_band_limits, normalize_cube


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_cube(path: Path, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    mosaic = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mosaic is None or mosaic.dtype != np.uint16 or mosaic.ndim != 2:
        raise ValueError(f"Invalid HOD26 uint16 mosaic: {path}")
    return normalize_cube(decode_x2cube(mosaic), low, high)


def _resize_pad(cube: np.ndarray, width: int = 1280):
    height = width // 2
    source_h, source_w = cube.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w, resized_h = int(round(source_w * scale)), int(round(source_h * scale))
    resized = cv2.resize(cube, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 16), np.float32)
    canvas[:resized_h, :resized_w] = resized
    return canvas, resized_w / source_w, resized_h / source_h, resized_w, resized_h


def _restore_boxes(
    boxes: np.ndarray, sx: float, sy: float, width: int, height: int, horizontal: bool
) -> np.ndarray:
    value = boxes.copy()
    value[:, 0::2] /= sx
    value[:, 1::2] /= sy
    if horizontal:
        x1 = value[:, 0].copy()
        value[:, 0] = width - value[:, 2]
        value[:, 2] = width - x1
    value[:, 0::2] = np.clip(value[:, 0::2], 0, width)
    value[:, 1::2] = np.clip(value[:, 1::2], 0, height)
    return value


def _primary_state(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("V6 inference requires an MMCV checkpoint")
    state = OrderedDict()
    for key, value in payload["state_dict"].items():
        key = key.removeprefix("module.")
        if not key.startswith("ema_"):
            state[key] = value
    return state, payload.get("meta", {})


def build_model(config_path: Path, checkpoint: Path, device: torch.device):
    import projects.models  # noqa: F401
    from mmcv import Config
    from mmcv.runner import wrap_fp16_model
    from mmdet.models import build_detector
    import hod26.v6  # noqa: F401

    config = Config.fromfile(str(config_path.resolve()))
    model = build_detector(
        config.model,
        train_cfg=config.get("train_cfg"),
        test_cfg=config.get("test_cfg"),
    )
    state, metadata = _primary_state(checkpoint)
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()
    model.CLASSES = HOD26_CLASSES
    wrap_fp16_model(model)
    model.to(device).eval()
    return model, config, metadata


def _gaussian_soft_nms(arrays: list[np.ndarray]) -> np.ndarray:
    from mmcv.ops import soft_nms

    arrays = [value.astype(np.float32, copy=False) for value in arrays if len(value)]
    if not arrays:
        return np.zeros((0, 5), np.float32)
    merged = np.concatenate(arrays)
    valid = (
        np.isfinite(merged).all(1)
        & (merged[:, 2] > merged[:, 0])
        & (merged[:, 3] > merged[:, 1])
        & (merged[:, 4] >= 0.0001)
    )
    merged = merged[valid]
    if not len(merged):
        return np.zeros((0, 5), np.float32)
    detections, _ = soft_nms(
        merged[:, :4], merged[:, 4], iou_threshold=0.55, sigma=0.5,
        min_score=0.0001, method="gaussian", offset=0,
    )
    return detections[np.argsort(-detections[:, 4])].astype(np.float32, copy=False)


@torch.inference_mode()
def predict_cube(
    model,
    cube: np.ndarray,
    *,
    filename: Path,
    device: torch.device,
) -> list[np.ndarray]:
    """Frozen two-view/Soft-NMS/max300 prediction for one normalized cube."""

    original_h, original_w = cube.shape[:2]
    candidates: list[list[np.ndarray]] = [[] for _ in HOD26_CLASSES]
    for horizontal in (False, True):
        view = np.ascontiguousarray(cube[:, ::-1]) if horizontal else cube
        canvas, sx, sy, resized_w, resized_h = _resize_pad(view, 1280)
        tensor = (
            torch.from_numpy(canvas.transpose(2, 0, 1).copy())
            .unsqueeze(0)
            .to(device)
        )
        meta = {
            "filename": str(filename.resolve()),
            "ori_filename": filename.name,
            "ori_shape": (original_h, original_w, 16),
            "img_shape": (resized_h, resized_w, 16),
            "pad_shape": canvas.shape,
            "scale_factor": np.asarray((sx, sy, sx, sy), np.float32),
            "flip": False,
            "flip_direction": None,
            "img_norm_cfg": {
                "mean": np.zeros(16, np.float32),
                "std": np.ones(16, np.float32),
                "to_rgb": False,
            },
        }
        result = model(
            return_loss=False, rescale=False, img=[tensor], img_metas=[[meta]]
        )[0]
        for class_id, detections in enumerate(result):
            detections = np.asarray(detections, np.float32)
            if len(detections):
                detections = detections[detections[:, 4] >= 0.0001].copy()
            if len(detections):
                detections[:, :4] = _restore_boxes(
                    detections[:, :4],
                    sx,
                    sy,
                    original_w,
                    original_h,
                    horizontal,
                )
                candidates[class_id].append(detections)
        del tensor, result

    merged = [_gaussian_soft_nms(arrays) for arrays in candidates]
    ranked = [
        (float(row[4]), class_id, row)
        for class_id, detections in enumerate(merged)
        for row in detections
    ]
    ranked.sort(key=lambda item: -item[0])
    selected: list[list[np.ndarray]] = [[] for _ in HOD26_CLASSES]
    for _, class_id, row in ranked[:300]:
        selected[class_id].append(row)
    return [
        np.asarray(rows, dtype=np.float32).reshape(-1, 5)
        if rows else np.zeros((0, 5), np.float32)
        for rows in selected
    ]


@torch.inference_mode()
def infer(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path = "storage/cache/test_manifest.json",
    data_root: str | Path = "data/raw",
    stats_path: str | Path = "auto",
    device_name: str = "cuda:0",
    limit: int | None = None,
):
    config_path, checkpoint_path = Path(config_path), Path(checkpoint_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    samples = manifest["samples"][:limit] if limit else manifest["samples"]
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model, config, metadata = build_model(config_path, checkpoint_path, device)
    if str(stats_path) == "auto":
        stats_path = (
            "storage/v6/dataset/full_band_stats.json"
            if metadata.get("hod26_v6_full_data_stage")
            else "storage/v6/dataset/fold0_band_stats.json"
        )
    low, high = load_band_limits(stats_path)
    rows, started = [], time.perf_counter()
    root = Path(data_root)
    for image_index, sample in enumerate(samples):
        cube = _load_cube(root / sample["image"], low, high)
        detections_by_class = predict_cube(
            model,
            cube,
            filename=root / sample["image"],
            device=device,
        )
        image_rows = []
        for class_id, detections in enumerate(detections_by_class):
            for x1, y1, x2, y2, score in detections:
                image_rows.append(
                    {
                        "image_id": str(sample["image_id"]),
                        "class_id": class_id,
                        "confidence": float(score),
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2),
                    }
                )
        rows.extend(image_rows)
        if (image_index + 1) % 25 == 0:
            elapsed = time.perf_counter() - started
            print(f"[V6] {image_index+1}/{len(samples)} {elapsed/(image_index+1):.2f}s/image", flush=True)
    rows.sort(key=lambda row: (int(row["image_id"]), -row["confidence"]))
    output = Path(output_path)
    write_submission(rows, output, include_id=True)
    report = validate_submission(
        output, manifest["samples"], num_classes=len(HOD26_CLASSES), include_id=True
    )
    report.update(
        architecture="CoSpec-DINO ViT-L V6",
        single_model=True,
        prediction_ensemble=False,
        checkpoint=str(checkpoint_path.resolve()),
        checkpoint_sha256=_sha256(checkpoint_path),
        checkpoint_epoch=metadata.get("epoch"),
        checkpoint_attempted_updates=metadata.get("iter"),
        checkpoint_successful_updates=metadata.get("hod26_v6_successful_updates"),
        config=str(config_path.resolve()),
        config_sha256=_sha256(config_path),
        stats=str(Path(stats_path).resolve()),
        stats_sha256=_sha256(stats_path),
        inference_protocol="1280 identity+horizontal; class-wise Gaussian Soft-NMS iou=.55 sigma=.5; threshold=.0001; global max300",
        processed_images=len(samples),
        elapsed_seconds=time.perf_counter() - started,
        cuda_peak_memory_bytes=(int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None),
    )
    audit = output.with_suffix(output.suffix + ".audit.json")
    report["audit"] = str(audit.resolve())
    atomic_json_dump(report, audit)
    if not report["valid"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6/co_spec_dino_vitl_fold0.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="storage/cache/test_manifest.json")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument(
        "--stats",
        default="auto",
        help="auto selects Fold0/full normalization from checkpoint lineage",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    _, report = infer(
        config_path=args.config, checkpoint_path=args.checkpoint,
        output_path=args.output, manifest_path=args.manifest,
        data_root=args.data_root, stats_path=args.stats,
        device_name=args.device, limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
