"""Native single-view inference for the converged V6 checkpoint.

This deliberately does not modify the source-locked champion inference module.
It uses one identity view and preserves the detector's native ``test_cfg``
output without cross-view fusion, external NMS, or score calibration.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from hod26.config import atomic_json_dump
from hod26.submission import validate_submission, write_submission

from .constants import HOD26_CLASSES
from .data import load_band_limits
from .infer import (
    _load_cube,
    _resize_pad,
    _restore_boxes,
    _sha256,
    build_model,
)


def _restore_native_result(
    result,
    *,
    sx: float,
    sy: float,
    width: int,
    height: int,
) -> list[np.ndarray]:
    """Validate and restore one model-native, globally capped result."""

    restored: list[np.ndarray] = []
    total = 0
    for detections in result:
        value = np.asarray(detections, np.float32).reshape(-1, 5).copy()
        if len(value):
            valid = (
                np.isfinite(value).all(1)
                & (value[:, 2] > value[:, 0])
                & (value[:, 3] > value[:, 1])
                & (value[:, 4] >= 0.0)
                & (value[:, 4] <= 1.0)
            )
            value = value[valid]
            value[:, :4] = _restore_boxes(
                value[:, :4], sx, sy, width, height, False
            )
        total += len(value)
        restored.append(value)
    if len(restored) != len(HOD26_CLASSES):
        raise RuntimeError(
            f"Native V6 output has {len(restored)} classes, expected "
            f"{len(HOD26_CLASSES)}"
        )
    if total > 300:
        raise RuntimeError(
            f"Native V6 test_cfg returned {total} detections; expected global max300"
        )
    return restored


@torch.inference_mode()
def predict_cube_native(
    model,
    cube: np.ndarray,
    *,
    filename: Path,
    device: torch.device,
) -> list[np.ndarray]:
    """Run exactly one identity view through the model-native test config."""

    original_h, original_w = cube.shape[:2]
    canvas, sx, sy, resized_w, resized_h = _resize_pad(cube, 1280)
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
    restored = _restore_native_result(
        result,
        sx=sx,
        sy=sy,
        width=original_w,
        height=original_h,
    )
    del tensor, result
    return restored


@torch.inference_mode()
def infer_native(
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
    model, _, metadata = build_model(config_path, checkpoint_path, device)
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
        filename = root / sample["image"]
        cube = _load_cube(filename, low, high)
        detections_by_class = predict_cube_native(
            model, cube, filename=filename, device=device
        )
        for class_id, detections in enumerate(detections_by_class):
            for x1, y1, x2, y2, score in detections:
                rows.append(
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
        if (image_index + 1) % 25 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[V6 raw] {image_index + 1}/{len(samples)} "
                f"{elapsed / (image_index + 1):.2f}s/image",
                flush=True,
            )
    rows.sort(key=lambda row: (int(row["image_id"]), -row["confidence"]))
    output = Path(output_path)
    write_submission(rows, output, include_id=True)
    report = validate_submission(
        output,
        manifest["samples"],
        num_classes=len(HOD26_CLASSES),
        include_id=True,
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
        inference_protocol=(
            "1280 identity single-view; model-native V6 test_cfg; "
            "no cross-view merge; no external score calibration"
        ),
        processed_images=len(samples),
        elapsed_seconds=time.perf_counter() - started,
        cuda_peak_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        inference_source=str(Path(__file__).resolve()),
        inference_source_sha256=_sha256(__file__),
    )
    audit = output.with_suffix(output.suffix + ".audit.json")
    report["audit"] = str(audit.resolve())
    atomic_json_dump(report, audit)
    if not report["valid"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    del model
    gc.collect()
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6/co_spec_dino_vitl_fold0.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="storage/cache/test_manifest.json")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--stats", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    _, report = infer_native(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        manifest_path=args.manifest,
        data_root=args.data_root,
        stats_path=args.stats,
        device_name=args.device,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
