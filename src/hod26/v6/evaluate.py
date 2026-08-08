"""Evaluate one V6 checkpoint with the frozen champion inference protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hod26.config import atomic_json_dump

from .data import apply_fixed_sensor_shift, load_band_limits
from .infer import _load_cube, build_model, predict_cube


MODES = ("original", "exposure_low", "spectral_slope", "band_response", "noise_blur")


def evaluate_modes(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    annotation_path: str | Path,
    image_root: str | Path,
    stats_path: str | Path,
    output_dir: str | Path,
    modes: list[str],
    device_name: str,
) -> dict[str, dict]:
    import hod26.v6  # noqa: F401
    from mmcv import Config
    from mmdet.datasets import build_dataset

    unknown = set(modes) - set(MODES)
    if unknown or not modes:
        raise ValueError(f"Invalid V6 evaluation modes: {sorted(unknown)}")
    config_path, checkpoint_path = Path(config_path), Path(checkpoint_path)
    config = Config.fromfile(str(config_path.resolve()))
    dataset = build_dataset(config.data.val)
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    if [int(info["id"]) for info in annotation["images"]] != [int(value) for value in dataset.img_ids]:
        raise RuntimeError("V6 evaluation annotation order differs from the dataset")
    low, high = load_band_limits(stats_path)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, _, metadata = build_model(config_path, checkpoint_path, device)
    root, destination = Path(image_root), Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    reports = {}
    for mode in modes:
        started, predictions = time.perf_counter(), []
        for index, info in enumerate(annotation["images"]):
            filename = root / info["file_name"]
            cube = _load_cube(filename, low, high)
            if mode != "original":
                cube = apply_fixed_sensor_shift(cube, mode, image_id=int(info["id"]))
            predictions.append(
                predict_cube(model, cube, filename=filename, device=device)
            )
            if (index + 1) % 50 == 0:
                print(
                    f"[V6:{mode}] {index + 1}/{len(annotation['images'])}",
                    flush=True,
                )
        metrics = dataset.evaluate(
            predictions,
            metric="bbox",
            classwise=True,
            proposal_nums=(1, 10, 100),
        )
        report = {
            "mode": mode,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": metadata.get("epoch"),
            "checkpoint_attempted_updates": metadata.get("iter"),
            "checkpoint_successful_updates": metadata.get("hod26_v6_successful_updates"),
            "inference_protocol": (
                "1280 identity+horizontal; class-wise Gaussian Soft-NMS "
                "iou=.55 sigma=.5; threshold=.0001; global max300"
            ),
            "images": len(predictions),
            "elapsed_seconds": time.perf_counter() - started,
            "metrics": {name: float(value) for name, value in metrics.items()},
        }
        atomic_json_dump(report, destination / f"{mode}.metrics.json")
        reports[mode] = report
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6/co_spec_dino_vitl_fold0.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotation", default="storage/v2/dataset/fold0/val.json")
    parser.add_argument("--image-root", default="data/raw/data_train/data_train/VIS")
    parser.add_argument("--stats", default="storage/v6/dataset/fold0_band_stats.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate_modes(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        annotation_path=args.annotation,
        image_root=args.image_root,
        stats_path=args.stats,
        output_dir=args.output_dir,
        modes=args.modes,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
