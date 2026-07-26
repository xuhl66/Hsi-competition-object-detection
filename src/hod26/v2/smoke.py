from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-step V2 pretrained/data/GPU memory verification"
    )
    parser.add_argument(
        "--config",
        default="configs/v2/deim_dfine_x_hsi_pilot.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="storage/pretrained/v2/deim_dfine_x_object365.pth",
    )
    parser.add_argument(
        "--upstream",
        default="storage/upstream/DEIM",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=11)
    parser.add_argument("--width", type=int, default=1024)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    upstream = Path(args.upstream).resolve()
    sys.path.insert(0, str(upstream))
    from hod26.v2.compat import install_deim_compatibility

    install_deim_compatibility()
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver import TASKS
    from hod26.v2.extensions import HSIDetSolver

    TASKS["hsi_detection"] = HSIDetSolver
    train_update = {"total_batch_size": args.batch_size}
    if args.width:
        train_update["collate_fn"] = {"widths": [args.width]}
    config = YAMLConfig(
        args.config,
        device=args.device,
        tuning=str(Path(args.checkpoint).resolve()),
        use_amp=True,
        output_dir="storage/v2/smoke",
        train_dataloader=train_update,
    )
    solver = HSIDetSolver(config)
    solver.train()
    samples, targets = next(iter(solver.train_dataloader))
    samples = samples.to(solver.device)
    targets = [
        {key: value.to(solver.device) for key, value in target.items()}
        for target in targets
    ]
    if solver.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(solver.device)
        torch.cuda.synchronize(solver.device)
    started = time.perf_counter()
    solver.optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=solver.device.type,
        dtype=torch.float16,
        enabled=solver.scaler is not None,
    ):
        outputs = solver.model(samples, targets=targets)
    with torch.autocast(device_type=solver.device.type, enabled=False):
        losses = solver.criterion(
            outputs,
            targets,
            epoch=0,
            step=0,
            global_step=0,
            epoch_step=len(solver.train_dataloader),
        )
    loss = sum(losses.values())
    if solver.scaler is not None:
        solver.scaler.scale(loss).backward()
        solver.scaler.unscale_(solver.optimizer)
        solver.scaler.step(solver.optimizer)
        solver.scaler.update()
    else:
        loss.backward()
        solver.optimizer.step()
    if solver.device.type == "cuda":
        torch.cuda.synchronize(solver.device)
    report = {
        "valid": bool(torch.isfinite(loss)),
        "input_shape": list(samples.shape),
        "loss": float(loss.detach()),
        "losses": {
            key: float(value.detach()) for key, value in losses.items()
        },
        "seconds": time.perf_counter() - started,
        "peak_gpu_gib": (
            torch.cuda.max_memory_allocated(solver.device) / 2**30
            if solver.device.type == "cuda"
            else 0.0
        ),
        "model_parameters": sum(
            parameter.numel() for parameter in solver.model.parameters()
        ),
    }
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
