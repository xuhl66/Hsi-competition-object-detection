from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F

from .config import (
    distributed_context,
    load_config,
    resolve_path,
    seed_everything,
)
from .evaluation import evaluate_model
from .model import (
    build_model,
    ensure_rgb_pretraining,
    unwrap_model,
    update_ema,
)
from .prepare import prepare
from .runtime import build_train_loaders, load_data_assets


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _move_training_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for key in ("img", "cls", "bboxes", "batch_idx"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    base_learning_rate: float,
    min_ratio: float,
) -> float:
    if global_step < warmup_steps:
        factor = 0.001 + 0.999 * global_step / max(warmup_steps, 1)
    else:
        progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        factor = min_ratio + (1.0 - min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )
    for group in optimizer.param_groups:
        group["lr"] = (
            base_learning_rate * float(group.get("lr_multiplier", 1.0)) * factor
        )
    return base_learning_rate * factor


def _checkpoint_payload(
    model: torch.nn.Module,
    ema: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    best_metric: float,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    bare = unwrap_model(model)
    return {
        "format_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "model": bare.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "model_summary": bare.summary(),
        "metrics": metrics,
        "created_at": datetime.now().astimezone().isoformat(),
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def train(args: argparse.Namespace) -> Path | None:
    config = load_config(args.config)
    rank, local_rank, world_size = distributed_context()
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda":
            torch.cuda.set_device(device)
    primary = rank in (-1, 0)
    seed = int(config["project"]["seed"]) + max(rank, 0)
    seed_everything(seed)

    if args.no_pretrained:
        config["model"]["pretrained_rgb"] = None
    elif config["model"].get("pretrained_rgb"):
        config["model"]["pretrained_rgb"] = str(
            resolve_path(config["model"]["pretrained_rgb"])
        )
        if primary:
            ensure_rgb_pretraining(config["model"]["pretrained_rgb"])
    if distributed:
        dist.barrier()

    required = [
        resolve_path(config["data"][key])
        for key in ("manifest", "test_manifest", "stats", "folds")
    ]
    if primary and not all(path.exists() for path in required):
        prepare(config)
    if distributed:
        dist.barrier()
    assets = load_data_assets(config)
    train_loader, validation_loader, sampler = build_train_loaders(
        config,
        assets,
        rank=max(rank, 0),
        world_size=world_size,
        full_data=args.full_data,
    )
    micro_batch = int(config["train"]["batch_size_per_gpu"]) * world_size
    target_global_batch = int(
        config["train"].get("global_batch_size", micro_batch)
    )
    if target_global_batch % micro_batch:
        raise ValueError(
            "train.global_batch_size must be divisible by "
            f"batch_size_per_gpu * world_size ({micro_batch})"
        )
    accumulation = max(1, target_global_batch // micro_batch)

    model = build_model(config, assets["classes"]).to(device)
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    start_epoch = 0
    global_step = 0
    best_metric = -1.0
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_state["model"], strict=True)
        start_epoch = int(resume_state["epoch"]) + 1
        global_step = int(resume_state["global_step"])
        best_metric = float(resume_state.get("best_metric", -1.0))

    bare_model = model
    parameter_groups = bare_model.parameter_groups(
        learning_rate=float(config["train"]["lr"]),
        rgb_multiplier=float(config["train"]["rgb_lr_multiplier"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    optimizer_name = str(config["train"]["optimizer"]).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=float(config["train"]["lr"]),
            betas=(0.9, 0.999),
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=float(config["train"]["lr"]),
            momentum=0.937,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state["scaler"])

    ema = copy.deepcopy(bare_model).eval() if primary else None
    if ema is not None:
        for parameter in ema.parameters():
            parameter.requires_grad_(False)
        if resume_state and resume_state.get("ema"):
            ema.load_state_dict(resume_state["ema"], strict=True)

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    if bool(config["train"].get("compile", False)):
        model = torch.compile(model)

    storage = resolve_path(config["project"]["storage"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_name = "full" if args.full_data else f"fold{config['data']['fold']}"
    run_name = f"{config['project']['run_name']}_{fold_name}_{timestamp}"
    run_directory = storage / "runs" / run_name
    checkpoint_directory = storage / "checkpoints" / run_name
    log_path = run_directory / "metrics.jsonl"
    if primary:
        run_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        (run_directory / "run_manifest.json").write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "config": config,
                    "model": bare_model.summary(),
                    "world_size": world_size,
                    "gradient_accumulation": accumulation,
                    "effective_global_batch": micro_batch * accumulation,
                    "device": str(device),
                    "train_images": len(train_loader.dataset),
                    "validation_images": len(validation_loader.dataset),
                    "external_pretraining_declaration": config["model"].get(
                        "declare_pretraining"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(bare_model.summary(), ensure_ascii=False, indent=2))
        print(f"Run directory: {run_directory}")

    epochs = int(config["train"]["epochs"])
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(config["train"]["warmup_epochs"]) * steps_per_epoch
    min_lr_ratio = float(config["train"]["min_lr_ratio"])
    widths = [int(value) for value in config["train"]["multi_scale"]["widths"]]
    use_multi_scale = bool(config["train"]["multi_scale"]["enabled"])
    ema_decay = float(config["train"]["ema_decay"])
    ema_tau = float(config["train"].get("ema_tau", 2000.0))
    clip_norm = float(config["train"]["clip_grad_norm"])
    patience = int(config["train"]["patience"])
    epochs_without_improvement = 0
    optimizer.zero_grad(set_to_none=True)
    stop_training = False

    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = torch.zeros(3, device=device)
        epoch_batches = 0
        started = time.time()

        for batch_index, batch in enumerate(train_loader):
            batch = _move_training_batch(batch, device)
            if use_multi_scale:
                width = widths[(global_step + batch_index) % len(widths)]
                height = width // 2
                if batch["img"].shape[-2:] != (height, width):
                    batch["img"] = F.interpolate(
                        batch["img"],
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
            synchronize = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            context = (
                model.no_sync()
                if distributed and not synchronize
                else nullcontext()
            )
            with context:
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss, loss_items = model(batch)
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()

            epoch_loss += loss_items.detach()
            epoch_batches += 1
            if synchronize:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                _set_learning_rate(
                    optimizer,
                    global_step,
                    total_steps,
                    warmup_steps,
                    float(config["train"]["lr"]),
                    min_lr_ratio,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if ema is not None:
                    update_ema(
                        ema,
                        model,
                        ema_decay,
                        updates=global_step,
                        tau=ema_tau,
                    )

                if primary and global_step % 20 == 0:
                    averaged = epoch_loss / max(epoch_batches, 1)
                    print(
                        f"epoch={epoch + 1}/{epochs} step={global_step} "
                        f"box={averaged[0]:.4f} cls+aux={averaged[1]:.4f} "
                        f"dfl={averaged[2]:.4f} lr={optimizer.param_groups[1]['lr']:.3e}"
                    )
                if args.smoke_steps and global_step >= args.smoke_steps:
                    break

        if distributed:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            count = torch.tensor(epoch_batches, device=device, dtype=torch.float32)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            averaged_loss = epoch_loss / max(float(count), 1.0)
        else:
            averaged_loss = epoch_loss / max(epoch_batches, 1)

        should_evaluate = (
            not args.full_data
            and len(validation_loader.dataset) > 0
            and (
                (epoch + 1) % int(config["train"]["eval_every"]) == 0
                or epoch + 1 == epochs
                or bool(args.smoke_steps)
            )
        )
        metrics = None
        improved = False
        if should_evaluate:
            if distributed:
                dist.barrier()
            if primary and ema is not None:
                evaluation_loader = (
                    itertools.islice(validation_loader, 2)
                    if args.smoke_steps
                    else validation_loader
                )
                metrics = evaluate_model(
                    ema,
                    evaluation_loader,
                    assets["classes"],
                    device,
                    confidence=float(config["inference"]["confidence"]),
                    nms_iou=float(config["inference"]["nms_iou"]),
                    max_detections=int(config["inference"]["max_detections"]),
                    amp=amp_enabled,
                )
                current = float(metrics["map_50_95"])
                improved = current > best_metric
                if improved:
                    best_metric = current
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += int(config["train"]["eval_every"])
                print(json.dumps(metrics, ensure_ascii=False, indent=2))
            if distributed:
                dist.barrier()

        if primary:
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "loss_box": float(averaged_loss[0]),
                "loss_cls_aux": float(averaged_loss[1]),
                "loss_dfl": float(averaged_loss[2]),
                "best_map_50_95": best_metric,
                "metrics": metrics,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            _append_jsonl(log_path, record)
            payload = _checkpoint_payload(
                model,
                ema,
                optimizer,
                scaler,
                config,
                epoch,
                global_step,
                best_metric,
                metrics,
            )
            _atomic_torch_save(payload, checkpoint_directory / "last.pt")
            if improved:
                _atomic_torch_save(payload, checkpoint_directory / "best.pt")
            if (epoch + 1) % int(config["train"]["save_every"]) == 0:
                _atomic_torch_save(
                    payload, checkpoint_directory / f"epoch_{epoch + 1:03d}.pt"
                )
            stop_training = (
                epochs_without_improvement >= patience and not args.full_data
            )

        if distributed:
            stop_tensor = torch.tensor(
                int(stop_training if primary else 0), device=device, dtype=torch.int32
            )
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())
        if args.smoke_steps and global_step >= args.smoke_steps:
            stop_training = True
        if stop_training:
            break

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if not primary:
        return None
    if args.full_data:
        return checkpoint_directory / "last.pt"
    best_path = checkpoint_directory / "best.pt"
    return best_path if best_path.exists() else checkpoint_directory / "last.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the final HOD26 detector")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
