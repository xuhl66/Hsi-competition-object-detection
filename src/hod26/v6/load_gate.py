"""Fail-closed public-bootstrap and ordinary-resume gate for V6."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch

from .constants import OFFICIAL_CHECKPOINT_SHA256, OFFICIAL_CODETR_COMMIT


REPO_ROOT = Path(__file__).resolve().parents[3]


class V6LoadGateError(RuntimeError):
    pass


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return (REPO_ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git(upstream: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={upstream}", "-C", str(upstream), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise V6LoadGateError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def verify_runtime_source_lock(path: str | Path) -> dict[str, Any]:
    lock_path = _resolve(path)
    if not lock_path.is_file():
        raise V6LoadGateError(f"Missing V6 runtime source lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("version") != "v6":
        raise V6LoadGateError("Unsupported V6 source-lock schema")
    files = lock.get("files")
    if not isinstance(files, Mapping) or not files:
        raise V6LoadGateError("V6 source lock has no files")
    for relative, expected in files.items():
        source = _resolve(relative)
        if not source.is_file():
            raise V6LoadGateError(f"Missing locked V6 source: {relative}")
        actual = sha256_file(source)
        if actual != expected:
            raise V6LoadGateError(
                f"V6 runtime source drift: {relative}; expected {expected}, got {actual}"
            )
    upstream = _resolve(lock["upstream"]["path"])
    commit = _git(upstream, "rev-parse", "HEAD").decode().strip()
    if commit != OFFICIAL_CODETR_COMMIT or commit != lock["upstream"]["commit"]:
        raise V6LoadGateError(f"Unexpected Co-DETR commit: {commit}")
    diff_sha = hashlib.sha256(_git(upstream, "diff", "--binary", "HEAD")).hexdigest()
    if diff_sha != lock["upstream"]["working_diff_sha256"]:
        raise V6LoadGateError("Co-DETR working diff differs from the V6 source lock")
    return {
        "source_lock": str(lock_path),
        "source_lock_sha256": sha256_file(lock_path),
        "locked_files": len(files),
        "upstream_commit": commit,
        "upstream_working_diff_sha256": diff_sha,
    }


def _build_schema(config_path: str | Path):
    import hod26.v6  # noqa: F401
    from mmcv import Config
    from mmcv.runner import build_optimizer
    from mmdet.models import build_detector

    config = Config.fromfile(str(_resolve(config_path)))
    disabled = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_detector(
                config.model,
                train_cfg=config.get("train_cfg"),
                test_cfg=config.get("test_cfg"),
            )
    finally:
        logging.disable(disabled)
    state = {
        name: (tuple(value.shape), str(value.dtype))
        for name, value in model.state_dict().items()
    }
    trainable = {
        name: (tuple(parameter.shape), str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = build_optimizer(model, config.optimizer)
    groups = optimizer.state_dict()["param_groups"]
    del optimizer, model
    gc.collect()
    return config, state, trainable, groups


def _require_tensor(
    value: Any, *, name: str, shape: tuple[int, ...], dtype: str
) -> None:
    if not torch.is_tensor(value):
        raise V6LoadGateError(f"Checkpoint value is not a tensor: {name}")
    if tuple(value.shape) != shape or str(value.dtype) != dtype:
        raise V6LoadGateError(
            f"Tensor schema mismatch {name}: {tuple(value.shape)}/{value.dtype} "
            f"!= {shape}/{dtype}"
        )
    if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise V6LoadGateError(f"Non-finite V6 tensor: {name}")


def _validate_primary_state(state_dict, schema, *, with_ema: bool) -> dict[str, int]:
    primary = set(schema)
    ema = {f"ema_{name.replace('.', '_')}": name for name in schema} if with_ema else {}
    expected = primary | set(ema)
    actual = set(state_dict)
    if expected != actual:
        raise V6LoadGateError(
            f"Strict V6 state mismatch; missing={sorted(expected-actual)[:8]}, "
            f"unexpected={sorted(actual-expected)[:8]}"
        )
    finite = 0
    for name, (shape, dtype) in schema.items():
        _require_tensor(state_dict[name], name=name, shape=shape, dtype=dtype)
        finite += int(state_dict[name].dtype.is_floating_point)
        if with_ema:
            ema_name = f"ema_{name.replace('.', '_')}"
            _require_tensor(state_dict[ema_name], name=ema_name, shape=shape, dtype=dtype)
            finite += int(state_dict[ema_name].dtype.is_floating_point)
    return {"primary_state_tensors": len(primary), "ema_state_tensors": len(ema), "finite_tensors": finite}


def _same(left, right) -> bool:
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return tuple(left) == tuple(right)
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=0)
        except (TypeError, ValueError):
            return False
    return left == right


def _validate_optimizer(saved, target_groups, trainable) -> dict[str, Any]:
    saved_groups, states = saved.get("param_groups"), saved.get("state")
    if not isinstance(saved_groups, list) or not isinstance(states, Mapping):
        raise V6LoadGateError("V6 resume checkpoint has no AdamW state")
    if len(saved_groups) != len(target_groups):
        raise V6LoadGateError("V6 optimizer group count changed")
    names_seen, ids_seen, steps = set(), set(), []
    moment_tensors = 0
    structural = ("group_name", "initial_lr", "lr_scale", "weight_decay", "betas", "eps")
    for index, (source, target) in enumerate(zip(saved_groups, target_groups, strict=True)):
        source_names, target_names = source.get("param_names"), target.get("param_names")
        if source_names != target_names:
            raise V6LoadGateError(f"V6 optimizer parameter order changed in group {index}")
        for field in structural:
            if field not in source or field not in target or not _same(source[field], target[field]):
                raise V6LoadGateError(f"V6 optimizer field changed: group {index} {field}")
        parameter_ids = source.get("params")
        if not isinstance(parameter_ids, list) or len(parameter_ids) != len(source_names):
            raise V6LoadGateError(f"Malformed V6 optimizer group {index}")
        for parameter_id, name in zip(parameter_ids, source_names, strict=True):
            if parameter_id in ids_seen or name in names_seen:
                raise V6LoadGateError("Duplicate V6 optimizer coordinate")
            ids_seen.add(parameter_id); names_seen.add(name)
            state = states.get(parameter_id)
            if not isinstance(state, Mapping):
                raise V6LoadGateError(f"Missing V6 AdamW state: {name}")
            for moment in ("exp_avg", "exp_avg_sq"):
                value = state.get(moment)
                shape, dtype = trainable[name]
                _require_tensor(
                    value, name=f"optimizer.{name}.{moment}",
                    shape=shape, dtype=dtype,
                )
                moment_tensors += 1
            step = state.get("step")
            step = float(step.item() if torch.is_tensor(step) else step)
            if step < 1 or not math.isclose(step, round(step), abs_tol=1e-6):
                raise V6LoadGateError(f"Invalid V6 AdamW step for {name}: {step}")
            steps.append(int(round(step)))
    if names_seen != set(trainable) or set(states) != ids_seen:
        raise V6LoadGateError("V6 optimizer/trainable state schema is incomplete")
    if max(steps) - min(steps) > 1:
        raise V6LoadGateError("V6 AdamW step families diverged by more than one update")
    return {
        "optimizer_parameter_names": len(names_seen),
        "optimizer_state_tensors": len(states),
        "finite_moment_tensors": moment_tensors,
        "optimizer_min_step": min(steps),
        "optimizer_max_step": max(steps),
    }


def _validate_scaler(meta: Mapping[str, Any]) -> dict[str, Any]:
    scaler = meta.get("fp16", {}).get("loss_scaler")
    if not isinstance(scaler, Mapping):
        raise V6LoadGateError("V6 resume checkpoint lost the AMP scaler")
    required = ("scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker")
    if any(field not in scaler for field in required):
        raise V6LoadGateError("V6 AMP scaler state is incomplete")
    scale = float(scaler["scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise V6LoadGateError("V6 AMP scaler is invalid")
    return {"scale": scale, "growth_tracker": int(scaler["_growth_tracker"])}


def validate(
    *,
    mode: str,
    checkpoint_path: str | Path,
    config_path: str | Path,
    source_lock: str | Path,
    provenance_path: str | Path | None = None,
) -> dict[str, Any]:
    runtime = verify_runtime_source_lock(source_lock)
    config, schema, trainable, groups = _build_schema(config_path)
    checkpoint = _resolve(checkpoint_path)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise V6LoadGateError(f"Missing V6 checkpoint: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload.get("meta"), Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise V6LoadGateError("Malformed V6 checkpoint payload")
    meta = payload["meta"]
    if meta.get("hod26_version") != "v6" or not meta.get("independent_lineage", meta.get("hod26_v6_independent_lineage")):
        raise V6LoadGateError("Checkpoint is not in the independent V6 lineage")
    report = {
        "gate": "passed",
        "mode": mode,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        **runtime,
    }
    if mode == "public_init":
        if "optimizer" in payload:
            raise V6LoadGateError("Public V6 initialization unexpectedly has optimizer state")
        report.update(_validate_primary_state(payload["state_dict"], schema, with_ema=False))
        provenance = _resolve(provenance_path or checkpoint.with_suffix(".pth.provenance.json"))
        if not provenance.is_file():
            raise V6LoadGateError("V6 public initialization provenance is missing")
        declared = json.loads(provenance.read_text(encoding="utf-8"))
        if (
            declared.get("output_sha256") != checkpoint_sha
            or declared.get("official_checkpoint_sha256") != OFFICIAL_CHECKPOINT_SHA256
            or declared.get("config_sha256") != sha256_file(_resolve(config_path))
            or declared.get("optimizer_state_inherited") is not False
        ):
            raise V6LoadGateError("V6 public initialization provenance is stale")
        report.update(
            provenance=str(provenance),
            provenance_sha256=sha256_file(provenance),
            optimizer="fresh",
            ema="fresh",
            fp16_scaler="fresh",
        )
    elif mode == "resume":
        report.update(_validate_primary_state(payload["state_dict"], schema, with_ema=True))
        if not isinstance(payload.get("optimizer"), Mapping):
            raise V6LoadGateError("V6 resume checkpoint has no optimizer")
        report.update(_validate_optimizer(payload["optimizer"], groups, trainable))
        report["fp16_scaler"] = _validate_scaler(meta)
        declared_provenance = meta.get("hod26_v6_initialization_provenance")
        if not isinstance(declared_provenance, str):
            raise V6LoadGateError("V6 resume lost initialization provenance")
        provenance = _resolve(declared_provenance)
        if (
            not provenance.is_file()
            or sha256_file(provenance) != meta.get("hod26_v6_initialization_provenance_sha256")
        ):
            raise V6LoadGateError("V6 resume initialization provenance changed")
        successful = int(meta.get("hod26_v6_successful_updates", -1))
        attempted = int(meta.get("iter", -1))
        if int(meta.get("hod26_v6_optimizer_min_step", -1)) != report["optimizer_min_step"]:
            raise V6LoadGateError("V6 optimizer minimum-step metadata changed")
        if (
            int(meta.get("hod26_v6_optimizer_max_step", -1)) != report["optimizer_max_step"]
            or int(meta.get("hod26_v6_optimizer_state_count", -1)) != report["optimizer_state_tensors"]
        ):
            raise V6LoadGateError("V6 optimizer state metadata changed")
        if successful != report["optimizer_max_step"] or attempted < successful:
            raise V6LoadGateError("V6 attempted/successful update metadata is inconsistent")
        if meta.get("hod26_v6_runtime_source_lock_sha256") != runtime["source_lock_sha256"]:
            raise V6LoadGateError("V6 resume source-lock lineage changed")
        report.update(
            epoch=int(meta.get("epoch", -1)),
            attempted_updates=attempted,
            successful_updates=successful,
            amp_skipped_updates=attempted - successful,
            ema_age=int(meta.get("hod26_v6_ema_age", -1)),
        )
        if report["epoch"] < 1 or report["ema_age"] != successful:
            raise V6LoadGateError("V6 resume epoch/EMA lineage is invalid")
    else:
        raise V6LoadGateError(f"Unknown V6 gate mode: {mode}")
    del payload, config
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("public_init", "resume"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/v6/co_spec_dino_vitl_fold0.py")
    parser.add_argument("--source-lock", default="configs/v6/runtime_source_lock.json")
    parser.add_argument("--provenance")
    args = parser.parse_args()
    report = validate(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        source_lock=args.source_lock,
        provenance_path=args.provenance,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
