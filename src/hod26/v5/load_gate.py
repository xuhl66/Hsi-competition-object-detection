"""Fail-closed state-loading checks for the active HOD26 V5 lineage.

This module never mutates a checkpoint or starts training.  It validates the
runtime source lock, the complete model/EMA schema, optimizer parameter order
and moments, FP16 scaler policy, and attempted-versus-successful update
continuity before a checkpoint is allowed into the training process.
"""

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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/v5/co_dino_vitl_fdr_salience_fold0.py"
DEFAULT_CONTRACT = REPO_ROOT / "configs/v5/state_load_contract.json"
STRUCTURAL_OPTIMIZER_FIELDS = (
    "group_name",
    "initial_lr",
    "lr_scale",
    "weight_decay",
    "betas",
    "eps",
    "amsgrad",
    "maximize",
)


class StateLoadGateError(RuntimeError):
    """Raised when a checkpoint cannot be proven safe to load."""


@dataclass(frozen=True)
class RuntimeSchema:
    model: dict[str, tuple[tuple[int, ...], str]]
    trainable_shapes: dict[str, tuple[int, ...]]
    optimizer_groups: list[dict[str, Any]]


@dataclass(frozen=True)
class ValidationResult:
    report: dict[str, Any]
    steps: dict[str, int]
    scaler: dict[str, Any] | None


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_contract(path: str | Path) -> dict[str, Any]:
    contract_path = _resolve(path)
    if not contract_path.is_file():
        raise StateLoadGateError(f"Missing state-load contract: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1:
        raise StateLoadGateError("Unsupported state-load contract schema")
    version = contract.get("hod26_version")
    if not isinstance(version, str) or not version:
        raise StateLoadGateError("State-load contract has no HOD26 version")
    optimizer = contract["optimizer"]
    if int(optimizer["parent_attempted_iterations"]) - int(
        optimizer["parent_successful_steps"]
    ) != int(optimizer["parent_amp_skipped_steps"]):
        raise StateLoadGateError("Parent attempted/successful/skip contract is invalid")
    if int(optimizer["bootstrap_stateful_states"]) + int(
        optimizer["fresh_states"]
    ) != int(optimizer["target_trainable_tensors"]):
        raise StateLoadGateError("Bootstrap inherited/fresh tensor totals are invalid")
    scaler = contract["fp16_scaler"]
    if not scaler["parent_was_serialized"] or float(scaler["parent_scale"]) <= 0:
        raise StateLoadGateError("Parent FP16 scaler declaration is invalid")
    return contract


def _run_git(upstream: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={upstream}",
            "-C",
            str(upstream),
            *arguments,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode(errors="replace").strip()
        raise StateLoadGateError(
            f"Unable to verify upstream Co-DETR ({' '.join(arguments)}): {error}"
        )
    return completed.stdout


def validate_runtime_sources(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Verify every runtime-critical source plus the pinned upstream tree."""

    checked: dict[str, str] = {}
    for relative, expected in contract["runtime_source_lock"].items():
        path = _resolve(relative)
        if not path.is_file():
            raise StateLoadGateError(f"Missing locked runtime source: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise StateLoadGateError(
                f"Runtime source drift: {relative}; expected {expected}, got {actual}"
            )
        checked[relative] = actual

    upstream_contract = contract["upstream"]
    upstream = _resolve(upstream_contract["path"])
    commit = _run_git(upstream, "rev-parse", "HEAD").decode().strip()
    if commit != upstream_contract["commit"]:
        raise StateLoadGateError(
            f"Unexpected Co-DETR commit: expected "
            f"{upstream_contract['commit']}, got {commit}"
        )
    diff_sha256 = hashlib.sha256(
        _run_git(upstream, "diff", "--binary", "HEAD")
    ).hexdigest()
    if diff_sha256 != upstream_contract["working_diff_sha256"]:
        raise StateLoadGateError(
            "Co-DETR working diff does not match the declared compatibility patch"
        )
    return {
        "locked_source_files": len(checked),
        "upstream_commit": commit,
        "upstream_working_diff_sha256": diff_sha256,
    }


def build_runtime_schema(config_path: str | Path) -> RuntimeSchema:
    """Build only enough of the current graph to capture strict schemas."""

    # Imports are intentionally local so the source-only gate remains cheap.
    import hod26.v5  # noqa: F401
    from mmcv import Config
    from mmcv.runner import build_optimizer
    from mmdet.models import build_detector

    config = Config.fromfile(str(_resolve(config_path)))
    previous_logging_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_detector(
                config.model,
                train_cfg=config.get("train_cfg"),
                test_cfg=config.get("test_cfg"),
            )
    finally:
        logging.disable(previous_logging_disable)
    model_schema = {
        name: (tuple(value.shape), str(value.dtype))
        for name, value in model.state_dict().items()
    }
    trainable_shapes = {
        name: tuple(parameter.shape)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = build_optimizer(model, config.optimizer)
    optimizer_groups = optimizer.state_dict()["param_groups"]
    del optimizer, model, config
    gc.collect()
    return RuntimeSchema(
        model=model_schema,
        trainable_shapes=trainable_shapes,
        optimizer_groups=optimizer_groups,
    )


def _ema_name(name: str) -> str:
    return f"ema_{name.replace('.', '_')}"


def _require_finite(tensor: torch.Tensor, label: str) -> None:
    if tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all()):
        raise StateLoadGateError(f"Non-finite tensor in checkpoint: {label}")


def _validate_model_state(
    state_dict: Mapping[str, Any],
    schema: RuntimeSchema,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    expected_primary = set(schema.model)
    ema_names = {_ema_name(name): name for name in schema.model}
    if len(ema_names) != len(schema.model):
        raise StateLoadGateError("EMA buffer-name collision in current model schema")
    expected = expected_primary | set(ema_names)
    actual = set(state_dict)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise StateLoadGateError(
            "Strict model/EMA schema mismatch; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    finite_tensors = 0
    for name, (shape, dtype) in schema.model.items():
        for checkpoint_name in (name, _ema_name(name)):
            value = state_dict[checkpoint_name]
            if not torch.is_tensor(value):
                raise StateLoadGateError(
                    f"Checkpoint entry is not a tensor: {checkpoint_name}"
                )
            if tuple(value.shape) != shape or str(value.dtype) != dtype:
                raise StateLoadGateError(
                    f"Model state mismatch for {checkpoint_name}: "
                    f"expected {shape}/{dtype}, got "
                    f"{tuple(value.shape)}/{value.dtype}"
                )
            _require_finite(value, checkpoint_name)
            if value.dtype.is_floating_point:
                finite_tensors += 1

    expected_model = contract["model"]
    if len(schema.model) != int(expected_model["state_tensors"]):
        raise StateLoadGateError("Current model tensor count violates contract")
    if len(ema_names) != int(expected_model["ema_pairs"]):
        raise StateLoadGateError("Current EMA pair count violates contract")
    return {
        "model_state_tensors": len(schema.model),
        "ema_pairs": len(ema_names),
        "finite_model_and_ema_tensors": finite_tensors,
    }


def _normalize(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return tuple(_normalize(item) for item in value)
    return value


def _same_optimizer_value(left: Any, right: Any) -> bool:
    left = _normalize(left)
    right = _normalize(right)
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=0)
        except (TypeError, ValueError):
            return False
    return left == right


def _step_value(state: Mapping[str, Any], name: str) -> int:
    if "step" not in state:
        raise StateLoadGateError(f"AdamW state has no step: {name}")
    value = state["step"]
    if torch.is_tensor(value):
        if value.numel() != 1 or not bool(torch.isfinite(value).all()):
            raise StateLoadGateError(f"Invalid AdamW step tensor: {name}")
        number = float(value.item())
    else:
        number = float(value)
    rounded = round(number)
    if number < 0 or not math.isclose(number, rounded, abs_tol=1e-6):
        raise StateLoadGateError(f"Non-integral AdamW step for {name}: {number}")
    return int(rounded)


def _validate_optimizer_groups(
    saved_optimizer: Mapping[str, Any],
    schema: RuntimeSchema,
) -> tuple[dict[str, Mapping[str, Any] | None], list[str]]:
    saved_groups = saved_optimizer.get("param_groups")
    if not isinstance(saved_groups, list):
        raise StateLoadGateError("Checkpoint has no optimizer param_groups")
    if len(saved_groups) != len(schema.optimizer_groups):
        raise StateLoadGateError(
            "Optimizer group-count mismatch: "
            f"{len(saved_groups)} != {len(schema.optimizer_groups)}"
        )

    named_states: dict[str, Mapping[str, Any] | None] = {}
    ordered_names: list[str] = []
    known_ids: set[Any] = set()
    saved_states = saved_optimizer.get("state")
    if not isinstance(saved_states, Mapping):
        raise StateLoadGateError("Checkpoint has no optimizer state mapping")

    for index, (saved, target) in enumerate(
        zip(saved_groups, schema.optimizer_groups, strict=True)
    ):
        saved_names = saved.get("param_names")
        target_names = target.get("param_names")
        if not isinstance(saved_names, list) or not isinstance(target_names, list):
            raise StateLoadGateError(f"Optimizer group {index} is missing param_names")
        if saved_names != target_names:
            mismatch = next(
                (
                    position
                    for position, pair in enumerate(
                        zip(saved_names, target_names, strict=False)
                    )
                    if pair[0] != pair[1]
                ),
                min(len(saved_names), len(target_names)),
            )
            raise StateLoadGateError(
                f"Optimizer parameter order mismatch in group {index} "
                f"at position {mismatch}"
            )
        saved_ids = saved.get("params")
        if not isinstance(saved_ids, list) or len(saved_ids) != len(saved_names):
            raise StateLoadGateError(f"Malformed optimizer group {index}")
        for field in STRUCTURAL_OPTIMIZER_FIELDS:
            if field not in saved or field not in target:
                raise StateLoadGateError(
                    f"Optimizer group {index} is missing structural field {field}"
                )
            if not _same_optimizer_value(saved[field], target[field]):
                raise StateLoadGateError(
                    f"Optimizer structural field changed in group {index}: "
                    f"{field}={saved[field]!r}, expected {target[field]!r}"
                )
        for parameter_id, name in zip(saved_ids, saved_names, strict=True):
            if parameter_id in known_ids:
                raise StateLoadGateError(
                    f"Duplicate optimizer parameter id: {parameter_id}"
                )
            if name in named_states:
                raise StateLoadGateError(f"Duplicate optimizer parameter name: {name}")
            known_ids.add(parameter_id)
            named_states[name] = saved_states.get(parameter_id)
            ordered_names.append(name)

    unknown_state_ids = sorted(set(saved_states) - known_ids, key=str)
    if unknown_state_ids:
        raise StateLoadGateError(
            f"Optimizer contains states for unknown ids: {unknown_state_ids[:5]}"
        )
    if set(ordered_names) != set(schema.trainable_shapes):
        missing = sorted(set(schema.trainable_shapes) - set(ordered_names))
        extra = sorted(set(ordered_names) - set(schema.trainable_shapes))
        raise StateLoadGateError(
            f"Optimizer/trainable schema mismatch; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return named_states, ordered_names


def _validate_optimizer_state(
    saved_optimizer: Mapping[str, Any],
    schema: RuntimeSchema,
    contract: Mapping[str, Any],
    *,
    mode: str,
    meta: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    named_states, ordered_names = _validate_optimizer_groups(saved_optimizer, schema)
    optimizer_contract = contract["optimizer"]
    if len(ordered_names) != int(optimizer_contract["target_trainable_tensors"]):
        raise StateLoadGateError("Target trainable tensor count violates contract")

    steps: dict[str, int] = {}
    missing_states: list[str] = []
    finite_moment_tensors = 0
    for name in ordered_names:
        state = named_states[name]
        if state is None:
            missing_states.append(name)
            continue
        shape = schema.trainable_shapes[name]
        for moment in ("exp_avg", "exp_avg_sq"):
            value = state.get(moment)
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise StateLoadGateError(f"AdamW {moment} shape mismatch for {name}")
            _require_finite(value, f"optimizer.{name}.{moment}")
            finite_moment_tensors += 1
        steps[name] = _step_value(state, name)

    fresh_prefixes = tuple(optimizer_contract["fresh_prefixes"])
    expected_fresh_count = int(optimizer_contract["fresh_states"])
    if mode == "bootstrap":
        if len(steps) != int(optimizer_contract["bootstrap_stateful_states"]):
            raise StateLoadGateError("Bootstrap optimizer state count mismatch")
        if len(missing_states) != expected_fresh_count or any(
            not name.startswith(fresh_prefixes) for name in missing_states
        ):
            raise StateLoadGateError(
                "Bootstrap fresh optimizer scope violates the declared prefixes"
            )
        declared_fresh = meta.get("fresh_optimizer_names")
        if not isinstance(declared_fresh, list) or set(declared_fresh) != set(
            missing_states
        ):
            raise StateLoadGateError(
                "Bootstrap metadata does not enumerate the exact fresh states"
            )
        expected_parent_step = int(optimizer_contract["parent_successful_steps"])
        if set(steps.values()) != {expected_parent_step}:
            raise StateLoadGateError(
                "Bootstrap inherited AdamW steps are not parent-continuous"
            )
        successful_local = 0
        skipped_local = 0
        fresh_names = set(missing_states)
    elif mode == "resume":
        if missing_states:
            raise StateLoadGateError(
                f"Resume checkpoint has missing optimizer states: "
                f"{missing_states[:5]}"
            )
        declared_fresh = meta.get("fresh_optimizer_names")
        if not isinstance(declared_fresh, list):
            raise StateLoadGateError("Resume metadata lost fresh_optimizer_names")
        fresh_names = set(declared_fresh)
        if len(fresh_names) != expected_fresh_count or any(
            not name.startswith(fresh_prefixes) for name in fresh_names
        ):
            raise StateLoadGateError("Resume fresh-state lineage is invalid")
        inherited_names = set(steps) - fresh_names
        fresh_step_values = {steps[name] for name in fresh_names}
        inherited_step_values = {steps[name] for name in inherited_names}
        if len(fresh_step_values) != 1 or len(inherited_step_values) != 1:
            raise StateLoadGateError(
                "AdamW steps are not uniform within fresh/inherited families"
            )
        successful_local = next(iter(fresh_step_values))
        inherited_step = next(iter(inherited_step_values))
        parent_step = int(optimizer_contract["parent_successful_steps"])
        if inherited_step != parent_step + successful_local:
            raise StateLoadGateError(
                "Inherited AdamW step does not equal parent step + "
                "successful child-lineage updates"
            )
        attempted_local = int(meta.get("iter", -1))
        skipped_local = attempted_local - successful_local
        if successful_local < 0 or skipped_local < 0:
            raise StateLoadGateError(
                "Successful optimizer updates exceed attempted runner iterations"
            )
    else:
        raise StateLoadGateError(f"Unknown checkpoint validation mode: {mode}")

    semantic_clones = [
        name
        for name in steps
        if name.startswith("neck.salience_high.") and name not in fresh_names
    ]
    if len(semantic_clones) != int(
        optimizer_contract["semantic_salience_clone_states"]
    ):
        raise StateLoadGateError("Semantic salience-clone state count changed")

    return (
        {
            "optimizer_parameter_order_verified": len(ordered_names),
            "optimizer_states": len(steps),
            "fresh_lineage_states": len(fresh_names),
            "semantic_salience_clone_states": len(semantic_clones),
            "finite_adam_moment_tensors": finite_moment_tensors,
            "successful_local_optimizer_steps": successful_local,
            "skipped_local_optimizer_steps": skipped_local,
        },
        steps,
    )


def _validate_scaler(
    meta: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any] | None:
    scaler_contract = contract["fp16_scaler"]
    scaler = meta.get("fp16", {}).get("loss_scaler")
    if mode == "bootstrap":
        if scaler_contract["bootstrap_inherited"]:
            if scaler is None:
                raise StateLoadGateError("Bootstrap must inherit the FP16 scaler")
        elif scaler is not None:
            raise StateLoadGateError(
                "Bootstrap unexpectedly contains an FP16 scaler despite the "
                "declared fresh-recalibration policy"
            )
        return None

    if scaler_contract["resume_required"] and not isinstance(scaler, Mapping):
        raise StateLoadGateError("Ordinary V5 resume requires an FP16 scaler")
    if scaler is None:
        return None
    required = (
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    )
    if any(field not in scaler for field in required):
        raise StateLoadGateError("FP16 scaler state is incomplete")
    scale = float(scaler["scale"])
    growth_tracker = int(scaler["_growth_tracker"])
    if (
        not math.isfinite(scale)
        or scale <= 0
        or growth_tracker < 0
        or float(scaler["growth_factor"]) <= 1
        or not 0 < float(scaler["backoff_factor"]) < 1
        or int(scaler["growth_interval"]) <= 0
    ):
        raise StateLoadGateError("FP16 scaler state is numerically invalid")
    return {
        "scale": scale,
        "growth_factor": float(scaler["growth_factor"]),
        "backoff_factor": float(scaler["backoff_factor"]),
        "growth_interval": int(scaler["growth_interval"]),
        "growth_tracker": growth_tracker,
    }


def _validate_meta(
    meta: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    mode: str,
    updates_per_epoch: int | None,
) -> dict[str, Any]:
    optimizer_contract = contract["optimizer"]
    if meta.get("hod26_version") != contract["hod26_version"]:
        raise StateLoadGateError("Checkpoint hod26_version violates the contract")
    if not meta.get("optimizer_state_transplant"):
        raise StateLoadGateError("Checkpoint lost optimizer transplant lineage")
    if int(meta.get("optimizer_parent_iter", -1)) != int(
        optimizer_contract["parent_attempted_iterations"]
    ):
        raise StateLoadGateError("Checkpoint optimizer parent iteration mismatch")
    if int(meta.get("optimizer_parent_epoch", -1)) != int(
        contract["lineage"]["parent_epoch"]
    ):
        raise StateLoadGateError("Checkpoint optimizer parent epoch mismatch")
    if (
        meta.get("optimizer_parent_sha256")
        != contract["lineage"]["parent_checkpoint_sha256"]
    ):
        raise StateLoadGateError("Checkpoint optimizer parent SHA-256 mismatch")
    prohibited_parent = contract["lineage"].get(
        "prohibited_parent_checkpoint_sha256"
    )
    if prohibited_parent is not None:
        if meta.get("prohibited_faulty_v5_e18_sha256") != prohibited_parent:
            raise StateLoadGateError(
                "Checkpoint lost the prohibited-parent lineage declaration"
            )
        if meta.get("optimizer_parent_sha256") == prohibited_parent:
            raise StateLoadGateError("Checkpoint uses the prohibited parent")
    if int(meta.get("parent_optimizer_states", -1)) != int(
        optimizer_contract["parent_optimizer_states"]
    ):
        raise StateLoadGateError("Checkpoint parent optimizer-state count mismatch")
    if int(meta.get("ema_inherited_updates", -1)) != int(
        contract["ema"]["inherited_attempted_age"]
    ):
        raise StateLoadGateError("Checkpoint EMA inherited age mismatch")

    epoch = int(meta.get("epoch", -1))
    iteration = int(meta.get("iter", -1))
    if mode == "bootstrap":
        if epoch != 0 or iteration != 0:
            raise StateLoadGateError("Bootstrap checkpoint must start at epoch/iter 0")
    else:
        if epoch <= 0 or iteration <= 0:
            raise StateLoadGateError("Resume checkpoint has no training progress")
        if updates_per_epoch is not None and iteration != epoch * updates_per_epoch:
            raise StateLoadGateError(
                f"Checkpoint is not at an epoch boundary: epoch={epoch}, "
                f"iter={iteration}, updates_per_epoch={updates_per_epoch}"
            )
    return {"epoch": epoch, "attempted_local_iterations": iteration}


def _validate_provenance(
    checkpoint: Path,
    checkpoint_sha256: str,
    provenance_path: str | Path | None,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    path = (
        _resolve(provenance_path)
        if provenance_path is not None
        else checkpoint.with_suffix(".pth.provenance.json")
    )
    if not path.is_file():
        raise StateLoadGateError(f"Missing bootstrap provenance: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("output_sha256") != checkpoint_sha256:
        raise StateLoadGateError("Bootstrap checkpoint/provenance SHA-256 mismatch")
    if report.get("parent_sha256") != contract["lineage"]["parent_checkpoint_sha256"]:
        raise StateLoadGateError("Bootstrap provenance parent SHA-256 mismatch")
    prohibited_parent = contract["lineage"].get(
        "prohibited_parent_checkpoint_sha256"
    )
    if prohibited_parent is not None and (
        report.get("prohibited_faulty_v5_e18_sha256") != prohibited_parent
        or report.get("parent_sha256") == prohibited_parent
    ):
        raise StateLoadGateError(
            "Bootstrap provenance violates the prohibited-parent declaration"
        )
    if int(report.get("parent_epoch", -1)) != int(contract["lineage"]["parent_epoch"]):
        raise StateLoadGateError("Bootstrap provenance parent epoch mismatch")
    if int(report.get("parent_iter", -1)) != int(
        contract["optimizer"]["parent_attempted_iterations"]
    ):
        raise StateLoadGateError("Bootstrap provenance parent iteration mismatch")
    config_source = contract["lineage"].get(
        "config_source", "configs/v5/co_dino_vitl_fdr_salience_fold0.py"
    )
    expected_config_sha256 = contract["runtime_source_lock"][config_source]
    if report.get("config_sha256") != expected_config_sha256:
        raise StateLoadGateError(
            f"Bootstrap provenance config SHA-256 mismatch: {config_source}"
        )
    expected_report_values = {
        "parent_optimizer_states": contract["optimizer"]["parent_optimizer_states"],
        "transplanted_optimizer_states": contract["optimizer"][
            "bootstrap_stateful_states"
        ],
        "fresh_optimizer_states": contract["optimizer"]["fresh_states"],
        "salience_cloned_states": contract["optimizer"][
            "semantic_salience_clone_states"
        ],
        "ema_inherited_updates": contract["ema"]["inherited_attempted_age"],
    }
    for field, expected in expected_report_values.items():
        if int(report.get(field, -1)) != int(expected):
            raise StateLoadGateError(f"Bootstrap provenance field mismatch: {field}")
    return {
        "provenance": str(path),
        "provenance_sha256": _sha256(path),
    }


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    schema: RuntimeSchema,
    contract: Mapping[str, Any],
    *,
    mode: str,
    updates_per_epoch: int | None,
) -> ValidationResult:
    if not isinstance(payload.get("meta"), Mapping):
        raise StateLoadGateError("Checkpoint has no metadata")
    if not isinstance(payload.get("state_dict"), Mapping):
        raise StateLoadGateError("Checkpoint has no model state_dict")
    if not isinstance(payload.get("optimizer"), Mapping):
        raise StateLoadGateError("Checkpoint has no optimizer")
    meta = payload["meta"]
    report = {
        "mode": mode,
        **_validate_meta(
            meta,
            contract,
            mode=mode,
            updates_per_epoch=updates_per_epoch,
        ),
        **_validate_model_state(payload["state_dict"], schema, contract),
    }
    optimizer_report, steps = _validate_optimizer_state(
        payload["optimizer"],
        schema,
        contract,
        mode=mode,
        meta=meta,
    )
    scaler = _validate_scaler(meta, contract, mode=mode)
    report.update(optimizer_report)
    report["fp16_scaler"] = scaler
    report["fp16_scaler_policy"] = (
        contract["fp16_scaler"]["bootstrap_policy"]
        if mode == "bootstrap"
        else "required_and_restored"
    )
    if mode == "bootstrap":
        report["declared_parent_fp16_scaler"] = {
            "serialized": bool(contract["fp16_scaler"]["parent_was_serialized"]),
            "scale": float(contract["fp16_scaler"]["parent_scale"]),
            "growth_tracker": int(contract["fp16_scaler"]["parent_growth_tracker"]),
        }
    report["ema_inherited_attempted_age"] = int(
        contract["ema"]["inherited_attempted_age"]
    )
    report["ema_update_order"] = contract["ema"]["current_lineage_update_order"]
    report["ema_smooths_control_buffers"] = bool(
        contract["ema"]["current_lineage_smooths_control_buffers"]
    )
    return ValidationResult(report=report, steps=steps, scaler=scaler)


def validate_checkpoint_path(
    checkpoint_path: str | Path,
    schema: RuntimeSchema,
    contract: Mapping[str, Any],
    *,
    mode: str,
    updates_per_epoch: int | None,
    provenance_path: str | Path | None = None,
) -> ValidationResult:
    checkpoint = _resolve(checkpoint_path)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise StateLoadGateError(f"Missing checkpoint: {checkpoint}")
    checkpoint_sha256 = _sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    try:
        result = validate_checkpoint_payload(
            payload,
            schema,
            contract,
            mode=mode,
            updates_per_epoch=updates_per_epoch,
        )
    finally:
        del payload
        gc.collect()
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        **result.report,
    }
    if mode == "bootstrap":
        report.update(
            _validate_provenance(
                checkpoint,
                checkpoint_sha256,
                provenance_path,
                contract,
            )
        )
    return ValidationResult(report=report, steps=result.steps, scaler=result.scaler)


def validate_smoke_continuity(
    bootstrap_path: str | Path,
    first_path: str | Path,
    second_path: str | Path,
    schema: RuntimeSchema,
    contract: Mapping[str, Any],
    *,
    updates_per_epoch: int,
    expected_attempted_delta: int,
) -> dict[str, Any]:
    bootstrap = validate_checkpoint_path(
        bootstrap_path,
        schema,
        contract,
        mode="bootstrap",
        updates_per_epoch=None,
    )
    first = validate_checkpoint_path(
        first_path,
        schema,
        contract,
        mode="resume",
        updates_per_epoch=updates_per_epoch,
    )
    second = validate_checkpoint_path(
        second_path,
        schema,
        contract,
        mode="resume",
        updates_per_epoch=updates_per_epoch,
    )
    attempted_delta = (
        second.report["attempted_local_iterations"]
        - first.report["attempted_local_iterations"]
    )
    successful_delta = (
        second.report["successful_local_optimizer_steps"]
        - first.report["successful_local_optimizer_steps"]
    )
    if attempted_delta != expected_attempted_delta:
        raise StateLoadGateError(
            f"Smoke attempted-update delta is {attempted_delta}, "
            f"expected {expected_attempted_delta}"
        )
    if successful_delta != expected_attempted_delta:
        raise StateLoadGateError(
            f"Smoke successful-update delta is {successful_delta}, "
            f"expected {expected_attempted_delta}"
        )
    if set(first.steps) != set(second.steps):
        raise StateLoadGateError("Smoke optimizer parameter sets changed on resume")
    wrong_steps = [
        name
        for name in first.steps
        if second.steps[name] - first.steps[name] != expected_attempted_delta
    ]
    if wrong_steps:
        raise StateLoadGateError(
            f"Smoke AdamW state did not advance for every tensor: " f"{wrong_steps[:5]}"
        )
    if first.scaler is None or second.scaler is None:
        raise StateLoadGateError("Smoke child checkpoints lost the FP16 scaler")
    if first.scaler["scale"] == second.scaler["scale"]:
        tracker_delta = second.scaler["growth_tracker"] - first.scaler["growth_tracker"]
        if tracker_delta != expected_attempted_delta:
            raise StateLoadGateError(
                "Smoke FP16 scaler growth tracker is not resume-continuous"
            )
    return {
        "mode": "two_segment_smoke",
        "bootstrap": bootstrap.report,
        "first": first.report,
        "second": second.report,
        "attempted_update_delta": attempted_delta,
        "successful_optimizer_step_delta": successful_delta,
        "all_optimizer_tensors_advanced": len(first.steps),
        "fp16_scaler_resume_continuous": True,
        "ema_pairs_present_in_all_segments": int(contract["model"]["ema_pairs"]),
    }


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed HOD26 checkpoint state-loading gate"
    )
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "sources",
        help="verify locked runtime sources and the upstream Co-DETR tree",
    )

    checkpoint = subparsers.add_parser(
        "checkpoint",
        help="strictly validate one bootstrap or ordinary-resume checkpoint",
    )
    checkpoint.add_argument("--checkpoint", required=True)
    checkpoint.add_argument("--mode", choices=("bootstrap", "resume"), required=True)
    checkpoint.add_argument("--config", default=str(DEFAULT_CONFIG))
    checkpoint.add_argument("--provenance")
    checkpoint.add_argument("--updates-per-epoch", type=int)

    smoke = subparsers.add_parser(
        "smoke",
        help="validate bootstrap -> child -> resumed-child state continuity",
    )
    smoke.add_argument("--bootstrap", required=True)
    smoke.add_argument("--first", required=True)
    smoke.add_argument("--second", required=True)
    smoke.add_argument("--config", default=str(DEFAULT_CONFIG))
    smoke.add_argument("--updates-per-epoch", type=int, required=True)
    smoke.add_argument("--expected-attempted-delta", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        contract = _load_contract(args.contract)
        source_report = validate_runtime_sources(contract)
        if args.command == "sources":
            _print_json({"gate": "passed", **source_report})
            return
        schema = build_runtime_schema(args.config)
        if args.command == "checkpoint":
            if args.mode == "resume" and args.updates_per_epoch is None:
                raise StateLoadGateError("Ordinary resume requires --updates-per-epoch")
            result = validate_checkpoint_path(
                args.checkpoint,
                schema,
                contract,
                mode=args.mode,
                updates_per_epoch=args.updates_per_epoch,
                provenance_path=args.provenance,
            )
            _print_json(
                {
                    "gate": "passed",
                    "runtime_sources": source_report,
                    **result.report,
                }
            )
            return
        result = validate_smoke_continuity(
            args.bootstrap,
            args.first,
            args.second,
            schema,
            contract,
            updates_per_epoch=args.updates_per_epoch,
            expected_attempted_delta=args.expected_attempted_delta,
        )
        _print_json(
            {
                "gate": "passed",
                "runtime_sources": source_report,
                **result,
            }
        )
    except StateLoadGateError as error:
        print(f"STATE LOAD GATE REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
