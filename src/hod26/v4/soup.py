"""Create one single-checkpoint EMA/SWA model soup from one V4 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from hod26.config import atomic_json_dump


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _primary_ema_state(payload: dict[str, Any]) -> OrderedDict[str, torch.Tensor]:
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("MMDetection checkpoint has no state_dict")
    # ExpMomentumEMAHook swaps EMA into ordinary parameters before checkpoint
    # saving. ema_* buffers at that point contain raw training parameters.
    return OrderedDict(
        (key, value) for key, value in state.items() if not key.startswith("ema_")
    )


def average_loaded_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    metadata: Sequence[Mapping[str, Any]],
    source_paths: Sequence[str | Path],
    output_path: str | Path,
    weights: Sequence[float] | None = None,
    source_sha256: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Average already-loaded primary EMA states into one inference checkpoint.

    Keeping this operation separate from checkpoint loading lets a systematic
    soup search load each 5.6 GiB training checkpoint only once.  The returned
    artifact is still one model/checkpoint; no prediction ensemble is created.
    """

    paths = [Path(path).resolve() for path in source_paths]
    if len(paths) < 2 or len(states) != len(paths) or len(metadata) != len(paths):
        raise ValueError("EMA/SWA soup requires at least two aligned states")
    raw_weights = list(weights or [1.0] * len(paths))
    if len(raw_weights) != len(paths) or any(value <= 0 for value in raw_weights):
        raise ValueError("weights must be positive and match checkpoint count")
    total = float(sum(raw_weights))
    normalized = [float(value) / total for value in raw_weights]

    configs = [meta.get("config") for meta in metadata]
    if not configs[0] or any(config != configs[0] for config in configs[1:]):
        raise ValueError("All soup members must come from the exact same V4 run/config")
    keys = list(states[0])
    if any(list(state) != keys for state in states[1:]):
        raise ValueError("Soup checkpoint state dictionaries do not match")

    averaged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in keys:
        tensors = [state[key] for state in states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"Shape mismatch for {key}")
        if tensors[0].dtype.is_floating_point:
            value = tensors[0].float().mul(normalized[0])
            for tensor, weight in zip(tensors[1:], normalized[1:]):
                value.add_(tensor.float(), alpha=weight)
            averaged[key] = value.to(dtype=tensors[0].dtype)
        else:
            averaged[key] = tensors[0].clone()

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    meta = dict(metadata[0])
    meta.update(
        {
            "hod26_weight_source": "same-run primary EMA weight soup",
            "hod26_soup_members": [str(path) for path in paths],
            "hod26_soup_weights": normalized,
            "single_checkpoint": True,
        }
    )
    torch.save({"meta": meta, "state_dict": averaged}, temporary)
    temporary.replace(destination)
    del averaged

    hashes = (
        list(source_sha256)
        if source_sha256 is not None
        else [_sha256(path) for path in paths]
    )
    if len(hashes) != len(paths):
        raise ValueError("source_sha256 must match checkpoint count")
    report = {
        "output": str(destination),
        "output_sha256": _sha256(destination),
        "members": [
            {
                "path": str(path),
                "sha256": sha256,
                "weight": weight,
                "epoch": meta.get("epoch"),
                "iter": meta.get("iter"),
            }
            for path, sha256, weight, meta in zip(
                paths, hashes, normalized, metadata
            )
        ],
        "weight_source": "EMA parameters saved as primary MMDetection state",
        "prediction_ensemble": False,
        "single_output_checkpoint": True,
    }
    atomic_json_dump(
        report, destination.with_suffix(destination.suffix + ".audit.json")
    )
    return report


def average_checkpoints(
    checkpoints: Sequence[str | Path],
    output_path: str | Path,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in checkpoints]
    if len(paths) < 2:
        raise ValueError("EMA/SWA soup requires at least two checkpoints")
    states: list[OrderedDict[str, torch.Tensor]] = []
    metadata: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        states.append(_primary_ema_state(payload))
        metadata.append(dict(payload.get("meta", {})))
        del payload
    return average_loaded_states(
        states,
        metadata,
        paths,
        output_path,
        weights,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Average converged same-run V4 EMA checkpoints"
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = average_checkpoints(args.checkpoints, args.output, args.weights)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
