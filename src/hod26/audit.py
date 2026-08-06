from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from hod26.config import atomic_json_dump, resolve_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _record_file(path: str | Path) -> dict[str, Any]:
    file_path = resolve_path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return {
        "path": str(file_path),
        "bytes": file_path.stat().st_size,
        "sha256": _sha256(file_path),
    }


def _collect_included_configs(path: str | Path) -> list[Path]:
    root = resolve_path(path)
    ordered: list[Path] = []
    visited: set[Path] = set()

    def visit(config_path: Path) -> None:
        config_path = config_path.resolve()
        if config_path in visited:
            return
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        visited.add(config_path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        includes = payload.get("__include__", [])
        if isinstance(includes, str):
            includes = [includes]
        if not isinstance(includes, list) or not all(
            isinstance(item, str) for item in includes
        ):
            raise ValueError(f"Invalid __include__ declaration in {config_path}")
        for include in includes:
            visit(config_path.parent / include)
        ordered.append(config_path)

    visit(root)
    return ordered


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    # A user-owned competition workspace may intentionally be launched from a
    # root shell. Scope the ownership exception to this exact working tree and
    # this subprocess only; never mutate global Git configuration.
    safe_directory = Path.cwd().resolve()
    return subprocess.run(
        ["git", "-c", f"safe.directory={safe_directory}", *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _git_record() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD").stdout.decode().strip()
    branch = _git("branch", "--show-current").stdout.decode().strip()
    status = _git("status", "--short").stdout.decode().splitlines()
    diff = _git("diff", "--binary", "HEAD").stdout
    return {
        "head": head,
        "branch": branch,
        "working_tree_dirty": bool(status),
        "status": status,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _environment_record() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _torchvision_version(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_count": torch.cuda.device_count(),
    }


def _torchvision_version() -> str | None:
    try:
        import torchvision
    except Exception:
        return None
    return torchvision.__version__


def _write_snapshot_archive(
    destination: Path,
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    archive_path = destination.with_name(f"{destination.stem}.snapshot.zip")
    if archive_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable source snapshot: {archive_path}"
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    index: list[dict[str, Any]] = []
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for group, records in groups.items():
            for position, record in enumerate(records):
                source = Path(record["path"])
                member = f"{group}/{position:03d}_{source.name}"
                archive.write(source, arcname=member)
                index.append(
                    {
                        "member": member,
                        "original_path": record["path"],
                        "sha256": record["sha256"],
                    }
                )
        index_bytes = json.dumps(
            index,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode()
        info = zipfile.ZipInfo("SNAPSHOT_INDEX.json")
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, index_bytes)
    temporary.replace(archive_path)
    return _record_file(archive_path)


def record_candidate(
    candidate: str,
    command: str,
    config_paths: Sequence[str | Path],
    source_paths: Sequence[str | Path],
    data_paths: Sequence[str | Path],
    *,
    initialization: str,
    parent_checkpoint: str | Path | None = None,
    lineage_manifest: str | Path | None = None,
    external_pretraining_provenance: str | Path | None = None,
    output_path: str | Path | None = None,
    notes: str | None = None,
) -> Path:
    """Capture one immutable candidate-stage reconstruction record."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", candidate):
        raise ValueError(
            "candidate must contain only letters, digits, dot, dash, underscore"
        )
    if not command.strip():
        raise ValueError("command must not be empty")
    if not config_paths:
        raise ValueError("At least one config is required")
    if not source_paths:
        raise ValueError("At least one source file is required")
    if not data_paths:
        raise ValueError("At least one immutable data view is required")

    config_files: list[Path] = []
    for config_path in config_paths:
        for included in _collect_included_configs(config_path):
            if included not in config_files:
                config_files.append(included)

    captured_at = datetime.now().astimezone()
    timestamp = captured_at.strftime("%Y%m%dT%H%M%S%z")
    destination = resolve_path(
        output_path
        or (
            f"storage/compliance/candidates/{candidate}/"
            f"candidate_manifest_{timestamp}.json"
        )
    )
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable candidate record: {destination}"
        )

    parent = _record_file(parent_checkpoint) if parent_checkpoint is not None else None
    lineage = _record_file(lineage_manifest) if lineage_manifest is not None else None
    external = (
        _record_file(external_pretraining_provenance)
        if external_pretraining_provenance is not None
        else None
    )
    configs = [_record_file(path) for path in config_files]
    sources = [_record_file(path) for path in source_paths]
    data_views = [_record_file(path) for path in data_paths]
    bundle_payload = {
        "configs": configs,
        "sources": sources,
        "data_views": data_views,
    }
    bundle_sha256 = hashlib.sha256(
        json.dumps(
            bundle_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    git_record = _git_record()
    environment_record = _environment_record()
    evidence = [record for record in (lineage, external) if record is not None]
    snapshot_archive = _write_snapshot_archive(
        destination,
        {
            "configs": configs,
            "sources": sources,
            "data_views": data_views,
            "evidence": evidence,
        },
    )

    manifest = {
        "schema_version": 1,
        "record_type": "candidate_training_stage",
        "candidate": candidate,
        "is_final_model": False,
        "captured_at": captured_at.isoformat(),
        "planned_launch_command": command.strip(),
        "initialization": initialization,
        "parent_checkpoint": parent,
        "lineage_manifest": lineage,
        "external_pretraining_provenance": external,
        "reconstruction_bundle": bundle_payload,
        "reconstruction_bundle_sha256": bundle_sha256,
        "source_snapshot_archive": snapshot_archive,
        "git": git_record,
        "environment": environment_record,
        "data_use_declaration": {
            "competition_annotations_only_for_task_finetuning": True,
            "declared_external_pretraining": external is not None,
            "external_hyperspectral_dataset_used": False,
            "test_or_ranking_labels_used": False,
        },
        "single_checkpoint_candidate": True,
        "notes": notes,
    }
    atomic_json_dump(manifest, destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record a version-neutral candidate snapshot before formal HOD26 "
            "training; this command never starts training"
        )
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--initialization", required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--lineage-manifest")
    parser.add_argument("--external-pretraining-provenance")
    parser.add_argument("--output")
    parser.add_argument("--notes")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    destination = record_candidate(
        candidate=args.candidate,
        command=args.command,
        config_paths=args.config,
        source_paths=args.source,
        data_paths=args.data,
        initialization=args.initialization,
        parent_checkpoint=args.parent_checkpoint,
        lineage_manifest=args.lineage_manifest,
        external_pretraining_provenance=args.external_pretraining_provenance,
        output_path=args.output,
        notes=args.notes,
    )
    print(destination)


if __name__ == "__main__":
    main()
