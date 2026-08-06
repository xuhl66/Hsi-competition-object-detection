#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

V4_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V4_PYTHON="$V4_ENV/bin/python"
if [[ ! -x "$V4_PYTHON" ]]; then
  echo "Missing V4 environment: $V4_PYTHON" >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f storage/v4/dataset/fold0_band_stats.json ]]; then
  "$V4_PYTHON" -m hod26.v4.prepare
fi

if [[ ! -f storage/pretrained/v4/co_dino_vitl_hod26_init.pth ]]; then
  "$V4_PYTHON" -m hod26.v4.checkpoint
fi

"$V4_PYTHON" - <<'PY'
import json
import hashlib
import subprocess
from pathlib import Path
from hod26.v4.checkpoint import sha256_file
from hod26.v4.constants import OFFICIAL_CODETR_COMMIT

required = (
    Path("storage/v4/dataset/fold0_band_stats.json"),
    Path("storage/pretrained/v4/co_dino_vitl_hod26_init.pth"),
    Path("storage/pretrained/v4/co_dino_vitl_hod26_init.pth.provenance.json"),
)
for path in required:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"V4 preparation failed: {path}")
provenance = json.loads(required[2].read_text(encoding="utf-8"))
actual_init_hash = sha256_file(required[1])
actual_config_hash = sha256_file(
    Path("configs/v4/co_dino_vitl_hsi_fold0.py")
)
if provenance.get("output_sha256") != actual_init_hash:
    raise SystemExit("V4 initialization/provenance SHA-256 mismatch")
if provenance.get("config_sha256") != actual_config_hash:
    raise SystemExit(
        "V4 initialization is stale for the current formal config; "
        "regenerate it with python -m hod26.v4.checkpoint"
    )
statistics = json.loads(required[0].read_text(encoding="utf-8"))
if (
    statistics.get("scope") != "fold0_train_only"
    or statistics.get("excluded_validation_fold") != 0
    or statistics.get("samples") != 2400
):
    raise SystemExit("V4 Fold0 statistics are not leakage-safe")
upstream = Path("storage/upstream/Co-DETR")
upstream_resolved = upstream.resolve()
git_prefix = [
    "git",
    "-c",
    f"safe.directory={upstream_resolved}",
    "-C",
    str(upstream_resolved),
]
commit = subprocess.run(
    [*git_prefix, "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if commit != OFFICIAL_CODETR_COMMIT:
    raise SystemExit(f"Unexpected Co-DETR commit: {commit}")
actual_patch = subprocess.run(
    [*git_prefix, "diff", "--binary", "HEAD"],
    check=True,
    capture_output=True,
).stdout
declared_patch = Path(
    "patches/codetr_v4_torch21_fp16.patch"
).read_bytes()
if actual_patch != declared_patch:
    raise SystemExit(
        "Co-DETR local changes do not exactly match the declared V4 patch"
    )
print("V4 preparation ready")
print(f"init_sha256={actual_init_hash}")
print(f"config_sha256={actual_config_hash}")
print(f"upstream_patch_sha256={hashlib.sha256(actual_patch).hexdigest()}")
PY
