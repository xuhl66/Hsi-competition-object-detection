#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
V6_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V6_PYTHON="$V6_ENV/bin/python"
[[ -x "$V6_PYTHON" ]] || { echo "Missing V6 environment: $V6_ENV" >&2; exit 2; }

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f storage/v6/dataset/fold0_band_stats.json || \
      ! -f storage/v6/dataset/full_band_stats.json ]]; then
  "$V6_PYTHON" -m hod26.v6.prepare
fi

if [[ ! -f storage/pretrained/v6/co_spec_dino_vitl_public_init.pth ]]; then
  "$V6_PYTHON" -m hod26.v6.checkpoint
fi

"$V6_PYTHON" -m hod26.v6.load_gate public_init \
  --checkpoint storage/pretrained/v6/co_spec_dino_vitl_public_init.pth \
  --config configs/v6/co_spec_dino_vitl_fold0.py \
  --source-lock configs/v6/runtime_source_lock.json

"$V6_PYTHON" - <<'PY'
import json
from pathlib import Path

expectations = {
    "storage/v6/dataset/fold0_band_stats.json": ("fold0_train_only", 2400),
    "storage/v6/dataset/full_band_stats.json": ("all_3000_official_train_images", 3000),
}
for filename, (scope, samples) in expectations.items():
    payload = json.loads(Path(filename).read_text(encoding="utf-8"))
    if payload.get("scope") != scope or int(payload.get("samples", -1)) != samples:
        raise SystemExit(f"V6 statistics contract failed: {filename}")
for filename, images in (
    ("storage/v2/dataset/fold0/train.json", 2400),
    ("storage/v2/dataset/fold0/val.json", 600),
    ("storage/v2/dataset/full.json", 3000),
):
    payload = json.loads(Path(filename).read_text(encoding="utf-8"))
    if len(payload["images"]) != images:
        raise SystemExit(f"V6 COCO view count failed: {filename}")
print("V6 preparation and public-only initialization gate passed")
PY
