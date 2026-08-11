#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash tools/infer_v6_raw.sh CHECKPOINT [OUTPUT.csv]" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CHECKPOINT="$(readlink -f "$1")"
OUTPUT="${2:-storage/v6/submissions/v6_raw_candidate.csv}"
[[ -f "$CHECKPOINT" ]] || { echo "Missing V6 checkpoint: $CHECKPOINT" >&2; exit 2; }
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
V6_PYTHON="storage/envs/codino-v4-smoke/bin/python"
CONFIG="${V6_INFER_CONFIG:-configs/v6/co_spec_dino_vitl_fold0.py}"
"$V6_PYTHON" -m hod26.v6.load_gate resume \
  --checkpoint "$CHECKPOINT" --config "$CONFIG" \
  --source-lock configs/v6/runtime_source_lock.json
exec "$V6_PYTHON" -m hod26.v6.raw_infer \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT" \
  --stats "${V6_INFER_STATS:-auto}"
