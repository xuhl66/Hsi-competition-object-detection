#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash tools/infer_v4_full_tta.sh CHECKPOINT [OUTPUT.csv]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
V4_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
CHECKPOINT="$1"
OUTPUT="${2:-storage/v4/submissions/co_dino_vitl_hsi_full_tta.csv}"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$V4_ENV/bin/python" -m hod26.v4.infer \
  --config configs/v4/co_dino_vitl_hsi_fold0.py \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --stats storage/v4/dataset/fold0_band_stats.json \
  --widths 1024 1152 1280 1408 \
  --flips none horizontal vertical both \
  --soft-nms-iou 0.65 \
  --soft-nms-sigma 0.5 \
  --pre-score 0.0001 \
  --output-score 0.0001 \
  --max-detections 300
