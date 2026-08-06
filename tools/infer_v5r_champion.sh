#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash tools/infer_v5r_champion.sh CHECKPOINT [OUTPUT.csv]" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CHECKPOINT="$1"
OUTPUT="${2:-storage/v5r/submissions/v5r_champion_protocol.csv}"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec storage/envs/codino-v4-smoke/bin/python -m hod26.v4.infer \
  --config configs/v5r/co_dino_vitl_fdr_prehead_fold0.py \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --stats storage/v4/dataset/fold0_band_stats.json \
  --widths 1280 \
  --flips none horizontal \
  --soft-nms-iou 0.55 \
  --soft-nms-sigma 0.5 \
  --soft-nms-method gaussian \
  --pre-score 0.0001 \
  --output-score 0.0001 \
  --max-detections 300
