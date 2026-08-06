#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

V4_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V4_PYTHON="$V4_ENV/bin/python"
RUN_DIR="${V4_OPT_RUN_DIR:-storage/v4/optimization/fold0_epoch64}"
OUTPUT="${V4_OPT_OUTPUT:-storage/v4/submissions/v4_soup_tta_nms_optimized.csv}"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export V4_SOUP_CPU_THREADS="${V4_SOUP_CPU_THREADS:-32}"
export V4_CACHE_CPU_THREADS="${V4_CACHE_CPU_THREADS:-8}"
export V4_CACHE_OPENCV_THREADS="${V4_CACHE_OPENCV_THREADS:-4}"
export V4_SEARCH_WORKERS="${V4_SEARCH_WORKERS:-32}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if pgrep -af '[t]ools/train.py.*co_dino_vitl_hsi_fold0.py' >/dev/null; then
  echo "Refusing optimization while V4 DDP training still owns the GPUs." >&2
  exit 2
fi

test -f storage/v4/runs/co_dino_vitl_hsi_fold0/best_bbox_mAP_epoch_30.pth
test -f storage/v4/runs/co_dino_vitl_hsi_fold0/epoch_42.pth
test -f storage/v4/runs/co_dino_vitl_hsi_fold0/epoch_50.pth
test -f storage/v4/runs/co_dino_vitl_hsi_fold0/epoch_54.pth
test -f storage/v4/runs/co_dino_vitl_hsi_fold0/epoch_64.pth
test -f storage/v4/dataset/fold0_band_stats.json
test -f storage/v2/dataset/fold0/val.json
test -f storage/cache/test_manifest.json

mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT")"

printf '%s\n' \
  "V4 post-convergence optimization" \
  "  source: one same-run EMA checkpoint or one same-run weight soup" \
  "  soup search: Epoch 30 anchor with Epoch 42/50/54/64 combinations and anchor-biased weights" \
  "  TTA cache: 1024/1152/1280/1408 x none/horizontal/vertical/both" \
  "  merge search: hard NMS, linear/gaussian Soft-NMS, WBF and box voting" \
  "  CPU: NUMA-local dual pools, ${V4_SEARCH_WORKERS} search workers per finalist" \
  "  metric: Fold0 COCO mAP@[.50:.95], maxDets=(1,10,100)" \
  "  stats: fold0-train-only normalization (identical to formal V4 training)" \
  "  output: $OUTPUT" \
  "  Kaggle upload: disabled"

exec "$V4_PYTHON" -m hod26.v4.optimize pipeline \
  --source-run storage/v4/runs/co_dino_vitl_hsi_fold0 \
  --run-dir "$RUN_DIR" \
  --output "$OUTPUT" \
  --config configs/v4/co_dino_vitl_hsi_fold0.py \
  --data-root data/raw \
  --stats storage/v4/dataset/fold0_band_stats.json \
  --upstream storage/upstream/Co-DETR \
  --test-manifest storage/cache/test_manifest.json \
  --val-annotations storage/v2/dataset/fold0/val.json
