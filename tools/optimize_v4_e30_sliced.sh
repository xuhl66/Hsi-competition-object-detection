#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

V4_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V4_PYTHON="$V4_ENV/bin/python"
CHECKPOINT="storage/v4/runs/co_dino_vitl_hsi_fold0/best_bbox_mAP_epoch_30.pth"
EXPECTED_SHA="2b96f909af97c93c01c2141d032eb391428b18aac76be5e38f4f3ae5f9198932"
RUN_DIR="${V4_SLICE_RUN_DIR:-storage/v4/optimization/e30_sliced_consistency}"
VAL_SLICES="$RUN_DIR/val_slices"
TEST_SLICES="$RUN_DIR/test_slices"
REPORT="$RUN_DIR/calibration.json"
OUTPUT="${V4_SLICE_OUTPUT:-storage/v4/submissions/submission_0730_c.csv}"
GLOBAL_VAL="storage/v4/optimization/fold0_epoch64/full_val/e30"
GLOBAL_TEST="storage/v4/optimization/fold0_epoch64/test/e30"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export V4_SLICE_CPU_THREADS="${V4_SLICE_CPU_THREADS:-8}"
export V4_SLICE_OPENCV_THREADS="${V4_SLICE_OPENCV_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if pgrep -af '[t]ools/train.py.*co_dino_vitl_hsi_fold0.py' >/dev/null; then
  echo "Refusing sliced inference while V4 DDP training owns the GPUs." >&2
  exit 2
fi
if pgrep -af '[h]od26.v4.sliced (cache|search|generate)' >/dev/null; then
  echo "Refusing duplicate V4 sliced pipeline." >&2
  exit 2
fi

test -x "$V4_PYTHON"
test -f "$CHECKPOINT"
test -f configs/v4/co_dino_vitl_hsi_fold0.py
test -d storage/upstream/Co-DETR
test -f storage/v4/dataset/fold0_band_stats.json
test -f storage/cache/train_manifest.json
test -f storage/cache/test_manifest.json
test -f storage/v2/dataset/fold0/val.json
test -f "$GLOBAL_VAL/w1280_none.npz"
test -f "$GLOBAL_VAL/w1280_horizontal.npz"
test -f "$GLOBAL_TEST/w1280_none.npz"
test -f "$GLOBAL_TEST/w1280_horizontal.npz"

ACTUAL_SHA="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "Checkpoint SHA256 mismatch: $ACTUAL_SHA" >&2
  exit 3
fi

mkdir -p "$VAL_SLICES" "$TEST_SLICES" "$(dirname "$OUTPUT")"

printf '%s\n' \
  "V4 e30 sliced consistency pipeline" \
  "  anchor: e30 EMA, SHA256=$EXPECTED_SHA" \
  "  global anchor: 1280 none/horizontal + Gaussian Soft-NMS sigma=0.5" \
  "  slices: four 60% corner crops, width=1280, 20% overlap" \
  "  selection: grouped 400-image tune / 200-image one-shot gate" \
  "  test labels: never read" \
  "  output: $OUTPUT" \
  "  Kaggle upload: disabled"

run_cache_pair() {
  local split="$1"
  local output_dir="$2"
  local pid0 pid1 code0 code1

  "$V4_PYTHON" -m hod26.v4.sliced cache \
    --checkpoint "$CHECKPOINT" \
    --checkpoint-sha256 "$EXPECTED_SHA" \
    --split "$split" \
    --slices tl bl \
    --output-dir "$output_dir" \
    --device cuda:0 \
    --config configs/v4/co_dino_vitl_hsi_fold0.py \
    --data-root data/raw \
    --stats storage/v4/dataset/fold0_band_stats.json \
    --upstream storage/upstream/Co-DETR \
    --test-manifest storage/cache/test_manifest.json \
    --val-annotations storage/v2/dataset/fold0/val.json &
  pid0=$!

  "$V4_PYTHON" -m hod26.v4.sliced cache \
    --checkpoint "$CHECKPOINT" \
    --checkpoint-sha256 "$EXPECTED_SHA" \
    --split "$split" \
    --slices tr br \
    --output-dir "$output_dir" \
    --device cuda:1 \
    --config configs/v4/co_dino_vitl_hsi_fold0.py \
    --data-root data/raw \
    --stats storage/v4/dataset/fold0_band_stats.json \
    --upstream storage/upstream/Co-DETR \
    --test-manifest storage/cache/test_manifest.json \
    --val-annotations storage/v2/dataset/fold0/val.json &
  pid1=$!

  code0=0
  code1=0
  wait "$pid0" || code0=$?
  if (( code0 != 0 )); then
    kill "$pid1" 2>/dev/null || true
  fi
  wait "$pid1" || code1=$?
  if (( code0 != 0 || code1 != 0 )); then
    echo "Slice cache failure: GPU0=$code0 GPU1=$code1" >&2
    exit 4
  fi
}

run_cache_pair val "$VAL_SLICES"

if [[ "${V4_SLICE_REUSE_CALIBRATION:-0}" == "1" ]]; then
  test -f "$REPORT"
  echo "Reusing existing calibration report: $REPORT"
else
  "$V4_PYTHON" -m hod26.v4.sliced search \
    --global-cache-dir "$GLOBAL_VAL" \
    --slice-cache-dir "$VAL_SLICES" \
    --val-annotations storage/v2/dataset/fold0/val.json \
    --train-manifest storage/cache/train_manifest.json \
    --output "$REPORT"
fi

run_cache_pair test "$TEST_SLICES"

"$V4_PYTHON" -m hod26.v4.sliced generate \
  --global-cache-dir "$GLOBAL_TEST" \
  --slice-cache-dir "$TEST_SLICES" \
  --calibration-report "$REPORT" \
  --test-manifest storage/cache/test_manifest.json \
  --output "$OUTPUT"

echo "Pipeline complete. CSV generated locally; no Kaggle command was executed."
