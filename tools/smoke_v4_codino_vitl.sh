#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
bash tools/prepare_v4_codino_vitl.sh

V4_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SMOKE_DIR="storage/v4/smoke/co_dino_vitl_hsi_object_aware"

"$V4_ENV/bin/torchrun" \
  --standalone \
  --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py \
  configs/v4/co_dino_vitl_hsi_smoke.py \
  --launcher pytorch \
  --work-dir "$SMOKE_DIR" \
  --seed 20260728 \
  --cfg-options runner.max_epochs=1

"$V4_ENV/bin/torchrun" \
  --standalone \
  --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py \
  configs/v4/co_dino_vitl_hsi_smoke.py \
  --launcher pytorch \
  --work-dir "$SMOKE_DIR" \
  --seed 20260728 \
  --cfg-options \
    "custom_hooks.1.resume_from=$REPO_ROOT/$SMOKE_DIR/epoch_1.pth"

mapfile -t SMOKE_LOGS < <(
  find "$SMOKE_DIR" -maxdepth 1 -type f -name '*.log.json' \
    -printf '%T@ %p\n' | sort -nr | head -2 | cut -d' ' -f2-
)
if [[ "${#SMOKE_LOGS[@]}" -ne 2 ]]; then
  echo "Could not identify both V4 smoke logs" >&2
  exit 2
fi
TRAIN_ROWS="$(
  jq -s '[.[] | select(.mode == "train")] | length' "${SMOKE_LOGS[@]}"
)"
NONFINITE_ROWS="$(
  jq -s \
    '[.[] | select(.mode == "train") |
      select(
        .loss == null or (.loss | isnan or isinfinite) or
        .grad_norm == null or (.grad_norm | isnan or isinfinite) or
        .loss_object_activation == null or
          (.loss_object_activation | isnan or isinfinite) or
        .object_activation_intersection == null or
          (.object_activation_intersection | isnan or isinfinite) or
        .object_activation_difference == null or
          (.object_activation_difference | isnan or isinfinite)
      )] | length' \
    "${SMOKE_LOGS[@]}"
)"
EPOCHS="$(
  jq -sr \
    '[.[] | select(.mode == "train") | .epoch] | unique | sort | join(",")' \
    "${SMOKE_LOGS[@]}"
)"
if [[ "$TRAIN_ROWS" -ne 4 || "$NONFINITE_ROWS" -ne 0 || "$EPOCHS" != "1,2" ]]; then
  echo "V4 smoke numerical/resume verification failed: rows=$TRAIN_ROWS nonfinite=$NONFINITE_ROWS epochs=$EPOCHS" >&2
  exit 2
fi
printf '%s\n' \
  "V4 dual-GPU smoke ready" \
  "  finite train updates: $TRAIN_ROWS/4" \
  "  resume epochs observed: $EPOCHS" \
  "  Mosaic/object loss/DDP/EMA/checkpoint/validation: passed"
