#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
bash tools/prepare_v5r.sh

V5R_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V5R_PYTHON="$V5R_ENV/bin/python"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SMOKE_DIR="${V5R_SMOKE_DIR:-storage/v5r/smoke/run_$(date +%Y%m%d_%H%M%S)}"
STATEFUL_INIT="$REPO_ROOT/storage/pretrained/v5r/v4_e30_stateful_to_v5r_init.pth"
CONTRACT="$REPO_ROOT/configs/v5r/state_load_contract.json"
FORMAL_CONFIG="$REPO_ROOT/configs/v5r/co_dino_vitl_fdr_prehead_fold0.py"
SMOKE_CONFIG="$REPO_ROOT/configs/v5r/co_dino_vitl_fdr_prehead_smoke.py"

if [[ -e "$SMOKE_DIR" ]]; then
  echo "Refusing to reuse an existing V5R smoke directory: $SMOKE_DIR" >&2
  exit 2
fi
mkdir -p "$SMOKE_DIR"

"$V5R_ENV/bin/torchrun" \
  --standalone --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py \
  "$SMOKE_CONFIG" \
  --launcher pytorch --work-dir "$SMOKE_DIR" --seed 20260731 \
  --cfg-options \
    runner.max_epochs=1 \
    "custom_hooks.1.resume_from=$STATEFUL_INIT"

test -f "$SMOKE_DIR/epoch_1.pth"
"$V5R_ENV/bin/torchrun" \
  --standalone --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py \
  "$SMOKE_CONFIG" \
  --launcher pytorch --work-dir "$SMOKE_DIR" --seed 20260731 \
  --cfg-options \
    runner.max_epochs=2 \
    "custom_hooks.1.resume_from=$REPO_ROOT/$SMOKE_DIR/epoch_1.pth"

test -f "$SMOKE_DIR/epoch_2.pth"
"$V5R_PYTHON" -m hod26.v5.load_gate \
  --contract "$CONTRACT" smoke \
  --bootstrap "$STATEFUL_INIT" \
  --first "$SMOKE_DIR/epoch_1.pth" \
  --second "$SMOKE_DIR/epoch_2.pth" \
  --config "$FORMAL_CONFIG" \
  --updates-per-epoch 2 \
  --expected-attempted-delta 2

mapfile -t LOGS < <(
  find "$SMOKE_DIR" -maxdepth 1 -type f -name '*.log.json' \
    -printf '%T@ %p\n' | sort -nr | head -2 | cut -d' ' -f2-
)
[[ "${#LOGS[@]}" -eq 2 ]]
TRAIN_ROWS="$(jq -s '[.[] | select(.mode=="train")] | length' "${LOGS[@]}")"
NONFINITE="$(jq -s \
  '[.[] | select(.mode=="train") |
    select(
      .loss == null or (.loss | isnan or isinfinite) or
      .grad_norm == null or (.grad_norm | isnan or isinfinite) or
      .loss_salience == null or (.loss_salience | isnan or isinfinite) or
      (
        .salience_transition > 0 and
        (
          .loss_pre_bbox == null or
          (.loss_pre_bbox | isnan or isinfinite) or
          .dn_loss_pre_bbox == null or
          (.dn_loss_pre_bbox | isnan or isinfinite) or
          .loss_fgl == null or
          (.loss_fgl | isnan or isinfinite) or
          .fdr_main_matched_iou == null or
          (.fdr_main_matched_iou | isnan or isinfinite) or
          .fdr_dn_matched_iou == null or
          (.fdr_dn_matched_iou | isnan or isinfinite)
        )
      )
    )] | length' "${LOGS[@]}")"
EPOCHS="$(jq -sr '[.[] | select(.mode=="train") | .epoch] | unique | sort | join(",")' "${LOGS[@]}")"
if [[ "$TRAIN_ROWS" -ne 4 || "$NONFINITE" -ne 0 || "$EPOCHS" != "1,2" ]]; then
  echo "V5R smoke failed: rows=$TRAIN_ROWS nonfinite=$NONFINITE epochs=$EPOCHS" >&2
  exit 2
fi
printf '%s\n' \
  "V5R dual-GPU two-segment smoke passed" \
  "  finite updates: $TRAIN_ROWS/4; resumed epochs: $EPOCHS" \
  "  all 1096 optimizer tensors, 1144 EMA pairs and FP16 scaler: continuous" \
  "  inherited/fresh AdamW step equations and AMP accounting: passed" \
  "  supervised pre-head, DN pre-head, full FDR/LQE/GO-LSD and native aux isolation: passed" \
  "  post-successful-step EMA and unsmoothed transition buffers: passed" \
  "  1280 Mosaic/DDP/backward/checkpoint/validation: passed" \
  "  artifact: $SMOKE_DIR"

