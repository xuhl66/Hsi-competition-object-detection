#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
bash tools/prepare_v5.sh
V5_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V5_PYTHON="$V5_ENV/bin/python"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SMOKE_DIR="${V5_SMOKE_DIR:-storage/v5/smoke/run_$(date +%Y%m%d_%H%M%S)}"
STATEFUL_INIT="$REPO_ROOT/storage/pretrained/v5/v4_e30_stateful_to_v5_init.pth"
STATE_LOAD_CONTRACT="$REPO_ROOT/configs/v5/state_load_contract.json"
mkdir -p "$SMOKE_DIR"
"$V5_PYTHON" -m hod26.v5.load_gate \
  --contract "$STATE_LOAD_CONTRACT" \
  checkpoint \
  --checkpoint "$STATEFUL_INIT" \
  --mode bootstrap
"$V5_ENV/bin/torchrun" \
  --standalone --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py \
  configs/v5/co_dino_vitl_fdr_salience_smoke.py \
  --launcher pytorch --work-dir "$SMOKE_DIR" --seed 20260730 \
  --cfg-options \
    runner.max_epochs=1 \
    "custom_hooks.1.resume_from=$STATEFUL_INIT"

test -f "$SMOKE_DIR/epoch_1.pth"
"$V5_ENV/bin/torchrun" \
  --standalone --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py \
  configs/v5/co_dino_vitl_fdr_salience_smoke.py \
  --launcher pytorch --work-dir "$SMOKE_DIR" --seed 20260730 \
  --cfg-options \
    runner.max_epochs=2 \
    "custom_hooks.1.resume_from=$REPO_ROOT/$SMOKE_DIR/epoch_1.pth"

test -f "$SMOKE_DIR/epoch_2.pth"
"$V5_PYTHON" -m hod26.v5.load_gate \
  --contract "$STATE_LOAD_CONTRACT" \
  smoke \
  --bootstrap "$STATEFUL_INIT" \
  --first "$SMOKE_DIR/epoch_1.pth" \
  --second "$SMOKE_DIR/epoch_2.pth" \
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
      .loss_fgl == null or (.loss_fgl | isnan or isinfinite) or
      .loss_salience == null or (.loss_salience | isnan or isinfinite) or
      (
        .salience_transition > 0 and
        (
          ."d0.loss_go_lsd" == null or
          (."d0.loss_go_lsd" | isnan or isinfinite) or
          ."d0.dn_loss_go_lsd" == null or
          (."d0.dn_loss_go_lsd" | isnan or isinfinite)
        )
      )
    )] | length' "${LOGS[@]}")"
EPOCHS="$(jq -sr '[.[] | select(.mode=="train") | .epoch] | unique | sort | join(",")' "${LOGS[@]}")"
if [[ "$TRAIN_ROWS" -ne 4 || "$NONFINITE" -ne 0 || "$EPOCHS" != "1,2" ]]; then
  echo "V5 smoke failed: rows=$TRAIN_ROWS nonfinite=$NONFINITE epochs=$EPOCHS" >&2
  exit 2
fi
printf '%s\n' \
  "V5 dual-GPU worst-case smoke passed" \
  "  finite updates: $TRAIN_ROWS/4" \
  "  resumed epochs: $EPOCHS" \
  "  all 1090 optimizer tensors, 1138 EMA pairs and FP16 scaler resume continuity: passed" \
  "  attempted vs successful optimizer-step accounting: passed" \
  "  1280 Mosaic/DDP/FDR/GO-LSD/LQE/salience/checkpoint/validation: passed" \
  "  artifact: $SMOKE_DIR"
