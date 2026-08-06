#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ "${V5_AUDIT_ONLY:-0}" != "1" ]] &&
  pgrep -af '[t]ools/train.py.*co_dino_vitl_fdr_salience_fold0.py' >/dev/null; then
  echo "Refusing to launch: an active V5 formal training process already exists." >&2
  exit 2
fi
bash tools/prepare_v5.sh

V5_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V5_PYTHON="$V5_ENV/bin/python"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

V5_MAX_EPOCHS="${V5_MAX_EPOCHS:-84}"
[[ "$V5_MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
  echo "V5_MAX_EPOCHS must be a positive integer" >&2
  exit 2
}
WORK_DIR="$REPO_ROOT/storage/v5/runs/co_dino_vitl_fdr_salience_fold0"
STATEFUL_INIT="$REPO_ROOT/storage/pretrained/v5/v4_e30_stateful_to_v5_init.pth"
STATE_LOAD_CONTRACT="$REPO_ROOT/configs/v5/state_load_contract.json"
CFG_OPTIONS=("runner.max_epochs=$V5_MAX_EPOCHS")
if [[ -e "$WORK_DIR/latest.pth" || -L "$WORK_DIR/latest.pth" ]]; then
  RESUME_CHECKPOINT="$(readlink -f "$WORK_DIR/latest.pth")"
  test -f "$RESUME_CHECKPOINT"
  "$V5_PYTHON" -m hod26.v5.load_gate \
    --contract "$STATE_LOAD_CONTRACT" \
    checkpoint \
    --checkpoint "$RESUME_CHECKPOINT" \
    --mode resume \
    --updates-per-epoch 1406
  SELECTED_CHECKPOINT="$RESUME_CHECKPOINT"
  INITIALIZATION_DECLARATION="Ordinary V5 full-state resume from the selected epoch checkpoint after strict model/EMA/optimizer/scaler/lineage validation"
  CFG_OPTIONS+=("custom_hooks.1.resume_from=$RESUME_CHECKPOINT")
  echo "State-load gate passed for ordinary V5 continuation: $RESUME_CHECKPOINT"
else
  test -f "$STATEFUL_INIT"
  "$V5_PYTHON" -m hod26.v5.load_gate \
    --contract "$STATE_LOAD_CONTRACT" \
    checkpoint \
    --checkpoint "$STATEFUL_INIT" \
    --mode bootstrap
  SELECTED_CHECKPOINT="$STATEFUL_INIT"
  INITIALIZATION_DECLARATION="V4 e30 raw+EMA plus every parent AdamW state; exact spectral collapse; official D-FINE-X FDR/LQE; 60 genuinely new tensors fresh; parent FP16 scaler exists but is explicitly reset for changed-graph recalibration"
  CFG_OPTIONS+=("custom_hooks.1.resume_from=$STATEFUL_INIT")
  echo "State-load gate passed for V4-e30 -> V5 bootstrap: $STATEFUL_INIT"
fi

TOTAL_ATTEMPTED_UPDATES=$((V5_MAX_EPOCHS * 1406))
printf '%s\n' \
  "V5 formal run contract" \
  "  initialization: V4 e30 raw+EMA and all 1015 parent AdamW states; parent attempted iter=42180, successful AdamW step=42168" \
  "  mapping: exact raw16 reparameterization; official D-FINE-X FDR/LQE; 15 declared semantic salience moment clones" \
  "  fresh state scope: 60 genuinely new FDR/LQE tensors; FP16 scaler deliberately recalibrates from 1 although V4 serialized scale=8" \
  "  ordinary resume: strict model/EMA schema, optimizer name/order/shape, moments, scaler, lineage and successful-step gate" \
  "  devices/global batch: 2 x RTX 4090; 1 image/GPU; global batch 2" \
  "  budget coordinate: ${V5_MAX_EPOCHS} epochs x 1406 = ${TOTAL_ATTEMPTED_UPDATES} attempted updates; successful steps and AMP skips are audited separately" \
  "  stages: transition 0-4k; Mosaic@0.18 4k-36k; clean high-resolution 36k-90k; polish 90k-118104" \
  "  validation/checkpoint: every 2 epochs = 2812 attempted updates; model/EMA/optimizer/scaler resume state; RNG stream is not bitwise-restored" \
  "  expected wall time: about 44-52 hours for 84 epochs after final GO-LSD smoke calibration" \
  "  stopping: no early stop; epoch 84 is soft and must be extended if boundary metrics rise"

LAUNCH_COMMAND="CUDA_VISIBLE_DEVICES=0,1 V5_MAX_EPOCHS=$V5_MAX_EPOCHS bash tools/train_v5.sh"
"$V5_PYTHON" -m hod26.audit \
  --candidate v5_co_dino_vitl_fdr_salience_fold0 \
  --command "$LAUNCH_COMMAND" \
  --config configs/v5/candidate_manifest.yaml \
  --source configs/v5/state_load_contract.json \
  --source configs/v5/co_dino_vitl_fdr_salience_fold0.py \
  --source configs/v4/co_dino_vitl_hsi_fold0.py \
  --source src/hod26/config.py \
  --source src/hod26/data.py \
  --source src/hod26/submission.py \
  --source src/hod26/v4/__init__.py \
  --source src/hod26/v4/compat.py \
  --source src/hod26/v4/model.py \
  --source src/hod26/v4/data.py \
  --source src/hod26/v4/hooks.py \
  --source src/hod26/v4/optim.py \
  --source src/hod26/v4/infer.py \
  --source src/hod26/v5/__init__.py \
  --source src/hod26/v5/model.py \
  --source src/hod26/v5/data.py \
  --source src/hod26/v5/hooks.py \
  --source src/hod26/v5/optim.py \
  --source src/hod26/v5/checkpoint.py \
  --source src/hod26/v5/stateful.py \
  --source src/hod26/v5/load_gate.py \
  --source docs/v5_training_contract.md \
  --source tools/prepare_v5.sh \
  --source tools/smoke_v5.sh \
  --source tools/train_v5.sh \
  --source tools/infer_v5_champion.sh \
  --data storage/v2/dataset/fold0/train.json \
  --data storage/v2/dataset/fold0/val.json \
  --data storage/v4/dataset/fold0_band_stats.json \
  --parent-checkpoint "$SELECTED_CHECKPOINT" \
  --lineage-manifest storage/pretrained/v5/v4_e30_stateful_to_v5_init.pth.provenance.json \
  --external-pretraining-provenance storage/pretrained/v2/deim_dfine_x_object365.pth.provenance.json \
  --initialization "$INITIALIZATION_DECLARATION" \
  --notes "V5 local attempted-update coordinates restart at zero for transition/LR/data staging; inherited AdamW step starts at 42168 and EMA attempted age at 42180. The 84-epoch horizon is 118104 attempted updates; successful steps and AMP skips are checkpoint-audited."

if [[ "${V5_AUDIT_ONLY:-0}" == "1" ]]; then
  echo "V5 audit-only verification complete; formal training was not started."
  exit 0
fi

exec "$V5_ENV/bin/torchrun" \
  --standalone \
  --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py \
  configs/v5/co_dino_vitl_fdr_salience_fold0.py \
  --launcher pytorch \
  --work-dir "$WORK_DIR" \
  --seed 20260730 \
  --cfg-options "${CFG_OPTIONS[@]}"
