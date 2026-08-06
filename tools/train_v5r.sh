#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CONFIG=configs/v5r/co_dino_vitl_fdr_prehead_fold0.py
WORK_DIR="$REPO_ROOT/storage/v5r/runs/co_dino_vitl_fdr_prehead_fold0"

if [[ "${V5R_AUDIT_ONLY:-0}" != "1" ]] &&
  pgrep -af '[t]ools/train.py.*co_dino_vitl_fdr_prehead_fold0.py' >/dev/null; then
  echo "Refusing to launch: an active V5R formal process already exists." >&2
  exit 2
fi
bash tools/prepare_v5r.sh

V5R_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V5R_PYTHON="$V5R_ENV/bin/python"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

V5R_MAX_EPOCHS="${V5R_MAX_EPOCHS:-96}"
[[ "$V5R_MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
  echo "V5R_MAX_EPOCHS must be a positive integer" >&2
  exit 2
}
STATEFUL_INIT="$REPO_ROOT/storage/pretrained/v5r/v4_e30_stateful_to_v5r_init.pth"
CONTRACT="$REPO_ROOT/configs/v5r/state_load_contract.json"
CFG_OPTIONS=("runner.max_epochs=$V5R_MAX_EPOCHS")

if [[ -e "$WORK_DIR/latest.pth" || -L "$WORK_DIR/latest.pth" ]]; then
  RESUME_CHECKPOINT="$(readlink -f "$WORK_DIR/latest.pth")"
  test -f "$RESUME_CHECKPOINT"
  "$V5R_PYTHON" -m hod26.v5.load_gate \
    --contract "$CONTRACT" checkpoint \
    --checkpoint "$RESUME_CHECKPOINT" \
    --mode resume \
    --config "$CONFIG" \
    --updates-per-epoch 1406
  SELECTED_CHECKPOINT="$RESUME_CHECKPOINT"
  INITIALIZATION_DECLARATION="Ordinary V5R state-continuous resume after strict model/EMA/optimizer/scaler/source/lineage validation"
  CFG_OPTIONS+=("custom_hooks.1.resume_from=$RESUME_CHECKPOINT")
  echo "State-load gate passed for ordinary V5R continuation: $RESUME_CHECKPOINT"
else
  if [[ -d "$WORK_DIR" ]] &&
    find "$WORK_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Refusing dirty V5R bootstrap: work directory exists and is non-empty." >&2
    echo "Archive it explicitly; this script will never reuse or clear it silently." >&2
    exit 2
  fi
  "$V5R_PYTHON" -m hod26.v5.load_gate \
    --contract "$CONTRACT" checkpoint \
    --checkpoint "$STATEFUL_INIT" \
    --mode bootstrap \
    --config "$CONFIG"
  SELECTED_CHECKPOINT="$STATEFUL_INIT"
  INITIALIZATION_DECLARATION="Clean V4 e30 raw+EMA+1015 exact AdamW states; official D-FINE-X FDR/LQE and V4-cloned independent pre-head/salience weights with 81 explicitly fresh optimizer coordinates; faulty V5 e18 excluded"
  CFG_OPTIONS+=("custom_hooks.1.resume_from=$STATEFUL_INIT")
  echo "State-load gate passed for clean V4-e30 -> V5R bootstrap."
fi

TOTAL_ATTEMPTED_UPDATES=$((V5R_MAX_EPOCHS * 1406))
printf '%s\n' \
  "V5R formal run contract" \
  "  sole parent: V4 e30 SHA256=2b96f909...; attempted iter=42180; successful AdamW step=42168; faulty V5 e18 SHA256=14b8b7a9... is explicitly prohibited" \
  "  inherited: V4 raw+EMA and all 1015 exact-coordinate AdamW states; preserved trainable parameters=99.5742%" \
  "  fresh optimizer: FDR 36 + LQE 24 + independent pre-bbox 6 + multiscale salience 15 = 81 tensors; weights/EMA still use declared official/V4 sources" \
  "  repair: supervised pre-anchor; native Co-DINO aux isolation; continuous native prediction/reference coordinates; localization LR=1x; query-position clamp" \
  "  EMA: inherited attempted age 42180; learned weights update after successful optimizer.step; AMP skips do not age EMA; transition buffers are copied, never smoothed" \
  "  devices/global batch: 2 x RTX 4090; 1 image/GPU; global batch 2" \
  "  budget: ${V5R_MAX_EPOCHS} x 1406 = ${TOTAL_ATTEMPTED_UPDATES} attempted updates; optimizer success and AMP skips are separately audited" \
  "  stages: synchronized transition 0-12k; no Mosaic before e10; Mosaic@0.18 e10-e36; clean high-resolution e36-e72; localization polish e72-e96" \
  "  validation/checkpoint: every 2 epochs = 2812 attempted updates; full model/EMA/optimizer/scaler continuation; RNG is state-continuous but not bitwise-restored" \
  "  expected wall time: about 48-58 hours for 96 epochs on the local dual-4090 host" \
  "  stopping: no early stop; epoch 96 is a soft horizon and must extend from the same checkpoint if boundary metrics rise"

LAUNCH_COMMAND="CUDA_VISIBLE_DEVICES=0,1 V5R_MAX_EPOCHS=$V5R_MAX_EPOCHS bash tools/train_v5r.sh"
"$V5R_PYTHON" -m hod26.audit \
  --candidate v5r_co_dino_vitl_fdr_prehead_fold0 \
  --command "$LAUNCH_COMMAND" \
  --config configs/v5r/candidate_manifest.yaml \
  --source configs/v5r/state_load_contract.json \
  --source "$CONFIG" \
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
  --source src/hod26/v5/__init__.py \
  --source src/hod26/v5/model.py \
  --source src/hod26/v5/data.py \
  --source src/hod26/v5/hooks.py \
  --source src/hod26/v5/load_gate.py \
  --source src/hod26/v5r/__init__.py \
  --source src/hod26/v5r/model.py \
  --source src/hod26/v5r/hooks.py \
  --source src/hod26/v5r/optim.py \
  --source src/hod26/v5r/checkpoint.py \
  --source src/hod26/v5r/stateful.py \
  --source docs/v5r_failure_and_training_contract.md \
  --source tools/prepare_v5r.sh \
  --source tools/smoke_v5r.sh \
  --source tools/train_v5r.sh \
  --source tools/infer_v5_champion.sh \
  --data storage/v2/dataset/fold0/train.json \
  --data storage/v2/dataset/fold0/val.json \
  --data storage/v4/dataset/fold0_band_stats.json \
  --parent-checkpoint "$SELECTED_CHECKPOINT" \
  --lineage-manifest storage/pretrained/v5r/v4_e30_stateful_to_v5r_init.pth.provenance.json \
  --external-pretraining-provenance storage/pretrained/v2/deim_dfine_x_object365.pth.provenance.json \
  --initialization "$INITIALIZATION_DECLARATION" \
  --notes "V5R is a clean lineage. Local transition/LR/data coordinates start at zero; inherited AdamW states start at successful step 42168 and EMA attempted age 42180. The faulty V5 e18 checkpoint is inference-only evidence and cannot enter training."

if [[ "${V5R_AUDIT_ONLY:-0}" == "1" ]]; then
  echo "V5R audit-only verification complete; formal training was not started."
  exit 0
fi

mkdir -p "$WORK_DIR"
exec "$V5R_ENV/bin/torchrun" \
  --standalone \
  --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py \
  "$CONFIG" \
  --launcher pytorch \
  --work-dir "$WORK_DIR" \
  --seed 20260731 \
  --cfg-options "${CFG_OPTIONS[@]}"

