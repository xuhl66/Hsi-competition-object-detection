#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CONFIG="configs/v6/co_spec_dino_vitl_fold0.py"
WORK_DIR="$REPO_ROOT/storage/v6/runs/co_spec_dino_vitl_fold0"
V6_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V6_PYTHON="$V6_ENV/bin/python"

if [[ "${V6_AUDIT_ONLY:-0}" != "1" ]] && \
  pgrep -af '[t]ools/train.py.*co_spec_dino_vitl_fold0.py' >/dev/null; then
  echo "Refusing to launch: an active V6 Fold0 process already exists." >&2
  exit 2
fi
bash tools/prepare_v6.sh

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

V6_MAX_EPOCHS="${V6_MAX_EPOCHS:-80}"
[[ "$V6_MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
  echo "V6_MAX_EPOCHS must be a positive integer" >&2; exit 2;
}
INIT="$REPO_ROOT/storage/pretrained/v6/co_spec_dino_vitl_public_init.pth"
CFG_OPTIONS=("runner.max_epochs=$V6_MAX_EPOCHS")

if [[ -e "$WORK_DIR/latest.pth" || -L "$WORK_DIR/latest.pth" ]]; then
  RESUME_CHECKPOINT="$(readlink -f "$WORK_DIR/latest.pth")"
  [[ -f "$RESUME_CHECKPOINT" ]] || { echo "Broken V6 latest.pth" >&2; exit 2; }
  "$V6_PYTHON" -m hod26.v6.load_gate resume \
    --checkpoint "$RESUME_CHECKPOINT" --config "$CONFIG" \
    --source-lock configs/v6/runtime_source_lock.json
  CFG_OPTIONS+=("custom_hooks.1.resume_from=$RESUME_CHECKPOINT")
  SELECTED="$RESUME_CHECKPOINT"
  INITIALIZATION="Strict ordinary V6 state-continuous resume; model/EMA/AdamW/scaler/source lock verified"
else
  if [[ -d "$WORK_DIR" ]] && \
    find "$WORK_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Refusing dirty V6 bootstrap: $WORK_DIR is non-empty without latest.pth." >&2
    echo "Archive the directory explicitly; this script never clears it." >&2
    exit 2
  fi
  "$V6_PYTHON" -m hod26.v6.load_gate public_init \
    --checkpoint "$INIT" --config "$CONFIG" \
    --source-lock configs/v6/runtime_source_lock.json
  SELECTED="$INIT"
  INITIALIZATION="One official public Co-DINO ViT-L O365-to-COCO checkpoint; fresh V6 AdamW/EMA/scaler; no V4/V5/V5R parent"
fi

TOTAL_ATTEMPTED=$((V6_MAX_EPOCHS * 1200))
printf '%s\n' \
  "V6 Fold0 formal run contract" \
  "  architecture: one CoSpec-DINO ViT-L; one shared inference decoder" \
  "  initialization: $INITIALIZATION" \
  "  external source: public Co-DINO SHA256=733d2ccde180a55151a68a6cab7c9f42b117d24d38d6197b37caf3189243256c" \
  "  trainable scope: full 368.12M-parameter model; not head-only" \
  "  LR peaks: detector=1.25e-5; V6 spectral/fusion=1.0e-4; class coordinates=5.0e-5; ViT-L top~=1.25e-5 with 0.90 layer decay" \
  "  budget: ${V6_MAX_EPOCHS} x 1200 = ${TOTAL_ATTEMPTED} attempted updates; successful AdamW updates and AMP skips audited separately" \
  "  stages: 0-6k stabilize/open fusion; 6k-56.4k capacity+Mosaic; 56.4k-80.4k clean; 80.4k-96k AP75/weak-class polish" \
  "  validation/checkpoint: every 2 epochs = 2400 attempted updates; 18-class AP, AP75, APs, weak4 and strong14" \
  "  resume: full model/raw+EMA/optimizer/scaler; source and parameter-order gate; state-continuous, not bitwise RNG resume" \
  "  stopping: no early stopping; 96k is a soft horizon and extends from the same checkpoint if any promotion metric is boundary-rising"

LAUNCH_COMMAND="CUDA_VISIBLE_DEVICES=0,1 V6_MAX_EPOCHS=$V6_MAX_EPOCHS bash tools/launch_v6_detached.sh"
"$V6_PYTHON" -m hod26.audit \
  --candidate v6_cospec_dino_vitl_fold0 \
  --command "$LAUNCH_COMMAND" \
  --config configs/v6/candidate_manifest.yaml \
  --source configs/v6/runtime_source_lock.json \
  --source "$CONFIG" \
  --source src/hod26/v6/constants.py \
  --source src/hod26/v6/compat.py \
  --source src/hod26/v6/data.py \
  --source src/hod26/v6/model.py \
  --source src/hod26/v6/optim.py \
  --source src/hod26/v6/hooks.py \
  --source src/hod26/v6/checkpoint.py \
  --source src/hod26/v6/load_gate.py \
  --source docs/v6_diagnosis_and_design.md \
  --source tools/prepare_v6.sh \
  --source tools/smoke_v6.sh \
  --source tools/train_v6.sh \
  --source tools/launch_v6_detached.sh \
  --data storage/v2/dataset/fold0/train.json \
  --data storage/v2/dataset/fold0/val.json \
  --data storage/v6/dataset/fold0_band_stats.json \
  --parent-checkpoint "$SELECTED" \
  --external-pretraining-provenance storage/pretrained/v6/co_spec_dino_vitl_public_init.pth.provenance.json \
  --initialization "$INITIALIZATION" \
  --notes "Independent V6 lineage. V4/V5/V5R checkpoints are prohibited. Epoch 80/96k is a soft horizon, not a convergence declaration."

if [[ "${V6_AUDIT_ONLY:-0}" == "1" ]]; then
  echo "V6 audit-only verification complete; formal training was not started."
  exit 0
fi

mkdir -p "$WORK_DIR"
exec "$V6_ENV/bin/torchrun" \
  --standalone --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py "$CONFIG" \
  --launcher pytorch --work-dir "$WORK_DIR" --seed 20260808 \
  --cfg-options "${CFG_OPTIONS[@]}"
