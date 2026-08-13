#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

[[ "${V6_FULL_SCRATCH_CONFIRMED:-}" == "YES" ]] || {
  echo "Refusing the all-3,000 formal run: set V6_FULL_SCRATCH_CONFIRMED=YES." >&2
  exit 2
}

CONFIG="configs/v6/co_spec_dino_vitl_full_scratch.py"
WORK_DIR="$REPO_ROOT/storage/v6/runs/co_spec_dino_vitl_full_scratch"
V6_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V6_PYTHON="$V6_ENV/bin/python"
PUBLIC_INIT="$REPO_ROOT/storage/pretrained/v6/co_spec_dino_vitl_public_init.pth"
MAX_EPOCHS="${V6_FULL_SCRATCH_MAX_EPOCHS:-80}"

[[ "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
  echo "V6_FULL_SCRATCH_MAX_EPOCHS must be a positive integer." >&2
  exit 2
}
(( MAX_EPOCHS >= 80 )) || {
  echo "Refusing a truncated Full run: the audited soft horizon is at least 80 epochs." >&2
  exit 2
}
if [[ "${V6_FULL_SCRATCH_AUDIT_ONLY:-0}" != "1" ]] && \
  pgrep -af '[t]ools/train.py.*co_spec_dino_vitl_(fold0|full|full_scratch).py' >/dev/null; then
  echo "Refusing to launch: another V6 Fold0/Full training process is active." >&2
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

# These values are duplicated into checkpoint metadata by both the generic V6
# checkpoint hook and the Full-specific contract hook.
export V6_FULL_START_UPDATE=0
export V6_FULL_END_UPDATE=120000
export V6_FULL_START_RATIO=0.02

CFG_OPTIONS=("runner.max_epochs=$MAX_EPOCHS")
if [[ -e "$WORK_DIR/latest.pth" || -L "$WORK_DIR/latest.pth" ]]; then
  RESUME_CHECKPOINT="$(readlink -f "$WORK_DIR/latest.pth")"
  [[ -f "$RESUME_CHECKPOINT" ]] || {
    echo "Broken V6 Full latest.pth: $WORK_DIR/latest.pth" >&2
    exit 2
  }
  "$V6_PYTHON" -m hod26.v6.full_gate resume \
    --checkpoint "$RESUME_CHECKPOINT" \
    --config "$CONFIG" \
    --source-lock configs/v6/runtime_source_lock_full_scratch.json \
    --target-max-epochs "$MAX_EPOCHS"
  CFG_OPTIONS+=("custom_hooks.1.resume_from=$RESUME_CHECKPOINT")
  SELECTED="$RESUME_CHECKPOINT"
  MODE="strict_full_state_resume"
  INITIALIZATION="Strict all-3,000 resume: model/raw+EMA+AdamW+scaler and Full lineage restored"
else
  if [[ -d "$WORK_DIR" ]] && \
    find "$WORK_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Refusing dirty Full bootstrap directory without latest.pth: $WORK_DIR" >&2
    echo "Archive it explicitly; this launcher never deletes training evidence." >&2
    exit 2
  fi
  "$V6_PYTHON" -m hod26.v6.full_gate public_init \
    --checkpoint "$PUBLIC_INIT" \
    --config "$CONFIG" \
    --source-lock configs/v6/runtime_source_lock_full_scratch.json
  SELECTED="$PUBLIC_INIT"
  MODE="fresh_public_initialized_full_retrain"
  INITIALIZATION="Audited public Co-DINO ViT-L only; fresh V6 AdamW/EMA/scaler; no e34/V4/V5/V5R parent"
fi

TOTAL_ATTEMPTED=$((MAX_EPOCHS * 1500))
printf '%s\n' \
  "V6 all-3,000 complete retraining contract" \
  "  mode: $MODE" \
  "  architecture: one CoSpec-DINO ViT-L; full 368.12M trainable model" \
  "  initialization: $INITIALIZATION" \
  "  data: all 3,000 official labeled images from update zero" \
  "  global batch: 2; 1,500 attempted updates/epoch" \
  "  budget: $MAX_EPOCHS epochs = $TOTAL_ATTEMPTED attempted updates; default soft horizon=120,000" \
  "  LR successful-update points: 0/.02, 1875/1, 7500/1, 70500/.30, 100500/.06, 115000/.015, 120000/.004" \
  "  augmentation exposure stages: 0-7.5k stabilize; 7.5k-70.5k capacity+Mosaic; 70.5k-100.5k clean; 100.5k-120k polish" \
  "  EMA: post-successful-step, 2,500-update exposure-scaled warmup" \
  "  validation: disabled; all labeled images are training data" \
  "  checkpoints: every 2 epochs/3,000 attempted updates; unlimited retention" \
  "  predeclared candidates: e34/51k, e40/60k, e46/69k, e52/78k, e60/90k" \
  "  resume: state-continuous and fail-closed; epoch-boundary RNG is not bitwise-restored" \
  "  stopping: no early stop; e80 is a soft horizon and a later extension resumes the same checkpoint"

LAUNCH_COMMAND="CUDA_VISIBLE_DEVICES=0,1 V6_FULL_SCRATCH_CONFIRMED=YES V6_FULL_SCRATCH_MAX_EPOCHS=$MAX_EPOCHS bash tools/launch_v6_full_scratch_detached.sh"
"$V6_PYTHON" -m hod26.audit \
  --candidate v6_cospec_dino_vitl_full_scratch \
  --command "$LAUNCH_COMMAND" \
  --config configs/v6/candidate_manifest_full_scratch.yaml \
  --source configs/v6/runtime_source_lock_full_scratch.json \
  --source "$CONFIG" \
  --source src/hod26/v6/full.py \
  --source src/hod26/v6/full_gate.py \
  --source docs/v6_full_scratch_contract.md \
  --source tools/train_v6_full_scratch.sh \
  --source tools/launch_v6_full_scratch_detached.sh \
  --source tools/smoke_v6_full_scratch.sh \
  --data storage/v2/dataset/full.json \
  --data storage/v6/dataset/full_band_stats.json \
  --parent-checkpoint "$SELECTED" \
  --external-pretraining-provenance storage/pretrained/v6/co_spec_dino_vitl_public_init.pth.provenance.json \
  --initialization "$INITIALIZATION" \
  --notes "All-3,000 complete retrain from the sole public checkpoint. No 200-image sentinel and no V6 e34 continuation. Candidate epochs were fixed before training."

if [[ "${V6_FULL_SCRATCH_AUDIT_ONLY:-0}" == "1" ]]; then
  echo "V6 Full audit-only verification complete; formal training was not started."
  exit 0
fi

mkdir -p "$WORK_DIR"
exec "$V6_ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py "$CONFIG" \
  --launcher pytorch --no-validate --work-dir "$WORK_DIR" --seed 20260808 \
  --cfg-options "${CFG_OPTIONS[@]}"
