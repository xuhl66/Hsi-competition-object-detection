#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
bash tools/prepare_v4_codino_vitl.sh

V4_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V4_PYTHON="$V4_ENV/bin/python"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

V4_MAX_EPOCHS="${V4_MAX_EPOCHS:-72}"
if [[ ! "$V4_MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "V4_MAX_EPOCHS must be a positive integer" >&2
  exit 2
fi

WORK_DIR="$REPO_ROOT/storage/v4/runs/co_dino_vitl_hsi_fold0"
CFG_OPTIONS=("runner.max_epochs=$V4_MAX_EPOCHS")
LATEST_CHECKPOINT="$WORK_DIR/latest.pth"
if [[ -e "$LATEST_CHECKPOINT" || -L "$LATEST_CHECKPOINT" ]]; then
  RESUME_CHECKPOINT="$(readlink -f "$LATEST_CHECKPOINT")"
  if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "Broken V4 latest checkpoint: $LATEST_CHECKPOINT" >&2
    exit 2
  fi
  # The EMA hook must register its buffers before runner.resume() loads them.
  # Passing resume_from to the hook preserves raw weights, EMA weights,
  # optimizer, scaler, epoch and update count as one continuous run.
  CFG_OPTIONS+=("custom_hooks.1.resume_from=$RESUME_CHECKPOINT")
  echo "Resume-safe V4 continuation: $RESUME_CHECKPOINT"
else
  echo "Fresh V4 initialization: storage/pretrained/v4/co_dino_vitl_hod26_init.pth"
fi

TOTAL_UPDATES=$((V4_MAX_EPOCHS * 1406))
printf '%s\n' \
  "V4 formal run contract" \
  "  initialization: official Co-DINO ViT-L COCO -> semantic 18-class transfer -> zero-start 16-band/object-aware refinement" \
  "  devices/global batch: 2 x RTX 4090, 1 image/GPU, global batch 2" \
  "  exposure/update budget: 2812 exposures/epoch, 1406 updates/epoch, ${V4_MAX_EPOCHS} epochs = ${TOTAL_UPDATES} updates" \
  "  model: 369146994 parameters; 1500-query Co-DINO + collaborative ROI/ATSS + object activation/CAFR" \
  "  stages: stabilize [0,4), strong+native16 Mosaic@0.25 [4,40), refine [40,56), localization [56,${V4_MAX_EPOCHS}); MixUp off" \
  "  validation/checkpoint: every 2 epochs = 2812 updates; classwise AP, AP75, small AP and weak4 AP; keep 12 + best" \
  "  continuation: complete EMA/optimizer/FP16/epoch/update resume; LR clamps at 0.003x after update 101232" \
  "  expected wall time for 72 epochs: about 30-36 hours on this host" \
  "  stopping: no automatic early stop; stop only after a multi-window plateau at low LR with best away from the boundary"

LAUNCH_COMMAND="V4_MAX_EPOCHS=$V4_MAX_EPOCHS bash tools/train_v4_codino_vitl.sh"
"$V4_PYTHON" -m hod26.audit \
  --candidate v4_co_dino_vitl_hsi_fold0 \
  --command "$LAUNCH_COMMAND" \
  --config configs/v4/candidate_manifest.yaml \
  --source configs/v4/co_dino_vitl_hsi_fold0.py \
  --source src/hod26/v4/__init__.py \
  --source src/hod26/data.py \
  --source src/hod26/v4/constants.py \
  --source src/hod26/v4/compat.py \
  --source src/hod26/v4/model.py \
  --source src/hod26/v4/data.py \
  --source src/hod26/v4/optim.py \
  --source src/hod26/v4/hooks.py \
  --source src/hod26/v4/checkpoint.py \
  --source patches/codetr_v4_torch21_fp16.patch \
  --source docs/v4_training_contract.md \
  --source tools/prepare_v4_codino_vitl.sh \
  --source tools/train_v4_codino_vitl.sh \
  --data storage/v2/dataset/fold0/train.json \
  --data storage/v2/dataset/fold0/val.json \
  --data storage/v4/dataset/fold0_band_stats.json \
  --parent-checkpoint storage/pretrained/v4/co_dino_vitl_hod26_init.pth \
  --external-pretraining-provenance \
    storage/pretrained/co-detr/pytorch_model.pth.provenance.json \
  --initialization \
    "official Co-DINO ViT-L COCO checkpoint; semantic 18-class transfer; zero-start native 16-band P2-P5 fusion and object-aware refinement" \
  --notes \
    "72 epochs is a 101232-update soft horizon, not an automatic convergence claim. Native 16-band Mosaic is strong-stage-only at p=0.25; MixUp is disabled."

if [[ "${V4_AUDIT_ONLY:-0}" == "1" ]]; then
  echo "V4 audit-only verification complete; formal training was not started."
  exit 0
fi

exec "$V4_ENV/bin/torchrun" \
  --standalone \
  --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py \
  configs/v4/co_dino_vitl_hsi_fold0.py \
  --launcher pytorch \
  --work-dir "$WORK_DIR" \
  --seed 20260728 \
  --cfg-options "${CFG_OPTIONS[@]}"
