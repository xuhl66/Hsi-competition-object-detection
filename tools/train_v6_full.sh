#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ "${V6_FULL_PROMOTED:-}" == "YES" ]] || {
  echo "Refusing V6 full-data stage: set V6_FULL_PROMOTED=YES only after Fold0 passes every promotion gate." >&2
  exit 2
}
bash tools/prepare_v6.sh

CONFIG="configs/v6/co_spec_dino_vitl_full.py"
WORK_DIR="$REPO_ROOT/storage/v6/runs/co_spec_dino_vitl_full"
V6_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V6_PYTHON="$V6_ENV/bin/python"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if pgrep -af '[t]ools/train.py.*co_spec_dino_vitl_full.py' >/dev/null; then
  echo "Refusing to launch: an active V6 full-data process exists." >&2; exit 2
fi

if [[ -e "$WORK_DIR/latest.pth" || -L "$WORK_DIR/latest.pth" ]]; then
  PARENT="$(readlink -f "$WORK_DIR/latest.pth")"
  MODE="resume_full"
else
  PARENT="${V6_FULL_PARENT:-}"
  [[ -n "$PARENT" && -f "$PARENT" ]] || {
    echo "First full-data launch requires V6_FULL_PARENT=/absolute/promoted_fold0.pth" >&2
    exit 2
  }
  MODE="promoted_fold0_to_full"
  if [[ -d "$WORK_DIR" ]] && find "$WORK_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Refusing dirty V6 full-data bootstrap directory." >&2; exit 2
  fi
fi

"$V6_PYTHON" -m hod26.v6.load_gate resume \
  --checkpoint "$PARENT" --config "$CONFIG" \
  --source-lock configs/v6/runtime_source_lock.json \
  >"${TMPDIR:-/tmp}/hod26_v6_full_parent_gate.json"

mapfile -t COORDINATES < <(
  "$V6_PYTHON" - "$PARENT" "$MODE" <<'PY'
import math
import sys
import torch
from hod26.v6.optim import piecewise_cosine_ratio

checkpoint, mode = sys.argv[1:]
payload = torch.load(checkpoint, map_location="cpu")
meta = payload["meta"]
start = int(meta["hod26_v6_successful_updates"])
epoch = int(meta["epoch"])
fold_points = [(0,.02),(1500,1),(6000,1),(56400,.30),(80400,.06),(92000,.015),(96000,.004)]
start_ratio = min(0.06, piecewise_cosine_ratio(start, fold_points))
if mode == "promoted_fold0_to_full":
    desired = max(1500, int(math.ceil((0.20 * start) / 1500.0)) * 1500)
else:
    # Preserve the already-declared endpoint from the first full checkpoint.
    desired = int(meta.get("hod26_v6_full_target_update", start + 1500)) - start
    desired = max(1500, int(math.ceil(desired / 1500.0)) * 1500)
full_epochs = desired // 1500
print(start)
print(start + desired)
print(f"{start_ratio:.12g}")
print(epoch + full_epochs)
print(full_epochs)
PY
)
V6_FULL_START_UPDATE="${COORDINATES[0]}"
V6_FULL_END_UPDATE="${COORDINATES[1]}"
V6_FULL_START_RATIO="${COORDINATES[2]}"
V6_FULL_TARGET_MAX_EPOCH="${COORDINATES[3]}"
FULL_EPOCHS="${COORDINATES[4]}"
export V6_FULL_START_UPDATE V6_FULL_END_UPDATE V6_FULL_START_RATIO V6_FULL_TARGET_MAX_EPOCH

printf '%s\n' \
  "V6 all-3000 continuation contract" \
  "  mode: $MODE" \
  "  parent: $PARENT" \
  "  data: all 3,000 official labeled images; no external images" \
  "  optimizer/EMA/scaler: strict state-continuous continuation" \
  "  successful-update interval: $V6_FULL_START_UPDATE -> $V6_FULL_END_UPDATE" \
  "  extra full-data epochs: $FULL_EPOCHS at 1,500 attempted updates/epoch" \
  "  LR ratio: $V6_FULL_START_RATIO -> 0.004; clean 1280 only; no Mosaic/MixUp" \
  "  validation: disabled because all labeled images are now training data" \
  "  checkpoint: every full-data epoch"

if [[ "${V6_FULL_AUDIT_ONLY:-0}" == "1" ]]; then
  echo "V6 full-data audit-only verification complete; training was not started."
  exit 0
fi

mkdir -p "$WORK_DIR"
exec "$V6_ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py "$CONFIG" \
  --launcher pytorch --no-validate --work-dir "$WORK_DIR" --seed 20260808 \
  --cfg-options "custom_hooks.1.resume_from=$PARENT"
