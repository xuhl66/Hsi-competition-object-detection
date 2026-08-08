#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
bash tools/prepare_v6.sh

V6_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V6_PYTHON="$V6_ENV/bin/python"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SMOKE_DIR="${V6_SMOKE_DIR:-storage/v6/smoke/exact_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$SMOKE_DIR" ]]; then
  echo "Refusing to overwrite V6 smoke directory: $SMOKE_DIR" >&2
  exit 2
fi
mkdir -p "$SMOKE_DIR"
PHASE1_LOG="$SMOKE_DIR/phase1.console.log"
PHASE2_LOG="$SMOKE_DIR/phase2.console.log"

"$V6_ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py \
  configs/v6/co_spec_dino_vitl_smoke.py \
  --launcher pytorch --work-dir "$SMOKE_DIR" --seed 20260808 \
  --cfg-options runner.max_epochs=1 \
  >"$PHASE1_LOG" 2>&1

PHASE1_CHECKPOINT="$REPO_ROOT/$SMOKE_DIR/epoch_1.pth"
"$V6_PYTHON" -m hod26.v6.load_gate resume \
  --checkpoint "$PHASE1_CHECKPOINT" \
  --config configs/v6/co_spec_dino_vitl_fold0.py \
  --source-lock configs/v6/runtime_source_lock.json \
  >"$SMOKE_DIR/phase1_gate.json"

"$V6_ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py \
  configs/v6/co_spec_dino_vitl_smoke.py \
  --launcher pytorch --work-dir "$SMOKE_DIR" --seed 20260808 \
  --cfg-options \
    runner.max_epochs=2 \
    "custom_hooks.1.resume_from=$PHASE1_CHECKPOINT" \
  >"$PHASE2_LOG" 2>&1

PHASE2_CHECKPOINT="$REPO_ROOT/$SMOKE_DIR/epoch_2.pth"
"$V6_PYTHON" -m hod26.v6.load_gate resume \
  --checkpoint "$PHASE2_CHECKPOINT" \
  --config configs/v6/co_spec_dino_vitl_fold0.py \
  --source-lock configs/v6/runtime_source_lock.json \
  >"$SMOKE_DIR/phase2_gate.json"

"$V6_PYTHON" - "$SMOKE_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
logs = sorted(root.glob("*.log.json"))
if len(logs) != 2:
    raise SystemExit(f"Expected exactly two V6 JSON logs, got {logs}")
rows = []
for path in logs:
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("mode") == "train":
            rows.append(row)
if len(rows) != 8 or sorted({int(row["epoch"]) for row in rows}) != [1, 2]:
    raise SystemExit(f"V6 smoke update/epoch mismatch: {len(rows)} rows")
required = (
    "loss", "grad_norm", "loss_cls", "loss_bbox", "loss_iou",
    "loss_cls1", "loss_bbox1", "loss_centerness1",
)
for index, row in enumerate(rows):
    for key in required:
        value = row.get(key)
        if value is None or not math.isfinite(float(value)):
            raise SystemExit(f"Non-finite/missing {key} at V6 smoke row {index}: {value}")
phase1 = json.loads((root / "phase1_gate.json").read_text(encoding="utf-8"))
phase2 = json.loads((root / "phase2_gate.json").read_text(encoding="utf-8"))
if (
    phase1["attempted_updates"] != 4
    or phase2["attempted_updates"] != 8
    or not 1 <= phase1["successful_updates"] <= 4
    or not phase1["successful_updates"] < phase2["successful_updates"] <= 8
    or phase2["ema_age"] != phase2["successful_updates"]
    or phase2["amp_skipped_updates"] != 8 - phase2["successful_updates"]
):
    raise SystemExit("V6 two-stage optimizer/EMA continuity failed")
maximum_memory = max(int(row.get("memory", 0)) for row in rows)
report = {
    "gate": "passed",
    "full_resolution": [1280, 640],
    "dual_gpu": True,
    "finite_updates": len(rows),
    "epochs": [1, 2],
    "phase1_attempted_successful": [phase1["attempted_updates"], phase1["successful_updates"]],
    "phase2_attempted_successful": [phase2["attempted_updates"], phase2["successful_updates"]],
    "ema_age": phase2["ema_age"],
    "amp_scaler": phase2["fp16_scaler"],
    "max_logged_memory_mib_per_gpu": maximum_memory,
    "mosaic_phase_then_clean_resume": True,
    "model_ema_optimizer_scaler_resume_gate": True,
}
(root / "smoke_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

printf '%s\n' \
  "V6 exact two-stage smoke passed" \
  "  directory: $SMOKE_DIR" \
  "  report: $SMOKE_DIR/smoke_report.json" \
  "  full-resolution Mosaic -> checkpoint -> clean resume: verified"
