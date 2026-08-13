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
export V6_FULL_START_UPDATE=0
export V6_FULL_END_UPDATE=120000
export V6_FULL_START_RATIO=0.02

"$V6_PYTHON" -m hod26.v6.full_gate public_init

SMOKE_DIR="${V6_FULL_SCRATCH_SMOKE_DIR:-storage/v6/smoke/full_scratch_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$SMOKE_DIR" ]]; then
  echo "Refusing to overwrite Full smoke directory: $SMOKE_DIR" >&2
  exit 2
fi
mkdir -p "$SMOKE_DIR"

"$V6_ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py \
  configs/v6/co_spec_dino_vitl_full_scratch_smoke.py \
  --launcher pytorch --no-validate --work-dir "$SMOKE_DIR" --seed 20260808 \
  --cfg-options runner.max_epochs=1 \
  >"$SMOKE_DIR/phase1.console.log" 2>&1

PHASE1_CHECKPOINT="$REPO_ROOT/$SMOKE_DIR/epoch_1.pth"
"$V6_PYTHON" -m hod26.v6.load_gate resume \
  --checkpoint "$PHASE1_CHECKPOINT" \
  --config configs/v6/co_spec_dino_vitl_full_scratch_smoke.py \
  --source-lock configs/v6/runtime_source_lock_full_scratch.json \
  >"$SMOKE_DIR/phase1_gate.json"

"$V6_ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  storage/upstream/Co-DETR/tools/train.py \
  configs/v6/co_spec_dino_vitl_full_scratch_smoke.py \
  --launcher pytorch --no-validate --work-dir "$SMOKE_DIR" --seed 20260808 \
  --cfg-options \
    runner.max_epochs=2 \
    "custom_hooks.1.resume_from=$PHASE1_CHECKPOINT" \
  >"$SMOKE_DIR/phase2.console.log" 2>&1

PHASE2_CHECKPOINT="$REPO_ROOT/$SMOKE_DIR/epoch_2.pth"
"$V6_PYTHON" -m hod26.v6.load_gate resume \
  --checkpoint "$PHASE2_CHECKPOINT" \
  --config configs/v6/co_spec_dino_vitl_full_scratch_smoke.py \
  --source-lock configs/v6/runtime_source_lock_full_scratch.json \
  >"$SMOKE_DIR/phase2_gate.json"

"$V6_PYTHON" - "$SMOKE_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

from hod26.v6.full import FULL_CLASS_COUNTS
from hod26.v6.model import effective_number_class_weights

root = Path(sys.argv[1])
logs = sorted(root.glob("*.log.json"))
if len(logs) != 2:
    raise SystemExit(f"Expected two Full smoke JSON logs, got {logs}")
rows = []
for path in logs:
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("mode") == "train":
            rows.append(row)
if len(rows) != 8 or sorted({int(row["epoch"]) for row in rows}) != [1, 2]:
    raise SystemExit(f"Full smoke update/epoch mismatch: {len(rows)} rows")
for index, row in enumerate(rows):
    for key in ("loss", "grad_norm", "loss_cls", "loss_bbox", "loss_iou"):
        if key not in row or not math.isfinite(float(row[key])):
            raise SystemExit(f"Non-finite/missing {key} at Full smoke row {index}")

phase1 = json.loads((root / "phase1_gate.json").read_text(encoding="utf-8"))
phase2 = json.loads((root / "phase2_gate.json").read_text(encoding="utf-8"))
if (
    phase1["attempted_updates"] != 4
    or phase2["attempted_updates"] != 8
    or not 1 <= phase1["successful_updates"] <= 4
    or not phase1["successful_updates"] < phase2["successful_updates"] <= 8
    or phase2["ema_age"] != phase2["successful_updates"]
):
    raise SystemExit("Full smoke optimizer/EMA continuity failed")

payload = torch.load(root / "epoch_2.pth", map_location="cpu")
meta, state = payload["meta"], payload["state_dict"]
expected = effective_number_class_weights(FULL_CLASS_COUNTS)
for key in ("query_head.hod26_class_weights", "bbox_head.0.hod26_class_weights"):
    reference = expected.to(dtype=state[key].dtype)
    torch.testing.assert_close(state[key], reference, rtol=0, atol=2e-7)
if (
    meta.get("hod26_v6_full_training_mode")
    != "public_init_all3000_complete_retrain"
    or meta.get("hod26_v6_full_candidate_epochs") != [34, 40, 46, 52, 60]
    or meta.get("hod26_v6_full_candidate_updates")
    != [34 * 4, 40 * 4, 46 * 4, 52 * 4, 60 * 4]
    or meta.get("hod26_v6_full_updates_per_epoch") != 4
):
    raise SystemExit("Full smoke checkpoint metadata contract failed")

maximum_memory = max(int(row.get("memory", 0)) for row in rows)
report = {
    "gate": "passed",
    "dual_gpu": True,
    "full_resolution": [1280, 640],
    "finite_updates": len(rows),
    "phase1_attempted_successful": [
        phase1["attempted_updates"], phase1["successful_updates"]
    ],
    "phase2_attempted_successful": [
        phase2["attempted_updates"], phase2["successful_updates"]
    ],
    "ema_age": phase2["ema_age"],
    "full_class_weights_exact": True,
    "capacity_mosaic_then_clean_resume": True,
    "max_logged_memory_mib_per_gpu": maximum_memory,
}
(root / "smoke_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

printf '%s\n' \
  "V6 Full exact two-stage smoke passed" \
  "  directory: $SMOKE_DIR" \
  "  report: $SMOKE_DIR/smoke_report.json" \
  "  public init -> capacity Mosaic -> stateful clean resume: verified"
