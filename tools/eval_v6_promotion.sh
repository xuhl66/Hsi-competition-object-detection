#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ $# -eq 2 ]] || {
  echo "Usage: bash tools/eval_v6_promotion.sh CHECKPOINT OUTPUT_DIR" >&2
  exit 2
}
CHECKPOINT="$(readlink -f "$1")"
OUTPUT_DIR="$2"
V6_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V6_PYTHON="$V6_ENV/bin/python"
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Refusing existing output: $OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"

"$V6_PYTHON" -m hod26.v6.load_gate resume \
  --checkpoint "$CHECKPOINT" \
  --config configs/v6/co_spec_dino_vitl_fold0.py \
  --source-lock configs/v6/runtime_source_lock.json \
  >"$OUTPUT_DIR/checkpoint_gate.json"

CUDA_VISIBLE_DEVICES=0 "$V6_PYTHON" -m hod26.v6.evaluate \
  --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT_DIR" --device cuda:0 \
  --modes original exposure_low spectral_slope \
  >"$OUTPUT_DIR/gpu0.console.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 "$V6_PYTHON" -m hod26.v6.evaluate \
  --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT_DIR" --device cuda:0 \
  --modes band_response noise_blur \
  >"$OUTPUT_DIR/gpu1.console.log" 2>&1 &
PID1=$!
wait "$PID0"
wait "$PID1"

"$V6_PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
modes = ("original", "exposure_low", "spectral_slope", "band_response", "noise_blur")
reports = {
    mode: json.loads((root / f"{mode}.metrics.json").read_text(encoding="utf-8"))
    for mode in modes
}
baseline = reports["original"]["metrics"]
summary = {
    "gate": "passed",
    "inference_protocol": reports["original"]["inference_protocol"],
    "original": {
        "bbox_mAP": baseline["bbox_mAP"],
        "bbox_mAP_75": baseline["bbox_mAP_75"],
        "bbox_mAP_s": baseline["bbox_mAP_s"],
        "bbox_weak4_mAP": baseline["bbox_weak4_mAP"],
        "bbox_strong14_mAP": baseline["bbox_strong14_mAP"],
    },
    "sensor_shift_bbox_mAP": {
        mode: reports[mode]["metrics"]["bbox_mAP"] for mode in modes[1:]
    },
    "sensor_shift_delta": {
        mode: reports[mode]["metrics"]["bbox_mAP"] - baseline["bbox_mAP"]
        for mode in modes[1:]
    },
}
(root / "promotion_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "V6 frozen-protocol Fold0 and sensor-shift evaluation passed: $OUTPUT_DIR/promotion_summary.json"
