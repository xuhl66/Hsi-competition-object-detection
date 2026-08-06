#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
V5R_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V5R_PYTHON="$V5R_ENV/bin/python"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"

CONFIG=configs/v5r/co_dino_vitl_fdr_prehead_fold0.py
CONTRACT=configs/v5r/state_load_contract.json
PARENT=storage/v4/runs/co_dino_vitl_hsi_fold0/best_bbox_mAP_epoch_30.pth
FAULTY_V5=storage/v5/runs/co_dino_vitl_fdr_salience_fold0/best_bbox_mAP_epoch_18.pth
EXTERNAL=storage/pretrained/v2/deim_dfine_x_object365.pth
BASE_INIT=storage/pretrained/v5r/v4_e30_dfine_to_v5r_init.pth
BASE_REPORT="${BASE_INIT%.pth}.pth.provenance.json"
STATEFUL=storage/pretrained/v5r/v4_e30_stateful_to_v5r_init.pth
STATEFUL_REPORT="${STATEFUL%.pth}.pth.provenance.json"

test -x "$V5R_PYTHON"
test -f "$CONFIG"
test -f "$CONTRACT"
test -f "$PARENT"
test -f "$FAULTY_V5"
test -f "$EXTERNAL"
test -f storage/pretrained/v2/deim_dfine_x_object365.pth.provenance.json
test -f storage/v4/dataset/fold0_band_stats.json

CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print $1}')"
PARENT_SHA="$(sha256sum "$PARENT" | awk '{print $1}')"
FAULTY_SHA="$(sha256sum "$FAULTY_V5" | awk '{print $1}')"
[[ "$PARENT_SHA" == "2b96f909af97c93c01c2141d032eb391428b18aac76be5e38f4f3ae5f9198932" ]]
[[ "$FAULTY_SHA" == "14b8b7a9987cfc18ac32d6139b41d7f31fe664686b38f7e951c6a992c3751ff5" ]]
[[ "$PARENT_SHA" != "$FAULTY_SHA" ]]

REBUILD_BASE="$(
  "$V5R_PYTHON" - "$BASE_INIT" "$BASE_REPORT" "$CONFIG_SHA" <<'PY'
import json
import sys
from pathlib import Path

checkpoint, report_path, config_sha = map(Path, sys.argv[1:])
if not checkpoint.is_file() or not report_path.is_file():
    print(1)
else:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(int(
        report.get("config_sha256") != str(config_sha)
        or report.get("parent_sha256")
        != "2b96f909af97c93c01c2141d032eb391428b18aac76be5e38f4f3ae5f9198932"
        or report.get("prohibited_faulty_v5_e18_sha256")
        != "14b8b7a9987cfc18ac32d6139b41d7f31fe664686b38f7e951c6a992c3751ff5"
    ))
PY
)"
if [[ "$REBUILD_BASE" == "1" ]]; then
  "$V5R_PYTHON" -m hod26.v5r.checkpoint
fi

BASE_SHA="$(
  "$V5R_PYTHON" - "$BASE_INIT" "$BASE_REPORT" <<'PY'
import json
import sys
from pathlib import Path
from hod26.v5.checkpoint import sha256_file

checkpoint, report_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
actual = sha256_file(checkpoint)
if actual != report["output_sha256"]:
    raise SystemExit("V5R base initialization SHA/provenance mismatch")
if report["pre_bbox_tensors_from_parent"] != 6:
    raise SystemExit("V5R independent pre-bbox mapping is incomplete")
if report["dfine_fdr_lqe_tensors"] != 60 or not report["transition_zero"]:
    raise SystemExit("V5R FDR/LQE mapping is incomplete")
print(actual)
PY
)"

REBUILD_STATEFUL="$(
  "$V5R_PYTHON" - "$STATEFUL" "$STATEFUL_REPORT" "$CONFIG_SHA" "$BASE_SHA" <<'PY'
import json
import sys
from pathlib import Path

checkpoint, report_path, config_sha, base_sha = map(Path, sys.argv[1:])
if not checkpoint.is_file() or not report_path.is_file():
    print(1)
else:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(int(
        report.get("config_sha256") != str(config_sha)
        or report.get("base_v5r_initialization_sha256") != str(base_sha)
        or report.get("salience_cloned_states") != 0
        or report.get("fresh_optimizer_states") != 81
    ))
PY
)"
if [[ "$REBUILD_STATEFUL" == "1" ]]; then
  "$V5R_PYTHON" -m hod26.v5r.stateful
fi

"$V5R_PYTHON" -m hod26.v5.load_gate \
  --contract "$CONTRACT" \
  sources
"$V5R_PYTHON" -m hod26.v5.load_gate \
  --contract "$CONTRACT" \
  checkpoint \
  --checkpoint "$STATEFUL" \
  --mode bootstrap \
  --config "$CONFIG"

"$V5R_PYTHON" - "$STATEFUL" "$STATEFUL_REPORT" "$CONFIG_SHA" <<'PY'
import json
import sys
from pathlib import Path
from hod26.v5.checkpoint import sha256_file

checkpoint, report_path, config_sha = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report["output_sha256"] != sha256_file(checkpoint):
    raise SystemExit("V5R stateful initialization SHA/provenance mismatch")
if report["config_sha256"] != str(config_sha):
    raise SystemExit("V5R stateful initialization config mismatch")
expected = {
    "parent_optimizer_states": 1015,
    "target_trainable_tensors": 1096,
    "transplanted_optimizer_states": 1015,
    "fresh_optimizer_states": 81,
    "fresh_fdr_states": 36,
    "fresh_lqe_states": 24,
    "fresh_pre_bbox_states": 6,
    "fresh_salience_states": 15,
    "salience_cloned_states": 0,
    "ema_inherited_updates": 42180,
}
for field, value in expected.items():
    if report.get(field) != value:
        raise SystemExit(f"V5R stateful field mismatch: {field}")
print("V5R clean stateful preparation ready")
print(f"stateful_init_sha256={report['output_sha256']}")
print(
    "optimizer_states="
    f"{report['transplanted_optimizer_states']} exact V4 coordinates, "
    f"{report['fresh_optimizer_states']} declared new/semantic coordinates"
)
PY

