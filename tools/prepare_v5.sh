#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
V5_ENV="$REPO_ROOT/storage/envs/codino-v4-smoke"
V5_PYTHON="$V5_ENV/bin/python"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
STATE_LOAD_CONTRACT=configs/v5/state_load_contract.json

test -x "$V5_PYTHON"
test -f storage/v4/runs/co_dino_vitl_hsi_fold0/best_bbox_mAP_epoch_30.pth
test -f storage/pretrained/v2/deim_dfine_x_object365.pth
test -f storage/pretrained/v2/deim_dfine_x_object365.pth.provenance.json
test -f storage/v4/dataset/fold0_band_stats.json
test -f "$STATE_LOAD_CONTRACT"
"$V5_PYTHON" -m hod26.v5.load_gate \
  --contract "$STATE_LOAD_CONTRACT" \
  sources

INIT=storage/pretrained/v5/v4_e30_dfine_to_v5_init.pth
CONFIG=configs/v5/co_dino_vitl_fdr_salience_fold0.py
CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print $1}')"
INIT_REPORT="${INIT%.pth}.pth.provenance.json"
INIT_REPORT_CONFIG_SHA="$(
  "$V5_PYTHON" - "$INIT_REPORT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(json.loads(path.read_text(encoding="utf-8"))["config_sha256"] if path.is_file() else "")
PY
)"
if [[ ! -f "$INIT" || "$INIT_REPORT_CONFIG_SHA" != "$CONFIG_SHA" ]]; then
  "$V5_PYTHON" -m hod26.v5.checkpoint
fi

"$V5_PYTHON" - <<'PY'
import json
from pathlib import Path
from hod26.v5.checkpoint import (
    DFINE_X_SHA256,
    V4_E30_SHA256,
    sha256_file,
)

parent = Path("storage/v4/runs/co_dino_vitl_hsi_fold0/best_bbox_mAP_epoch_30.pth")
external = Path("storage/pretrained/v2/deim_dfine_x_object365.pth")
initial = Path("storage/pretrained/v5/v4_e30_dfine_to_v5_init.pth")
provenance = initial.with_suffix(".pth.provenance.json")
if sha256_file(parent) != V4_E30_SHA256:
    raise SystemExit("V4 e30 SHA mismatch")
if sha256_file(external) != DFINE_X_SHA256:
    raise SystemExit("D-FINE SHA mismatch")
report = json.loads(provenance.read_text(encoding="utf-8"))
if report["output_sha256"] != sha256_file(initial):
    raise SystemExit("V5 initialization SHA/provenance mismatch")
if report["config_sha256"] != sha256_file(
    Path("configs/v5/co_dino_vitl_fdr_salience_fold0.py")
):
    raise SystemExit(
        "V5 initialization is stale; rerun python -m hod26.v5.checkpoint"
    )
if report["dfine_fdr_lqe_tensors"] != 60 or not report["transition_zero"]:
    raise SystemExit("V5 FDR/LQE mapping is incomplete")
print("V5 preparation ready")
print(f"init_sha256={report['output_sha256']}")
PY

STATEFUL=storage/pretrained/v5/v4_e30_stateful_to_v5_init.pth
STATEFUL_REPORT="${STATEFUL%.pth}.pth.provenance.json"
REBUILD_STATEFUL="$(
  "$V5_PYTHON" - "$STATEFUL" "$STATEFUL_REPORT" "$CONFIG_SHA" \
    "$INIT_REPORT" <<'PY'
import json
import sys
from pathlib import Path

stateful, report_path, config_sha, base_report_path = map(Path, sys.argv[1:])
if not stateful.is_file() or not report_path.is_file():
    print(1)
    raise SystemExit
report = json.loads(report_path.read_text(encoding="utf-8"))
base = json.loads(base_report_path.read_text(encoding="utf-8"))
stale = (
    report.get("config_sha256") != str(config_sha)
    or report.get("base_v5_initialization_sha256") != base.get("output_sha256")
)
print(int(stale))
PY
)"
if [[ "$REBUILD_STATEFUL" == "1" ]]; then
  "$V5_PYTHON" -m hod26.v5.stateful
fi

"$V5_PYTHON" - "$STATEFUL" "$STATEFUL_REPORT" "$CONFIG_SHA" <<'PY'
import json
import sys
from pathlib import Path

from hod26.v5.checkpoint import V4_E30_SHA256, sha256_file

stateful, report_path, config_sha = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report["output_sha256"] != sha256_file(stateful):
    raise SystemExit("V5 stateful initialization SHA/provenance mismatch")
if report["config_sha256"] != str(config_sha):
    raise SystemExit("V5 stateful initialization config mismatch")
if report["parent_sha256"] != V4_E30_SHA256:
    raise SystemExit("V5 optimizer parent SHA mismatch")
if report["parent_epoch"] != 30 or report["parent_iter"] != 42180:
    raise SystemExit("V5 optimizer parent runner position mismatch")
if report["parent_optimizer_states"] != 1015:
    raise SystemExit("V5 did not account for every V4 optimizer state")
if report["transplanted_optimizer_states"] != 1030:
    raise SystemExit("V5 optimizer transplant tensor count mismatch")
if report["fresh_optimizer_states"] != 60:
    raise SystemExit("Only the 60 new FDR/LQE tensors may start fresh")
if report["ema_inherited_updates"] != 42180:
    raise SystemExit("V5 EMA age was not inherited")
print("V5 stateful preparation ready")
print(f"stateful_init_sha256={report['output_sha256']}")
print(
    "optimizer_states="
    f"{report['transplanted_optimizer_states']} transplanted, "
    f"{report['fresh_optimizer_states']} genuinely new"
)
PY
