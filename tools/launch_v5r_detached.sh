#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORK_DIR="$REPO_ROOT/storage/v5r/runs/co_dino_vitl_fdr_prehead_fold0"
LOG_DIR="$REPO_ROOT/storage/v5r/launch_logs"
PID_FILE="$LOG_DIR/v5r_detached.pid"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MAX_EPOCHS="${V5R_MAX_EPOCHS:-96}"
VERIFY_SECONDS="${V5R_DETACHED_VERIFY_SECONDS:-8}"

[[ "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
  echo "V5R_MAX_EPOCHS must be a positive integer" >&2
  exit 2
}
[[ "$VERIFY_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "V5R_DETACHED_VERIFY_SECONDS must be a positive integer" >&2
  exit 2
}

if pgrep -af '[t]ools/train_v5r.sh|[t]ools/train.py.*co_dino_vitl_fdr_prehead_fold0.py' >/dev/null; then
  echo "Refusing to launch: an active V5R preparation or training process exists." >&2
  pgrep -af '[t]ools/train_v5r.sh|[t]ools/train.py.*co_dino_vitl_fdr_prehead_fold0.py' >&2 || true
  exit 2
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(tr -cd '0-9' <"$PID_FILE")"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Refusing to launch: detached V5R PID $OLD_PID is still alive." >&2
    exit 2
  fi
fi

if [[ -d "$WORK_DIR" ]] &&
  find "$WORK_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Refusing dirty bootstrap: $WORK_DIR is non-empty." >&2
  echo "Archive the previous run explicitly before starting a clean lineage." >&2
  exit 2
fi

command -v nohup >/dev/null
command -v setsid >/dev/null
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
CONSOLE_LOG="$LOG_DIR/v5r_${STAMP}.console.log"
ln -sfn "$(basename "$CONSOLE_LOG")" "$LOG_DIR/latest.console.log"

printf '%s\n' \
  "V5R detached launch contract" \
  "  repo: $REPO_ROOT" \
  "  CUDA_VISIBLE_DEVICES: $CUDA_DEVICES" \
  "  V5R_MAX_EPOCHS: $MAX_EPOCHS" \
  "  console log: $CONSOLE_LOG" \
  "  transport: nohup + new setsid session + stdin=/dev/null" \
  "  note: survives SSH logout; it cannot survive power loss or host reboot"

if [[ "${V5R_DETACHED_DRY_RUN:-0}" == "1" ]]; then
  echo "Dry-run passed; formal training was not started."
  exit 0
fi

nohup setsid env \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  V5R_MAX_EPOCHS="$MAX_EPOCHS" \
  bash "$REPO_ROOT/tools/train_v5r.sh" \
  >"$CONSOLE_LOG" 2>&1 </dev/null &
DETACHED_PID=$!
printf '%s\n' "$DETACHED_PID" >"$PID_FILE"

sleep "$VERIFY_SECONDS"
if ! kill -0 "$DETACHED_PID" 2>/dev/null; then
  echo "Detached V5R process exited during startup. Last console lines:" >&2
  tail -80 "$CONSOLE_LOG" >&2 || true
  exit 1
fi

PROCESS_STATE="$(ps -o stat= -p "$DETACHED_PID" | tr -d '[:space:]')"
PROCESS_TTY="$(ps -o tty= -p "$DETACHED_PID" | tr -d '[:space:]')"
PROCESS_SID="$(ps -o sid= -p "$DETACHED_PID" | tr -d '[:space:]')"
if [[ "$PROCESS_STATE" == Z* || "$PROCESS_TTY" != "?" ]]; then
  echo "Detached-process verification failed: state=$PROCESS_STATE tty=$PROCESS_TTY" >&2
  exit 1
fi

printf '%s\n' \
  "Detached V5R startup is alive." \
  "  PID: $DETACHED_PID" \
  "  SID: $PROCESS_SID" \
  "  TTY: $PROCESS_TTY" \
  "  PID file: $PID_FILE" \
  "  follow console: tail -f '$CONSOLE_LOG'" \
  "  quick status: ps -o pid,ppid,pgid,sid,tty,stat,etime,cmd -p $DETACHED_PID"
