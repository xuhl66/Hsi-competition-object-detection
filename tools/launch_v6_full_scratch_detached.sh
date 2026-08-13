#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

[[ "${V6_FULL_SCRATCH_CONFIRMED:-}" == "YES" ]] || {
  echo "Set V6_FULL_SCRATCH_CONFIRMED=YES to acknowledge the 120k-update run." >&2
  exit 2
}

WORK_DIR="$REPO_ROOT/storage/v6/runs/co_spec_dino_vitl_full_scratch"
LOG_DIR="$REPO_ROOT/storage/v6/launch_logs"
PID_FILE="$LOG_DIR/v6_full_scratch_detached.pid"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MAX_EPOCHS="${V6_FULL_SCRATCH_MAX_EPOCHS:-80}"
VERIFY_SECONDS="${V6_DETACHED_VERIFY_SECONDS:-10}"

[[ "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid Full max epochs." >&2; exit 2; }
(( MAX_EPOCHS >= 80 )) || { echo "Full max epochs cannot be below 80." >&2; exit 2; }
[[ "$VERIFY_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid verify seconds." >&2; exit 2; }

if pgrep -af '[t]ools/train_v6.*\.sh|[t]ools/train.py.*co_spec_dino_vitl_(fold0|full|full_scratch).py' >/dev/null; then
  echo "Refusing Full launch: another V6 process is active." >&2
  pgrep -af '[t]ools/train_v6.*\.sh|[t]ools/train.py.*co_spec_dino_vitl_(fold0|full|full_scratch).py' >&2 || true
  exit 2
fi
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(tr -cd '0-9' <"$PID_FILE")"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Refusing Full launch: detached PID $OLD_PID is alive." >&2
    exit 2
  fi
fi
if [[ -d "$WORK_DIR" ]] && \
  find "$WORK_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q . && \
  [[ ! -e "$WORK_DIR/latest.pth" && ! -L "$WORK_DIR/latest.pth" ]]; then
  echo "Refusing dirty Full work directory without a resumable latest.pth." >&2
  exit 2
fi

command -v nohup >/dev/null
command -v setsid >/dev/null
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
CONSOLE_LOG="$LOG_DIR/v6_full_scratch_${STAMP}.console.log"
ln -sfn "$(basename "$CONSOLE_LOG")" "$LOG_DIR/latest_full_scratch.console.log"

printf '%s\n' \
  "V6 detached all-3,000 complete retraining launch" \
  "  repo: $REPO_ROOT" \
  "  CUDA_VISIBLE_DEVICES: $CUDA_DEVICES" \
  "  max epochs: $MAX_EPOCHS" \
  "  work directory: $WORK_DIR" \
  "  console log: $CONSOLE_LOG" \
  "  transport: nohup + new setsid session + stdin=/dev/null" \
  "  survives SSH logout; cannot survive power loss or host reboot"

if [[ "${V6_DETACHED_DRY_RUN:-0}" == "1" ]]; then
  echo "V6 Full detached dry-run passed; no training was started."
  exit 0
fi

nohup setsid env \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  V6_FULL_SCRATCH_CONFIRMED=YES \
  V6_FULL_SCRATCH_MAX_EPOCHS="$MAX_EPOCHS" \
  bash "$REPO_ROOT/tools/train_v6_full_scratch.sh" \
  >"$CONSOLE_LOG" 2>&1 </dev/null &
DETACHED_PID=$!
printf '%s\n' "$DETACHED_PID" >"$PID_FILE"
sleep "$VERIFY_SECONDS"
if ! kill -0 "$DETACHED_PID" 2>/dev/null; then
  echo "Detached V6 Full process exited during startup:" >&2
  tail -100 "$CONSOLE_LOG" >&2 || true
  exit 1
fi
STATE="$(ps -o stat= -p "$DETACHED_PID" | tr -d '[:space:]')"
TTY="$(ps -o tty= -p "$DETACHED_PID" | tr -d '[:space:]')"
SID="$(ps -o sid= -p "$DETACHED_PID" | tr -d '[:space:]')"
if [[ "$STATE" == Z* || "$TTY" != "?" ]]; then
  echo "V6 Full detach verification failed: state=$STATE tty=$TTY" >&2
  exit 1
fi
printf '%s\n' \
  "Detached V6 Full startup is alive." \
  "  PID: $DETACHED_PID" \
  "  SID: $SID" \
  "  TTY: $TTY" \
  "  PID file: $PID_FILE" \
  "  follow: tail -f '$CONSOLE_LOG'"
