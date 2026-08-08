#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ "${V6_FULL_PROMOTED:-}" == "YES" ]] || {
  echo "Set V6_FULL_PROMOTED=YES only after Fold0 promotion." >&2; exit 2;
}
LOG_DIR="$REPO_ROOT/storage/v6/launch_logs"
PID_FILE="$LOG_DIR/v6_full_detached.pid"
VERIFY_SECONDS="${V6_DETACHED_VERIFY_SECONDS:-10}"
[[ "$VERIFY_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid verify seconds" >&2; exit 2; }
if pgrep -af '[t]ools/train_v6.sh|[t]ools/train_v6_full.sh|[t]ools/train.py.*co_spec_dino_vitl_(fold0|full).py' >/dev/null; then
  echo "Refusing full-data launch: another V6 Fold0/full process is active." >&2
  pgrep -af '[t]ools/train_v6.sh|[t]ools/train_v6_full.sh|[t]ools/train.py.*co_spec_dino_vitl_(fold0|full).py' >&2 || true
  exit 2
fi
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(tr -cd '0-9' <"$PID_FILE")"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Refusing full-data launch: detached V6 PID $OLD_PID is alive." >&2; exit 2
  fi
fi
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/v6_full_${STAMP}.console.log"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/latest_full.console.log"
printf '%s\n' \
  "V6 detached full-data launch" \
  "  log: $LOG" \
  "  transport: nohup + setsid + stdin=/dev/null" \
  "  Fold0 parent: ${V6_FULL_PARENT:-resume latest full checkpoint}"
if [[ "${V6_DETACHED_DRY_RUN:-0}" == "1" ]]; then
  echo "V6 full detached dry-run passed; no training started."; exit 0
fi
nohup setsid env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  V6_FULL_PROMOTED=YES \
  V6_FULL_PARENT="${V6_FULL_PARENT:-}" \
  bash "$REPO_ROOT/tools/train_v6_full.sh" >"$LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
sleep "$VERIFY_SECONDS"
kill -0 "$PID" 2>/dev/null || { tail -100 "$LOG" >&2; exit 1; }
TTY="$(ps -o tty= -p "$PID" | tr -d '[:space:]')"
STATE="$(ps -o stat= -p "$PID" | tr -d '[:space:]')"
[[ "$TTY" == "?" && "$STATE" != Z* ]] || { echo "Detach verification failed" >&2; exit 1; }
printf '%s\n' "Detached V6 full-data startup is alive." "  PID: $PID" "  TTY: $TTY" "  follow: tail -f '$LOG'"
