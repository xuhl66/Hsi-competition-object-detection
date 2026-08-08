#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

checkpoint="${1:-storage/v3/runs/hf_hsi_deim_x_fold0/best.pth}"
if [[ $# -gt 0 ]]; then
  shift
fi

exec storage/envs/deim-v2/bin/python -m hod26.v2.infer \
  --config configs/v3/hf_hsi_deim_x_infer.yaml \
  --checkpoint "${checkpoint}" \
  "$@"
