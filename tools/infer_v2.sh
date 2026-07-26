#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: tools/infer_v2.sh CHECKPOINT [OUTPUT.csv]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint="$1"
output="${2:-storage/v2/submissions/deim_dfine_x_hsi.csv}"
cd "${repo_root}"
exec storage/envs/deim-v2/bin/python -m hod26.v2.infer \
  --config configs/v2/deim_dfine_x_hsi_infer.yaml \
  --checkpoint "${checkpoint}" \
  --output "${output}"
