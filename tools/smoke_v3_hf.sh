#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

checkpoint="storage/v2/submissions/checkpoints/v2_long_e228_map071226.pth"
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing submitted V2 e228 checkpoint: ${checkpoint}" >&2
  exit 2
fi

export OMP_NUM_THREADS=8
exec storage/envs/deim-v2/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m hod26.v2.train \
  --config configs/v3/hf_hsi_deim_x_smoke.yaml \
  --tuning "${checkpoint}" \
  --use-amp \
  "$@"
