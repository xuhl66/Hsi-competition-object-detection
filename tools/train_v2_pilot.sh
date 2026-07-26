#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
run_dir="storage/v2/runs/deim_dfine_x_hsi_fold0_pilot"
mkdir -p "${run_dir}"
if [[ -f "${run_dir}/last.pth" && " $* " != *" --resume "* ]]; then
  echo "Refusing to overwrite an existing V2 pilot." >&2
  echo "Resume with: tools/train_v2_pilot.sh --resume ${run_dir}/last.pth" >&2
  exit 2
fi
initialization=(--tuning storage/pretrained/v2/deim_dfine_x_object365.pth)
for argument in "$@"; do
  if [[ "${argument}" == "--resume" || "${argument}" == "-r" ]]; then
    initialization=()
    break
  fi
done
export OMP_NUM_THREADS=8
exec storage/envs/deim-v2/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m hod26.v2.train \
  --config configs/v2/deim_dfine_x_hsi_pilot.yaml \
  "${initialization[@]}" \
  --use-amp \
  "$@"
