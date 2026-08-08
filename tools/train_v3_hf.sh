#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

run_dir="storage/v3/runs/hf_hsi_deim_x_fold0"
v2_checkpoint="storage/v2/submissions/checkpoints/v2_long_e228_map071226.pth"
v3_checkpoint="${run_dir}/last.pth"
mkdir -p "${run_dir}"

explicit_initialization=false
for argument in "$@"; do
  case "${argument}" in
    --resume|-r|--resume=*|-r=*|--tuning|-t|--tuning=*|-t=*)
      explicit_initialization=true
      break
      ;;
  esac
done

initialization_arguments=()
if [[ "${explicit_initialization}" == false ]]; then
  if [[ -f "${v3_checkpoint}" ]]; then
    initialization_arguments=(--resume "${v3_checkpoint}")
  elif [[ -f "${v2_checkpoint}" ]]; then
    initialization_arguments=(--tuning "${v2_checkpoint}")
  else
    echo "No V3 resume or submitted V2 e228 checkpoint was found." >&2
    echo "Expected ${v3_checkpoint} or ${v2_checkpoint}." >&2
    exit 2
  fi
fi

echo "V3 HF-HSI-DEIM-X: convergence-controlled, 100000-update guard."
if [[ ${#initialization_arguments[@]} -gt 0 ]]; then
  echo "Automatic initialization: ${initialization_arguments[*]}"
else
  echo "Using the explicit initialization supplied on the command line."
fi

export OMP_NUM_THREADS=8
exec storage/envs/deim-v2/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m hod26.v2.train \
  --config configs/v3/hf_hsi_deim_x_fold0.yaml \
  "${initialization_arguments[@]}" \
  --use-amp \
  "$@"
