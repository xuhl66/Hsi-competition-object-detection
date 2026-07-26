#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

run_dir="storage/v2/long_runs/deim_dfine_x_hsi_fold0_long"
pilot_checkpoint="storage/v2/runs/deim_dfine_x_hsi_fold0_pilot/last.pth"
long_checkpoint="${run_dir}/last.pth"
mkdir -p "${run_dir}"

explicit_resume=false
for argument in "$@"; do
  case "${argument}" in
    --resume|-r|--resume=*|-r=*)
      explicit_resume=true
      break
      ;;
  esac
done

resume_arguments=()
if [[ "${explicit_resume}" == false ]]; then
  if [[ -f "${long_checkpoint}" ]]; then
    resume_arguments=(--resume "${long_checkpoint}")
  elif [[ -f "${pilot_checkpoint}" ]]; then
    resume_arguments=(--resume "${pilot_checkpoint}")
  else
    echo "No resumable V2 checkpoint was found." >&2
    echo "Expected ${pilot_checkpoint} or ${long_checkpoint}." >&2
    exit 2
  fi
fi

echo "V2 long run: convergence-controlled, 100000-update safety ceiling."
if [[ ${#resume_arguments[@]} -gt 0 ]]; then
  echo "Automatic resume: ${resume_arguments[1]}"
else
  echo "Using the explicit resume checkpoint from command-line arguments."
fi

export OMP_NUM_THREADS=8
exec storage/envs/deim-v2/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m hod26.v2.train \
  --config configs/v2/deim_dfine_x_hsi_long.yaml \
  "${resume_arguments[@]}" \
  --use-amp \
  "$@"
