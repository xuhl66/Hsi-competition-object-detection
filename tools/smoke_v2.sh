#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
exec storage/envs/deim-v2/bin/python -m hod26.v2.smoke \
  --config configs/v2/deim_dfine_x_hsi_pilot.yaml \
  --checkpoint storage/pretrained/v2/deim_dfine_x_object365.pth \
  "$@"

