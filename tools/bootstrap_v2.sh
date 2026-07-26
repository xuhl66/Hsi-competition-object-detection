#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
storage_root="${repo_root}/storage"
upstream_root="${storage_root}/upstream/DEIM"
environment_root="${storage_root}/envs/deim-v2"
checkpoint="${storage_root}/pretrained/v2/deim_dfine_x_object365.pth"
checkpoint_url="https://drive.usercontent.google.com/download?id=1RMNrHh3bYN0FfT5ZlWhXtQxkG23xb2xj&export=download&confirm=t"
checkpoint_sha="f240a1128280eb6ce0fff911e8811a8dff669e18be25f5160db7b4263452277d"
upstream_commit="09d35d53d39ee3145a1e61e3a989b28b9468d1dd"

mkdir -p "${storage_root}/upstream" "${storage_root}/pretrained/v2"
if [[ ! -d "${upstream_root}/.git" ]]; then
  git clone --filter=blob:none \
    https://github.com/Intellindust-AI-Lab/DEIM.git \
    "${upstream_root}"
fi
git -C "${upstream_root}" fetch origin "${upstream_commit}"
git -C "${upstream_root}" checkout --detach "${upstream_commit}"

if [[ ! -f "${checkpoint}" ]]; then
  curl -L --fail --retry 3 --retry-delay 2 \
    "${checkpoint_url}" \
    -o "${checkpoint}.part"
  mv "${checkpoint}.part" "${checkpoint}"
fi
printf '%s  %s\n' "${checkpoint_sha}" "${checkpoint}" | sha256sum --check -

if [[ ! -x "${environment_root}/bin/python" ]]; then
  python -m venv --system-site-packages "${environment_root}"
fi
"${environment_root}/bin/python" -m pip install \
  "faster-coco-eval>=1.6.5" tensorboard
"${environment_root}/bin/python" -m pip install \
  --no-build-isolation --no-deps -e "${repo_root}"
"${environment_root}/bin/python" -m hod26.v2.prepare

echo "V2 bootstrap complete: ${environment_root}"
