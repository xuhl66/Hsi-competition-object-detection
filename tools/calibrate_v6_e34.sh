#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

python_bin="$repo/storage/envs/codino-v4-smoke/bin/python"
checkpoint="$repo/storage/v6/runs/co_spec_dino_vitl_fold0/best_bbox_mAP_epoch_34.pth"
work="$repo/storage/v6/calibration/e34"
mkdir -p "$work" "$repo/storage/v6/submissions"

export PYTHONPATH="$repo/src:$repo/storage/upstream/Co-DETR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

case "${1:-}" in
  val)
    "$python_bin" -m hod26.v6.load_gate resume \
      --config configs/v6/co_spec_dino_vitl_fold0.py \
      --checkpoint "$checkpoint" >/dev/null
    "$python_bin" -m hod26.v6.calibration extract \
      --checkpoint "$checkpoint" \
      --records storage/v2/dataset/fold0/val.json \
      --records-kind annotation \
      --image-root data/raw/data_train/data_train/VIS \
      --output "$work/val_queries.npz" \
      --device "${V6_CAL_DEVICE:-cuda:0}"
    "$python_bin" -m hod26.v6.calibration diagnose \
      --cache "$work/val_queries.npz" \
      --pool "$work/val_pool.npz" \
      --annotation storage/v2/dataset/fold0/val.json \
      --output "$work/calibration_report.json"
    ;;
  test)
    test -f "$work/calibration_report.json"
    "$python_bin" -m hod26.v6.load_gate resume \
      --config configs/v6/co_spec_dino_vitl_fold0.py \
      --checkpoint "$checkpoint" >/dev/null
    "$python_bin" -m hod26.v6.calibration extract \
      --checkpoint "$checkpoint" \
      --records storage/cache/test_manifest.json \
      --records-kind manifest \
      --image-root data/raw \
      --output "$work/test_queries.npz" \
      --device "${V6_CAL_DEVICE:-cuda:0}"
    "$python_bin" -m hod26.v6.calibration emit \
      --cache "$work/test_queries.npz" \
      --pool "$work/test_pool.npz" \
      --calibration "$work/calibration_report.json" \
      --manifest storage/cache/test_manifest.json \
      --checkpoint "$checkpoint" \
      --output storage/v6/submissions/0811_c.csv
    ;;
  fixed-val)
    test -f "$work/val_queries.npz"
    test -f "$work/val_queries.npz.audit.json"
    "$python_bin" -m hod26.v6.calibration fixed-diagnose \
      --cache "$work/val_queries.npz" \
      --annotation storage/v2/dataset/fold0/val.json \
      --output "$work/fixed_selected_margin_report.json"
    ;;
  fixed-test)
    test -f "$work/fixed_selected_margin_report.json"
    "$python_bin" -m hod26.v6.load_gate resume \
      --config configs/v6/co_spec_dino_vitl_fold0.py \
      --checkpoint "$checkpoint" >/dev/null
    if [[ ! -f "$work/test_queries_fixed.npz" ]]; then
      "$python_bin" -m hod26.v6.calibration extract \
        --checkpoint "$checkpoint" \
        --records storage/cache/test_manifest.json \
        --records-kind manifest \
        --image-root data/raw \
        --output "$work/test_queries_fixed.npz" \
        --device "${V6_CAL_DEVICE:-cuda:0}"
    fi
    "$python_bin" -m hod26.v6.calibration fixed-emit \
      --cache "$work/test_queries_fixed.npz" \
      --calibration "$work/fixed_selected_margin_report.json" \
      --manifest storage/cache/test_manifest.json \
      --checkpoint "$checkpoint" \
      --output storage/v6/submissions/0811_d.csv
    ;;
  *)
    echo "usage: CUDA_VISIBLE_DEVICES=0 bash tools/calibrate_v6_e34.sh {val|test|fixed-val|fixed-test}" >&2
    exit 2
    ;;
esac
