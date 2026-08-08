# HOD Challenge 2026 — single-model competition workspace

> This is an evolving candidate workspace, not a frozen final submission.
> V3 is one candidate and later V4/V5/... versions may replace it. The
> version-neutral final-delivery procedure is in
> [docs/code_review/README.md](docs/code_review/README.md).

The original V1 baseline in this repository used:

```text
16-bit 4×4 spectral mosaic
  -> exact X2Cube demosaicing
  -> 16-band normalization
  -> HYPER-DET dual-stream backbone
  -> HUFF fusion at P2/P3/P4/P5
  -> object-aware auxiliary supervision
  -> single YOLO11-style DFL detector
  -> COCO mAP@[.50:.95] validation
  -> submission.csv with an explicit id column
```

The architecture is a single detector. It does not fuse predictions from
multiple checkpoints. Training candidates and cross-validation are used only
for model selection; a final submission is produced by one checkpoint.

## Storage

Heavy outputs are accessed through `storage/`, which points to the dedicated
mounted disk. Source code and configuration remain in this project.

## Environment

```bash
source .venv/bin/activate
pip install -e .
```

## Prepare and audit

```bash
hod26-prepare --config configs/final.yaml
```

This creates manifests, robust band statistics and deterministic balanced
folds. The official raw files under `data/raw/` are never modified.

## Train

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=1 hod26-train --config configs/final.yaml
```

The configured effective global batch is 36: one GPU automatically uses two
micro-batches, while two GPUs use one micro-batch each. This keeps the
optimization schedule unchanged when a single-GPU checkpoint is resumed under
DDP.

Two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  -m hod26.train --config configs/final.yaml
```

Final 3,000-image training (fixed 120 epochs, no validation holdout):

```bash
CUDA_VISIBLE_DEVICES=1 hod26-train \
  --config configs/full_120.yaml --full-data
```

## Declared external pretraining

The RGB branch can be initialized from the public Ultralytics YOLO11l COCO
checkpoint. This use is recorded in the run manifest and must be declared to
the organizer. The hyperspectral branch and all fusion modules start from
scratch. Set `model.pretrained_rgb: null` to run without external weights.

## Submission

```bash
hod26-infer --config configs/final.yaml --checkpoint storage/checkpoints/FINAL.pt
hod26-submit-check --config configs/final.yaml \
  --csv storage/submissions/submission.csv
```

The rules currently require:

```csv
id,image_id,class_id,confidence,x1,y1,x2,y2
```
