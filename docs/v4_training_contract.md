# HOD26 V4 training contract

V4 is the full-capacity Co-DINO ViT-L route, not a baseline or a short
architecture pilot. Its selection target is the official
`mAP@[.50:.95]`; AP50, speed and parameter count are diagnostic only.

## Architecture

- Official Co-DINO ViT-L: 24 transformer blocks, 1024 embedding width,
  16 attention heads, activation checkpointing and 1,500 queries.
- All 16 X2Cube bands enter a learnable broadband proxy for the pretrained
  ViT and a separate native spectral/spatial P2--P5 path. First spectral
  differences are included in both nonlinear paths.
- Four gated P2--P5 fusion blocks start at exact identity, so the official
  pretrained detector is preserved at update zero while the HSI path learns.
- A box-supervised OSSDet-style activation head predicts a low-level object
  mask. Its intersection/difference objective uses the published
  `alpha=0.6`, `gamma=0.1` setting. The mask gates P2 foreground/background
  evidence and guides CAFR-style bottom-up P2-to-P5 cross-spectral
  refinement. Every feature-changing output projection is zero initialized,
  so this complete branch is an exact identity at update zero while its mask
  supervision starts learning immediately.
- The Co-DINO query head, RPN/ROI head and ATSS head train collaboratively.
  Inference uses one Co-DINO query detector and one checkpoint.
- There are 369,146,994 trainable parameters and 18 output classes.

The initialization is the pinned official Co-DINO ViT-L COCO checkpoint.
Shape-compatible tensors transfer exactly; 19 class tensors receive
semantic COCO-to-HOD26 row initialization; the new HSI tensors use
deterministic initialization, and all feature-changing HSI/object-aware
residuals start at zero. The checkpoint, config, upstream commit and required
compatibility patch are SHA-256 verified before launch.

## Formal Fold0 run

Fold0 training contains 2,400 unique images and controlled 2x image
repetition for `car`, `e-bike`, `people` and `stone_block`. This yields
2,812 exposures and 1,406 attempted optimizer updates per epoch with two GPUs.
`stone_block` is deliberately not sampled 4x.

The initial 72-epoch soft horizon is 101,232 attempted updates:

| Epochs | Update range | Policy |
|---|---:|---|
| 0--4 | 0--5,624 | pretrained stabilization |
| 4--40 | 5,624--56,240 | strong spectral/spatial augmentation |
| 40--56 | 56,240--78,736 | refinement |
| 56--72 | 78,736--101,232 | high-resolution localization |

The strong stage applies native four-scene, 16-band Mosaic with probability
0.25. It draws four distinct scenes through the same bounded repeat sampler,
rescales all boxes exactly and never mixes spectral channels. Mosaic is
disabled in stabilization, refinement and localization so final convergence
returns to the real image distribution. MixUp remains disabled: linear
mixtures of two full hyperspectral scenes do not have physically faithful
hard-box labels, and the local V2 phase evidence favored Mosaic without
MixUp.

AdamW uses ViT layer-wise LR decay, an 8x multiplier for new HSI/fusion
and object-aware parameters and a 4x multiplier for remapped class
parameters. The continuous absolute-update LR curve ends at 0.003x and
remains there for any extension. EMA is active for the entire run.
All per-pixel channel LayerNorm operations in SFP, the native spectral path
and the object-aware path compute their statistics and affine transform in
FP32 under AMP, then return the original activation dtype. This preserves the
same parameters and graph while preventing half-precision variance backward
overflow.

Validation and complete checkpoints occur every two epochs (2,812 updates).
Validation uses standard COCO `maxDets=(1,10,100)`; the Kaggle page specifies
the IoU sweep but does not publish a different detection cap.
The checkpoint contains raw parameters, EMA parameters, optimizer, FP16
scaler, epoch and absolute attempted iteration. It does not preserve the
Python/NumPy/CUDA/worker RNG stream or sampler cursor, so resume is
training-state continuous rather than bitwise-identical. There is no automatic
early stopping.

The locked epoch-30 checkpoint has `runner.iter=42,180` but AdamW
`step=42,168`; the corresponding logs contain exactly 12 AMP-skipped steps.
EMA age follows attempted iterations because the V4 hook runs every iteration.
Future reports must keep attempted iterations, successful optimizer steps and
AMP skips separate.

The run may stop only after several validation windows establish a plateau,
the LR is in its low tail, the best checkpoint is not at the boundary, and
official mAP, AP75, small-object AP and weak-class AP have stopped improving.
If epoch 72 is still improving, continue the same checkpoint:

```bash
V4_MAX_EPOCHS=88 bash tools/train_v4_codino_vitl.sh
```

## Engineering readiness

The exact formal graph passed 11 focused tests plus dual-RTX-4090 forward,
backward, DDP, dynamic loss scaling, EMA, validation, full checkpoint and
epoch-1-to-epoch-2 resume smoke tests. The strict smoke observed four finite
training updates out of four, including finite object-activation losses and
gradient norms. An additional fixed-seed stress run completed eight out of
eight finite 1280-wide updates with native four-scene Mosaic forced on every
sample. The measured peak was 21,193 MiB on a 24,564 MiB GPU, leaving about
3.3 GiB headroom. Smoke metrics are deliberately not treated as accuracy
evidence.

## Post-convergence inference

The original contract proposed weight soup and 16-view inference as a
post-convergence option. Public-leaderboard evidence supersedes that proposal.
The locked V4 reference is the epoch-30 EMA checkpoint with two 1280-wide
full-image views (identity and horizontal), followed by class-wise Gaussian
Soft-NMS (`iou_threshold=0.55`, `sigma=0.5`,
`pre_score=output_score=0.0001`) and global top-300 detections per image.
It reached Fold0 `mAP@[.50:.95]=0.71711748` and public `0.67686`.

The e54 12-view bundle improved the same Fold0 by `0.00620` but reduced public
score by `0.00087`; e30 four-slice refinement improved Fold0 by only
`0.000385` and reduced public score by `0.00004`. Therefore V5 and subsequent
model-capability versions use their own single converged EMA checkpoint with
the locked two-view protocol. Soup, SWA, full TTA, slicing and other secondary
post-processing are deferred until model-level progress has credibly
plateaued. CSV generation still includes the contiguous official `id` column
and an audit report, and no CSV is uploaded automatically.
