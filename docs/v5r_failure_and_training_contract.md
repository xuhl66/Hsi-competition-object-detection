# HOD26 V5 failure audit and clean V5R training contract

## Decision

The original V5 run is terminated and frozen as failure evidence. Its best
epoch-18 checkpoint (`SHA-256 14b8b7a9987cfc18ac32d6139b41d7f31fe664686b38f7e951c6a992c3751ff5`)
may be used for inference-only leaderboard diagnosis, but no original V5
model, EMA or optimizer state may initialize V5R.

V5R is a new clean lineage. Its sole training parent is V4 epoch 30
(`SHA-256 2b96f909af97c93c01c2141d032eb391428b18aac76be5e38f4f3ae5f9198932`);
the only second source is the already declared official D-FINE-X
Objects365-to-COCO checkpoint, restricted to FDR and LQE tensors.

## What failed in V5

The failure was structural and reproducible in the log, not an SSH/network
event. Validation rose from 0.702 at epoch 10 to 0.719 at epoch 18, then fell
to 0.689/0.682 at epochs 20/22. The training signature began during epoch 15:
main and DN localization losses diverged while encoder and collaborative-head
localization remained stable. Around the e14-to-e16 window, representative
main bbox/IoU loss moved from about 0.09/0.39 to 0.17/0.75 and DN bbox/IoU
from about 0.09/0.31 to 0.94/2.71, while encoder bbox/IoU stayed near
0.09/0.39. FGL simultaneously shrank because its official IoU weighting
self-silences after predicted boxes deteriorate.

Four implementation defects jointly explain the signature:

1. V5 used the first unsupervised native coarse prediction as the fixed FDR
   base reference. Official D-FINE instead owns an independent 4-D pre-bbox
   head and explicitly supervises both matching and DN pre-outputs.
2. V5 left its FDR runtime reference installed on the shared decoder.
   Co-DETR's RPN/ROI and ATSS auxiliary passes reused that decoder and then
   applied native regression branches again. One FDR parameter family thus
   received incompatible primary and auxiliary coordinate semantics; stable
   auxiliary losses concealed primary FDR drift.
3. FDR and LQE were grouped with new HSI modules at 8x detector LR despite
   already carrying mature external weights. Their Adam moment magnitude
   accelerated around the failure window.
4. Update zero bypassed to literal V4, but update one entered a custom decoder
   whose cached official prediction came from the unnormalized internal
   reference-update branch. Native Co-DINO advances references with that
   branch but supervises a separate prediction recomputed from the normalized
   intermediate state. The old implementation therefore was not functionally
   continuous across the first two updates.

The old EMA also ran before `optimizer.step` and smoothed `v5_transition`.
Validation at epoch 22 therefore saw transition 0.944 while the raw training
graph was already at 1.0. That lag was not the primary collapse cause, but it
made diagnostics less faithful.

## V5R structural repair

V5R keeps the high-potential 16-band exact reparameterization, multiscale
salience, FDR, LQE, FGL and GO-LSD ideas, with these non-negotiable repairs:

- a separate pre-bbox head is cloned from the domain-trained V4 decoder
  layer-0 regression head and receives matching plus DN cls/L1/GIoU losses;
- the fixed FDR reference is the detached supervised pre-box, matching the
  official D-FINE topology;
- primary native predictions and next-layer native references remain separate
  Co-DINO coordinates and are independently blended into FDR, making the
  transition continuous;
- every Co-DETR auxiliary decoder call temporarily removes FDR/pre-head
  runtime references and executes the native Co-DINO decoder;
- FDR, LQE and pre-bbox parameters use detector LR (1x), not HSI LR (8x);
- D-FINE's query-position clamp to [-10, 10] is applied;
- transition is extended to 12,000 attempted updates, and strong Mosaic starts
  only at epoch 10, after structural migration is complete;
- logs expose main/DN/pre matched IoU, FDR-vs-native displacement, corner
  magnitude and box saturation on every active training row.

FGL retains the published IoU weighting rather than adding an unvalidated
floor. The supervised pre-anchor addresses the upstream failure that caused
FGL to self-silence.

## Exact state inheritance declaration

The V4 parent is at runner iter 42,180 and AdamW step 42,168; 12 parent AMP
overflows are explicitly preserved in the record.

- All 1,015 exact-coordinate V4 optimizer tensors inherit raw weights, EMA
  weights, AdamW `step`, `exp_avg` and `exp_avg_sq`.
- The two 31-to-16 convolution weights are algebraically collapsed. Their
  Adam moments use the first 16 raw-input coordinates, whose gradient
  coordinate is unchanged.
- FDR (36 tensors) and LQE (24 tensors) use official pretrained weights and
  EMA values but fresh optimizer state.
- The independent pre-bbox head (6 tensors) clones V4 layer-0 raw/EMA weights
  but starts fresh optimizer state because it is now a separate supervised
  coordinate.
- The three multiscale salience heads (15 tensors) clone V4 object-head
  raw/EMA weights but start fresh optimizer state. V5's heuristic cross-scale
  moment cloning is deliberately not repeated.

The result is 1,015 inherited and 81 explicitly fresh optimizer tensors out
of 1,096 trainable tensors. Inherited tensors cover 99.5742% of trainable
parameters by element count.

The V4 parent did serialize an FP16 scaler at scale 8.0. V5R deliberately
recalibrates the changed graph from the configured fresh scaler; every
ordinary V5R resume must restore the child scaler.

## EMA and resume semantics

V5R learned-weight EMA updates only after a successful optimizer step. AMP
overflow does not age or alter learned EMA weights. Transition buffers are
copied into their EMA counterparts and never smoothed, so validation sees the
same structural coordinate used by the last training update.

The checkpoint restores model/raw EMA, optimizer, FP16 scaler, epoch and
attempted-iteration state. Epoch checkpoints still do not serialize every
Python/NumPy/CUDA/worker RNG stream or sampler cursor, so this is
state-continuous rather than bitwise resume.

Before formal launch, the fail-closed gate must prove exact source/upstream
identity, complete model/EMA schema, optimizer group/name/order/shape and
hyperparameters, finite weights/moments, inherited/fresh step equations,
scaler policy, parent/provenance SHA and explicit exclusion of faulty V5 e18.
A two-segment dual-GPU smoke must then prove that all optimizer states advance
through a real child resume and that pre-head, DN, FDR, native auxiliary,
EMA, checkpoint, validation and inference paths remain finite.

## Formal long schedule

Global batch remains two (one image per RTX 4090), producing 1,406 attempted
updates and 2,812 sample exposures per epoch. The 96-epoch soft horizon is
134,976 attempted updates:

- 0-12k: synchronized structural transition and LR stabilization, no Mosaic;
- epoch 10-36: capacity phase with native 16-band Mosaic at probability 0.18;
- epoch 36-72: clean high-resolution refinement;
- epoch 72-96: low-LR localization polish.

Validation and complete checkpoints occur every two epochs (2,812 attempted
updates). There is no automatic early stop. Epoch 96 is not permission to
stop if the best point is at the boundary or mAP, AP75, small-object AP or
weak-class AP remains on a credible rising trend; extend the same checkpoint
at low LR.

Promotion continues to use the frozen public-board protocol: one best EMA
checkpoint, width 1280, identity plus horizontal views, classwise Gaussian
Soft-NMS at IoU 0.55 with sigma 0.5, thresholds 0.0001 and top 300. Soup, SWA,
slicing, WBF and larger TTA do not enter model-version comparison.
