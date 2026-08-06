# HOD26 V5 formal training contract

V5 is one detector and one final checkpoint. It starts from the public-board
winning V4 epoch-30 EMA checkpoint. The two redundant 31-input convolutions are
algebraically collapsed to the 16 independent sensor bands. The six Co-DINO
decoder layers receive D-FINE FDR and LQE tensors from the declared official
D-FINE-X Objects365-to-COCO checkpoint; its category head is never imported.

The new localization and multilevel salience paths transition continuously
from the V4 function over updates 0--4,000. Update zero executes the literal V4
forward; zero-valued graph anchors keep the dormant V5 parameters DDP-safe.
FGL uses the D-FINE global-optimal union of encoder and decoder Hungarian
assignments, while earlier distributions distil the final layer with
IoU-weighted positives.

V5 must not repeat V3's optimizer-amnesia failure. The V4 epoch-30 checkpoint
contains both raw train weights and EMA weights after 42,180 attempted runner
iterations. Its 1,015 trained parameter tensors are at AdamW step 42,168:
the V4 logs and checkpoint jointly prove 12 AMP-skipped optimizer steps. V5
inherits every parent state by parameter name. The two 31-to-16 convolutions
preserve their raw-band gradient statistics. Only the 60 genuinely new
D-FINE FDR/LQE tensors start with empty AdamW state.

The 15 new scale-aware salience tensors clone weights, EMA values and AdamW
moments from their V4 parent head. Weight/EMA cloning is a sound semantic
initialization; moment cloning across feature scales is an explicit heuristic,
not an exact coordinate transformation. It is retained for this already
running V5 lineage and must not become the default for later versions without
new evidence.

The parent did serialize an FP16 scaler: scale 8.0 with growth tracker 1,535.
V5 deliberately starts its changed graph from scale 1.0 for numerical
recalibration rather than inheriting that scaler. This is an explicit
bootstrap policy, not missing parent data. Every ordinary V5-to-V5 resume must
restore the child scaler. EMA values and their 42,180-attempt age are inherited,
while V5 uses a new local attempted-iteration coordinate for transition, LR
and data stages.

Formal training is 2 GPUs × 1 image, 2,812 sample exposures and 1,406 attempted
updates per epoch. Successful optimizer steps are read from AdamW state at
each checkpoint; AMP skips are `runner.iter - successful_local_steps`.

The 84-epoch horizon is 118,104 attempted updates:

- 0--about 4k: parent-preserving stabilization, no Mosaic;
- about 4k--36k: capacity phase, native 16-band Mosaic at probability 0.18;
- about 36k--90k: clean high-resolution localization;
- about 90k--118,104: low-LR localization polish.

Validation and a resumable checkpoint occur every two epochs (2,812 attempted
updates). The checkpoint restores model/raw EMA, optimizer, FP16 scaler,
epoch and local schedule position, but not Python/NumPy/CUDA/worker RNG or the
sampler cursor; it is state-continuous, not bitwise-identical. Final dual-4090
worst-case smoke calibrates the 84-epoch wall time at roughly 44--52 hours,
including regular validation. There is no automatic early stopping. Epoch 84
is a soft horizon:
continue from the same checkpoint when the best point is at the boundary or
mAP, AP75, small-object AP, or weak-class AP is still improving. Stop only
after multiple validation windows plateau at the planned low LR.

## Fail-closed loading gate

`configs/v5/state_load_contract.json` pins all runtime-critical sources, the
Co-DETR commit and compatibility diff, model/EMA tensor counts, optimizer
families, parent attempted/successful steps, scaler policy and current EMA
semantics. Before bootstrap or ordinary resume,
`python -m hod26.v5.load_gate checkpoint` verifies:

- exact model and EMA name/shape/dtype with finite values;
- exact optimizer group, parameter name/order/shape and structural
  hyperparameters, so PyTorch cannot silently zip state onto reordered params;
- finite Adam moments and the inherited/fresh step equations;
- checkpoint/provenance SHA for bootstrap and a complete scaler for resume;
- source, upstream and lineage identity.

The two-segment smoke additionally proves all 1,090 optimizer tensors advance
by the successful update delta, all 1,138 EMA pairs survive both checkpoints,
and scaler state resumes. The gate was deployed without mutating the already
running process; its exact launch snapshot and existing stateful smoke were
validated read-only.

The current lineage keeps V4-compatible EMA ordering: EMA updates before
`optimizer.step` and also smooths floating buffers. Consequently validation
sees a delayed transition (approximately 0.09 at epoch 2, 0.30 at epoch 4,
0.60 at epoch 8 and 0.982 at epoch 30). Early checkpoints therefore cannot be
used to reject V5. Correcting EMA order or excluding control buffers is a
new-version change, not an allowed mid-run resume edit.

Promotion uses the already public-board-validated protocol: one best EMA
checkpoint, width 1280, identity plus horizontal flip, classwise Gaussian
Soft-NMS (`iou=0.55`, `sigma=0.5`), thresholds `0.0001`, and top 300. Soup,
SWA, slicing, WBF, and full TTA are not part of V5 promotion.
