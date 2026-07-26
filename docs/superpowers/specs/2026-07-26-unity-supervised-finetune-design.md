# Unity-Supervised Rotation-Aware Fusion Fine-Tuning Design

## Goal

Fine-tune the existing real-gymnastics A4--A9 rotation-aware fusion
checkpoints with Unity native 3D keypoint supervision, then compare them with
their zero-shot versions, single-view SAM3D, deterministic fusion, SAM3D-2D
triangulation, and oracle-2D triangulation under the existing strict Unity16
evaluation protocol.

The primary research question is whether Unity supervision transfers to an
unseen rotation direction, rather than whether a model can memorize adjacent
frames from the same synthetic sequence.

## Dataset and Leakage Boundary

The Unity benchmark contains:

- `continuous_left_060_r00`: 97 frames, angles from -60 to 0 degrees;
- `continuous_right_060_r00`: 97 frames, angles from 0 to 60 degrees;
- `static_sweep`: five independent samples at -90, -45, 0, 45, and 90
  degrees.

Training uses two direction-held-out folds:

| Fold | Training sequence | Test sequence |
|---|---|---|
| `left_to_right` | `continuous_left_060_r00` | `continuous_right_060_r00` |
| `right_to_left` | `continuous_right_060_r00` | `continuous_left_060_r00` |

The five static samples are evaluation-only OOD diagnostics. They must never
be used for training, validation, checkpoint selection, early stopping, or
hyperparameter selection.

Frames are never randomly split. Overlapping windows may occur only within a
training sequence. No window may contain frames from both the training and
test direction. Test Unity GT must not be read by the training path.

## Experiment Matrix

The matrix contains all existing rotation-aware variants:

```text
A4, A5, A6, A7, A8, A9
```

Each variant is fine-tuned for both folds and three fixed seeds:

```text
0, 1, 2
```

This produces 36 fine-tuning runs. Each run starts from its matching existing
real-gymnastics checkpoint. Training from scratch is outside the primary
matrix because the Unity training fold contains only 97 frames.

All variants and seeds use identical optimization settings. Test performance
must not influence the training duration or selected checkpoint.

## Input, Target, and Differentiable Mapping

Inputs remain synchronized cam0/cam1 SAM3D MHR70 3D keypoints and their
validity masks from the completed Unity inference cache. No Unity GT-derived
feature, bounding box, visibility flag, or confidence is supplied to the
network.

The network continues to emit MHR70 fused keypoints. A differentiable mapping
selects the exact approved Unity16 subset:

- direct MHR70 mappings for neck, arms, hands, legs, and feet;
- mean of the two hip joints for `Hips`;
- mean of the big- and small-toe tips for each `Toes` joint;
- logical AND validity for every derived joint.

Unity native 3D keypoints provide the supervised target only for the training
fold.

## Sequence-Level Differentiable Sim3

The SAM3D and Unity coordinate frames differ, while the formal benchmark
metric intentionally ignores a single global similarity transform per
sequence. The supervised loss therefore estimates one differentiable Sim3
per training window using every valid frame and Unity16 joint in that window.
It must not estimate a separate transform per frame.

The implementation uses a batched, masked Umeyama fit:

1. compute masked prediction and target centroids;
2. estimate the cross-covariance over all valid frame-joint pairs;
3. use `torch.linalg.svd` with determinant correction for a proper rotation;
4. estimate one positive scale and translation per window;
5. apply that transform to every prediction in the window.

A fold must fail explicitly if a window has fewer than three valid
non-degenerate correspondences or produces a non-finite transform. Tests
must confirm that gradients reach the model output and that a single
transform cannot hide a frame-specific deformation.

## Training Loss

The main supervised objective is masked Smooth-L1 distance between aligned
predicted Unity16 points and Unity GT:

```text
L_total = L_unity_3d + 0.1 * L_existing_self_supervised
```

`L_unity_3d` is averaged only over valid homologous joints and padded frames
are excluded. The existing self-supervised physical and temporal objectives
remain unchanged and act as regularization against catastrophic overfitting.

The main experiment does not add an angle-specific supervised loss because
the primary target is real 3D keypoint accuracy. Trunk angle MAE remains an
independent evaluation metric.

## Optimization Protocol

Every run uses:

| Setting | Value |
|---|---:|
| Window length | 32 frames |
| Training stride | 8 frames |
| Epochs | 100 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Seeds | `0, 1, 2` |
| Checkpoint used for evaluation | final epoch |

The full matching A4--A9 model is fine-tuned. No layer is frozen. Existing
deterministic corruptions and masks remain available as self-supervised
regularization, with stable seeds derived from the fold, run seed, epoch, and
window identity.

The final epoch is used unconditionally. The implementation must not load a
"best" checkpoint chosen using the held-out direction or static sweep.

## Architecture and CLI

Add isolated supervised training components:

```text
configs/benchmarks/unity_supervised.yaml
src/gymnastics/benchmarks/unity/supervised.py
src/gymnastics/benchmarks/unity/supervised_loss.py
```

Extend `gymnastics benchmark unity` with:

```text
finetune
finetune-matrix
evaluate-finetuned
report-finetuned
```

`finetune` runs or resumes one ablation/fold/seed combination.
`finetune-matrix` executes the complete 36-run matrix and skips only runs
whose final checkpoint and provenance pass validation.
`evaluate-finetuned` performs inference on the held-out direction and static
sweep, then evaluates through the existing Unity16 one-Sim3-per-sequence
implementation. `report-finetuned` regenerates machine-readable and
human-readable comparisons from saved artifacts.

## Output Isolation and Provenance

All new artifacts go below:

```text
local/runs/unity_benchmark/supervised_finetune/
  fold_left_to_right/<ablation>/seed_<seed>/
  fold_right_to_left/<ablation>/seed_<seed>/
  evaluation/
  report/
```

Each run records:

- fold and exact train/test sequence IDs;
- seed and resolved optimization configuration;
- initial real-data checkpoint path and SHA-256;
- final checkpoint SHA-256;
- git commit;
- Unity manifest hash;
- SAM3D cache identity;
- epoch-level supervised, self-supervised, and total loss;
- explicit statement that Unity GT was used for training;
- confirmation that static and held-out sequences were excluded from
  training.

New outputs must not modify existing real-data runs or zero-shot Unity
artifacts.

## Evaluation and Ranking

The official comparison uses the existing strict Unity16 evaluator and fits
one Sim3 per complete evaluation sequence. It reports:

- MPJPE, median, and p95 in millimetres;
- trunk angle MAE and RMSE;
- per-sequence, per-joint, and Unity-visibility partitions;
- results for each fold and seed;
- mean, standard deviation, minimum, and maximum across seeds;
- a macro average over the two held-out directions;
- static-sweep OOD diagnostics.

The report includes:

- zero-shot A4--A9;
- Unity-supervised A4--A9;
- cam0 and cam1;
- all nine deterministic fusion methods;
- SAM3D-2D triangulation;
- oracle-2D triangulation as a diagnostic;
- GT-fitted joint-weight fusion as a diagnostic.

Unity-supervised methods form a clearly labelled ranking group. They must not
be described as external zero-shot generalization.

Success is assessed at three thresholds:

1. improvement over the matching zero-shot variant;
2. improvement over `avg_world_face_ref` at 166.537 mm;
3. approaching or exceeding `triangulation_sam3d2d` at 30.259 mm.

No statistical significance claim is made because the benchmark contains one
avatar and two continuous sequences.

## Failure Handling and Resumption

Training fails without creating a final checkpoint when:

- a fold contains overlapping train/test sample IDs;
- Unity GT from a held-out or static sample reaches a training batch;
- initial checkpoint ablation does not match the requested ablation;
- a differentiable Sim3 is degenerate or non-finite;
- a loss or gradient is non-finite;
- provenance is incomplete.

Atomic checkpoint and metrics writes prevent interrupted runs from appearing
complete. Resumption accepts a run only when its final checkpoint, resolved
configuration, fold identity, source hashes, and provenance all match.

## Verification

Tests must cover:

- exact direction fold membership and absence of sample overlap;
- exclusion of all five static samples from training;
- differentiable Unity16 mapping and derived-joint gradients;
- masked sequence/window-level Sim3 recovery and gradient propagation;
- inability of one Sim3 to hide frame-specific deformation;
- supervised loss masks and padded-frame exclusion;
- rejection of held-out GT in training batches;
- checkpoint ablation and provenance validation;
- deterministic seeds and resumable matrix behavior;
- CLI routing and artifact layout;
- ranking separation between zero-shot, Unity-supervised, and diagnostic
  methods.

Before reporting results, run the focused tests, the complete repository test
suite, Ruff on changed files, and a report consistency audit against all 36
expected runs.

## Interpretation Boundary

This experiment measures synthetic in-domain adaptation and transfer to the
opposite rotation direction for one avatar and one environment. It does not
establish human-population generalization. A strong supervised result would
show that the fusion architecture can learn accurate 3D shape from Unity
supervision; it would not prove that the same accuracy transfers back to real
gymnastics videos.
