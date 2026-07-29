# Unity Extrinsic Learning Design

## Objective

Add three calibrated learned baselines to the existing Unity benchmark and
report them without mixing incompatible input regimes:

1. `extrinsic_gate`: fuse two monocular SAM3D 3D pose streams after applying
   the exact Unity relative camera rotation.
2. `extrinsic_residual_tcn`: refine the calibrated equal-average base with a
   bounded temporal residual.
3. `learnable_triangulation`: predict per-joint, per-view confidence from
   SAM3D 2D tracks and use it in differentiable algebraic triangulation with
   the exact Unity projection matrices.

The first two are calibrated 3D-to-3D fusion methods. The third is a calibrated
2D-to-3D method and must be reported in a separate input-regime block.

## Experimental Contract

- Unity native 3D keypoints are the only supervised target and evaluation GT.
- Use the existing strict direction-transfer folds:
  `continuous_left_060_r00 -> continuous_right_060_r00` and the reverse.
- Run seeds 0, 1, and 2 for every method and fold.
- The held-out direction and `static_sweep` never enter training, checkpoint
  selection, normalization fitting, or hyperparameter tuning.
- Train for the configured fixed epoch count and evaluate the final checkpoint;
  no test-set early stopping is permitted.
- Use one sequence-level Sim3 at evaluation, matching every existing Unity
  benchmark result.
- Report `static_sweep` only as an OOD diagnostic.
- Preserve immutable sample IDs, camera provenance, manifest hash, source-cache
  hash, resolved configuration, and checkpoint hash.

## Camera Geometry

For column vectors, the exact relative camera rotation from cam1 coordinates
to cam0 coordinates is:

```text
R_cam1_to_cam0 =
    world_to_camera(cam0)[0:3,0:3]
    @ camera_to_world(cam1)[0:3,0:3]
```

The SAM3D pose streams are pelvis-centred before rotation because their
translation is not metrically calibrated. Translation from the Unity cameras
is therefore not applied to monocular 3D poses. The projection matrices used
by learnable triangulation retain the full exact camera geometry.

## Model 1: Extrinsic Gate

Inputs are cam0 MHR70 3D points, cam1 MHR70 3D points rotated into cam0, joint
validity, absolute inter-view disagreement, per-view velocity magnitude, and
the flattened exact relative rotation. A compact shared MLP followed by a
three-block temporal convolution predicts one sigmoid gate per frame and
joint. The output is:

```text
gate * cam0 + (1 - gate) * rotated_cam1
```

Single-view joints bypass the gate and copy the valid view. The gate model has
no unconstrained residual, so its prediction remains inside the two-view
segment.

## Model 2: Extrinsic Residual TCN

The base is the validity-aware equal average of cam0 and rotated cam1. The
network consumes the same calibrated features as the gate model. A compact
dilated TCN predicts a per-joint 3D residual bounded by 50 mm using `tanh`.
The residual is zero for padded frames and joints with neither view valid.

## Model 3: Learnable Algebraic Triangulation

The model consumes normalized SAM3D 2D coordinates, joint validity, temporal
velocity, and the two exact 3x4 pixel projection matrices. A shared
per-view/per-joint confidence network predicts positive confidence in
`[0.05, 1.0]`. Each pair of DLT equations is multiplied by the corresponding
confidence, and the 3D point is the dehomogenized smallest right singular
vector. Missing views produce an invalid joint rather than a fabricated point.

Training maps MHR70 outputs to the Unity16 semantic evaluation subset before
the supervised loss. For the triangulation method, the loss is metric masked
Smooth-L1 in Unity world coordinates. For the two monocular-3D fusion methods,
one differentiable Sim3 is fit per training window before masked Smooth-L1,
matching the ambiguity of SAM3D monocular scale and coordinate convention.

## Software Boundaries

- `extrinsic_models.py`: pure differentiable geometry, feature construction,
  the three model classes, and a common prediction contract.
- `extrinsic_training.py`: fold-safe datasets, fixed-epoch training,
  checkpoint/provenance validation, inference, and matrix orchestration.
- `extrinsic_evaluation.py`: shared Unity evaluator adapters and strict
  2-fold x 3-seed aggregation.
- `cli.py`: explicit `extrinsic-train`, `extrinsic-infer`, and
  `extrinsic-evaluate` stages plus an all-stage command.
- Tests use compact synthetic camera/pose fixtures and assert observable
  geometry, leakage boundaries, checkpoint identity, and aggregation.

## Paper Organization

The Results section will be reorganized into:

1. primary real-gymnastics results without extrinsics;
2. real-gymnastics calibrated deterministic baselines;
3. Unity native-GT validation, grouped by input regime;
4. supervised direction-transfer and OOD diagnostics;
5. robustness, ablation, and cohort analyses.

A single overview table will state for every result family: dataset, GT,
training supervision, camera information, evaluation unit, and headline
metric. Captions and discussion will distinguish measured evidence from
diagnostics and avoid a single unfair cross-regime ranking.

## Acceptance Criteria

- All three methods run for both folds and three seeds.
- Every completed cell has a validated checkpoint and provenance record.
- Held-out and static sample IDs are absent from each training artifact.
- Evaluation produces finite direction-held-out metrics and static diagnostics.
- Existing Unity and rotation-aware tests remain green.
- The paper asset generator reads result artifacts, emits grouped tables and an
  overview figure/table, manuscript checks pass, and LaTeX compiles to PDF.

