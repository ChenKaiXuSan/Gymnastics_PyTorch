# Real-Data Fitted-Camera Feature Pilot Design

## Objective

Test whether fitted camera information improves the existing A6
rotation-aware self-supervised fusion network on the collected two-view
gymnastics dataset. This is a person-disjoint pilot on outer fold 0 and uses
the same G0--G5 feature ablations as the Unity study.

## Data audit and selected approach

Three camera-fitting strategies were considered:

1. One global rig fitted from all training people.
2. One consensus rig per recovered rig cluster.
3. One input-level camera fitted independently for each person.

The global-rig audit failed: fitting from 96 training people produced a 4.86%
inlier ratio, 23.36 px held-out reprojection error, and a near-identity
rotation inconsistent with the physical views. The collected videos contain
two mirror-image rig groups and person/session variation. A single fixed rig
is therefore invalid.

The experiment uses per-person input-level self-calibration. The existing
camera audit covers all 137 people and was produced from synchronized SAM3D
observations. Median held-out reprojection error is 6.27 px; 123/137 people
are at or below 10 px. Rotation and unit translation direction are used as
features; metric translation scale is excluded.

## Split and leakage boundary

Use the existing cohort-stratified outer `fold_00`:

- train: 96 people;
- validation: 27 people;
- test: 14 people;
- seeds: 0, 1, and 2.

Each test person's camera is estimated at inference from that person's own
synchronized SAM3D observations. This is explicitly a transductive,
input-level self-calibration setting. Test camera fitting receives neither
triangulated 3D nor labels.

Triangulated 3D is unavailable to model construction, camera feature
construction, training, validation, checkpoint selection, and inference. It
is loaded only after all test predictions are frozen.

## Model and ablations

Reuse the corresponding fold-0 A6 checkpoint for each seed:

- seed 0: `all137_a6_e100_seed0`;
- seed 1: `all137_a6_s1_e100`;
- seed 2: `all137_a6_s2_e100`.

The A6 backbone remains frozen. Only the zero-initialized
`camera_conditioner` and `camera_delta_head` parameters are optimized. This
isolates the incremental effect of camera input and avoids the added-capacity
confound observed in the Unity experiment.

- G0: unmodified frozen A6 checkpoint; no additional training.
- G1: per-person pose and intrinsics.
- G2: G1 plus fit-quality features.
- G3: G2 plus per-frame/per-joint epipolar and ray geometry.
- G4: all camera features with FiLM conditioning.
- G5: G4 with a deterministic 30-degree wrong-camera perturbation.

Training uses the existing A6 self-supervised losses for 10 epochs with
batch size 32. G1--G5 use identical optimization settings. The primary
comparison is G4 versus matched G0; G5 is the required negative control.

## Data flow

1. Load prepared canonical A6 trials and split-cycle frame maps.
2. Load per-person fitted camera audit and calibrated intrinsics.
3. Load only the 2D keypoints addressed by each trial's face/side frame maps.
4. Undistort 2D points and construct the existing 19 global and 8 joint
   camera features.
5. Train only camera-specific parameters on the 96 training people.
6. Select the final epoch without pseudo-GT.
7. Infer the 14 held-out people.
8. Load triangulated 3D and evaluate with the existing per-cycle static Sim3
   protocol, then aggregate cycles within person before group statistics.

## Outputs and success criteria

Write isolated artifacts under
`local/runs/fitted_camera_real/fold_00/{G0,...,G5}/seed_{0,1,2}`.

Required outputs:

- checkpoint and strict provenance for every trained cell;
- one inference NPZ per test cycle;
- camera-fit audit by person;
- metrics by cycle, person, method, and seed;
- paired G1--G5 versus G0 comparisons;
- direct G4 versus G5 negative-control comparison;
- Markdown experiment report.

A correct-camera claim is allowed only if G4 improves person-level MPJPE over
G0 and also beats G5. Otherwise the result is reported as no evidence that the
network used correct camera geometry.

## Limitations

This first run covers one 14-person outer test fold. It is a prespecified
pilot gate, not a complete 137-person OOF result. Expansion to all ten outer
folds is justified only if the pilot passes the G4-versus-G5 control.
