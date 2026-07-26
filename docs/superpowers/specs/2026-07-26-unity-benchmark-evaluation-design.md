# Unity Benchmark External Evaluation Design

## 1. Purpose

Implement the current two-view gymnastics pose-processing and fusion evaluation
on `/home/data/xchen/gymnastics/unity_benchmark`, using Unity's native 3D
keypoints as the external ground truth.

The benchmark has two synchronized rendered views (`cam0` and `cam1`), exact
camera parameters, exact 2D projections, and exact 3D skeleton coordinates. The
evaluation will measure how well the existing SAM3D-Body and fusion methods
transfer to this independently generated dataset.

The experiment must answer:

- How accurate is SAM3D-Body from each Unity camera independently?
- Do the existing deterministic two-view fusion methods improve on either
  single view?
- Do the existing rotation-aware A4--A9 checkpoints transfer zero-shot?
- How does fusion compare with calibrated two-view triangulation?
- How much of the error is associated with Unity's self-occluded joints?

This benchmark is an external evaluation, not a new training set. Unity 3D
ground truth must not be used for training, checkpoint selection, or ordinary
fusion decisions.

## 2. Confirmed Scope

The experiment includes:

- `cam0` single-view SAM3D-Body;
- `cam1` single-view SAM3D-Body;
- all nine existing deterministic fusion methods;
- existing rotation-aware A4--A9 checkpoints, evaluated zero-shot;
- triangulation of SAM3D-predicted 2D keypoints using Unity calibration;
- oracle triangulation of Unity's exact 2D keypoints as a geometry diagnostic;
- Unity native 3D keypoints as the only evaluation ground truth;
- a strict homologous subset of approximately 16 joints;
- one sequence-level Sim3 alignment per evaluated sequence, matching the
  current evaluation protocol.

The experiment does not include:

- triangulated points as ground truth;
- training or fine-tuning on Unity data;
- checkpoint selection using Unity metrics;
- approximate comparison of all 22 Unity joints against semantically different
  MHR70 joints;
- per-frame Procrustes alignment as the main metric.

## 3. Dataset Contract

### 3.1 Source data

The adapter reads the Unity benchmark in place and never modifies it:

```text
/home/data/xchen/gymnastics/unity_benchmark/
├── manifest.jsonl
├── skeleton.json
├── cameras.json
├── generation_config.json
├── metadata.json
├── run_summary.json
└── images/
    ├── cam0/
    └── cam1/
```

The inspected dataset contains 199 samples and 398 PNG images:

- five static samples at -90, -45, 0, 45, and 90 degrees;
- 97 frames in `continuous_left_060_r00`;
- 97 frames in `continuous_right_060_r00`;
- 22 Unity 3D joints and 22 projected 2D joints per camera and sample.

`sample_id`, not `frame_index`, is the persistent frame key. `frame_index`
restarts within sequences and therefore must never be used as a global image
identifier.

### 3.2 Evaluation sequences

The adapter exposes three evaluation sequences:

1. `static_sweep`, formed from the five ordered static angles;
2. `continuous_left_060_r00`;
3. `continuous_right_060_r00`.

The static samples are grouped so that the shared evaluation protocol can
estimate one sequence-level Sim3 transformation. Every output retains the
original `sample_id`, source `sequence_id`, `frame_index`, phase, visibility,
and `actual_angle_deg`.

### 3.3 Coordinate conventions

The implementation treats Unity 3D coordinates as meters in its documented
world frame: X right, Y up, and Z forward. Image coordinates use a top-left
origin. Camera matrices are row-major 4x4 matrices and camera depth is positive.

These conventions must be verified by reprojection and oracle triangulation
tests before full inference results are interpreted.

## 4. Joint Correspondence

Only anatomically homologous joints are included in the main comparison.
Derived MHR70 points use fixed, documented formulas and require every source
joint to be valid.

| Unity joint | MHR70 prediction used for evaluation |
|---|---|
| `Hips` | midpoint of `left_hip` and `right_hip` |
| `Neck` | `neck` |
| `LeftUpperArm` | `left_shoulder` |
| `LeftLowerArm` | `left_elbow` |
| `LeftHand` | `left_wrist` |
| `RightUpperArm` | `right_shoulder` |
| `RightLowerArm` | `right_elbow` |
| `RightHand` | `right_wrist` |
| `LeftUpperLeg` | `left_hip` |
| `LeftLowerLeg` | `left_knee` |
| `LeftFoot` | `left_ankle` |
| `LeftToes` | mean of `left_big_toe_tip` and `left_small_toe_tip` |
| `RightUpperLeg` | `right_hip` |
| `RightLowerLeg` | `right_knee` |
| `RightFoot` | `right_ankle` |
| `RightToes` | mean of `right_big_toe_tip` and `right_small_toe_tip` |

Unity `Spine`, `Chest`, `UpperChest`, `Head`, `LeftShoulder`, and
`RightShoulder` are excluded because MHR70 has no exact point with the same
skeleton semantics. The mapping is defined once and reused by single-view,
fusion, and SAM3D-2D triangulation evaluation.

Oracle triangulation starts from Unity joints directly, but its ranked summary
is still restricted to the same 16-joint subset for comparability.

## 5. Architecture

Add a modular benchmark package under `src/gymnastics/benchmarks/unity/`.
Responsibilities are separated so that expensive inference can be cached and
later evaluation can be repeated without a GPU:

```text
dataset adapter
  -> SAM3D inference for cam0 and cam1
  -> synchronized pose-pair construction
  -> deterministic fusion
  -> zero-shot rotation-aware inference
  -> SAM3D-2D and oracle-2D triangulation
  -> common Unity-GT evaluation
  -> tables, figures, and Markdown report
```

The intended module boundaries are:

- dataset and manifest parsing;
- camera parsing, projection, and triangulation;
- MHR70-to-Unity joint mapping;
- SAM3D image inference and cache validation;
- adapters into existing deterministic and rotation-aware fusion code;
- common alignment and metric computation;
- report generation and command-line orchestration.

The precise filenames may follow existing repository conventions during
implementation, but these responsibilities must remain independently testable.

### 5.1 Staged command interface

The command-line interface supports:

```text
infer -> triangulate -> fuse -> evaluate -> report
```

It also provides a full-run entry point that executes missing stages in order.
Each stage validates upstream metadata before reusing cached artifacts. A
completed stage is not recomputed unless explicitly requested.

### 5.2 SAM3D inference

SAM3D-Body processes `cam0` and `cam1` independently. The estimator should be
loaded once per inference process rather than once per sequence. Each cached
sample records at least:

- `sample_id`;
- camera name;
- source image path;
- predicted MHR70 3D keypoints;
- predicted MHR70 2D keypoints;
- fields needed by the current fusion code;
- inference/checkpoint metadata.

Missing detections or invalid arrays are explicit failed samples. They are never
silently removed from denominators or synchronized pairs.

### 5.3 Synchronization

Unity views are natively synchronized. Pose pairs are joined by `sample_id`
within an evaluation sequence, with temporal offset fixed to zero. No DTW,
audio alignment, split-cycle record, or estimated fallback offset is used.

### 5.4 Deterministic fusion

The benchmark adapter calls the existing pure deterministic fusion functions
instead of reimplementing their geometry. `cam0` is the reference view where a
method requires one. All nine existing method names remain unchanged.

`sim3_face_stable_joint_weight` derives weights from evaluation ground truth in
the current experiment definition. It will be run only to preserve the complete
matrix and will be labelled `GT_LEAKY_DIAGNOSTIC`. It is excluded from valid
method ranking and recommendations.

### 5.5 Rotation-aware fusion

A4--A9 use the existing checkpoints trained on the real gymnastics dataset.
The Unity adapter constructs the same MHR70 pose-pair contract expected by the
current inference layer. There is no Unity training, fine-tuning, validation
selection, or checkpoint selection.

Every ablation is reported separately. The experiment may not select and
present an ablation as though it had been selected without observing Unity
ground truth; comparison across all A4--A9 is explicitly post-hoc external
evaluation.

## 6. Triangulation Experiments

### 6.1 SAM3D-2D triangulation

`triangulation_sam3d2d` triangulates corresponding MHR70 2D predictions from
`cam0` and `cam1` using the known Unity projection matrices. The cameras are
already synchronized, so no temporal alignment or triangulated pseudo-GT
directory from the human dataset is involved.

The resulting MHR70 world points are mapped to the same 16 Unity joints and
evaluated using the same sequence-level Sim3 and metric implementation as every
other valid method.

### 6.2 Oracle-2D triangulation

`triangulation_oracle2d` triangulates Unity's exact projected 2D joints with the
same Unity camera model. It is an implementation diagnostic, not a deployable
method and not part of the valid ranking.

It reports:

- raw world-coordinate 3D error before Sim3;
- reprojection error in both cameras;
- the common Sim3-aligned metrics on the 16-joint subset.

Oracle failure beyond a small documented numerical tolerance blocks
interpretation of the SAM3D-2D triangulation result until the camera convention
or implementation is corrected.

## 7. Evaluation Protocol

### 7.1 Alignment

For each method and each of the three evaluation sequences:

1. collect all mutually valid frames and mapped joints;
2. estimate one similarity transformation over the pooled sequence points;
3. apply that single transformation to every prediction in the sequence;
4. compute Euclidean errors without any further per-frame alignment.

The fit includes only the agreed evaluation joint subset. Per-frame Procrustes
may be implemented as an explicitly labelled diagnostic but cannot replace the
main protocol.

### 7.2 Primary pose metrics

All distances are reported in millimetres:

- MPJPE;
- median joint error;
- P95 joint error;
- per-sequence MPJPE;
- per-joint MPJPE;
- pooled per-frame error arrays.

The main overall MPJPE pools all valid mapped joint samples after the
per-sequence alignment. Per-sequence values are always shown so the five-frame
static group cannot hide or dominate behaviour in the continuous sequences.

### 7.3 Visibility analysis

Unity visibility flags define two additional result partitions:

- target joint visible in the evaluated camera/view;
- target joint self-occluded.

For fused and triangulated methods, the report includes camera-specific
visibility patterns where informative, including visible in both, one, or
neither camera. These partitions are secondary diagnostics; the main metric
uses all valid Unity GT joints.

### 7.4 Rotation-angle analysis

The task-specific secondary evaluation reports trunk-rotation MAE and RMSE in
degrees. Predicted rotation is derived from the fused or single-view 3D
keypoints using one fixed pelvis/thorax coordinate convention shared across all
methods.

The same geometry is applied to Unity 3D ground truth as a construct-validity
reference, and the result is compared with `actual_angle_deg`. Sign and neutral
offset conventions are fixed once at the evaluator level and cannot be fitted
separately for each method.

### 7.5 Fair ranking

The primary ranking includes:

- `cam0`;
- `cam1`;
- the eight leakage-free deterministic methods;
- A4--A9 zero-shot outputs;
- `triangulation_sam3d2d`.

The diagnostic table includes:

- `sim3_face_stable_joint_weight`;
- `triangulation_oracle2d`;
- geometry and reprojection audits.

Unity GT is accessible only to the evaluator, the oracle diagnostic, and the
explicitly leaky deterministic diagnostic. Runtime checks and code boundaries
should make accidental GT access by valid inference/fusion paths difficult.

## 8. Output Contract

All new artifacts are isolated from the real-person experiments:

```text
local/runs/unity_benchmark/
├── sam3d/
│   ├── cam0/
│   └── cam1/
├── triangulation/
│   ├── sam3d2d/
│   └── oracle2d/
├── fusion/
│   ├── deterministic/
│   └── rotation_aware/
├── evaluation/
│   ├── metrics_summary.csv
│   ├── metrics_by_sequence.csv
│   ├── metrics_by_joint.csv
│   ├── metrics_by_visibility.csv
│   └── per_frame_errors.npz
└── report/
    ├── unity_benchmark_report.md
    ├── results.json
    └── figures/
```

Machine-readable metadata records:

- resolved configuration;
- source dataset path and inspected counts;
- joint mapping version;
- checkpoint paths;
- code commit;
- exact commands;
- timestamps and runtime;
- expected and completed sample counts;
- exclusions and explicit failure reasons.

The Markdown report separates valid methods from diagnostics and includes
dataset integrity, pose metrics, visibility results, angle results, failure
accounting, and interpretation.

## 9. Validation Strategy

Before the full 398-image GPU run:

1. unit-test manifest parsing, sequence grouping, and `sample_id` joins;
2. unit-test all 16 joint mapping formulas and validity propagation;
3. test units and sequence-level Sim3 on synthetic data;
4. test projection/triangulation round trips with Unity camera parameters;
5. require oracle triangulation to satisfy documented raw 3D and reprojection
   tolerances;
6. run a small end-to-end subset through SAM3D, synchronization, at least one
   fusion method, and evaluation;
7. validate output schemas and resume behaviour.

The full run then verifies for every method:

- expected samples and all three sequences are present;
- joint dimensions and names match;
- arrays are finite or carry an explicit invalid mask;
- no frame is silently dropped;
- checkpoint and configuration identity are recorded;
- valid methods never read Unity 3D GT before evaluation.

Targeted repository tests are used because the repository-wide default pytest
configuration also collects optional and third-party modules with unavailable
dependencies.

## 10. Reporting and Interpretation Boundaries

The report may identify which valid method has the lowest benchmark error and
describe paired per-frame or per-joint differences. It must not make
population-level statistical significance or generalization claims because the
current Unity dataset contains one avatar, one environment, five static poses,
and two continuous sequences.

Conclusions must explicitly distinguish:

- accuracy against real Unity 3D ground truth;
- diagnostic oracle geometry performance;
- GT-leaky method performance;
- zero-shot transfer of checkpoints trained on the real gymnastics dataset;
- limitations caused by synthetic appearance, self-occlusion, and limited
  sequence diversity.

## 11. Completion Criteria

The work is complete when:

- the modular Unity adapter and staged CLI are implemented;
- targeted unit and integration tests pass;
- SAM3D predictions exist for all successfully detected images in both views;
- the two triangulation experiments, nine deterministic methods, and A4--A9
  have been evaluated or carry explicit failure records;
- the common metrics and visibility/angle analyses are generated;
- the oracle camera audit passes;
- the final report contains reproducible commands, complete tables, findings,
  and limitations;
- no Unity raw file or existing real-person run artifact has been modified.
