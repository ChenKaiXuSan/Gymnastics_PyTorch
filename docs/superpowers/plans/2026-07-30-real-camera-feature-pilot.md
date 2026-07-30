# Real-Data Fitted-Camera Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a leakage-safe, fold-0, three-seed G0--G5 fitted-camera experiment on the collected 137-person gymnastics dataset.

**Architecture:** A real-data adapter joins existing canonical A6 cycle caches with synchronized SAM3D 2D points and the per-person camera audit. Dataset wrappers attach the existing 19-dimensional global and 8-dimensional joint camera features. A standalone pilot runner freezes the A6 backbone, trains only camera-specific parameters, performs test-only inference, and loads triangulated pseudo-GT only in the evaluation command.

**Tech Stack:** Python 3.10, PyTorch, NumPy, OpenCV, existing rotation-aware fusion modules, pytest, YAML.

## Global Constraints

- Use `conda run -n gymnastic ...` for every project command.
- Use `local/runs/cohort_cycle/folds/fold_00.json` with 96 train, 27 validation, and 14 test people.
- Use seeds 0, 1, and 2 with their matching existing A6 checkpoints.
- G0 is the unchanged A6 checkpoint; G1--G5 train only `camera_conditioner` and `camera_delta_head`.
- Triangulated 3D is inaccessible until evaluation.
- Per-person camera estimation is a declared transductive input operation.
- G4 must beat both G0 and wrong-camera G5 before claiming correct camera information.
- Store generated artifacts only under `local/runs/fitted_camera_real/fold_00`.

---

### Task 1: Real camera feature and cycle adapter

**Files:**
- Create: `src/gymnastics/fusion/rotation_aware/real_camera_data.py`
- Test: `tests/rotation_aware/test_real_camera_data.py`

**Interfaces:**
- Consumes: canonical `PosePairTrial`, SAM3D frame NPZs, calibrated intrinsics/distortion, and `estimated_extrinsics.json`.
- Produces: `RealCameraTrial`, `load_real_camera_trials(...)`, `CameraWindowDataset`, and `CameraCompleteCycleDataset`.

- [ ] **Step 1: Write failing camera-audit and alignment tests**

Create fixtures with two aligned frame maps and synthetic SAM3D NPZ outputs.
Assert that:

```python
trials = load_real_camera_trials(
    raw_trials=[trial],
    sam3d_person_root=sam3d_root,
    camera_audit_path=audit_path,
    face_calibration_path=face_calibration,
    side_calibration_path=side_calibration,
    ablation="G4",
)
assert trials[0].camera_features.joint_features.shape == (4, 70, 8)
assert trials[0].camera_fit.person_id == "1"
assert not hasattr(trials[0], "triangulated_3d")
```

Also assert that G0 carries no camera data, G1/G2 mask the documented feature
subsets, G5 deterministically perturbs the rotation by 30 degrees, and missing
SAM3D frames fail with an explicit path.

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_real_camera_data.py -q
```

Expected: import failure because `real_camera_data` does not exist.

- [ ] **Step 3: Implement immutable real-data contracts**

Implement:

```python
@dataclass(frozen=True)
class PersonCameraFit:
    person_id: str
    fitted: FittedRelativeCamera
    method: str
    rig_cluster: int
    bone_cv_pct: float

@dataclass(frozen=True)
class RealCameraTrial:
    canonical_trial: CanonicalTrial
    camera_fit: PersonCameraFit | None
    camera_features: CameraFeatureSequence | None
    ablation: str
```

`load_person_camera_fit(...)` reads only camera-audit fields, normalizes
translation to unit length, and records a deterministic fit-ID vector without
loading triangulated outputs.

- [ ] **Step 4: Implement aligned 2D loading**

Load `pred_keypoints_2d` for every `face_map`/`side_map` pair, undistort with
the declared calibration and `P=K`, and pass pixel coordinates to the existing
camera feature builder. Apply the same G1--G5 masks as the Unity experiment.

- [ ] **Step 5: Implement camera dataset wrappers**

Wrap `PosePairWindowDataset` and `PosePairCompleteCycleDataset`. Resolve
features by `(person_id, trial_id)` and attach padded tensors:

```python
sample["camera_global_features"]  # [19]
sample["camera_joint_features"]   # [T,70,8]
sample["camera_valid"]            # [T,70]
```

G0 returns the underlying sample unchanged.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/rotation_aware/test_real_camera_data.py \
  tests/rotation_aware/test_camera_conditioning.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/gymnastics/fusion/rotation_aware/real_camera_data.py \
  tests/rotation_aware/test_real_camera_data.py
git commit -m "feat: adapt fitted camera features to real cycles"
```

### Task 2: Frozen-backbone training and test inference

**Files:**
- Create: `src/gymnastics/fusion/rotation_aware/real_camera_training.py`
- Test: `tests/rotation_aware/test_real_camera_training.py`

**Interfaces:**
- Consumes: one source A6 checkpoint, split-bound real camera trials, and `RealCameraTrainingConfig`.
- Produces: `train_real_camera_cell(...)`, `infer_real_camera_cell(...)`, checkpoints, provenance, and standard `fused_sequence.npz` files.

- [ ] **Step 1: Write failing freeze/provenance tests**

Assert:

```python
run = train_real_camera_cell(..., ablation="G4", seed=0)
payload = torch.load(run.checkpoint, weights_only=False)
assert payload["provenance"]["triangulated_3d_available_to_training"] is False
assert payload["provenance"]["test_people_available_to_training"] is False
assert payload["trainable_parameter_prefixes"] == [
    "camera_conditioner.",
    "camera_delta_head.",
]
```

Compare the source and trained checkpoints and assert every non-camera tensor
is bit-identical. Assert G0 reuses the source checkpoint and performs no
optimizer step.

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_real_camera_training.py -q
```

Expected: import failure because `real_camera_training` does not exist.

- [ ] **Step 3: Implement source expansion and freezing**

Seed Python, NumPy, and Torch before constructing the camera branch. Load A6
with `strict=False`, accepting only camera parameter names. Set
`requires_grad=False` on every non-camera parameter and build AdamW from only
the camera parameter list.

- [ ] **Step 4: Implement training**

Use existing `train_one_epoch` and `validate` with the source checkpoint's
loss/corruption configurations. Run 10 epochs, retain the checkpoint with the
best finite validation score, and write epoch metrics atomically.

- [ ] **Step 5: Implement camera-aware inference**

Use 128-frame windows and stride 64. Pass the matching
`CameraFeatureBundle`, restore predictions into the face world frame, and
write standard fields required by `discover_method_sequences`, including
frame maps, timestamps, joint validity, ablation, seed, checkpoint hash, and
camera-fit provenance.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/rotation_aware/test_real_camera_training.py \
  tests/rotation_aware/test_real_camera_data.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/gymnastics/fusion/rotation_aware/real_camera_training.py \
  tests/rotation_aware/test_real_camera_training.py
git commit -m "feat: train frozen camera branch on real data"
```

### Task 3: Pseudo-GT-isolated evaluation and report

**Files:**
- Create: `src/gymnastics/fusion/rotation_aware/real_camera_evaluation.py`
- Test: `tests/rotation_aware/test_real_camera_evaluation.py`

**Interfaces:**
- Consumes: complete frozen test inference matrix and triangulated references.
- Produces: `metrics_by_cycle.csv`, `metrics_by_person.csv`,
  `metrics_by_method.csv`, `paired_comparisons.csv`, and
  `real_camera_feature_report.md`.

- [ ] **Step 1: Write failing aggregation and negative-control tests**

Provide 14-person synthetic rows for three seeds and G0--G5. Assert that
cycles are pooled within person first, methods are paired on
`(person_id, seed)`, and the report refuses a camera claim whenever G5 matches
or beats G4.

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_real_camera_evaluation.py -q
```

Expected: import failure because `real_camera_evaluation` does not exist.

- [ ] **Step 3: Implement evaluation**

Load test inference first, then call `load_triangulated_references` and
`evaluate_person_trials(..., alignment="similarity")`. Reject any run whose
provenance does not prove test-person and pseudo-GT isolation.

- [ ] **Step 4: Implement statistics and report**

Report person-level mean/median MPJPE, seed standard deviation, paired
G1--G5-minus-G0 deltas, a person-clustered bootstrap 95% descriptive interval,
improved-person counts, and direct G4-minus-G5 results. Include camera audit
coverage and the fixed-rig failure observed before the experiment.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_real_camera_evaluation.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gymnastics/fusion/rotation_aware/real_camera_evaluation.py \
  tests/rotation_aware/test_real_camera_evaluation.py
git commit -m "feat: evaluate real camera feature pilot"
```

### Task 4: Configuration, CLI, execution, and validation

**Files:**
- Create: `configs/fusion/real_camera_pilot.yaml`
- Create: `src/gymnastics/fusion/rotation_aware/real_camera_cli.py`
- Test: `tests/rotation_aware/test_real_camera_cli.py`
- Modify: `docs/superpowers/plans/2026-07-30-real-camera-feature-pilot.md`

**Interfaces:**
- Produces standalone commands `train-matrix`, `evaluate`, and `report`.

- [ ] **Step 1: Write failing CLI tests**

Assert exact parsing of:

```bash
python -m gymnastics.fusion.rotation_aware.real_camera_cli train-matrix \
  --config configs/fusion/real_camera_pilot.yaml --device cuda:0
python -m gymnastics.fusion.rotation_aware.real_camera_cli evaluate \
  --config configs/fusion/real_camera_pilot.yaml
```

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_real_camera_cli.py -q
```

Expected: import failure because `real_camera_cli` does not exist.

- [ ] **Step 3: Implement config and resumable CLI**

The YAML declares all paths, source checkpoints by seed, G0--G5, seeds 0--2,
10 epochs, batch size 32, learning rate `1e-4`, weight decay `1e-4`, window
128/32/64, and output root. The CLI validates exactly 18 cells, reuses
complete artifacts, reports failures without silently retrying, and never
imports evaluation code during training.

- [ ] **Step 4: Run the full verification suite**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/rotation_aware tests/unity_benchmark -q
git diff --check
```

Expected: all tests pass.

- [ ] **Step 5: Run the experiment**

Run one matrix process per GPU, with disjoint seed subsets:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n gymnastic \
  python -m gymnastics.fusion.rotation_aware.real_camera_cli train-matrix \
  --config configs/fusion/real_camera_pilot.yaml --seed 0 --seed 2 --device cuda:0
CUDA_VISIBLE_DEVICES=1 conda run -n gymnastic \
  python -m gymnastics.fusion.rotation_aware.real_camera_cli train-matrix \
  --config configs/fusion/real_camera_pilot.yaml --seed 1 --device cuda:0
```

Monitor process state and output growth every 30 seconds. Hard timeout is six
hours. Do not auto-retry failed cells.

- [ ] **Step 6: Evaluate after training**

Run:

```bash
conda run -n gymnastic \
  python -m gymnastics.fusion.rotation_aware.real_camera_cli evaluate \
  --config configs/fusion/real_camera_pilot.yaml
```

Expected: a complete 18-cell report under
`local/runs/fitted_camera_real/fold_00/evaluation`.

- [ ] **Step 7: Validate artifacts and record conclusions**

Verify 18 cell provenance records, 18 checkpoint/source records, expected
test-cycle coverage, no NaNs in primary person-level metrics, and report
G0--G5 ranking plus G4-versus-G5 interpretation. Append exact measured results
to this plan.

- [ ] **Step 8: Commit**

```bash
git add configs/fusion/real_camera_pilot.yaml \
  src/gymnastics/fusion/rotation_aware/real_camera_cli.py \
  tests/rotation_aware/test_real_camera_cli.py \
  docs/superpowers/plans/2026-07-30-real-camera-feature-pilot.md
git commit -m "feat: run real-data fitted-camera pilot"
```

---

## Execution Record — 2026-07-30

Status: **completed; preregistered camera claim not supported**.

### Executed matrix

- Fold: `local/runs/cohort_cycle/folds/fold_00.json`
- People: 96 train / 27 validation / 14 test
- Cycles: 654 train / 181 validation / 93 test
- Cells: G0--G5 x seeds 0, 1, 2 = 18
- G0: unchanged A6 source weights, zero optimizer steps
- G1--G5: 10 epochs, batch size 32, AdamW `1e-4`, only
  `camera_conditioner.*` and `camera_delta_head.*` trainable
- Evaluation: person-pooled MPJPE after per-frame similarity alignment to
  triangulated pseudo-GT; all reported distances below are millimetres
- Outputs: `local/runs/fitted_camera_real/fold_00`

The A6 complete-cycle ROM term was not re-optimized during camera adaptation.
The already learned A6 rotation prior remained frozen, while the added camera
branch was trained with the window-level source objectives. This avoids the
batch-size-32 mismatch that would otherwise update an arbitrary 44/654 complete
cycles per epoch.

### Final ranking

| Rank | Cell | Mean person MPJPE (mm) | Median (mm) | SD of seed means (mm) |
|---:|---|---:|---:|---:|
| 1 | G0 | 60.5777 | 58.0508 | 0.8337 |
| 2 | G1 | 60.8000 | 58.1869 | 0.5970 |
| 3 | G2 | 60.8043 | 58.1890 | 0.6021 |
| 4 | G3 | 60.8411 | 58.1853 | 0.4909 |
| 5 | G4 | 60.8499 | 58.1773 | 0.4885 |
| 6 | G5 | 60.8504 | 58.1792 | 0.4875 |

G0 is the best cell. The 60.58 mm fold-0 number is not directly comparable to
the 64.05 mm all-137-person main-table result because this pilot evaluates only
the 14 held-out fold-0 people and averages three source seeds.

### Paired comparisons

| Cell | Baseline | Mean delta MPJPE (mm) | Person-clustered bootstrap 95% CI (mm) | Improved people |
|---|---|---:|---:|---:|
| G1 | G0 | +0.2223 | [+0.1734, +0.2716] | 0/14 |
| G2 | G0 | +0.2267 | [+0.1771, +0.2763] | 0/14 |
| G3 | G0 | +0.2635 | [+0.2061, +0.3186] | 0/14 |
| G4 | G0 | +0.2723 | [+0.1953, +0.3458] | 0/14 |
| G5 | G0 | +0.2728 | [+0.1959, +0.3463] | 0/14 |
| G4 | G5 | -0.0005 | [-0.0010, -0.00005] | 9/14 |

Correct-camera G4 is only 0.0005 mm better than wrong-camera G5, while both are
about 0.27 mm worse than G0. The difference is numerically negligible and does
not rescue the primary comparison. Therefore the fitted-camera claim fails the
predeclared requirement that G4 beat both G0 and G5 in a meaningful way.

### Interpretation

The added branch consistently perturbs an already strong frozen A6 solution.
The near identity of G4 and G5 shows that the trained residual is effectively
insensitive to whether the supplied camera rotation is correct. Plausible
contributors are:

1. the frozen A6 representation already captures the useful two-view and torso
   rotation information;
2. per-person camera fits are noisy (median held-out reprojection error
   6.2677 px);
3. window-level self-supervision has no explicit geometric objective forcing
   the branch to use epipolar or ray-angle features.

This experiment should be reported as a negative ablation or appendix result,
not promoted to the paper mainline. A follow-up would need an explicit
geometry-consistency loss, confidence gating, and preferably independently
calibrated camera labels.

### Integrity and completeness audit

- 18 checkpoints and 18 provenance records
- 1,674 inference sequences (18 x 93)
- 252 person rows (18 x 14)
- 738 non-camera checkpoint tensors verified bit-identical to their matching
  A6 source tensors
- all primary cycle/person/method MPJPE values finite
- every run records
  `triangulated_3d_available_to_training: false` and
  `test_people_available_to_training: false`
- compact observation cache contains 928/928 cycles; the selective loader was
  verified exactly equal to the standard SAM3D 2D loader (`max_abs = 0`)

Primary report:
`local/runs/fitted_camera_real/fold_00/evaluation/real_camera_feature_report.md`.
