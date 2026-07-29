# Fitted-Camera Feature Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leakage-safe fitted-camera features to the existing A6 trunk-rotation fusion network and run the Unity G0--G5 direction-transfer experiment.

**Architecture:** A new Unity camera-feature builder fits one relative rig from training-direction SAM3D 2D points and produces global and joint features. `RotationAwareFusionModel` accepts an optional typed camera bundle, encodes it additively for G1--G3 or through FiLM for G4--G5, while leaving the camera-free API byte-compatible. A separate runner trains only with existing A6 self-supervised losses, freezes artifacts, and evaluates against Unity-native 3D afterward.

**Tech Stack:** Python 3.11, PyTorch, NumPy, OpenCV, OmegaConf/YAML, pytest, existing `gymnastic` conda environment.

## Global Constraints

- Use `conda run -n gymnastic ...` for all project Python commands.
- Do not read Unity-native 3D or private triangulated 3D during G-series training.
- Fit cameras from training-fold SAM3D 2D only; held-out direction and static data are evaluation-only.
- Preserve current A0--A9 behavior and checkpoint loading when camera conditioning is disabled.
- Run two direction folds and seeds 0, 1, and 2.
- Store new artifacts under `local/runs/unity_benchmark/camera_feature_fusion`.

---

### Task 1: Fitted camera and feature contract

**Files:**
- Create: `src/gymnastics/benchmarks/unity/camera_features.py`
- Test: `tests/unity_benchmark/test_camera_features.py`

**Interfaces:**
- Produces: `FittedRelativeCamera`, `CameraFeatureSequence`,
  `fit_relative_camera_from_training_2d(...)`, and
  `build_camera_feature_sequence(...)`.
- `CameraFeatureSequence.global_features` has shape `[Cg]`;
  `joint_features` has shape `[T,70,Cj]`; `valid` has shape `[T,70]`.

- [ ] **Step 1: Write failing tests for a known synthetic rig**

```python
def test_fitted_camera_recovers_training_rig_without_evaluation_frames():
    fitted = fit_relative_camera_from_training_2d(
        pixels_train, valid_train, intrinsics, threshold_px=2.0
    )
    assert geodesic_deg(fitted.rotation_face_to_side, expected_rotation) < 2.0
    assert fitted.fit_sample_count == len(train_sample_ids)
    assert not set(fitted.fit_sample_ids) & set(test_sample_ids)
```

- [ ] **Step 2: Run the focused test and verify the missing-module failure**

Run: `conda run -n gymnastic python -m pytest tests/unity_benchmark/test_camera_features.py -q`

Expected: FAIL because `camera_features` does not exist.

- [ ] **Step 3: Implement immutable fitted-camera and feature dataclasses**

```python
@dataclass(frozen=True)
class FittedRelativeCamera:
    rotation_face_to_side: np.ndarray
    translation_direction_face_to_side: np.ndarray
    inlier_ratio: float
    holdout_reprojection_px: float
    fit_sample_ids: np.ndarray

@dataclass(frozen=True)
class CameraFeatureSequence:
    global_features: np.ndarray
    joint_features: np.ndarray
    valid: np.ndarray
    schema: tuple[str, ...]
```

Use `estimate_relative_pose` on even frames for audit, score odd frames with
`reprojection_error`, then refit on all training frames. Decompose Unity
projection matrices only for intrinsics; never use exact relative extrinsics in
the fitted result.

- [ ] **Step 4: Implement finite normalized features**

Global features contain rotation 6D, unit translation, normalized intrinsics,
inlier ratio, and `log1p(holdout_px)/log(101)`. Joint features contain normalized
pixels, symmetric Sampson/epipolar residual, ray intersection angle, and
validity flags. Invalid joints are zeroed.

- [ ] **Step 5: Verify focused tests pass**

Run: `conda run -n gymnastic python -m pytest tests/unity_benchmark/test_camera_features.py -q`

Expected: PASS.

### Task 2: Optional camera conditioning in A6

**Files:**
- Create: `src/gymnastics/fusion/rotation_aware/camera.py`
- Modify: `src/gymnastics/fusion/rotation_aware/model.py`
- Modify: `src/gymnastics/fusion/rotation_aware/training.py`
- Modify: `src/gymnastics/fusion/rotation_aware/inference.py`
- Test: `tests/rotation_aware/test_camera_conditioning.py`

**Interfaces:**
- Consumes: batched `CameraFeatureBundle(global_features, joint_features, valid)`.
- Produces: `CameraConditioningConfig` and optional
  `RotationAwareFusionModel(..., camera_config=...)`.

- [ ] **Step 1: Write failing tests for compatibility and camera gradients**

```python
def test_camera_disabled_model_is_exactly_legacy_compatible():
    legacy = RotationAwareFusionModel(SPEC, hidden_channels=16)
    explicit = RotationAwareFusionModel(
        SPEC, hidden_channels=16, camera_config=None
    )
    explicit.load_state_dict(legacy.state_dict())
    torch.testing.assert_close(run(legacy), run(explicit), atol=0, rtol=0)

def test_film_camera_features_reach_gate_and_receive_gradient():
    model = RotationAwareFusionModel(
        SPEC,
        hidden_channels=16,
        camera_config=CameraConditioningConfig(12, 8, mode="film"),
    )
    output = run(model, camera_features=bundle)
    output.fused_kpts.square().mean().backward()
    assert finite_nonzero_gradient(model.camera_conditioner)
```

- [ ] **Step 2: Run tests and verify signature/import failures**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_camera_conditioning.py -q`

Expected: FAIL because the camera API is absent.

- [ ] **Step 3: Implement typed bundle validation and additive/FiLM encoders**

```python
@dataclass(frozen=True)
class CameraFeatureBundle:
    global_features: Tensor
    joint_features: Tensor
    valid: Tensor

@dataclass(frozen=True)
class CameraConditioningConfig:
    global_channels: int
    joint_channels: int
    mode: Literal["additive", "film"]
```

Validate batch, frame, joint, dtype, and finiteness. Zero invalid joint rows.
For additive mode, add a zero-initialized projected camera embedding to
`fused_features`. For FiLM, apply bounded `gamma` and `beta`, both initialized
to zero, so the source A6 function is initially exact.

- [ ] **Step 4: Thread optional camera data through training and inference**

`_forward_prepared` reads the three camera tensors when present and passes a
bundle to `model.forward`. Inference accepts an optional per-sequence camera
feature provider; legacy callers pass none and remain unchanged.

- [ ] **Step 5: Run camera and legacy model tests**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_camera_conditioning.py tests/rotation_aware/test_model.py -q`

Expected: PASS with legacy swap-invariance unchanged when cameras are disabled.

### Task 3: Leakage-safe Unity camera-guided dataset

**Files:**
- Create: `src/gymnastics/benchmarks/unity/camera_guided_data.py`
- Test: `tests/unity_benchmark/test_camera_guided_data.py`

**Interfaces:**
- Consumes: one training `UnityFold`, cached SAM3D 2D/3D, and a feature subset.
- Produces: `UnityCameraGuidedSequence` without any GT field and
  `UnityCameraGuidedWindowDataset`.

- [ ] **Step 1: Write failing tests proving GT absence and fit isolation**

```python
def test_camera_guided_training_sequence_contains_no_unity_gt():
    sequence = build_camera_guided_sequences(..., fold=fold)["train"]
    assert not hasattr(sequence, "gt_unity16_m")
    assert set(sequence.camera.fit_sample_ids) <= set(sequence.sample_ids)
    assert not set(sequence.camera.fit_sample_ids) & set(test_ids)
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `conda run -n gymnastic python -m pytest tests/unity_benchmark/test_camera_guided_data.py -q`

- [ ] **Step 3: Implement sequence and window contracts**

Build raw and canonical trials through the existing Unity adapter. Load 2D only
for camera fitting/features. Attach padded `camera_global_features`,
`camera_joint_features`, and `camera_valid` tensors in each window. The sequence
type exposes no native 3D reference.

- [ ] **Step 4: Implement feature masks for G0--G5**

G0 returns no camera tensors. G1 selects global pose/intrinsics. G2 adds fit
quality. G3/G4 select all global and joint geometry. G5 deterministically rotates
the fitted camera by 30 degrees before constructing G4 features.

- [ ] **Step 5: Verify dataset tests**

Run: `conda run -n gymnastic python -m pytest tests/unity_benchmark/test_camera_guided_data.py -q`

Expected: PASS.

### Task 4: G-series training, inference, and provenance

**Files:**
- Create: `src/gymnastics/benchmarks/unity/camera_guided_training.py`
- Modify: `src/gymnastics/benchmarks/unity/cli.py`
- Modify: `configs/benchmarks/unity_supervised.yaml`
- Test: `tests/unity_benchmark/test_camera_guided_training.py`
- Test: `tests/unity_benchmark/test_supervised_cli.py`

**Interfaces:**
- Produces: `CameraGuidedTrainingConfig`, `CameraGuidedRun`,
  `train_camera_guided_run(...)`, `run_camera_guided_inference(...)`.
- CLI stages: `camera-feature-train`, `camera-feature-train-matrix`,
  `camera-feature-evaluate`, and `camera-feature-report`.

- [ ] **Step 1: Write failing tests for one-epoch training and strict provenance**

```python
def test_camera_guided_training_uses_only_self_supervised_losses(tmp_path):
    run = train_camera_guided_run(..., epochs=1)
    history = json.loads(run.history_path.read_text())
    assert set(history[0]) == {"epoch", *LossBreakdown_fields}
    assert "unity_3d" not in run.resolved_config_path.read_text()
```

- [ ] **Step 2: Run focused tests and verify missing API failures**

Run: `conda run -n gymnastic python -m pytest tests/unity_benchmark/test_camera_guided_training.py tests/unity_benchmark/test_supervised_cli.py -q`

- [ ] **Step 3: Implement source-checkpoint expansion and training**

Load the A6 checkpoint metadata, instantiate G0 without cameras or G1--G5 with
the correct camera config, copy all matching A6 weights, verify initial G output
equals A6 because the camera projection is zero-initialized, then train with
`train_one_epoch` and existing `LossConfig`/`CorruptionConfig`.

- [ ] **Step 4: Save immutable run artifacts**

Save final checkpoint, resolved YAML, history JSON, fitted-camera JSON,
feature-schema JSON, source and final SHA-256 values, Git commit, train sample
IDs, and a declaration that Unity GT was unavailable to training.

- [ ] **Step 5: Implement held-out and static inference**

Load only the held-out and static camera-guided sequences after training.
Windowed inference passes the stored fitted-camera features and writes standard
`MethodSequence` files.

- [ ] **Step 6: Add CLI/config matrix**

Configure G0--G5, folds `left_to_right/right_to_left`, seeds `0/1/2`, 100
epochs, hidden channels inherited from A6, and CUDA when available.

- [ ] **Step 7: Verify training and CLI tests**

Run: `conda run -n gymnastic python -m pytest tests/unity_benchmark/test_camera_guided_training.py tests/unity_benchmark/test_supervised_cli.py -q`

Expected: PASS.

### Task 5: Evaluation and G0--G5 report

**Files:**
- Create: `src/gymnastics/benchmarks/unity/camera_guided_evaluation.py`
- Test: `tests/unity_benchmark/test_camera_guided_evaluation.py`

**Interfaces:**
- Consumes: complete two-fold, three-seed G0--G5 inference matrix.
- Produces: `metrics_by_sequence.csv`, `by_method.csv`,
  `comparisons_vs_g0.csv`, and `camera_feature_report.md`.

- [ ] **Step 1: Write failing aggregation tests**

```python
def test_camera_guided_report_averages_seeds_then_folds():
    summary = aggregate_camera_guided_results(complete_rows)
    assert summary.loc["G4", "heldout_mpjpe_mm"] == expected_macro
    assert summary.loc["G4", "delta_vs_g0_mm"] == expected_delta
```

- [ ] **Step 2: Run and verify missing-module failure**

Run: `conda run -n gymnastic python -m pytest tests/unity_benchmark/test_camera_guided_evaluation.py -q`

- [ ] **Step 3: Implement native-GT evaluation after training**

Reuse the established one-Sim3-per-sequence Unity evaluator. Add axial-angle
MAE, ROM error, peak timing, fitted-camera audit metrics, and camera-ablation
diagnostics. Reject incomplete matrices.

- [ ] **Step 4: Implement paired comparisons**

Compute per-cell G1--G5 minus G0, macro-average seeds within direction and then
directions, and report bootstrap confidence intervals over the six paired cells
as descriptive intervals because the benchmark contains one avatar/rig.

- [ ] **Step 5: Verify evaluation tests**

Run: `conda run -n gymnastic python -m pytest tests/unity_benchmark/test_camera_guided_evaluation.py -q`

Expected: PASS.

### Task 6: Full verification and experiment execution

**Files:**
- Modify: `docs/superpowers/plans/2026-07-29-camera-feature-fusion.md`

**Interfaces:**
- Produces the final verified code and experiment artifacts.

- [ ] **Step 1: Run focused suites**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/rotation_aware \
  tests/unity_benchmark -q
```

Expected: all tests pass.

- [ ] **Step 2: Run the G0--G5 matrix**

Run:

```bash
conda run -n gymnastic python -m gymnastics.benchmarks.unity.cli \
  --config configs/benchmarks/unity_supervised.yaml \
  camera-feature-train-matrix
```

Expected: 36 completed cells, each with strict provenance.

- [ ] **Step 3: Evaluate and report**

Run:

```bash
conda run -n gymnastic python -m gymnastics.benchmarks.unity.cli \
  --config configs/benchmarks/unity_supervised.yaml \
  camera-feature-evaluate
conda run -n gymnastic python -m gymnastics.benchmarks.unity.cli \
  --config configs/benchmarks/unity_supervised.yaml \
  camera-feature-report
```

Expected: complete CSV and Markdown artifacts under
`local/runs/unity_benchmark/camera_feature_fusion/evaluation`.

- [ ] **Step 4: Verify artifact completeness and rerun tests**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/rotation_aware \
  tests/unity_benchmark -q
git status --short
```

Expected: tests pass; only planned source, test, config, and documentation files
are tracked changes. Experiment outputs remain ignored.

- [ ] **Step 5: Record measured conclusions**

Update this plan's execution notes with the exact G0--G5 rankings, paired
differences, camera-fit audit, negative-control result, limitations, and paths
to every report artifact. Do not modify the manuscript until the evidence
supports a mainline change.

## Execution notes (2026-07-29)

- Completed the full two-fold, three-seed G0--G5 matrix: 36 checkpoints,
  36 provenance records, and 72 held-out/static inference files; no failed
  cells.
- The first matrix exposed a Unity transfer failure in A6: the raw residual
  reached approximately `-6.2e7`, saturating every valid coordinate at the
  `tanh` bound. All six outputs were therefore byte-identical. Those artifacts
  are retained under
  `local/runs/unity_benchmark/camera_feature_fusion_saturation_audit`.
- Added a zero-initialized, independently bounded camera-motion residual bypass
  and a regression test proving it retains gradients when the original A6 head
  is saturated. G0 remains the original A6 behavior.
- Final held-out continuous MPJPE ranking (mm):
  G5 175.903, G4 175.905, G3 175.909, G2 175.932, G1 175.934, G0 176.088.
- G4 minus G0 was `-0.183 mm` (`-0.104%`), with descriptive paired-cell 95%
  interval `[-0.197, -0.168] mm` and improvement in 6/6 fold/seed cells.
  Angle MAE changed from 45.223 degrees (G0) to 45.252 degrees (G4).
- The wrong-camera G5 control was 0.001 mm better than G4 in the macro result
  and better in 5/6 paired cells. The static diagnostic was likewise
  indistinguishable (G5 271.221 mm, G4 271.222 mm). Therefore the experiment
  does not support a correct-camera geometry claim; the small improvement is
  consistent with the added residual capacity.
- Camera-fit audits were valid: left-to-right inlier ratio 0.637 and held-out
  reprojection 2.084 px; right-to-left inlier ratio 0.620 and held-out
  reprojection 1.967 px.
- Primary report:
  `local/runs/unity_benchmark/camera_feature_fusion/evaluation/camera_feature_report.md`.
  Machine-readable tables are beside it in `by_method.csv`,
  `comparisons_vs_g0.csv`, and `metrics_by_sequence.csv`.
- This result should remain a separate negative/diagnostic study and should not
  replace the paper mainline.
