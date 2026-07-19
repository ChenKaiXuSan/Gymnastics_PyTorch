# Rotation-Aware Fusion Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-supervised, trunk-rotation-aware face/side 3D KPT fusion mainline while preserving the current nine deterministic fusion methods as comparison baselines.

**Architecture:** Add an isolated `fuse.rotation_aware` package that adapts current SAM3D and split-cycle records into typed trials, performs differentiable canonical geometry, builds symmetric temporal features, and predicts bounded residuals over a deterministic quality-weighted base. Training is self-supervised; triangulated pseudo-GT is imported only by the evaluation layer.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyYAML, PyTorch, pytest, ruff, mypy.

## Global Constraints

- Prefix every project-code and test command with `conda run -n gymnastic`.
- Keep `python -m fuse` and all existing `logs/fuse_experiments` outputs unchanged.
- Use `logs/split_cycle/person_<id>/alignment_record_<id>.json` as the only mainline time-offset source.
- Do not use triangulated 3D data in training, pseudo-target construction, fusion weights, or checkpoint selection.
- Do not use RGB, 2D KPT, camera parameters, mesh vertices, or manually labelled angles.
- Resolve MHR70 roles through `SkeletonSpec`; do not hard-code a joint count in model or geometry APIs.
- Use trial-level scale, never per-frame scale.
- Preserve view-swap invariance and recompute fused trunk kinematics from fused KPT.
- Aggregate final metrics by person; cycles are training/evaluation units, not final ranking units.
- Every production behavior follows a red-green-refactor TDD cycle.

---

### Task 1: Data Contracts, SkeletonSpec, and Split-Cycle Trial Adapter

**Files:**
- Create: `configs/fuse/skeleton_mhr70.yaml`
- Create: `configs/fuse/rotation_aware.yaml`
- Create: `fuse/rotation_aware/__init__.py`
- Create: `fuse/rotation_aware/config.py`
- Create: `fuse/rotation_aware/schema.py`
- Create: `fuse/rotation_aware/data.py`
- Test: `tests/rotation_aware/test_data.py`

**Interfaces:**
- Consumes: current `load_sam3d_world_by_frame`, `load_split_alignment_offset`, `build_aligned_timeline`, and MHR70 names.
- Produces: `SkeletonSpec`, `PosePairTrial`, `load_skeleton_spec`, `load_person_trials`, `write_person_cache`, and `load_cached_trial`.

- [ ] **Step 1: Write failing contract and adapter tests**

```python
def test_load_person_trials_uses_split_cycle_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_sam3d_world_by_frame", fake_sam3d_loader)
    trials = data.load_person_trials("1", sam3d_root, split_root, spec)
    assert trials[0].trial_id == "cycle_000"
    assert trials[0].face_map.tolist() == [10, 11, 12]
    assert trials[0].side_map.tolist() == [7, 8, 9]


def test_pose_pair_trial_builds_finite_nonzero_valid_mask():
    points = np.array([[[1, 2, 3], [0, 0, 0], [np.nan, 1, 2]]], dtype=np.float32)
    valid = valid_from_points(points)
    assert valid.tolist() == [[True, False, False]]
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_data.py -q`

Expected: collection fails because `fuse.rotation_aware` does not exist.

- [ ] **Step 3: Implement immutable contracts and YAML-backed SkeletonSpec**

```python
@dataclass(frozen=True)
class PosePairTrial:
    face: np.ndarray
    side: np.ndarray
    valid_face: np.ndarray
    valid_side: np.ndarray
    timestamps: np.ndarray
    face_map: np.ndarray
    side_map: np.ndarray
    joint_names: tuple[str, ...]
    person_id: str
    trial_id: str
    fps: float
```

Validate shapes, joint order, monotonic frame maps, and required roles at construction boundaries. Derive pelvis and thorax as configured virtual roles rather than synthetic entries in the 70-joint output.

- [ ] **Step 4: Implement split-only person/cycle loading and compact cache**

Read each person's split record, align once using its offset, slice trials by recorded face/side cycle ranges, and save compressed trial arrays plus source/config metadata under `logs/fuse_rotation_aware/cache/person_<id>/`.

- [ ] **Step 5: Run focused and existing fuse tests**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_data.py tests/test_fuse_experiment_matrix.py -q`

Expected: all tests pass; existing fuse test count remains 12.

### Task 2: Differentiable Canonical Geometry and Trunk Kinematics

**Files:**
- Create: `fuse/rotation_aware/geometry.py`
- Create: `fuse/rotation_aware/trunk.py`
- Test: `tests/rotation_aware/test_geometry.py`
- Test: `tests/rotation_aware/test_trunk.py`

**Interfaces:**
- Consumes: `SkeletonSpec` and torch tensors `[B,T,J,3]` with masks `[B,T,J]`.
- Produces: `CanonicalTransform`, `CanonicalizedPose`, `canonicalize_pose`, `restore_pose`, `build_pelvis_frame`, `build_thorax_frame`, `relative_rotation`, `axial_rotation_angle`, and `extract_trunk_features`.

- [ ] **Step 1: Write known-geometry failing tests**

```python
def test_known_thorax_rotation_is_thirty_degrees():
    pose, valid = synthetic_mhr70_pose(theta_deg=30.0)
    theta, theta_valid = axial_rotation_angle_from_points(pose, valid, spec)
    assert theta_valid.all()
    torch.testing.assert_close(theta, torch.deg2rad(torch.tensor(30.0)), atol=8.7e-3, rtol=0)


def test_canonical_round_trip():
    canonical = canonicalize_pose(points, valid, spec)
    restored = restore_pose(canonical.points, canonical.transform)
    torch.testing.assert_close(restored[valid], points[valid], atol=1e-5, rtol=0)
```

- [ ] **Step 2: Run tests and verify missing-symbol failures**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_geometry.py tests/rotation_aware/test_trunk.py -q`

- [ ] **Step 3: Implement safe frame construction and trial-scale canonicalization**

Use safe normalization, explicit determinant correction, frame validity masks, previous-valid-frame fallback only for transform continuity, and a robust per-trial torso-length median scale.

- [ ] **Step 4: Implement relative SO(3), wrapped angle, omega, and alpha**

```python
def circular_diff(a: Tensor, b: Tensor) -> Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def relative_rotation(pelvis_rotation: Tensor, thorax_rotation: Tensor) -> Tensor:
    return pelvis_rotation.transpose(-1, -2) @ thorax_rotation
```

- [ ] **Step 5: Verify invariance, finite outputs, and gradients**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_geometry.py tests/rotation_aware/test_trunk.py -q`

Expected: translation, global rotation, scale, circular-boundary, degeneracy, round-trip, and known-angle tests pass.

### Task 3: Pose, Quality, Disagreement, and Deterministic Mainline Baselines

**Files:**
- Create: `fuse/rotation_aware/features.py`
- Create: `fuse/rotation_aware/base_fusion.py`
- Test: `tests/rotation_aware/test_features.py`
- Test: `tests/rotation_aware/test_base_fusion.py`

**Interfaces:**
- Consumes: canonicalized face/side tensors, trunk features, SkeletonSpec bones, and masks.
- Produces: `FeatureBundle`, `extract_pose_features`, `compute_quality_features`, `compute_disagreement_features`, and `quality_weighted_fusion`.

- [ ] **Step 1: Write failing feature and fusion tests**

```python
def test_identical_views_have_zero_disagreement():
    features = compute_disagreement_features(pose, pose, trunk, trunk, valid, valid)
    torch.testing.assert_close(features.coordinate_abs_delta, torch.zeros_like(pose))


def test_base_fusion_falls_back_to_only_valid_view():
    out = quality_weighted_fusion(face, side, face_valid, torch.zeros_like(side_valid), qf, qs)
    torch.testing.assert_close(out.points[face_valid], face[face_valid])
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_features.py tests/rotation_aware/test_base_fusion.py -q`

- [ ] **Step 3: Implement pose velocity/bone features and fixed robust quality**

Quality includes robust shoulder/hip/torso deviations, local rigidity, angular outliers, frame degeneracy, and valid ratio. Loss-facing quality tensors are detached.

- [ ] **Step 4: Implement disagreement and deterministic baselines**

Implement canonical arithmetic mean and quality-weighted mean with explicit both-invalid handling and swap-invariant output.

- [ ] **Step 5: Verify focused suite**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_features.py tests/rotation_aware/test_base_fusion.py -q`

### Task 4: Window Dataset and Reproducible Synthetic Corruption

**Files:**
- Create: `fuse/rotation_aware/corruptions.py`
- Create: `fuse/rotation_aware/dataset.py`
- Test: `tests/rotation_aware/test_corruptions.py`
- Test: `tests/rotation_aware/test_dataset.py`

**Interfaces:**
- Consumes: cached `PosePairTrial`, person-level split membership, window config, and deterministic seeds.
- Produces: `CorruptionBatch`, `apply_corruptions`, `PosePairWindowDataset`, and `collate_pose_pair_windows`.

- [ ] **Step 1: Write failing reproducibility and leakage tests**

```python
def test_corruption_is_reproducible_and_reference_is_unchanged():
    before = face.clone()
    a = apply_corruptions(face, side, valid_face, valid_side, seed=17, config=cfg)
    b = apply_corruptions(face, side, valid_face, valid_side, seed=17, config=cfg)
    torch.testing.assert_close(a.corrupted_face, b.corrupted_face)
    torch.testing.assert_close(face, before)


def test_subjects_do_not_cross_splits():
    manifest = build_split_manifest(fold_json)
    assert not (set(manifest.train) & set(manifest.val))
    assert not (set(manifest.train) & set(manifest.test))
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_corruptions.py tests/rotation_aware/test_dataset.py -q`

- [ ] **Step 3: Implement seven corruption families with exact masks**

Implement joint dropout, temporal block dropout, spike noise, random-walk drift, thorax rotation bias, freeze segment, and integer time shift. Preserve unmodified references and write fixed evaluation manifests.

- [ ] **Step 4: Implement 128-frame windowing and padding masks**

Train stride is 32, evaluation stride is 64, and padding never contributes to any loss.

- [ ] **Step 5: Verify focused suite**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_corruptions.py tests/rotation_aware/test_dataset.py -q`

### Task 5: Swap-Invariant Rotation-Aware TCN

**Files:**
- Create: `fuse/rotation_aware/model.py`
- Test: `tests/rotation_aware/test_model.py`

**Interfaces:**
- Consumes: canonical poses, `FeatureBundle`, cross-view features, and masks.
- Produces: `FusionOutput` and `RotationAwareFusionModel.forward`.

- [ ] **Step 1: Write failing model contract tests**

```python
def test_model_is_view_swap_invariant():
    out_lr = model(face, side, features_face, features_side, cross)
    out_rl = model(side, face, features_side, features_face, swapped_cross)
    torch.testing.assert_close(out_lr.fused_kpts, out_rl.fused_kpts, atol=1e-5, rtol=0)


def test_model_output_is_base_plus_delta_and_has_finite_gradient():
    out = model(face, side, face_features, side_features, cross, valid_face, valid_side)
    torch.testing.assert_close(out.fused_kpts, out.base_kpts + out.delta_kpts)
    out.fused_kpts.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_model.py -q`

- [ ] **Step 3: Implement shared encoders, symmetric fusion, and residual TCN**

Use shared view encoders and only symmetric mean/absolute-difference combinations. Use six non-causal dilated residual blocks with dilations 1, 2, 4, 8, 16, 32.

- [ ] **Step 4: Implement bounded joint residuals and fused kinematics**

```python
delta = max_delta_by_joint * torch.tanh(raw_delta)
fused = base.points + delta
fused_theta, fused_r_pt = trunk_kinematics(fused, output_valid, spec)
```

- [ ] **Step 5: Verify focused suite**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_model.py -q`

### Task 6: Self-Supervised Losses and Training Engine

**Files:**
- Create: `fuse/rotation_aware/losses.py`
- Create: `fuse/rotation_aware/training.py`
- Test: `tests/rotation_aware/test_losses.py`
- Test: `tests/rotation_aware/test_training.py`

**Interfaces:**
- Consumes: `FusionOutput`, corruption masks, unmodified references, quality, masks, and loss config.
- Produces: `LossBreakdown`, `compute_self_supervised_losses`, `train_one_epoch`, `validate`, `save_checkpoint`, and `load_checkpoint`.

- [ ] **Step 1: Write failing masked-loss and tiny-overfit tests**

```python
def test_perfect_prediction_has_zero_mask_loss():
    losses = compute_self_supervised_losses(perfect_output, perfect_batch, cfg)
    assert losses.mask.item() == pytest.approx(0.0, abs=1e-7)


def test_padding_and_invalid_points_contribute_no_loss():
    losses_a = compute_self_supervised_losses(output, batch, cfg)
    output.fused_kpts[~batch.loss_mask] = 1e6
    losses_b = compute_self_supervised_losses(output, batch, cfg)
    torch.testing.assert_close(losses_a.total, losses_b.total)
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_losses.py tests/rotation_aware/test_training.py -q`

- [ ] **Step 3: Implement pseudo-target selection and nine masked losses**

Implement corruption recovery, high-consensus identity, circular axial rotation, SO(3), trial bone length, local rigidity, adaptive temporal acceleration, minimal residual, and complete-cycle ROM.

- [ ] **Step 4: Implement deterministic training, validation score, and checkpoint metadata**

Checkpoint selection uses corruption recovery, bone CV, rotation consistency, identity preservation, and ROM retention. It must not import triangulation modules.

- [ ] **Step 5: Verify tiny overfit and finite training**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_losses.py tests/rotation_aware/test_training.py -q`

Expected: eight deterministic samples reduce loss without NaN.

### Task 7: CLI, Long-Sequence Inference, and Unified Person-Level Evaluation

**Files:**
- Create: `fuse/rotation_aware/inference.py`
- Create: `fuse/rotation_aware/evaluation.py`
- Create: `fuse/rotation_aware/visualization.py`
- Create: `fuse/rotation_aware/cli.py`
- Create: `fuse/rotation_aware/__main__.py`
- Test: `tests/rotation_aware/test_inference.py`
- Test: `tests/rotation_aware/test_evaluation.py`
- Test: `tests/rotation_aware/test_visualization.py`
- Test: `tests/rotation_aware/test_cli.py`

**Interfaces:**
- Consumes: cached trials, checkpoint, current compact-output conventions, and current triangulated evaluator.
- Produces: `prepare`, `train`, `infer`, and `evaluate` subcommands; compatible `fused_sequence.npz`; per-person CSV/JSON reports.

- [ ] **Step 1: Write failing CLI/output/evaluation tests**

```python
def test_inference_output_contains_compatible_and_mainline_fields(tmp_path):
    result = run_inference(model, trial, spec, output_root=tmp_path, run_id="test")
    with np.load(result.sequence_path) as data:
        assert {"kpts_world", "kpts_body", "kpts_fused_canonical", "face_map", "side_map"} <= set(data.files)


def test_evaluation_aggregates_cycles_by_person():
    rows = evaluate_person_trials("1", [cycle_a, cycle_b], references)
    assert len(rows.person_metrics) == 1
    assert rows.person_metrics[0].person_id == "1"
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_inference.py tests/rotation_aware/test_evaluation.py tests/rotation_aware/test_cli.py -q`

- [ ] **Step 3: Implement overlap-add inference and compatible export**

Average overlapping window predictions with deterministic taper weights, restore canonical output to face reference, and derive current unscaled `kpts_body` from restored points.

- [ ] **Step 4: Implement person-level self-supervised and external evaluation**

Triangulated imports live only in `evaluation.py`. Report existing baselines and new model together without modifying old output files. `visualization.py` writes theta/omega/quality curves and an optional four-skeleton animation from saved arrays without changing them.

- [ ] **Step 5: Implement CLI and smoke each subcommand**

Run: `conda run -n gymnastic python -m fuse.rotation_aware --help`

Expected: `prepare`, `train`, `infer`, and `evaluate` are listed.

### Task 8: Regression, Quality Gates, and Research Documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md` only by appending the approved mainline commands without overwriting existing user edits
- Create: `docs/rotation_aware_fusion.md`
- Create: `tests/rotation_aware/test_end_to_end.py`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: an end-to-end smoke path, reproducible commands, and explicit comparison/mainline research documentation.

- [ ] **Step 1: Write failing end-to-end synthetic test**

The test creates two synthetic views, prepares one cached trial, performs a tiny train, runs overlap inference, and verifies finite `[T,70,3]` output and person-level metrics.

- [ ] **Step 2: Verify RED, then add only required integration glue**

Run: `conda run -n gymnastic python -m pytest tests/rotation_aware/test_end_to_end.py -q`

- [ ] **Step 3: Document research boundaries and commands**

Document old methods as comparison experiments, `rotation_aware_self_supervised` as the paper mainline, split-only alignment, no-GT training, output fields, and run directory semantics.

- [ ] **Step 4: Run complete focused and regression suites**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware tests/test_fuse_experiment_matrix.py tests/test_sam3d_triangulation.py tests/test_compare_fused_triangulated.py -q
conda run -n gymnastic ruff check fuse/rotation_aware tests/rotation_aware
conda run -n gymnastic mypy fuse/rotation_aware
```

- [ ] **Step 5: Run one-person real-data smoke test**

```bash
conda run -n gymnastic python -m fuse.rotation_aware prepare --person 1
conda run -n gymnastic python -m fuse.rotation_aware infer --person 1 --checkpoint logs/fuse_rotation_aware/runs/tiny_smoke/checkpoints/best.pt --run-id tiny_smoke
conda run -n gymnastic python -m fuse.rotation_aware evaluate --person 1 --run-id tiny_smoke
```

Verify finite output, frame-map compatibility, no writes under existing method directories, and a single person-level metric row.
