# Unity Benchmark External Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible Unity external benchmark for both SAM3D single views, nine deterministic fusion methods, zero-shot A4--A9 rotation-aware checkpoints, SAM3D-2D triangulation, and oracle-2D triangulation against Unity native 3D keypoints.

**Architecture:** Add a focused `gymnastics.benchmarks.unity` package with immutable dataset contracts, exact Unity camera geometry, cached SAM3D inference, adapters into existing fusion functions, one common sequence-level Sim3 evaluator, and report generation. Extend the unified CLI with `gymnastics benchmark unity <stage>` so GPU inference, CPU fusion/evaluation, and reporting can be resumed independently.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, PyTorch, OmegaConf/PyYAML, pandas, matplotlib, pytest, existing SAM3D-Body and rotation-aware fusion modules.

## Global Constraints

- Run project code, tests, scripts, and Python tooling with `conda run -n gymnastic ...`.
- Read `/home/data/xchen/gymnastics/unity_benchmark` in place and never modify it.
- Write all generated artifacts below `local/runs/unity_benchmark`.
- Use Unity native 3D keypoints as the only evaluation ground truth.
- Do not train, fine-tune, select checkpoints, or tune valid fusion methods with Unity ground truth.
- Evaluate exactly the 16 approved homologous joints.
- Fit one Sim3 per method and evaluation sequence; do not use per-frame Procrustes as the main metric.
- Treat `sim3_face_stable_joint_weight` and `triangulation_oracle2d` as diagnostics excluded from the valid ranking.
- Use existing A4--A9 checkpoints without retraining.
- Record every missing detection or excluded point explicitly; do not silently drop frames.
- Do not alter existing real-person results under `local/runs/fuse_experiments` or `local/runs/fuse_rotation_aware`.

---

## File Structure

Create:

- `configs/benchmarks/unity.yaml` — dataset, output, SAM3D, skeleton, checkpoint, and evaluation defaults.
- `src/gymnastics/benchmarks/__init__.py` — benchmark namespace.
- `src/gymnastics/benchmarks/unity/__init__.py` — public Unity benchmark API.
- `src/gymnastics/benchmarks/unity/schema.py` — immutable frame, camera, benchmark, and method-output contracts.
- `src/gymnastics/benchmarks/unity/dataset.py` — manifest parsing, integrity validation, and three-sequence grouping.
- `src/gymnastics/benchmarks/unity/mapping.py` — fixed 16-joint MHR70-to-Unity mapping and validity propagation.
- `src/gymnastics/benchmarks/unity/geometry.py` — Unity projection matrices, DLT triangulation, and reprojection.
- `src/gymnastics/benchmarks/unity/sam3d.py` — one-estimator-per-process image inference and cache loading.
- `src/gymnastics/benchmarks/unity/fusion.py` — deterministic and zero-shot rotation-aware adapters.
- `src/gymnastics/benchmarks/unity/evaluation.py` — sequence-level Sim3, pose, visibility, and angle metrics.
- `src/gymnastics/benchmarks/unity/report.py` — CSV/JSON/Markdown tables and figures.
- `src/gymnastics/benchmarks/unity/cli.py` — staged and full-run orchestration.
- `tests/unity_benchmark/__init__.py` — test package marker.
- `tests/unity_benchmark/conftest.py` — compact synthetic Unity fixture.
- `tests/unity_benchmark/test_dataset_mapping.py` — dataset and mapping tests.
- `tests/unity_benchmark/test_geometry.py` — camera/oracle triangulation tests.
- `tests/unity_benchmark/test_sam3d_cache.py` — inference-cache tests with a fake estimator.
- `tests/unity_benchmark/test_fusion.py` — deterministic and checkpoint adapter tests.
- `tests/unity_benchmark/test_evaluation_report.py` — metrics, leakage partition, output, and report tests.
- `tests/unity_benchmark/test_cli.py` — unified and staged CLI tests.

Modify:

- `src/gymnastics/cli.py` — expose `benchmark unity`.
- `tests/structure/test_cli.py` — require the new top-level benchmark command.

No existing deterministic or rotation-aware algorithm implementation should be
copied or changed unless a narrowly scoped reusable public helper is required
and covered by its existing tests.

---

### Task 1: Dataset Contracts, Configuration, and 16-Joint Mapping

**Files:**
- Create: `configs/benchmarks/unity.yaml`
- Create: `src/gymnastics/benchmarks/__init__.py`
- Create: `src/gymnastics/benchmarks/unity/__init__.py`
- Create: `src/gymnastics/benchmarks/unity/schema.py`
- Create: `src/gymnastics/benchmarks/unity/dataset.py`
- Create: `src/gymnastics/benchmarks/unity/mapping.py`
- Create: `tests/unity_benchmark/__init__.py`
- Create: `tests/unity_benchmark/conftest.py`
- Create: `tests/unity_benchmark/test_dataset_mapping.py`

**Interfaces:**
- Produces: `load_unity_benchmark(root: Path) -> UnityBenchmark`
- Produces: `group_evaluation_sequences(benchmark: UnityBenchmark) -> dict[str, tuple[UnityFrame, ...]]`
- Produces: `map_mhr70_to_unity(points: np.ndarray, valid: np.ndarray | None = None) -> MappedPose`
- Produces: constants `EVALUATION_JOINT_NAMES`, `UNITY_JOINT_INDICES`, and `MHR70_MAPPING_VERSION`

- [ ] **Step 1: Write failing manifest and grouping tests**

Create a two-record synthetic manifest fixture and assert immutable shapes,
absolute image paths, global `sample_id` identity, and static grouping:

```python
def test_loads_manifest_and_groups_static_samples(unity_root: Path) -> None:
    benchmark = load_unity_benchmark(unity_root)
    assert benchmark.joint_names == tuple(UNITY_22)
    assert [frame.sample_id for frame in benchmark.frames] == [0, 1, 2]
    groups = group_evaluation_sequences(benchmark)
    assert tuple(groups) == ("static_sweep", "continuous_left_060_r00")
    assert [frame.sample_id for frame in groups["static_sweep"]] == [0, 1]
    assert groups["continuous_left_060_r00"][0].image_paths["cam0"].is_absolute()
```

- [ ] **Step 2: Write failing 16-joint mapping tests**

Construct a `[2, 70, 3]` array whose joint index is encoded in its value and
assert direct and derived mappings:

```python
def test_maps_exact_sixteen_homologous_joints() -> None:
    points = np.zeros((2, 70, 3), dtype=np.float32)
    for index in range(70):
        points[:, index] = index + 1
    mapped = map_mhr70_to_unity(points)
    assert mapped.points.shape == (2, 16, 3)
    assert mapped.joint_names == EVALUATION_JOINT_NAMES
    np.testing.assert_allclose(
        mapped.points[:, mapped.index("Hips")],
        0.5 * (points[:, 9] + points[:, 10]),
    )
    np.testing.assert_allclose(
        mapped.points[:, mapped.index("LeftToes")],
        0.5 * (points[:, 15] + points[:, 16]),
    )
```

Add a second test that invalidating either hip or either toe source invalidates
the corresponding derived target.

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark/test_dataset_mapping.py -q
```

Expected: collection fails because `gymnastics.benchmarks.unity` does not exist.

- [ ] **Step 4: Implement immutable contracts**

Define:

```python
@dataclass(frozen=True)
class UnityCamera:
    camera_id: str
    image_size: tuple[int, int]
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
    clip_projection: np.ndarray

@dataclass(frozen=True)
class UnityFrame:
    sample_id: int
    sequence_id: str
    frame_index: int
    sample_type: str
    phase: str
    time_seconds: float
    actual_angle_deg: float
    image_paths: Mapping[str, Path]
    gt_world_m: np.ndarray
    gt_available: np.ndarray
    gt_pixels: Mapping[str, np.ndarray]
    visible: Mapping[str, np.ndarray]

@dataclass(frozen=True)
class UnityBenchmark:
    root: Path
    joint_names: tuple[str, ...]
    cameras: Mapping[str, UnityCamera]
    frames: tuple[UnityFrame, ...]

@dataclass(frozen=True)
class MappedPose:
    points: np.ndarray
    valid: np.ndarray
    joint_names: tuple[str, ...]

    def index(self, name: str) -> int:
        return self.joint_names.index(name)
```

Copy arrays into read-only storage in `__post_init__`; validate shapes and
finite scalar metadata.

- [ ] **Step 5: Implement manifest validation and grouping**

`load_unity_benchmark` must:

- require `manifest.jsonl`, `skeleton.json`, and `cameras.json`;
- validate strictly increasing, unique `sample_id`;
- validate 22-element 3D and per-camera 2D lists;
- resolve image paths against `root` and require files to exist;
- validate that both cameras exist and dimensions agree with metadata;
- create camera contracts from `cameras.json`;
- expose explicit `gt_available` and `visible` masks.

`group_evaluation_sequences` maps all `sample_type == "static"` frames to
`static_sweep`, sorted by `actual_angle_deg`; continuous frames retain their
`sequence_id` and are sorted by `frame_index`.

- [ ] **Step 6: Implement the exact mapping table**

Use MHR70 indices from `gymnastics.common.skeletons.mhr70.pose_info` and define
the approved outputs in this order:

```python
EVALUATION_JOINT_NAMES = (
    "Hips", "Neck",
    "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
)
```

Direct mappings use MHR70 `neck`, shoulders, elbows, wrists, hips, knees, and
ankles. `Hips` is the mean of MHR70 hips. Each `Toes` joint is the mean of its
big- and small-toe tips. Derived validity is the logical AND of every source.

- [ ] **Step 7: Add resolved configuration**

Create `configs/benchmarks/unity.yaml` with:

```yaml
paths:
  dataset_root: /home/data/xchen/gymnastics/unity_benchmark
  output_root: local/runs/unity_benchmark
  sam3d_config: configs/sam3d/sam3d_body.yaml
  skeleton: configs/fusion/skeleton_mhr70.yaml
checkpoints:
  A4: local/runs/fuse_rotation_aware/runs/all137_a4_e100_seed0/checkpoints/best.pt
  A5: local/runs/fuse_rotation_aware/runs/all137_a5_e100_seed0/checkpoints/best.pt
  A6: local/runs/fuse_rotation_aware/runs/all137_a6_e100_seed0/checkpoints/best.pt
  A7: local/runs/fuse_rotation_aware/runs/all137_a7_e100_seed0/checkpoints/best.pt
  A8: local/runs/fuse_rotation_aware/runs/all137_a8_e100_seed0/checkpoints/best.pt
  A9: local/runs/fuse_rotation_aware/runs/all137_a9_e100_seed0/checkpoints/best.pt
data:
  fps: 60.0
evaluation:
  alignment: similarity
  camera_reference: cam0
```

- [ ] **Step 8: Run mapping tests**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark/test_dataset_mapping.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add configs/benchmarks/unity.yaml \
  src/gymnastics/benchmarks \
  tests/unity_benchmark
git commit -m "feat: add Unity benchmark data contracts"
```

---

### Task 2: Unity Camera Geometry and Both Triangulation Baselines

**Files:**
- Create: `src/gymnastics/benchmarks/unity/geometry.py`
- Create: `tests/unity_benchmark/test_geometry.py`
- Modify: `src/gymnastics/benchmarks/unity/schema.py`

**Interfaces:**
- Consumes: `UnityCamera`, `UnityBenchmark`, and `UnityFrame`
- Produces: `pixel_projection(camera: UnityCamera) -> np.ndarray`
- Produces: `project_world(points_m: np.ndarray, camera: UnityCamera) -> tuple[np.ndarray, np.ndarray]`
- Produces: `triangulate_pixels(cam0_pixels: np.ndarray, cam1_pixels: np.ndarray, cam0: UnityCamera, cam1: UnityCamera) -> np.ndarray`
- Produces: `run_oracle_triangulation(benchmark: UnityBenchmark, output_root: Path) -> tuple[MethodSequence, ...]`
- Produces: `run_sam3d_triangulation(benchmark: UnityBenchmark, cache_root: Path, output_root: Path) -> tuple[MethodSequence, ...]`

- [ ] **Step 1: Write failing projection and DLT tests**

Use the real first Unity record for a read-only geometry test:

```python
def test_projection_reproduces_manifest_pixels(real_unity_root: Path) -> None:
    benchmark = load_unity_benchmark(real_unity_root)
    frame = benchmark.frames[0]
    for camera_id, camera in benchmark.cameras.items():
        pixels, depth = project_world(frame.gt_world_m, camera)
        mask = frame.gt_available
        np.testing.assert_allclose(
            pixels[mask], frame.gt_pixels[camera_id][mask], atol=1e-3
        )
        assert np.all(depth[mask] > 0)

def test_oracle_dlt_recovers_world_points(real_unity_root: Path) -> None:
    benchmark = load_unity_benchmark(real_unity_root)
    frame = benchmark.frames[0]
    reconstructed = triangulate_pixels(
        frame.gt_pixels["cam0"], frame.gt_pixels["cam1"],
        benchmark.cameras["cam0"], benchmark.cameras["cam1"],
    )
    np.testing.assert_allclose(
        reconstructed[frame.gt_available],
        frame.gt_world_m[frame.gt_available],
        atol=1e-4,
    )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark/test_geometry.py -q
```

Expected: import failure for `geometry`.

- [ ] **Step 3: Implement the exact Unity pixel projection**

Derive a 3x4 pixel projection matrix from the documented clip-space matrices:

```python
def pixel_projection(camera: UnityCamera) -> np.ndarray:
    clip_from_world = camera.clip_projection @ camera.world_to_camera
    sx = (camera.image_size[0] - 1) / 2.0
    sy = (camera.image_size[1] - 1) / 2.0
    return np.stack(
        (
            sx * (clip_from_world[0] + clip_from_world[3]),
            sy * (clip_from_world[3] - clip_from_world[1]),
            clip_from_world[3],
        )
    )
```

`project_world` multiplies homogeneous points by this matrix and separately
computes documented positive depth from `inverse(camera_to_world)`.

- [ ] **Step 4: Implement finite-mask DLT triangulation**

Use `cv2.triangulatePoints` with the two pixel projection matrices. Preserve
shape, put `NaN` at invalid correspondences, reject points whose homogeneous
denominator is numerically zero, and do not require visible status: exact
available 2D coordinates remain geometrically valid when self-occluded.

- [ ] **Step 5: Implement oracle and SAM3D-2D sequence outputs**

Define a shared method contract:

```python
@dataclass(frozen=True)
class MethodSequence:
    method: str
    sequence_id: str
    sample_ids: np.ndarray
    points: np.ndarray
    valid: np.ndarray
    joint_names: tuple[str, ...]
    metadata: Mapping[str, object]
```

Oracle triangulates 22 Unity joints, selects the same 16 target indices, and
stores raw reprojection errors. SAM3D-2D triangulates all 70 predicted MHR
joints, then calls `map_mhr70_to_unity`.

- [ ] **Step 6: Run geometry tests**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark/test_geometry.py -q
```

Expected: projection maximum absolute error is below `1e-3 px`; oracle raw 3D
maximum error is below `1e-4 m`.

- [ ] **Step 7: Commit**

```bash
git add src/gymnastics/benchmarks/unity/geometry.py \
  src/gymnastics/benchmarks/unity/schema.py \
  tests/unity_benchmark/test_geometry.py
git commit -m "feat: add Unity calibrated triangulation"
```

---

### Task 3: Cached SAM3D Inference for Unity Images

**Files:**
- Create: `src/gymnastics/benchmarks/unity/sam3d.py`
- Create: `tests/unity_benchmark/test_sam3d_cache.py`

**Interfaces:**
- Consumes: `UnityBenchmark`, OmegaConf SAM3D config, and camera selection
- Produces: `run_sam3d_inference(benchmark: UnityBenchmark, camera_id: str, output_root: Path, config_path: Path, device: str, force: bool = False, estimator_factory: Callable | None = None) -> InferenceSummary`
- Produces: `load_sam3d_camera_cache(root: Path, camera_id: str, sample_ids: Sequence[int]) -> CachedPose`

Define the produced contracts in `sam3d.py`:

```python
@dataclass(frozen=True)
class InferenceSummary:
    camera_id: str
    expected: int
    completed: int
    reused: int
    failed: tuple[Mapping[str, object], ...]
    summary_path: Path

@dataclass(frozen=True)
class CachedPose:
    camera_id: str
    sample_ids: np.ndarray
    points_3d: np.ndarray
    points_2d: np.ndarray
    valid_3d: np.ndarray
    valid_2d: np.ndarray
    failures: Mapping[int, str]
```

- [ ] **Step 1: Write failing fake-estimator cache tests**

Inject a fake estimator whose `process_one_image` returns one MHR70 dictionary:

```python
def test_inference_loads_estimator_once_and_resumes(
    unity_root: Path, tmp_path: Path
) -> None:
    factory = FakeEstimatorFactory()
    benchmark = load_unity_benchmark(unity_root)
    first = run_sam3d_inference(
        benchmark, "cam0", tmp_path, CONFIG, "cpu",
        estimator_factory=factory,
    )
    second = run_sam3d_inference(
        benchmark, "cam0", tmp_path, CONFIG, "cpu",
        estimator_factory=factory,
    )
    assert factory.calls == 1
    assert first.completed == len(benchmark.frames)
    assert second.reused == len(benchmark.frames)
```

Also test that an empty detector result creates a failure record containing the
`sample_id` and does not create a fake pose.

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark/test_sam3d_cache.py -q
```

Expected: import failure for `sam3d`.

- [ ] **Step 3: Implement lazy heavyweight imports**

Keep module import CPU-testable. Import `cv2`, OmegaConf, and
`gymnastics.sam3d.infer.setup_sam_3d_body` only inside the default estimator
factory. Override the loaded SAM3D configuration device with the requested
device.

- [ ] **Step 4: Implement clean per-sample caches**

For each `sample_id`, read the matching image and select the largest detected
person with the existing `select_best_person`. Save:

```text
sam3d/<camera_id>/<sample_id:08d>.npz
  pred_keypoints_3d [70,3]
  pred_keypoints_2d [70,2]
  valid_3d [70]
  valid_2d [70]
  sample_id scalar
  camera_id scalar
  source_image scalar
  metadata JSON scalar
```

Write a camera-level `summary.json` with expected, completed, reused, failed,
checkpoint, config path, and failure reasons. Use temporary files followed by
atomic replacement.

- [ ] **Step 5: Validate cache identity on load**

`load_sam3d_camera_cache` requires exact requested sample IDs in requested
order. Missing files raise a message listing IDs unless those IDs have explicit
failure records; explicit failures remain invalid frames in downstream arrays.

- [ ] **Step 6: Run cache tests**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark/test_sam3d_cache.py -q
```

Expected: all tests pass without importing SAM3D third-party modules.

- [ ] **Step 7: Commit**

```bash
git add src/gymnastics/benchmarks/unity/sam3d.py \
  tests/unity_benchmark/test_sam3d_cache.py
git commit -m "feat: cache Unity SAM3D inference"
```

---

### Task 4: Deterministic Fusion Adapter

**Files:**
- Create: `src/gymnastics/benchmarks/unity/fusion.py`
- Create: `tests/unity_benchmark/test_fusion.py`

**Interfaces:**
- Consumes: synchronized MHR70 caches and grouped Unity sequences
- Produces: `build_pose_pair_trial(...) -> PosePairTrial`
- Produces: `run_deterministic_fusion(benchmark: UnityBenchmark, cache_root: Path, output_root: Path, methods: Sequence[str] = ALL_METHODS) -> tuple[MethodSequence, ...]`
- Produces: `estimate_leaky_joint_weights(face: np.ndarray, aligned_side: np.ndarray, unity_gt: np.ndarray, gt_valid: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Write failing deterministic parity tests**

Use small valid MHR70 arrays and compare the adapter outputs with existing pure
functions:

```python
@pytest.mark.parametrize("method", ALL_METHODS)
def test_runs_every_named_deterministic_method(method: str, pose_pair) -> None:
    output = fuse_deterministic_sequence(
        method, pose_pair.face, pose_pair.side,
        leaky_weights=np.full((70, 2), 0.5, dtype=np.float32)
        if method == "sim3_face_stable_joint_weight" else None,
    )
    assert output.shape == pose_pair.face.shape
    assert np.isfinite(output).all()
```

Assert `avg_body_current` is exactly
`experiment_matrix.current_body_average(face, side)` and that the adapter
rejects unknown method names.

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark/test_fusion.py::test_runs_every_named_deterministic_method -q
```

Expected: import failure for `fusion`.

- [ ] **Step 3: Extract only orchestration, reuse all algorithms**

Import and call:

- `current_body_average`;
- `root_align_to_reference`;
- `sim3_align_to_reference`;
- `fuse_weighted`;
- `bodypart_weights`;
- `smooth_sim3_alignment`;
- `smooth_sequence`;
- `STABLE_SIM3_JOINTS`;
- `ALL_METHODS`.

Mirror the existing method dispatch without file-system/person assumptions.
`cam0` is the face/reference argument and the synchronized offset is zero.

- [ ] **Step 4: Implement the explicit leaky diagnostic**

Map face, aligned side, and Unity GT to the 16-joint set. Compute per-joint
single-view errors after the same sequence-level Sim3 evaluation alignment,
then expand weights back to affected MHR70 sources deterministically. Store
`ranking_group=diagnostic`, `leakage=unity_gt_joint_weights`, and the weights in
metadata.

No valid method function receives the benchmark or Unity GT arguments.

- [ ] **Step 5: Save per-method sequence files**

Write:

```text
fusion/deterministic/<method>/<sequence_id>.npz
```

with MHR70 `points`, validity, sample IDs, method, camera reference, and
diagnostic metadata. Use the mapping module only in evaluation; preserve full
MHR70 for consistent downstream use.

- [ ] **Step 6: Run deterministic tests plus existing regression tests**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark/test_fusion.py \
  tests/test_fuse_experiment_matrix.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/gymnastics/benchmarks/unity/fusion.py \
  tests/unity_benchmark/test_fusion.py
git commit -m "feat: adapt deterministic fusion to Unity"
```

---

### Task 5: Zero-Shot A4--A9 Rotation-Aware Adapter

**Files:**
- Modify: `src/gymnastics/benchmarks/unity/fusion.py`
- Modify: `tests/unity_benchmark/test_fusion.py`

**Interfaces:**
- Consumes: `PosePairTrial`, skeleton YAML, and one existing `best.pt`
- Produces: `load_rotation_aware_model(checkpoint: Path, skeleton_path: Path, device: str) -> LoadedRotationAware`
- Produces: `run_rotation_aware_fusion(benchmark: UnityBenchmark, cache_root: Path, output_root: Path, checkpoints: Mapping[str, Path], device: str = "cpu") -> tuple[MethodSequence, ...]`

Define:

```python
@dataclass(frozen=True)
class LoadedRotationAware:
    model: RotationAwareFusionModel
    skeleton: SkeletonSpec
    ablation: str
    hidden_channels: int
    checkpoint_path: Path
    checkpoint_sha256: str
    provenance: Mapping[str, object]
```

- [ ] **Step 1: Write failing checkpoint-contract test**

Create a tiny valid checkpoint using the repository model and existing
`save_checkpoint`, then assert ablation, hidden width, and twist-residual
selection are restored from checkpoint metadata rather than active Unity
configuration:

```python
def test_loads_rotation_model_from_checkpoint_metadata(
    tiny_checkpoint: Path, skeleton_path: Path
) -> None:
    loaded = load_rotation_aware_model(tiny_checkpoint, skeleton_path, "cpu")
    assert loaded.ablation == "A8"
    assert loaded.hidden_channels == 8
    assert loaded.checkpoint_path == tiny_checkpoint
```

- [ ] **Step 2: Write failing zero-shot trial test**

Monkeypatch `run_inference` and verify three Unity sequences are passed as
MHR70 `PosePairTrial` instances with `person_id="unity"`, zero-offset equal
sample maps, 60 Hz timestamps, and no Unity GT in `source_metadata`.

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark/test_fusion.py -k rotation -q
```

Expected: missing rotation-aware adapter symbols.

- [ ] **Step 4: Implement safe checkpoint loading**

Read `training_config` from the checkpoint, instantiate:

```python
model = RotationAwareFusionModel(
    skeleton,
    hidden_channels=int(training["hidden_channels"]),
    twist_residual=str(training["ablation"]) in {"A8", "A9"},
)
payload = load_checkpoint(checkpoint, model, map_location=device)
model.to(device).eval()
```

Validate skeleton metadata, checkpoint existence, non-empty provenance, and
requested ablation matching saved ablation. Compute and store SHA256.

- [ ] **Step 5: Implement zero-shot inference**

Build one `PosePairTrial` for each evaluation sequence from synchronized
cam0/cam1 3D predictions. Call existing `run_inference` with window length 128
and stride 64. Run under `torch.inference_mode()`. Save benchmark copies under:

```text
fusion/rotation_aware/<ablation>/<sequence_id>.npz
```

The adapter metadata records `training_source=real_gymnastics`,
`unity_training=false`, checkpoint path/hash, and saved training provenance.
No Unity GT arrays are passed into model loading or inference.

- [ ] **Step 6: Run rotation-aware and existing inference tests**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark/test_fusion.py \
  tests/rotation_aware/test_inference.py \
  tests/rotation_aware/test_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/gymnastics/benchmarks/unity/fusion.py \
  tests/unity_benchmark/test_fusion.py
git commit -m "feat: add zero-shot Unity fusion evaluation"
```

---

### Task 6: Common Sim3 Evaluation, Visibility, Angle Metrics, and Report

**Files:**
- Create: `src/gymnastics/benchmarks/unity/evaluation.py`
- Create: `src/gymnastics/benchmarks/unity/report.py`
- Create: `tests/unity_benchmark/test_evaluation_report.py`

**Interfaces:**
- Consumes: mapped `MethodSequence` objects and `UnityBenchmark`
- Produces: `evaluate_method_sequence(candidate: MethodSequence, ground_truth: MethodSequence, visibility: Mapping[str, np.ndarray]) -> EvaluationResult`
- Produces: `evaluate_all(...) -> EvaluationBundle`
- Produces: `write_report(bundle: EvaluationBundle, output_root: Path, provenance: Mapping[str, object]) -> Path`

Define:

```python
@dataclass(frozen=True)
class EvaluationResult:
    method: str
    sequence_id: str
    sample_ids: np.ndarray
    joint_names: tuple[str, ...]
    errors_m: np.ndarray
    valid: np.ndarray
    aligned_points_m: np.ndarray
    summary: Mapping[str, float | int]
    angle_errors_deg: np.ndarray
    metadata: Mapping[str, object]

@dataclass(frozen=True)
class EvaluationBundle:
    results: tuple[EvaluationResult, ...]
    failures: tuple[Mapping[str, object], ...]
    valid_ranking: tuple[Mapping[str, object], ...]
    diagnostics: tuple[Mapping[str, object], ...]
    tables: Mapping[str, tuple[Mapping[str, object], ...]]
    provenance: Mapping[str, object]
```

- [ ] **Step 1: Write failing sequence-level Sim3 tests**

Create two frames related by one known Sim3 and assert zero error. Then perturb
the second frame with an extra rotation and assert the error remains non-zero,
proving the evaluator did not fit per-frame transforms:

```python
def test_similarity_alignment_is_one_transform_per_sequence() -> None:
    gt = make_two_frame_pose()
    candidate = apply_known_sim3(gt)
    result = sequence_joint_errors(candidate, gt)
    assert result.valid.all()
    assert result.errors.max() < 1e-6

    candidate[1] = candidate[1] @ rotation_z(np.deg2rad(20))
    result = sequence_joint_errors(candidate, gt)
    assert result.errors[1].mean() > 1e-3
```

- [ ] **Step 2: Write failing summary, visibility, and ranking tests**

Assert:

- metres are converted to millimetres exactly once;
- summary contains mean, median, and P95;
- cam0/cam1 visibility partitions are counted;
- diagnostic methods are absent from `valid_ranking`;
- all expected method IDs appear either in results or explicit failures.

- [ ] **Step 3: Write failing angle tests**

Build pelvis and shoulder axes with known axial rotations. Assert the shared
angle extractor returns correct signed degrees, neutral offset is common across
methods, and wrapped residuals use `[-180, 180)`:

```python
assert angular_residual_deg(np.array([179.0]), np.array([-179.0])) == pytest.approx([-2.0])
```

- [ ] **Step 4: Run tests and verify they fail**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark/test_evaluation_report.py -q
```

Expected: imports fail for evaluation and report.

- [ ] **Step 5: Implement one-transform sequence evaluation**

Reuse `fit_similarity` and `apply_sim3` from the deterministic module. Fit on
all mutually valid `[T, 16, 3]` points in one sequence. Return aligned
predictions, `[T, 16]` errors in metres, and the fit transform. Do not call
`align_candidate(..., "procrustes")`.

- [ ] **Step 6: Implement pose and visibility tables**

Produce row-oriented records for:

- `metrics_summary.csv`;
- `metrics_by_sequence.csv`;
- `metrics_by_joint.csv`;
- `metrics_by_visibility.csv`;
- `per_frame_errors.npz`.

Main overall MPJPE pools valid points after each sequence's shared alignment.
Visibility rows include `cam0_visible`, `cam0_occluded`, `cam1_visible`,
`cam1_occluded`, `both_visible`, `one_visible`, and `neither_visible` where
applicable.

- [ ] **Step 7: Implement the angle metric**

Use hips to define the pelvis transverse axis and shoulders to define the
thorax transverse axis after projecting both onto the plane normal to the
pelvis vertical axis. Fix sign from the right-handed coordinate construction
and use neutral/static-zero records for one evaluator-level zero offset. Report
MAE/RMSE against `actual_angle_deg`, plus the same metric for Unity GT-derived
angles to show construct validity.

- [ ] **Step 8: Implement report generation**

The Markdown report includes:

1. dataset and completion audit;
2. valid method ranking;
3. diagnostic-only table;
4. per-sequence results;
5. visible versus occluded results;
6. rotation-angle metrics;
7. failed samples/methods;
8. checkpoint and command provenance;
9. limitations and careful descriptive conclusions.

Generate compact MPJPE and visibility figures with matplotlib. Save JSON with
the complete machine-readable bundle.

- [ ] **Step 9: Run evaluation/report tests**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark/test_evaluation_report.py -q
```

Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/gymnastics/benchmarks/unity/evaluation.py \
  src/gymnastics/benchmarks/unity/report.py \
  tests/unity_benchmark/test_evaluation_report.py
git commit -m "feat: evaluate and report Unity benchmark"
```

---

### Task 7: Staged CLI and End-to-End Dry Run

**Files:**
- Create: `src/gymnastics/benchmarks/unity/cli.py`
- Create: `tests/unity_benchmark/test_cli.py`
- Modify: `src/gymnastics/cli.py`
- Modify: `tests/structure/test_cli.py`

**Interfaces:**
- Produces: `gymnastics benchmark unity inspect`
- Produces: `gymnastics benchmark unity infer`
- Produces: `gymnastics benchmark unity triangulate`
- Produces: `gymnastics benchmark unity fuse`
- Produces: `gymnastics benchmark unity evaluate`
- Produces: `gymnastics benchmark unity report`
- Produces: `gymnastics benchmark unity run`

- [ ] **Step 1: Write failing unified CLI tests**

Assert top-level help lists `benchmark`; nested help lists all seven Unity
stages; `inspect` on the real dataset reports 199 samples, 398 images, 22
joints, and three evaluation sequences.

- [ ] **Step 2: Run CLI tests and verify they fail**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark/test_cli.py \
  tests/structure/test_cli.py -q
```

Expected: `benchmark` is absent.

- [ ] **Step 3: Wire the unified dispatcher**

Add:

```python
"benchmark:unity": ("gymnastics.benchmarks.unity.cli", "main", True)
```

Add nested `benchmark` and `unity` parsers and return
`benchmark:unity` from `_target_key`.

- [ ] **Step 4: Implement stage parsers and orchestration**

Every stage accepts `--config` defaulting to
`configs/benchmarks/unity.yaml`. Inference accepts `--camera {cam0,cam1}`,
`--device`, and `--force`. Fuse accepts repeatable `--method` and
`--ablation`. `run` executes CPU inspect/oracle, requires completed SAM3D
caches, then executes triangulation, fusion, evaluation, and report; it does
not hide expensive two-GPU process management.

- [ ] **Step 5: Run all targeted tests**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark \
  tests/test_fuse_experiment_matrix.py \
  tests/test_sam3d_triangulation.py \
  tests/test_compare_fused_triangulated.py \
  tests/rotation_aware/test_inference.py \
  tests/rotation_aware/test_cli.py \
  tests/structure/test_cli.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Run a real-data CPU integrity and oracle dry run**

Run:

```bash
conda run -n gymnastic gymnastics benchmark unity inspect \
  --config configs/benchmarks/unity.yaml
conda run -n gymnastic gymnastics benchmark unity triangulate \
  --config configs/benchmarks/unity.yaml --oracle-only
```

Expected: inspect reports 199 samples and three sequences; oracle raw 3D maximum
error is below `0.1 mm` and reprojection maximum is below `0.01 px`.

- [ ] **Step 7: Commit**

```bash
git add src/gymnastics/benchmarks/unity/cli.py \
  src/gymnastics/cli.py \
  tests/unity_benchmark/test_cli.py \
  tests/structure/test_cli.py
git commit -m "feat: expose Unity benchmark CLI"
```

---

### Task 8: Full GPU Experiment and Evidence-Backed Conclusions

**Files:**
- Generate only: `local/runs/unity_benchmark/**`

**Interfaces:**
- Consumes: committed implementation, two available RTX 3090 GPUs, existing SAM3D weights, and A4--A9 checkpoints
- Produces: `local/runs/unity_benchmark/report/unity_benchmark_report.md`
- Produces: complete CSV, JSON, NPZ, and figures from the design contract

- [ ] **Step 1: Record the code and environment baseline**

Run:

```bash
git status --short
git rev-parse HEAD
conda run -n gymnastic python -V
conda run -n gymnastic python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
nvidia-smi
```

Expected: implementation worktree clean; Python/PyTorch versions and two GPUs
are recorded in run metadata.

- [ ] **Step 2: Run a four-image SAM3D smoke test**

Run each camera on two selected sample IDs:

```bash
conda run -n gymnastic gymnastics benchmark unity infer \
  --config configs/benchmarks/unity.yaml \
  --camera cam0 --device cuda:0 --sample-id 0 --sample-id 5
conda run -n gymnastic gymnastics benchmark unity infer \
  --config configs/benchmarks/unity.yaml \
  --camera cam1 --device cuda:1 --sample-id 0 --sample-id 5
```

Expected: four cache files with `[70,3]` and `[70,2]` finite prediction arrays,
or explicit detection failures.

- [ ] **Step 3: Run both full-view SAM3D jobs**

After presenting these exact commands for experiment confirmation, launch:

```bash
conda run -n gymnastic gymnastics benchmark unity infer \
  --config configs/benchmarks/unity.yaml \
  --camera cam0 --device cuda:0
conda run -n gymnastic gymnastics benchmark unity infer \
  --config configs/benchmarks/unity.yaml \
  --camera cam1 --device cuda:1
```

Run them concurrently in separate processes. Monitor logs and GPU utilisation.
Do not restart completed samples. Expected: 199 accounted samples per camera.

- [ ] **Step 4: Run all CPU/GPU-light downstream stages**

Run:

```bash
conda run -n gymnastic gymnastics benchmark unity triangulate \
  --config configs/benchmarks/unity.yaml
conda run -n gymnastic gymnastics benchmark unity fuse \
  --config configs/benchmarks/unity.yaml \
  --device cuda:0
conda run -n gymnastic gymnastics benchmark unity evaluate \
  --config configs/benchmarks/unity.yaml
conda run -n gymnastic gymnastics benchmark unity report \
  --config configs/benchmarks/unity.yaml
```

Expected: cam0, cam1, nine deterministic methods, A4--A9,
`triangulation_sam3d2d`, and `triangulation_oracle2d` are all accounted for.

- [ ] **Step 5: Verify artifacts and numerical invariants**

Run:

```bash
conda run -n gymnastic python -m pytest tests/unity_benchmark -q
conda run -n gymnastic gymnastics benchmark unity inspect \
  --config configs/benchmarks/unity.yaml --verify-results
```

Require:

- 199 expected samples per camera;
- all three evaluation sequences per successful method;
- no unreported missing frames;
- oracle raw error below the documented tolerance;
- valid ranking excludes both diagnostics;
- metrics contain no unexplained NaN;
- report and JSON agree on the top-ranked valid method.

- [ ] **Step 6: Read and cross-check the report**

Use `metrics_summary.csv`, `metrics_by_sequence.csv`,
`metrics_by_visibility.csv`, and `results.json` to verify every numerical claim
in `unity_benchmark_report.md`. State:

- the best valid method and MPJPE;
- comparison with cam0 and cam1;
- comparison with SAM3D-2D triangulation;
- whether A4--A9 transfer improves pose or angle metrics;
- visible versus occluded degradation;
- oracle geometry quality;
- all failure counts and dataset limitations.

- [ ] **Step 7: Final implementation verification**

Run:

```bash
git status --short
git log --oneline --decorate -10
conda run -n gymnastic python -m pytest \
  tests/unity_benchmark \
  tests/test_fuse_experiment_matrix.py \
  tests/test_sam3d_triangulation.py \
  tests/test_compare_fused_triangulated.py \
  tests/rotation_aware/test_inference.py \
  tests/rotation_aware/test_cli.py \
  tests/structure/test_cli.py -q
```

Expected: implementation worktree clean and all targeted tests pass. Report
generated artifacts remain ignored under `local/runs/unity_benchmark`.
