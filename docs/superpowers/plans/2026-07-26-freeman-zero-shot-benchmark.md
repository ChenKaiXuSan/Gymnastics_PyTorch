# FreeMan Zero-Shot Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable full-release FreeMan benchmark that selects one near-orthogonal synchronized camera pair per session, runs SAM3D-Body on both views, evaluates existing deterministic and rotation-aware fusion methods zero-shot against FreeMan's markerless 3D reference, and produces subject-balanced reports.

**Architecture:** Add a focused `gymnastics.benchmarks.freeman` package split into download/preflight, immutable dataset contracts, camera pairing, streaming SAM3D inference, fusion adapters, evaluation, reporting, and staged CLI orchestration. Large data stays below ignored `local/`; extraction is limited to one subject at a time; valid inference and fusion APIs never receive FreeMan reference 3D arrays.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, SciPy, pandas, PyYAML/OmegaConf, PyTorch, Hugging Face `hf` CLI and `huggingface_hub`, pytest, existing SAM3D-Body and rotation-aware fusion modules.

## Global Constraints

- Run project code, tests, scripts, and Python tooling with `conda run -n gymnastic ...`.
- Use the gated Hugging Face dataset repository `wjwow/FreeMan`.
- Download and preserve the complete release, including subject archives, shared annotation archives, metadata, and split archive volumes.
- Process every valid session from all 40 subjects in both `30FPS` and `60FPS` subsets when present.
- Select exactly two synchronized views per session from camera geometry, targeting a 90-degree horizontal optical-axis separation.
- Use frame stride `1` for headline results.
- Do not train, fine-tune, select checkpoints, select views, align time, or tune valid methods with FreeMan reference 3D.
- Treat FreeMan 3D as a public markerless multi-view reference, not independent marker-based motion-capture ground truth.
- Keep archives below `local/datasets/freeman` and generated results below `local/runs/freeman_benchmark`; both paths are Git-ignored.
- Never require more than one subject's extracted videos to coexist.
- Never delete downloaded archives automatically.
- Reject unknown camera or joint schemas instead of guessing.
- Use one Sim3 fit per complete session for the primary metric; do not use per-frame fitting as the primary metric.
- Aggregate headline comparisons by subject so long sessions do not dominate.
- Run leakage-prone deterministic methods only as labelled diagnostics and exclude them from valid rankings.

---

## File Structure

Create:

- `configs/benchmarks/freeman.yaml` — repository, storage, subject/FPS, camera-pair, SAM3D, checkpoint, and metric defaults.
- `src/gymnastics/benchmarks/__init__.py` — benchmark namespace.
- `src/gymnastics/benchmarks/freeman/__init__.py` — public FreeMan benchmark API.
- `src/gymnastics/benchmarks/freeman/schema.py` — immutable archive, camera, session, pair, prediction, and preflight contracts.
- `src/gymnastics/benchmarks/freeman/download.py` — Hub inventory, authentication/disk checks, resumable download, checksum verification, multipart extraction, and safe cleanup.
- `src/gymnastics/benchmarks/freeman/dataset.py` — official directory/session loader and annotation/video validation.
- `src/gymnastics/benchmarks/freeman/pairing.py` — camera conversion and deterministic near-orthogonal pair selection.
- `src/gymnastics/benchmarks/freeman/mapping.py` — versioned COCO17-to-MHR70 joint correspondence.
- `src/gymnastics/benchmarks/freeman/sam3d.py` — streaming video inference and atomic cache management.
- `src/gymnastics/benchmarks/freeman/fusion.py` — deterministic and zero-shot rotation-aware adapters.
- `src/gymnastics/benchmarks/freeman/evaluation.py` — sequence-level alignment, spatial/temporal metrics, aggregation, and statistics.
- `src/gymnastics/benchmarks/freeman/report.py` — CSV/JSON/Markdown report generation.
- `src/gymnastics/benchmarks/freeman/cli.py` — staged commands and one-subject extraction lifecycle.
- `tests/freeman_benchmark/__init__.py` — test package marker.
- `tests/freeman_benchmark/conftest.py` — compact official-layout fixture with two FPS subsets and eight cameras.
- `tests/freeman_benchmark/test_download.py` — Hub manifest, disk reserve, split archives, extraction, and cleanup tests.
- `tests/freeman_benchmark/test_dataset_pairing.py` — session contract, camera parsing, pairing, and frame consistency tests.
- `tests/freeman_benchmark/test_mapping.py` — exact 17-joint correspondence and validity tests.
- `tests/freeman_benchmark/test_sam3d.py` — fake-estimator streaming/cache/resume tests.
- `tests/freeman_benchmark/test_fusion.py` — deterministic and rotation-aware isolation tests.
- `tests/freeman_benchmark/test_evaluation_report.py` — metrics, aggregation, statistics, and report wording tests.
- `tests/freeman_benchmark/test_cli.py` — staged CLI, orchestration, and dry-run tests.

Modify:

- `src/gymnastics/cli.py` — expose `gymnastics benchmark freeman`.
- `tests/structure/test_cli.py` — require the new `benchmark` command.
- `README.md` — document inspection and staged execution without adding local data paths to Git.

Do not modify deterministic or rotation-aware algorithms. Adapters must call
their existing pure functions and `run_inference`.

---

### Task 1: Configuration, Immutable Contracts, and Hub Preflight

**Files:**
- Create: `configs/benchmarks/freeman.yaml`
- Create: `src/gymnastics/benchmarks/__init__.py`
- Create: `src/gymnastics/benchmarks/freeman/__init__.py`
- Create: `src/gymnastics/benchmarks/freeman/schema.py`
- Create: `src/gymnastics/benchmarks/freeman/download.py`
- Create: `tests/freeman_benchmark/__init__.py`
- Create: `tests/freeman_benchmark/test_download.py`

**Interfaces:**
- Produces: `load_config(path: Path) -> dict[str, Any]`
- Produces: `fetch_hub_inventory(api: Any, repo_id: str, revision: str) -> tuple[ArchiveEntry, ...]`
- Produces: `run_preflight(config: Mapping[str, Any], *, runner: Runner = subprocess.run, api: Any | None = None) -> PreflightReport`
- Produces: `download_release(config: Mapping[str, Any], report: PreflightReport, *, runner: Runner = subprocess.run) -> Path`
- Produces: `validate_downloads(entries: Sequence[ArchiveEntry], archive_root: Path) -> tuple[Path, ...]`

Define the injected command-runner type once in `download.py`:

```python
Runner = Callable[..., subprocess.CompletedProcess[str]]
```

- [ ] **Step 1: Write failing immutable-contract and inventory tests**

Use fake Hugging Face siblings with `rfilename`, `size`, and
`lfs.sha256`. Assert deterministic ordering, read-only tuples, split-volume
retention, and subject coverage:

```python
def test_inventory_preserves_all_release_files(fake_hf_api) -> None:
    entries = fetch_hub_inventory(fake_hf_api, "wjwow/FreeMan", "main")
    assert tuple(item.path for item in entries) == tuple(
        sorted(item.path for item in entries)
    )
    assert {item.path for item in entries} >= {
        "cameras.zip",
        "keypoints2d.zip",
        "keypoints3d.zip",
        "motions.zip",
        "subj01.zip",
        "subj40.zip",
        "subj02.01",
    }
    assert all(item.size > 0 for item in entries)
```

Define frozen contracts:

```python
@dataclass(frozen=True)
class ArchiveEntry:
    path: str
    size: int
    sha256: str | None

@dataclass(frozen=True)
class PreflightReport:
    repo_id: str
    revision: str
    hf_executable: Path
    authenticated_user: str
    access_granted: bool
    archive_root: Path
    required_bytes: int
    free_bytes: int
    reserve_bytes: int
    entries: tuple[ArchiveEntry, ...]
```

- [ ] **Step 2: Write failing preflight tests**

Inject a fake command runner and fake disk usage. Cover:

```python
def test_preflight_rejects_missing_hf(config) -> None:
    with pytest.raises(RuntimeError, match="hf executable"):
        run_preflight(config, runner=missing_hf_runner)

def test_preflight_rejects_unapproved_gate(config, fake_hf_api) -> None:
    fake_hf_api.raise_for_dataset = GatedRepoError("pending")
    with pytest.raises(RuntimeError, match="gated access"):
        run_preflight(config, runner=authenticated_runner, api=fake_hf_api)

def test_preflight_enforces_remaining_bytes_plus_reserve(config, fake_hf_api) -> None:
    with pytest.raises(RuntimeError, match="free space"):
        run_preflight(
            config,
            runner=authenticated_runner,
            api=fake_hf_api,
            disk_usage=lambda _: usage(free=99, required=100),
        )
```

Also assert existing valid files are subtracted from `required_bytes`, while a
same-sized file with a bad checksum remains required.

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_download.py -q
```

Expected: collection fails because `gymnastics.benchmarks.freeman` does not
exist.

- [ ] **Step 4: Implement the resolved YAML configuration**

Use these committed defaults:

```yaml
repository:
  repo_id: wjwow/FreeMan
  revision: main
paths:
  archive_root: local/datasets/freeman/archives
  manifest_root: local/datasets/freeman/manifests
  work_root: local/datasets/freeman/work
  output_root: local/runs/freeman_benchmark
download:
  reserve_bytes: 107374182400
  verify_sha256: true
dataset:
  fps_subsets: [30, 60]
  subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
  frame_stride: 1
pairing:
  target_angle_deg: 90.0
  world_up_axis: [0.0, 0.0, 1.0]
  minimum_axis_norm: 1.0e-8
sam3d:
  config: configs/sam3d/sam3d_body.yaml
  device: 0
  checkpoint_id: sam3d-body
rotation_aware:
  config: configs/fusion/rotation_aware.yaml
  run_ids: [paper_a4, paper_a5, paper_a6]
evaluation:
  pck_thresholds_mm: [50, 100, 150]
  auc_max_threshold_mm: 150
  minimum_subject_coverage: 0.95
  bootstrap_samples: 10000
  random_seed: 20260726
```

`load_config` resolves project-relative paths against `PROJECT_ROOT`, rejects
an FPS outside `{30, 60}`, rejects subjects outside `1..40`, and requires
`frame_stride >= 1`.

- [ ] **Step 5: Implement authenticated inventory and storage preflight**

`fetch_hub_inventory` calls
`api.dataset_info(repo_id, revision=revision, files_metadata=True)`, converts
every non-directory sibling to `ArchiveEntry`, and verifies that `subj01.zip`
through `subj40.zip` are represented. A subject may also have published
numeric volumes such as `.01`, `.02`, and `.03`; these remain separate required
entries.

`run_preflight`:

1. resolves `hf` with `shutil.which`;
2. runs `hf auth whoami`;
3. obtains the gated dataset inventory;
4. checks each completed local blob by size and optional SHA256;
5. computes remaining bytes;
6. uses `shutil.disk_usage(archive_root.parent)` and requires
   `free_bytes >= required_bytes + reserve_bytes`;
7. writes no files before all checks pass.

- [ ] **Step 6: Implement resumable complete-release download**

Invoke the current CLI as:

```python
command = [
    str(report.hf_executable), "download", report.repo_id,
    "--repo-type", "dataset",
    "--revision", report.revision,
    "--local-dir", str(report.archive_root),
]
```

After the command succeeds, validate every inventory entry. Write
`remote_inventory.json` and `download_state.json` atomically using
`path.with_suffix(path.suffix + ".tmp")` followed by `Path.replace`.
`download_state.json` records repository revision, authenticated user,
timestamp, byte counts, filenames, sizes, and observed SHA256 values.

- [ ] **Step 7: Run tests and commit**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_download.py -q
```

Expected: all tests pass.

Commit:

```bash
git add configs/benchmarks/freeman.yaml \
  src/gymnastics/benchmarks/__init__.py \
  src/gymnastics/benchmarks/freeman/__init__.py \
  src/gymnastics/benchmarks/freeman/schema.py \
  src/gymnastics/benchmarks/freeman/download.py \
  tests/freeman_benchmark/__init__.py \
  tests/freeman_benchmark/test_download.py
git commit -m "feat: add FreeMan download preflight"
```

---

### Task 2: Multipart Extraction and Safe Subject Lifecycle

**Files:**
- Modify: `src/gymnastics/benchmarks/freeman/download.py`
- Modify: `tests/freeman_benchmark/test_download.py`

**Interfaces:**
- Consumes: `ArchiveEntry`, resolved `paths.archive_root` and `paths.work_root`
- Produces: `subject_archive_set(subject_id: int, entries: Sequence[ArchiveEntry]) -> tuple[ArchiveEntry, ...]`
- Produces: `extract_shared_annotations(entries: Sequence[ArchiveEntry], archive_root: Path, work_root: Path, *, runner: Runner = subprocess.run) -> Path`
- Produces: `extract_subject(subject_id: int, archive_root: Path, work_root: Path, *, runner: Runner = subprocess.run) -> Path`
- Produces: `cleanup_subject_workspace(subject_id: int, subject_root: Path, work_root: Path) -> None`

- [ ] **Step 1: Write failing split-volume and traversal tests**

```python
def test_subject_archive_set_orders_split_parts_before_zip(entries) -> None:
    selected = subject_archive_set(2, entries)
    assert [item.path for item in selected] == [
        "subj02.01", "subj02.02", "subj02.03", "subj02.zip"
    ]

def test_extract_rejects_archive_path_traversal(tmp_path, fake_7z_runner) -> None:
    fake_7z_runner.listing = ["Path = ../../escape.mp4"]
    with pytest.raises(RuntimeError, match="unsafe archive member"):
        extract_subject(1, tmp_path / "archives", tmp_path / "work",
                        runner=fake_7z_runner)
```

Add tests proving cleanup rejects `/`, the work root itself, another subject,
and a symlink escaping the work root.

Add a shared-annotation test proving only `cameras.zip`, `keypoints2d.zip`,
`keypoints3d.zip`, and official root text lists are materialized below
`work/shared`; `motions.zip` and `bbox2d.zip` remain preserved archives but are
not extracted because the benchmark does not consume them.

- [ ] **Step 2: Run the focused failure**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_download.py -q
```

Expected: new extraction tests fail because the functions are absent.

- [ ] **Step 3: Implement archive-format detection and extraction**

Require `7z` from `shutil.which`. Test the published `.zip` directly with
`7z t`. When numeric pieces exist and the direct test fails, inspect every
piece before reconstruction:

- exactly one first piece must begin with a ZIP local-file header
  (`PK\x03\x04`);
- exactly one final piece must contain the ZIP end-of-central-directory
  signature (`PK\x05\x06`) in its tail;
- numbered middle pieces must be contiguous from `.01`;
- order is the detected first piece, numeric middle pieces, and detected final
  piece, never an assumed extension order.

Stream those pieces into
`work_root / f"subject_{subject_id:02d}.reconstructed.zip"` with
`shutil.copyfileobj`; do not load a volume into memory. Validate the
reconstructed archive with `7z t` before listing or extracting it.

Run `7z l -slt <validated-archive>` next, parse each `Path =` member, and
reject:

- absolute members;
- members containing `..`;
- members whose resolved destination is not below
  `work_root / f"subject_{subject_id:02d}"`.

Extract with:

```python
[
    seven_zip, "x", str(archive_root / f"subj{subject_id:02d}.zip"),
    f"-o{subject_root}", "-y",
]
```

Extract the validated archive into
`subject_root.with_name(subject_root.name + ".partial")`, validate that at
least one `30FPS/videos/*/vframes/c01.mp4` or
`60FPS/videos/*/vframes/c01.mp4` exists, then atomically rename the directory.
Delete only the validated reconstructed ZIP after successful extraction; the
published `.01`/`.02`/`.03`/`.zip` files remain untouched.

`extract_shared_annotations` applies the same member-path checks, publishes
through `work/shared.partial`, and requires `session_list.txt`, a validation
split file, camera JSON, 2D annotations, and 3D annotations before renaming to
`work/shared`. Its manifest hashes every consumed archive and allows an
identical valid shared tree to be reused.

- [ ] **Step 4: Implement scoped cleanup**

`cleanup_subject_workspace` resolves both paths and requires:

```python
subject_root.parent == work_root.resolve()
subject_root.name == f"subject_{subject_id:02d}"
not subject_root.is_symlink()
```

Only then call `shutil.rmtree(subject_root)`. The function never touches the
archive root and returns without error when the exact subject directory does
not exist.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_download.py -q
```

Expected: all tests pass.

Commit:

```bash
git add src/gymnastics/benchmarks/freeman/download.py \
  tests/freeman_benchmark/test_download.py
git commit -m "feat: add safe FreeMan subject extraction"
```

---

### Task 3: Official Dataset Contract, COCO17 Mapping, and Camera Pairing

**Files:**
- Create: `src/gymnastics/benchmarks/freeman/dataset.py`
- Create: `src/gymnastics/benchmarks/freeman/pairing.py`
- Create: `src/gymnastics/benchmarks/freeman/mapping.py`
- Create: `tests/freeman_benchmark/conftest.py`
- Create: `tests/freeman_benchmark/test_dataset_pairing.py`
- Create: `tests/freeman_benchmark/test_mapping.py`

**Interfaces:**
- Consumes: official `30FPS`/`60FPS` layout and `FreeManCamera` from `schema.py`
- Produces: `load_subject_sessions(subject_root: Path, shared_root: Path, fps_values: Sequence[int]) -> tuple[FreeManSession, ...]`
- Produces: `select_camera_pair(session: FreeManSession, *, target_angle_deg: float, world_up: np.ndarray) -> SelectedPair`
- Produces: `load_session_reference(session: FreeManSession) -> ReferenceSequence`
- Produces: `map_mhr70_to_freeman(points: np.ndarray, valid: np.ndarray | None = None) -> MappedPose`
- Produces: constants `FREEMAN_COCO17_NAMES` and `MAPPING_VERSION`

- [ ] **Step 1: Build a compact official-layout fixture**

The fixture contains:

```text
fixture/
├── shared/
│   ├── 30FPS/
│   │   ├── session_list.txt
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   ├── test.txt
│   │   ├── cameras/<session>.json
│   │   ├── keypoints2d/<session>.npy
│   │   └── keypoints3d/<session>.npy
│   └── 60FPS/...
└── subject_01/
    ├── 30FPS/videos/<session>/vframes/c01.mp4 ... c08.mp4
    └── 60FPS/videos/<session>/vframes/c01.mp4 ... c08.mp4
```

Camera JSON elements use the official keys:

```python
{
    "name": "c01",
    "size": [1920, 1080],
    "matrix": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
    "rotation": [0.0, 0.0, 0.0],
    "translation": [0.0, 0.0, 0.0],
    "distortions": [0.0, 0.0, 0.0, 0.0, 0.0],
}
```

Write tiny MP4 files with OpenCV so frame-count and FPS checks exercise the
real loader.

- [ ] **Step 2: Write failing dataset and frame-contract tests**

```python
def test_loads_both_fps_subsets_and_exact_subject(freeman_fixture) -> None:
    sessions = load_subject_sessions(
        freeman_fixture.subject_root,
        freeman_fixture.shared_root,
        fps_values=(30, 60),
    )
    assert {(item.fps, item.subject_id) for item in sessions} == {(30, 1), (60, 1)}
    assert all(tuple(item.video_paths) == tuple(f"c{i:02d}" for i in range(1, 9))
               for item in sessions)

def test_rejects_internal_frame_gap(freeman_fixture) -> None:
    freeman_fixture.remove_video_frame("c03", frame=2)
    with pytest.raises(ValueError, match="frame correspondence"):
        load_subject_sessions(
            freeman_fixture.subject_root,
            freeman_fixture.shared_root,
            fps_values=(30, 60),
        )
```

Also reject duplicate session IDs, a session whose `_subjNN` suffix disagrees
with the extracted subject, fewer than eight camera records, a non-3x3
intrinsic matrix, non-finite rotations, 2D shape other than `[8,F,17,3]`,
3D shape other than `[F,17,3]`, and inconsistent annotation/video frames.
Trailing frames may be truncated only when the exclusion count and reason are
stored in the session contract.

- [ ] **Step 3: Write failing camera-pair tests**

Construct eight cameras around the origin. Assert:

```python
def test_selects_pair_closest_to_ninety_degrees(session) -> None:
    pair = select_camera_pair(
        session, target_angle_deg=90.0, world_up=np.array([0.0, 0.0, 1.0])
    )
    assert (pair.view_a, pair.view_b) == ("c01", "c03")
    assert pair.separation_deg == pytest.approx(90.0)
    assert pair.reference_view == "c01"
```

Add ties proving smaller camera-height difference wins before lexical camera
IDs. Add degenerate optical-axis and zero world-up rejection.

- [ ] **Step 4: Write failing exact mapping tests**

Use official COCO17 order:

```python
FREEMAN_COCO17_NAMES = (
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-wrist", "right-wrist", "left-hip", "right-hip",
    "left-knee", "right-knee", "left-ankle", "right-ankle",
)
```

Every target has an exact MHR70 name. Encode MHR indices in a `[2,70,3]` test
array and assert all 17 outputs, especially MHR `left-wrist` index 62 and
`right-wrist` index 41. Invalid source joints must invalidate only their exact
mapped target.

- [ ] **Step 5: Run tests and verify failure**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_dataset_pairing.py \
  tests/freeman_benchmark/test_mapping.py -q
```

Expected: collection fails because the dataset, pairing, and mapping modules do
not exist.

- [ ] **Step 6: Implement immutable dataset contracts**

Add to `schema.py`:

```python
@dataclass(frozen=True)
class FreeManCamera:
    name: str
    size: tuple[int, int]
    matrix: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    distortions: np.ndarray

@dataclass(frozen=True)
class FreeManSession:
    session_id: str
    subject_id: int
    fps: int
    split: str
    scenario: str | None
    action: str | None
    video_paths: Mapping[str, Path]
    cameras: Mapping[str, FreeManCamera]
    keypoints2d_path: Path
    keypoints3d_path: Path
    frame_ids: np.ndarray
    excluded_trailing_frames: Mapping[str, int]

@dataclass(frozen=True)
class SelectedPair:
    session_id: str
    view_a: str
    view_b: str
    reference_view: str
    separation_deg: float
    target_error_deg: float
    height_difference: float

@dataclass(frozen=True)
class ReferenceSequence:
    session_id: str
    points_m: np.ndarray
    valid: np.ndarray
    frame_ids: np.ndarray
    joint_names: tuple[str, ...]

@dataclass(frozen=True)
class MappedPose:
    points: np.ndarray
    valid: np.ndarray
    joint_names: tuple[str, ...]
```

Copy NumPy arrays, mark them read-only, and wrap mappings with
`MappingProxyType`.

- [ ] **Step 7: Implement the official loader**

Read `session_list.txt` and the three official split files. Accept exactly one
validation filename from the two official-code/release variants `valid.txt`
and `validation.txt`; reject a directory containing both or neither. Parse
subject IDs with `r"_subj(\d+)$"`. Load the object-array annotations using:

```python
k2 = np.load(path, allow_pickle=True)[0]["keypoints2d"]
k3 = np.load(path, allow_pickle=True)[0]["keypoints3d_optim"]
```

Populate `scenario` and `action` only from an official release metadata mapping
when one exists. Do not infer semantic labels from opaque session-name
substrings; use `None` when the repository supplies no mapping.

For 3D, accept `keypoints3d_optim` as the committed reference field. Record
available `keypoints3d_smoothnet32`, `keypoints3d_smoothnet`, and raw
`keypoints3d` fields in inspection metadata but never silently substitute them.
Validity is finite XYZ; values remain in the documented FreeMan metric unit and
are normalized to metres only when inspection confirms millimetres.

`load_session_reference` reads only `keypoints3d_optim`, joins the exact
`session.frame_ids`, attaches `FREEMAN_COCO17_NAMES`, and returns the immutable
`ReferenceSequence`. It is used only by evaluation and is not imported by
pairing, SAM3D inference, or fusion modules.

Open every selected video with OpenCV and validate `CAP_PROP_FPS`,
`CAP_PROP_FRAME_COUNT`, and sequential decodability for the synthetic fixture.
The production inspection path samples decoding at first, middle, and last
frames; inference detects internal failures while streaming.

- [ ] **Step 8: Implement geometry-only camera selection**

Convert OpenCV Rodrigues rotation vector `rvec` to world-to-camera `R`.
Compute:

```python
camera_center_world = -R.T @ tvec
optical_axis_world = R.T @ np.array([0.0, 0.0, 1.0])
horizontal = axis - dot(axis, up_hat) * up_hat
```

Use unsigned separation
`degrees(arccos(clip(abs(dot(a_hat, b_hat)), 0, 1)))`. Rank by:

```python
(
    abs(separation_deg - target_angle_deg),
    abs(dot(center_a - center_b, up_hat)),
    min(view_a, view_b),
    max(view_a, view_b),
)
```

The selected pair contains lexical `view_a < view_b`, and the reference view is
always `view_a`.

- [ ] **Step 9: Implement and test COCO17 mapping**

Build the table from names, not raw literal indices:

```python
MHR_INDICES = tuple(MHR70_INDEX[name] for name in FREEMAN_COCO17_NAMES)
```

`map_mhr70_to_freeman` accepts `[T,70,3]`, validates optional `[T,70]`
validity, returns `[T,17,3]`, and zeroes invalid outputs while retaining an
explicit boolean mask.

- [ ] **Step 10: Run tests and commit**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_dataset_pairing.py \
  tests/freeman_benchmark/test_mapping.py -q
```

Expected: all tests pass.

Commit:

```bash
git add src/gymnastics/benchmarks/freeman/schema.py \
  src/gymnastics/benchmarks/freeman/dataset.py \
  src/gymnastics/benchmarks/freeman/pairing.py \
  src/gymnastics/benchmarks/freeman/mapping.py \
  tests/freeman_benchmark/conftest.py \
  tests/freeman_benchmark/test_dataset_pairing.py \
  tests/freeman_benchmark/test_mapping.py
git commit -m "feat: add FreeMan dataset and camera pairing"
```

---

### Task 4: Streaming SAM3D Inference and Atomic Resume Cache

**Files:**
- Create: `src/gymnastics/benchmarks/freeman/sam3d.py`
- Create: `tests/freeman_benchmark/test_sam3d.py`

**Interfaces:**
- Consumes: `FreeManSession`, `SelectedPair`, existing
  `gymnastics.sam3d.infer.setup_sam_3d_body` and `select_best_person`
- Produces: `infer_subject_sessions(sessions: Sequence[FreeManSession], pairs: Mapping[str, SelectedPair], config: Mapping[str, Any], *, estimator_factory: Callable | None = None) -> tuple[InferenceArtifact, ...]`
- Produces: `load_inference(path: Path) -> ViewPrediction`
- Produces: `validate_inference(path: Path, expected: InferenceIdentity) -> bool`

- [ ] **Step 1: Write failing streaming and resume tests**

Use a fake estimator that records calls and returns deterministic
`pred_keypoints_2d`, `pred_keypoints_3d`, `pred_cam_t`, and `bbox`.

```python
def test_streams_both_selected_views_with_one_estimator(fake_session, fake_estimator) -> None:
    artifacts = infer_subject_sessions(
        [fake_session], {fake_session.session_id: pair("c01", "c03")},
        config, estimator_factory=lambda _: fake_estimator,
    )
    assert fake_estimator.load_count == 1
    assert fake_estimator.frames == [
        ("c01", 0), ("c01", 1), ("c01", 2),
        ("c03", 0), ("c03", 1), ("c03", 2),
    ]
    assert {item.view_id for item in artifacts} == {"c01", "c03"}
```

Add tests that:

- failed detections create invalid frame rows instead of shortening arrays;
- an identical valid artifact is reused with zero estimator calls;
- a source checksum, pair identity, frame stride, or checkpoint change forces
  recomputation;
- corrupt `.npz` files are replaced;
- `.partial` output is never treated as complete.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_sam3d.py -q
```

Expected: collection fails because `freeman.sam3d` does not exist.

- [ ] **Step 3: Implement compact prediction contracts**

Add:

```python
@dataclass(frozen=True)
class ViewPrediction:
    session_id: str
    subject_id: int
    fps: float
    view_id: str
    frame_ids: np.ndarray
    points3d: np.ndarray
    points2d: np.ndarray
    valid3d: np.ndarray
    valid2d: np.ndarray
    metadata: Mapping[str, Any]

@dataclass(frozen=True)
class InferenceArtifact:
    path: Path
    session_id: str
    view_id: str
    frames: int
    valid_frames: int

@dataclass(frozen=True)
class InferenceIdentity:
    session_id: str
    subject_id: int
    fps: int
    view_id: str
    source_video_sha256: str
    source_frame_count: int
    frame_stride: int
    sam3d_config_sha256: str
    checkpoint_id: str
```

Shapes are `[T,70,3]`, `[T,70,2]`, and masks `[T,70]`.

- [ ] **Step 4: Implement one-estimator streaming inference**

Load SAM3D configuration with existing `gymnastics.common.config.load_config`.
Build the estimator once per subject process. For each selected view:

1. open `cv2.VideoCapture`;
2. decode sequentially;
3. skip frames excluded by explicit diagnostic stride;
4. call `estimator.process_one_image(img=frame, bboxes=None)`;
5. select the largest bounding box with `select_best_person(outputs, verbose=False)`;
6. write predicted fields into preallocated chunk lists;
7. append zero points and false masks for failed detections;
8. reject an internal video decode failure before the declared valid end.

Do not retain video frames or mesh outputs. Validity requires finite,
non-zero XYZ for 3D and finite XY for 2D.

- [ ] **Step 5: Implement cache identity and atomic publication**

The identity JSON includes:

```python
{
    "session_id": session.session_id,
    "subject_id": session.subject_id,
    "fps": session.fps,
    "view_id": view_id,
    "source_video_sha256": sha256(video_path),
    "source_frame_count": len(session.frame_ids),
    "frame_stride": config["dataset"]["frame_stride"],
    "sam3d_config_sha256": sha256(config_path),
    "checkpoint_id": config["sam3d"]["checkpoint_id"],
}
```

Save to:

```text
local/runs/freeman_benchmark/sam3d/
  subject_01/<session_id>/c01/prediction.npz
  subject_01/<session_id>/c01/metadata.json
```

Write both files under a sibling `.partial` directory, load and validate them,
then rename to the final view directory. `validate_inference` checks identity,
required arrays, exact shapes, monotonic frame IDs, finiteness under valid
masks, and metadata agreement.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_sam3d.py -q
```

Expected: all tests pass.

Commit:

```bash
git add src/gymnastics/benchmarks/freeman/sam3d.py \
  src/gymnastics/benchmarks/freeman/schema.py \
  tests/freeman_benchmark/test_sam3d.py
git commit -m "feat: add resumable FreeMan SAM3D inference"
```

---

### Task 5: Leakage-Isolated Deterministic Fusion Adapter

**Files:**
- Create: `src/gymnastics/benchmarks/freeman/fusion.py`
- Create: `tests/freeman_benchmark/test_fusion.py`

**Interfaces:**
- Consumes: two `ViewPrediction` objects and existing deterministic helpers
- Produces: `fuse_deterministic(pair: PosePairInput, methods: Sequence[str] = ALL_METHODS) -> tuple[MethodPrediction, ...]`
- Produces: `save_method_prediction(prediction: MethodPrediction, output_root: Path) -> Path`
- Produces: constant `METHOD_CLASSIFICATION: Mapping[str, str]`

- [ ] **Step 1: Write failing deterministic method-matrix tests**

```python
def test_runs_all_nine_registered_methods_without_reference_3d(pose_pair) -> None:
    outputs = fuse_deterministic(pose_pair)
    assert tuple(item.method for item in outputs) == ALL_METHODS
    assert all(item.points.shape == pose_pair.view_a.points3d.shape for item in outputs)
    assert METHOD_CLASSIFICATION["sim3_face_stable_joint_weight"] == "GT_LEAKY_DIAGNOSTIC"
```

`PosePairInput` intentionally has no reference field:

```python
@dataclass(frozen=True)
class PosePairInput:
    session_id: str
    subject_id: int
    fps: float
    view_a: ViewPrediction
    view_b: ViewPrediction
```

Assert mismatched frame IDs fail instead of invoking DTW.

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_fusion.py::test_runs_all_nine_registered_methods_without_reference_3d -q
```

Expected: failure because `fuse_deterministic` is absent.

- [ ] **Step 3: Implement the leakage-free methods with existing helpers**

Use the existing functions from
`gymnastics.fusion.deterministic.experiment_matrix`:

- `current_body_average`;
- `root_align_to_reference`;
- `sim3_align_to_reference`;
- `fuse_weighted`;
- `bodypart_weights`;
- `smooth_sequence`;
- `STABLE_SIM3_JOINTS`.

Keep `view_a` as the reference. Implement the eight valid methods exactly as
the existing experiment matrix.

For `sim3_face_stable_joint_weight`, do not accept reference 3D. Emit the equal
0.5 fallback already used when triangulated reference is unavailable, set
metadata:

```python
{
    "classification": "GT_LEAKY_DIAGNOSTIC",
    "joint_weight_source": "unavailable_external_reference_equal_fallback",
    "excluded_from_ranking": True,
}
```

This preserves the nine-method output matrix without leaking FreeMan reference
3D or presenting the fallback as the original oracle method.

- [ ] **Step 4: Implement method prediction cache**

```python
@dataclass(frozen=True)
class MethodPrediction:
    method: str
    session_id: str
    subject_id: int
    points: np.ndarray
    valid: np.ndarray
    frame_ids: np.ndarray
    metadata: Mapping[str, Any]
```

Write one atomic compressed NPZ per method under:

```text
local/runs/freeman_benchmark/fusion/deterministic/
  <method>/subject_01/<session_id>/fused_sequence.npz
```

Cache identity includes both view inference metadata hashes and selected-pair
metadata.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_fusion.py -q
```

Expected: deterministic tests pass; rotation-aware tests added next remain
absent.

Commit:

```bash
git add src/gymnastics/benchmarks/freeman/fusion.py \
  src/gymnastics/benchmarks/freeman/schema.py \
  tests/freeman_benchmark/test_fusion.py
git commit -m "feat: adapt deterministic fusion to FreeMan"
```

---

### Task 6: Zero-Shot Rotation-Aware Checkpoint Adapter

**Files:**
- Modify: `src/gymnastics/benchmarks/freeman/fusion.py`
- Modify: `tests/freeman_benchmark/test_fusion.py`

**Interfaces:**
- Consumes: `PosePairInput`, existing `PosePairTrial`,
  `load_skeleton_spec`, `RotationAwareFusionModel`, `load_checkpoint`, and
  `run_inference`
- Produces: `build_rotation_aware_trial(pair: PosePairInput) -> PosePairTrial`
- Produces: `fuse_rotation_aware(pair: PosePairInput, checkpoint: Path, run_id: str, config: Mapping[str, Any]) -> MethodPrediction`

- [ ] **Step 1: Write failing trial-contract and checkpoint tests**

```python
def test_builds_exact_zero_offset_mhr70_trial(pose_pair) -> None:
    trial = build_rotation_aware_trial(pose_pair)
    np.testing.assert_array_equal(trial.face_map, pose_pair.view_a.frame_ids)
    np.testing.assert_array_equal(trial.side_map, pose_pair.view_b.frame_ids)
    np.testing.assert_allclose(
        trial.timestamps,
        pose_pair.view_a.frame_ids / pose_pair.fps,
    )
    assert trial.source_metadata["temporal_alignment"] == "native_zero_offset"
```

Mock the model and `run_inference`; assert FreeMan reference arrays cannot be
passed through the call graph. Reject a checkpoint with a non-MHR70 skeleton
hash or training provenance indicating FreeMan data.

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_fusion.py -q
```

Expected: rotation-aware tests fail because the adapter is absent.

- [ ] **Step 3: Implement exact synchronized trial construction**

Map `view_a` to the existing `PosePairTrial.face` fields and `view_b` to
`.side`. Use original frame IDs for both maps and timestamps `frame_id / fps`.
Require exact equality of frame IDs and joint order `MHR70_NAMES`. Store:

```python
{
    "dataset": "FreeMan",
    "session_id": pair.session_id,
    "subject_id": pair.subject_id,
    "temporal_alignment": "native_zero_offset",
    "reference_view": pair.view_a.view_id,
    "zero_shot": True,
    "reference_3d_consumed": False,
}
```

- [ ] **Step 4: Implement checkpoint loading and inference**

Resolve the rotation-aware skeleton from
`configs/fusion/skeleton_mhr70.yaml`. Reuse checkpoint construction logic from
`gymnastics.fusion.rotation_aware.cli` through existing public
`load_checkpoint` and model configuration fields. Call:

```python
result = run_inference(
    model,
    trial,
    skeleton,
    output_root=temporary_output_root,
    run_id=run_id,
    window_length=config["inference"]["window_length"],
    stride=config["inference"]["stride"],
    provenance=checkpoint_provenance,
    resolved_config=config,
)
```

Load `kpts_world` and `joint_valid` from the resulting NPZ and republish a
`MethodPrediction` whose method is `f"rotation_aware:{run_id}"`, frame IDs come
from the exact pair, and metadata comes from checkpoint provenance, under:

```text
local/runs/freeman_benchmark/fusion/rotation_aware/
  <run_id>/subject_01/<session_id>/fused_sequence.npz
```

Checkpoint metadata must state the original gymnastics training source,
checkpoint SHA256, ablation, zero-shot status, and no FreeMan reference access.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_fusion.py -q
```

Expected: all fusion tests pass.

Commit:

```bash
git add src/gymnastics/benchmarks/freeman/fusion.py \
  tests/freeman_benchmark/test_fusion.py
git commit -m "feat: add FreeMan zero-shot rotation-aware fusion"
```

---

### Task 7: Sequence-Level Metrics, Subject Aggregation, and Statistical Tests

**Files:**
- Create: `src/gymnastics/benchmarks/freeman/evaluation.py`
- Create: `tests/freeman_benchmark/test_evaluation_report.py`

**Interfaces:**
- Consumes: `ReferenceSequence`, `MethodPrediction`, and the fixed mapping
- Produces: `evaluate_session(prediction: MethodPrediction, reference: ReferenceSequence, thresholds_mm: Sequence[float]) -> SessionMetrics`
- Produces: `aggregate_metrics(rows: Sequence[SessionMetrics]) -> EvaluationTables`
- Produces: `paired_method_tests(subject_table: pd.DataFrame, *, seed: int, bootstrap_samples: int) -> pd.DataFrame`

- [ ] **Step 1: Write failing spatial and temporal metric tests**

Construct a two-frame 17-joint reference and apply one known scale, rotation,
translation, plus a controlled residual:

```python
def test_sequence_sim3_removes_one_static_frame_transform() -> None:
    metrics = evaluate_session(prediction, reference, thresholds_mm=(50, 100))
    assert metrics.sim3_mpjpe_mm == pytest.approx(expected_residual_mm)
    assert metrics.root_mpjpe_mm == pytest.approx(expected_root_error_mm)
    assert metrics.pa_mpjpe_mm <= metrics.sim3_mpjpe_mm
```

Add:

- frame-dependent rotations remain visible in sequence-level Sim3 MPJPE;
- PA-MPJPE removes per-frame rigid similarity and is labelled secondary;
- velocity error divides by `1/fps`;
- acceleration error divides by `(1/fps)^2`;
- PCK thresholds and AUC are expressed in millimetres;
- invalid joints do not enter numerators or denominators;
- missing frame IDs fail closed rather than shifting frames.

- [ ] **Step 2: Write failing subject-balanced aggregation tests**

Use one subject with one long session and another with one short session. Assert
the headline overall is the arithmetic mean of subject means, not a pooled
frame mean. Assert aggregation columns include FPS, official split, subject,
session, method, evaluated joints, valid frames, valid points, and coverage.

For statistics, assert matched subject intersection, paired effect
`candidate - baseline`, bootstrap 95% confidence interval with the configured
seed, Wilcoxon signed-rank p-value, and Holm-adjusted p-value.

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_evaluation_report.py -q
```

Expected: collection fails because `freeman.evaluation` does not exist.

- [ ] **Step 4: Implement reference mapping and session alignment**

`load_session_reference` returns the committed
`keypoints3d_optim` COCO17 reference. `evaluate_session` joins by exact
`frame_ids`, then fits one Umeyama Sim3 over all mutually valid points in the
session. Require at least three non-collinear valid points.

Compute:

- sequence-Sim3 MPJPE, median, P95, and maximum;
- pelvis/root-relative MPJPE using midpoint of left/right hips;
- per-frame PA-MPJPE as a secondary diagnostic;
- PCK at every configured threshold;
- normalized AUC from 0 through `auc_max_threshold_mm`;
- per-joint Euclidean error;
- first-difference velocity error;
- second-difference acceleration error;
- valid-frame and valid-joint coverage;
- excluded frames and failure reasons.

- [ ] **Step 5: Implement immutable metric rows and aggregation**

Define:

```python
@dataclass(frozen=True)
class SessionMetrics:
    subject_id: int
    session_id: str
    fps: int
    split: str
    scenario: str | None
    action: str | None
    method: str
    classification: str
    frames_total: int
    frames_valid: int
    valid_points: int
    sim3_mpjpe_mm: float
    root_mpjpe_mm: float
    pa_mpjpe_mm: float
    velocity_error_mm_s: float
    acceleration_error_mm_s2: float
    pck: Mapping[int, float]
    auc: float
    coverage: float

@dataclass(frozen=True)
class EvaluationTables:
    by_session: pd.DataFrame
    by_subject: pd.DataFrame
    by_method: pd.DataFrame
    by_joint: pd.DataFrame
    by_split: pd.DataFrame
    by_scenario: pd.DataFrame
    paired_statistics: pd.DataFrame
    failures: pd.DataFrame
```

`aggregate_metrics` produces dataframes by session, subject, method, joint,
FPS, official split, and non-null official scenario/action metadata. Valid
ranking rows include `VALID` only; diagnostics are written separately.

- [ ] **Step 6: Implement matched statistical comparison**

Preserve `view_a` and `view_b` as two declared baselines without choosing one
after observing FreeMan errors. For every valid fusion method and each
single-view baseline:

1. inner-join subject means with the baseline;
2. report mean and median paired difference;
3. bootstrap subjects with replacement for a deterministic 95% CI;
4. run `scipy.stats.wilcoxon`;
5. apply Holm correction over all valid method comparisons.

Do not run inferential statistics when fewer than ten matched subjects exist;
emit status `insufficient_subject_coverage`.

- [ ] **Step 7: Run tests and commit**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_evaluation_report.py -q
```

Expected: evaluation tests pass; report tests added next remain absent.

Commit:

```bash
git add src/gymnastics/benchmarks/freeman/evaluation.py \
  src/gymnastics/benchmarks/freeman/schema.py \
  tests/freeman_benchmark/test_evaluation_report.py
git commit -m "feat: add FreeMan benchmark metrics"
```

---

### Task 8: Machine-Readable Outputs and Markerless-Reference Report

**Files:**
- Create: `src/gymnastics/benchmarks/freeman/report.py`
- Modify: `tests/freeman_benchmark/test_evaluation_report.py`

**Interfaces:**
- Consumes: `EvaluationTables`, pair-selection records, failure records,
  resolved config, dataset/download manifests
- Produces: `write_report(tables: EvaluationTables, context: ReportContext, output_root: Path) -> ReportOutputs`

Define:

```python
@dataclass(frozen=True)
class ReportContext:
    resolved_config: Mapping[str, Any]
    dataset_manifest: Mapping[str, Any]
    download_manifest: Mapping[str, Any]
    camera_pairs: pd.DataFrame
    checkpoint_metadata: Mapping[str, Any]
    code_commit: str

@dataclass(frozen=True)
class ReportOutputs:
    markdown: Path
    results_json: Path
    csv_paths: Mapping[str, Path]
```

- [ ] **Step 1: Write failing output and wording tests**

```python
def test_report_separates_valid_and_diagnostic_methods(tables, context, tmp_path) -> None:
    outputs = write_report(tables, context, tmp_path)
    text = outputs.markdown.read_text(encoding="utf-8")
    assert "public markerless multi-view reference" in text
    assert "independent marker-based motion capture" in text
    assert "sim3_face_stable_joint_weight" in text
    assert "excluded from valid ranking" in text
    assert "all 40 subjects" in text
```

Assert generated files:

```text
evaluation/metrics_by_session.csv
evaluation/metrics_by_subject.csv
evaluation/metrics_by_method.csv
evaluation/metrics_by_joint.csv
evaluation/metrics_by_split.csv
evaluation/metrics_by_scenario.csv
evaluation/paired_statistics.csv
evaluation/failures.csv
evaluation/results.json
report/freeman_benchmark_report.md
report/camera_pairs.csv
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_evaluation_report.py -q
```

Expected: report tests fail because `write_report` is absent.

- [ ] **Step 3: Implement deterministic CSV and JSON exports**

Sort rows by subject, FPS, session, method, and joint. Use explicit `NaN` JSON
conversion to `null`. `results.json` records:

- repository ID and revision;
- archive inventory hash;
- resolved configuration;
- selected camera pairs and geometry diagnostics;
- mapping version and 17 joint names;
- SAM3D checkpoint identity;
- rotation-aware checkpoint identities;
- method classification;
- dataset coverage and failure counts;
- subject-balanced headline metrics;
- code commit from `git rev-parse HEAD`.

- [ ] **Step 4: Implement the Markdown report**

The report contains:

1. protocol and zero-shot statement;
2. explicit FreeMan reference provenance limitation;
3. downloaded/processed coverage for 40 subjects and both FPS subsets;
4. camera pair distribution and target-angle error;
5. valid method ranking;
6. separated oracle/leaky diagnostic table;
7. per-FPS, per-split, per-joint, and available official scenario/action results;
8. temporal quality metrics;
9. paired statistics;
10. failures and exclusions;
11. reproducibility commands and artifact paths.

The report generator refuses to call a result "complete" if processed subject
coverage is below `evaluation.minimum_subject_coverage`.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_evaluation_report.py -q
```

Expected: all evaluation/report tests pass.

Commit:

```bash
git add src/gymnastics/benchmarks/freeman/report.py \
  tests/freeman_benchmark/test_evaluation_report.py
git commit -m "feat: report FreeMan zero-shot results"
```

---

### Task 9: Staged CLI and One-Subject-at-a-Time Orchestration

**Files:**
- Create: `src/gymnastics/benchmarks/freeman/cli.py`
- Modify: `src/gymnastics/cli.py`
- Modify: `tests/structure/test_cli.py`
- Create: `tests/freeman_benchmark/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: every prior stage API
- Produces: `make_parser() -> argparse.ArgumentParser`
- Produces: `main(argv: Sequence[str] | None = None) -> int`
- Produces: top-level routes `benchmark:freeman`

- [ ] **Step 1: Write failing unified and staged CLI tests**

```python
def test_unified_cli_exposes_freeman_benchmark() -> None:
    result = run_cli("benchmark", "freeman", "--help")
    assert result.returncode == 0
    for stage in ("inspect", "download", "infer", "fuse",
                  "evaluate", "report", "run"):
        assert stage in result.stdout
```

Mock stage functions and assert:

- `inspect` performs preflight and dataset inspection without download or
  inference mutation;
- `download` calls preflight before download;
- `infer`, `fuse`, `evaluate`, and `report` only call their stage;
- `run` processes subjects in numeric order;
- one subject is extracted, inferred, validated, and cleaned before the next;
- a failed inference preserves the subject workspace for diagnosis;
- `--subject 3 7`, `--fps 30`, `--frame-stride 4`, and `--force-stage infer`
  override only their documented scopes;
- `--frame-stride != 1` marks results diagnostic and non-headline.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_cli.py \
  tests/structure/test_cli.py -q
```

Expected: failure because the benchmark route does not exist.

- [ ] **Step 3: Implement staged parser**

Expose:

```text
gymnastics benchmark freeman inspect
gymnastics benchmark freeman download
gymnastics benchmark freeman infer
gymnastics benchmark freeman fuse
gymnastics benchmark freeman evaluate
gymnastics benchmark freeman report
gymnastics benchmark freeman run
```

Every subcommand accepts
`--config configs/benchmarks/freeman.yaml`. Stage-specific options are:

```text
--subject ID [ID ...]
--fps {30,60} [{30,60} ...]
--frame-stride N
--force-stage {inspect,infer,fuse,evaluate,report}
--keep-workspace
--dry-run
```

`download` has no subject subset because the confirmed scope retains the
complete release.

- [ ] **Step 4: Implement the run state machine**

`run` performs:

```text
preflight
  -> complete-release download validation
  -> shared annotation extraction/validation
  -> for subject 01..40:
       extract subject
       load sessions from 30FPS and 60FPS
       select/store camera pair per session
       infer both views
       validate inference
       deterministic fusion
       zero-shot rotation-aware fusion for available run IDs
       validate fusion
       cleanup exact subject workspace
  -> evaluate all cached methods
  -> aggregate
  -> report
```

Write `local/runs/freeman_benchmark/run_state.json` atomically after every
stage and subject. It records `pending`, `running`, `complete`, or `failed`
with error type/message and artifact hashes. A rerun starts from the first
invalid state.

If a subject fails after extraction, mark it failed and preserve its workspace.
Continue only when the error is session-local and the configured minimum
coverage can still be reached. Unknown schemas, systematic camera convention
failure, missing shared annotations, or disk reserve violations abort.

- [ ] **Step 5: Wire unified CLI**

Add:

```python
"benchmark:freeman": (
    "gymnastics.benchmarks.freeman.cli",
    "main",
    True,
)
```

Create nested `benchmark` and `freeman` parsers in `gymnastics.cli._parser`,
and return `benchmark:freeman` from `_target_key`.

- [ ] **Step 6: Add README commands**

Document:

```bash
conda run -n gymnastic gymnastics benchmark freeman inspect
conda run -n gymnastic gymnastics benchmark freeman download
conda run -n gymnastic gymnastics benchmark freeman run
```

State that the dataset is gated, approximately 829 GB compressed, results are
zero-shot, FreeMan 3D is markerless multi-view reference, and `local/` remains
untracked.

- [ ] **Step 7: Run tests and commit**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark \
  tests/structure/test_cli.py -q
```

Expected: all FreeMan and CLI tests pass.

Commit:

```bash
git add src/gymnastics/benchmarks/freeman/cli.py \
  src/gymnastics/cli.py \
  tests/freeman_benchmark/test_cli.py \
  tests/structure/test_cli.py \
  README.md
git commit -m "feat: expose staged FreeMan benchmark"
```

---

### Task 10: Full Verification, Local Preflight, and Experiment Handoff

**Files:**
- Modify only if verification reveals an in-scope defect: files created in Tasks 1–9
- Generate ignored artifacts: `local/runs/freeman_benchmark/inspect/*`

**Interfaces:**
- Consumes: installed project, local `hf`/Hugging Face identity, storage state,
  SAM3D checkpoints, rotation-aware checkpoints
- Produces: verified unit-test result, dry inspection report, and an explicit
  run/blocker status

- [ ] **Step 1: Run all focused tests**

Run:

```bash
conda run -n gymnastic python -m pytest tests/freeman_benchmark -q
```

Expected: all tests pass.

- [ ] **Step 2: Run affected existing tests**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/structure/test_cli.py \
  tests/test_fuse_experiment_matrix.py \
  tests/rotation_aware/test_inference.py \
  tests/rotation_aware/test_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run complete repository tests**

Run:

```bash
conda run -n gymnastic python -m pytest -q
```

Expected: all non-optional tests pass.

- [ ] **Step 4: Verify formatting and Git isolation**

Run:

```bash
git diff --check
git status --short
git check-ignore -v \
  local/datasets/freeman \
  local/runs/freeman_benchmark
```

Expected: no whitespace errors; both large-data paths match the `local/`
ignore rule; pre-existing unrelated worktree changes remain untouched.

- [ ] **Step 5: Run real local inspection**

Run:

```bash
conda run -n gymnastic gymnastics benchmark freeman inspect
```

Expected outcomes:

- success writes the authenticated identity, repository revision, archive
  inventory, required bytes, free bytes, safety reserve, checkpoint discovery,
  and current stage readiness under the ignored inspect directory;
- missing `hf`, pending gated access, insufficient storage, or missing
  checkpoints produces a precise non-zero blocker without downloading data.

- [ ] **Step 6: Start the complete release only when preflight succeeds**

Run:

```bash
conda run -n gymnastic gymnastics benchmark freeman run
```

This is a resumable long-running command. If gated access is not approved or
required checkpoints are unavailable, do not bypass the check; report the
exact blocker and leave the implementation verified and ready to resume.

- [ ] **Step 7: Final evidence review**

Confirm:

- `run_state.json` identifies every subject `01..40`;
- every processed session has a stored two-camera selection;
- every selected pair uses direct synchronized frame IDs;
- no valid fusion metadata reports FreeMan reference consumption;
- `metrics_by_subject.csv` contains subject-balanced values;
- diagnostics are excluded from the valid ranking;
- the Markdown report uses the markerless-reference wording;
- Git does not list downloaded or generated data.

- [ ] **Step 8: Commit verification-only fixes if any**

If verification required no code change, do not create an empty commit. If a
focused defect was fixed, stage only the FreeMan files changed and commit:

```bash
git commit -m "fix: verify FreeMan benchmark pipeline"
```
