# Unity-Supervised Fusion Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune real-gymnastics A4--A9 checkpoints with Unity native 3D supervision under direction-held-out cross-validation, run all 36 experiments, and report their accuracy against the existing zero-shot and triangulation baselines.

**Architecture:** Add an isolated supervised data adapter, differentiable Unity16/Sim3 loss, resumable fine-tuning runner, held-out inference/evaluation aggregator, and report generator inside `gymnastics.benchmarks.unity`. Reuse the existing rotation-aware model, corruption, feature, self-supervised loss, inference, and common Unity evaluator; never modify existing real-data artifacts or zero-shot results.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, PyYAML, existing SAM3D/rotation-aware modules, CSV/JSON/Markdown, pytest, Ruff, `gymnastic` conda environment.

## Global Constraints

- Run project code, tests, scripts, and Python tooling through the `gymnastic` conda environment.
- Read `/home/data/xchen/gymnastics/unity_benchmark` and existing SAM3D caches in place.
- Write all new artifacts below `local/runs/unity_benchmark/supervised_finetune`.
- Use only the training direction's Unity GT in each training fold.
- Never use the held-out direction or `static_sweep` for training, validation, checkpoint selection, early stopping, or hyperparameter selection.
- Use folds `left_to_right` and `right_to_left`, ablations A4--A9, and seeds 0, 1, and 2.
- Use window length 32, training stride 8, 100 epochs, AdamW, and learning rate `1e-4`.
- Fine-tune the full matching pretrained model and always evaluate the final epoch.
- Map model output to exactly the approved Unity16 joints.
- Fit one differentiable Sim3 per training window and one NumPy Sim3 per complete evaluation sequence; never use per-frame Procrustes.
- Use `L_total = L_unity_3d + 0.1 * L_existing_self_supervised`.
- Keep zero-shot, Unity-supervised, and diagnostic ranking groups separate.
- Treat `triangulation_oracle2d` and `sim3_face_stable_joint_weight` as diagnostics.
- Do not modify `local/runs/fuse_rotation_aware` or existing Unity zero-shot outputs.
- Fail on split leakage, non-finite Sim3, loss, gradient, or incomplete provenance.

---

## File Structure

Create:

- `configs/benchmarks/unity_supervised.yaml` — fixed matrix, optimization, loss, and output settings.
- `src/gymnastics/benchmarks/unity/supervised_data.py` — folds, supervised sequence contracts, cache/GT joining, and target-bearing window dataset.
- `src/gymnastics/benchmarks/unity/supervised_loss.py` — differentiable MHR70-to-Unity16 mapping, masked Umeyama Sim3, and supervised objective.
- `src/gymnastics/benchmarks/unity/supervised.py` — model initialization, epoch training, atomic checkpoints, resume validation, and held-out inference.
- `src/gymnastics/benchmarks/unity/supervised_evaluation.py` — run discovery, common evaluation, seed/fold aggregation, CSV/JSON/Markdown report.
- `tests/unity_benchmark/test_supervised_data.py`
- `tests/unity_benchmark/test_supervised_loss.py`
- `tests/unity_benchmark/test_supervised_training.py`
- `tests/unity_benchmark/test_supervised_evaluation.py`
- `tests/unity_benchmark/test_supervised_cli.py`

Modify:

- `src/gymnastics/benchmarks/unity/cli.py` — add four supervised stages.
- `src/gymnastics/benchmarks/unity/__init__.py` — expose stable supervised interfaces.
- `src/gymnastics/benchmarks/unity/dataset.py` — add an optional sequence filter
  so the training command does not materialize held-out GT records.
- `tests/structure/test_cli.py` only if its exact subcommand contract requires the new stage names.
- `docs/superpowers/specs/2026-07-26-unity-supervised-finetune-design.md` only if implementation uncovers a contradiction; do not silently change the approved protocol.

---

### Task 1: Fixed Configuration, Fold Contracts, and Leakage Audit

**Files:**
- Create: `configs/benchmarks/unity_supervised.yaml`
- Create: `src/gymnastics/benchmarks/unity/supervised_data.py`
- Create: `tests/unity_benchmark/test_supervised_data.py`
- Modify: `src/gymnastics/benchmarks/unity/dataset.py`
- Modify: `tests/unity_benchmark/test_dataset_mapping.py`

**Interfaces:**
- Consumes: `UnityBenchmark`, `group_evaluation_sequences`, `load_sam3d_camera_cache`, `build_pose_pair_trial`, `canonicalize_trial`.
- Produces:
  - `UnityFold`
  - `UNITY_SUPERVISED_FOLDS`
  - `UnitySupervisedSequence`
  - sequence-filtered `load_unity_benchmark`
  - `build_supervised_sequence`
  - `build_supervised_sequences`
  - `select_supervised_fold`
  - `audit_fold_isolation`

- [ ] **Step 1: Write failing fold and leakage tests**

Add:

```python
from pathlib import Path

import pytest

from gymnastics.benchmarks.unity.dataset import load_unity_benchmark
from gymnastics.benchmarks.unity.supervised_data import (
    UNITY_SUPERVISED_FOLDS,
    audit_fold_isolation,
    build_supervised_sequences,
    select_supervised_fold,
)


UNITY_ROOT = Path("/home/data/xchen/gymnastics/unity_benchmark")
SAM3D_ROOT = Path("local/runs/unity_benchmark/sam3d")
SKELETON = Path("configs/fusion/skeleton_mhr70.yaml")


def test_direction_folds_are_exact_and_static_is_evaluation_only() -> None:
    assert UNITY_SUPERVISED_FOLDS["left_to_right"].train_sequence == (
        "continuous_left_060_r00"
    )
    assert UNITY_SUPERVISED_FOLDS["left_to_right"].test_sequence == (
        "continuous_right_060_r00"
    )
    assert UNITY_SUPERVISED_FOLDS["right_to_left"].train_sequence == (
        "continuous_right_060_r00"
    )
    assert UNITY_SUPERVISED_FOLDS["right_to_left"].test_sequence == (
        "continuous_left_060_r00"
    )


def test_real_unity_fold_has_no_sample_or_sequence_leakage() -> None:
    benchmark = load_unity_benchmark(UNITY_ROOT)
    sequences = build_supervised_sequences(
        benchmark,
        SAM3D_ROOT,
        skeleton_path=SKELETON,
        fps=60.0,
    )
    train, test, static = select_supervised_fold(
        sequences, UNITY_SUPERVISED_FOLDS["left_to_right"]
    )

    audit_fold_isolation(
        UNITY_SUPERVISED_FOLDS["left_to_right"], train, test, static
    )
    assert len(train.sample_ids) == 97
    assert len(test.sample_ids) == 97
    assert len(static.sample_ids) == 5
    assert not set(train.sample_ids) & set(test.sample_ids)
    assert not set(train.sample_ids) & set(static.sample_ids)


def test_sequence_filtered_loader_materializes_only_training_direction() -> None:
    benchmark = load_unity_benchmark(
        UNITY_ROOT,
        sequence_ids=("continuous_left_060_r00",),
    )

    assert len(benchmark.frames) == 97
    assert {
        frame.sequence_id for frame in benchmark.frames
    } == {"continuous_left_060_r00"}
```

Add a negative test that replaces `test.sample_ids[0]` with a training ID and
asserts `ValueError` contains `sample leakage`.

- [ ] **Step 2: Run tests and verify collection fails**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_data.py -q
```

Expected: FAIL because `gymnastics.benchmarks.unity.supervised_data` does not
exist.

- [ ] **Step 3: Add the fixed supervised configuration**

Create:

```yaml
base_config: configs/benchmarks/unity.yaml

paths:
  output_root: local/runs/unity_benchmark/supervised_finetune
  skeleton: configs/fusion/skeleton_mhr70.yaml

matrix:
  ablations: [A4, A5, A6, A7, A8, A9]
  folds: [left_to_right, right_to_left]
  seeds: [0, 1, 2]

window:
  length: 32
  train_stride: 8

training:
  epochs: 100
  batch_size: 4
  learning_rate: 0.0001
  weight_decay: 0.0001
  optimizer: adamw
  device: cuda

loss:
  unity_3d_weight: 1.0
  self_supervised_weight: 0.1
  smooth_l1_beta_m: 0.02

evaluation:
  window_length: 32
  stride: 8
  static_sequence: static_sweep
  alignment: one_sim3_per_sequence
```

- [ ] **Step 4: Implement immutable fold and sequence contracts**

Implement:

```python
@dataclass(frozen=True)
class UnityFold:
    name: str
    train_sequence: str
    test_sequence: str


UNITY_SUPERVISED_FOLDS = MappingProxyType(
    {
        "left_to_right": UnityFold(
            "left_to_right",
            "continuous_left_060_r00",
            "continuous_right_060_r00",
        ),
        "right_to_left": UnityFold(
            "right_to_left",
            "continuous_right_060_r00",
            "continuous_left_060_r00",
        ),
    }
)


@dataclass(frozen=True)
class UnitySupervisedSequence:
    sequence_id: str
    sample_ids: np.ndarray
    raw_trial: PosePairTrial
    canonical_trial: PosePairTrial
    gt_unity16_m: np.ndarray
    gt_valid: np.ndarray
```

Copy arrays into read-only storage in `__post_init__` and validate:

```text
sample_ids       [T]
raw/canonical    [T,70,3]
gt_unity16_m     [T,16,3]
gt_valid         [T,16]
```

- [ ] **Step 5: Add sequence-filtered manifest loading**

Extend `load_unity_benchmark` to return `UnityBenchmark` from positional
`root: Path` and keyword-only
`sequence_ids: Sequence[str] | None = None`.

When `sequence_ids` is supplied, skip nonmatching records immediately after
reading `sample_id`, `sequence_id`, and `sample_type`; do not call
`_world_points`, `_image_points`, or image validation for skipped records.
The default `None` path must remain byte-for-byte compatible at the contract
level with all existing callers. Reject an empty requested subset.

- [ ] **Step 6: Join the existing cache to Unity GT**

Implement `build_supervised_sequence` returning `UnitySupervisedSequence`
from positional `benchmark: UnityBenchmark`, `sam3d_root: Path`, and
`sequence_id: str`, plus keyword-only `skeleton_path: Path` and `fps: float`.

Implement `build_supervised_sequences` returning
`Mapping[str, UnitySupervisedSequence]` from positional
`benchmark: UnityBenchmark` and `sam3d_root: Path`, plus keyword-only
`skeleton_path: Path` and `fps: float`.

`build_supervised_sequence` performs the five operations below for exactly one
sequence. `build_supervised_sequences` calls it for each of the three
evaluation sequences:

1. load cam0/cam1 SAM3D caches by exact `sample_id`;
2. create the raw `PosePairTrial` with `build_pose_pair_trial`;
3. canonicalize once with `canonicalize_trial`;
4. select GT through `select_unity_evaluation_joints`;
5. assert exact sample order and return an immutable result.

- [ ] **Step 7: Implement explicit isolation audit**

Implement:

```python
def select_supervised_fold(
    sequences: Mapping[str, UnitySupervisedSequence],
    fold: UnityFold,
) -> tuple[
    UnitySupervisedSequence,
    UnitySupervisedSequence,
    UnitySupervisedSequence,
]:
    train = sequences[fold.train_sequence]
    test = sequences[fold.test_sequence]
    static = sequences["static_sweep"]
    audit_fold_isolation(fold, train, test, static)
    return train, test, static


def audit_fold_isolation(
    fold: UnityFold,
    train: UnitySupervisedSequence,
    test: UnitySupervisedSequence,
    static: UnitySupervisedSequence,
) -> None:
    if train.sequence_id != fold.train_sequence:
        raise ValueError("training sequence does not match fold")
    if test.sequence_id != fold.test_sequence:
        raise ValueError("test sequence does not match fold")
    if static.sequence_id != "static_sweep":
        raise ValueError("static diagnostic sequence is missing")
    groups = (
        set(train.sample_ids.tolist()),
        set(test.sample_ids.tolist()),
        set(static.sample_ids.tolist()),
    )
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("sample leakage across Unity supervised fold")
    if tuple(map(len, groups)) != (97, 97, 5):
        raise ValueError("unexpected Unity supervised fold sizes")
```

Require exact sequence identities, pairwise-disjoint sample IDs, 97 training
frames, 97 test frames, and 5 static frames. Raise before constructing a
training dataset.

- [ ] **Step 8: Run fold tests**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_data.py \
  tests/unity_benchmark/test_dataset_mapping.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add configs/benchmarks/unity_supervised.yaml \
  src/gymnastics/benchmarks/unity/dataset.py \
  src/gymnastics/benchmarks/unity/supervised_data.py \
  tests/unity_benchmark/test_dataset_mapping.py \
  tests/unity_benchmark/test_supervised_data.py
git commit -m "feat: define Unity supervised direction folds"
```

---

### Task 2: Target-Bearing Training Windows

**Files:**
- Modify: `src/gymnastics/benchmarks/unity/supervised_data.py`
- Modify: `tests/unity_benchmark/test_supervised_data.py`

**Interfaces:**
- Consumes: `UnitySupervisedSequence`, `PosePairWindowDataset`,
  `WindowConfig`, `SplitManifest`.
- Produces: `UnitySupervisedWindowDataset`.

- [ ] **Step 1: Write failing window-target tests**

Add:

```python
def test_supervised_windows_align_gt_with_global_frame_indices() -> None:
    benchmark = load_unity_benchmark(UNITY_ROOT)
    sequences = build_supervised_sequences(
        benchmark, SAM3D_ROOT, skeleton_path=SKELETON, fps=60.0
    )
    train, _, _ = select_supervised_fold(
        sequences, UNITY_SUPERVISED_FOLDS["left_to_right"]
    )
    dataset = UnitySupervisedWindowDataset(
        train, skeleton_path=SKELETON, length=32, stride=8
    )

    assert len(dataset) == 10
    first = dataset[0]
    last = dataset[len(dataset) - 1]
    assert first["gt_unity16_m"].shape == (32, 16, 3)
    assert first["gt_valid"].shape == (32, 16)
    assert first["sample_ids"].tolist() == train.sample_ids[:32].tolist()
    assert last["sample_ids"][-1].item() == train.sample_ids[-1]
    assert first["training_sequence_id"] == train.sequence_id
```

Add a test using a 5-frame synthetic sequence with `length=8` that asserts the
last three `sample_ids` equal `-1`, `gt_valid` is false there, and
`padding_mask` is false there.

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_data.py \
  -k "supervised_windows" -q
```

Expected: FAIL because `UnitySupervisedWindowDataset` is undefined.

- [ ] **Step 3: Implement the dataset wrapper**

Implement a dataset that:

1. constructs an internal `PosePairWindowDataset` for the canonical training
   trial with a manifest containing only that sequence identity;
2. receives the internal sample's `global_frame_index`;
3. copies matching Unity16 GT, validity, and global `sample_ids`;
4. fills padded target positions with zero and IDs with `-1`;
5. adds `training_sequence_id`;
6. never stores or receives a held-out/static sequence.

Use the existing start calculation via `PosePairWindowDataset`; do not create
a second window indexing algorithm.

- [ ] **Step 4: Run data tests**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_data.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gymnastics/benchmarks/unity/supervised_data.py \
  tests/unity_benchmark/test_supervised_data.py
git commit -m "feat: add Unity supervised training windows"
```

---

### Task 3: Differentiable Unity16 Mapping and Window-Level Sim3

**Files:**
- Create: `src/gymnastics/benchmarks/unity/supervised_loss.py`
- Create: `tests/unity_benchmark/test_supervised_loss.py`

**Interfaces:**
- Produces:
  - `torch_map_mhr70_to_unity16(points, valid)`
  - `DifferentiableSim3`
  - `masked_window_sim3(prediction, target, valid)`
  - `apply_torch_sim3(points, transform)`

- [ ] **Step 1: Write failing differentiable mapping tests**

Add:

```python
def test_torch_mapping_matches_numpy_and_propagates_derived_gradients() -> None:
    points = torch.arange(
        2 * 70 * 3, dtype=torch.float32
    ).reshape(2, 70, 3)
    points.requires_grad_(True)
    valid = torch.ones((2, 70), dtype=torch.bool)

    mapped, mapped_valid = torch_map_mhr70_to_unity16(points, valid)
    expected = map_mhr70_to_unity(
        points.detach().numpy(), valid.numpy()
    )

    np.testing.assert_allclose(
        mapped.detach().numpy(), expected.points, rtol=0, atol=0
    )
    assert torch.equal(
        mapped_valid, torch.from_numpy(np.array(expected.valid, copy=True))
    )
    mapped[:, 0].sum().backward()
    assert torch.all(points.grad[:, 9] == 0.5)
    assert torch.all(points.grad[:, 10] == 0.5)
```

- [ ] **Step 2: Write failing one-Sim3 and gradient tests**

Add:

```python
def test_masked_window_sim3_recovers_one_transform_and_has_gradients() -> None:
    target = torch.randn(2, 12, 16, 3)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    prediction = (
        1.7 * torch.einsum("btjc,cd->btjd", target, rotation)
        + torch.tensor([2.0, -1.0, 0.4])
    )
    prediction.requires_grad_(True)
    valid = torch.ones((2, 12, 16), dtype=torch.bool)

    transform = masked_window_sim3(prediction, target, valid)
    aligned = apply_torch_sim3(prediction, transform)
    loss = torch.linalg.vector_norm(aligned - target, dim=-1).mean()

    assert loss.item() < 1e-4
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
```

Add another test that rotates only the second frame after constructing the
global transform and asserts its residual remains above `1e-2`; this proves
the implementation is not fitting one transform per frame.

Add degenerate tests for fewer than three usable points and collinear
zero-variance targets; each must raise `ValueError` with `degenerate Sim3`.

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_loss.py -q
```

Expected: FAIL because the supervised loss module does not exist.

- [ ] **Step 4: Implement differentiable mapping**

Implement the exact source table from `mapping.MHR70_EVALUATION_SOURCES`
without converting tensors to NumPy. Direct joints use indexing; hips and
toes stack their two source tensors and take the mean across the new source
axis. Derived validity is `torch.all`.
Zero invalid outputs with `torch.where`.

- [ ] **Step 5: Implement batched masked Umeyama**

Define:

```python
@dataclass(frozen=True)
class DifferentiableSim3:
    scale: torch.Tensor       # [B]
    rotation: torch.Tensor    # [B,3,3]
    translation: torch.Tensor # [B,3]
```

Flatten only `[T,J]` inside each batch item. Use masked centroids and
cross-covariance, `torch.linalg.svd`, determinant correction, positive scale,
and translation. Check at least three usable points, target/prediction
variance above `1e-10`, and finite outputs. Do not detach any transform
component.

- [ ] **Step 6: Run supervised loss tests**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_loss.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/gymnastics/benchmarks/unity/supervised_loss.py \
  tests/unity_benchmark/test_supervised_loss.py
git commit -m "feat: add differentiable Unity supervision geometry"
```

---

### Task 4: Masked Supervised Objective

**Files:**
- Modify: `src/gymnastics/benchmarks/unity/supervised_loss.py`
- Modify: `tests/unity_benchmark/test_supervised_loss.py`

**Interfaces:**
- Produces:
  - `UnitySupervisedLossConfig`
  - `UnitySupervisedLoss`
  - `compute_unity_supervised_loss(output, batch, config, *, self_supervised)`

- [ ] **Step 1: Write failing mask, padding, and non-finite tests**

Add:

```python
from types import SimpleNamespace


def supervised_loss_fixture():
    fused = torch.randn(1, 8, 70, 3, requires_grad=True)
    output = SimpleNamespace(
        fused_kpts=fused,
        valid=torch.ones((1, 8, 70), dtype=torch.bool),
    )
    target, target_valid = torch_map_mhr70_to_unity16(
        fused.detach(), output.valid
    )
    batch = {
        "gt_unity16_m": target.clone(),
        "gt_valid": target_valid.clone(),
        "padding_mask": torch.ones((1, 8), dtype=torch.bool),
    }
    return output, batch


def test_supervised_loss_uses_only_valid_non_padded_unity16_points() -> None:
    output, batch = supervised_loss_fixture()
    config = UnitySupervisedLossConfig(
        unity_3d_weight=1.0,
        self_supervised_weight=0.1,
        smooth_l1_beta_m=0.02,
    )

    self_loss = output.fused_kpts.sum() * 0.0
    original = compute_unity_supervised_loss(
        output, batch, config, self_supervised=self_loss
    )
    batch["gt_unity16_m"][~batch["gt_valid"]] = 1e6
    batch["gt_unity16_m"][~batch["padding_mask"]] = -1e6
    changed = compute_unity_supervised_loss(
        output, batch, config, self_supervised=self_loss
    )

    torch.testing.assert_close(original.unity_3d, changed.unity_3d)
```

Add assertions that:

- `total == unity_3d + 0.1 * self_supervised`;
- a NaN in a valid target raises `FloatingPointError`;
- a non-finite total raises `FloatingPointError`;
- every parameter receiving a gradient has only finite gradients.

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_loss.py \
  -k "supervised_loss" -q
```

Expected: FAIL because the loss config and function are undefined.

- [ ] **Step 3: Implement the supervised loss objects**

Implement:

```python
@dataclass(frozen=True)
class UnitySupervisedLossConfig:
    unity_3d_weight: float = 1.0
    self_supervised_weight: float = 0.1
    smooth_l1_beta_m: float = 0.02


@dataclass(frozen=True)
class UnitySupervisedLoss:
    unity_3d: torch.Tensor
    self_supervised: torch.Tensor
    total: torch.Tensor
```

Define the exact signature:

```python
def compute_unity_supervised_loss(
    output: FusionOutput,
    batch: Mapping[str, object],
    config: UnitySupervisedLossConfig,
    *,
    self_supervised: torch.Tensor,
) -> UnitySupervisedLoss:
    mapped, mapped_valid = torch_map_mhr70_to_unity16(
        output.fused_kpts, output.valid
    )
    target = batch["gt_unity16_m"]
    target_valid = batch["gt_valid"].bool()
    padding = batch["padding_mask"].bool()
    common = mapped_valid & target_valid & padding[:, :, None]
    transform = masked_window_sim3(mapped, target, common)
    aligned = apply_torch_sim3(mapped, transform)
    point_loss = torch.nn.functional.smooth_l1_loss(
        aligned,
        target,
        beta=config.smooth_l1_beta_m,
        reduction="none",
    ).sum(dim=-1)
    unity_3d = point_loss[common].mean()
    total = (
        config.unity_3d_weight * unity_3d
        + config.self_supervised_weight * self_supervised
    )
    if not torch.isfinite(total):
        raise FloatingPointError("Unity supervised loss is non-finite")
    return UnitySupervisedLoss(unity_3d, self_supervised, total)
```

It accepts the prepared batch and the already computed existing
self-supervised scalar.
Map `output.fused_kpts`, combine model/GT/padding validity, fit one window
Sim3, apply `torch.nn.functional.smooth_l1_loss` with
`reduction="none"`,
reduce over valid points, and form the exact weighted total.

- [ ] **Step 4: Run all supervised loss tests**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_loss.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gymnastics/benchmarks/unity/supervised_loss.py \
  tests/unity_benchmark/test_supervised_loss.py
git commit -m "feat: add masked Unity supervised objective"
```

---

### Task 5: One-Run Fine-Tuning and Atomic Provenance

**Files:**
- Create: `src/gymnastics/benchmarks/unity/supervised.py`
- Create: `tests/unity_benchmark/test_supervised_training.py`

**Interfaces:**
- Consumes:
  - `load_rotation_aware_model`
  - `UnitySupervisedWindowDataset`
  - private package-local training helpers `_forward_window`
  - `compute_self_supervised_losses`
  - `compute_unity_supervised_loss`
- Produces:
  - `UnityFineTuneConfig`
  - `UnityFineTuneRun`
  - `train_supervised_epoch`
  - `run_supervised_finetune`
  - `validate_completed_run(run, expected)`

- [ ] **Step 1: Write a failing one-step training test**

Build a tiny skeleton/model and two synthetic windows, then assert:

```python
def test_supervised_epoch_updates_model_and_reports_three_losses() -> None:
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    metrics = train_supervised_epoch(
        model,
        loader,
        optimizer,
        skeleton,
        loss_config=unity_loss_config,
        self_supervised_config=LossConfig(),
        corruption_config=CorruptionConfig(enabled_families=()),
        seed=0,
        epoch=0,
        device="cpu",
    )

    assert set(metrics) >= {
        "unity_3d_loss",
        "self_supervised_loss",
        "total_loss",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
```

- [ ] **Step 2: Write failing checkpoint/provenance tests**

Assert a completed run contains:

```text
fold
train_sequence
test_sequence
static_excluded_from_training = true
unity_gt_supervision = true
seed
ablation
source_checkpoint_sha256
final_checkpoint_sha256
git_commit
unity_manifest_sha256
sam3d_cache_identity
resolved_config
history with exactly 100 epochs in a full run
```

Use a two-epoch test configuration for speed. Corrupt the saved fold name and
assert `validate_completed_run` returns false. Request A5 with an A4 source
checkpoint and assert `ValueError` contains `checkpoint ablation mismatch`.

- [ ] **Step 3: Run training tests and verify failure**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_training.py -q
```

Expected: FAIL because `supervised.py` does not exist.

- [ ] **Step 4: Implement configuration and run paths**

Implement:

```python
@dataclass(frozen=True)
class UnityFineTuneConfig:
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    window_length: int = 32
    train_stride: int = 8
    device: str = "cuda"


@dataclass(frozen=True)
class UnityFineTuneRun:
    ablation: str
    fold: str
    seed: int
    train_sequence: str
    test_sequence: str
    run_root: Path
    final_checkpoint: Path
    metrics_path: Path
    provenance_path: Path
```

Use:

```text
<output>/fold_<fold>/<ablation>/seed_<seed>/
  final.pt
  history.json
  provenance.json
  resolved_config.yaml
```

- [ ] **Step 5: Define exact orchestration signatures**

Implement `run_supervised_finetune` returning `UnityFineTuneRun` with positional
argument `train_sequence: UnitySupervisedSequence`, followed by keyword-only
arguments:

```text
ablation: str
fold: UnityFold
seed: int
source_checkpoint: Path
skeleton_path: Path
output_root: Path
config: UnityFineTuneConfig
loss_config: UnitySupervisedLossConfig
self_supervised_config: LossConfig
corruption_config: CorruptionConfig
```

Implement `validate_completed_run` returning `bool` with positional
`run: UnityFineTuneRun` and keyword-only:

```text
source_checkpoint_sha256: str
resolved_config: Mapping[str, object]
unity_manifest_sha256: str
```

The runner must not accept a `UnityBenchmark`, a sequence mapping, held-out
sequence, or static sequence. This makes held-out GT unavailable by
construction once the training loader is created.

- [ ] **Step 6: Implement one supervised epoch**

For every prepared window:

1. call the existing `_forward_window` so canonical feature extraction,
   corruptions, validity, and model behavior stay identical to real training;
2. compute the existing self-supervised loss with
   `compute_self_supervised_losses`;
3. compute the Unity loss;
4. fail on a non-finite component;
5. backpropagate;
6. inspect all non-null gradients and fail if any are non-finite;
7. call AdamW step;
8. return arithmetic means of the three scalar losses.

- [ ] **Step 7: Implement deterministic full-run orchestration**

`run_supervised_finetune` must:

1. seed Python, NumPy, and PyTorch;
2. audit fold isolation before constructing the loader;
3. load the matching pretrained ablation;
4. construct AdamW with the exact config;
5. train exactly `epochs`;
6. append and atomically rewrite `history.json` after each epoch;
7. save only `final.pt` after the final epoch;
8. write provenance and final SHA-256 atomically;
9. never call the test/static evaluator.

Use a temporary suffix followed by `Path.replace` for JSON, YAML, and
checkpoint publication.

- [ ] **Step 8: Implement strict resume validation**

`validate_completed_run` returns true only if all four artifacts exist and:

- requested ablation/fold/seed/config match;
- history length equals configured epochs;
- source and final hashes match;
- training/static/test identities satisfy the approved protocol;
- `unity_gt_supervision` and static exclusion flags are true.

Otherwise return false so the matrix command reruns the exact cell.

- [ ] **Step 9: Run training tests**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_training.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/gymnastics/benchmarks/unity/supervised.py \
  tests/unity_benchmark/test_supervised_training.py
git commit -m "feat: train Unity-supervised fusion runs"
```

---

### Task 6: Held-Out and Static Inference

**Files:**
- Modify: `src/gymnastics/benchmarks/unity/supervised.py`
- Modify: `tests/unity_benchmark/test_supervised_training.py`

**Interfaces:**
- Produces:
  - `run_finetuned_inference`
  - `discover_completed_runs`

- [ ] **Step 1: Write failing inference isolation tests**

Add a completed two-epoch fixture and assert:

```python
def test_finetuned_inference_writes_only_heldout_and_static_sequences() -> None:
    outputs = run_finetuned_inference(
        run,
        sequences,
        skeleton_path=SKELETON,
        window_length=32,
        stride=8,
        device="cpu",
    )

    assert {item.sequence_id for item in outputs} == {
        run.test_sequence,
        "static_sweep",
    }
    assert all(item.metadata["ranking_group"] == "unity_supervised" for item in outputs)
    assert all(item.metadata["unity_gt_used_for_training"] for item in outputs)
    assert not (
        run.run_root / "inference" / f"{run.train_sequence}.npz"
    ).exists()
```

Assert output metadata contains fold, seed, source/final checkpoint hashes,
and `evaluation_gt_loaded_after_training=true`.

- [ ] **Step 2: Run the new test and verify failure**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_training.py \
  -k "finetuned_inference" -q
```

Expected: FAIL because the inference function is undefined.

- [ ] **Step 3: Implement completed-run discovery**

Scan only the exact 36 configured paths. Reject duplicate or unexpected fold,
ablation, and seed identities. Return runs in deterministic
`fold, ablation, seed` order.

Implement `discover_completed_runs` returning an immutable tuple of
`UnityFineTuneRun` with positional `output_root: Path` and keyword-only
`expected_cells: Sequence[tuple[str, str, int]]` and
`resolved_config: Mapping[str, object]`.

- [ ] **Step 4: Implement held-out inference**

Load `final.pt` into the matching model, then call existing `run_inference`
for:

- the held-out continuous direction;
- `static_sweep`.

Use raw synchronized SAM3D trials, window length 32, stride 8, and CPU
inference unless the existing inference path gains explicit tensor-device
support. Save normalized `MethodSequence` files under each run's
`inference/` directory and include the ranking group
`unity_supervised`.

Implement `run_finetuned_inference` returning an immutable tuple of
`MethodSequence` with positional `run: UnityFineTuneRun` and
`sequences: Mapping[str, UnitySupervisedSequence]`, followed by keyword-only
`skeleton_path: Path`, `window_length: int`, `stride: int`, and
`device: str = "cpu"`.

- [ ] **Step 5: Run training/inference tests**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_training.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gymnastics/benchmarks/unity/supervised.py \
  tests/unity_benchmark/test_supervised_training.py
git commit -m "feat: infer held-out Unity supervised runs"
```

---

### Task 7: Unified Evaluation and Seed/Fold Aggregation

**Files:**
- Create: `src/gymnastics/benchmarks/unity/supervised_evaluation.py`
- Create: `tests/unity_benchmark/test_supervised_evaluation.py`

**Interfaces:**
- Consumes:
  - existing `evaluate_method_sequence`
  - existing zero-shot `results.json`
  - fine-tuned `MethodSequence` artifacts.
- Produces:
  - `FineTunedRunEvaluation`
  - `FineTunedEvaluationBundle`
  - `evaluate_finetuned_runs(benchmark, runs, sequences)`
  - `aggregate_finetuned_results(rows)`
  - `build_finetuned_bundle(results, failures, provenance)`

- [ ] **Step 1: Define evaluation result contracts**

Implement:

```python
@dataclass(frozen=True)
class FineTunedRunEvaluation:
    ablation: str
    fold: str
    seed: int
    split_kind: str
    evaluation: EvaluationResult


@dataclass(frozen=True)
class FineTunedEvaluationBundle:
    run_results: Sequence[FineTunedRunEvaluation]
    failures: Sequence[Mapping[str, object]]
    tables: Mapping[str, Sequence[Mapping[str, object]]]
    supervised_ranking: Sequence[Mapping[str, object]]
    static_diagnostics: Sequence[Mapping[str, object]]
    provenance: Mapping[str, object]
```

- [ ] **Step 2: Write failing common-evaluator test**

Use a transformed synthetic prediction and assert the supervised evaluator
calls the same one-Sim3-per-sequence semantics:

```python
def test_finetuned_evaluation_uses_common_sequence_sim3() -> None:
    result = evaluate_finetuned_sequence(
        candidate,
        reference,
        visibility=visibility,
        actual_angles_deg=angles,
        fold="left_to_right",
        ablation="A4",
        seed=0,
    )

    assert result.summary["mpjpe_mm"] < 1e-3
    assert result.metadata["ranking_group"] == "unity_supervised"
```

Perturb one frame with an extra rotation and assert its mean error exceeds
`1e-2` metres.

- [ ] **Step 3: Write failing aggregation tests**

Construct six rows for one ablation: three seeds in each fold. Assert:

```python
rows = [
    {"ablation": "A4", "fold": "left_to_right", "seed": 0, "mpjpe_mm": 100.0},
    {"ablation": "A4", "fold": "left_to_right", "seed": 1, "mpjpe_mm": 110.0},
    {"ablation": "A4", "fold": "left_to_right", "seed": 2, "mpjpe_mm": 120.0},
    {"ablation": "A4", "fold": "right_to_left", "seed": 0, "mpjpe_mm": 200.0},
    {"ablation": "A4", "fold": "right_to_left", "seed": 1, "mpjpe_mm": 210.0},
    {"ablation": "A4", "fold": "right_to_left", "seed": 2, "mpjpe_mm": 220.0},
]
summary = aggregate_finetuned_results(rows)
row = summary[0]
assert row["ablation"] == "A4"
assert row["folds"] == 2
assert row["seeds"] == 3
assert row["runs"] == 6
assert row["macro_mpjpe_mm"] == pytest.approx(160.0)
assert row["seed_std_mpjpe_mm"] == pytest.approx(10.0)
```

The macro result must first average seeds within each fold, then give each
direction equal weight. Assert a missing seed raises `ValueError` containing
`incomplete 2x3 matrix`.

- [ ] **Step 4: Run evaluation tests and verify failure**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_evaluation.py -q
```

Expected: FAIL because the evaluation module does not exist.

- [ ] **Step 5: Implement run-level evaluation**

For each fine-tuned run:

- evaluate its held-out continuous sequence;
- evaluate static as a separate OOD diagnostic;
- retain fold, ablation, seed, ranking group, source/final hashes;
- use the existing Unity16 visibility masks and actual angles;
- use the existing `_angle_offset` convention;
- never evaluate the training sequence as held-out evidence.

Implement `evaluate_finetuned_runs` returning an immutable tuple of
`FineTunedRunEvaluation` with positional `benchmark: UnityBenchmark`,
`runs: Sequence[UnityFineTuneRun]`, and
`sequences: Mapping[str, UnitySupervisedSequence]`.

Implement `aggregate_finetuned_results` returning an immutable tuple of
summary mappings from positional `rows: Sequence[Mapping[str, object]]`.

- [ ] **Step 6: Implement exact matrix aggregation**

Return immutable tables:

```text
run_results
by_fold
by_ablation
by_sequence
by_joint
by_visibility
static_diagnostics
```

For each A4--A9 require exactly two folds and seeds `{0,1,2}`. Report mean,
sample standard deviation, minimum, and maximum. Continuous macro metrics
weight the two folds equally. Static metrics remain diagnostic and do not
enter the continuous macro ranking.

- [ ] **Step 7: Build the immutable bundle**

Implement `build_finetuned_bundle` with positional
`results: Sequence[FineTunedRunEvaluation]`, keyword-only
`failures: Sequence[Mapping[str, object]]`, and
`provenance: Mapping[str, object]`. Populate every table listed above and
sort the supervised ranking by `macro_mpjpe_mm`.

- [ ] **Step 8: Run evaluation tests**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_evaluation.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/gymnastics/benchmarks/unity/supervised_evaluation.py \
  tests/unity_benchmark/test_supervised_evaluation.py
git commit -m "feat: evaluate Unity supervised fusion matrix"
```

---

### Task 8: Supervised Report and Ranking Separation

**Files:**
- Modify: `src/gymnastics/benchmarks/unity/supervised_evaluation.py`
- Modify: `tests/unity_benchmark/test_supervised_evaluation.py`

**Interfaces:**
- Produces:
  `write_finetuned_report(bundle, output_root, *, baseline_results) -> Path`.

- [ ] **Step 1: Write failing report tests**

Assert the report writes:

```text
evaluation/run_results.csv
evaluation/by_fold.csv
evaluation/by_ablation.csv
evaluation/by_joint.csv
evaluation/by_visibility.csv
report/results.json
report/unity_supervised_finetune_report.md
report/figures/zero_shot_vs_supervised_mpjpe.png
```

Assert Markdown contains:

```text
Unity-Supervised Training
Direction-Held-Out Results
Zero-Shot vs Fine-Tuned
Static OOD Diagnostic
Triangulation Comparison
Interpretation Boundary
```

Assert `triangulation_oracle2d` and
`sim3_face_stable_joint_weight` are never included in a valid or supervised
ranking and every supervised row states `Unity GT used for training`.

- [ ] **Step 2: Run report tests and verify failure**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_evaluation.py \
  -k "report" -q
```

Expected: FAIL because `write_finetuned_report` is undefined.

- [ ] **Step 3: Implement machine-readable outputs**

Use `csv.DictWriter`, strict JSON-compatible scalars, atomic publication, and
`np.savez_compressed` for per-frame errors. The JSON root must contain:

```json
{
  "supervised_ranking": [],
  "zero_shot_ranking": [],
  "valid_nonlearned_ranking": [],
  "diagnostics": [],
  "tables": {},
  "provenance": {}
}
```

Implement `write_finetuned_report` returning `Path` with positional
`bundle: FineTunedEvaluationBundle` and `output_root: Path`, plus keyword-only
`baseline_results: Mapping[str, object]`.

- [ ] **Step 4: Implement concise conclusions**

Generate statements that answer:

1. which fine-tuned ablation has the lowest direction-macro MPJPE;
2. its absolute and percentage change from matching zero-shot;
3. whether it beats 166.537 mm;
4. whether it approaches or beats 30.259 mm;
5. how left-to-right and right-to-left differ;
6. whether static OOD behavior is consistent;
7. that the benchmark contains one avatar and does not establish
   population-level significance.

- [ ] **Step 5: Run report tests**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src MPLCONFIGDIR=local/cache/matplotlib \
  python -m pytest tests/unity_benchmark/test_supervised_evaluation.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gymnastics/benchmarks/unity/supervised_evaluation.py \
  tests/unity_benchmark/test_supervised_evaluation.py
git commit -m "feat: report Unity supervised fusion results"
```

---

### Task 9: CLI and Resumable 36-Run Matrix

**Files:**
- Modify: `src/gymnastics/benchmarks/unity/cli.py`
- Modify: `src/gymnastics/benchmarks/unity/__init__.py`
- Create: `tests/unity_benchmark/test_supervised_cli.py`
- Modify: `tests/structure/test_cli.py` only when needed by the structural assertion.

**Interfaces:**
- Produces CLI stages:
  - `finetune`
  - `finetune-matrix`
  - `evaluate-finetuned`
  - `report-finetuned`

- [ ] **Step 1: Write failing CLI parsing tests**

Add:

```python
def test_unity_cli_exposes_supervised_stages() -> None:
    help_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gymnastics",
            "benchmark",
            "unity",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for stage in (
        "finetune",
        "finetune-matrix",
        "evaluate-finetuned",
        "report-finetuned",
    ):
        assert stage in help_result.stdout
```

Add parser tests for:

```text
finetune --ablation A4 --fold left_to_right --seed 0 --device cpu
finetune-matrix --device cuda
evaluate-finetuned
report-finetuned
```

- [ ] **Step 2: Write failing matrix resume test**

Monkeypatch the one-run function and completion validator. Assert the matrix
enumerates exactly:

```python
{
    (ablation, fold, seed)
    for ablation in ("A4", "A5", "A6", "A7", "A8", "A9")
    for fold in ("left_to_right", "right_to_left")
    for seed in (0, 1, 2)
}
```

Mark two cells complete and assert they are skipped while the remaining 34
are invoked.

- [ ] **Step 3: Run CLI tests and verify failure**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_cli.py -q
```

Expected: FAIL because the new stages are absent.

- [ ] **Step 4: Implement configuration loading**

Load `unity_supervised.yaml`, then load its `base_config`. Resolve the dataset,
SAM3D cache, skeleton, source checkpoints, matrix, training, loss, and
evaluation settings into immutable runtime configuration. Reject any matrix
that is not exactly six ablations, two approved folds, and three approved
seeds.

For `finetune`, call `load_unity_benchmark` with only the fold's training
sequence ID, build exactly one `UnitySupervisedSequence`, and pass only that
sequence to `run_supervised_finetune`. The held-out and static records are
loaded only inside `evaluate-finetuned`, after every final checkpoint has
been validated.

- [ ] **Step 5: Implement one-run and matrix stages**

`finetune` requires explicit ablation, fold, and seed and prints the run root,
source hash, final hash, and final three losses.

`finetune-matrix` iterates deterministic fold/ablation/seed order, validates
each cell before skipping, runs incomplete cells, and prints completed,
reused, and failed counts. It exits nonzero if any cell fails.

- [ ] **Step 6: Implement evaluation/report stages**

`evaluate-finetuned` requires all 36 valid final runs, performs missing
held-out/static inference, evaluates, aggregates, and writes the report.

`report-finetuned` requires saved evaluation artifacts and regenerates only
the report without retraining.

- [ ] **Step 7: Run CLI and Unity benchmark tests**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m pytest \
  tests/unity_benchmark/test_supervised_cli.py \
  tests/unity_benchmark/test_cli.py \
  tests/structure/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/gymnastics/benchmarks/unity/cli.py \
  src/gymnastics/benchmarks/unity/__init__.py \
  tests/unity_benchmark/test_supervised_cli.py \
  tests/structure/test_cli.py
git commit -m "feat: expose Unity supervised fine-tuning CLI"
```

---

### Task 10: Focused Integration and Protocol Audit

**Files:**
- Modify the supervised modules and tests only when a failing audit exposes a
  concrete defect.

**Interfaces:**
- Produces a locally verified implementation ready for the real matrix.

- [ ] **Step 1: Run all Unity and rotation-aware focused tests**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src MPLCONFIGDIR=local/cache/matplotlib \
  python -m pytest \
  tests/unity_benchmark \
  tests/rotation_aware/test_model.py \
  tests/rotation_aware/test_losses.py \
  tests/rotation_aware/test_training.py \
  tests/rotation_aware/test_inference.py \
  tests/rotation_aware/test_cli.py \
  tests/structure/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run Ruff on all changed Python files**

Run:

```bash
conda run -n gymnastic python -m ruff check \
  src/gymnastics/benchmarks/unity \
  tests/unity_benchmark \
  tests/structure/test_cli.py
```

Expected: `All checks passed!`

- [ ] **Step 3: Run a two-epoch CPU smoke run**

Use a temporary config override or direct test entrypoint with:

```text
A4
left_to_right
seed 0
epochs 2
device cpu
```

Verify:

- two history entries;
- finite losses;
- exact fold provenance;
- final checkpoint reloads;
- inference contains only right and static;
- held-out result uses 97 frames and static uses 5.

- [ ] **Step 4: Audit that test GT cannot reach training**

Instrument the dataset constructor in a test to record every sample ID
requested during two epochs. Assert the observed set is a subset of the
training fold and disjoint from both held-out and static IDs.

- [ ] **Step 5: Commit any audit fixes**

If Steps 1--4 required changes:

```bash
git add src/gymnastics/benchmarks/unity tests/unity_benchmark
git commit -m "fix: enforce Unity supervised protocol audit"
```

If no changes were required, do not create an empty commit.

---

### Task 11: Execute the 36 Fine-Tuning Runs

**Files:**
- Generated only below `local/runs/unity_benchmark/supervised_finetune`.

**Interfaces:**
- Consumes the completed implementation and existing A4--A9 checkpoints.
- Produces 36 validated final checkpoints and histories.

- [ ] **Step 1: Record pre-run environment**

Run:

```bash
git rev-parse HEAD
nvidia-smi
conda run -n gymnastic python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

Save the resolved device inventory in the matrix provenance.

- [ ] **Step 2: Run one full A4 cell first**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src MPLCONFIGDIR=local/cache/matplotlib \
  python -m gymnastics benchmark unity finetune \
  --supervised-config configs/benchmarks/unity_supervised.yaml \
  --ablation A4 \
  --fold left_to_right \
  --seed 0 \
  --device cuda
```

Verify 100 finite history entries, matching hashes, and reloadable final
checkpoint before starting the remaining matrix.

- [ ] **Step 3: Run or resume the full matrix**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src MPLCONFIGDIR=local/cache/matplotlib \
  python -m gymnastics benchmark unity finetune-matrix \
  --supervised-config configs/benchmarks/unity_supervised.yaml \
  --device cuda
```

Poll progress at intervals below 60 seconds. If GPU access is limited to one
device, run sequentially. Do not change epochs, learning rate, or seed count
after observing held-out results.

- [ ] **Step 4: Validate the exact matrix**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src python -m gymnastics \
  benchmark unity finetune-matrix \
  --supervised-config configs/benchmarks/unity_supervised.yaml \
  --device cuda
```

Expected: 36 reused, 0 incomplete, 0 failed.

---

### Task 12: Evaluate, Report, and State the Experimental Conclusion

**Files:**
- Generated below:
  - `local/runs/unity_benchmark/supervised_finetune/evaluation`
  - `local/runs/unity_benchmark/supervised_finetune/report`

**Interfaces:**
- Produces final CSV, JSON, Markdown, figures, and user-facing conclusions.

- [ ] **Step 1: Run held-out/static inference and evaluation**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src MPLCONFIGDIR=local/cache/matplotlib \
  python -m gymnastics benchmark unity evaluate-finetuned \
  --supervised-config configs/benchmarks/unity_supervised.yaml
```

Expected: 36 continuous held-out run results, 36 static diagnostics, 0
explicit failures.

- [ ] **Step 2: Verify machine-readable consistency**

Check:

```text
6 ablations
2 folds per ablation
3 seeds per fold
97 held-out frames per continuous run
5 static frames per static run
16 joints
no NaN in primary aggregated MPJPE
zero-shot and diagnostic rows unchanged from the previous results.json
```

- [ ] **Step 3: Regenerate the report**

Run:

```bash
conda run -n gymnastic env PYTHONPATH=src MPLCONFIGDIR=local/cache/matplotlib \
  python -m gymnastics benchmark unity report-finetuned \
  --supervised-config configs/benchmarks/unity_supervised.yaml
```

- [ ] **Step 4: Run complete repository verification**

Run:

```bash
conda run -n gymnastic python -m ruff check \
  src/gymnastics/benchmarks/unity \
  tests/unity_benchmark \
  tests/structure/test_cli.py

conda run -n gymnastic env PYTHONPATH=src MPLCONFIGDIR=local/cache/matplotlib \
  python -m pytest -q

git diff --check
git status --short
```

Expected: Ruff passes, the full test suite passes, no whitespace errors, and
only intentional tracked changes remain.

- [ ] **Step 5: Commit final implementation fixes**

If evaluation exposed an implementation defect, add its regression test,
apply the minimal fix, rerun Step 4, then commit:

```bash
git add src/gymnastics/benchmarks/unity tests/unity_benchmark
git commit -m "fix: finalize Unity supervised evaluation"
```

Do not commit generated `local/runs` artifacts.

- [ ] **Step 6: Report the conclusion**

Report:

- the complete supervised MPJPE ranking with mean and standard deviation;
- left-to-right and right-to-left results;
- matching zero-shot deltas;
- comparison with 166.537 mm direct-3D fusion;
- comparison with 30.259 mm SAM3D-2D triangulation;
- static OOD behavior;
- best/worst joint changes;
- whether Unity supervision improves direction-held-out generalization;
- the one-avatar/one-environment limitation;
- exact report and CSV file links;
- verification evidence and branch/commit state.

Use measured artifacts only. Do not claim statistical significance or
human-population generalization.
