# Task 4 Report: Window Dataset and Reproducible Synthetic Corruption

## Scope

- Added `fuse/rotation_aware/corruptions.py`.
- Added `fuse/rotation_aware/dataset.py`.
- Added focused tests in `tests/rotation_aware/test_corruptions.py` and `tests/rotation_aware/test_dataset.py`.
- Did not modify existing fuse behavior, logs, AGENTS instructions, GPU paths, or triangulation code.

## RED Evidence

1. Ran:

   ```bash
   conda run -n gymnastic python -m pytest tests/rotation_aware/test_corruptions.py tests/rotation_aware/test_dataset.py -q
   ```

   Result: collection failed as expected with `ModuleNotFoundError` for the absent `fuse.rotation_aware.corruptions` and `fuse.rotation_aware.dataset` modules.

2. Added the invalid-target integer-shift regression before changing its production implementation. The same focused command then failed with `test_time_shift_leaves_reference_invalid_targets_and_masks_untouched`, demonstrating that time shifts rewrote originally invalid targets.

## GREEN Evidence

1. Focused Task 4 verification:

   ```bash
   conda run -n gymnastic python -m pytest tests/rotation_aware/test_corruptions.py tests/rotation_aware/test_dataset.py -q
   ```

   Result: `15 passed in 1.46s`.

2. Rotation-aware regression suite:

   ```bash
   conda run -n gymnastic python -m pytest tests/rotation_aware -q
   ```

   Result: `53 passed in 1.56s`.

3. Whitespace verification:

   ```bash
   git diff --check
   ```

   Result: exit 0.

## Implementation Notes

- `apply_corruptions` uses a fresh CPU-local `torch.Generator` per invocation and returns untouched cloned references, corrupted validity masks, and exact dynamic changed-point masks.
- Implemented joint dropout, temporal block dropout, spike noise, random-walk drift, thorax-centered axial rotation bias, freeze segment, and integer time shift.
- `write_corruption_manifest` writes sorted, deterministic per-window seeds and a JSON-normalized config for fixed evaluation replay.
- `build_split_manifest` reads only `person_id`/person membership from existing fold JSON and rejects train/validation/test overlap.
- Windows default to length 128, train stride 32, eval stride 64; short windows are zero padded, their validity is false, and `loss_mask` excludes padding and either-view-invalid joints.

## Self-Review

- Confirmed no hard-coded MHR70 joint count in the new production modules.
- Confirmed all corruption families derive their masks from actual changed outputs or validity changes, rather than from sampled intent.
- Confirmed integer shifts cannot alter originally invalid targets, keeping masks compatible with reference-validity masking.
- Confirmed no labels are read or propagated from fold JSON entries.
- Confirmed only the four Task 4 code/test files and this report will be staged; pre-existing untracked `.superpowers` files remain untouched.

## Concerns

- Thorax rotation accepts a `thorax_joint_index` supplied by its caller. The training/configuration integration should derive that index from `SkeletonSpec.role("thorax")` when it assembles the corruption config.
- The dataset consumes already-loaded cached `PosePairTrial` records; cache discovery/loading remains intentionally owned by the reviewed Task 1 cache adapter.

## Post-Commit Lint Cleanup

Controller verification identified Ruff `F401` for the unused `Mapping` import in `fuse/rotation_aware/corruptions.py:9`. Removed only that import; no behavior changed.

1. Focused Ruff:

   ```bash
   conda run -n gymnastic ruff check fuse/rotation_aware/corruptions.py
   ```

   Result: `All checks passed!`

2. Focused Task 4 pytest:

   ```bash
   conda run -n gymnastic python -m pytest tests/rotation_aware/test_corruptions.py tests/rotation_aware/test_dataset.py -q
   ```

   Result: `15 passed in 1.17s`.

## Review Requested Changes

### RED Evidence

After adding the review regression tests, ran:

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_corruptions.py tests/rotation_aware/test_dataset.py -q
```

Result: `13 failed, 5 passed in 1.24s`. The failures demonstrated that `apply_corruptions` did not accept `skeleton`, default corruption raised because `thorax_joint_index` was missing, and `PosePairWindowDataset` did not accept or enforce a `SplitManifest`.

### Changes

- Removed the caller-provided `thorax_joint_index` configuration field.
- Added optional `SkeletonSpec` input to `apply_corruptions`; thorax rotation dynamically resolves the `thorax` role for each view and frame. Midpoint roles use their primary pair when valid and their configured fallback pair otherwise.
- Kept generic default calls safe: when no skeleton is supplied, default corruption runs all non-semantic families and does not raise.
- Made `PosePairWindowDataset` require `manifest` and a `train`/`val`/`test` split. It rejects every trial whose person is outside that split rather than silently filtering it.

### GREEN Evidence

1. Focused CPU tests:

   ```bash
   conda run -n gymnastic python -m pytest tests/rotation_aware/test_corruptions.py tests/rotation_aware/test_dataset.py -q
   ```

   Result: `18 passed in 1.20s`.

2. Full rotation-aware CPU tests:

   ```bash
   conda run -n gymnastic python -m pytest tests/rotation_aware -q
   ```

   Result: `56 passed in 1.54s`.

3. Ruff:

   ```bash
   conda run -n gymnastic ruff check fuse/rotation_aware tests/rotation_aware
   ```

   Result: `All checks passed!`

4. Scoped mypy:

   ```bash
   conda run -n gymnastic mypy --follow-imports=skip fuse/rotation_aware/corruptions.py fuse/rotation_aware/dataset.py
   ```

   Result: `Success: no issues found in 2 source files`.

5. Plain mypy was also run:

   ```bash
   conda run -n gymnastic mypy fuse/rotation_aware/corruptions.py fuse/rotation_aware/dataset.py
   ```

   Result: failed with six pre-existing import-following errors in `fuse/save.py` and `fuse/experiment_matrix.py`; neither file is owned by Task 4 and neither was changed.
