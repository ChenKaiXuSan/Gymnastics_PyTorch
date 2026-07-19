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
