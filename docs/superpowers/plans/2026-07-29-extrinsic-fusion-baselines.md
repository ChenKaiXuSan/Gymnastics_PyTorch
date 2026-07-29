# Extrinsic Fusion Baselines Implementation Plan

**Goal:** Add two leakage-free deterministic fusion baselines that use the
estimated side-to-face camera rotation, evaluate them separately from the
existing methods, and report relevant calibrated deep-learning comparators.

**Design:** The estimated extrinsics follow the triangulation convention
`X_side = R X_face + t`. SAM3D poses are root-relative, so translation is not
applied. Each side pose is pelvis-centred, rotated into the face axes with the
row-vector operation `X_side_centered @ R`, and restored at the face pelvis.
`extrinsic_r_average` averages the two poses equally.
`extrinsic_r_quality_average` uses the repository's existing fixed
rotation-aware quality score to compute per-frame normalized face/side weights.
Neither method reads triangulated 3D keypoints during fusion.

**Isolation:** Run the two new methods under
`local/runs/fuse_extrinsic_baselines`, leaving the verified nine-method CSV and
artifacts unchanged. The paper generator reads both CSVs and emits separate
tables for methods without and with camera extrinsics.

---

### Task 1: Specify behavior with failing tests

**Files:**
- Modify: `tests/test_fuse_experiment_matrix.py`

1. Test that estimated extrinsics are loaded for a requested person and reject
   missing or malformed rotations.
2. Test the rotation convention with a known `R`, including pelvis restoration.
3. Test equal averaging after extrinsic rotation.
4. Test quality weighting, including equal-score fallback and preference for
   the higher-quality view.
5. Run:
   `conda run -n gymnastic python -m pytest tests/test_fuse_experiment_matrix.py -q`
   and confirm failures are caused by the missing new API.

### Task 2: Implement and integrate the two methods

**Files:**
- Modify: `src/gymnastics/fusion/deterministic/experiment_matrix.py`

1. Define `NO_EXTRINSIC_METHODS`, `EXTRINSIC_METHODS`, and combined
   `ALL_METHODS`.
2. Add strict extrinsics JSON loading and SO(3) validation.
3. Add side-to-face rotation alignment, equal fusion, and quality-weighted
   fusion helpers.
4. Reuse `compute_quality_features` and `extract_trunk_features` with
   `configs/fusion/skeleton_mhr70.yaml`.
5. Add CLI options `--extrinsics-path` and `--skeleton-path`.
6. Load these resources only when an extrinsic method is requested.
7. Record `uses_camera_extrinsics`, source path, rotation convention, and
   quality source in each saved config.
8. Run the focused test file until it passes.

### Task 3: Run and validate the external-parameter experiment

1. Run both methods for all discovered participants:

   `conda run -n gymnastic gymnastics fuse deterministic
   --methods extrinsic_r_average extrinsic_r_quality_average
   --out-dir local/runs/fuse_extrinsic_baselines`

2. Verify 274 finite person-method rows, 137 people, two methods, and complete
   metadata.
3. Compare mean person-level MPJPE with the existing nine-method matrix without
   changing the latter.

### Task 4: Split manuscript result tables

**Files:**
- Modify: `paper/image_and_vision_computing/scripts/generate_paper_assets.py`
- Modify: `paper/image_and_vision_computing/scripts/check_manuscript.py`
- Modify: `paper/image_and_vision_computing/sections/06_results.tex`
- Generate: `paper/image_and_vision_computing/tables/deterministic_baselines.tex`
- Generate: `paper/image_and_vision_computing/tables/extrinsic_baselines.tex`

1. Keep the existing nine-method verification invariants.
2. Add a strict loader for the two-method, 274-row extrinsics CSV.
3. Emit independent captions and labels that explicitly state whether camera
   extrinsics are used.
4. Add source-grounded result text after obtaining the actual metrics.
5. Run the paper generator, manuscript checker, and LaTeX build.

### Task 5: Investigate calibrated deep-learning models

1. Search primary papers and official repositories for calibrated multi-view
   3D human pose models.
2. Extract whether each method consumes intrinsics/extrinsics, supported view
   count, supervision, code availability, and expected adaptation cost.
3. Recommend a small shortlist suitable for a two-view, single-person,
   MHR70-keypoint experiment, while separating directly runnable models from
   methods that would require retraining or skeleton adaptation.

### Task 6: Final verification

1. Run focused deterministic tests.
2. Run relevant paper generation/check/build commands.
3. Inspect `git diff --check` and `git status --short`.
4. Report exact metrics, table separation, literature shortlist, limitations,
   and changed files without committing unrelated user changes.
