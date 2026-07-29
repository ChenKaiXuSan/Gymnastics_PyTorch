# Unity Extrinsic Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, run, and report three calibrated learned baselines on the Unity native-3D benchmark.

**Architecture:** Add a focused calibrated-learning layer beside the existing Unity supervised A4--A9 pipeline. Pure model/geometry code is separated from fold-safe orchestration and evaluation, and the paper consumes only validated aggregate artifacts.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, pytest, Ruff, Elsevier CAS LaTeX.

## Global Constraints

- Use Unity native 3D as the only supervised target and evaluation GT.
- Use two direction-transfer folds and seeds 0, 1, and 2.
- Never load held-out or static GT in a training call.
- Use final fixed-epoch checkpoints; do not select on held-out results.
- Keep 3D-to-3D fusion and 2D-to-3D triangulation in separate paper groups.
- Run project Python commands with `conda run -n gymnastic`.

---

### Task 1: Calibrated Model Geometry

**Files:**
- Create: `src/gymnastics/benchmarks/unity/extrinsic_models.py`
- Create: `tests/unity_benchmark/test_extrinsic_models.py`

**Interfaces:**
- Produces: `relative_camera_rotation(benchmark) -> np.ndarray`
- Produces: `ExtrinsicGateModel`, `ExtrinsicResidualTCN`, and `LearnableTriangulationModel`
- Produces: `CalibratedPrediction(points, valid, diagnostics)`

- [ ] **Step 1: Write failing geometry and model-contract tests**

Test literal 90-degree rotation, validity-aware gate endpoints, 50 mm residual
bound, exact noise-free DLT recovery, and zero confidence/invalid-view
rejection.

- [ ] **Step 2: Run the new test module and verify RED**

Run:
`conda run -n gymnastic python -m pytest tests/unity_benchmark/test_extrinsic_models.py -q`

Expected: import failure because `extrinsic_models.py` does not exist.

- [ ] **Step 3: Implement the minimal calibrated models**

Use pelvis-centred row-vector rotation for monocular 3D and batched SVD for
weighted homogeneous DLT. Gate outputs use sigmoid; residuals use
`0.05 * tanh(raw_delta)`.

- [ ] **Step 4: Run the model tests and existing Unity geometry tests**

Run:
`conda run -n gymnastic python -m pytest tests/unity_benchmark/test_extrinsic_models.py tests/unity_benchmark/test_geometry.py -q`

Expected: all pass.

### Task 2: Leakage-Safe Training and Checkpoints

**Files:**
- Create: `src/gymnastics/benchmarks/unity/extrinsic_training.py`
- Create: `tests/unity_benchmark/test_extrinsic_training.py`
- Modify: `configs/benchmarks/unity_supervised.yaml`

**Interfaces:**
- Consumes: the three models and existing `UnitySupervisedSequence`
- Produces: `ExtrinsicTrainingConfig`, `ExtrinsicRun`, `train_extrinsic_run`
- Produces: `validate_extrinsic_run`, `run_extrinsic_inference`

- [ ] **Step 1: Write failing fold-isolation, loss, and checkpoint tests**

Construct a training sequence with known sample IDs and assert that only the
declared train direction reaches the dataset, final checkpoints preserve model
type/seed/fold, and corrupted hashes invalidate a completed run.

- [ ] **Step 2: Run the tests and verify RED**

Run:
`conda run -n gymnastic python -m pytest tests/unity_benchmark/test_extrinsic_training.py -q`

Expected: import failure because training orchestration is absent.

- [ ] **Step 3: Implement fixed-epoch CPU/GPU training**

Use AdamW, deterministic seed setup, masked Unity16 Smooth-L1, optional
window-level differentiable Sim3 for 3D-fusion models, atomic artifacts, and
strict provenance.

- [ ] **Step 4: Run training tests and supervised regression tests**

Run:
`conda run -n gymnastic python -m pytest tests/unity_benchmark/test_extrinsic_training.py tests/unity_benchmark/test_supervised_data.py tests/unity_benchmark/test_supervised_loss.py -q`

Expected: all pass.

### Task 3: Evaluation, Reporting, and CLI

**Files:**
- Create: `src/gymnastics/benchmarks/unity/extrinsic_evaluation.py`
- Create: `tests/unity_benchmark/test_extrinsic_evaluation.py`
- Modify: `src/gymnastics/benchmarks/unity/cli.py`
- Modify: `tests/unity_benchmark/test_supervised_cli.py`

**Interfaces:**
- Consumes: validated `ExtrinsicRun` artifacts and existing common evaluator
- Produces: per-run, per-fold, per-method, and static diagnostic CSV/JSON files
- Produces: CLI stages `extrinsic-train`, `extrinsic-infer`, `extrinsic-evaluate`

- [ ] **Step 1: Write failing aggregation and CLI tests**

Require an exact 2-fold x 3-seed matrix, separate input-regime labels, stable
ranking order, and CLI forwarding without hidden GT access.

- [ ] **Step 2: Run the tests and verify RED**

Run:
`conda run -n gymnastic python -m pytest tests/unity_benchmark/test_extrinsic_evaluation.py tests/unity_benchmark/test_supervised_cli.py -q`

- [ ] **Step 3: Implement evaluation and stage orchestration**

Reuse `evaluate_method_sequence`; write atomically to
`local/runs/unity_benchmark/extrinsic_learning`.

- [ ] **Step 4: Run all Unity benchmark tests**

Run:
`conda run -n gymnastic python -m pytest tests/unity_benchmark -q`

Expected: all pass.

### Task 4: Execute the Full Matrix

**Files:**
- Generate only ignored artifacts below: `local/runs/unity_benchmark/extrinsic_learning`

**Interfaces:**
- Consumes: 199 immutable Unity samples and existing SAM3D caches
- Produces: 18 validated cells, inference sequences, aggregate tables, and a result report

- [ ] **Step 1: Run two folds x three methods x three seeds**

Use the configured fixed epoch count on CPU when CUDA is unavailable.

- [ ] **Step 2: Validate every artifact and evaluate**

Reject incomplete cells, non-finite metrics, sample-identity mismatch, or
unexpected run directories.

- [ ] **Step 3: Recompute aggregates from raw per-run outputs**

Verify 2 folds, 3 seeds, 6 held-out runs per method and report seed dispersion.

### Task 5: Consolidate and Review the IVC Paper

**Files:**
- Modify: `paper/image_and_vision_computing/scripts/generate_paper_assets.py`
- Modify: `paper/image_and_vision_computing/scripts/check_manuscript.py`
- Modify: `paper/image_and_vision_computing/sections/05_experimental_protocol.tex`
- Modify: `paper/image_and_vision_computing/sections/06_results.tex`
- Modify: `paper/image_and_vision_computing/sections/07_discussion.tex`
- Modify: `paper/image_and_vision_computing/sections/08_limitations.tex`
- Modify: `paper/image_and_vision_computing/sections/09_conclusion.tex`
- Create: grouped generated tables/figures under the paper directory

**Interfaces:**
- Consumes: validated real-data, Unity zero-shot, supervised, extrinsic, robustness, and cohort result artifacts
- Produces: one result-family overview and grouped evidence tables

- [ ] **Step 1: Inventory every numerical claim and its source artifact**

Mark completed evidence, diagnostics, and genuinely pending analyses.

- [ ] **Step 2: Reorganize the Results section by research question**

Lead with the primary claim, keep camera-information groups explicit, place
Unity and triangulation together as native-GT validation, and move diagnostics
after headline results.

- [ ] **Step 3: Run manuscript generation and checks**

Run:
`conda run -n gymnastic python paper/image_and_vision_computing/scripts/generate_paper_assets.py`

Run:
`conda run -n gymnastic python paper/image_and_vision_computing/scripts/check_manuscript.py`

- [ ] **Step 4: Compile and visually inspect the PDF**

Run:
`make -C paper/image_and_vision_computing all`

Inspect every results/table page for overflow, ordering, caption clarity, and
cross-reference correctness.

- [ ] **Step 5: Run reader/reviewer audit and revise blocking issues**

Audit contribution clarity, fairness, leakage, GT validity, statistical unit,
claim-evidence alignment, limitations, and reproducibility. Record residual
non-blocking concerns in a review report.

### Task 6: Final Verification and Integration

**Files:**
- Verify all touched source, tests, docs, and ignored paper artifacts

- [ ] **Step 1: Run focused and regression tests**

Run:
`conda run -n gymnastic python -m pytest tests/unity_benchmark tests/rotation_aware tests/test_fuse_experiment_matrix.py tests/freeman_benchmark -q`

- [ ] **Step 2: Run Ruff and diff checks**

Run:
`conda run -n gymnastic ruff check src/gymnastics/benchmarks/unity tests/unity_benchmark`

Run: `git diff --check`

- [ ] **Step 3: Integrate without deleting newer master work**

Merge the Unity branch into master, resolve CLI/docs conflicts by retaining
both Unity and FreeMan functionality, and preserve unrelated dirty files.

