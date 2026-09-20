# Workspace Instructions

- For commands that run project code, tests, scripts, or Python tooling in this workspace, use the `gymnastic` conda environment by default, for example `conda run -n gymnastic ...`.
- All code lives under `src/` with `PYTHONPATH=src`. There is no umbrella package: the four pipeline stages are the top-level packages and the only entry points (`python -m pose_estimation`, `python -m cycle_alignment`, `python -m pseudo_gt`, `python -m fusion`); `src/common/` is a shared library and `src/configs/` holds every configuration file, one sub-directory per stage.

## Repository Purpose

This repository is a SAM3D-Body-first pipeline for multi-view gymnastics motion
analysis. Its current research focus is to process face/side gymnastics videos
into 3D keypoints, segment movement cycles, triangulate a pseudo-GT reference,
experiment with multi-view 3D keypoint fusion, and analyse the fused motion
(cohort/repeated-cycle statistics, benchmarks).

In short, the active work is:

```text
two-view gymnastics videos
  -> SAM3D-Body keypoints
  -> temporal alignment and cycle segmentation
  -> triangulated 3D pseudo-GT
  -> face/side/fused 3D keypoint comparison
  -> analysis (metrics, benchmarks, cohort statistics)
```

## Active Pipeline

The current active pipeline is:

```text
/home/data/xchen/gymnastics/raw/person
  -> python -m pose_estimation run
  -> /home/data/xchen/gymnastics/sam3d_body_results/person
  -> python -m cycle_alignment align
  -> python -m pseudo_gt triangulate
  -> /home/data/xchen/gymnastics/sam3d_triangulated/person
  -> python -m fusion deterministic
  -> python -m fusion analyze / python -m fusion cohort-cycle / python -m fusion benchmark-*
```

Important details:

- `pose_estimation` runs SAM3D-Body inference on raw `face` and `side` videos.
- `cycle_alignment` estimates face/side temporal alignment and segments each
  person's motion into cycles.
- `pseudo_gt` uses split-cycle frame records and
  SAM3D 2D keypoints to triangulate 3D joints.
- `fusion` runs the face/side 3D keypoint fusion experiment matrix and evaluates
  each method against triangulated pseudo-GT.
- `fusion.analysis` contains comparison, metrics, reporting, and visualization tools.

## Key Entry Points

Use `conda run -n gymnastic ...` for these commands.

```bash
# Run SAM3D-Body on raw face/side videos.
python -m pose_estimation run

# Segment aligned motion into cycles.
python -m cycle_alignment align

# Add cycle middles (turn-around frames) to the private records / detect
# cycles for FreeMan and Unity. Training never detects cycles itself.
python -m cycle_alignment cycles private
python -m cycle_alignment cycles freeman
python -m cycle_alignment cycles unity
python -m cycle_alignment cycles index      # local/runs/cycle_records/{gymnastics,freeman,unity} + index.json

# Triangulate SAM3D face/side 2D keypoints into pseudo-GT 3D joints.
python -m pseudo_gt triangulate

# Run the fusion experiment matrix.
python -m fusion deterministic

# Metrics, reports, and the out-of-fold cohort analysis.
python -m fusion analyze
python -m fusion cohort-cycle
```

Focused verification commands:

```bash
python -m pytest tests/fusion/baselines/test_experiment_matrix.py -q
python -m pytest tests/pseudo_gt/test_triangulation.py tests/fusion/analysis/test_compare_fused_triangulated.py -q
```

## Module Responsibilities

`src/` is organised as the four pipeline stages, each with its own
``python -m <stage>`` entry point, plus the shared library `common/` and the
configuration tree `configs/`:

| Stage | Package | Role | Command |
|---|---|---|---|
| ① pose estimation | `src/pose_estimation/` | SAM3D-Body inference on the raw face/side videos; per-view 3D + 2D MHR70 keypoints. The pinned upstream SAM-3D-Body checkout is the submodule `pose_estimation/third_party/sam-3d-body`. | `python -m pose_estimation run` |
| ② cycle alignment | `src/cycle_alignment/` | Side-to-face offset, cycle segmentation with turn-around middles (`cycles.py`, `cycle_records.py`, `annotate_cycles.py`), split-cycle records. | `python -m cycle_alignment align`, `python -m cycle_alignment cycles ...` |
| ③ pseudo ground truth | `src/pseudo_gt/` | Chessboard intrinsics (`calibration.py`), per-person extrinsics (`estimate_extrinsics.py`), triangulation of SAM3D 2D keypoints into the evaluation reference (`sam3d_from_split_cycle.py`). Evaluation-only; training never imports it. | `python -m pseudo_gt calibrate`, `python -m pseudo_gt triangulate` |
| ④ fusion network | `src/fusion/` | **The proposed model**: cycle-aware dual-view fusion (data modules, model, losses, Lightning training, Hydra configs in `src/configs/fusion`). See `docs/cycle_aware_fusion.md`. | `python -m fusion train` |
| support | `src/fusion/keypoints/` | Shared 3D-keypoint representation used by ③, ④, the baselines and the benchmarks: `PosePairTrial`, `SkeletonSpec`, canonical body frame, trunk/quality features, person cache. No model code. | – |
| support | `src/fusion/baselines/` | Deterministic comparison matrix and classical baselines every model is compared against. | `python -m fusion deterministic` |
| support | `src/fusion/benchmarks/` | FreeMan and Unity public benchmarks (adapters, zero-shot and trained evaluation). | `python -m fusion benchmark-{freeman,freeman-train,unity}` |
| support | `src/fusion/analysis/` | Metrics, reports, cohort/repeated-cycle statistics, paper result tables. | `python -m fusion analyze`, `python -m fusion cohort-cycle` |
| support | `src/common/` | Project paths, config helpers, MHR70 metadata, the shared CLI dispatcher. Library only, no entry point. | – |
| config | `src/configs/` | `pose_estimation/`, `pseudo_gt/`, `fusion/` (Hydra tree of the model), `shared/` (MHR70 skeleton spec, fold files), `benchmarks/`, `analysis/`, `archive/` (old-model configs). | – |
| archive | `src/fusion/archive/rotation_aware/` | Frozen paper model (2026-09-19); kept only to regenerate published tables. See `archive/README.md`. | `python -m fusion rotation-aware` |
| – | `pegasus/` | All NQSV job scripts, named by stage: `fusion_*` / `submit_fusion_*` (the model, 5-fold), `archive_*` (old model), `benchmark_freeman_subject_qsub.sh` (SAM3D on FreeMan); see `pegasus/README.md`. | – |
| – | `local/` | Ignored checkpoints, videos, run outputs, and caches. | – |

## Current Fuse Direction

The current preferred fusion method is:

```text
avg_body_current
```

Current fuse behavior:

- Discover persons from `/home/data/xchen/gymnastics/sam3d_body_results/person`.
- Require split-cycle alignment records from `local/runs/split_cycle/person_<id>/alignment_record_<id>.json`.
- Use `offset_side_to_face` from split-cycle; do not fall back to a newly
  estimated keypoint-DTW offset.
- Use face as the reference view.
- Align side to face with Sim3 estimated from stable joints.
- Average face and aligned-side 3D keypoints.
- Smooth the fused 3D keypoints over time.
- Save compact outputs under `local/runs/fuse_experiments/<method>/person_<id>/fused_sequence.npz`.
- Evaluate against `/home/data/xchen/gymnastics/sam3d_triangulated/person`.

## Gymnastics Dataset Inventory

The active gymnastics dataset root is:

```text
/home/data/xchen/gymnastics
```

### Main Pipeline Data

| Type | Path | Coverage | Notes |
|---|---|---:|---|
| Raw two-view videos | `/home/data/xchen/gymnastics/raw/person` | 137 persons | Each person has `IDxx_face.MOV` and `IDxx_side.MOV`. |
| SAM3D-Body results | `/home/data/xchen/gymnastics/sam3d_body_results/person` | 137 persons | Each person has complete `face/*.npz` and `side/*.npz` SAM3D outputs. |
| Split-cycle alignment | `local/runs/split_cycle` | 137 persons | Active alignment records used by fuse and triangulation. |
| Triangulated pseudo-GT | `/home/data/xchen/gymnastics/sam3d_triangulated/person` | 137 persons | Evaluation reference for fuse; currently 928 cycle sequences. |
| Fuse experiments | `local/runs/fuse_experiments` | 137 persons x 9 methods | Contains compact fused 3D keypoints and metrics; `metrics_by_person.csv` has no NaN. |
| Rotation-aware runs | `local/runs/fuse_rotation_aware` | 137 persons, 928 cycles | A4/A5/A6 checkpoints, inference, and A0-A6 evaluation. |

Current key counts:

```text
raw/person:                  137 persons, 2 videos per person
sam3d_body_results/person:   137 persons, face/side complete
local/runs/split_cycle:            137 persons, alignment_record complete
sam3d_triangulated/person:   137 persons, 928 cycles
local/runs/fuse_experiments:       137 persons x 9 methods, 1233 fused sequences
local/runs/fuse_rotation_aware:    137 persons, 928 cycles per run, A0-A6 evaluated
```

The 137 persons are 80 elderly participants and 57 students.

### Active Research Flow

```text
raw face/side videos
  -> SAM3D-Body face/side keypoints
  -> split_cycle alignment and cycle segmentation
  -> triangulated pseudo-GT
  -> fuse experiment matrix
```

Fuse should use the split-cycle alignment offset from:

```text
local/runs/split_cycle/person_<id>/alignment_record_<id>.json
```

The current recommended fuse method is:

```text
avg_body_current
```

This method maps both views into a pelvis-centred, rotation-normalised body frame,
averages them there, and maps the result back into the face view's world frame.

It was selected on the regenerated triangulated pseudo-ground-truth (mean person
MPJPE 64.05 mm, better than every other leakage-free method on 69-100% of the 137
people, Holm-corrected Wilcoxon p < 1e-4). Note that `sim3_face_stable_joint_weight`
scores lower still (63.48 mm) but derives its per-joint weights from the
triangulated GT it is then evaluated against, so its number is optimistically
biased and it is not a valid recommendation.

### Historical Or Secondary Data

| Path | Approx. Size | Notes |
|---|---:|---|
| `/home/data/xchen/gymnastics/run_data` | 323G | Older run directory with previous SAM3D/Mediapipe-style outputs. |
| `/home/data/xchen/gymnastics/bak` | 140G | Backup data/results; flagged for deletion. |
| `local/archive/classification_removed_2026-09-19/` | 33G | Archived outputs of the removed motion-classification task (`train/`, `total_5_class/`); see its README. Nothing in the pipeline reads them. |
| `local/runs/calibration_vis` | 977M | Camera calibration parameters and visualizations. |

## Model Policy (2026-09-19)

All new modelling work uses only the cycle-aware architecture
(`fusion`, `python -m fusion train`). The
rotation-aware model was moved to `fusion.archive.rotation_aware`
and is frozen: it is kept solely to regenerate the Sports Engineering paper
artefacts (ablations A0–A11, B1/B2, FreeMan zero-shot and subject-disjoint
rows, cohort OOF). Do not add experiments, losses or configs to it; bug fixes
go to `fusion.keypoints` when they concern the shared infrastructure.

The deterministic `python -m fusion deterministic` experiment matrix (including
the classical baselines in `classical_baselines.py`) remains the comparison
suite for every model.

## Archived Rotation-Aware Model (paper reproduction only)

The archived method uses only SAM3D face/side 3D keypoints and the
split-cycle alignment offset during training. Triangulated 3D keypoints are
loaded only by the evaluation layer and are never used for pseudo-targets,
fusion weights, checkpoint selection, or training losses.

Reproduce the paper runs with:

```bash
conda run -n gymnastic python -m fusion rotation-aware prepare --config src/configs/archive/rotation_aware.yaml
conda run -n gymnastic python -m fusion rotation-aware train --config src/configs/archive/rotation_aware.yaml --run-id paper_a6 --ablation A6
conda run -n gymnastic python -m fusion rotation-aware infer --config src/configs/archive/rotation_aware.yaml --run-id paper_a6
conda run -n gymnastic python -m fusion rotation-aware evaluate --config src/configs/archive/rotation_aware.yaml --run-id paper_a6
```

Train A4, A5, and A6 under separate run IDs, then combine them with repeated
`--run-id` options:

```bash
conda run -n gymnastic python -m fusion rotation-aware evaluate --config src/configs/archive/rotation_aware.yaml --run-id paper_a4 --run-id paper_a5 --run-id paper_a6
```

New artifacts are isolated under `local/runs/fuse_rotation_aware/{cache,runs,inference,evaluation}`.
Do not write rotation-aware training outputs into `local/runs/fuse_experiments`.
