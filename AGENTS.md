# Workspace Instructions

- Run project code, tests and scripts in the project conda environment: `gymnastic` on the lab workstation, `sam_3d_body` on Pegasus/HP260146 (`direction` also works for fusion training; there is no `gymnastic` env there). Commands in the docs are written as plain `python -m ...`; prefix them with `conda run -n <env>` or activate the env first, and set `PYTHONPATH=src` when the package is not pip-installed.
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
$GYMNASTICS_DATA_ROOT/raw/person
  -> python -m pose_estimation run
  -> $GYMNASTICS_DATA_ROOT/sam3d_body_results/person
  -> python -m cycle_alignment align
  -> python -m pseudo_gt triangulate
  -> $GYMNASTICS_DATA_ROOT/sam3d_triangulated/person
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

Run these from the repository root with `PYTHONPATH=src` in the project environment (see the first bullet above).

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
python -m cycle_alignment cycles sportspose   # after `python -m fusion benchmark-sportspose infer`
python -m cycle_alignment cycles index      # local/runs/cycle_records/{gymnastics,freeman,unity,sportspose} + index.json

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
| support | `src/fusion/baselines/` | Deterministic comparison matrix and classical baselines every model is compared against (incl. the calibration-free depth-aware rule `avg_body_depthaware`). | `python -m fusion deterministic` |
| support | `src/fusion/external/` | External learned baselines (VideoPose3D-style TCN, SmoothNet, MetaPose-style MLP, MUC-style weights) on the model's input/output contract, trained with the same folds, windows and losses; never zero-shot. | `python -m fusion train model=external_<name>` |
| support | `src/fusion/benchmarks/` | FreeMan, Unity and SportsPose public benchmarks (adapters, view selection, SAM3D caches, zero-shot and trained evaluation). | `python -m fusion benchmark-{freeman,freeman-train,unity,sportspose}` |
| support | `src/fusion/analysis/` | Metrics, reports, cohort/repeated-cycle statistics, paper result tables. | `python -m fusion analyze`, `python -m fusion cohort-cycle` |
| support | `src/common/` | Project paths, config helpers, MHR70 metadata, the shared CLI dispatcher. Library only, no entry point. | – |
| config | `src/configs/` | `pose_estimation/`, `pseudo_gt/`, `fusion/` (Hydra tree of the model), `shared/` (MHR70 skeleton spec, fold files), `benchmarks/`, `analysis/`, `archive/` (old-model configs). | – |
| archive | `src/fusion/archive/rotation_aware/` | Frozen paper model (2026-09-19); kept only to regenerate published tables. See `archive/README.md`. | `python -m fusion rotation-aware` |
| – | `pegasus/` | All NQSV job scripts, named by stage: `fusion_*` / `submit_fusion_*` (the model, 5-fold), `archive_*` (old model), `benchmark_freeman_subject_qsub.sh` (SAM3D on FreeMan); see `pegasus/README.md`. | – |
| – | `local/` | Ignored checkpoints, videos, run outputs, and caches. | – |

## Current Fuse Direction

The current preferred deterministic fusion method is:

```text
avg_body_depthaware      (2026-09-21; calibration-free depth-aware body average, alpha 0.8)
```

It replaces `avg_body_current` (the plain body-frame average, kept as the
control): each view's precision is discounted along its own camera optical
axis, obtained from the view's own canonicalisation rotation. Selected on the
FreeMan 8-camera reference (alpha 0.75–0.875 flat), never on the private
triangulated reference, which is built from the same image-plane coordinates
and rewards this rule far beyond its true gain (see
`docs/cycle_aware_fusion.md` §1.5). The learned model (`python -m fusion train`,
architecture v1.1) uses the same rule as its base pose. Everything below
describes the `avg_body_current` pipeline the rule shares.

The previous preferred fusion method was:

```text
avg_body_current
```

Current fuse behavior:

- Discover persons from `$GYMNASTICS_DATA_ROOT/sam3d_body_results/person`.
- Require split-cycle alignment records from `local/runs/split_cycle/person_<id>/alignment_record_<id>.json`.
- Use `offset_side_to_face` from split-cycle; do not fall back to a newly
  estimated keypoint-DTW offset.
- Use face as the reference view.
- Align side to face with Sim3 estimated from stable joints.
- Average face and aligned-side 3D keypoints.
- Smooth the fused 3D keypoints over time.
- Save compact outputs under `local/runs/fuse_experiments/<method>/person_<id>/fused_sequence.npz`.
- Evaluate against `$GYMNASTICS_DATA_ROOT/sam3d_triangulated/person`.

## Gymnastics Dataset Inventory

The private dataset root is defined once, in `src/common/paths.py`
(`DATA_ROOT`). It is taken from the `GYMNASTICS_DATA_ROOT` environment
variable when set; otherwise the first existing known machine root is used
(`/work/1/HP260146/chenkaixu/gymnastics` on HP260146, `/home/data/xchen/gymnastics`
on the lab workstation). Importing `common.paths` exports the resolved value back
into the environment, so every `src/configs/**/*.yaml` can interpolate it with
`${oc.env:GYMNASTICS_DATA_ROOT}`. Paths below are written relative to that root
as `$GYMNASTICS_DATA_ROOT`; never hard-code a machine root outside
`common/paths.py` (`tests/structure/test_data_entry_config.py` enforces this).

### Main Pipeline Data

| Type | Path | Coverage | Notes |
|---|---|---:|---|
| Raw two-view videos | `$GYMNASTICS_DATA_ROOT/raw/person` | 137 persons | Each person has `IDxx_face.MOV` and `IDxx_side.MOV`. |
| SAM3D-Body results | `$GYMNASTICS_DATA_ROOT/sam3d_body_results/person` | 137 persons | Each person has complete `face/*.npz` and `side/*.npz` SAM3D outputs. |
| Split-cycle alignment | `local/runs/split_cycle` | 137 persons | Active alignment records used by fuse and triangulation. |
| Triangulated pseudo-GT | `$GYMNASTICS_DATA_ROOT/sam3d_triangulated/person` | 137 persons | Evaluation reference for fuse; currently 928 cycle sequences. |
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

The recommended fuse method is `avg_body_depthaware` (see "Current Fuse
Direction"); the previous recommendation, `avg_body_current`, remains the
control row. All numbers below are the archived matrix protocol (70 joints,
one similarity alignment per cycle); current reporting uses the model protocol
of `docs/cycle_aware_fusion.md` §1.5.

`avg_body_current` maps both views into a pelvis-centred, rotation-normalised body frame,
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
| `$GYMNASTICS_DATA_ROOT/run_data` | 323G | Older run directory with previous SAM3D/Mediapipe-style outputs. |
| `$GYMNASTICS_DATA_ROOT/bak` | 140G | Backup data/results; flagged for deletion. |
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
python -m fusion rotation-aware prepare --config src/configs/archive/rotation_aware.yaml
python -m fusion rotation-aware train --config src/configs/archive/rotation_aware.yaml --run-id paper_a6 --ablation A6
python -m fusion rotation-aware infer --config src/configs/archive/rotation_aware.yaml --run-id paper_a6
python -m fusion rotation-aware evaluate --config src/configs/archive/rotation_aware.yaml --run-id paper_a6
```

Train A4, A5, and A6 under separate run IDs, then combine them with repeated
`--run-id` options:

```bash
python -m fusion rotation-aware evaluate --config src/configs/archive/rotation_aware.yaml --run-id paper_a4 --run-id paper_a5 --run-id paper_a6
```

New artifacts are isolated under `local/runs/fuse_rotation_aware/{cache,runs,inference,evaluation}`.
Do not write rotation-aware training outputs into `local/runs/fuse_experiments`.
