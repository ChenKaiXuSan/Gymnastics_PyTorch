# Workspace Instructions

- For commands that run project code, tests, scripts, or Python tooling in this workspace, use the `gymnastic` conda environment by default, for example `conda run -n gymnastic ...`.

## Repository Purpose

This repository is a SAM3D-Body-first pipeline for multi-view gymnastics motion
analysis. Its current research focus is to process face/side gymnastics videos
into 3D keypoints, segment movement cycles, triangulate a pseudo-GT reference,
experiment with multi-view 3D keypoint fusion, and train/evaluate motion
classification models.

In short, the active work is:

```text
two-view gymnastics videos
  -> SAM3D-Body keypoints
  -> temporal alignment and cycle segmentation
  -> triangulated 3D pseudo-GT
  -> face/side/fused 3D keypoint comparison
  -> classification and analysis
```

## Active Pipeline

The current active pipeline is:

```text
/home/data/xchen/gymnastics/raw/person
  -> gymnastics sam3d
  -> /home/data/xchen/gymnastics/sam3d_body_results/person
  -> gymnastics align
  -> gymnastics triangulate
  -> /home/data/xchen/gymnastics/sam3d_triangulated/person
  -> gymnastics fuse deterministic
  -> gymnastics classify and gymnastics analyze
```

Important details:

- `gymnastics.sam3d` runs SAM3D-Body inference on raw `face` and `side` videos.
- `gymnastics.alignment` estimates face/side temporal alignment and segments each
  person's motion into cycles.
- `gymnastics.triangulation` uses split-cycle frame records and
  SAM3D 2D keypoints to triangulate 3D joints.
- `gymnastics.fusion` runs the face/side 3D keypoint fusion experiment matrix and evaluates
  each method against triangulated pseudo-GT.
- `gymnastics.classification` trains classification models from prepared motion data and
  precomputed fold/index mappings.
- `gymnastics.analysis` contains comparison, metrics, reporting, and visualization tools.

## Key Entry Points

Use `conda run -n gymnastic ...` for these commands.

```bash
# Run SAM3D-Body on raw face/side videos.
gymnastics sam3d

# Segment aligned motion into cycles.
gymnastics align

# Triangulate SAM3D face/side 2D keypoints into pseudo-GT 3D joints.
gymnastics triangulate

# Run the fusion experiment matrix.
gymnastics fuse deterministic

# Train/evaluate classification models.
gymnastics classify
```

Focused verification commands:

```bash
python -m pytest tests/test_fuse_experiment_matrix.py -q
python -m pytest tests/test_sam3d_triangulation.py tests/test_compare_fused_triangulated.py -q
```

## Module Responsibilities

| Module | Current Role |
|---|---|
| `src/gymnastics/sam3d/` | SAM3D-Body inference and keypoint extraction from raw videos. |
| `src/gymnastics/alignment/` | Face/side time alignment, audio/keypoint offset selection, cycle segmentation, and split-cycle videos. |
| `src/gymnastics/triangulation/` | 3D triangulation from SAM3D 2D keypoints, camera helpers, and visualizations. |
| `src/gymnastics/fusion/` | Deterministic and rotation-aware multi-view fusion. |
| `src/gymnastics/classification/` | Person-level splits and ST-GCN/TCN/Mamba-style classification. |
| `src/gymnastics/analysis/` | Label analysis, metric comparison, plotting, reports, and result inspection. |
| `src/gymnastics/calibration/` | Camera calibration utilities. |
| `src/gymnastics/common/` | Shared paths and canonical MHR70 metadata. |
| `legacy/prepare_dataset/` | Old DPT/RAFT/YOLO/Detectron2 preprocessing path, kept for reference. |
| `third_party/` | Pinned upstream SAM3 and SAM-3D-Body repositories. |
| `local/` | Ignored checkpoints, videos, run outputs, and caches. |

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

### Labels And Splits

| Type | Path | Notes |
|---|---|---|
| Label table | `/home/data/xchen/gymnastics/suwabe_label.xlsx` | Label spreadsheet. |
| Label table | `/home/data/xchen/gymnastics/高齢者体回旋20260513.xlsx` | Japanese-named label/score spreadsheet. |
| Person-level folds | `/home/data/xchen/gymnastics/index_mapping/camera_pairs_by_person_folds` | `fold_00.json`, `fold_01.json`, `fold_02.json`; split by person with train/val/test ratio `7/2/1`. |
| Label analysis | `/home/data/xchen/gymnastics/label_analysis_output` | Label analysis outputs. |
| 5-class label analysis | `/home/data/xchen/gymnastics/5_classes_label_analysis_output` | Five-class label analysis outputs. |

The fold files are regenerated by `python -m gymnastics.classification.splits.main_5_classes`.
They read alignment records from `local/runs/split_cycle` and fused frame maps from
`local/runs/fuse_experiments/<method>/person_<id>/fused_sequence.npz`.

### Historical Or Secondary Data

| Path | Approx. Size | Notes |
|---|---:|---|
| `/home/data/xchen/gymnastics/run_data` | 323G | Older run directory with previous SAM3D/Mediapipe-style outputs. |
| `/home/data/xchen/gymnastics/bak` | 140G | Backup data/results; flagged for deletion. |
| `local/runs/train` | 8.5G | Training logs and outputs. |
| `local/runs/total_5_class` | 25G | Five-class experiment outputs. |
| `local/runs/calibration_vis` | 977M | Camera calibration parameters and visualizations. |

## Rotation-Aware Paper Mainline

The deterministic `gymnastics fuse deterministic` experiment matrix remains the comparison
suite. The paper mainline is the isolated, self-supervised method:

```text
rotation_aware_self_supervised
```

It uses only SAM3D face/side 3D keypoints and the split-cycle alignment offset
during training. Triangulated 3D keypoints are loaded only by the evaluation
layer and are never used for pseudo-targets, fusion weights, checkpoint
selection, or training losses.

Run the mainline with:

```bash
conda run -n gymnastic gymnastics fuse rotation-aware prepare --config configs/fusion/rotation_aware.yaml
conda run -n gymnastic gymnastics fuse rotation-aware train --config configs/fusion/rotation_aware.yaml --run-id paper_a6 --ablation A6
conda run -n gymnastic gymnastics fuse rotation-aware infer --config configs/fusion/rotation_aware.yaml --run-id paper_a6
conda run -n gymnastic gymnastics fuse rotation-aware evaluate --config configs/fusion/rotation_aware.yaml --run-id paper_a6
```

Train A4, A5, and A6 under separate run IDs, then combine them with repeated
`--run-id` options:

```bash
conda run -n gymnastic gymnastics fuse rotation-aware evaluate --config configs/fusion/rotation_aware.yaml --run-id paper_a4 --run-id paper_a5 --run-id paper_a6
```

New artifacts are isolated under `local/runs/fuse_rotation_aware/{cache,runs,inference,evaluation}`.
Do not write rotation-aware training outputs into `local/runs/fuse_experiments`.
