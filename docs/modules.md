# Module Map

This project is now organized around a SAM3D-Body-first data preparation flow.

## Active Pipeline Modules

### `SAM3Dbody/`

Runs SAM3D-Body inference on raw multi-view videos. The active entry point is:

```bash
python -m SAM3Dbody.main
```

Input: `/home/data/xchen/gymnastics/raw/person`

Output: `/home/data/xchen/gymnastics/sam3d_body_results/person`

Configuration: `configs/sam3d_body.yaml`

### `fuse/`

Runs the default fusion experiment matrix for face and side view SAM3D-Body
results. The matrix rebuilds temporal alignment, writes fused 3D keypoints, and
exports metrics.

Entry point:

```bash
python -m fuse
```

### `split_cycle/`

Segments fused motion sequences into individual action cycles.

Entry point:

```bash
python -m split_cycle.main
```

### `triangulation/sam3d_from_split_cycle.py`

Uses `split_cycle/person_<id>/alignment_record_<id>.json` to align face and side
SAM3D-Body frame indices, then triangulates `pred_keypoints_2d` from:

```text
sam3d_body_results/person/<id>/face/*.npz
sam3d_body_results/person/<id>/side/*.npz
```

Output:

```text
sam3d_triangulated/person/person_<id>/cycle_<idx>/
```

Configuration: `configs/sam3d_triangulation.yaml`

Central guide: `triangulation/README.md`

Report tooling: `triangulation/tools/generate_results_report.py`

### `project/train/`

Contains dataloaders, models, trainers, and evaluation logic for classification
tasks.

Entry point:

```bash
python -m project.train.train
```

Configuration: `configs/train.yaml`

### `analysis/`

Contains analysis, metrics, comparison, and visualization scripts/notebooks.

## Support Modules

### `triangulation/`

Support path for 3D pose triangulation from multi-view 2D keypoints.

Configuration: `configs/triangulation.yaml`

See `triangulation/README.md` for the active SAM3D triangulation workflow,
output structure, result reports, related consumers, and focused tests.

### `camera_calibration/`

Camera calibration utilities and scripts.

### `videopose3d/`

VideoPose3D code and utilities. This is currently a support/legacy-style module
rather than the primary data preparation path.

## Legacy Modules

### `legacy/prepare_dataset/`

Previous preprocessing pipeline based on DPT depth, RAFT optical flow, YOLOv11,
and Detectron2. It is kept for reference and reuse but is no longer part of the
active SAM3D-Body-only preparation flow.

Configuration: `configs/legacy/prepare_dataset.yaml`
