# Triangulation

This directory is the central entry point for SAM3D-Body triangulation work.
It contains the code that triangulates face/side 2D keypoints into 3D joints,
the supporting visualization helpers, and the reporting tools for generated
triangulated results.

## Main Workflow

Triangulate SAM3D-Body 2D keypoints using cycle-level face/side alignment
records from `split_cycle`:

```bash
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle
```

Quick smoke test on one person/cycle:

```bash
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle --person 1 --max-cycles 1 --max-frames 2
```

The legacy/support triangulation entry point is:

```bash
conda run -n gymnastic python -m triangulation.main
```

## Inputs

The SAM3D split-cycle triangulation path uses:

```text
/home/data/xchen/gymnastics/sam3d_body_results/person/<id>/face/*.npz
/home/data/xchen/gymnastics/sam3d_body_results/person/<id>/side/*.npz
logs/split_cycle/person_<id>/alignment_record_<id>.json
configs/sam3d_triangulation.yaml
```

The older support entry point uses `configs/triangulation.yaml`.

## Outputs

The generated triangulated dataset is written outside the repo:

```text
/home/data/xchen/gymnastics/sam3d_triangulated/person
```

Each processed cycle is stored as:

```text
person_<id>/cycle_<idx>/summary.json
person_<id>/cycle_<idx>/joints_3d/*.json
person_<id>/cycle_<idx>/joints_3d_sequence.npz
person_<id>/cycle_<idx>/visualization/*.png
person_<id>/cycle_<idx>/cycle_<idx>_3d.mp4
```

`joints_3d_sequence.npz` contains:

```text
joints_3d.npy
frame_records.npy
```

Camera pose visualizations are stored under:

```text
/home/data/xchen/gymnastics/sam3d_triangulated/person/_camera
```

## Result Reports

Generate a consolidated quality report and CSV details:

```bash
conda run -n gymnastic python triangulation/tools/generate_results_report.py
```

The report files are written to:

```text
logs/analysis/triangulated_results/
```

Current report artifacts:

```text
logs/analysis/triangulated_results/triangulated_results_report.md
logs/analysis/triangulated_results/triangulated_cycle_details.csv
logs/analysis/triangulated_results/triangulated_person_summary.csv
```

## Related Code

- `sam3d_from_split_cycle.py`: active SAM3D-Body face/side triangulation path.
- `main.py`: older support triangulation path.
- `camera_position_mapping.py`: camera calibration and pose selection helpers.
- `load.py`: keypoint loading utilities.
- `save.py`: 3D joint output helpers.
- `vis/`: 3D pose and frame/video visualization helpers.
- `tools/`: reporting and maintenance utilities for triangulated outputs.

## Related Consumers

- `analysis/compare_fused_triangulated.py` compares face/side/fused SAM3D-Body
  3D keypoints against the triangulated reference.
- `fuse/experiment_matrix.py` can evaluate fusion variants against the
  triangulated dataset.

## Tests

Focused tests:

```bash
conda run -n gymnastic python -m pytest tests/test_sam3d_triangulation.py tests/test_compare_fused_triangulated.py
```
