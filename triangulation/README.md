# Triangulation

This directory is the central entry point for SAM3D-Body triangulation work.
It contains the code that triangulates face/side 2D keypoints into 3D joints,
the supporting visualization helpers, and the reporting tools for generated
triangulated results.

## Camera Extrinsics

Intrinsics come from chessboard calibration, but the extrinsics used to be taken
from a synthetic layout in `configs/sam3d_triangulation.yaml` (four cameras on a
3.5 m circle, face/side assumed exactly 90 deg apart with a 4.95 m baseline)
shared by every person. The rig was in fact re-positioned between sessions, so
that single assumed pose fits nobody well: held-out reprojection error runs about
25 px on average and reaches 58 px on the worst people.

Estimate the real per-person extrinsics from the SAM3D correspondences first:

```bash
conda run -n gymnastic python -m triangulation.estimate_extrinsics
```

This writes `logs/analysis/extrinsics/estimated_extrinsics.json`, which
`sam3d_from_split_cycle` picks up via the `extrinsics` block in the config. Pass
`--assumed-extrinsics` to that script to force the legacy layout instead.

Compare the two sources on reprojection error, shape error against SAM3D's own
monocular 3D, and bone-length stability:

```bash
conda run -n gymnastic python triangulation/tools/compare_extrinsics.py
```

## Metric Scale

Two-view geometry is scale-free, so reprojection error cannot detect a wrong
baseline length: scaling the baseline scales the whole reconstruction and leaves
every projection untouched (locked in by
`tests/test_estimate_extrinsics.py::test_reprojection_error_is_blind_to_baseline_scale`).
Metric scale therefore rests entirely on SAM3D's monocular 3D, which this module
rescales from SAM3D's assumed focal length to the calibrated one.

What that does and does not affect:

* **Method rankings and relative comparisons are unaffected.** `fuse` measures
  error after a per-sequence Sim3 fit, which absorbs any uniform scale error.
* **Absolute millimetres are proportional to it.** If the true scale is `s` times
  the current one, every reported MPJPE is off by the same `s`.

Two ways to handle it without a calibration target:

```bash
# Report errors as a fraction of each subject's own body size -- cancels s exactly.
conda run -n gymnastic python analysis/normalize_by_body_scale.py
```

That tool also prints the recovered limb-chain lengths and the implied `s` for a
range of candidate true statures, which bounds the scale error from anthropometry
alone.

To pin `s` down properly, the recordings were made on a matted hall floor whose
seams form a regular grid in both views. Judo tatami are manufactured to
standard dimensions, so measuring one mat edge in a frame gives a direct metric
reference; that is the cheapest route to a calibrated scale if it is needed.

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
- `estimate_extrinsics.py`: recovers per-person face/side extrinsics from the data.
- `main.py`: older support triangulation path.
- `camera_position_mapping.py`: synthetic camera layout, used for the pose figures
  and as the fallback when `extrinsics.source` is `assumed`.
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
conda run -n gymnastic python -m pytest tests/test_sam3d_triangulation.py \
    tests/test_estimate_extrinsics.py tests/test_compare_fused_triangulated.py
```
