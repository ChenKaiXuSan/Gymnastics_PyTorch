# Runbook

## Data Root

The default data root is:

```text
/home/data/xchen/gymnastics
```

Override it with:

```bash
export GYMNASTICS_DATA_ROOT=/path/to/gymnastics
```

## Active Pipeline

Run SAM3D-Body dataset preparation:

```bash
conda run -n drivefusion python -m SAM3Dbody.main
```

Run the default multi-view fusion experiment matrix:

```bash
conda run -n drivefusion python -m fuse
```

Run cycle segmentation:

```bash
conda run -n drivefusion python -m split_cycle.main
```

Triangulate SAM3D-Body 2D keypoints using the `split_cycle` face/side alignment
records:

```bash
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle
```

The output is written under:

```text
/home/data/xchen/gymnastics/sam3d_triangulated/person
```

For each person and cycle, the script saves:

```text
person_<id>/cycle_<idx>/joints_3d/*.json
person_<id>/cycle_<idx>/joints_3d_sequence.npz
person_<id>/cycle_<idx>/visualization/*.png
person_<id>/cycle_<idx>/cycle_<idx>_3d.mp4
```

Quick smoke test on one person/cycle:

```bash
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle --person 1 --max-cycles 1 --max-frames 2
```

Regenerate the triangulated-result quality report:

```bash
conda run -n gymnastic python triangulation/tools/generate_results_report.py
```

The consolidated triangulation guide is:

```text
triangulation/README.md
```

Run classifier training:

```bash
conda run -n drivefusion python -m project.train.train
```

## Legacy Preprocessing

The old YOLO/Detectron2/DPT/RAFT preprocessing code is stored in:

```text
legacy/prepare_dataset/
```

Its configuration is stored in:

```text
configs/legacy/prepare_dataset.yaml
```
