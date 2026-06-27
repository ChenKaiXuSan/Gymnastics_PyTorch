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
conda run -n gymnastic python -m split_cycle.main
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

## One-Person SAM3D-Body, Split-Cycle, And Triangulation Run

Use this when adding or checking one person, for example person `46`.

### 1. Check Required Inputs

The single-person flow starts from raw face/side videos:

```text
/home/data/xchen/gymnastics/raw/person/46/ID46_face.MOV
/home/data/xchen/gymnastics/raw/person/46/ID46_side.MOV
```

Quick check:

```bash
ls /home/data/xchen/gymnastics/raw/person/46/ID46_face.MOV /home/data/xchen/gymnastics/raw/person/46/ID46_side.MOV
```

### 2. Run SAM3D-Body For That Person

Generate per-frame SAM3D-Body predictions for both views:

```bash
conda run -n gymnastic python -m SAM3Dbody.main infer.person_list=[46] infer.gpu=[0] infer.workers_per_gpu=1
```

Use a different GPU by changing `infer.gpu=[0]`. To run all configured people, use the default `infer.person_list=[-1]` from `configs/sam3d_body.yaml`.

Expected inference outputs:

```text
/home/data/xchen/gymnastics/sam3d_body_results/person/46/face/*_sam3d_body.npz
/home/data/xchen/gymnastics/sam3d_body_results/person/46/side/*_sam3d_body.npz
```

Expected visual/log outputs:

```text
/home/data/xchen/gymnastics/sam3d_body_results/logs/46/face/visualization/
/home/data/xchen/gymnastics/sam3d_body_results/logs/46/side/visualization/
/home/data/xchen/gymnastics/sam3d_body_results/logs/person_logs/46.log
```

Check that both views produced frame outputs:

```bash
find /home/data/xchen/gymnastics/sam3d_body_results/person/46 -maxdepth 2 -type f -name '*_sam3d_body.npz' | head
```

Count face/side outputs:

```bash
find /home/data/xchen/gymnastics/sam3d_body_results/person/46/face -type f -name '*_sam3d_body.npz' | wc -l
find /home/data/xchen/gymnastics/sam3d_body_results/person/46/side -type f -name '*_sam3d_body.npz' | wc -l
```

### 3. Generate Split-Cycle Alignment

Run face/side synchronization and cycle segmentation for that person:

```bash
conda run -n gymnastic python -m split_cycle.main --person 46 --threads 1
```

Expected output:

```text
logs/split_cycle/person_46/alignment_record_46.json
logs/split_cycle/person_46/theta_unwrap.png
logs/split_cycle/person_46/face/cycle_*.mp4
logs/split_cycle/person_46/side/cycle_*.mp4
```

Check that the alignment record exists:

```bash
ls logs/split_cycle/person_46/alignment_record_46.json
```

### 4. Run Triangulation For That Person

Triangulate the SAM3D 2D keypoints using the split-cycle alignment:

```bash
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle --person 46
```

For a quick smoke test before the full run:

```bash
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle --person 46 --max-cycles 1 --max-frames 2
```

Expected output:

```text
/home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/summary.json
/home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/cycle_000/summary.json
/home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/cycle_000/joints_3d_sequence.npz
/home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/cycle_000/visualization/*.png
/home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/cycle_000/cycle_000_3d.mp4
```

### 5. Inspect The Result

Read the per-person summary:

```bash
sed -n '1,220p' /home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/summary.json
```

Check generated cycle summaries:

```bash
find /home/data/xchen/gymnastics/sam3d_triangulated/person/person_46 -maxdepth 2 -name summary.json | sort -V
```

Inspect a representative visualization frame:

```text
/home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/cycle_000/visualization/000000.png
```

Important fields to look at in each cycle summary:

```text
processed_frames
missing_pairs
face_reprojection_error_mean_px
side_reprojection_error_mean_px
```

`missing_pairs` should ideally be `0`. Large reprojection errors mean the 3D skeleton may still look continuous but the face/side projection agreement is poor.

### 6. Refresh The Consolidated Report

After the full triangulation run, regenerate the report:

```bash
conda run -n gymnastic python triangulation/tools/generate_results_report.py
```

Report outputs:

```text
logs/analysis/triangulated_results/triangulated_results_report.md
logs/analysis/triangulated_results/triangulated_person_summary.csv
logs/analysis/triangulated_results/triangulated_cycle_details.csv
```

### Common Failure Points

- If SAM3D-Body skips the person, check that both raw videos exist under `raw/person/46/`.
- If SAM3D-Body starts but writes no `.npz` files, check `sam3d_body_results/logs/person_logs/46.log`.
- If triangulation skips the person, check that `alignment_record_46.json` exists.
- If split-cycle skips the person, check that both raw videos exist under `raw/person/46/`.
- If SAM3D keypoints cannot be loaded, check that both `face` and `side` `.npz` folders exist under `sam3d_body_results/person/46/`.
- If only a quick smoke test was run, rerun triangulation without `--max-cycles` and `--max-frames` before using the result as final data.

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
