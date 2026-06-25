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

Run multi-view fusion:

```bash
conda run -n drivefusion python -m fuse.main
```

Run cycle segmentation:

```bash
conda run -n drivefusion python -m split_cycle.main
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
