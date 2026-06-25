# Current Pipeline

The active dataset preparation path now uses SAM3D-Body directly.

```text
/home/data/xchen/gymnastics/raw/person
  -> SAM3Dbody
  -> /home/data/xchen/gymnastics/sam3d_body_results/person
  -> fuse
  -> split_cycle
  -> project/train
  -> analysis
```

## Active Modules

- `SAM3Dbody/`: runs SAM3D-Body inference on `face` and `side` videos.
- `fuse/`: aligns and fuses the SAM3D-Body results from multiple views.
- `split_cycle/`: segments fused motion into cycles.
- `project/train/`: trains and evaluates classifiers from prepared motion data.
- `analysis/`: compares, visualizes, and reports results.

## Legacy Modules

- `legacy/prepare_dataset/`: previous DPT, RAFT, YOLOv11, and Detectron2 preprocessing pipeline.
- `configs/legacy/prepare_dataset.yaml`: configuration for the legacy preprocessing pipeline.

The legacy pipeline is kept for reference, but it is no longer part of the active
SAM3D-Body-only data preparation flow.
