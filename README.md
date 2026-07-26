# Gymnastics Motion Analysis

A SAM3D-Body-first pipeline for multi-view gymnastics motion analysis. Paired
face/side videos are converted into 3D keypoints, temporally aligned and split
into movement cycles, triangulated into a pseudo-reference, fused across views,
and optionally used for motion classification.

## Active pipeline

```text
paired face/side videos
  -> SAM3D-Body keypoints
  -> temporal alignment and cycle segmentation
  -> triangulated 3D pseudo-reference
  -> deterministic or rotation-aware fusion
  -> analysis and optional classification
```

All active Python code is installed from one package:

```text
src/gymnastics/
├── sam3d/              # inference orchestration and project adapters
├── alignment/          # face/side alignment and cycle segmentation
├── triangulation/      # extrinsics and pseudo-reference reconstruction
├── fusion/
│   ├── deterministic/  # nine-method comparison matrix
│   └── rotation_aware/ # self-supervised paper method
├── classification/     # splits, datasets, models, training, evaluation
├── analysis/           # metrics, reports, statistics, visualization
├── calibration/        # camera calibration
└── common/             # canonical paths and skeleton metadata
```

## Installation

The project uses the `gymnastic` Conda environment for research commands.

```bash
git submodule update --init --recursive
conda run -n gymnastic python -m pip install -e ".[analysis,training,test]"
```

SAM3 and SAM-3D-Body are pinned below `third_party/`. Project code imports them
through the adapter in `gymnastics.sam3d`; upstream source is not duplicated in
the installed package.

## Commands

Run commands from the repository root:

```bash
# Extract SAM3D-Body keypoints.
conda run -n gymnastic gymnastics sam3d

# Align face/side timelines and segment cycles.
conda run -n gymnastic gymnastics align

# Estimate per-person camera extrinsics.
conda run -n gymnastic gymnastics triangulate estimate-extrinsics

# Build the triangulated pseudo-reference.
conda run -n gymnastic gymnastics triangulate

# Run the deterministic fusion matrix.
conda run -n gymnastic gymnastics fuse deterministic --methods avg_body_current

# Run the rotation-aware paper method.
conda run -n gymnastic gymnastics fuse rotation-aware --help

# Train/evaluate classifiers.
conda run -n gymnastic gymnastics classify

# Analyze saved sequences.
conda run -n gymnastic gymnastics analyze

# Calibrate cameras.
conda run -n gymnastic gymnastics calibrate
```

Configuration is grouped by domain under `configs/`.

## Data and local assets

The external dataset defaults to `/home/data/xchen/gymnastics` and can be
overridden with `GYMNASTICS_DATA_ROOT`.

Large or generated local material is kept under the ignored `local/` root:

```text
local/
├── checkpoints/       # model weights
├── calibration_inputs/ # calibration videos
├── runs/              # alignment, fusion, training, and analysis outputs
└── cache/             # local caches and migration backups
```

These files remain on the workstation and are not tracked by Git.

## Repository boundaries

- `src/gymnastics/`: active project-owned Python code.
- `tests/`: tests mirroring the active package.
- `configs/`: runtime configuration grouped by domain.
- `docs/`: current workflow, module, and runbook documentation.
- `notebooks/`: exploratory analysis separated from importable code.
- `scripts/`: operational scripts only.
- `legacy/`: frozen historical code excluded from installation and default tests.
- `third_party/`: pinned upstream repositories.
- `paper/neurocomputing/`: local manuscript workspace; generated builds are not
  part of the Python package.

## Verification

```bash
conda run -n gymnastic python -m pytest -q
conda run -n gymnastic python -m compileall -q src/gymnastics
```

Additional workflow documentation:

- [Current pipeline](docs/current_pipeline.md)
- [Runbook](docs/runbook.md)
- [Module map](docs/modules.md)
- [Rotation-aware fusion](docs/rotation_aware_fusion.md)
- [Triangulation](docs/triangulation.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
