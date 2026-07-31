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

# Train the rotation-conditioned and cross-view-only attention ablations.
# All production rotation-aware configs use the same fixed 137-person
# train/validation/test split (96/27/14).
conda run -n gymnastic gymnastics fuse rotation-aware train \
  --config configs/fusion/rotation_aware_cross_attention.yaml \
  --run-id paper137_a10_b64_e100_s0 --ablation A10
conda run -n gymnastic gymnastics fuse rotation-aware train \
  --config configs/fusion/rotation_aware_cross_attention.yaml \
  --run-id paper137_a11_b64_e100_s0 --ablation A11

# Train/evaluate classifiers.
conda run -n gymnastic gymnastics classify

# Analyze saved sequences.
conda run -n gymnastic gymnastics analyze

# Calibrate cameras.
conda run -n gymnastic gymnastics calibrate
```

Configuration is grouped by domain under `configs/`.

## Results

The current evidence summary, including cohort definitions, headline fusion and
classification results, failure coverage, and unfinished experiments, is in
[docs/results_summary.md](docs/results_summary.md).

Regenerate the detailed local tables from the saved per-person/fold artefacts:

```bash
conda run -n gymnastic python -m gymnastics.analysis.project_results
```

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

### FreeMan zero-shot benchmark

FreeMan is a gated Hugging Face dataset of approximately 829 GB compressed.
The benchmark uses all 40 subjects, selects synchronized near-orthogonal camera
pairs, and evaluates the gymnastics-trained fusion models zero-shot against
FreeMan's markerless multi-view 3D reference. That reference is not independent
marker-based motion capture.

```bash
conda run -n gymnastic gymnastics benchmark freeman inspect
conda run -n gymnastic gymnastics benchmark freeman download
conda run -n gymnastic gymnastics benchmark freeman run
```

Downloaded archives, extracted subject workspaces, predictions, and reports all
remain under ignored `local/` paths.

## Repository boundaries

- `src/gymnastics/`: active project-owned Python code.
- `tests/`: tests mirroring the active package.
- `configs/`: runtime configuration grouped by domain.
- `docs/`: current workflow, module, and runbook documentation.
- `notebooks/`: exploratory analysis separated from importable code.
- `scripts/`: operational scripts only.
- `legacy/`: frozen historical code excluded from installation and default tests.
- `third_party/`: pinned upstream repositories.
- `paper/image_and_vision_computing/`: local manuscript workspace; generated builds are not
  part of the Python package.

## Verification

```bash
conda run -n gymnastic python -m pytest -q
conda run -n gymnastic python -m compileall -q src/gymnastics
```

Additional workflow documentation:

- [Current pipeline](docs/current_pipeline.md)
- [Results summary](docs/results_summary.md)
- [Runbook](docs/runbook.md)
- [Module map](docs/modules.md)
- [Rotation-aware fusion](docs/rotation_aware_fusion.md)
- [Triangulation](docs/triangulation.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
