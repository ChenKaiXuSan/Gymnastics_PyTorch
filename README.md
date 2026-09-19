# Gymnastics Motion Analysis

A SAM3D-Body-first pipeline for multi-view gymnastics motion analysis. Paired
face/side videos are converted into 3D keypoints, temporally aligned and split
into movement cycles, triangulated into a pseudo-reference, fused across views,
and analysed (fusion metrics, public benchmarks, cohort statistics).

## Active pipeline

```text
paired face/side videos
  -> SAM3D-Body keypoints
  -> temporal alignment and cycle segmentation
  -> triangulated 3D pseudo-reference
  -> deterministic or rotation-aware fusion
  -> analysis, benchmarks and cohort statistics
```

All active Python code is installed from one package:

```text
src/gymnastics/
├── sam3d/              # inference orchestration and project adapters
├── alignment/          # face/side alignment and cycle segmentation
├── triangulation/      # extrinsics and pseudo-reference reconstruction
├── fusion/
│   ├── deterministic/  # nine-method comparison matrix
│   ├── rotation_aware/ # self-supervised paper method
│   └── cycle_aware/    # cycle-aware dual-view fusion (Lightning + Hydra)
├── benchmarks/         # Unity native-3D and FreeMan public-data benchmarks
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

# Train the cycle-aware dual-view fusion model (Hydra overrides).
conda run -n gymnastic gymnastics fuse cycle-aware experiment=smoke

# Train the rotation-conditioned and cross-view-only attention ablations.
# All production rotation-aware configs use the same fixed 137-person
# train/validation/test split (96/27/14).
conda run -n gymnastic gymnastics fuse rotation-aware train \
  --config configs/fusion/rotation_aware_cross_attention.yaml \
  --run-id paper137_a10_b64_e100_s0 --ablation A10
conda run -n gymnastic gymnastics fuse rotation-aware train \
  --config configs/fusion/rotation_aware_cross_attention.yaml \
  --run-id paper137_a11_b64_e100_s0 --ablation A11

# Analyze saved sequences.
conda run -n gymnastic gymnastics analyze

# Calibrate cameras.
conda run -n gymnastic gymnastics calibrate
```

Configuration is grouped by domain under `configs/`.

## Results

The current evidence summary, including cohort definitions, headline fusion
results, failure coverage, and unfinished experiments, is in
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

## Cycle-aware dual-view fusion (Architecture v1.0)

`gymnastics.fusion.cycle_aware` is a second, self-contained learned fusion
model that treats the repeated-cycle structure of the recorded motion as a
first-class signal. It is trained with PyTorch Lightning, configured with
Hydra, and shares the SAM3D-Body inputs, the canonical body frame, and the
skeleton metadata of the rest of the repository. The long-form description is
in [docs/cycle_aware_fusion.md](docs/cycle_aware_fusion.md).

### Research goal

Two uncalibrated monocular 3D pose estimates of the same person (View A/B)
fail at different joints and different times. The model learns, without any
3D labels, *how much to trust each view for every joint at every time step*
and applies a small bounded correction, using local motion (velocity) and
cycle-scale motion (phase, periodic recurrence) as the evidence.

### Architecture

```text
View A/B 3D Kpts  [B, T, J, 3]
      ↓
Spatial Transformer            (attention over joints, per frame)      -> F_pose
      +
Short Temporal Transformer     (local band, ~0.25 cycle, per joint)    -> F_short
      +
Long Temporal Transformer      (whole window + phase encoding)         -> F_long
      ↓
Motion Fusion                  F_motion = MLP([F_short ; F_long])
      ↓
FiLM                           H = (1 + γ(F_motion)) · F_pose + β(F_motion)
      ↓
Motion-Guided A/B
      ↓
Bidirectional Cross-Attention  H_A <-> H_B (same frame, over joints)
      ↓
Joint-Wise Reliability         [w_A, w_B] = softmax(R),  w_A + w_B = 1
      ↓
Weighted Pose Fusion           P_base = w_A · P_A + w_B · P_B   (original inputs)
      ↓
Residual Refinement            P_hat = P_base + ΔP  (bounded)
      ↓
Final 3D Pose
```

Both views go through the *same* encoder weights; the model is symmetric
under swapping the views.

### Project structure

```text
src/gymnastics/fusion/cycle_aware/
├── skeleton.py           common joint set (mhr70 / mhr70_major), bones, mirrors
├── sample.py             DualViewSample + FusionBatch contracts, canonicalisation
├── phase.py              cycle phase, phase normalisation, phase encoding, cycle estimation
├── velocity.py           physical velocity from timestamps
├── outputs.py            PoseFusionOutput dataclass
├── modules/              spatial / short / long transformers, motion fusion, FiLM,
│                         cross-view attention, reliability, weighted fusion, residual
├── model.py              CycleAwareFusionModel + CycleAwareModelConfig
├── losses.py             recovery, periodicity, symmetry, residual objectives
├── corruptions.py        joint/distal masks, noise, depth drift, frame dropouts
├── metrics.py            Procrustes / translation aligned MPJPE
├── data/                 base DataModule, windows, sample cache, and one adapter per
│                         dataset (gymnastics, freeman, unity) plus synthetic
├── lightning_module.py   training / validation / test / predict steps
└── train.py              Hydra entry point (`gymnastics fuse cycle-aware`)
configs/cycle_aware/      Hydra groups: model, data, loss, corruption, trainer, optimizer, experiment
tests/cycle_aware/        unit and integration tests
```

### Dataset interface

Every adapter returns `DualViewSample` objects: two `[T, J, 3]` views in the
pelvis-centred canonical body frame, `[T, J]` validity masks, strictly
increasing physical timestamps, optional complete-cycle ranges, and an
optional reference pose (evaluation only). The shared windowing code turns
samples into `[B, T, J, ·]` batches; the model never sees dataset-specific
structure.

| Adapter | Source | Cycles + middles (precomputed) | Reference |
|---|---|---|---|
| `gymnastics` | rotation-aware person cache or SAM3D + split-cycle records | `alignment_record_<id>.json` (`gymnastics align`, middles via `gymnastics align cycles private`) | triangulated pseudo-reference |
| `freeman` | zero-shot benchmark SAM3D cache | `local/runs/cycle_records/freeman` (`gymnastics align cycles freeman`) | `keypoints3d_optim` (COCO17) |
| `unity` | Unity manifest + SAM3D camera cache | `local/runs/cycle_records/unity` (`gymnastics align cycles unity`) | native 3D (Unity22) |
| `synthetic` | generated in memory | exact | generating motion |

Cycle detection (cycle start = right-wrist azimuth crossing, middle =
turn-around extremum) lives entirely in `gymnastics.alignment`; the training
package only reads the record files, so run the `align cycles` step before
training:

```bash
conda run -n gymnastic gymnastics align cycles private      # adds "mid" to the 137 records
conda run -n gymnastic gymnastics align cycles freeman      # local/runs/cycle_records/freeman
conda run -n gymnastic gymnastics align cycles unity        # local/runs/cycle_records/unity
conda run -n gymnastic gymnastics align cycles index        # unified tree + index.json + README.md
```

### Configuration

Hydra composes `configs/cycle_aware/config.yaml` with the groups `model`,
`data`, `loss`, `corruption`, `trainer`, `optimizer`, and the optional
`experiment` presets. Every command-line argument is an override:

```bash
conda run -n gymnastic gymnastics fuse cycle-aware print_config=true data=freeman
```

### Training

```bash
# Smoke run on synthetic data (CPU, seconds).
conda run -n gymnastic gymnastics fuse cycle-aware experiment=smoke

# Private data, fixed 96/27/14 split, 50 epochs.
conda run -n gymnastic gymnastics fuse cycle-aware data=gymnastics trainer.max_epochs=50

# FreeMan, subject-disjoint split (cycle records from `gymnastics align cycles freeman`).
conda run -n gymnastic gymnastics fuse cycle-aware data=freeman

# Unity direction-transfer fold.
conda run -n gymnastic gymnastics fuse cycle-aware data=unity data.options.fold=right_to_left
```

Outputs (resolved config, CSV logs, checkpoints, `result.json`) are written
below `local/runs/cycle_aware/<run_name>`.

Cross-validation (5 folds, single seed, 50 epochs is the fixed protocol):
`folds_dir=configs/cycle_aware/folds/gymnastics` runs the folds sequentially;
on the cluster use the job scripts in `pegasus/` (one gpu job per fold) and
`python -m gymnastics.fusion.cycle_aware.summarize <sweep_dir>` afterwards.

### Testing

```bash
conda run -n gymnastic python -m pytest tests/cycle_aware -q
```

### Ablation studies

Each module has a Hydra switch under `model.*` and a ready-made preset under
`configs/cycle_aware/experiment/`:

```bash
conda run -n gymnastic gymnastics fuse cycle-aware data=gymnastics experiment=no_film
conda run -n gymnastic gymnastics fuse cycle-aware data=gymnastics model.cross_view.enabled=false
```

Presets: `no_film`, `no_cross_view`, `no_short_motion`, `no_long_motion`,
`no_phase`, `equal_reliability`, `no_residual`, `pose_only`, `full_skeleton`,
`full_context` (long-term context = whole sequence). The long-term context is
`num_cycles` (0.5, 1, 2, ... or `null` for the full sequence).

## Repository boundaries

- `src/gymnastics/`: active project-owned Python code.
- `tests/`: tests mirroring the active package.
- `configs/`: runtime configuration grouped by domain.
- `docs/`: current workflow, module, and runbook documentation.
- `notebooks/`: exploratory analysis separated from importable code.
- `scripts/`: operational scripts only.
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
- [Cycle-aware fusion](docs/cycle_aware_fusion.md)
- [Triangulation](docs/triangulation.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
