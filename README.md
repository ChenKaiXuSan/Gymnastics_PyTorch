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
  -> deterministic or cycle-aware fusion
  -> analysis, benchmarks and cohort statistics
```

All active Python code is installed from one package:

```text
src/
├── pose_estimation/    # ① SAM3D-Body inference            python -m pose_estimation
├── cycle_alignment/    # ② offset, cycles, cycle records    python -m cycle_alignment {align,cycles}
├── pseudo_gt/          # ③ calibration, extrinsics, triangulated reference
│                       #                                    python -m pseudo_gt {calibrate,estimate-extrinsics,triangulate}
├── fusion/             # ④ the proposed cycle-aware fusion network   python -m fusion train
│   ├── keypoints/      #    shared 3D-keypoint representation (trial schema, skeleton, body frame, cache)
│   ├── baselines/      #    deterministic matrix + classical baselines   python -m fusion deterministic
│   ├── external/       #    external baselines: model.py = published architectures on our contract (python -m fusion train model=external_*)
│   │                   #                       published/ = the authors' own code, trained per fold (python -m fusion external-published)
│   ├── benchmarks/     #    FreeMan / Unity / Fit3D                      python -m fusion benchmark-*
│   ├── analysis/       #    metrics, reports, cohort statistics          python -m fusion analyze | cohort-cycle
│   └── archive/        #    frozen paper model (rotation_aware)          python -m fusion rotation-aware
├── common/             # shared library: paths, config helpers, MHR70 metadata, CLI dispatcher
└── configs/            # all configuration files, one sub-directory per stage
```

## Installation

```bash
git submodule update --init --recursive
python -m pip install -e ".[analysis,training,test]"
```

### Environment

Every command in this repository is written as a plain `python -m ...` call and
assumes the project conda environment is active (or is run through
`conda run -n <env> ...`). Which environment that is depends on the machine:

| Machine | Environment | Notes |
|---|---|---|
| Lab workstation | `gymnastic` | Full stack, installed with `pip install -e .`. |
| Pegasus / HP260146 | `sam_3d_body` | Used by every `pegasus/*.sh` job script; `direction` also works for fusion training. |

Neither cluster environment has the package installed, so set
`PYTHONPATH=src` there (the job scripts do this). The data root is resolved by
`src/common/paths.py` (see [Data and local assets](#data-and-local-assets)).

### Third-party checkouts

All upstream source is referenced, never copied into the package:

| Path | Contents |
|---|---|
| `src/pose_estimation/third_party/sam-3d-body` | SAM-3D-Body, pinned as a submodule; imported through the adapter in `pose_estimation`. |
| `src/fusion/external/third_party/{VideoPose3D,CanonPose,MHFormer,MDVPose}` | The published baselines, pinned as submodules and trained in place by `python -m fusion external-published`. |
| `src/fusion/external/third_party/metapose` | MetaPose, vendored rather than pinned (TensorFlow, local patches); see its `VENDORED.md`. |

`git submodule update --init --recursive` (above) fetches all of them. The
baseline submodules are configured with `ignore = untracked`, so the
checkpoints and caches written inside them during a run do not show up as
changes of this repository. Baseline weights live under `local/checkpoints`.

## Commands

Run commands from the repository root:

```bash
# Extract SAM3D-Body keypoints.
python -m pose_estimation run

# Align face/side timelines and segment cycles.
python -m cycle_alignment align

# Estimate per-person camera extrinsics.
python -m pseudo_gt estimate-extrinsics

# Build the triangulated pseudo-reference.
python -m pseudo_gt triangulate

# Run the deterministic fusion matrix.
python -m fusion deterministic --methods avg_body_current

# Train the cycle-aware dual-view fusion model (the active model; Hydra overrides).
python -m fusion train experiment=smoke

# Archived paper model (reproduction only; see src/fusion/archive/README.md).
python -m fusion rotation-aware --help

# Analyze saved sequences.
python -m fusion analyze

# Calibrate cameras.
python -m pseudo_gt calibrate
```

Configuration is grouped by domain under `configs/`.

## Results

The current evidence summary, including cohort definitions, headline fusion
results, failure coverage, and unfinished experiments, is in
[docs/results_summary.md](docs/results_summary.md).

Regenerate the detailed local tables from the saved per-person/fold artefacts:

```bash
python -m fusion.analysis.project_results
```

## Data and local assets

The external dataset root is resolved once in `src/common/paths.py`: set
`GYMNASTICS_DATA_ROOT` to override it, otherwise the first known machine root
that exists is used. Configs interpolate it with `${oc.env:GYMNASTICS_DATA_ROOT}`.

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
python -m fusion benchmark-freeman inspect
python -m fusion benchmark-freeman download
python -m fusion benchmark-freeman run
```

Downloaded archives, extracted subject workspaces, predictions, and reports all
remain under ignored `local/` paths.

## Cycle-aware dual-view fusion (Architecture v1.2)

`fusion` is a second, self-contained learned fusion
model that treats the repeated-cycle structure of the recorded motion as a
first-class signal. It is trained with PyTorch Lightning, configured with
Hydra, and shares the SAM3D-Body inputs, the canonical body frame, and the
skeleton metadata of the rest of the repository. The long-form description is
in [docs/cycle_aware_fusion.md](docs/cycle_aware_fusion.md).

### Research goal

Two uncalibrated monocular 3D pose estimates of the same person (View A/B)
fail at different joints and different times. The base pose combines the two
views with a calibration-free geometric prior (v1.1: each view's precision is
discounted along its own camera-depth axis, obtained from the view's own
canonicalisation), and the model learns, without any 3D labels, a small
bounded correction on top of that prior, using local motion (velocity) and
cycle-scale motion (phase, periodic recurrence) as the evidence. Since v1.2
(2026-09-25) both views get equal weight: the v1.1 head that learned *how
much to trust each view for every joint at every time step* lost to equal
weights and is switched off (`model=v1_1 loss=v3` restores it).

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
(Joint-Wise Reliability)       v1.1 only; v1.2: w_A = w_B = 1/2
      ↓
Depth-Aware Pose Fusion        Λ_v = w_v (I − α d_v d_vᵀ),  P_base = (Λ_A+Λ_B)⁻¹(Λ_A P_A + Λ_B P_B)
                               (original inputs, α = 0.8 fixed; zero parameters in v1.2)
      ↓
Residual Refinement            P_hat = P_base + ΔP,  ΔP = 0.25 tanh(MLP([½(C_A+C_B) ; |C_A−C_B| ; P_base]))
      ↓
Final 3D Pose
```

Both views go through the *same* encoder weights; the model is symmetric
under swapping the views.

### Project structure

```text
src/fusion/
├── skeleton.py           common joint set (mhr70 / mhr70_major), bones, mirrors
├── sample.py             DualViewSample + FusionBatch contracts, canonicalisation
├── phase.py              cycle phase, phase normalisation, phase encoding, cycle estimation
├── velocity.py           physical velocity from timestamps
├── outputs.py            PoseFusionOutput dataclass
├── modules/              spatial / short / long transformers, motion fusion, FiLM,
│                         cross-view attention, reliability, depth-aware fusion, residual
├── model.py              CycleAwareFusionModel + CycleAwareModelConfig (ARCHITECTURE_VERSION)
├── losses.py             recovery (v3), cross-cycle (v2), periodicity, symmetry, residual objectives
├── corruptions.py        joint/distal masks, noise, depth drift, frame dropouts
├── metrics.py            Procrustes / translation aligned MPJPE
├── data/                 base DataModule, windows, sample cache, and one adapter per
│                         dataset (gymnastics, freeman, unity, fit3d) plus synthetic
├── lightning_module.py   training / validation / test / predict steps
└── train.py              Hydra entry point (`python -m fusion train`)
src/configs/fusion/      Hydra groups: model, data, loss, corruption, trainer, optimizer, experiment
tests/fusion/        unit and integration tests
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
| `gymnastics` | rotation-aware person cache or SAM3D + split-cycle records | `alignment_record_<id>.json` (`python -m cycle_alignment align`, middles via `python -m cycle_alignment cycles private`) | triangulated pseudo-reference |
| `freeman` | zero-shot benchmark SAM3D cache | `local/runs/cycle_records/freeman` (`python -m cycle_alignment cycles freeman`) | `keypoints3d_optim` (COCO17) |
| `unity` | Unity manifest + SAM3D camera cache | `local/runs/cycle_records/unity` (`python -m cycle_alignment cycles unity`) | native 3D (Unity22) |
| `fit3d` | Fit3D release + the external SAM3D cache (`python -m fusion benchmark-fit3d`) | `local/runs/cycle_records/fit3d` (`python -m cycle_alignment cycles fit3d`; bounds are the release's repetition annotations) | multi-view fitted 3D (25 joints, metres) |
| `synthetic` | generated in memory | exact | generating motion |

Cycle detection (cycle start = right-wrist azimuth crossing, middle =
turn-around extremum) lives entirely in `cycle_alignment`; the training
package only reads the record files, so run the `align cycles` step before
training:

```bash
python -m cycle_alignment cycles private      # adds "mid" to the 137 records
python -m cycle_alignment cycles freeman      # local/runs/cycle_records/freeman
python -m cycle_alignment cycles unity        # local/runs/cycle_records/unity
python -m cycle_alignment cycles fit3d        # local/runs/cycle_records/fit3d
python -m cycle_alignment cycles index        # unified tree + index.json + README.md
```

### Configuration

Hydra composes `src/configs/fusion/config.yaml` with the groups `model`,
`data`, `loss`, `corruption`, `trainer`, `optimizer`, and the optional
`experiment` presets. Every command-line argument is an override:

```bash
python -m fusion train print_config=true data=freeman
```

### Training

```bash
# Smoke run on synthetic data (CPU, seconds).
python -m fusion train experiment=smoke

# Private data, fixed 96/27/14 split, 50 epochs.
python -m fusion train data=gymnastics trainer.max_epochs=50

# FreeMan, subject-disjoint split (cycle records from `python -m cycle_alignment cycles freeman`).
python -m fusion train data=freeman

# Unity direction-transfer fold.
python -m fusion train data=unity data.options.fold=right_to_left

# Fit3D (prepare once: select the view pair, then the repetition records).
python -m fusion benchmark-fit3d select-views
python -m cycle_alignment cycles fit3d
python -m fusion train data=fit3d data.fold_json=src/configs/fusion/folds/fit3d/fold_01.json
```

Outputs (resolved config, CSV logs, checkpoints, `result.json`) are written
below `local/runs/cycle_aware/<run_name>`.

Objectives (v2, default): leave-one-cycle-out cross-cycle pose target as the
main supervision, corruption-labelled reliability, feature-level periodicity
and half-cycle mirror symmetry, L1 residual regulariser; no target is built
from the current window's own two views (`loss=v1_recovery` keeps the earlier
recovery objective). See [docs/cycle_aware_fusion.md](docs/cycle_aware_fusion.md).

Cross-validation (5 folds, single seed, 50 epochs is the fixed protocol):
`folds_dir=src/configs/fusion/folds/gymnastics` runs the folds sequentially;
on the cluster use the job scripts in `pegasus/` (one gpu job per fold) and
`python -m fusion.summarize <sweep_dir>` afterwards.

### Testing

```bash
python -m pytest tests/fusion -q
```

### Ablation studies

Each module has a Hydra switch under `model.*` and a ready-made preset under
`src/configs/fusion/experiment/`:

```bash
python -m fusion train data=gymnastics experiment=no_film
python -m fusion train data=gymnastics model.cross_view.enabled=false
```

Presets: `no_film`, `no_cross_view`, `no_short_motion`, `no_long_motion`,
`no_phase`, `equal_reliability`, `no_residual`, `pose_only`, `full_skeleton`,
`full_context` (long-term context = whole sequence). The long-term context is
`num_cycles` (0.5, 1, 2, ... or `null` for the full sequence).

## Repository boundaries

- `src/`: active project-owned Python code.
- `tests/`: tests mirroring the active package.
- `configs/`: runtime configuration grouped by domain.
- `docs/`: current workflow, module, and runbook documentation.
- `pegasus/`: NQSV job scripts for the cluster (one per stage family); see `pegasus/README.md`.
- `paper/image_and_vision_computing/`: local manuscript workspace; generated builds are not
  part of the Python package.

## Verification

```bash
python -m pytest -q
python -m compileall -q src
```

Additional workflow documentation:

- [Pipeline and runbook](docs/pipeline.md) (Chinese)
- [Module map](docs/modules.md)
- [Cycle-aware fusion](docs/cycle_aware_fusion.md) (the active model)
- [Triangulation](docs/triangulation.md)
- [Results summary](docs/results_summary.md)
- [docs/research/](docs/research/) dated experiment reports; [docs/archive/](docs/archive/) retired-model notes

## License

Apache License 2.0. See [LICENSE](LICENSE).
