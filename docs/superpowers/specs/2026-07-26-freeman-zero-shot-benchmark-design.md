# FreeMan Zero-Shot Benchmark Design

## Goal

Add a reproducible external benchmark that evaluates the existing gymnastics
multi-view 3D keypoint fusion methods on the complete FreeMan release without
training or tuning on FreeMan.

The benchmark uses all 40 FreeMan subjects. For each session it selects two
synchronised RGB views whose horizontal viewing directions are closest to a
90-degree separation, runs SAM3D-Body independently on both views, applies the
existing deterministic and rotation-aware fusion methods, and evaluates the
results against the FreeMan 3D keypoint reference.

FreeMan 3D keypoints are a public markerless multi-view reference, not
independent marker-based motion-capture ground truth. Reports and manuscript
text must preserve this distinction.

## Scope

The benchmark will:

- download the complete gated `wjwow/FreeMan` Hugging Face dataset;
- retain all downloaded archives under the ignored `local/` tree;
- inspect and validate the official session lists, camera files, videos, and
  2D/3D annotations;
- process all 40 subjects;
- select exactly two views per session from camera geometry;
- run SAM3D-Body at every available frame by default;
- cache all expensive inference results and resume interrupted work;
- evaluate single-view, deterministic fusion, and available zero-shot
  rotation-aware checkpoints;
- write machine-readable and Markdown result summaries;
- keep all downloaded and generated material outside Git tracking.

The benchmark will not:

- train or fine-tune any model on FreeMan;
- use FreeMan 3D reference keypoints inside fusion, checkpoint selection, view
  selection, temporal alignment, or hyperparameter tuning;
- describe FreeMan annotations as Vicon, Qualisys, or independent motion
  capture;
- convert FreeMan into the private gymnastics dataset layout;
- require all archives and all extracted videos to coexist.

## Repository And Storage Layout

All large files remain ignored:

```text
local/
├── datasets/
│   └── freeman/
│       ├── archives/             # exact Hugging Face files
│       ├── manifests/            # local inventory and checksum state
│       └── work/                 # one-subject extraction workspace
└── runs/
    └── freeman_benchmark/
        ├── inspect/
        ├── sam3d/
        ├── fusion/
        ├── evaluation/
        └── report/
```

The Hugging Face release is approximately 829 GB. The project filesystem
currently has approximately 1.4 TB free, so the compressed release can be
retained but a second complete extracted copy cannot.

Processing is subject-serial at the extraction boundary:

1. validate the selected subject archive and required shared annotation files;
2. extract one subject into `local/datasets/freeman/work/subject_<id>/`;
3. run or resume inference for that subject;
4. validate that every completed inference artifact is readable;
5. remove only that subject's disposable extraction workspace;
6. continue to the next subject.

Archive deletion is never automatic. Extraction cleanup targets only the
validated per-subject work directory.

## Hugging Face Access And Download

The implementation uses the current `hf` CLI, not the deprecated
`huggingface-cli`.

Before downloading, `inspect` and `download` must verify:

- the `hf` executable exists;
- a local Hugging Face identity is authenticated;
- access to the gated `wjwow/FreeMan` repository has been granted;
- the target filesystem has enough free space for the remaining archives plus
  a configurable safety reserve;
- partial files and completed files are distinguishable;
- expected split archives such as `.z01`, `.z02`, and `.z03` are retained.

Downloads use `--local-dir` to avoid a second full Hub cache. They are
resumable and never write into a tracked directory.

The download stage records repository revision, filenames, byte sizes, and
local checksums in a manifest. A subsequent run downloads only missing or
invalid files.

## Dataset Contract

Add a dedicated package:

```text
src/gymnastics/benchmarks/freeman/
├── __init__.py
├── schema.py
├── dataset.py
├── download.py
├── pairing.py
├── mapping.py
├── sam3d.py
├── fusion.py
├── evaluation.py
├── report.py
└── cli.py
```

The immutable dataset contract contains:

- repository revision and archive inventory;
- subject and session identifiers;
- view identifiers and video paths;
- per-view camera intrinsics and extrinsics;
- frame rate and frame count;
- selected two-view pair and pair-selection diagnostics;
- FreeMan 2D and 3D keypoint arrays;
- validity masks and frame correspondence;
- official train/validation/test membership for reporting only.

The loader rejects duplicate session IDs, inconsistent frame counts, missing
camera records, invalid rotations, non-finite annotations, and archive/session
disagreement.

Official splits are retained in reports, but the headline zero-shot result uses
all valid sessions from all 40 subjects because no FreeMan data are used for
training or tuning.

## Two-View Selection

FreeMan camera poses vary by session. View selection therefore runs per
session, not once globally.

For each pair of available cameras:

1. derive the optical-axis direction in world coordinates;
2. project the axes onto the world horizontal plane;
3. compute their unsigned angular separation;
4. reject degenerate axes and cameras with invalid calibration;
5. rank pairs by absolute distance from 90 degrees;
6. break ties by smaller camera-height difference;
7. break remaining ties lexicographically by view ID.

Selection uses camera geometry only. It must not inspect FreeMan 3D keypoint
errors or fusion results.

The selected pair is labelled `view_a` and `view_b`; the report must not call a
view "face" or "side" unless that orientation is supported by dataset metadata.
The reference view is deterministically the lexicographically first selected
view.

## Frame And Temporal Contract

FreeMan views are synchronised. The benchmark uses direct frame-index
correspondence and does not estimate a DTW offset.

For every selected session:

- frame indices must exist in both selected videos;
- annotation and video rates must be reconciled explicitly;
- unmatched trailing frames are excluded with a recorded reason;
- internal dropped-frame gaps invalidate the affected interval rather than
  silently shifting one view;
- default frame stride is `1`;
- a non-default stride is allowed only as an explicit diagnostic option and
  is never used for the headline full-data result.

## SAM3D-Body Inference

SAM3D-Body runs independently for the two selected views. The benchmark reuses
the existing project inference implementation instead of copying third-party
logic.

Inference artifacts are stored per subject, session, view, and frame range.
Each artifact records:

- source video identity and checksum;
- camera and frame metadata;
- SAM3D configuration and checkpoint identity;
- MHR70 2D and 3D keypoints;
- confidences and validity;
- completion status.

Writes are atomic. A session is complete only when its artifact passes schema,
shape, finiteness, and frame-count validation. Interrupted or corrupt artifacts
are recomputed without invalidating completed sessions.

## Skeleton Mapping

Evaluation uses a fixed, named intersection between MHR70 and FreeMan body
joints. The mapping is versioned in code and emitted in every report.

The mapping layer:

- validates the FreeMan joint-name order instead of assuming raw indices;
- maps only semantically equivalent body joints;
- excludes face and hand joints not represented by both skeletons;
- propagates per-frame and per-joint validity;
- reports the number of evaluated joints and excluded observations.

If the release lacks joint names, the loader requires an explicitly versioned
FreeMan schema supported by tests. Unknown schemas fail closed.

## Evaluated Methods

The benchmark evaluates:

- selected `view_a` alone;
- selected `view_b` alone;
- every registered deterministic fusion method;
- available rotation-aware paper checkpoints in zero-shot mode.

Leakage-prone or oracle-labelled methods remain separated from valid methods
in reports. They cannot become the headline recommendation.

No method receives FreeMan reference 3D keypoints, FreeMan-derived joint
weights, or FreeMan evaluation metrics during inference.

## Coordinate Alignment And Metrics

FreeMan and SAM3D outputs may use different world frames. The primary
comparison therefore uses one sequence-level Sim3 transformation estimated on
the mapped valid joints. It is estimated once per complete session, not once
per frame, so framewise alignment cannot hide temporal errors.

Reports include:

- sequence-level-Sim3 MPJPE;
- root-relative MPJPE;
- PA-MPJPE as a secondary pose-shape metric;
- PCK and AUC under explicitly recorded thresholds;
- per-joint error;
- velocity error;
- acceleration error;
- valid-frame and valid-joint coverage;
- failure counts by stage and reason.

Metrics are aggregated per session, per subject, official split, action or
scenario when metadata permit, and over the complete dataset. Subject-level
means are the main unit for method comparison so long sessions do not dominate
the headline result.

Pairwise method comparisons use matched subjects and report effect sizes,
confidence intervals, and multiplicity-corrected non-parametric tests when
coverage is sufficient.

## CLI

Expose the staged interface:

```text
gymnastics benchmark freeman inspect
gymnastics benchmark freeman download
gymnastics benchmark freeman infer
gymnastics benchmark freeman fuse
gymnastics benchmark freeman evaluate
gymnastics benchmark freeman report
gymnastics benchmark freeman run
```

All commands accept a configuration file under:

```text
configs/benchmarks/freeman.yaml
```

The configuration defines dataset paths, safety reserve, subject selection,
frame stride, device, SAM3D configuration, checkpoint discovery, output paths,
and evaluation thresholds. The committed default selects all 40 subjects,
frame stride `1`, and two views per session.

Every stage supports dry inspection, clear progress summaries, idempotent
reruns, and explicit `--force` only for scoped recomputation.

## Error Handling

The benchmark stops before mutation when authentication, access, or disk
checks fail.

Per-session data failures are recorded and skipped only when the remaining
dataset can still be evaluated without changing frame correspondence.
Systematic failures, unknown schemas, missing shared annotations, invalid
camera conventions, or coverage below configured thresholds abort evaluation.

No exception handler may convert corrupted data into zero-valued keypoints or
silently substitute another camera pair.

## Tests

Tests use compact synthetic fixtures and mocked inference. They do not download
FreeMan or require a GPU.

Coverage includes:

- archive inventory and split-volume handling;
- disk-reserve and gated-access preflight;
- per-subject extraction lifecycle and safe cleanup targeting;
- session/camera/schema validation;
- deterministic 90-degree pair selection and tie breaking;
- exact frame correspondence and dropped-frame rejection;
- MHR70-to-FreeMan joint mapping;
- resumable atomic inference caches;
- deterministic and rotation-aware adapter isolation from reference 3D;
- sequence-level alignment and all reported metrics;
- subject-balanced aggregation;
- CLI routing, staged reruns, and report generation.

An optional integration test can inspect an already downloaded local FreeMan
installation but is excluded from default pytest runs.

## Acceptance Criteria

Implementation is complete when:

- the committed default targets all 40 subjects and two views per session;
- preflight correctly reports the current local authentication and storage
  state;
- download is resumable and preserves every FreeMan archive and split volume;
- processing never requires more than one extracted subject;
- all selected sessions have reproducible camera-pair records;
- SAM3D inference and fusion resume without repeating valid work;
- evaluation never exposes FreeMan reference 3D to a fusion method;
- reports distinguish public markerless reference results from independent
  marker-based ground truth;
- unit tests pass in the `gymnastic` conda environment;
- all large inputs and generated outputs remain ignored by Git.
