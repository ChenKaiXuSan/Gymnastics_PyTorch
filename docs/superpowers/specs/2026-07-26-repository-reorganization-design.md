# Repository Reorganization Design

## Objective

Replace the current collection of top-level Python packages with one standard
`src/gymnastics` package. Breaking import and command compatibility is accepted.
The result must make active research code, historical code, third-party source,
paper sources, local assets, configuration, tests, and generated outputs
unambiguous.

## Approved constraints

- Use a single `src/gymnastics/` package.
- Do not preserve legacy `python -m <old-package>` entry points or imports.
- Keep `paper/image_and_vision_computing/` in the project, while keeping its generated build
  output untracked.
- Keep `legacy/prepare_dataset/` as frozen reference code outside the installed
  package.
- Represent SAM3 and SAM-3D-Body as pinned third-party repositories rather than
  duplicate maintained source trees.
- Keep checkpoints, calibration videos, logs, and caches locally, but do not
  track them with Git.
- Do not push this reorganization to GitHub.

## Target layout

```text
Gymnastics_PyTorch/
├── pyproject.toml
├── src/gymnastics/
│   ├── cli.py
│   ├── sam3d/
│   ├── alignment/
│   ├── triangulation/
│   ├── fusion/
│   │   ├── deterministic/
│   │   └── rotation_aware/
│   ├── classification/
│   ├── analysis/
│   ├── calibration/
│   └── common/
├── configs/
├── tests/
├── docs/
├── notebooks/
├── scripts/
├── paper/image_and_vision_computing/
├── legacy/prepare_dataset/
├── third_party/
└── local/
    ├── checkpoints/
    ├── calibration_inputs/
    ├── runs/
    └── cache/
```

## Module boundaries

- `sam3d` owns project-specific video discovery, inference orchestration, result
  loading/saving, and visualization adapters. Upstream model implementation does
  not live here.
- `alignment` owns face/side time alignment and cycle segmentation.
- `triangulation` owns calibration consumption, extrinsic estimation, and
  pseudo-reference reconstruction.
- `fusion.deterministic` owns the nine-method comparison matrix.
- `fusion.rotation_aware` owns self-supervised training, inference, and
  evaluation used by the paper.
- `classification` owns person-level folds, datasets, models, training, and
  evaluation.
- `analysis` owns metrics, statistical comparison, reports, and visualization.
- `calibration` owns camera calibration.
- `common` contains only genuinely shared path, I/O, geometry, and skeleton
  definitions. In particular, MHR70 has one canonical definition.

## Public command interface

One console command exposes subcommands:

```text
gymnastics sam3d
gymnastics align
gymnastics triangulate
gymnastics fuse deterministic
gymnastics fuse rotation-aware
gymnastics classify
gymnastics analyze
gymnastics calibrate
```

Subcommands delegate to existing domain `main()` functions during the first
reorganization. Deeper CLI normalization is out of scope unless required to
make the unified dispatcher work.

## Configuration and paths

Configuration mirrors the package domains below `configs/`. Repository-relative
defaults resolve from a single project-root utility. Runtime paths point below
`local/` unless they intentionally reference the external dataset root
`/home/data/xchen/gymnastics`.

Local assets are migrated without deletion:

- `checkpoint/` and `ckpt/` -> `local/checkpoints/`
- `camera_calibration/input_video/` -> `local/calibration_inputs/`
- `local/runs/` -> `local/runs/`
- Python/tool caches -> ignored cache directories

## Third-party policy

The current third-party directories are Git links without a valid
`.gitmodules`. Their remote URLs and exact commits are recorded, `.gitmodules`
is restored, and both repositories remain pinned. Before removing the tracked
`SAM3Dbody/sam_3d_body` duplicate, compare it with the SAM-3D-Body checkout.
Project-specific differences move into the `gymnastics.sam3d` adapter; upstream
implementation remains third-party code.

## Tests and quality gates

- Configure pytest to collect only `tests/`.
- Exclude `third_party`, `legacy`, `.worktrees`, `local`, paper builds, and
  caches.
- Update every test import to `gymnastics.*`.
- Add structure tests for the new package, unified CLI, canonical MHR70
  metadata, ignored local assets, and forbidden legacy imports.
- Preserve the established core baseline of 300 passing tests.
- Keep tests needing optional training or SAM3D dependencies marked as optional
  rather than breaking collection.
- Run import compilation, focused tests, the full `tests/` suite, configuration
  scans, and stale-import scans before completion.

## Migration safety

The work happens in place because the user explicitly requested direct project
modification. Existing uncommitted changes are preserved. Moves are performed
before deletions, local large files are relocated rather than erased, and no
Git commit or push is performed automatically.
