# Repository Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the repository into one installable `src/gymnastics` package with explicit domain boundaries and isolated local/third-party assets.

**Architecture:** Move active code into domain packages under `src/gymnastics`, expose one console dispatcher, mirror domains in configuration and tests, and keep non-source material outside the installed package. Preserve algorithms while intentionally replacing all old import and command paths.

**Tech Stack:** Python 3.10, setuptools via `pyproject.toml`, pytest, Hydra/OmegaConf, PyTorch.

## Global Constraints

- Breaking old imports and commands is allowed.
- Preserve local checkpoints, calibration videos, logs, and caches without Git tracking.
- Keep paper sources local and untracked according to the current repository policy.
- Keep historical preprocessing under `legacy/`.
- Do not push to GitHub.
- Run project Python commands with `conda run -n gymnastic`.

---

### Task 1: Lock the new package contract

**Files:**
- Create: `tests/structure/test_repository_layout.py`
- Create: `tests/structure/test_cli.py`
- Create: `pyproject.toml`
- Create: `src/gymnastics/__init__.py`
- Create: `src/gymnastics/__main__.py`
- Create: `src/gymnastics/cli.py`
- Remove: `setup.py`
- Remove: `setup.cfg`
- Remove: `MANIFEST.in`

**Interfaces:**
- Produces: `gymnastics.cli.main(argv: Sequence[str] | None = None) -> int`
- Produces: console script `gymnastics = gymnastics.cli:main`

- [ ] Write structure tests that require `src/gymnastics`, a `pyproject.toml`
  pytest configuration restricted to `tests`, and the new CLI command names.
- [ ] Run the tests and verify they fail because the package and dispatcher do
  not exist.
- [ ] Create the minimal package, dispatcher, and packaging metadata.
- [ ] Run the structure and CLI tests and verify they pass.

### Task 2: Establish canonical shared metadata

**Files:**
- Create: `src/gymnastics/common/skeletons/mhr70.py`
- Create: `src/gymnastics/common/paths.py`
- Modify: all consumers of `fuse.metadata.mhr70`,
  `split_cycle.metadata.mhr70`, and SAM3D-local metadata.
- Remove: duplicate project-owned MHR70 modules after migration.

**Interfaces:**
- Produces: `MHR70_NAMES`, `MHR70_INDEX`, and project/local root helpers.

- [ ] Add a test proving every active consumer imports the same MHR70 object.
- [ ] Run it and verify the missing canonical module failure.
- [ ] Move the canonical definition and update active consumers.
- [ ] Run metadata and affected fusion/alignment tests.

### Task 3: Move the active pipeline packages

**Files:**
- Move: `SAM3Dbody/{main,infer,load,save,vis}.py` ->
  `src/gymnastics/sam3d/`
- Move: `split_cycle/` -> `src/gymnastics/alignment/`
- Move: `triangulation/` -> `src/gymnastics/triangulation/`
- Move: deterministic `fuse/` modules ->
  `src/gymnastics/fusion/deterministic/`
- Move: `fuse/rotation_aware/` ->
  `src/gymnastics/fusion/rotation_aware/`
- Modify: corresponding tests and internal imports.

**Interfaces:**
- Produces domain `main()` functions used by the unified CLI.

- [ ] Update one domain's tests to the desired import and verify collection
  fails before its move.
- [ ] Move that domain, update imports, and run its focused tests.
- [ ] Repeat for alignment, triangulation, deterministic fusion, rotation-aware
  fusion, and SAM3D adapters.
- [ ] Scan active Python for forbidden old absolute imports.

### Task 4: Move classification, analysis, calibration, and notebooks

**Files:**
- Move: `project/cross_validation/` ->
  `src/gymnastics/classification/splits/`
- Move: `project/train/` -> `src/gymnastics/classification/`
- Move: `analysis/*.py` -> `src/gymnastics/analysis/`
- Move: `camera_calibration/main.py` ->
  `src/gymnastics/calibration/`
- Move: active notebooks -> `notebooks/analysis/` and
  `notebooks/calibration/`
- Remove: `analysis/main copy.py` after confirming it has no unique behavior.

**Interfaces:**
- Produces classification, analysis, and calibration `main()` entry points.

- [ ] Update affected tests/import smoke checks to desired package paths and
  verify failure.
- [ ] Move each domain and repair only its internal imports.
- [ ] Run classification-independent tests and import compilation.
- [ ] Confirm notebooks and source files are separated.

### Task 5: Normalize configuration and CLI dispatch

**Files:**
- Move: root YAML files into matching `configs/<domain>/` directories.
- Modify: config lookups in every domain.
- Modify: `src/gymnastics/cli.py`.
- Create: `tests/structure/test_config_paths.py`.

**Interfaces:**
- Consumes: each domain's `main()` function.
- Produces: unified command dispatch and stable config discovery.

- [ ] Add tests for each command-to-domain mapping and each config path.
- [ ] Verify the tests fail against placeholder dispatch.
- [ ] Implement lazy command imports so optional heavyweight dependencies do
  not break CLI help or test collection.
- [ ] Run CLI/config tests and `gymnastics --help`.

### Task 6: Repair third-party and local asset layout

**Files:**
- Create: `.gitmodules`
- Modify: `.gitignore`
- Create: `third_party/README.md`
- Move local assets into `local/`.

**Interfaces:**
- Produces pinned third-party checkout metadata and ignored local runtime roots.

- [ ] Record current remotes and commits for both third-party repositories.
- [ ] Compare project-owned SAM3D implementation with upstream and retain any
  project-specific adapter changes.
- [ ] Restore valid submodule metadata.
- [ ] Move checkpoints, calibration inputs, and logs without deleting content.
- [ ] Verify `git check-ignore` covers every local runtime path and
  `git submodule status` succeeds.

### Task 7: Documentation, stale-code audit, and full verification

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/current_pipeline.md`
- Modify: `docs/modules.md`
- Modify: `docs/runbook.md`
- Modify: paper generation scripts only where imports or source paths changed.

**Interfaces:**
- Produces documentation matching the new commands and paths.

- [ ] Replace all active documentation commands and package paths.
- [ ] Search for old module imports, old commands, duplicate metadata, generated
  caches, and code outside `src/gymnastics`.
- [ ] Run `python -m compileall src/gymnastics`.
- [ ] Run focused domain tests.
- [ ] Run the complete `pytest` command configured by `pyproject.toml`.
- [ ] Run `git diff --check` and inspect `git status`.
- [ ] Confirm no files are staged, committed, or pushed.
