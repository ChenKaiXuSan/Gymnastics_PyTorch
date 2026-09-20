# Module Map

All active code lives under `src/` (`PYTHONPATH=src`, no umbrella package).
The four pipeline stages are the top-level packages and the only entry points;
`common/` is a shared library and `configs/` the configuration tree.

| Stage | Package | Responsibility | Primary command |
|---|---|---|---|
| ① | `pose_estimation` | SAM3D-Body inference on paired face/side videos. | `python -m pose_estimation run` |
| ② | `cycle_alignment` | Align face/side timelines, segment movement cycles, write cycle records. | `python -m cycle_alignment align` |
| ③ | `pseudo_gt` | Camera calibration, per-person extrinsics, triangulated 3D pseudo-reference (evaluation only). | `python -m pseudo_gt calibrate`, `python -m pseudo_gt triangulate` |
| ④ | `fusion` | **The proposed model.** Cycle-aware dual-view fusion (transformer encoders, FiLM, cross-view reliability) trained with Lightning and configured with Hydra. | `python -m fusion train` |
| support | `fusion.keypoints` | Shared 3D-keypoint representation: `PosePairTrial`, `SkeletonSpec`, canonical body frame, trunk/quality features, person cache. | – |
| support | `fusion.baselines` | Deterministic fusion methods and classical baselines. | `python -m fusion deterministic` |
| support | `fusion.benchmarks` | FreeMan and Unity benchmarks. | `python -m fusion benchmark-{freeman,freeman-train,unity}` |
| support | `fusion.analysis` | Metrics, comparisons, reports, cohort statistics. | `python -m fusion analyze`, `python -m fusion cohort-cycle` |
| support | `common` | Canonical project paths and MHR70 metadata. | Imported by other packages |
| archive | `fusion.archive.rotation_aware` | Frozen paper model (2026-09-19); kept for reproducing published tables. | `python -m fusion rotation-aware` |

## Boundaries

- Domain packages can depend on `common`.
- Analysis may read outputs from every pipeline stage but does not participate
  in training or inference.
- Fusion training cannot import triangulated pseudo-reference data. Only its
  evaluation layer may read that data.
- Project-specific SAM3D adapters may import the pinned third-party checkout;
  upstream source must not be copied into `src/gymnastics`.
- Runtime files belong below `local/`, never inside an importable package.

## Supporting directories

| Directory | Purpose |
|---|---|
| `configs/` | Domain-aligned YAML configuration. |
| `tests/` | Automated verification of active code. |
| `notebooks/` | Exploratory work that is not imported by production code. |
| `scripts/` | Monitoring, bootstrap, and other operational commands. |
| `third_party/` | Pinned upstream Git submodules. |
| `paper/image_and_vision_computing/` | Local manuscript sources and generated paper assets. |
| `local/` | Ignored checkpoints, videos, runs, caches, and migration backups. |
