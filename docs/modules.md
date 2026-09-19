# Module Map

All active code belongs to the `gymnastics` package under `src/`.

| Package | Responsibility | Primary command |
|---|---|---|
| `gymnastics.sam3d` | Discover paired videos and orchestrate SAM3D-Body inference. | `gymnastics sam3d` |
| `gymnastics.alignment` | Align face/side timelines and segment movement cycles. | `gymnastics align` |
| `gymnastics.triangulation` | Estimate camera extrinsics and reconstruct the 3D pseudo-reference. | `gymnastics triangulate` |
| `gymnastics.fusion.deterministic` | Run and evaluate the nine deterministic fusion methods. | `gymnastics fuse deterministic` |
| `gymnastics.fusion.core` | Shared fusion infrastructure: `PosePairTrial`, `SkeletonSpec`, canonical body frame, trunk/quality features, person cache. | – |
| `gymnastics.fusion.cycle_aware` | **Active model.** Cycle-aware dual-view fusion (transformer encoders, FiLM, cross-view reliability) trained with Lightning and configured with Hydra. | `gymnastics fuse cycle-aware` |
| `gymnastics.fusion.archive.rotation_aware` | Archived paper model (frozen 2026-09-19); train/infer/evaluate kept for reproducing published tables. | `gymnastics fuse rotation-aware` |
| `gymnastics.analysis` | Compute metrics, compare methods, create reports, and visualize results. | `gymnastics analyze` |
| `gymnastics.calibration` | Calibrate cameras from local image/video inputs. | `gymnastics calibrate` |
| `gymnastics.common` | Canonical project paths, geometry helpers, and MHR70 metadata. | Imported by other domains |

## Boundaries

- Domain packages can depend on `gymnastics.common`.
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
