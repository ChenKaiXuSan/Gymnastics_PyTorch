# Module Map

All active code belongs to the `gymnastics` package under `src/`. The
package is organised as four pipeline stages plus supporting packages.

| Stage | Package | Responsibility | Primary command |
|---|---|---|---|
| ① | `gymnastics.pose_estimation` | SAM3D-Body inference on paired face/side videos. | `gymnastics sam3d` |
| ② | `gymnastics.cycle_alignment` | Align face/side timelines, segment movement cycles, write cycle records. | `gymnastics align` |
| ③ | `gymnastics.pseudo_gt` | Camera calibration, per-person extrinsics, triangulated 3D pseudo-reference (evaluation only). | `gymnastics calibrate`, `gymnastics triangulate` |
| ④ | `gymnastics.fusion` | **The proposed model.** Cycle-aware dual-view fusion (transformer encoders, FiLM, cross-view reliability) trained with Lightning and configured with Hydra. | `gymnastics fuse cycle-aware` |
| support | `gymnastics.keypoints` | Shared 3D-keypoint representation: `PosePairTrial`, `SkeletonSpec`, canonical body frame, trunk/quality features, person cache. | – |
| support | `gymnastics.baselines` | Deterministic fusion methods and classical baselines. | `gymnastics fuse deterministic` |
| support | `gymnastics.benchmarks` | FreeMan and Unity benchmarks. | `gymnastics benchmark ...` |
| support | `gymnastics.analysis` | Metrics, comparisons, reports, cohort statistics. | `gymnastics analyze`, `gymnastics cohort-cycle` |
| support | `gymnastics.common` | Canonical project paths and MHR70 metadata. | Imported by other packages |
| archive | `gymnastics.archive.rotation_aware` | Frozen paper model (2026-09-19); kept for reproducing published tables. | `gymnastics fuse rotation-aware` |

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
