# Estimated-Extrinsics Fusion Results

Date: 2026-07-29

## Experiment

- People: 137
- Evaluated split cycles: 928
- Evaluation: sequence-level similarity-aligned MPJPE to the regenerated
  triangulated pseudo-reference
- External geometry:
  `local/runs/analysis/extrinsics/estimated_extrinsics.json`
- Aligned input cache:
  `local/runs/fuse_rotation_aware/cache`
- Result CSV:
  `local/runs/fuse_extrinsic_baselines/metrics_by_person.csv`
- Camera convention: `X_side = R X_face + t`
- Fusion convention: pelvis-centre the root-relative side pose, map it to face
  axes with the row-vector operation `X_side_centered @ R`, restore the face
  pelvis, and exclude camera translation.

The immutable split-cycle cache was verified against the original per-frame
SAM3D loading path for people 1, 36, 70, and 104. On all cached frame pairs,
Extrinsic-R fused coordinates were exactly equal (maximum absolute difference
0).

## Main results

| Group | Method | Mean | SD | Median | 95% bootstrap CI |
|---|---|---:|---:|---:|---:|
| Uses estimated extrinsics | Extrinsic-R average | **0.062031** | 0.016571 | 0.059187 | [0.059266, 0.064807] |
| Uses estimated extrinsics | Extrinsic-R quality average | 0.063251 | 0.016794 | 0.060174 | [0.060512, 0.066134] |
| No camera extrinsics | Body-frame average | 0.064045 | 0.016092 | 0.062259 | [0.061481, 0.066774] |

Extrinsic-R average is 3.15% below body-frame average and is lower for 118/137
people. Its paired difference is -0.002015 (10,000-sample bootstrap 95% CI
[-0.002383, -0.001655]; Holm-adjusted Wilcoxon
`p = 2.66e-16`).

Extrinsic-R quality average is 1.24% below body-frame average and is lower for
89/137 people. Its paired difference is -0.000794 (95% CI
[-0.001201, -0.000387]; Holm-adjusted `p = 1.96e-4`).

Quality weighting is worse than equal Extrinsic-R averaging by 0.001220 on
average (95% CI [0.001038, 0.001406]) and is better for only 19/137 people
(Holm-adjusted `p = 2.50e-19`). The equal-weight method is therefore the
preferred simple extrinsic-assisted baseline.

## Calibration dependence

- Holdout reprojection error versus Extrinsic-R MPJPE:
  Spearman `rho = 0.354`, `p = 2.19e-5`.
- Per-person extrinsics: 129 people, mean Extrinsic-R MPJPE 0.061095, mean
  holdout reprojection error 6.56 px.
- Cluster-consensus fallback: 8 people, mean Extrinsic-R MPJPE 0.077110, mean
  holdout reprojection error 18.63 px.

The fallback comparison is descriptive because the groups are defined by
calibration success and are highly imbalanced. It nevertheless agrees with the
continuous correlation: poorer calibration is associated with poorer fused
agreement.

## Interpretation

The estimated camera rotation provides a small but consistent improvement over
the best leakage-free pose-only deterministic baseline. The result does not
establish absolute 3D accuracy: the extrinsics and triangulated evaluator use
shared upstream paired-video evidence. It should be reported in a separate
camera-extrinsics table and treated as a calibration-assisted comparator, not as
the new calibration-free paper mainline.
