# Estimated-Extrinsics Fusion Results

Date: 2026-07-29
Unified-evaluator update: 2026-08-01

## Experiment

- People: 137
- Evaluated split cycles: 928
- Evaluation: one similarity alignment per cycle followed by framewise hip
  centring against the regenerated triangulated pseudo-reference
- External geometry:
  `local/runs/analysis/extrinsics/estimated_extrinsics.json`
- Aligned input cache:
  `local/runs/fuse_rotation_aware/cache`
- Result CSV:
  `paper/sports_engineering/generated/extrinsic_person_metrics_matched_137.csv`
- Camera convention: `X_side = R X_face + t`
- Fusion convention: pelvis-centre the root-relative side pose, map it to face
  axes with the row-vector operation `X_side_centered @ R`, restore the face
  pelvis, and exclude camera translation.

The immutable split-cycle cache was verified against the original per-frame
SAM3D loading path for people 1, 36, 70, and 104. On all cached frame pairs,
Extrinsic-R fused coordinates were exactly equal (maximum absolute difference
0).

## Main results

| Group | Method | Mean MPJPE (mm) | SD (mm) |
|---|---|---:|---:|
| Uses estimated extrinsics | Extrinsic-R average | **63.074** | 14.962 |
| Uses estimated extrinsics | Extrinsic-R quality average | 64.133 | 15.301 |
| No camera extrinsics | Body-frame average | 65.249 | 14.523 |

Extrinsic-R average is 3.33% below body-frame average and is lower for 109/137
people. Its paired difference is -2.175 mm (10,000-sample bootstrap 95% CI
[-2.622, -1.730] mm; Holm-adjusted Wilcoxon `p = 1.24e-14`).

Extrinsic-R quality average is 1.71% below body-frame average and is lower for
89/137 people. Its paired difference is -1.116 mm (95% CI
[-1.613, -0.632] mm; Holm-adjusted `p = 4.92e-5`).

Quality weighting is worse than equal Extrinsic-R averaging by 1.059 mm on
average and is better for only 21/137 people. The equal-weight method is
therefore the preferred simple extrinsic-assisted baseline.

## Calibration dependence

- Holdout reprojection error versus Extrinsic-R MPJPE:
  Spearman `rho = 0.347`, `p = 3.35e-5`.
- Per-person extrinsics: 129 people, mean Extrinsic-R MPJPE 62.050 mm, mean
  holdout reprojection error 6.56 px.
- Cluster-consensus fallback: 8 people, mean Extrinsic-R MPJPE 79.592 mm, mean
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
