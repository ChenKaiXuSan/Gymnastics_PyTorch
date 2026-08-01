# Estimated-Extrinsics Fusion Results

Date: 2026-07-29
Unified-evaluator update: 2026-08-01

## Experiment

- Dataset inventory: 137 people and 928 split cycles
- Primary paper comparison: the fixed 14-person held-out test split
- Secondary analysis: all 137 people, including training and validation people
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

## Held-out main-paper results

| Group | Method | Mean MPJPE (mm) | SD (mm) |
|---|---|---:|---:|
| Uses estimated extrinsics | Extrinsic-R average | **59.081** | 9.481 |
| Uses estimated extrinsics | Extrinsic-R quality average | 59.595 | 9.276 |
| No camera extrinsics | Body-frame average | 60.829 | 8.964 |

Extrinsic-R average is 2.87% below body-frame average and is lower for 10/14
held-out people. Its paired difference is -1.747 mm (10,000-sample participant
bootstrap 95% CI [-2.921, -0.474] mm; Holm-adjusted Wilcoxon `p = 0.0491`).

Extrinsic-R quality average is 2.03% below body-frame average and is lower for
10/14 held-out people. Its paired difference is -1.233 mm (95% CI
[-2.383, -0.000] mm; Holm-adjusted `p = 0.0785`). The bootstrap interval and
Holm-adjusted test straddle conventional decision boundaries, so this row is
reported without a significance claim.

The equal-weight method has the lower held-out descriptive mean and is retained
as the camera-assisted comparator.

Table 1's A2 arithmetic output (60.845 mm) and Table 2's regenerated
`avg_body_current` baseline (60.829 mm) cover the same 14 people, matched
frames, valid points and evaluator. They are independently materialized
sequences, so the observed 0.0165-mm aggregate difference is retained rather
than forcing the two rows to be numerically identical.

## Secondary all-participant results (Online Resource)

| Group | Method | Mean MPJPE (mm) | SD (mm) |
|---|---|---:|---:|
| Uses estimated extrinsics | Extrinsic-R average | **63.074** | 14.962 |
| Uses estimated extrinsics | Extrinsic-R quality average | 64.133 | 15.301 |
| No camera extrinsics | Body-frame average | 65.249 | 14.523 |

Across all 137 people, Extrinsic-R average is 3.33% below body-frame average
and is lower for 109 people. Its paired difference is -2.175 mm (95% CI
[-2.622, -1.730] mm; Holm-adjusted Wilcoxon `p = 1.24e-14`). The quality
average is 1.71% lower, improves 89 people and has a paired difference of
-1.116 mm (95% CI [-1.613, -0.632] mm; Holm-adjusted `p = 4.92e-5`). These
values are descriptive because the cohort includes people used to train or
select the learned methods; they are not substituted for the 14-person primary
comparison.

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

On the fixed held-out cohort, estimated camera rotation has a small advantage
over the pose-only deterministic baseline. The all-participant analysis has the
same direction but is secondary. Neither result establishes absolute 3D
accuracy: the extrinsics and triangulated evaluator use shared upstream
paired-video evidence. The method is therefore treated as a camera-assisted
comparator, not as the new calibration-free paper mainline. With only 14
held-out participants and two adjusted comparisons, the borderline Extrinsic-R
result should be interpreted as modest evidence requiring independent
replication rather than a definitive ranking.
