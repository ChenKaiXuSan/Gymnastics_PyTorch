# Source and Evidence Audit

This file records the provenance of claims used in the manuscript. It is a
working audit, not supplementary material.

## Repository Evidence

| Claim | Artifact | Population | Alignment | Reference | Seeds | Status |
|---|---|---|---|---|---:|---|
| Deterministic matrix | `local/runs/fuse_experiments/metrics_by_person.csv` | 137 people, 9 methods | Evaluator-specific hip centering | Same-video triangulated pseudo-reference | n/a | Verified; pseudo-reference-fitted joint weights are marked leaky |
| Primary learned MPJPE | `local/runs/analysis/project_results/learned_results_by_split.csv` | Held-out test, 14 people | One static sequence-level Sim3, then hip centering | Same-video triangulated pseudo-reference | 1 | Verified; A6 60.78 mm, A2/A5 indistinguishable |
| Paired learned tests | `local/runs/analysis/project_results/learned_test_comparisons.csv` | 14 paired people | Same as learned MPJPE | Same-video triangulated pseudo-reference | 1 | Verified; Holm correction across nine A6 comparisons |
| A3-relative retention | Same learned split CSV; evaluator `reference_kpts` | 14 test people for primary table | No external alignment in denominator | A3 quality-weighted base saved by inference | 1 | Verified; not triangulated retention |
| Validation corruption | Rotation-aware validation reports | 27 validation people | Clean-to-corrupted self displacement | Each method's clean output | 1 | Verified; not a test-set accuracy result |
| Unity zero-shot | `local/runs/unity_benchmark/report/results.json` | 199 samples, 3 sequences, one avatar/environment | One Sim3 per sequence | Unity native 3D, Unity16 | 1 private checkpoint | Verified; A6 178.506 mm, direct average 166.537 mm |
| Unity calibrated learning | `local/runs/unity_benchmark/extrinsic_learning/report/results.json` | Two direction-held-out folds | One Sim3 per sequence | Unity native 3D, Unity16 | 3 | Verified; input regimes reported separately |
| Cohort primary models | `local/runs/cohort_cycle/analysis/statistics_midcycle_v2/core_mixed_models.csv` | 137 people, 928 cycles | n/a | OOF A6 model-derived descriptors | OOF seed 0 | Verified; cycle position centered at 0.5 |
| Cohort source sensitivity | `local/runs/cohort_cycle/analysis/statistics_midcycle_v2/sensitivity_mixed_models.csv` | 137 people, 928 cycles per source | n/a | OOF A6, face, side, deterministic | n/a | Verified; same mixed-model estimand, 32/32 converged |

The private triangulated stream is derived from the same two videos and is not
independent capture. Unity native 3D is independent synthetic reference data,
but its one-avatar/environment design does not establish population validity.

## Literature Evidence

Primary paper pages or DOI records were checked for the following claims:

- SAM 3D Body input representation and MHR output:
  <https://arxiv.org/abs/2602.15989>
- Learnable and confidence-weighted triangulation:
  <https://openaccess.thecvf.com/content_ICCV_2019/html/Iskakov_Learnable_Triangulation_of_Human_Pose_ICCV_2019_paper.html>
- Cross-view image/2D feature fusion followed by 3D recovery:
  <https://openaccess.thecvf.com/content_ICCV_2019/html/Qiu_Cross_View_Fusion_for_3D_Human_Pose_Estimation_ICCV_2019_paper.html>
- Camera-disentangled calibrated multi-view fusion:
  <https://openaccess.thecvf.com/content_CVPR_2020/html/Remelli_Lightweight_Multi-View_3D_Pose_Estimation_Through_Camera-Disentangled_Representation_CVPR_2020_paper.html>
- Generalizable and uncalibrated triangulation:
  <https://openaccess.thecvf.com/content/CVPR2022/html/Bartol_Generalizable_Human_Pose_Triangulation_CVPR_2022_paper.html>
  and
  <https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_Probabilistic_Triangulation_for_Uncalibrated_Multi-View_3D_Human_Pose_Estimation_ICCV_2023_paper.html>
- Closest self-supervised cross-view method, which fuses 2D observations to
  train a monocular estimator:
  <https://openaccess.thecvf.com/content/ACCV2022/html/Kim_Cross-View_Self-Fusion_for_Self-Supervised_3D_Human_Pose_Estimation_in_the_ACCV_2022_paper.html>
- Multi-view self-supervision with unknown or known calibration:
  <https://openaccess.thecvf.com/content/CVPR2022/html/Usman_MetaPose_Fast_3D_Pose_From_Multiple_Views_Without_3D_Supervision_CVPR_2022_paper.html>
  and
  <https://openaccess.thecvf.com/content/CVPR2024/html/Srivastav_SelfPose3d_Self-Supervised_Multi-Person_Multi-View_3d_Pose_Estimation_CVPR_2024_paper.html>
- Temporal convolutional pose estimation:
  <https://openaccess.thecvf.com/content_CVPR_2019/html/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.html>
- OpenCap calibrated smartphone pipeline and Mocap validation:
  <https://doi.org/10.1371/journal.pcbi.1011462>
- Continuous rotation representations:
  <https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html>
- Similarity-transform estimation:
  <https://doi.org/10.1109/34.88573>

## Venue Evidence

- Neurocomputing Guide for Authors:
  <https://www.sciencedirect.com/journal/neurocomputing/publish/guide-for-authors>
- Official Elsevier CAS template archive:
  <https://assets.ctfassets.net/o78em1y1w4i4/5uFmLZJTPDMAUjFnHRpjj8/6f19a979146eb93263763d87a894ab0d/els-cas-templates.zip>
- Downloaded archive SHA-256:
  `36d97da01c6bbd134f315bff6c3de553735e2550444a6ddd4f869ddc67a20757`

## Unresolved Before Submission

- Add repeated seeds for the full private held-out learned comparison.
- Add diverse real public-benchmark or synchronized independent-capture
  validation; Unity is limited synthetic evidence.
- Finalize the institutional postal address, ethics/consent statement, data
  availability decision, author biography, and author photograph.
