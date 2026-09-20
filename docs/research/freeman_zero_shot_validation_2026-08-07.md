# FreeMan Zero-Shot Validation Report (2026-08-07)

## Purpose

The manuscript's evidence for the rotation-aware fusion pipeline previously
had two layers: the private 137-person trunk-rotation cohort (triangulated
pseudo-reference) and the Unity synthetic benchmark (exact cameras). Neither
is public human data, leaving the obvious reviewer question — do the
conclusions hold outside data we constructed ourselves? This experiment
answers it with a strictly zero-shot, domain-matched evaluation on the public
FreeMan dataset (Wang et al., CVPR 2024), and simultaneously supplies the
multi-seed robustness evidence the private A4–A9 experiment lacks (seed 0
only there).

The claim under test is ranking transfer, not absolute accuracy: FreeMan's 3D
reference is itself a markerless multi-view reconstruction, so agreement with
it certifies cross-dataset consistency against an external reference.

## Cohort and selection

- 10 subjects, 552 sessions (60% of the 912 released sessions), frame stride 1.
- Subject 1 (pipeline-continuity baseline) plus the nine subjects with the
  strongest trunk-rotation content in the release, ranked from the public
  reference annotations by top-150 twist-session membership and per-subject
  top-five twist range (`trunk_twist_ranking.csv`). One candidate (subject 31)
  was skipped for incomplete twist annotations. Selection reads no method
  output and alters no method parameter; it scopes the claim to the
  trunk-rotation motion domain that the paper targets.

## Protocol

1. Per session, select the two of eight calibrated views whose horizontal
   optical axes are closest to 90° (achieved mean absolute deviation 1.606°
   over 552 pairs). View roles are assigned by the selector.
2. Run SAM3D-Body independently per view at every frame. The estimator
   consumes no dataset calibration (intrinsics estimated from video, fixed
   per session view).
3. Fuse with 14 methods: 2 single views, 2 naive baselines (no rotation
   normalization), 6 deterministic normalized methods, and the zero-shot
   rotation-aware checkpoints A4, A5 (seed 0) and A6 (seeds 0/1/2), all
   trained on the private data and frozen. The pseudo-reference-fitted
   per-joint weighting variant has no fitting target here and is excluded as
   a diagnostic.
4. Evaluate on the 17 joints shared between MHR70 and the FreeMan skeleton,
   under two protocols: session-level Sim3 MPJPE (one similarity transform
   per session — strict, keeps per-frame monocular depth drift) and per-frame
   PA-MPJPE (Procrustes — isolates pose shape). Subject-balanced aggregation;
   detection failures recorded explicitly (coverage ≥ 99.7% per method).
5. FreeMan reference coordinates are centimetres (verified from anatomical
   segment lengths); `reference_scale_to_m: 0.01`.

No FreeMan data influenced training, checkpoint selection, view pairing,
temporal alignment, or any hyperparameter.

## Results (subject-balanced, 10 subjects / 552 sessions)

| Group | Method | Sim3 MPJPE (mm) | PA-MPJPE (mm) |
|---|---|---:|---:|
| Single view | View A | 360.333 | 106.852 |
| | View B | 361.985 | 106.342 |
| Naive fusion | World-coordinate average | 368.342 | 129.823 |
| | Root translation to view A | 368.489 | 129.823 |
| Deterministic | Body-frame average | 362.521 | 103.184 |
| | Sim3, stable joints | 362.554 | 103.167 |
| | Sim3, stable + body-part weights | 362.551 | 103.212 |
| | Sim3, all joints | 364.198 | 103.265 |
| | Sim3, stable + smoothed side stream | 362.280 | 103.000 |
| | Sim3, stable + smoothed fusion | 362.187 | 102.898 |
| Rotation-aware | A4 spatial objectives | 364.363 | 103.114 |
| (zero-shot) | A5 + rotation/temporal | 364.218 | 103.199 |
| | A6 full model (3 seeds, mean ± SD) | 364.815 ± 0.738 | 103.317 ± 0.048 |

### Findings

1. **The private-data ordering transfers zero-shot.** On PA-MPJPE, every
   coordinate-normalizing fusion method (102.9–103.4 mm) beats both single
   views (106.3/106.9 mm), and the naive baselines that skip rotation
   normalization collapse to 129.8 mm. Normalization, not the choice among
   normalized methods, carries most of the benefit — the RQ1 conclusion
   reproduced on public data.
2. **Learned ≈ deterministic, again.** A4/A5/A6 sit inside the deterministic
   cluster, consistent with the private RQ3 result that the learned model
   matches but does not beat arithmetic fusion on position agreement.
3. **Multi-seed stability.** The three A6 seeds agree within 0.05 mm
   PA-MPJPE — the first across-seed evidence in the project.
4. **The strict Sim3 protocol is a common-mode floor.** All methods,
   including single views, fall in a 360–368 mm band with single views
   nominally lowest: one similarity per session cannot absorb per-frame
   monocular depth drift, which both input streams share, so fusion can
   neither help nor harm under this metric. Both protocols are reported to
   keep this failure mode visible. (A session-median-intrinsics ablation
   ruled out per-frame FoV-estimation jitter as the drift cause.)

## Engineering notes

- Ran on Pegasus (1× H100 PCIe + 48 cores per node), one subject per qsub
  job, 24 h queue limit; subject 22 (117 sessions) needed one cache-resumed
  resubmission.
- A batched inference path (32-frame windows through detection, full-mode
  SAM3D including hand decoders; per-session-view intrinsics; decode-prefetch
  thread) replaced the per-frame loop after a parity validation: mean 0.85 mm
  / median 0.38 mm keypoint difference on the 17 evaluation joints, identical
  failed-frame sets, end-to-end metrics identical to two decimals. Observed
  ~5–7× throughput at stride 1 (~0.2–0.35 s/frame·view vs ~1.05 s).
- Cache identities are keyed by the SAM3D config hash; the per-frame caches
  were reused under an audited `identity_aliases` declaration rather than by
  rewriting metadata.

## Artefacts

- Aggregate report: `local/runs/freeman_benchmark_cluster/report/freeman_benchmark_report.md`
- Per-session/joint/method/subject CSVs + paired statistics:
  `local/runs/freeman_benchmark_cluster/evaluation/`
- Manuscript integration: `paper/image_and_vision_computing/`
  (protocol §FreeMan, results RQ5, `tables/freeman_zero_shot.tex`)
- Configs: `src/configs/benchmarks/freeman_cluster.yaml`,
  `src/configs/pose_estimation/sam3d_body_freeman_batched.yaml`
- Job script: `pegasus/benchmark_freeman_subject_qsub.sh`

## Remaining work

- 30 remaining subjects (~175 GPU-h batched) to upgrade the domain-matched
  claim to full-release evidence; kept as rebuttal ammunition for now.
- Marker-based (or IMU) validation for absolute accuracy remains outstanding
  and is outside FreeMan's capability.
