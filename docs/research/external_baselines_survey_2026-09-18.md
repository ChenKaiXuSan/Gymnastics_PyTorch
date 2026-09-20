# External baseline survey for the SR fusion paper (2026-09-18)

Purpose: the current experiment matrix in `paper/sports_engineering/` compares
only internal variants (single views, naive averaging, coordinate-system
variants, the A4–A9 ablation ladder, estimated-rotation comparators).
This note lists published methods that can serve as external baselines, split
into those that run in the paper's exact setting (per-view monocular 3D, known
temporal sync, no calibration, no 3D labels) and those that only serve as
calibrated upper bounds. Licences and repository status were checked on
2026-09-18.

## Setting recap

- Input: two SAM 3D Body sequences (MHR70, 70 joints) per session, face and
  side view, synchronized, uncalibrated.
- Evaluation: private triangulated pseudo-reference (14 held-out people) and
  FreeMan (10 subjects, 552 sessions, 17 shared joints, Sim3 and PA protocols).
- FreeMan ships calibration, so calibrated methods can be run there as upper
  bounds even though the paper's method does not use it.

## A. Classical / non-learned fusion

| Method | Reference | Code | Inputs | Runs uncalibrated? | Effort |
|---|---|---|---|---|---|
| Confidence-weighted Kalman skeleton fusion | Moon et al., Int J Adv Robot Syst 2016, DOI 10.5772/62415; Lee et al., Sensors 2022, DOI 10.3390/s22093155 | none (re-implement) | per-view 3D joints + confidence, rigid registration | yes (replace their marker registration with Procrustes/pelvis frame) | 0.5–1 day |
| Per-frame Procrustes/Umeyama + robust averaging (geometric median / Huber) | Umeyama, TPAMI 1991 | numpy | per-view 3D | yes | hours |
| Anipose RANSAC triangulation + spatiotemporal filtering | Karashchuk et al., Cell Reports 2021 | github.com/lambdaloop/anipose (BSD-2) | 2D + calibration | no (upper bound) | 1 day |
| Extrinsics from monocular 3D pose sets, then triangulation | Lee, Nishino, Nobuhara, IEEE RA-L 2025, arXiv 2502.12546; Takahashi et al., CVPRW 2018 | none | per-view monocular 3D | yes, but must be re-implemented | 2–3 days |
| DMMR (joint SMPL + extrinsics optimisation) | Huang et al., 3DV 2021, arXiv 2110.10355 | github.com/boycehbz/DMMR (MIT) | 2D keypoints, no calibration | yes; SMPL joints need mapping | 1–2 days |
| Generalizable Human Pose Triangulation | Bartol et al., CVPR 2022, arXiv 2110.00280 | github.com/kristijanbartol/general-3d-humans (MIT); weights gated | 2D keypoints; uncalibrated path "WIP" | risky | 2–4 days |

## B. Self- / weakly supervised multi-view methods

| Method | Venue | Code / licence | Inputs | Runs uncalibrated? | Effort |
|---|---|---|---|---|---|
| MetaPose (Usman et al.) | CVPR 2022, arXiv 2108.04869 | google-research/metapose (Apache-2.0, TF) + public checkpoints | per-view 2D → per-view monocular 3D + uncertainty → multi-view refinement, no calibration or 3D labels | yes; closest published analogue to post-estimation fusion | 2–4 days |
| MDVPose (Xu et al.) | ACM MM 2024 | github.com/iGame-Lab/MDVPose (no licence file) | per-view 2D keypoint sequences (Halpe-26); Procrustes multi-view consistency on MotionBERT | yes; wild-inference script exists | 1–2 days |
| Multi-view Pose Fusion for occlusion-aware 3D HPE (Bragagnolo et al.) | ECCVW 2024, arXiv 2408.15810 | method code not released | per-view monocular 3D skeletons + extrinsics; weights 1/reprojection error + limb-symmetry optimisation | needs calibration for weights; re-implementable with a calibration-free weight | 1–3 days |
| Two Views Are Better than One (Ingwersen et al.) | CVPRW 2025, arXiv 2311.12421 | github.com/ChristianIngwersen/SportsPose | two views at train only (sequence Procrustes consistency), monocular at test | training recipe, not fusion; sports-domain evidence | 2–3 days |
| CanonPose / ElePose (Wandt et al.) | CVPR 2021 / 2022 | github.com/bastianwandt/CanonPose, ElePose | multi-view 2D, no calibration; canonical-frame lifter | yes as "learned lifter + canonical average" | 2–3 days |
| Probabilistic Triangulation (Jiang et al.) | ICCV 2023, arXiv 2309.04756 | github.com/bymaths/probabilistic_triangulation | images / heatmaps, camera pose as distribution | yes but image-based | 3–5 days |
| UPose3D (Davoodnia et al.) | ECCV 2024, arXiv 2404.14634 | none | 2D multi-view + temporal | cite only | – |
| Cross-View Self-Fusion (Kim et al.) | ACCV 2022 | none | multi-view images | cite only | – |
| EpipolarPose (Kocabas et al.) | CVPR 2019 | github.com/mkocabas/EpipolarPose (non-commercial) | multi-view images at train | training recipe | 2–3 days |
| SelfPose3D (Srivastav et al.) | CVPR 2024, arXiv 2404.02041 | github.com/CAMMA-public/SelfPose3d | multi-view images + calibration | upper bound only | 3–5 days |
| Learnable Triangulation (Iskakov et al.) | ICCV 2019 | github.com/karfly/learnable-triangulation-pytorch (MIT) | images + calibration | upper bound only | 1–2 days |
| Cross View Fusion (Qiu et al.) | ICCV 2019 | github.com/microsoft/multiview-human-pose-estimation-pytorch (MIT) | images + calibration | upper bound only | 2–3 days |
| Unconstrained MV-HPE with algebraic priors (Qin et al.) | arXiv 2604.24312 (2026) | none | 4+ views, uncalibrated | cite as state of the art | – |

## C. Temporal refinement / denoising networks (generic learned fusion)

| Method | Venue | Code / licence | Note | Effort |
|---|---|---|---|---|
| SmoothNet (Zeng et al.) | ECCV 2022, arXiv 2112.13715 | github.com/cure-lab/SmoothNet (Apache-2.0) | noisy 3D sequence in, refined 3D out; windows 8–64; public checkpoints. Feed the body-frame average; zero-shot and retrained variants | 0.5–2 days |
| DeciWatch (Zeng et al.) | ECCV 2022, arXiv 2203.08713 | github.com/cure-lab/DeciWatch (Apache-2.0) | DenoiseNet stage usable alone | 1–2 days |
| MotionAGFormer (Mehraban et al.) | WACV 2024 | github.com/TaatiTeam/MotionAGFormer (Apache-2.0) | generic sequence model with the same two-view 3D input as A4–A6, trained with the same self-supervised losses | 2–3 days |
| MotionBERT (Zhu et al.) | ICCV 2023 | github.com/Walter0807/MotionBERT (Apache-2.0) | 2D→3D lifter; supervised use would break the label-free protocol | 2–3 days |
| VideoPose3D (Pavllo et al.) | CVPR 2019 | github.com/facebookresearch/VideoPose3D (CC BY-NC) | 1D TCN; simplest "any temporal network" control | 1–2 days |
| D3PRefiner (Yan et al.) | arXiv 2401.03914 | none | cite only | – |

## D. Foundation / recent mesh recovery

| Method | Venue | Code | Note |
|---|---|---|---|
| MUC: Mixture of Uncalibrated Cameras (Zhu et al.) | AAAI 2025, arXiv 2403.05055 | github.com/AbsterZhu/MUC (no licence file) | per-view SMPLer-X predictions + learned per-view per-joint weights; explicitly "instead of averaging"; closest mesh-level analogue. 3–5 days |
| EasyRet3D (Yin et al.) | WACV 2025 | not released | auto-calibration + stitching of per-view SMPL; cite only |
| Multi-HMR, WHAM, Human3R, 4DHumans/TokenHMR | ECCV 2024 / CVPR 2024 / 2025 / 2024 | various | monocular only; usable as alternative per-view estimators, not fusion baselines |
| Multi-camera self-calibration in sports mocap (Yang et al.) | arXiv 2604.17567 (2026) | project page only | 2D keypoints + known stick length; relates to the estimated-rotation comparator |

## E. Sports-science two-camera systems

| System | Reference | Code | Note |
|---|---|---|---|
| OpenCap | Uhlrich et al., PLOS Comput Biol 2023 | github.com/opencap-org/opencap-core (Apache-2.0) | ≥2 iPhones + checkerboard calibration; triangulation → LSTM augmenter → OpenSim IK. Calibrated upper bound; outputs joint angles. 3–5 days |
| Pose2Sim | Pagnon et al., JOSS 2022; Sensors 2021/2022 | github.com/perfanalytics/pose2sim (BSD-3) | two-camera supported, confidence-weighted triangulation, Butterworth/Kalman filters. Best calibrated classical upper bound on FreeMan. 1–2 days |
| Theia3D | commercial | closed | cite validation studies only |

## FreeMan-reported baselines (Wang et al., CVPR 2024, arXiv 2309.05073)

- Reference: HRNet-w48 2D → calibrated 8-view triangulation → smoothness and
  bone-length optimisation → SMPLify; 17 COCO joints.
- Single-view 2D→3D lifting, FreeMan→FreeMan: SimpleBaseline 90.53 MPJPE /
  54.17 PA-MPJPE; MHFormer 93.00 / 63.50 (mm).
- Multi-view (4 views, VoxelPose, 13 joints): 26.07 mm MPJPE.
- No two-view or uncalibrated numbers are reported; the 90 mm single-view vs
  26 mm four-view calibrated gap brackets the paper's regime.

## Ranked shortlist

1. Confidence-weighted Kalman fusion (Moon 2016 / Lee 2022 re-implementation).
   Runs in the exact uncalibrated setting; the canonical classical answer to
   "why not just filter?"; under one day.
2. Per-joint weighted mean + limb-symmetry refinement (Bragagnolo 2024 style),
   with a calibration-free weight (uncalibrated) and FreeMan reprojection
   weights (calibrated upper bound); 1–3 days.
3. SmoothNet zero-shot and retrained on the body-frame average; standard
   plug-and-play temporal refiner; about one day.
4. MetaPose; published learned aggregation of per-view monocular estimates
   without calibration or 3D labels; strongest external deep-learning baseline;
   2–4 days (TensorFlow, format adapters).
5. Pose2Sim (or plain DLT + Kalman) with FreeMan calibration; calibrated
   upper bound from the sports-engineering community; 1–2 days.
6. MDVPose or MUC as a second recent deep-learning baseline; MDVPose is easier,
   MUC is closer in spirit; 2–5 days.

Calibrated upper bounds only: Learnable Triangulation, Cross View Fusion,
SelfPose3D, VoxelPose (already in the FreeMan paper), OpenCap, Anipose.
Cite-only (no code): UPose3D, Kim 2022, EasyRet3D, Lee/Nishino/Nobuhara 2025,
Qin 2026, D3PRefiner, Theia3D.

## Implementation status (2026-09-18)

Implemented as shared label-free methods in
`src/gymnastics/baselines/classical_baselines.py`, registered in
`experiment_matrix.BASELINE_METHODS` and dispatched by the private matrix,
the FreeMan benchmark and the Unity benchmark alike:

| Method id | Row | Notes |
|---|---|---|
| `kalman_body_fusion` | Kalman filter fusion | constant-velocity KF per joint axis in the pelvis-centred body frame; two measurements per frame; per-view per-joint measurement variance from robust second differences; process noise from low-passed acceleration |
| `kalman_rts_body_fusion` | Kalman RTS smoother fusion | same filter plus Rauch–Tung–Striebel backward pass |
| `jitter_weighted_body_average` | Reliability-weighted average | per-frame per-joint weights = inverse local second-difference RMS (Bragagnolo-style reliability without calibration) |
| `butterworth_body_average` | Butterworth 6 Hz | 4th-order zero-phase low-pass on the body-frame average (standard biomechanics filter) |
| `smoothnet_body_average` | SmoothNet zero-shot | public Human3.6M/FCN 3D checkpoint (window 32) applied to the body-frame average in mm; weights under `local/weights/smoothnet/` |

Runs: private `local/runs/fuse_external_baselines/` (137 people, compact
NPZ per method, re-evaluated on the 14 held-out people by
`paper/sports_engineering/scripts/generate_main_matrix.py`); FreeMan
`local/runs/freeman_external_baselines/` (`gymnastics benchmark freeman-train
baselines`); Unity `local/runs/unity_benchmark/` (`gymnastics benchmark unity
fuse --config configs/benchmarks/unity_hp260146.yaml --method ...` then
`evaluate`; previous evaluation backed up as `evaluation.bak_2026-09-18`).

The cross-dataset main matrix (rows = method blocks, columns = private /
FreeMan / Unity) is generated by `generate_main_matrix.py` into
`paper/sports_engineering/generated/main_matrix.{csv,tex}`.

Planned next: a plain-TCN learned baseline (B1) trained with the same
self-supervised losses on the private data (and FreeMan folds), and MetaPose /
Pose2Sim as calibrated or external deep-learning references if time permits.
