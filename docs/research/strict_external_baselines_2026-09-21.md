# Strict external baselines (2026-09-21)

Package: `src/fusion/external/published/` (`python -m fusion external-published`),
job script `pegasus/external_trained_qsub.sh`, outputs
`local/runs/external_published/<method>/<dataset>/summary_<mode>.json`.

## Why a second external-baseline family

The earlier "external learned" rows (`fusion/external/`: TCN, SmoothNet, MetaPose-
style MLP, MUC-style weights) swap only the network inside our pipeline: our
canonicalisation, our phase windows, our losses, our base fusion. They test
architectures, not published methods, and they are kept for that purpose
(`docs/research/main_matrix_external_baselines_2026-09-19.md`). The rows
below run each published method with **its own data flow and its own recipe**
on the same two SAM3D views, and only the *evaluation* is shared: the method's
3D output replaces both views on the trial's frames through the
`trial_transform` hook of the DataModules, so folds, phase windows, test
subjects and the metric (per-frame PA-MPJPE on the major joints the method
predicts) are exactly those of the model protocol
(`docs/cycle_aware_fusion.md` §1.5).

Rule for the main table (user decision 2026-09-21): **every learned external
baseline is trained on our data, per fold, with its released recipe. No
zero-shot rows in the main table.** Zero-shot numbers (released checkpoints)
go to an appendix only.

## Shared input: SAM3D 2D keypoints

All methods lift 2D detections, so the input is the SAM3D-Body 2D MHR70
keypoints of each view (`keypoints2d.py`; private cache
`local/runs/external_published/keypoints2d/gymnastics/subject_<id>/{face,side}.npz`,
FreeMan / SportsPose read their benchmark caches). Layout conversions in
`mapping.py`: MHR70 → COCO-17 → Human3.6M-17 (VideoPose3D, MetaPose) or
MPII-16 (CanonPose); the reverse map scatters the predicted joints into the
MHR70 slots (legs, shoulders, elbows, wrists, neck, nose), so a method is
scored on the intersection of its skeleton with the 20 major joints (13 for
H36M-17 methods). SAM3D emits no detection confidences: validity (1 / 0)
is used where a method expects confidences (CanonPose) and a single
Gaussian per joint where it expects heatmaps (MetaPose).

## Methods

| # | Method | Supervision | How it is run here | Deviations from the release |
|---|---|---|---|---|
| 1 | **CanonPose** (Wandt et al., CVPR 2021) | self-supervised (reprojection + view-consistency + camera-consistency, no 3D labels, no calibration) | trained per fold on the training subjects' two-view 2D keypoints with the released `train.py` defaults (Adam 1e-4, batch 32, 100 epochs, milestones 30/60/90, γ 0.1, weight decay 1e-5, loss weights 1 / 1 / 0.1); the loop is reproduced verbatim as a function (`canonpose.py`); two-view result = average of the canonical poses of both views (`canonical_average`; `per_view` also reported) | skeleton-morphing network of Sec. 4.2 skipped (needs 2D GT); confidences = validity |
| 2 | **MetaPose** (Usman et al., CVPR 2022) | label-free (stage 1 = weak-perspective bundle adjustment with a GMM heatmap likelihood; stage 2 = network trained with the reprojection loss `fwd` and the `soln` losses to the stage-1 optimum) | stage 1 with the released objective/functions in a batched port validated against the official solver (`metapose_s1.py`); stage 2 trained per fold with the authors' `train_metapose.py` on the training subjects' frames, test subjects predicted (`metapose_s2.py --mode train`); the record field `pose3d`, which the script uses only for the early-stopping / checkpoint metric `val_pred_pmpjpe`, holds the stage-1 optimum, so model selection is label-free too | monocular initialisation from the official VideoPose3D lifter instead of EpipolarPose (any monocular lifter fills that role); heatmap GMM = one Gaussian at the SAM3D keypoint (σ = 2 % of the box); schedule capped (default 30 epochs × 4 stages, patience 5; released 300 × up to 10, patience 50) – recorded in the summary; 2 cameras |
| 3 | **MHFormer** (Li et al., CVPR 2022) | supervised (3D MPJPE) | released 81-frame configuration (`main.py`: Adam amsgrad 1e-3, x0.95 per epoch and x0.5 every 5 epochs, 19 epochs, batch 256, flip augmentation and flip TTA, centre frame, all window frames supervised); trained per fold on the training subjects' two views as monocular samples with the reference joints rotated into each camera (`mhformer.py`); **FreeMan and SportsPose only** (the private data has no independent 3D reference) | SAM3D 2D instead of CPN detections (missing joints interpolated in time, frames without reference masked); selection on the fold's validation subjects (release: test set); 81 instead of 351 frames (4x cheaper, also a released configuration) |
| 4 | **MDVPose** (Ma et al.; "multiple dynamic views via single-view pretraining with Procrustes alignment", the MotionBERT multi-view fine-tuning the user asked for as "two views") | supervised (MPJPE + 0.5 scale-normalised MPJPE + 20 velocity) + 0.002 Procrustes multi-view consistency | DSTformer from the MotionBERT H36M checkpoint (`walterzhu/MotionBERT` on Hugging Face, `local/checkpoints/motionbert`), released multi-view config (AdamW 3e-4, wd 0.01, x0.97 per epoch, 243-frame clips stride 81, flip aug/TTA), one batch = three clip pairs = the released batch of six clips, the consistency term inside each pair (`mdvpose.py`); FreeMan / SportsPose only | 30 epochs instead of 60 (9.7 GPU-h per fold otherwise); flip drawn per clip pair (the released loader flips cameras independently, contradicting its own multi-view term); selection on validation subjects; frame validity masks replace the release's padding marker |
| A | VideoPose3D (Pavllo et al., CVPR 2019) | supervised on Human3.6M | released `pretrained_h36m_detectron_coco.bin`, per view, both views Procrustes-averaged | zero-shot -> **appendix only**: FreeMan-repetitive 86.3 +- 6.3 mm (13 joints) |
| B | MetaPose released `ckpt/h36m/cam2` | -- | `--stage s2 --released` | zero-shot -> appendix only |

MUC (Lee et al.) was dropped: the release needs a OneDrive checkpoint and
an SMPL-X account, and its SMPL-X-space fusion does not map onto keypoint
inputs without rewriting the method.

## Environments and jobs

* CanonPose runs in the main env (`sam_3d_body`, torch).
* MetaPose stages 1–2 run in a TensorFlow env: `metapose_gpu`
  (TF 2.15.1 + CUDA 12 pip wheels, tfds 4.9.2, protobuf 4.25,
  tensorflow-metadata 1.14, numpy 1.26; `PYTHONNOUSERSITE=1` because
  `~/.local` pollutes it) or the CPU env `metapose` (TF 2.8);
  `metapose_pipeline.metapose_python()` picks `metapose_gpu` when present
  (`GYMNASTICS_METAPOSE_PYTHON` overrides). The released checkpoint
  reproduces 44.1 mm PMPJPE on the released `h36m/opt/2/test` records in
  that env (sanity check, 2026-09-21).
* Jobs: `qsub -A SKIING -q gpu -v METHOD=metapose,DATASET=freeman,STAGE=prepare::s1 pegasus/external_trained_qsub.sh`,
  then `STAGE=train,FOLDS=fold_01` per fold, then `STAGE=evaluate`;
  CanonPose: `STAGE=prepare`, `STAGE=train,FOLDS=...`, `STAGE=evaluate`.
  NQSV spools the job log until the job ends.

## Findings while running them

* **CanonPose converges to the mirrored world.** The self-supervised
  objective is invariant to a global reflection (mirrored pose with mirrored
  cameras re-projects identically under weak perspective). All five
  gymnastics folds trained fine (reprojection residual 4 % of the input
  scale, the two views' canonical poses agree to 31 mm) yet scored
  147-155 mm PA-MPJPE against the reference because the pose is mirrored
  (40 mm with the mirror undone). The release leaves the choice to its
  Human3.6M convention; here the one bit per trained model is set from the
  method's own input modality -- whether the canonical pose or its mirror
  agrees better with SAM3D's per-view 3D on the first 20 trials -- never
  from the reference (`CanonPoseTrialTransform`, logged as "reflection
  resolved").
* CanonPose released loop: 265 s/epoch on a FreeMan fold (374k frames,
  batch 32, one Python loop per subject and camera); the camera-consistency
  term vectorised with one within-subject permutation per camera and the
  released sum-of-subject-means normalisation (unit-tested against the loop)
  -> 97 s/epoch, 2.7 h per 100-epoch fold on an H100.
* MetaPose: the tfds record serialiser needs ~1 h per fold; records are now
  written once per subject in parallel (`--stage shards`) and folds
  recombine subjects by TFRecord concatenation with a 64-record random
  validation head (`train_metapose` validates on the first 64 records of
  its training split). The script keeps stage checkpoints at the literal
  `/tmp/best-model`; `metapose_launch.py` gives every run a private path
  (concurrent jobs on one node would overwrite each other). One FreeMan
  stage-1 optimum of 710k is degenerate (scale 40, collapsed pose) and is
  excluded from training. Gymnastics stage 0: 27 s/epoch, early-stopped
  after 16 epochs at val PMPJPE 0.032 bbox units.
* MHFormer-81 on a FreeMan fold: 1.5 M windows per epoch (both views,
  flips), 888 s/epoch -> 4.7 h per fold. MDVPose: 580 s/epoch with three
  clip pairs per batch (887 s with one) -> 30 epochs = 4.8 h per fold.

## Results (model protocol: 5 folds, phase windows, per-frame PA-MPJPE, mm; joints = major joints the skeleton covers)

| Method | Supervision | Gymnastics (private) | FreeMan-repetitive | SportsPose |
|---|---|---:|---:|---:|
| CanonPose, trained per fold, `canonical_average` | none | **30.0 ± 0.9** (13 j) | 53.4 ± 4.3 (12 j) | waits for the SAM3D cache |
| CanonPose, `per_view` | none | 37.6 ± 1.4 | 62.9 ± 3.7 | -- |
| MetaPose stage 1 (optimisation only, a published ablation) | none | 44.3 ± 0.9 (14 j) | 55.7 ± 4.8 (13 j) | -- |
| MetaPose stage 2 trained, README default `fwd` objective | none | 127.8 ± 51.4 | (stage 1 crashes in cuSOLVER `gesvd`) | -- |
| MetaPose stage 2 trained, `ts` (student of stage 1) | none | 76.9 ± 9.1 | folds trained by the other session, evaluation pending | -- |
| MHFormer-81, trained per fold, `procrustes_average` | 3D reference | n/a (no independent reference) | 44.1 ± 6.2 (13 j; folds 40.5 / 35.9 / 54.0 / 47.5 / 42.8) | -- |
| MHFormer-81, `per_view` | 3D reference | n/a | 51.0 ± 5.5 | -- |
| MDVPose, trained per fold (30 epochs), `procrustes_average` | 3D reference | n/a | **40.3 ± 6.5** (13 j; folds 35.7 / 33.1 / 51.4 / 43.5 / 38.0) | -- |
| MDVPose, `per_view` | 3D reference | n/a | 45.6 ± 5.6 | -- |
| VideoPose3D-243, trained per fold (80 epochs), `procrustes_average` | 3D reference | n/a | 46.6 ± 6.5 (13 j; folds 42.7 / 38.6 / 56.9 / 50.8 / 44.3) | -- |
| VideoPose3D-243, `per_view` | 3D reference | n/a | 52.8 ± 5.8 | -- |
| VideoPose3D H36M checkpoint, zero-shot (appendix only) | -- | -- | 86.3 ± 6.3 | -- |

Notes: MHFormer fold_04 and MDVPose folds 04/05 hit the 5.5 h job limit
(epoch 18/19, 19/30 and 29/30); their checkpoints are the best
validation-subject epochs (11, 18, 2), which had already been reached, so
they were not re-run. MetaPose `init` (the monocular VideoPose3D
initialisation) scores 92.6 / 93.7 mm. The private column can only hold
label-free methods: the triangulated pseudo-reference is built from the
same two views and is the evaluation reference, so supervising a method
with it would be circular (training never reads it, by project policy).

Reference rows for the same protocol come from `python -m fusion analyze`
(the model: 18.4 mm on the private data, 20 joints); external rows are
scored on the 12-14 major joints their skeleton covers, so the comparison
table must re-aggregate the model on those joints.

## Status (2026-09-22 noon)

* Framework, mappings, evaluator, five adapters (CanonPose, MetaPose,
  MHFormer, MDVPose, VideoPose3D-trained): done, tested
  (`tests/fusion/external/test_published.py`); `python -m fusion
  external-published report` tabulates every `summary_*.json`.
* Gymnastics: CanonPose and MetaPose (S1, S2 fwd, S2 ts) evaluated.
* FreeMan: CanonPose, MetaPose S1, MHFormer, MDVPose, VideoPose3D-trained
  evaluated; MetaPose S2 (fwd) fails inside
  cuSOLVER's `gesvd` in stage 1 on this dataset (twice; the launcher's
  opt-in CPU-SVD pin exists for it), the `ts` folds were trained by the
  concurrent session.
* SportsPose column waits for the user's `sam3d_sp` inference jobs.
