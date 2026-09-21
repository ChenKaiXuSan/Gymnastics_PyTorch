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
| 3 | **MHFormer** (Li et al., CVPR 2022) | supervised (3D labels) | trained with its recipe on the reference joints of the training subjects; **FreeMan and SportsPose only** (the private data has no independent 3D reference) | — (pending) |
| 4 | **Two Views Are Better than One** (MotionBERT-based, `mpjpe_weight: 1.0`) | supervised | same as 3, FreeMan / SportsPose rows only | — (pending; flagged to the user) |
| A | VideoPose3D (Pavllo et al., CVPR 2019) | supervised on Human3.6M | released `pretrained_h36m_detectron_coco.bin`, per view, both views Procrustes-averaged | zero-shot → **appendix only**: FreeMan-repetitive 86.3 ± 6.3 mm (13 joints) |
| B | MetaPose released `ckpt/h36m/cam2` | — | `--stage s2 --released` | zero-shot → appendix only |

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

## Status (2026-09-21)

* Framework, mappings, evaluator: done, tested (`tests/fusion/external/test_published.py`).
* CanonPose: FreeMan and gymnastics inputs prepared (gymnastics: 928
  trials, 147,297 frames); 3-epoch timing run on FreeMan fold_01 submitted
  (job 15512); full 5-fold campaigns follow once the epoch time is known.
* MetaPose: subject-01 FreeMan pipeline validated end to end with the
  released checkpoint; per-fold training path validated on CPU (smoke run);
  FreeMan full `prepare::s1` job 15511 running; per-fold training next.
* MHFormer, Two Views: not started (supervised; FreeMan / SportsPose rows only).
* SportsPose column waits for the user's `sam3d_sp` inference jobs.
