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
FreeMan / Fit3D read their benchmark caches). Layout conversions in
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
| 3 | **MHFormer** (Li et al., CVPR 2022) | supervised (3D MPJPE) | released 81-frame configuration (`main.py`: Adam amsgrad 1e-3, x0.95 per epoch and x0.5 every 5 epochs, 19 epochs, batch 256, flip augmentation and flip TTA, centre frame, all window frames supervised); trained per fold on the training subjects' two views as monocular samples with the reference joints rotated into each camera (`mhformer.py`); **FreeMan and Fit3D only** (the private data has no independent 3D reference) | SAM3D 2D instead of CPN detections (missing joints interpolated in time, frames without reference masked); selection on the fold's validation subjects (release: test set); 81 instead of 351 frames (4x cheaper, also a released configuration) |
| 4 | **MDVPose** (Ma et al.; "multiple dynamic views via single-view pretraining with Procrustes alignment", the MotionBERT multi-view fine-tuning the user asked for as "two views") | supervised (MPJPE + 0.5 scale-normalised MPJPE + 20 velocity) + 0.002 Procrustes multi-view consistency | DSTformer from the MotionBERT H36M checkpoint (`walterzhu/MotionBERT` on Hugging Face, `local/checkpoints/motionbert`), released multi-view config (AdamW 3e-4, wd 0.01, x0.97 per epoch, 243-frame clips stride 81, flip aug/TTA), one batch = three clip pairs = the released batch of six clips, the consistency term inside each pair (`mdvpose.py`); FreeMan / Fit3D only | 30 epochs instead of 60 (9.7 GPU-h per fold otherwise); flip drawn per clip pair (the released loader flips cameras independently, contradicting its own multi-view term); selection on validation subjects; frame validity masks replace the release's padding marker |
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

## Comparison table (12 common joints, 5 folds, per-frame PA-MPJPE, mm)

Every published method predicts a different skeleton (12-14 of the 20 major
joints), so the table is computed on their intersection -- shoulders,
elbows, wrists, hips, knees, ankles (`evaluate.COMPARISON_JOINTS`). The
Procrustes alignment and the error use exactly those joints for every row,
ours included: the model's fold checkpoints run through the same evaluator
(`python -m fusion external-published model --run <sweep> --joints
comparison12`), which on all 20 joints reproduces the sweep's own numbers.

| Method | Supervision | Gymnastics (137) | FreeMan-rep (37) | Fit3D (8) |
|---|---|---:|---:|---:|
| **Ours v1.1 (self-supervised)** | none | **19.3 ± 1.0** | **41.4 ± 6.2** | 47.7 ± 4.6 |
| Ours, closed-form rule only (no learning) | none | 18.0 ± 1.0 | 41.0 ± 6.2 | 47.6 ± 4.8 |
| Ours, base pose (learned weights) | none | 20.1 ± 1.0 | 41.5 ± 6.2 | 48.2 ± 4.6 |
| Ours v1.1, reference-supervised (upper bound) | 3D ref | n/a | 37.3 ± 6.3 | 24.9 ± 3.2 |
| MDVPose | 3D ref | n/a | 39.8 ± 5.8 | **21.7 ± 2.2** |
| MHFormer-81 | 3D ref | n/a | 43.6 ± 5.5 | 27.2 ± 2.4 |
| VideoPose3D-243 trained | 3D ref | n/a | 46.3 ± 5.8 | 27.1 ± 2.5 |
| CanonPose | none | 28.0 ± 0.9 | 53.4 ± 4.3 | 103.9 ± 18.3 |
| MetaPose stage 1 | none | 43.3 ± 0.9 | 55.9 ± 5.0 | 58.6 ± 4.0 |
| MetaPose stage 2 (`ts`) | none | 75.5 ± 10.3 | 68.4 ± 1.5 | 66.6 ± 5.3 |
| VideoPose3D H36M checkpoint, zero-shot (appendix) | -- | n/a | 83.8 ± 6.4 | 49.4 ± 2.4 |
| Single view (face) | -- | 31.7 ± 1.0 | 48.6 ± 4.5 | 53.2 ± 4.0 |

Supervised methods have no private-data row: its only 3D is the triangulated
pseudo-reference, which is derived from the same two views and is the
evaluation reference, so training on it would be circular.

### Paired statistics over subjects

Fold-level tests floor at p = 0.0625 for n = 5; the folds are
subject-disjoint, so each subject is a held-out measurement of both methods
(`python -m fusion external-published compare`, bootstrap CI over subjects,
Wilcoxon signed-rank, Holm across the rows of one dataset; full output in
`local/runs/external_published/stats/`). Positive difference = ours better.

Gymnastics, 137 participants:

| Comparison | Theirs | Ours | Diff [95 % CI] | Ours better | p_Holm |
|---|---:|---:|---|---:|---:|
| Closed-form rule | 18.1 | 19.5 | -1.37 [-1.64, -1.12] | 17/137 | 2.2e-21 |
| No-residual ablation | 18.3 | 19.5 | -1.20 [-1.44, -0.99] | 16/137 | 1.6e-21 |
| Base pose (learned weights) | 20.3 | 19.5 | +0.80 [+0.67, +0.93] | 128/137 | 2.5e-22 |
| CanonPose | 28.2 | 19.5 | +8.74 [+8.01, +9.48] | 136/137 | 2.5e-23 |
| MetaPose S1 | 43.4 | 19.5 | +23.90 [+22.35, +25.44] | 136/137 | 2.5e-23 |
| MetaPose S2-ts | 75.6 | 19.5 | +56.13 [+53.99, +58.30] | 137/137 | 2.5e-23 |
| Face view | 31.9 | 19.5 | +12.46 [+11.86, +13.05] | 137/137 | 2.5e-23 |

FreeMan, 37 subjects:

| Comparison | Theirs | Ours | Diff [95 % CI] | Ours better | p_Holm |
|---|---:|---:|---|---:|---:|
| MDVPose | 39.8 | 40.6 | -0.82 [-1.91, +0.25] | 16/37 | 0.12 |
| MHFormer | 44.0 | 40.6 | +3.38 [+1.92, +4.99] | 29/37 | 1.6e-04 |
| VideoPose3D-trained | 46.6 | 40.6 | +5.96 [+4.30, +7.72] | 31/37 | 1.3e-07 |
| CanonPose | 52.8 | 40.6 | +12.20 [+10.65, +13.71] | 36/37 | 2.6e-10 |
| MetaPose S1 | 54.4 | 40.6 | +13.76 [+10.72, +16.64] | 36/37 | 2.4e-07 |
| MetaPose S2-ts | 68.0 | 40.6 | +27.31 [+23.90, +30.70] | 36/37 | 2.6e-10 |
| VideoPose3D zero-shot | 83.7 | 40.6 | +43.06 [+39.05, +47.56] | 37/37 | 1.6e-10 |
| Closed-form rule | 40.4 | 40.6 | -0.26 [-0.43, -0.11] | 11/37 | 9.2e-03 |
| Ours, reference-supervised | 37.6 | 40.6 | -3.04 [-3.99, -1.99] | 7/37 | 1.9e-05 |

Fit3D, 8 subjects (the Wilcoxon floor is 2/2^8 = 0.0078, so p_Holm cannot go
below 0.07 with nine comparisons):

| Comparison | Theirs | Ours | Diff [95 % CI] | Ours better | p |
|---|---:|---:|---|---:|---:|
| MDVPose | 22.1 | 47.5 | -25.41 [-31.17, -19.58] | 0/8 | 7.8e-03 |
| MHFormer | 27.0 | 47.5 | -20.45 [-24.94, -15.67] | 0/8 | 7.8e-03 |
| VideoPose3D-trained | 27.4 | 47.5 | -20.13 [-25.97, -14.16] | 0/8 | 7.8e-03 |
| Ours, reference-supervised | 25.2 | 47.5 | -22.25 [-26.68, -17.96] | 0/8 | 7.8e-03 |
| CanonPose | 99.6 | 47.5 | +52.08 [+37.25, +65.01] | 8/8 | 7.8e-03 |
| MetaPose S1 | 58.3 | 47.5 | +10.78 [+9.60, +11.83] | 8/8 | 7.8e-03 |
| MetaPose S2-ts | 66.8 | 47.5 | +19.31 [+16.67, +23.06] | 8/8 | 7.8e-03 |
| VideoPose3D zero-shot | 49.5 | 47.5 | +1.97 [-1.53, +5.23] | 5/8 | 0.25 |
| Closed-form rule | 47.4 | 47.5 | -0.08 [-0.18, +0.02] | 2/8 | 0.20 |

### Reading

1. **Against label-free methods we win everywhere and by a wide margin**:
   CanonPose and both MetaPose variants lose on 136-137 of 137 private
   participants and on 36/37 FreeMan subjects, p_Holm < 1e-9 throughout.
2. **Against reference-supervised methods the picture depends on how far the
   dataset's reference skeleton is from SAM3D's.** On FreeMan we beat
   MHFormer and VideoPose3D and tie with MDVPose (CI crosses zero, 16/37).
   On Fit3D every supervised method is 20-25 mm ahead -- but so is *our own
   architecture trained with the reference* (24.9 vs 47.7). The two input
   views alone are 53 mm, the closed form 47.6: fusion can only remove the
   part of the error the two views disagree on, while a supervised lifter
   also learns the systematic SAM3D -> Fit3D skeleton offset that survives
   Procrustes alignment. The self-supervised setting cannot see that offset
   by construction. Within the label-free family the ranking is unchanged:
   we beat MetaPose S1 by 10.8 mm and CanonPose by 52 mm on all 8 subjects.
3. **The learned part does not pay for itself on clean data.** The
   closed-form rule is 1.4 mm better than the full model on the private
   data (17/137 subjects better, p_Holm 2e-21) and 0.3 mm better on FreeMan;
   the no-residual ablation matches the rule. The learned reliability
   weights (base pose) *are* worse than equal weights, and the residual
   recovers part of that but not all. The learned components currently
   justify themselves only under corruption (20-joint sweep: model 23.0 vs
   its base 29.5 mm), and even there the equal-weight rule scores 21.3.

## Per-method results on their own joint sets (12-14 joints)

| Method | Supervision | Gymnastics (private) | FreeMan-repetitive | Fit3D |
|---|---|---:|---:|---:|
| CanonPose, trained per fold, `canonical_average` | none | **30.0 ± 0.9** (13 j) | 53.4 ± 4.3 (12 j) | 102.8 ± 18.1 (13 j; folds 128.2 / 111.9 / 73.8 / 94.5 / 105.3) |
| CanonPose, `per_view` | none | 37.6 ± 1.4 | 62.9 ± 3.7 | 102.7 ± 21.3 (12 j) |
| MetaPose stage 1 (optimisation only, a published ablation) | none | 44.3 ± 0.9 (14 j) | 55.7 ± 4.8 (13 j) | running |
| MetaPose stage 2 trained, README default `fwd` objective | none | 127.8 ± 51.4 | not run (stage 1 crashes in cuSOLVER `gesvd`; the worst variant on the private data) | -- |
| MetaPose stage 2 trained, `ts` (student of stage 1) | none | 76.9 ± 9.1 | 68.0 ± 1.6 (13 j; folds 65.5 / 66.7 / 69.2 / 69.2 / 69.6) | running |
| MHFormer-81, trained per fold, `procrustes_average` | 3D reference | n/a (no independent reference) | 44.1 ± 6.2 (13 j; folds 40.5 / 35.9 / 54.0 / 47.5 / 42.8) | 29.6 ± 2.4 (14 j; folds 30.8 / 28.7 / 33.6 / 26.7 / 28.2) |
| MHFormer-81, `per_view` | 3D reference | n/a | 51.0 ± 5.5 | 29.2 ± 2.4 (12 j) |
| MDVPose, trained per fold (30 epochs), `procrustes_average` | 3D reference | n/a | **40.3 ± 6.5** (13 j; folds 35.7 / 33.1 / 51.4 / 43.5 / 38.0) | **24.4 ± 2.0** (14 j; folds 22.4 / 23.4 / 27.3 / 22.5 / 26.4) |
| MDVPose, `per_view` | 3D reference | n/a | 45.6 ± 5.6 | 23.6 ± 2.5 (12 j) |
| VideoPose3D-243, trained per fold (80 epochs), `procrustes_average` | 3D reference | n/a | 46.6 ± 6.5 (13 j; folds 42.7 / 38.6 / 56.9 / 50.8 / 44.3) | 29.2 ± 2.5 (14 j; folds 27.3 / 28.8 / 33.7 / 26.5 / 29.6) |
| VideoPose3D-243, `per_view` | 3D reference | n/a | 52.8 ± 5.8 | 28.8 ± 2.6 (12 j) |
| VideoPose3D H36M checkpoint, zero-shot (appendix only) | -- | -- | 86.3 ± 6.3 | 47.8 ± 2.4 (14 j) |
| *Ours v1.1, label-free (last epoch, sweep test)* | none | 18.4 (20 j) | 41.8 (13 j) | 49.5 ± 3.5 (14 j; folds 53.6 / 45.9 / 53.1 / 49.4 / 45.4) |
| *Ours v1.1, reference-supervised (last epoch, `final.ckpt`)* | 3D reference | n/a | 38.0 ± 6.9 (13 j; folds 33.4 / 30.2 / 49.6 / 41.8 / 34.9) | 24.8 ± 3.1 (14 j; folds 22.8 / 24.1 / 30.3 / 21.2 / 25.7) |

Notes: MHFormer fold_04 and MDVPose folds 04/05 hit the 5.5 h job limit
(epoch 18/19, 19/30 and 29/30); their checkpoints are the best
validation-subject epochs (11, 18, 2), which had already been reached, so
they were not re-run. MetaPose `init` (the monocular VideoPose3D
initialisation) scores 92.6 / 93.7 mm. The private column can only hold
label-free methods: the triangulated pseudo-reference is built from the
same two views and is the evaluation reference, so supervising a method
with it would be circular (training never reads it, by project policy).

Fit3D (replaced SportsPose on 2026-09-23, commit 16423d5): 8 subjects with
reference, 47 fitness exercises, 4 synchronised cameras at 50 fps, 296
sequences with annotated repetitions (1526, used as the cycles); folds hold
1/1/2/2/2 test subjects. Its rig only offers ~46, 132 and 178 degree camera
separations, so the selected pair is ~132 degrees apart (a weaker depth
baseline than FreeMan's 90 degrees), and the `joints3d_25` reference maps
onto 14 of the 20 major joints (no heels, toes, eyes, ears).

Reference rows for the same protocol come from `python -m fusion analyze`
(the model: 18.4 mm on the private data, 20 joints); external rows are
scored on the 12-14 major joints their skeleton covers, so the comparison
table must re-aggregate the model on those joints.


## Status (2026-09-24)

* Fit3D column complete for CanonPose, MHFormer, MDVPose, VideoPose3D
  (trained and zero-shot) and our model (label-free and reference-supervised);
  MetaPose on Fit3D is running (`mpts_f3_*` jobs of the concurrent session).
* Model rows are scored from `final.ckpt` (`--checkpoint auto`); for runs
  that predate it they fall back to `last.ckpt`, which is exact for the
  label-free runs (validation improved to the last epoch).

## Status (2026-09-24)

* Framework, mappings, evaluator, five adapters (CanonPose, MetaPose,
  MHFormer, MDVPose, VideoPose3D-trained): done, tested
  (`tests/fusion/external/test_published.py`, 8 tests).
  `python -m fusion external-published report` tabulates every
  `summary_*.json`; `... model --variant rule|base|face|side` scores our own
  closed form and inputs through the same evaluator; `... compare` runs the
  paired per-subject statistics.
* Gymnastics: CanonPose and MetaPose (S1, S2 `fwd`, S2 `ts`) evaluated.
* FreeMan: all five methods evaluated; MetaPose S2 `fwd` not run (crashes
  inside cuSOLVER `gesvd` in stage 1, twice).
* Fit3D: complete -- CanonPose, MetaPose (S1 and S2-`ts`), MHFormer,
  MDVPose, VideoPose3D (trained and zero-shot).
* v1.1 ablations (gymnastics, 8 presets) and seeds 1-2 for both main sweeps
  are in `local/runs/cycle_aware/`; seed spread is 0.07 mm, an order of
  magnitude below the effects discussed above.
