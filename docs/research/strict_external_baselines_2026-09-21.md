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

Updated on 2026-09-25 to the final architecture v1.2. The v1.2 rows for the
private data and FreeMan are the `*_v1_1_equal_reliability_5fold_seed0` sweeps,
whose training is identical to v1.2. v1.1 is kept as an archived row.

| Method | Supervision | Gymnastics (137) | FreeMan-rep (37) | Fit3D (8) |
|---|---|---:|---:|---:|
| **Ours v1.2 (final, self-supervised)** | none | **18.2 ± 1.1** | **41.0 ± 6.2** | 47.7 ± 4.7 |
| Ours, closed-form rule only (no learning) | none | 18.0 ± 1.0 | 41.0 ± 6.2 | 47.6 ± 4.8 |
| Ours v1.1 (archived; learned view weights) | none | 19.3 ± 1.0 | 41.4 ± 6.2 | 47.7 ± 4.6 |
| Ours v1.2, reference-supervised (upper bound) | 3D ref | n/a | 37.8 ± 6.2 | 25.1 ± 3.2 |
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

Fold-level tests floor at p = 0.0625 for n = 5. The folds are
subject-disjoint, so each subject is a held-out measurement of both methods
(`python -m fusion external-published compare`, bootstrap CI over subjects,
Wilcoxon signed-rank, Holm across the rows of one dataset). Full output is in
`local/runs/external_published/stats/v1_2_*.json`; the v1.1 versions are the
files without the prefix. Ours = v1.2; positive difference = ours better.
Subject means differ slightly from the fold means above.

Gymnastics, 137 participants:

| Comparison | Theirs | Ours | Diff [95 % CI] | Ours better | p_Holm |
|---|---:|---:|---|---:|---:|
| Closed-form rule | 18.1 | 18.3 | -0.17 [-0.20, -0.15] | 16/137 | 1.0e-20 |
| v1.1 (archived) | 19.5 | 18.3 | +1.19 [+0.95, +1.46] | 114/137 | 4.7e-19 |
| CanonPose | 28.2 | 18.3 | +9.94 [+9.15, +10.73] | 137/137 | 2.2e-23 |
| MetaPose S1 | 43.4 | 18.3 | +25.09 [+23.52, +26.65] | 136/137 | 2.2e-23 |
| MetaPose S2-ts | 75.6 | 18.3 | +57.33 [+55.18, +59.50] | 137/137 | 2.2e-23 |
| Face view | 31.9 | 18.3 | +13.65 [+13.01, +14.28] | 137/137 | 2.2e-23 |
| Side view | 37.7 | 18.3 | +19.46 [+18.70, +20.24] | 137/137 | 2.2e-23 |

FreeMan, 37 subjects:

| Comparison | Theirs | Ours | Diff [95 % CI] | Ours better | p_Holm |
|---|---:|---:|---|---:|---:|
| MDVPose | 39.8 | 40.3 | -0.52 [-1.56, +0.49] | 15/37 | 0.30 |
| MHFormer | 44.0 | 40.3 | +3.68 [+2.25, +5.26] | 32/37 | 1.1e-05 |
| VideoPose3D-trained | 46.6 | 40.3 | +6.26 [+4.65, +7.97] | 33/37 | 2.6e-08 |
| CanonPose | 52.8 | 40.3 | +12.50 [+10.95, +14.01] | 36/37 | 2.9e-10 |
| MetaPose S1 | 54.4 | 40.3 | +14.05 [+10.95, +16.97] | 36/37 | 2.4e-07 |
| MetaPose S2-ts | 68.0 | 40.3 | +27.61 [+24.19, +31.02] | 36/37 | 2.9e-10 |
| VideoPose3D zero-shot | 83.7 | 40.3 | +43.36 [+39.29, +47.97] | 37/37 | 1.6e-10 |
| Closed-form rule | 40.4 | 40.3 | +0.04 [+0.01, +0.06] | 31/37 | 1.5e-04 |
| v1.1 (archived) | 40.6 | 40.3 | +0.30 [+0.13, +0.48] | 26/37 | 3.8e-03 |
| Ours v1.2, reference-supervised | 38.0 | 40.3 | -2.30 [-3.20, -1.32] | 7/37 | 7.4e-05 |

Fit3D, 8 subjects. The Wilcoxon floor is 2/2^8 = 0.0078, so p_Holm cannot go
below 0.086 with eleven comparisons; the raw p is given:

| Comparison | Theirs | Ours | Diff [95 % CI] | Ours better | p |
|---|---:|---:|---|---:|---:|
| MDVPose | 22.1 | 47.5 | -25.37 [-31.16, -19.45] | 0/8 | 7.8e-03 |
| MHFormer | 27.0 | 47.5 | -20.40 [-24.92, -15.59] | 0/8 | 7.8e-03 |
| VideoPose3D-trained | 27.4 | 47.5 | -20.09 [-25.98, -14.08] | 0/8 | 7.8e-03 |
| Ours v1.2, reference-supervised | 25.5 | 47.5 | -21.99 [-26.45, -17.56] | 0/8 | 7.8e-03 |
| CanonPose | 99.6 | 47.5 | +52.12 [+37.36, +64.98] | 8/8 | 7.8e-03 |
| MetaPose S1 | 58.3 | 47.5 | +10.82 [+9.69, +11.85] | 8/8 | 7.8e-03 |
| MetaPose S2-ts | 66.8 | 47.5 | +19.35 [+16.69, +23.11] | 8/8 | 7.8e-03 |
| VideoPose3D zero-shot | 49.5 | 47.5 | +2.01 [-1.52, +5.31] | 5/8 | 0.20 |
| Closed-form rule | 47.4 | 47.5 | -0.04 [-0.10, +0.01] | 2/8 | 0.31 |
| v1.1 (archived) | 47.5 | 47.5 | +0.04 [-0.06, +0.14] | 6/8 | 0.38 |

Reference-supervised v1.2 against the supervised methods
(`stats/v1_2_<dataset>_refsup_vs_supervised.json`): FreeMan beats MDVPose
+1.78 [+0.96, +2.58] (32/37, p_Holm 1.3e-4), MHFormer +5.98 (35/37) and
VideoPose3D +8.57 (36/37). Fit3D loses to MDVPose -3.38 (1/8, p 0.016) and
is ahead of MHFormer +1.59 and VideoPose3D +1.91 (6/8, n.s.).

### Reading

1. **Against label-free methods we win everywhere and by a wide margin.**
   CanonPose and both MetaPose variants lose on 136-137 of 137 private
   participants, 36/37 FreeMan subjects and 8/8 Fit3D subjects
   (p_Holm < 1e-6 on the first two).
2. **Against reference-supervised methods the result depends on how far the
   dataset's reference skeleton is from SAM3D's.** On FreeMan the
   self-supervised v1.2 beats MHFormer and VideoPose3D and ties with MDVPose
   (CI crosses zero, 15/37). Trained with the reference, it beats MDVPose
   too (+1.8 mm, 32/37). On Fit3D every supervised method is 20-25 mm ahead,
   and so is our own architecture trained with the reference (25.1 vs 47.7).
   The two input views alone are 53 mm and the closed form is 47.6. Fusion can
   only remove the part of the error the two views disagree on, while a
   supervised lifter also learns the systematic SAM3D -> Fit3D skeleton
   offset that survives Procrustes alignment. The self-supervised setting
   cannot see that offset by construction.
3. **On clean data v1.2 matches the closed-form rule.** It is 0.17 mm
   behind on the private data, 0.04 mm ahead on FreeMan and within 0.04 mm
   on Fit3D. It beats the archived v1.1 by 1.2 mm on the private data
   (114/137) and 0.3 mm on FreeMan (26/37). The learned residual earns its
   place under corruption, where v1.2 beats the rule on almost every subject
   of all three datasets ("Where the learned part helps" and "Final model
   v1.2" below).

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


## Where the learned part helps (2026-09-25)

All numbers in this section: 12 comparison joints, 5 folds, per-frame
PA-MPJPE in mm, seed 0, unless stated. "Rule" is the closed-form
equal-weight depth-aware fusion (alpha 0.8, zero parameters); "learned
weights" is the v1.1 base pose `P_base` with the reliability head's weights;
"v1.1" is the full model; "equal weights + residual" is v1.1 trained with
`model.reliability.enabled=false` (runs `*_equal_reliability_*`, called
**v1.2** below).

### Test-time corruption sweep

`python -m fusion external-published corruption` re-scores finished
checkpoints with the training corruption switched on at test time. The level
scales every corruption probability (level 1 = the training setting); the
magnitudes stay fixed.

| Dataset | Level | Rule | Learned weights | v1.1 | Learned weights, no residual | v1.2 |
|---|---:|---:|---:|---:|---:|---:|
| Gymnastics | 0 | **18.0** | 20.1 | 19.3 | 18.2 | 18.2 |
| | 0.5 | **18.7** | 22.6 | 20.3 | 20.0 | **18.7** |
| | 1 | 21.4 | 28.5 | 23.3 | 24.5 | **20.5** |
| | 2 | 30.1 | 37.6 | 30.0 | 33.1 | **26.4** |
| FreeMan | 0 | **41.0** | 41.5 | 41.4 | -- | **41.0** |
| | 1 | 42.7 | 46.0 | 43.5 | -- | **42.0** |
| | 2 | 47.9 | 52.6 | 46.8 | -- | **45.3** |

Gymnastics ablations at level 1: v1.1 23.3, no cross-view 23.7, no FiLM
22.9, no long motion 23.3, no short motion 23.3, no phase 23.3, pose branch
only 22.8, no residual 24.5, v1.2 20.5. The encoder branches move the result
by less than 0.9 mm. What matters is whether the weights are learned and
whether the residual is present.

Paired over subjects (subject means, so they differ slightly from the fold
means above; difference = other minus v1.2, positive = v1.2 better; Wilcoxon
with Holm correction over the 14 tests):

| Dataset | Level | v1.2 vs rule | v1.2 better | p_Holm | v1.2 vs v1.1 | v1.2 better | p_Holm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gymnastics (137) | 0 | -0.17 | 16 | 5e-20 | +1.19 | 114 | 4e-18 |
| | 0.5 | +0.04 | 78 | 0.08 | +1.67 | 129 | 5e-22 |
| | 1 | +0.85 | 124 | 6e-20 | +2.77 | 136 | 4e-23 |
| | 2 | +3.64 | 136 | 4e-23 | +3.57 | 137 | 4e-23 |
| FreeMan (37) | 0 | +0.04 | 31 | 2e-4 | +0.30 | 26 | 4e-3 |
| | 1 | +0.63 | 36 | 1e-9 | +1.23 | 35 | 6e-10 |
| | 2 | +2.57 | 37 | 1e-10 | +1.32 | 36 | 2e-9 |

Reading: the learned reliability weights are worse than equal weights at
every level on both datasets. The residual is what makes the learned model
robust. v1.2 ties the rule on clean data (-0.17 mm on the private data,
+0.04 mm on FreeMan) and beats it under corruption on almost every subject.

### Strata on real (uncorrupted) data

`... analysis --what strata`, model against the rule, per frame. Difference =
rule minus model (positive = model better). Updated to v1.2 on 2026-09-25;
v1.1 in brackets.

| Stratum | Gymnastics | FreeMan | Fit3D |
|---|---:|---:|---:|
| Observed input < 80 % | no frames | +4.0 (111 frames, both ~0.9 m) [+1.9] | +0.2 (432 frames) [+0.2] |
| Observed input >= 95 % | -0.17 (all 65,024 frames) [-1.3] | +0.04 [-0.4] | -0.04 [-0.1] |
| Turn-around / between / mid-swing | -0.16 / -0.19 / -0.15 | +0.05 / +0.04 / +0.04 | -0.02 / -0.04 / -0.05 |
| Slow / medium / fast | -0.17 / -0.16 / -0.18 | +0.02 / +0.04 / +0.07 | -0.05 / -0.05 / -0.02 |
| Elderly / students | -0.16 / -0.19 | -- | -- |

On real data v1.2 stays within 0.2 mm of the rule in every stratum (v1.1
lost up to 1.5 mm). SAM3D never abstains, so real data almost never has the
degraded input the learned part was built for. The model is ahead in the
< 80 % observed stratum, which holds 111 FreeMan and 432 Fit3D frames.

### Measurement-level errors

`... analysis --what measurement`: trunk twist (shoulder line vs hip line)
per cycle. Each cell gives the range-of-motion error in degrees and the peak
angular-velocity error in degrees/s. Updated to v1.2.

| Variant | Gymnastics (1016 cycles) | FreeMan (4166) | Fit3D (1668) |
|---|---:|---:|---:|
| Face only | 3.36 / 69.0 | 7.08 / 72.8 | 4.84 / 38.6 |
| Side only | 6.37 / 15.8 | 7.07 / 73.0 | 4.79 / 39.0 |
| Rule | **2.70** / 22.4 | **6.79** / 73.3 | 4.84 / 34.5 |
| v1.2 | 2.74 / 28.2 | 6.80 / 73.7 | 4.94 / **34.0** |
| v1.1 (archived) | 2.76 / 24.3 | 6.79 / 73.5 | 4.97 / 34.2 |

Fusion lowers the ROM error compared with either view on the private data
and FreeMan. Peak angular velocity is underestimated after fusion (ratio
0.85-0.89 on the private data), because averaging two views smooths the
peak. The private column is scored against the triangulated pseudo-reference
and shares its bias.

### Bias of a two-view pseudo-reference (FreeMan)

`python -m fusion external-published pseudo-reference build` triangulates
the two selected FreeMan views with the released cameras (the same
construction as the private reference). The same predictions are then scored
against both references (`data.options.reference_source`).

| Method | Release reference | Two-view triangulated | Change |
|---|---:|---:|---:|
| Rule | 41.0 | **15.7** | -62 % |
| v1.2 | 41.0 | 15.8 | -61 % |
| v1.1 (archived) | 41.4 | 17.9 | -57 % |
| Face only | 48.6 | 31.9 | -34 % |
| MDVPose | **39.8** | 32.2 | -19 % |
| CanonPose | 53.4 | 38.7 | -28 % |

The pseudo-reference rewards two-view fusion 3x more than a supervised
lifter and flips the ranking: MDVPose leads against the release reference,
and the rule leads by 16.5 mm against the pseudo-reference. Private-data
margins between fusion and external methods are therefore inflated and are
reported only among label-free methods.

### Cost

`... cost` (GPU timing from `local/runs/external_published/cost/cost_table.json`; temporal context = frames the method reads per output):

| Method | Parameters | ms / frame | Temporal context | 3D labels | Calibration | Pretrained |
|---|---:|---:|---:|---|---|---|
| Ours, rule | 0 | 0.0006 | 128 | no | no | no |
| Ours, v1.2 (final) | 1.05 M | 0.065 | 128 | no | no | no |
| CanonPose | 10.6 M | 0.004 | 1 | no | no | no |
| MetaPose S1 | 0 | iterative (100 Adam steps) | 1 | no | no | monocular init |
| MHFormer-81 | 19.8 M | 5.0 | 81 | yes | camera-frame reference | no |
| MDVPose | 42.5 M | 0.074 | 243 | yes | camera-frame reference | MotionBERT H36M |
| VideoPose3D-243 | 17.0 M | 0.62 | 243 | yes | camera-frame reference | no |

## Depth discount alpha (2026-09-25)

`python -m fusion external-published alpha --dataset <d> --joints all`
(20 joints) sweeps the rule over alpha in {0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8,
0.85, 0.875, 0.9, 0.95, 0.98}. alpha = 1 is excluded because the system
becomes singular when the optical axes are not orthogonal. Outputs are in
`local/runs/external_published/alpha/<dataset>/`.

Pooled over all subjects:

| alpha | 0 | 0.3 | 0.5 | 0.6 | 0.7 | 0.75 | **0.8** | 0.85 | 0.875 | 0.9 | 0.95 | 0.98 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FreeMan | 43.03 | 41.91 | 41.20 | 40.92 | 40.73 | **40.70** | 40.72 | 40.81 | 40.90 | 41.01 | 41.32 | 41.59 |
| Fit3D | 52.24 | 51.41 | 50.71 | 50.31 | 49.85 | 49.60 | 49.35 | 49.11 | 49.00 | 48.91 | **48.86** | 48.89 |
| Gymnastics (diagnostic only) | 27.46 | 24.26 | 21.65 | 20.21 | 18.73 | 18.01 | 17.33 | 16.74 | 16.50 | 16.31 | **16.14** | 16.22 |

Fold-wise calibration (alpha selected on each fold's validation subjects,
scored on its test subjects; fold mean ± sd):

| Setting | FreeMan | Fit3D |
|---|---:|---:|
| Plain average (alpha 0) | 43.68 ± 6.65 | 52.24 ± 3.23 |
| Fixed alpha 0.8 | 41.41 ± 6.98 | 49.42 ± 3.53 |
| alpha 0.98 | 42.27 ± 6.92 | 48.99 ± 3.65 |
| One alpha from validation | 41.39 ± 6.98 (0.75 in every fold) | **48.97 ± 3.62** (0.9-0.95) |
| Per-joint alpha from validation | 41.44 ± 6.98 | 49.35 ± 3.54 |

Conclusions:

1. **alpha = 0.8 stays fixed.** On FreeMan, which has an independent
   reference, the curve is U-shaped with its minimum at 0.75, and 0.8 is
   0.02 mm off. On Fit3D the best value (0.95) gains 0.45 mm (about 1 %),
   which changes no ranking. Between 0.75 and 0.95 the error varies by
   less than 0.5 mm on both datasets. Every existing result, all run at
   0.8, stands.
2. **Per-joint alpha is rejected.** It is worse than a single alpha on both
   datasets (selection overfits the validation subjects) and is not
   implemented in the model.
3. **Depth awareness itself is the gain.** Going from the plain average to
   0.8 saves 2.3 mm on FreeMan and 2.8 mm on Fit3D, several times more than
   any further tuning of alpha.
4. **The private reference cannot select alpha.** Its curve falls steeply
   up to 0.95: the 0.8 -> 0.95 step is worth 1.2 mm there, against
   0.02-0.5 mm on the datasets with an independent reference. This matches
   the pseudo-reference bias above (the triangulated reference is built
   from the same image-plane coordinates that alpha -> 1 trusts).

Cross-dataset transfer of the dataset-wide per-joint tables (fold mean,
20 joints). Joints the source reference does not cover (toes and heels,
plus the neck on FreeMan) take the source's global alpha.

| Scored on | Fixed 0.8 | FreeMan table | Fit3D table |
|---|---:|---:|---:|
| FreeMan | **41.41** | (own) | 42.08 |
| Fit3D | **49.42** | 49.67 | (own) |
| Gymnastics (diagnostic only) | 17.33 | 17.87 | 16.55 |

5. **Per-joint tables do not transfer.** On both datasets with an
   independent reference, the other dataset's table is worse than the fixed
   0.8 (+0.67 and +0.25 mm). On the private data the Fit3D table (higher
   alphas) looks 0.8 mm better and the FreeMan table 0.5 mm worse. That
   ordering follows each table's mean alpha, which is the pseudo-reference
   artefact again, not a transfer gain.

## Architecture decision: v1.2 (final, 2026-09-25)

v1.2 is **a model version, not a loss version**: v1.1 with
`model.reliability.enabled=false`. No code changes. The reliability head is
bypassed, both views get weight 1/2 (the only valid view takes weight 1),
`P_base` becomes exactly the closed-form rule, and the residual head reads
`(C_A + C_B) / 2` instead of `w_A C_A + w_B C_B`. The loss config stays
`loss/v3`. `L_rel` has no gradient path any more, so the effective objective
is `L_rec + 0.01 L_res`. The learned part is now only the bounded residual:
a fixed geometric prior (alpha) gives the accuracy, and the residual gives
robustness under damaged input.

Configs since commit dd76abc: `model/v1_2.yaml` + `loss/v4.yaml` (v3 with the
reliability weight at 0) are the Hydra defaults; v1.1 is `model=v1_1
loss=v3` (since the archive: `model=archive/v1_1 loss=archive/v3`). v1.2 runs: private data and FreeMan = the `*_v1_1_equal_reliability_5fold_seed0`
sweeps (identical training: with the head off `L_rel` has no gradient);
Fit3D = `fit3d_v1_2_5fold_seed0` (2026-09-25). Module ablations of v1.2 on the
private data: `gymnastics_v1_2_<preset>_5fold_seed0` (results below).

## Final model v1.2: Fit3D and module ablations (2026-09-25)

12 comparison joints, 5 folds, seed 0.

Corruption sweep of the final model on every dataset (fold means, mm; the
private and FreeMan v1.2 rows are the `*_v1_1_equal_reliability_*` sweeps):

| Dataset | Variant | Level 0 | 0.5 | 1 | 2 |
|---|---|---:|---:|---:|---:|
| Gymnastics | Rule | **18.0** | **18.7** | 21.4 | 30.1 |
| | v1.1 (archived) | 19.3 | 20.3 | 23.3 | 30.0 |
| | **v1.2** | 18.2 | **18.7** | **20.5** | **26.4** |
| FreeMan | Rule | **41.0** | -- | 42.7 | 47.9 |
| | v1.1 (archived) | 41.4 | -- | 43.5 | 46.8 |
| | **v1.2** | **41.0** | -- | **42.0** | **45.3** |
| Fit3D | Rule | **47.6** | 48.0 | 49.0 | 53.2 |
| | v1.1 (archived) | 47.7 | 48.1 | 49.3 | 52.7 |
| | **v1.2** | 47.7 | **47.9** | **48.7** | **51.6** |

Fit3D (8 subjects, so the smallest attainable Wilcoxon p is 0.0078): v1.2 ties
the rule on clean data (-0.04 mm, 2/8 subjects better, p 0.31) and beats both
the rule and v1.1 on 8/8 subjects at levels 1 and 2 (rule +0.38 / +1.58 mm,
v1.1 +0.66 / +1.17 mm, p 0.0078). The same pattern holds on all three
datasets.

Module ablations of v1.2 on the private data (`gymnastics_v1_2_<preset>_5fold_seed0`),
paired over the 137 subjects (subject means, so v1.2 reads 18.29 rather than
the 18.2 fold mean). Difference = ablation minus v1.2, so positive means the
component helps. Holm correction over the 18 tests:

| Removed | Clean | Level 1 | Level 2 | v1.2 better (level 2) | p_Holm (level 2) |
|---|---:|---:|---:|---:|---:|
| Motion branches + FiLM (pose only) | +0.20 | +0.34 | **+0.64** | 129/137 | 1e-21 |
| FiLM | +0.18 | +0.30 | **+0.58** | 126/137 | 2e-20 |
| Cross-view attention | +0.11 | +0.25 | **+0.48** | 122/137 | 3e-20 |
| Long-motion branch | +0.04 | +0.09 | +0.27 | 108/137 | 8e-15 |
| Short-motion branch | +0.07 | +0.06 | +0.12 | 102/137 | 1e-6 |
| Phase encoding | +0.03 | +0.02 | +0.02 | 75/137 | 0.24 (n.s.) |

Absolute errors for v1.2: 18.29 clean, 20.68 at level 1, 26.57 at level 2.
The residual itself is not in the table: v1.2 without it is the rule
(18.0 / 21.4 / 30.1).

Reading: with the reliability head gone, every encoder component except the
phase encoding now helps. The effect is significant, small on clean data
(0.03-0.20 mm) and roughly three times larger at level 2 (up to 0.64 mm),
which is consistent with the residual acting as a corruption-recovery term.
This reverses the v1.1 ablation, where removing FiLM or the motion branches
*improved* the clean error: in v1.1 the encoder also fed the harmful
reliability weights.

## Final model v1.2: module ablations on all three datasets and the reference-supervised row (2026-09-25)

Each ablation (`<data>_v1_2_<preset>_5fold_seed0`) is scored at corruption levels
0/1/2 and paired over subjects against v1.2 at the same level. Difference =
ablation minus v1.2 (positive = the component helps), with the number of
subjects where v1.2 is better. Holm correction is applied within each dataset
(18 tests).

| Removed | Gymnastics (137) clean / L1 / L2 | FreeMan (37) clean / L1 / L2 | Fit3D (8) clean / L1 / L2 |
|---|---|---|---|
| Motion branches + FiLM (pose only) | +0.20 / +0.34 / **+0.64** (129) | +0.03 / +0.18 / **+0.59** (36) | -0.01 / +0.10 / +0.28 (8/8) |
| FiLM | +0.18 / +0.30 / **+0.58** (126) | +0.03 / +0.17 / **+0.54** (36) | -0.03 / +0.07 / +0.23 (8/8) |
| Cross-view attention | +0.11 / +0.25 / **+0.48** (122) | -0.01 / +0.20 / **+0.72** (36) | -0.01 / +0.07 / +0.27 (8/8) |
| Long-motion branch | +0.04 / +0.09 / +0.27 (108) | +0.00 / +0.07 / +0.21 (30) | +0.02 / +0.03 / +0.04 (6/8) |
| Short-motion branch | +0.07 / +0.06 / +0.12 (102) | +0.02 / +0.08 / +0.20 (34) | +0.01 / +0.04 / +0.15 (7/8) |
| Phase encoding | +0.03 / +0.02 / +0.02 (75), n.s. | -0.01 / -0.02 / **-0.06** (9, p_Holm 0.002) | +0.01 / +0.02 / +0.00 (3/8) |

v1.2 absolute errors (subject means): gymnastics 18.29 / 20.68 / 26.57,
FreeMan 40.35 / 41.34 / 44.64, Fit3D 47.45 / 48.43 / 51.36.

Significance: on the private data every row except the phase encoding is
significant at all three levels (p_Holm <= 5e-4). On FreeMan every row
except the phase encoding is significant at levels 1 and 2 (p_Holm <= 2e-4),
and none is significant on clean data. Fit3D has 8 subjects, so the smallest
Wilcoxon p is 0.0078 and no row survives the Holm correction over 18 tests.
The direction still matches the other two datasets: pose only, FiLM and
cross-view are better than their ablation on 8/8 subjects at level 2.

Reading: the encoder components matter for recovery from damaged input,
not for clean accuracy. The ranking is the same on the two datasets with
enough subjects: the FiLM-conditioned motion branches and cross-view
attention carry most of the effect (0.5-0.7 mm at level 2), and the
long- and short-motion branches contribute 0.1-0.3 mm. The phase encoding
never helps. It is not significant on the private data or Fit3D and is
slightly harmful on FreeMan at level 2 (removing it gains 0.06 mm).
**Decision (2026-09-25): the phase encoding stays in v1.2.** The effect is
at most 0.06 mm, and it is reported as a component without measurable
contribution.

Reference-supervised final model (`experiment=reference_supervised`, v1.2 +
loss v4 with the dataset reference as the recovery target), 12 joints:

| | FreeMan | Fit3D |
|---|---:|---:|
| v1.2, reference-supervised | **37.8** | 25.1 |
| v1.1, reference-supervised (archived) | 37.3 | 24.9 |
| MDVPose | 39.8 | **21.7** |
| MHFormer | 43.6 | 27.2 |
| VideoPose3D (trained) | 46.3 | 27.1 |

Paired over subjects, v1.2-refsup beats MDVPose on FreeMan (+1.78 mm, CI
0.96-2.58, 32/37, p 1e-4), MHFormer (+6.0, 35/37) and VideoPose3D (+8.6,
36/37). On Fit3D it beats MHFormer and VideoPose3D on 6/8 subjects (n.s.)
and loses to MDVPose (-3.4 mm, 1/8, p 0.016). Compared with v1.1-refsup it
is 0.43 mm worse on FreeMan (5/37, p 3e-6) and 0.21 mm worse on Fit3D (n.s.).
With a real target, learned weights can carry a little signal.

## Real SAM3D failures: does v1.2 beat the rule where the inputs are wrong? (2026-09-26)

The corruption sweep only shows recovery from *synthetic* damage.
`python -m fusion external-published analysis --what failures`
(`failure_strata.py`) labels every test frame by real input failures and
compares v1.2 with the rule, per frame and per subject (subjects with >= 10
frames in a level; Wilcoxon). 12 joints; difference = rule minus v1.2, so
positive = v1.2 better.

| Stratum (share of frames) | Gymnastics (pseudo-GT) | FreeMan | Fit3D |
|---|---|---|---|
| View disagreement, 0-50 % (label-free) | -0.09 (26/137) | +0.02 (29/37) | -0.03 (3/8) |
| View disagreement, 90-99 % | -0.40 (6/82) | +0.11 (29/31) | -0.11 (2/8) |
| View disagreement, top 1 % | -1.00 (3/24) | +0.20 (19/22) | -0.07 (3/8) |
| Worse view error, top 1 % | -0.83 (0/19) | -0.21 (18/23, n.s.) | +0.23 (7/8, p 0.04) |
| One view with a limb > 60 deg off (face / side) | -0.44 / -1.36 (0.4 % / 0.1 %) | +0.09 / +0.14 (3.3 % / 3.7 %; 27/30, 30/31) | +0.59 / +0.61 (0.2 % each; 5/6, 6/7) |
| Both views with a limb > 60 deg off | -1.16 (< 0.1 %) | -0.18 (1.2 %) | +0.05 (0.3 %) |
| Worse view > 2x the better | -0.26 (14.8 %) | +0.12 (4.9 %; 32/34) | +1.10 (0.3 %; 6/6, p 0.03) |
| Left-right swap in a view | < 0.1 % of frames | < 0.1 % | none |

Reading:

1. **SAM3D rarely fails grossly in these data.** Left-right swaps are
   practically absent. A limb more than 60 degrees off occurs in 0.5 % of the
   private frames, 8 % of FreeMan and 0.7 % of Fit3D.
2. **The rule already handles a single failing view.** When one FreeMan view
   has a gross limb error (face 81 mm, side 52 mm), the rule gives 50.4 mm,
   better than the good view alone. This leaves little for a learned
   correction to recover.
3. **On the datasets with an independent reference, v1.2 is ahead of the rule
   in the failure strata, but by very little.** FreeMan: +0.1-0.2 mm in the
   hard strata, consistent over subjects (27/30 to 32/34). Fit3D: +0.6-1.1 mm
   in the 0.2-0.3 % of frames where one view fails (p 0.03 with 6-7
   subjects). The gain grows with the difficulty of the frame, which matches
   the corruption sweep, but it is 0.3-1 % of the error.
4. **Frames where both views fail are not rescued.** Top-1 % frames of FreeMan
   (~200-350 mm) are equal for v1.2 and the rule.
5. **The private column goes the other way** (v1.2 loses up to 1 mm on the
   hardest frames). Its reference is triangulated from the same 2D keypoints
   and favours the rule by construction (see the pseudo-reference section),
   so it is not evidence either way.

Conclusion: on real inputs the learned residual is a small, consistent
safeguard that gains most where one view fails. The accuracy of the method
comes from the closed-form depth-aware rule. The paper should present the
rule as the main contribution and the network as a robustness component,
with the synthetic corruption sweep as its main evidence and these strata as
the real-data check.

## Status (2026-09-25)

* Robustness, strata, measurement, pseudo-reference, cost and alpha
  results above. Code: `corruption_sweep.py`, `analysis_rows.py`,
  `pseudo_reference.py`, `cost_table.py`, `alpha_calibration.py`, all in
  `src/fusion/external/published/`.
* 2026-09-25 evening: the comparison table, paired statistics, strata,
  measurement, pseudo-reference and cost rows are all on v1.2.
* Every v1.2 run is done: main results, corruption sweeps and module
  ablations on three datasets, plus the reference-supervised rows.
* Optional, not run: external methods under test-time corruption.

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
