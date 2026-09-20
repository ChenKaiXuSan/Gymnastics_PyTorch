# FreeMan subject-disjoint training of the rotation-aware model (2026-09-18)

## Why

The paper's FreeMan rows (Table 4 in `paper/sports_engineering/manuscript.tex`)
apply private-data checkpoints zero-shot. A reviewer can ask whether the
learned model would beat deterministic body-frame averaging if it were trained
on the target domain. Because the model is self-supervised, FreeMan training
needs no FreeMan labels; the only thing that must be protected is subject
disjointness between training and evaluation.

## Design

- Data: the two-view SAM3D-Body predictions already cached by the zero-shot
  benchmark (`local/runs/freeman_benchmark_cluster/sam3d`), same camera pair
  per session, native zero temporal offset, 25 fps. One rotation-aware
  "person" is one FreeMan subject; one trial is one session. The FreeMan 3D
  reference is never opened while building caches, training, validating or
  selecting checkpoints.
- Cohort: the ten paper subjects (1, 7, 9, 12, 15, 19, 22, 29, 35, 36; 552
  sessions) are the evaluation cohort. Five folds partition them into test
  pairs balanced by frame count: (15, 22), (1, 12), (35, 36), (9, 29),
  (7, 19). Each fold trains on the other eight paper subjects.
- Validation: the seven cached non-paper subjects (2, 4, 5, 6, 8, 10, 11;
  61 sessions) are the checkpoint-selection set for every fold, so all folds
  select on identical data and no evaluation subject ever enters training or
  validation.
- Model and objective: identical to the private A6 protocol (same network,
  nine self-supervised losses, Adam 1e-3, batch 32, 128-frame windows with
  train stride 32, self-supervised checkpoint score).
- Epoch budget: one fold has about 13.5k training windows versus 1,555 on the
  private data, so 12 epochs give about 5.1k window-optimizer steps, matching
  the private 100-epoch run (about 4.9k steps). Training is from random
  initialisation, as in the private protocol.
- Evaluation: every session of the ten subjects is fused with the checkpoint
  of the fold that holds its subject out, then scored with the zero-shot
  benchmark evaluator (17 shared joints, one Sim3 per session and per-frame
  Procrustes, subject-balanced). A guard refuses any checkpoint whose recorded
  split lists the subject under train or val.

## Commands

```bash
export PYTHONPATH=src
PY=/home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body/bin/python
$PY -m fusion benchmark-freeman-train prepare-cache   # CPU, ~1 min
$PY -m fusion benchmark-freeman-train write-folds
$PY -m fusion benchmark-freeman-train plan --seed 0   # prints the five train commands
for f in fold_01 fold_02 fold_03 fold_04 fold_05; do
  qsub -v FOLD=$f,SEED=0 scripts/freeman_train_qsub_fold.sh
done
$PY -m fusion benchmark-freeman-train evaluate --seed 0
$PY -m fusion benchmark-freeman-train compare  --seed 0
```

Config: `src/configs/archive/rotation_aware_freeman.yaml`. Folds:
`src/configs/shared/folds/freeman/fold_0{1..5}.json`. Runs:
`local/runs/fuse_rotation_aware_freeman/runs/freeman_fold_0X_a6_e12_s0`.
Evaluation and paired comparison: `local/runs/freeman_trained_fusion/`.

## Status

- 2026-09-18: code, configs, folds and caches in place; unit tests in
  `tests/freeman_benchmark/test_training.py` (19 passed). A one-epoch CPU
  smoke run (batch 2, subjects 05/06/15) completed and its checkpoint was
  pushed through the out-of-fold evaluator on two subject-15 sessions, giving
  values in the same range as the zero-shot rows. Five fold jobs submitted to
  the gpu queue as NQS requests 5994–5998 (logs:
  `local/runs/fuse_rotation_aware_freeman/joblogs/fold_0X_s0.log`).
- Login-node caveat: the per-user memory quota is 16 GB, so training must
  run on the gpu queue; a batch-32 CPU attempt was OOM-killed.
- After the jobs finish: run `evaluate --seed 0` then `compare --seed 0`,
  then add the "A6, FreeMan-trained (subject-disjoint OOF)" row to Table 4
  of the manuscript and Table 6 of Online Resource 1.
