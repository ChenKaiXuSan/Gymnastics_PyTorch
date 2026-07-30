# Neurocomputing Manuscript

This directory contains the English manuscript for the rotation-aware,
self-supervised two-view 3D pose fusion study. It uses the official Elsevier CAS
LaTeX template retained under `template/`.

## Build

From the repository root:

```bash
make -C paper/neurocomputing
```

The command regenerates the verified deterministic table and figure, audits the
manuscript, and writes `paper/neurocomputing/build/manuscript.pdf`.

When running from a linked worktree without the ignored `logs/` directory, set
the source repository explicitly:

```bash
GYMNASTICS_SOURCE_ROOT=/home/workspace/kaixu/code/Gymnastics_PyTorch \
  make -C paper/neurocomputing
```

## Evidence Status

The regenerated deterministic experiment matrix contains nine methods and 137
people (1,233 person--method rows). The same upstream inventory contains 137
people with SAM3D outputs, split-cycle alignment records, and triangulated
pseudo-reference directories, comprising 928 triangulated cycle directories.
The lowest numerical deterministic mean belongs to the
pseudo-reference-fitted joint-weight diagnostic and is excluded from
recommendation because it leaks the evaluation target; `avg_body_current` is
the lowest-mean leakage-free method.

The final learned comparison uses the frozen 96/27/14 person
training/validation/test split. The complete A4--A9 evaluation, per-family
robustness analysis, limited Unity native-3D benchmark, and person-disjoint
cohort analysis have all been incorporated into the manuscript.

## Learned Experiments

After the branch is integrated into a checkout that contains
`logs/split_cycle`, run:

```bash
conda run -n gymnastic python -m fuse.rotation_aware prepare \
  --config configs/fuse/rotation_aware.yaml
conda run -n gymnastic python -m fuse.rotation_aware train \
  --config configs/fuse/rotation_aware.yaml --run-id paper_a4 --ablation A4
conda run -n gymnastic python -m fuse.rotation_aware train \
  --config configs/fuse/rotation_aware.yaml --run-id paper_a5 --ablation A5
conda run -n gymnastic python -m fuse.rotation_aware train \
  --config configs/fuse/rotation_aware.yaml --run-id paper_a6 --ablation A6
conda run -n gymnastic python -m fuse.rotation_aware infer \
  --config configs/fuse/rotation_aware.yaml --run-id paper_a4
conda run -n gymnastic python -m fuse.rotation_aware infer \
  --config configs/fuse/rotation_aware.yaml --run-id paper_a5
conda run -n gymnastic python -m fuse.rotation_aware infer \
  --config configs/fuse/rotation_aware.yaml --run-id paper_a6
conda run -n gymnastic python -m fuse.rotation_aware evaluate \
  --config configs/fuse/rotation_aware.yaml \
  --run-id paper_a4 --run-id paper_a5 --run-id paper_a6
```

The archived mainline uses the declared single training seed and reports that
limitation explicitly. Before revising numerical results, regenerate the
corresponding CSV/JSON artifacts, rerun `make`, and repeat the final
claim-evidence review.

## Submission Boundary

The PDF contains the complete learned experiment matrix, statistical analysis,
frozen evidence populations, and downstream cohort analysis. Submission still
requires final confirmation of participant ethics, author affiliation, and
data-sharing language.
