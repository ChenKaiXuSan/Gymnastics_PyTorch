# Sports Engineering submission candidate

This directory is an independent journal-formatted version of the two-view 3D
pose-fusion paper. The earlier manuscript remains unchanged in
`../neurocomputing/`.

The article is positioned as an evaluation of a low-infrastructure
post-estimation workflow. Its central evidence is that body-frame
canonicalization provides the clearest in-domain gain, while the complete
learned model does not outperform strong held-out baselines or transfer
zero-shot to the limited Unity benchmark. The repeated-cycle analysis is a
representation-sensitivity case study rather than a causal analysis of ageing.

## Contents

- `manuscript.tex`: single-file main article using the Springer Nature class.
- `online_resource_1.tex`: detailed ablations and secondary results.
- `references.bib`: bibliography used by the main article.
- `cohort_cycle_analysis.pdf`: main-article figure asset.
- `cover_letter.md`: journal-specific cover-letter draft.
- `SUBMISSION_CHECKLIST.md`: technical status and author-only blockers.
- `scripts/check_sports_engineering.py`: structural and evidence checks.
- `scripts/generate_comparison_tables.py`: regenerates the camera-extrinsic and
  14-person per-joint tables from the formal local experiment artifacts.
- `generated/`: source-checked CSV evidence and LaTeX table fragments. Main
  camera and joint tables use the fixed 14-person held-out set; secondary
  coordinate, deterministic and camera tables use all 137 participants. Every
  pseudo-reference table uses one similarity transform per cycle followed by
  framewise hip centring.
- `sn-jnl.cls`, `sn-mathphys-num.bst`: official Springer Nature template files.
- `appendix.sty`, `threeparttable.sty`, `vruler.sty`: local build dependencies.

## Build

From this directory:

```bash
conda run -n gymnastic make all
conda run -n gymnastic make check
```

`make all` validates and reuses the committed 137-person matched-metric cache,
then derives the 14-person main camera and joint rows plus the 137-person
secondary tables. This step
requires the formal `local/runs/fuse_experiments`,
`local/runs/fuse_extrinsic_baselines` and `local/runs/fuse_rotation_aware`
artifacts plus the triangulated pseudo-reference. The generated table fragments
are committed and included in the submission source archive, so the journal
build itself does not require access to participant data.

To rebuild every 137-person metric directly from compact fused sequences rather
than the protocol-checked cache, run:

```bash
conda run -n gymnastic python scripts/generate_comparison_tables.py
```

The PDFs are written to `build/manuscript.pdf` and
`build/online_resource_1.pdf`.

`make submission-check` intentionally fails while the ethics, full affiliation
and cover-letter author-action placeholders remain. This protects against
accidental upload of an administratively incomplete manuscript.

## Submission archive

After all blocking author fields have been resolved:

```bash
conda run -n gymnastic make submission-check
conda run -n gymnastic make package
```

The package target creates `submission/sports_engineering_source.zip` from the
flat source files and copies the compiled PDFs and cover letter into
`submission/`.
