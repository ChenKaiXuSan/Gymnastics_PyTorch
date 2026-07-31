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
- `sn-jnl.cls`, `sn-mathphys-num.bst`: official Springer Nature template files.
- `appendix.sty`, `threeparttable.sty`, `vruler.sty`: local build dependencies.

## Build

From this directory:

```bash
conda run -n gymnastic make all
conda run -n gymnastic make check
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
