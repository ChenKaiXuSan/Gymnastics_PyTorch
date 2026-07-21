# Neurocomputing Paper Design

## Objective

Create a complete, compilable English manuscript for submission to
*Neurocomputing*. The manuscript presents the rotation-aware self-supervised
fusion method as the main research contribution and uses gymnastics as a
challenging real-world evaluation setting rather than as the sole novelty.

The working title is:

> Rotation-Aware Self-Supervised Fusion of Temporally Aligned Two-View 3D
> Human Poses

## Author And Venue

- Target journal: *Neurocomputing*.
- First and currently sole author: Kaixu Chen.
- Affiliation: CCS.
- Corresponding email: chenkaixusan@gmail.com.
- Manuscript language: English.
- Citation style: numbered Elsevier references.

## Delivery Layout

All manuscript assets live under `paper/neurocomputing/`:

```text
paper/neurocomputing/
  README.md
  Makefile
  manuscript.tex
  references.bib
  sections/
  figures/
  tables/
  template/
  build/
```

The `template/` directory contains the official Elsevier CAS LaTeX template
downloaded from the link provided by the journal's Guide for Authors. The
manuscript uses the official class without modifying its source files.

## Research Positioning

The paper is method-led. Its central problem is learning a robust fused 3D
pose sequence from two temporally aligned but coordinate-inconsistent 3D pose
streams without motion-capture supervision.

The mainline method consumes only face-view and side-view SAM3D-Body 3D
keypoints together with the split-cycle temporal offset. It does not consume
RGB frames, camera parameters, 2D keypoints, meshes, manually annotated trunk
angles, or triangulated 3D poses during training or checkpoint selection.

The method contribution is organized around:

1. subject-level temporal alignment and trial construction;
2. differentiable body-centered canonicalization;
3. rotation-aware trunk and cross-view disagreement features;
4. a view-swap-invariant residual temporal convolutional fusion model;
5. self-supervision from reproducible corruption, consensus, geometry,
   temporal, and complete-cycle rotation objectives;
6. evaluation by person rather than by treating cycles as independent people.

The existing deterministic fusion matrix remains a comparison suite. The
strongest verified deterministic baseline is
`sim3_face_stable_smooth_kpt`.

## Evidence Boundary

Only values verified from repository artifacts may appear as observed results.
The current deterministic matrix contains 68 people and nine methods. Its
verified person-level mean MPJPE values are reported in repository coordinate
units, not converted to millimetres without an independently verified scale.

The triangulated sequences are described consistently as a
`triangulated pseudo-reference`, never as motion-capture ground truth or an
independent 3D ground truth. The paper explicitly states that the
pseudo-reference and the fused candidates share upstream observations and
therefore do not establish absolute 3D accuracy.

Full A4, A5, and A6 model results do not currently exist. Their table cells use
a dedicated visible LaTeX command, `\resultpending{}`, and the surrounding text
does not claim superiority. These markers are designed for deterministic
replacement after the experiments run.

## Manuscript Structure

The manuscript contains:

1. Introduction
2. Related Work
3. Problem Formulation
4. Rotation-Aware Self-Supervised Fusion
5. Experimental Protocol
6. Results
7. Discussion
8. Limitations
9. Conclusion

Front and back matter include an abstract of at most 250 words, one to seven
keywords, three to five highlights of at most 85 characters each, CRediT
contributions, funding, competing interests, data availability, and a
generative-AI disclosure.

## Results Presentation

The initial draft includes:

- the complete nine-method deterministic comparison table;
- person-level aggregation across all 68 people;
- clear unit and pseudo-reference caveats;
- an A0-A6 ablation table whose learned rows remain visibly pending;
- a planned robustness table for fixed synthetic corruptions;
- a planned rotation-preservation table for ROM, angular velocity, and angular
  jerk;
- explicit statistical analysis procedures for repeated seeds and
  person-paired comparisons.

The draft does not fabricate confidence intervals, p-values, learned-model
metrics, public-benchmark results, or downstream classification gains.

## Literature And Citation Policy

References must be traceable to primary publications or official project
pages. Bibliographic metadata and DOI values are checked before inclusion.
Claims about SAM3D-Body, multi-view pose estimation, temporal pose modelling,
self-supervised corruption, Procrustes/Sim3 alignment, and rotation
representations are cited near the relevant text.

The draft may identify a public multi-view benchmark as planned external
validation, but it must not imply that this experiment has already run.

## Figures

The initial paper includes reproducible, code-native diagrams or plots for:

- the complete data and evaluation pipeline;
- the rotation-aware fusion architecture;
- deterministic baseline performance;
- learned-model training and ablation results once available.

Figures use accessible colors and remain legible in grayscale. No generative
AI image is used in the manuscript.

## Build And Verification

The manuscript compiles locally with the repository's available LaTeX engine.
The preferred build command is `make` from `paper/neurocomputing/`, with a
documented fallback if `latexmk` is unavailable. Verification includes:

- successful PDF generation;
- no undefined citations or references;
- abstract and highlight length checks;
- a scan for accidental ground-truth claims;
- a scan showing every pending experimental value;
- confirmation that all reported deterministic values match the source CSV.

## Completion Boundary

This delivery is a complete manuscript draft, not a submission-ready empirical
claim. It becomes empirically complete after full A4, A5, and A6 training over
the declared person-level folds and repeated seeds, statistical analysis,
result insertion, and a final claim-evidence review.
