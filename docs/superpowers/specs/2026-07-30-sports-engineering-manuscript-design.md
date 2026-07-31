# Sports Engineering Manuscript Design

**Date:** 2026-07-30  
**Status:** Approved by the author in the current task

## Objective

Convert the existing rotation-aware two-view pose-fusion manuscript into a
submission candidate for *Sports Engineering* without running additional large
experiments. The revision must present the work as an applied sports-engineering
study, retain the honest interpretation of the existing evidence, and comply
with the journal's Research Article limits.

## Editorial Positioning

The paper will be positioned around a low-infrastructure workflow for
markerless gymnastics motion analysis:

1. uncalibrated face and side videos are converted to monocular 3D poses;
2. a self-supervised rotation-aware model fuses the synchronized views;
3. triangulated 3D is used only for evaluation, not for training;
4. the fused representation is audited in an exploratory comparison of
   elderly-labeled and student-labeled cohorts over repeated movement cycles.

The principal technical conclusion is not that the learned model is an
unqualified accuracy winner. The defensible conclusion is that body-frame
canonicalization provides most of the in-domain gain, while the learned model
matches strong in-domain baselines and has limited external transfer. The
cohort analysis is an application audit whose effects are explicitly described
as representation-sensitive and non-causal.

## Journal Constraints

The main article will follow the current *Sports Engineering* Research Article
guidance:

- no more than 4,000 words in the main body;
- a 150--250-word abstract;
- no more than 10 figures and tables combined;
- Introduction, Methods, Results, Discussion, and a one-paragraph Conclusion;
- an ethics statement in Methods;
- Statements and Declarations after the references;
- numbered citations in square brackets;
- single-blind author information on the title page.

## Deliverable Structure

The existing Neurocomputing-oriented source remains unchanged. A separate,
flat-source package will be created at:

`paper/sports_engineering/`

The package will contain:

- a single-file Springer Nature LaTeX manuscript;
- a concise Online Resource with detailed ablations and robustness results;
- the bibliography, journal class, bibliography style, and figure assets;
- a cover letter;
- a submission checklist identifying unresolved author-only information;
- automated manuscript and submission-readiness checks.

## Main-Article Evidence

The main article will retain only evidence needed to support the applied
argument:

- private held-out comparison of the proposed method and principal baselines;
- deterministic world-frame versus body-frame averaging result;
- Unity external benchmark;
- cohort-level mixed-model outcomes;
- pose-source sensitivity analysis;
- one compact workflow figure and one cohort-analysis figure.

Full ablation definitions, corruption settings, robustness measurements, and
secondary statistics will be moved to the Online Resource.

## Integrity Rules

- Do not invent ethics approval, consent wording, participant demographics,
  institutional addresses, or ORCID identifiers.
- Refer to the groups as `elderly-labeled` and `student-labeled`, because exact
  ages and demographic covariates are unavailable.
- Do not make causal claims about ageing.
- State clearly that triangulated 3D is evaluation-only.
- State clearly that the proposed learned model is statistically
  indistinguishable from the strongest in-domain baselines.
- Report the negative external-transfer result.
- Treat the cohort analysis as exploratory and representation-sensitive.

## Submission Blockers

The generated package is a journal-formatted submission candidate, not a
file that should be uploaded without author verification. Submission remains
blocked until the author supplies or confirms:

- ethics approval/exemption identifier and consent statement;
- complete institutional affiliation and postal address;
- author list, order, corresponding-author status, and ORCID identifiers;
- funding and competing-interest declarations;
- permission to share the stated data/code availability information.

