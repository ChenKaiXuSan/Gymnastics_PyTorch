# Sports Engineering Manuscript Implementation Plan

> **Goal:** Produce a journal-formatted, evidence-consistent submission
> candidate while preserving the existing manuscript and experiments.

## Task 1: Create an isolated submission package

- Create `paper/sports_engineering/`.
- Copy the official Springer Nature class and numerical bibliography style.
- Copy only the figure assets and bibliography required by the new manuscript.
- Keep the existing `paper/neurocomputing/` package unchanged.

## Task 2: Rewrite the main manuscript

- Use a single-file Springer Nature source with line numbering.
- Replace the method-centric title and framing with an applied
  sports-engineering title.
- Write a 150--250-word abstract without undefined abbreviations.
- Compress the Introduction to the unmet engineering need, research questions,
  and contributions.
- Consolidate data, method, evaluation, cohort analysis, and ethics into
  Methods.
- Restrict Results to the primary in-domain, external, and cohort findings.
- Interpret canonicalization, external transfer, and representation sensitivity
  in Discussion.
- End with one concise Conclusions paragraph.
- Keep the body at or below 4,000 words and the figure/table total at or below
  10.

## Task 3: Build an Online Resource

- Document the full A0--A9 ablation matrix.
- Include corruption settings and robustness results.
- Include secondary deterministic comparisons and statistical details.
- Ensure every supplementary number is traceable to the existing experiment
  artifacts.

## Task 4: Add submission materials

- Draft a journal-specific cover letter.
- Add an author-facing submission checklist.
- Document required files, compilation instructions, and unresolved blockers.
- Add automated checks for structure, abstract length, body length, figure/table
  count, citations, and unresolved submission placeholders.

## Task 5: Verify the package

- Compile the manuscript and Online Resource with the project conda
  environment.
- Run automated manuscript checks.
- Inspect representative PDF pages for layout defects.
- Produce a flat source archive only after compilation succeeds.
- Report unresolved author-only submission blockers separately from technical
  build status.

