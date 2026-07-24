# Neurocomputing Manuscript Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete, source-grounded, compilable English Neurocomputing manuscript for the rotation-aware two-view 3D pose fusion study.

**Architecture:** Keep the paper isolated under `paper/neurocomputing/`, vendor the unmodified official Elsevier CAS template, split prose into numbered section files, and generate quantitative tables and plots directly from repository CSV artifacts. Learned A4-A6 results remain visibly marked by a dedicated LaTeX command until full experiments exist; all currently stated numerical conclusions must be reproducible from `logs/fuse_experiments/metrics_by_person.csv`.

**Tech Stack:** Elsevier CAS LaTeX, BibTeX, latexmk, Python standard library, matplotlib, Make.

## Global Constraints

- Target journal is *Neurocomputing* and the manuscript is written in English.
- The current author is Kaixu Chen, CCS, chenkaixusan@gmail.com.
- Use the official Elsevier CAS LaTeX class without modifying class or bibliography-style sources.
- Treat `codex/rotation-aware-fusion` and its existing worktree as the implemented method source; do not merge or alter that branch while writing the paper.
- Do not report unrun A4, A5, or A6 values as results.
- Call triangulated evaluation data a `triangulated pseudo-reference`, never independent ground truth.
- Do not convert repository coordinate units to millimetres without an independently established scale.
- Preserve the user's unrelated `AGENTS.md` change and untracked split-cycle plan.
- Prefix project Python commands with `conda run -n gymnastic`.

---

### Task 1: Vendor The Official Template And Scaffold The Manuscript

**Files:**
- Create: `paper/neurocomputing/template/els-cas-templates.zip`
- Create: `paper/neurocomputing/template/README.md`
- Create: `paper/neurocomputing/README.md`
- Create: `paper/neurocomputing/Makefile`
- Create: `paper/neurocomputing/.gitignore`
- Create: `paper/neurocomputing/manuscript.tex`
- Create: `paper/neurocomputing/highlights.txt`

**Interfaces:**
- Consumes: official URL `https://assets.ctfassets.net/o78em1y1w4i4/5uFmLZJTPDMAUjFnHRpjj8/6f19a979146eb93263763d87a894ab0d/els-cas-templates.zip`.
- Produces: a local Elsevier CAS template archive, extracted class/style files, and a minimal manuscript that `latexmk` can compile.

- [ ] **Step 1: Download and fingerprint the official archive**

Download the archive directly from the link exposed by the Neurocomputing Guide for Authors, save the original zip, record the retrieval date, URL, and SHA-256 in the template README, and list its contents before extraction.

- [ ] **Step 2: Extract the unmodified CAS distribution**

Extract into `paper/neurocomputing/template/els-cas-templates/`. Select the long single-column CAS class suitable for a journal manuscript and expose the required `.cls`, `.sty`, and `.bst` files to the manuscript through `TEXINPUTS` and `BIBINPUTS` in the Makefile.

- [ ] **Step 3: Create the build contract**

The Makefile must implement:

```make
.PHONY: all check clean
all: check
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build manuscript.tex
check:
	conda run -n gymnastic python scripts/check_manuscript.py
clean:
	latexmk -C -outdir=build manuscript.tex
```

It may add environment variables required to find the vendored class and bibliography styles, but must preserve these targets.

- [ ] **Step 4: Compile the minimal shell**

Run: `make -C paper/neurocomputing`

Expected: `paper/neurocomputing/build/manuscript.pdf` exists and the log has no fatal error.

### Task 2: Generate Verified Baseline Assets

**Files:**
- Create: `paper/neurocomputing/scripts/generate_paper_assets.py`
- Create: `paper/neurocomputing/scripts/check_manuscript.py`
- Create: `paper/neurocomputing/tables/deterministic_baselines.tex`
- Create: `paper/neurocomputing/figures/deterministic_mpjpe.pdf`
- Create: `paper/neurocomputing/artifacts/deterministic_summary.csv`

**Interfaces:**
- Consumes: `logs/fuse_experiments/metrics_by_person.csv` with fields `person_id`, `method`, `eval_frames`, `valid_points`, `mpjpe`, `median`, `p95`, and `max_error`.
- Produces: person-level method summaries sorted by mean MPJPE, a matching LaTeX table, and a publication-quality plot.

- [ ] **Step 1: Implement source validation**

The generator must assert exactly nine methods, 68 unique people per method, finite metrics, and one row per person-method pair. It must fail if these contracts change.

- [ ] **Step 2: Compute person-level statistics**

For each method compute `n`, arithmetic mean, sample standard deviation, median, and interquartile range of person-level MPJPE. Preserve repository coordinate units.

- [ ] **Step 3: Emit a deterministic table and figure**

Use a colorblind-safe palette, show mean MPJPE with 95% bootstrap confidence intervals across people using a fixed seed, and annotate in the caption that the reference is triangulated and not independent ground truth. The table's verified ordering must begin with `sim3_face_stable_smooth_kpt` at mean `0.790351` before rounding.

- [ ] **Step 4: Verify generated values**

Run: `conda run -n gymnastic python paper/neurocomputing/scripts/generate_paper_assets.py`

Expected: the script reports nine methods, 68 people, 612 rows, and writes all three outputs without changing source CSV files.

### Task 3: Build A Verified Bibliography

**Files:**
- Create: `paper/neurocomputing/references.bib`
- Create: `paper/neurocomputing/artifacts/source_audit.md`

**Interfaces:**
- Consumes: primary publication pages, DOI records, and official project publications.
- Produces: BibTeX entries used by the manuscript and a concise provenance table containing title, year, venue, DOI or canonical URL, and the manuscript claim supported.

- [ ] **Step 1: Verify the closest method families**

Include primary sources for calibrated and uncalibrated multi-view triangulation, image/heatmap cross-view fusion, self-supervised multi-view pose learning, temporal 3D pose estimation, and video-based biomechanics. The corpus must include Iskakov et al. 2019, Qiu et al. 2019, Pavllo et al. 2019, Kim et al. 2022, MetaPose 2022, SelfPose3D 2024, and OpenCap 2023.

- [ ] **Step 2: Verify method foundations**

Include sources for Sim3/Procrustes alignment, continuous rotation representations, temporal convolutional networks, and the public pose benchmarks named as future external validation.

- [ ] **Step 3: Audit bibliography consistency**

Every BibTeX key must appear in `source_audit.md`; every DOI or canonical URL must resolve to the matching title; no citation may be added solely from an unverified secondary summary.

### Task 4: Draft Front Matter, Introduction, And Related Work

**Files:**
- Create: `paper/neurocomputing/sections/01_introduction.tex`
- Create: `paper/neurocomputing/sections/02_related_work.tex`
- Modify: `paper/neurocomputing/manuscript.tex`
- Modify: `paper/neurocomputing/highlights.txt`

**Interfaces:**
- Consumes: the approved research positioning and verified bibliography.
- Produces: a self-contained abstract no longer than 250 words, one to seven keywords, four compliant highlights, a problem-led introduction, and a related-work section with explicit closest-work contrasts.

- [ ] **Step 1: Write the abstract and contribution statement**

State the no-Mocap/no-camera-training setting, pose-level fusion problem, rotation-aware self-supervision, full 70-joint output, and evaluation plan. Do not state that the learned model outperforms baselines before results exist.

- [ ] **Step 2: Write the introduction**

End the introduction with four bounded contributions: problem formulation, rotation-aware swap-invariant residual fusion, self-supervised objectives, and person-level evaluation protocol.

- [ ] **Step 3: Write related work by research family**

Contrast the study against triangulation, image-level fusion, self-supervised multi-view estimation, temporal monocular refinement, and markerless biomechanics. Identify Cross-View Self-Fusion and MetaPose as close but input/output-distinct methods.

### Task 5: Draft The Mathematical Method

**Files:**
- Create: `paper/neurocomputing/sections/03_problem_formulation.tex`
- Create: `paper/neurocomputing/sections/04_method.tex`
- Create: `paper/neurocomputing/figures/pipeline.tex`
- Create: `paper/neurocomputing/figures/architecture.tex`

**Interfaces:**
- Consumes: implementation in `/home/workspace/kaixu/code/Gymnastics_PyTorch/.worktrees/rotation-aware-fusion/fuse/rotation_aware/` and its configuration files.
- Produces: equations and diagrams that match implemented data, geometry, feature, model, corruption, and loss contracts.

- [ ] **Step 1: Formalize inputs, masks, timing, and coordinate boundary**

Define paired streams, split-cycle offset, valid masks, cycle trials, 60 Hz physical time, canonical output, and face-reference compatibility output. Explicitly state that split-cycle alignment is upstream and fixed rather than learned by the fusion network.

- [ ] **Step 2: Formalize canonical geometry and trunk rotation**

Define pelvis and thorax frames, trial-level scale, canonical transform, relative rotation `R_p^T R_t`, wrapped axial angle, angular velocity, and angular acceleration with validity masks.

- [ ] **Step 3: Formalize symmetric residual fusion**

Describe fixed quality-weighted base fusion, shared view encoder, symmetric mean/absolute-difference aggregation, six-block dilated TCN, bounded per-joint residual, and recomputation of trunk kinematics from fused keypoints.

- [ ] **Step 4: Formalize self-supervision**

Describe seven corruption families and nine implemented objectives, making clear that pseudo-targets are formed only from unmodified input consensus or a quality-dominant view. Separate A4, A5, and A6 by enabled objective groups.

- [ ] **Step 5: Draw implementation-matched diagrams**

Use TikZ or PGF with no generated imagery. The diagrams must distinguish training-only self-supervision from evaluation-only triangulated pseudo-reference access.

### Task 6: Draft Experimental Protocol And Results

**Files:**
- Create: `paper/neurocomputing/sections/05_experimental_protocol.tex`
- Create: `paper/neurocomputing/sections/06_results.tex`
- Create: `paper/neurocomputing/tables/ablation_matrix.tex`
- Create: `paper/neurocomputing/tables/robustness_matrix.tex`
- Create: `paper/neurocomputing/tables/rotation_matrix.tex`

**Interfaces:**
- Consumes: dataset inventory, deterministic artifacts, implemented A0-A6 registry, and experiment configuration.
- Produces: a reproducible protocol and an evidence-bounded results section.

- [ ] **Step 1: Document data and splitting**

Report 68 people, two views, 458 recorded cycles where supported by the active inventory, 70 joints, split-only offsets, person-disjoint folds, and cycle/window construction. Explain that cycles are samples but people are the final statistical units.

- [ ] **Step 2: Document baselines and metrics**

Cover the nine legacy deterministic methods and A0-A6 learned-study registry. Report pseudo-reference MPJPE separately from no-reference structural, temporal, ROM, corruption-recovery, and swap metrics.

- [ ] **Step 3: Insert verified deterministic results**

Include the generated table and plot and restrict interpretation to relative agreement with the shared triangulated pseudo-reference.

- [ ] **Step 4: Insert visible learned-result markers**

Define `\resultpending{metric-name}` so pending cells render as `Experiment pending: metric-name`. Do not use invisible blanks, zeros, fabricated dashes, or prose implying completion.

- [ ] **Step 5: Pre-register statistical reporting**

Specify at least three seeds, paired person-level bootstrap intervals, paired nonparametric comparisons with multiplicity control, and effect sizes. Mark these analyses as planned until the underlying runs exist.

### Task 7: Draft Interpretation, Limitations, And Declarations

**Files:**
- Create: `paper/neurocomputing/sections/07_discussion.tex`
- Create: `paper/neurocomputing/sections/08_limitations.tex`
- Create: `paper/neurocomputing/sections/09_conclusion.tex`
- Create: `paper/neurocomputing/sections/declarations.tex`

**Interfaces:**
- Consumes: verified baseline findings and the approved evidence boundary.
- Produces: an interpretation that separates established evidence from hypotheses and complete Neurocomputing back matter.

- [ ] **Step 1: Interpret only observed baseline evidence**

Discuss the small but consistent ordering among deterministic methods without calling pseudo-reference agreement absolute accuracy. Present the learned approach as the tested hypothesis awaiting full runs.

- [ ] **Step 2: State limitations explicitly**

Cover shared SAM3D upstream bias, lack of independent Mocap, private two-view data, coordinate-scale ambiguity, fixed upstream temporal offset, and limited external generalization evidence.

- [ ] **Step 3: Add declarations**

Include Kaixu Chen's CRediT roles, no specific funding unless provided, no competing interests, a privacy-aware data statement, and the required generative-AI manuscript-preparation disclosure with human responsibility.

### Task 8: Compile And Audit The Full Package

**Files:**
- Modify: `paper/neurocomputing/scripts/check_manuscript.py`
- Modify: `paper/neurocomputing/README.md`
- Generate: `paper/neurocomputing/build/manuscript.pdf`
- Generate: `paper/neurocomputing/build/manuscript.log`

**Interfaces:**
- Consumes: the complete source package.
- Produces: a reproducible PDF and an audit report suitable for experiment-result handoff.

- [ ] **Step 1: Implement manuscript checks**

Check abstract word count, highlight count and character limits, unresolved citations/references, all `\resultpending` locations, forbidden independent-GT phrasing, author metadata, and equality between the deterministic table and generated summary.

- [ ] **Step 2: Build from clean state**

Run: `make -C paper/neurocomputing clean`

Run: `make -C paper/neurocomputing`

Expected: check succeeds and PDF is produced from a clean build.

- [ ] **Step 3: Inspect compiler diagnostics**

Run: `rg -n "Undefined|Citation.*undefined|Reference.*undefined|LaTeX Error|Fatal error" paper/neurocomputing/build/manuscript.log`

Expected: no matches.

- [ ] **Step 4: Verify repository scope**

Run: `git status --short`

Expected: manuscript and this plan are the only new task files; pre-existing unrelated changes remain untouched.

- [ ] **Step 5: Report empirical completion boundary**

The README must list the exact A4-A6 commands, result files expected, and the replacement workflow for `\resultpending` markers. The final delivery must say plainly that the paper is a complete draft but not submission-ready until those runs are inserted and reviewed.
