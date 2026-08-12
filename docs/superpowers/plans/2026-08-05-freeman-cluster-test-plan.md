# FreeMan Cluster Test Plan

**Goal:** Finish the FreeMan zero-shot validation of the rotation-aware fusion
checkpoints by migrating the interrupted benchmark run from the retired
dual-GPU workstation to the NQSV cluster (`gpu` queue, account `HP260146`),
producing first pilot numbers quickly and a full-protocol table afterwards.

**Starting state (verified 2026-08-05):**

- All FreeMan archives (773 GB) and 912 extracted session videos (771 GB) are
  on the cluster at
  `/work/HP260146/chenkaixu/public_datasets/multiview_human/FreeMan/`.
- SAM3D-Body checkpoint at `local/checkpoints/models/sam-3d-body-dinov3`;
  `sam_3d_body` conda env has torch 2.10 cu128.
- Rotation-aware checkpoints referenced by the benchmark
  (`all137_a4/a5/a6_e100_seed0`) verified present and loadable under
  `local/runs/fuse_rotation_aware/runs/`; A6 seeds 1/2 and A10/A11 also exist.
- Previous run stopped 2026-07-28 at ~13/22 sessions of subject 01 of 40.
- `configs/benchmarks/freeman.yaml` and the saved manifests embed workstation
  paths (`/home/workspace/kaixu/datasets/FreeMan/...`); the inspect stage must
  be re-run after re-pointing.
- Cluster `gpu` queue: 24 h elapse limit per request, GPU count effectively
  unlimited, queue currently busy (~26 running / 35 queued).
- Volume: subject 01 has 22 sessions, 36,737 frames; full release ≈ 912
  sessions, ≈ 2.8 M frame-inferences × 2 views at stride 1.

## Phase 0 — Port and smoke test (half a day)

- [ ] Create `configs/benchmarks/freeman_cluster.yaml` from `freeman.yaml`
      with `paths.archive_root/manifest_root/work_root` pointed at the
      cluster FreeMan tree (keep `output_root: local/runs/freeman_benchmark`).
- [ ] `benchmark freeman download --config configs/benchmarks/freeman_cluster.yaml`
      — verifies existing archives (sha256) and marks the stage complete; no
      re-download expected.
- [ ] `benchmark freeman inspect --config ...` — rebuilds manifests with
      cluster paths. Confirm session counts match the old manifests
      (subject 01: 22 sessions) and `reference_scale_to_m` is recorded.
- [ ] Submit one short `gpu`-queue job:
      `run --subject 1 --frame-stride 25` — end-to-end smoke of extract →
      infer → fuse → evaluate on a coarse stride. Confirms GPU node model,
      VRAM fit, and throughput (record frames/s for sizing Phase 2).
- [ ] Decide the fate of the 13 workstation-era `sam3d/subject_01` caches:
      keep if identity validation accepts them, otherwise let the pipeline
      re-infer (small loss).

## Phase 1 — Pilot numbers (1–2 days)

- [ ] Pick 3–4 subjects rich in trunk-twist-like sessions per
      `trunk_twist_ranking.csv` (e.g. subj 22, 12, 36) plus subject 01.
- [ ] One qsub job per subject, `--frame-stride 3`, single device.
- [ ] Extend `rotation_aware.run_ids` with `all137_a6_s1_e100`,
      `all137_a6_s2_e100` (and optionally `paper137_a10/a11`) — SAM3D caches
      are shared, so extra checkpoints only add cheap fuse/evaluate work.
- [ ] Run `evaluate` + `report` restricted to the pilot subjects
      (`--subject ...`); sanity-check:
      - fusion beats both single views per subject;
      - A6 vs A2 gap direction consistent with the internal 137-person data;
      - PCK@50/100/150 and AUC in plausible ranges;
      - camera pairs near 90° separation.
- [ ] Gate: if pilot metrics look wrong, stop and diagnose before spending
      full-run GPU hours.

## Phase 2 — Full protocol (paper table)

- [ ] Decide stride for the headline table: the design doc mandates stride 1;
      fall back to stride 2–3 only if queue pressure makes stride 1
      impractical, and report the stride honestly.
- [ ] Submit 40 per-subject qsub jobs (24 h limit each). The staged cache
      makes resubmission idempotent — a resubmitted job skips complete
      sessions, so stragglers just need another submission.
- [ ] Track completion via `run_state.json` / per-subject session manifests;
      `evaluation.minimum_subject_coverage: 0.95` requires ≥ 38/40 subjects.
- [ ] Final `report` runs once on a CPU node (aggregate only).

## Phase 3 — Integration

- [ ] Fold the report into `docs/results_summary.md` (resolves the
      "completed public synthetic benchmark evaluation" pending item).
- [ ] Manuscript wording: FreeMan reference is markerless multi-view, not
      independent motion capture; zero-shot means no FreeMan data touched
      training, checkpoint selection, pairing, or tuning.

## Risks

- GPU node model/VRAM unknown until the smoke job runs.
- Queue contention is the main wall-clock driver for Phase 2.
- Old subject-01 caches may fail identity validation (acceptable re-run cost).
- Per-subject extraction into `work_root` needs transient disk; Lustre has
  4.7 PB free — do not enable `--keep-workspace` so cleanup stays automatic.
