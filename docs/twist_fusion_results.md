# Twist-fusion ablation results (A6 → A7 → A8 → A9)

Goal: fuse the trunk axial rotation (体回旋 / twist) better than the baseline
rotation-aware network. Additive ablation ladder over A6 (paper baseline),
trained identically (all137, e100, seed 0, batch 32), scored against the
regenerated triangulated GT with per-sequence similarity alignment.

| ablation | what it adds | trunk-twist idea |
|---|---|---|
| A6 | baseline (full self-supervised objective) | — |
| A7 | **改法4** per-view-peak ROM anchor | ROM target = the wider view, not the average |
| A8 | A7 + **改法2** twist residual | fuse the twist in rotation space (graded pelvis-axis rotation) |
| A9 | A8 + **改法3** observed twist-rate anchor | bound the twist rate to the per-view observed ω |

## Results (137 persons, similarity-aligned)

| metric | A6 | A7 (+改法4) | A8 (+改法2) | A9 (+改法3) |
|---|---:|---:|---:|---:|
| **MPJPE (mm)** | 65.7 | 66.3 | 97.9 | — |
| **ROM retention** | 1.000 | **1.054** | 1.121 | — |
| **peak angular-velocity retention** | 1.000 | 1.087 | **1.911** | — |
| bone-length CV (lower=stiffer) | 0.0202 | 0.0203 | 0.0247 | — |
| rigidity | 0.0161 | 0.0163 | 0.0196 | — |
| joint jerk (lower=smoother) | 5480 | **4397** | 7150 | — |

Paired Wilcoxon (vs A6, 137 persons):
- A7: ROM retention +5.4% (p=7e-7, higher on 81% of people); jerk −20% (p=3e-24,
  lower on 100%); MPJPE +0.6 mm (negligible).
- A8: MPJPE +32 mm / +49% (p=3e-24); peak angular velocity +91% (p=2e-21);
  ROM retention +12% but rigidity and jerk both degrade.

## Findings

1. **改法4 (A7) works — a clean, modest win.** Anchoring the ROM target to the
   wider per-view range (instead of the coordinate average, which shrinks the
   twist) recovers +5.4% ROM retention and makes motion 20% smoother, at
   essentially no MPJPE cost. This confirms the core hypothesis: coordinate-space
   averaging shrinks the trunk twist, and the peak anchor un-shrinks it. Usable
   as-is with the existing model.

2. **改法2 alone (A8) over-rotates.** The twist residual pushes ROM further up
   but massively overshoots — peak angular velocity nearly doubles (unphysical),
   MPJPE degrades 49%, and rigidity/smoothness worsen. The twist knob is powerful
   but unconstrained: nothing bounds its magnitude, and the peak-ROM target just
   drives it larger. This is why 改法3 (an observation anchor on the twist
   magnitude) is not optional.

3. **The value is on the twist axis, not MPJPE.** A7's benefit is invisible in
   MPJPE (unchanged) and lives entirely in ROM retention and jerk. The network
   route should be evaluated and reported on trunk-twist fidelity, not per-frame
   position error.

## A9 (改法3) status — did NOT produce a usable model

A9 trained 100 epochs but is currently unusable, for two compounding reasons:

- **The observed twist-rate loss was ≈0 for every training epoch** (only the
  epoch-0 warmup was non-zero). As implemented, the ω-anchor's valid-frame mask
  almost never engages, so 改法3 barely participated in training. It needs its
  masking/weighting reworked before a retrain is worthwhile.
- **No checkpoint was saved.** best.pt is written only when `val_score >= best`
  (best starts at −∞), and A9's `val_score` was `nan` every epoch (A7's was a
  normal ~0.77). `nan >= −∞` is always False, so best.pt was never written and
  the trained weights were lost (there is no periodic/last checkpoint). The nan
  originates in validation scoring under the A9 loss path; worth making the
  checkpoint logic tolerate a nan score (e.g. save the last epoch as a fallback)
  regardless.

Next step for A9: fix the ω-anchor masking so 改法3 actually engages, guard the
checkpoint against a nan validation score, then retrain. Until then A9's effect
is unmeasured.

## Recommendation

- Ship **A7 (改法4)** as the twist-fidelity improvement: clean, significant,
  no downside.
- Treat **A8 (改法2)** as evidence that the twist residual needs a magnitude
  anchor — do not use it standalone.
- **改法3** is the right idea (bound the twist to observations) but its current
  implementation does not engage; rework before spending another training run.
