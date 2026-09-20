# Twist-fusion ablation results (A6 → A7 → A8 → A9)

Goal: fuse the trunk axial rotation (体回旋 / twist) better than the baseline
rotation-aware network. Additive ablation ladder over A6 (paper baseline),
trained identically (all137, seed 0, batch 32, e100), scored against the
regenerated triangulated GT with per-sequence similarity alignment.

> **A9 caveat:** the A9 run was torn down at **epoch 85/100** when the launching
> session ended. Its val_score had plateaued flat (~0.716) for the last ~25
> epochs, so best.pt (the max-val_score checkpoint, ~epoch 83) is converged and
> the numbers below are stable. The remaining 15 epochs would only extend the
> plateau; the effect sizes vs A7 (49% worse MPJPE, 2.35× overshoot) are far too
> large for that to change the conclusion. A clean 100-epoch rerun is available
> if an exact-protocol number is needed for publication.

| ablation | what it adds | trunk-twist idea |
|---|---|---|
| A6 | baseline (full self-supervised objective) | — |
| A7 | **改法4** per-view-peak ROM anchor | ROM target = the wider view, not the average |
| A8 | A7 + **改法2** twist residual | fuse the twist in rotation space (graded pelvis-axis rotation) |
| A9 | A8 + **改法3** observed twist-rate anchor | bound the twist rate to the per-view observed ω |

## Results (137 persons, similarity-aligned)

> **Evidence-status update (2026-07-26):** this 137-person table includes the
> 96 training and 27 validation people and is therefore descriptive, not a
> held-out generalization result. On the 14-person test split, A6/A7/A8/A9 MPJPE
> is 60.78/61.31/92.02/94.11 mm; A7 ROM retention is 0.948 (A6: 1.000) and
> peak-angular-velocity retention is 0.821 (A6: 1.000). Thus the all-person A7
> ROM gain does not reproduce on test, and A7 is not selected over A6 for the
> paper mainline.

| metric | A6 | A7 (+改法4) | A8 (+改法2) | A9 (+改法3) |
|---|---:|---:|---:|---:|
| **MPJPE (mm)** | **65.7** | 66.3 | 97.9 | 99.0 |
| **ROM retention** | 1.000 | **1.054** | 1.121 | 1.010 |
| **peak angular-velocity retention** (1.0=ideal) | **1.000** | 1.087 | 1.911 | 2.352 |
| bone-length CV (lower=stiffer) | **0.0202** | 0.0203 | 0.0247 | 0.0248 |
| rigidity | **0.0161** | 0.0163 | 0.0196 | 0.0197 |
| joint jerk (lower=smoother) | 5480 | **4397** | 7150 | 5822 |

Paired Wilcoxon (vs A6, 137 persons):
- A7: ROM retention +5.4% (p=7e-7, higher on 81% of people); jerk −20% (p=3e-24,
  lower on 100%); MPJPE +0.6 mm (negligible).
- A8: MPJPE +32 mm / +49% (p=3e-24); peak angular velocity +91% (p=2e-21);
  ROM retention +12% but rigidity and jerk both degrade.

Paired Wilcoxon, **A9 vs A7** (137 persons) — the head-to-head that matters:
- MPJPE **+33 mm / +49%** (p=3e-24, worse on 100% of people).
- peak angular-velocity retention **+116%** (1.09 → 2.35, p=1e-15, worse on 82%).
- ROM retention **−4.1%** (1.054 → 1.010, p=0.04) — A9 lost A7's ROM gain, back to baseline.
- joint jerk **+32%**, bone-CV/rigidity **+21–22%** (all p<1e-23) — rougher and less rigid.

Paired Wilcoxon, **A9 vs A8** (did 改法3 tame A8's overshoot? No):
- peak angular-velocity retention **+23%** (1.91 → 2.35, p=8e-19) — the overshoot got
  **worse**, not better; the one thing 改法3 was meant to fix.
- ROM retention −9.8% (p=1e-18); joint jerk −18.6% (p=3e-24, the only improvement).

## Findings

1. **改法4 (A7) is promising only in the descriptive all-person analysis.** Anchoring the ROM target to the
   wider per-view range (instead of the coordinate average, which shrinks the
   twist) recovers +5.4% ROM retention and makes motion 20% smoother, at
   essentially no all-person MPJPE cost. The held-out test split does not
   reproduce the ROM gain, so this supports a follow-up hypothesis rather than a
   deployable improvement.

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

4. **改法3 (A9) does not work — a clean negative result.** With the twist-rate
   anchor added on top of A8, A9 does **not** beat A7 on any axis, and it does not
   even achieve its own purpose (taming A8's overshoot): peak angular-velocity
   retention rose from A8's 1.91 to **2.35** (further from the ideal 1.0), while
   ROM fell back to baseline (1.01) and MPJPE stayed high (99 mm). So A9 is the
   worst of both worlds — it keeps A8's position error and rigidity cost but throws
   away the ROM gain, and makes the twist dynamics *faster/jerkier*, not slower.

   The likely reason is baked into the anchor's definition: 改法3 bounds the fused
   twist rate to the **per-view observed |ω| envelope**, but a single monocular
   view's angular velocity is noisy and peaks *higher* than the triangulated truth
   (per-view peak_ω ≈ 30 rad/s). So the "bound" sits above the real rate — it
   grants headroom rather than removing it, and the free twist residual + the
   peak-ROM push (改法4/改法2) spend that headroom overshooting. A rate anchor
   would need a *tighter, denoised* target (e.g. the cross-view-consistent rate),
   not the wider per-view envelope, and even then A8 already shows the twist
   residual itself is the source of the MPJPE/rigidity damage.

## A9 (改法3) — training-stability bug, found and fixed

The first A9 attempts produced `nan` val_scores and no usable model. Root-caused
to a latent gradient bug, now fixed (commits on `feat/twist-fusion`):

- In `trunk.py::_derivative`, ω = dθ/dt divides by the per-frame interval, which is
  **0 at padded / run-boundary frames**. The value was already zeroed, but the
  division's backward is `grad · (1/dt) = 0 · inf = nan`. A0–A8 never differentiate
  ω, so they never hit it; **A9 is the first ablation that backprops through the
  twist rate**, so it detonated the bug — every weight went `nan` in ~1 epoch while
  the nan-masked loss still *looked* healthy (~428). Fix: divide by a
  guaranteed-positive denominator (no forward-value change for any ablation; only
  the gradient becomes finite). Regression tests at both the trunk and loss level.
- Checkpointing now also persists weights when val_score is non-finite, so a run
  can never again finish with no checkpoint.

After the fix A9 trained cleanly (finite val_score ~0.716, loss 4193 → 3744,
inference nan-free on all 928 cycles). The negative result above is therefore
real, not an artifact of the earlier instability.

## Recommendation

- Keep **A6** as the paper mainline. Treat A7 (改法4) as a follow-up candidate:
  its descriptive all-person gain requires repeated-seed and held-out
  replication before deployment.
- Treat **A8 (改法2)** as evidence that a free twist residual over-rotates and
  costs MPJPE/rigidity — do not use it standalone.
- **Drop 改法3 in its current form.** Bounding to the per-view rate envelope does
  not constrain the twist (the envelope is looser than the truth). If the twist
  residual is pursued further, the constraint has to come from a denoised,
  cross-view-consistent rate/ROM target, not a single view's peak.
