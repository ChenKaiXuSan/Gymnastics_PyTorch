# Archived fusion configurations

The final architecture of the cycle-aware fusion model is **v1.2**
(`model/v1_2.yaml` + `loss/v4.yaml`, the Hydra defaults since 2026-09-25).
Everything below is archived: kept so that published and earlier runs can be
reproduced, not to be extended. The model code still implements every
version through its switches, so the archived configs run unchanged.

| Config | What it is | Selected with |
|---|---|---|
| `model/archive/v1.yaml` | Architecture v1.0: scalar convex base `w_A P_A + w_B P_B` (no depth prior) | `model=archive/v1` |
| `model/archive/v1_1.yaml` | Architecture v1.1: depth-aware base with learned reliability weights | `model=archive/v1_1` |
| `loss/archive/v1_recovery.yaml` | Loss v1: recovery toward the plain two-view average | `loss=archive/v1_recovery` |
| `loss/archive/v2.yaml` | Loss v2: v1 + cross-cycle, periodicity and symmetry terms | `loss=archive/v2` |
| `loss/archive/v3.yaml` | Loss v3: recovery toward the depth-aware rule + `L_rel` 0.02 + `L_res` 0.01 | `loss=archive/v3` |

Archived presets (`experiment=archive/<name>`), each pinned to the model and
loss it ran with:

| Preset | Model / loss | Notes |
|---|---|---|
| `v1` | v1 / v1_recovery | loss v1 on the v1.0 base |
| `v2` | v1 / v2 | loss v2 on the v1.0 base (`gymnastics_v2_5fold_seed0`) |
| `v1_0_base` | v1 / v3 | alpha = 0 base ablation |
| `equal_reliability` | v1_1 / v3 | `*_v1_1_equal_reliability_*` (numerically the v1.2 model: with the head off `L_rel` has no gradient) |
| `no_residual` | v1_1 / v3 | `*_v1_1_no_residual_*` (on v1.2 this would be the closed-form rule with nothing to learn) |
| `measurement` | v1 / v1_recovery | position-level periodicity / symmetry priors |
| `periodicity_contrastive`, `periodicity_cosine` | v1 / v2 | `shortdiag_*` |
| `reference_supervised_v1` | v1 / v1_recovery | reference target, loss v1 (`*_v1_reference_supervised_*`) |
| `reference_supervised_v3` | v1_1 / v3 | `*_v1_1_refsup_*` |

Why v1.2 replaced v1.1:
`docs/research/strict_external_baselines_2026-09-21.md`, "Where the learned
part helps". The v1.1 ablation is in
`docs/research/cycle_aware_ablation_v1_1_2026-09-23.md`.
