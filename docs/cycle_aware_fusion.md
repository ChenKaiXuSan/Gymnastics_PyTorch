# Cycle-Aware Dual-View 3D Pose Fusion (Architecture v1.0)

`fusion` fuses two independent monocular 3D pose
estimates of one person (View A = face camera, View B = side camera on the
private data) into one refined sequence, treating the repeated-cycle
structure of the motion as a first-class signal. It is a pure PyTorch model
wrapped by PyTorch Lightning and configured with Hydra; it is independent of
the rotation-aware model except for the shared SAM3D inputs, the canonical
body frame, and the MHR70 skeleton metadata.

## 1. Method

### 1.1 Inputs and coordinate frame

Both views are mapped independently into the pelvis-centred canonical body
frame (`fusion.keypoints.geometry.canonicalize_pose`): origin
at the hip midpoint, x from left to right hip, y along pelvis→thorax, z
completing a right-handed frame, and lengths divided by the median torso
length of the sequence. The two views are therefore directly comparable
without calibration; the transform of View A is stored so the fused pose can
be mapped back into its world frame.

### 1.2 Cycles, middles and phase normalisation

Cycle detection is **not** part of the training code. It runs once, offline,
in `cycle_alignment` (`cycle_alignment/cycles.py`) and writes record files
(`cycle_alignment/cycle_records.py`) that the DataModules read through
`fusion/data/cycle_records.py`:

| Dataset | Command | Record |
|---|---|---|
| private | `python -m cycle_alignment align` (boundaries) + `python -m cycle_alignment cycles private` (middles) | `local/runs/split_cycle/person_<id>/alignment_record_<id>.json` |
| FreeMan | `python -m cycle_alignment cycles freeman` | `local/runs/cycle_records/freeman/subject_NN/<session>.json` |
| Unity | `python -m cycle_alignment cycles unity` | `local/runs/cycle_records/unity/subject_<seq>/<seq>.json` |

One shared definition is used everywhere: the right-wrist azimuth in the
pelvis body frame (smoothed, unwrapped) starts a cycle each time it crosses
the reference angle upwards (counter-clockwise), and the **middle** of a
cycle is the turn-around frame, the extremum of that signal inside the cycle.
Every record stores the detection settings that produced it.

For an annotated cycle `[start, mid, end)` the phase of frame `t` is
`phi = (t - start) / (end - start)` and `half_index` is 0 on `[start, mid)`
(outward motion) and 1 on `[mid, end)` (return). Before windowing, every
complete cycle is resampled to `S = samples_per_cycle` samples by linear
interpolation (`phase.normalize_sample_to_phase`): each half gets `S / 2`
samples, so the middle lands exactly at phase 0.5 and the time-reversed
mirror of sample `k` of a cycle is sample `S - k` (the hook for the symmetry
objective). Physical timestamps are kept, so the sample interval `delta_t`
still measures real time and velocities are physical:

```
v[t] = (P[t] - P[t-1]) / delta_t[t]
```

Sequences whose record lists no cycles keep their native sampling and an
invalid phase; the model then degrades gracefully to plain temporal
modelling.

### 1.3 Network

All representations are `[B, T, J, C]`. One encoder (shared weights) is
applied to each view:

| Stage | Operation | Module |
|---|---|---|
| Pose Branch | attention over the `J` joints of one frame → `F_pose` | `modules/spatial_transformer.py` |
| Short motion | per-joint temporal attention restricted to a band of ≈ `cycle_ratio · S` samples on `[P ; v]` → `F_short` | `modules/short_motion.py` |
| Long motion | per-joint temporal attention over the whole window on `[P ; v ; sin 2πφ ; cos 2πφ]` → `F_long` | `modules/long_motion.py` |
| Motion fusion | `F_motion = MLP([F_short ; F_long])` | `modules/motion_fusion.py` |
| FiLM | `H = (1 + γ(F_motion)) · F_pose + β(F_motion)` with zero-initialised γ, β | `modules/film.py` |
| Cross-view | `C_A = H_A + Attn(H_A ← H_B)`, `C_B = H_B + Attn(H_B ← H_A)` over the joints of the same frame | `modules/cross_view_attention.py` |
| Reliability | `R_A = g([C_A ; C_B ; flags])`, `R_B = g([C_B ; C_A ; flags])`, `[w_A, w_B] = softmax([R_A, R_B])` | `modules/reliability.py` |
| Weighted fusion | `P_base = w_A · P_A + w_B · P_B` on the **original** inputs | `modules/weighted_fusion.py` |
| Residual | `ΔP = max_delta · tanh(MLP([w_A C_A + w_B C_B ; |C_A − C_B| ; P_base]))`, `P_hat = P_base + ΔP` | `modules/residual_refinement.py` |

Tensor folding is documented in each module: the spatial transformer folds
`B·T` and attends over `J`; the temporal transformers fold `B·J` and attend
over `T`; cross-view attention folds `B·T` and attends over `J` of the other
view. Invalid joints and padding frames are excluded from attention through
explicit masks and produce zero features.

Properties enforced by tests (`tests/fusion`):

* `w_A + w_B = 1` for every `(b, t, j)`; a joint valid in one view only takes
  that view; a joint valid in neither is flagged invalid.
* At initialisation FiLM and the residual are identities, so the model equals
  the reliability-weighted fusion.
* Swapping the views swaps the outputs (symmetry).
* Every parameter receives a gradient in a full forward/backward pass.

### 1.4 Objectives (version 2, default `loss=v2`)

No target is built from the current window's own two views.  Five terms
(`losses.py`):

```
L = 1.0 · L_cycle + 0.05 · L_rel + 0.1 · L_period + 0.1 · L_sym + 0.01 · L_res
```

* **`L_cycle` (main pose supervision)**: leave-one-cycle-out cross-cycle
  target (`cycle_target.py`, computed by the DataModule and shipped as
  `cycle_target` / `cycle_confidence`).  For cycle *i* the candidates are the
  canonical poses of *both* views at the same normalised phase in the other
  cycles (±`neighbors`, default 2); the target is their per-joint median and
  the confidence `C = 1 / (1 + (MAD/τ)²)` (τ = 0.05 torso lengths, zero
  below `min_candidates`).  Confidence-weighted Huber distance.  On the
  private data the same phase of the neighbouring cycle is 3× closer than the
  other view of the same frame (0.048 vs 0.144 torso lengths).
* **`L_rel`**: cross-entropy on the reliability logits where synthetic
  corruption damaged exactly one view (label = the undamaged view); the
  only term that trains `w_A / w_B` directly.
* **`L_period`**: `1 − cos(F_M(φ, i), F_M(φ, i+1))` on the motion feature of
  adjacent cycles.
* **`L_sym`**: `1 − cos(F_M(φ, j), F_M(φ + 0.5, mirror(j)))` within a cycle,
  mirror = left/right joint swap.  The two halves of a trunk-rotation cycle
  are mirror states (twist +0.36 / −0.05 / −0.38 / +0.10 rad at phases
  0 / 0.25 / 0.5 / 0.75 on the private data), but only approximately, hence
  the small weight.
* **`L_res`**: L1 norm of `ΔP`; with the recovery target gone this is what
  anchors the refinement to the measurements.

Synthetic corruption (`corruptions.py`: joint masks, distal-joint blocks,
Gaussian jitter, depth drift, temporal and contiguous frame dropout) is still
applied to the training inputs: it provides the labels of `L_rel`, and the
cross-cycle target is taken from the clean inputs of the *other* cycles, so
it is unaffected.

**Version 1 (`loss=v1_recovery`, preset `experiment=v1`)** is the earlier
recovery objective and stays available for reproduction:

```
L = w_rec · L_recovery + w_per · L_periodicity + w_sym · L_symmetry + w_half · L_half_symmetry + w_res · L_residual
```

`L_recovery` reproduces the clean two-view target under corruption (average
where the views agree within `consensus_distance`, the single valid view
otherwise).  On clean inputs that target *is* the average, so a v1 model
equals the reliability-weighted average on clean data (5-fold private
result: 27.5 mm for both) and only differs under corruption (−44 % on
corrupted joints).  `recovery_target=reference` with
`data.train_with_reference=true` (preset `experiment=reference_supervised`,
v1 family) replaces the target by the attached reference pose, brought into
the frame of the detached weighted base by a per-frame similarity transform,
on datasets whose reference is independent of the inputs (FreeMan, Unity);
the private adapter refuses it.  A checkpoint trained that way is applied to
the private data with `checkpoint=<path> test_only=true` (zero-shot) or
fine-tuned label-free with `checkpoint=<path>`.  Position-level periodicity /
half-cycle symmetry (`experiment=measurement`) hurt every metric on the
private data.

### 1.5 Evaluation

Validation and test windows may carry a reference pose (triangulated
pseudo-reference, FreeMan `keypoints3d_optim`, Unity native 3D). Because the
fused pose lives in the canonical frame of View A, the primary metric is the
Procrustes-aligned MPJPE (`metrics.py`); translation-aligned MPJPE is also
logged. Both are reported for `P_hat` and for `P_base` so the contribution
of the residual is visible. References are never attached to training
windows.

## 2. Datasets

Each adapter documents its source layout, skeleton, coordinate system,
synchronisation, ground truth and cycle availability in its module
docstring.

| Adapter | Module | Subject id | Cycles + middles | Reference |
|---|---|---|---|---|
| Private gymnastics | `data/gymnastics.py` | person id | `alignment_record_<id>.json` (`require_cycle_mids`); consecutive cycles concatenated | triangulated pseudo-reference matched by face/side frame pairs |
| FreeMan | `data/freeman.py` | subject number | `cycle_records/freeman` (`require_cycle_records`) | `keypoints3d_optim` scaled to metres, COCO17 → MHR70 positions |
| Unity | `data/unity.py` | sequence id | `cycle_records/unity` (`require_cycle_records`) | native Unity22 joints → MHR70 positions |
| Synthetic | `data/synthetic.py` | generated | exact | generating motion |

The private adapter uses the rotation-aware person cache by default
(`options.source: cache`) and the paper split
`src/configs/shared/folds/paper_137_a6_split.json`. FreeMan reads the SAM3D
cache of the zero-shot benchmark (`local/runs/freeman_benchmark_cluster`);
Unity reads the benchmark manifest plus `local/runs/unity_benchmark/sam3d`.
Converted samples can be cached with `data.cache_dir` (on by default for the
three real datasets).

## 3. Configuration

```
src/configs/fusion/
├── config.yaml              root: samples_per_cycle, num_cycles, seed, run_name, output_root
├── model/v1.yaml            architecture and ablation switches
├── data/{synthetic,gymnastics,freeman,unity}.yaml   (+ _common.yaml)
├── loss/default.yaml
├── corruption/{default,none}.yaml
├── trainer/{default,debug}.yaml
├── optimizer/default.yaml
└── experiment/*.yaml        @package _global_ presets (ablations, smoke)
```

`samples_per_cycle` and `num_cycles` are defined once at the root and
interpolated into both the model and the data window, so the training window
is always `num_cycles · samples_per_cycle` samples. `num_cycles` may be
fractional (0.5, 1, 2, ...) or `null`, which gives the long-term branch the
whole sequence as context (one padded window per sequence; preset
`experiment=full_context`).

## 4. Commands

```bash
# Print the composed configuration.
conda run -n gymnastic python -m fusion train print_config=true data=gymnastics

# Smoke run (synthetic, CPU, seconds).
conda run -n gymnastic python -m fusion train experiment=smoke

# Private data.
conda run -n gymnastic python -m fusion train data=gymnastics trainer.max_epochs=50

# Precompute cycles + middles (once per dataset), then build the unified tree
# local/runs/cycle_records/{gymnastics,freeman,unity} + index.json + README.md.
conda run -n gymnastic python -m cycle_alignment cycles private
conda run -n gymnastic python -m cycle_alignment cycles freeman
conda run -n gymnastic python -m cycle_alignment cycles unity
conda run -n gymnastic python -m cycle_alignment cycles index

# FreeMan (subject-disjoint) on a subject subset.
conda run -n gymnastic python -m fusion train data=freeman 'data.options.subjects=[1,2,3,4,5,6]'

# Unity direction transfer.
conda run -n gymnastic python -m fusion train data=unity data.options.fold=left_to_right

# Ablations.
conda run -n gymnastic python -m fusion train data=gymnastics experiment=no_film
conda run -n gymnastic python -m fusion train data=gymnastics model.reliability.enabled=false
```

Every run writes `config.yaml`, `logs/` (CSV), `checkpoints/` and
`result.json` below `local/runs/cycle_aware/<run_name>`. `python -m
fusion.train` is equivalent to the CLI.

On a large shared CPU box cap the threads and keep the DataLoader in-process
(`trainer.num_threads=32 data.num_workers=0`); the default of one torch
thread per core plus forked workers oversubscribes the machine. Measured on
HP260146 (96 cores, no GPU): about 0.5 s per batch of 8 windows, i.e. roughly
1.5 min per epoch over the 96 training people.

`ta_mpjpe` (translation-only alignment) is logged only for references that
share the canonical body frame (synthetic data); for the triangulated,
FreeMan and Unity references use `pa_mpjpe`.

## 4.1 Cross-validation protocol

The fixed protocol is **5-fold subject-disjoint cross-validation, single seed
(0), 50 epochs**, run separately per dataset.

* Fold files: `src/configs/fusion/folds/gymnastics/fold_01..05.json` (137
  people, stratified by elderly / student cohort; every person is tested
  exactly once, the validation people of fold *k* are the test people of fold
  *k + 1*) and the existing `src/configs/shared/folds/freeman/fold_01..05.json`.
* `data.fold_json=<file>` selects one fold (it also restricts which subjects
  are loaded); `folds_dir=<dir>` runs every fold sequentially in one process
  and writes `summary.{json,csv}`.
* Cluster: `bash pegasus/submit_cycle_aware_5fold.sh gymnastics [experiment]`
  submits one gpu job per fold (`pegasus/cycle_aware_fold_qsub.sh`); collect
  with `PYTHONPATH=src python -m fusion.summarize
  local/runs/cycle_aware/<sweep>`.
* Unity is evaluation-only (two short single-sweep sequences, no cycles).

## 5. Tests

```bash
conda run -n gymnastic python -m pytest tests/fusion -q
```

Coverage: skeleton variants, sample contract, velocity, phase utilities,
every module (shapes, masking, identities, symmetry), the full model
(forward, backward, ablation switches, swap symmetry), corruptions, losses,
metrics, windowing, the four DataModules (injected loaders), Hydra
composition, a Lightning `fast_dev_run`, and a two-epoch end-to-end run with
checkpoint reload.

## 6. Extending

* New dataset: subclass `DualViewDataModule`, implement `load_samples` (return
  `DualViewSample` objects, e.g. through `sample_from_pose_pair_trial`) and
  `default_split`, add `src/configs/fusion/data/<name>.yaml`, and register the
  name in `data/__init__.build_datamodule`.
* New ablation: add a `@package _global_` preset under
  `src/configs/fusion/experiment/` that sets `model.<block>.enabled`.
* New loss: add a term to `losses.compute_losses` and a weight to
  `LossConfig` / `loss/default.yaml`.
