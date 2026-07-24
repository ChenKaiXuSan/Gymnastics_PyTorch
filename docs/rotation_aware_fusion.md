# Rotation-Aware Self-Supervised Fusion

## Research Boundary

`rotation_aware_self_supervised` is the paper mainline for rotation-aware,
self-supervised face/side SAM3D-Body 3D-keypoint fusion. It consumes only the
two SAM3D 3D keypoint streams and produces complete 70-joint sequences.

The existing `python -m fuse` command, its nine methods, and
`logs/fuse_experiments` remain deterministic comparison experiments. The
established comparison baseline is `avg_body_current`: on the regenerated
triangulated pseudo-ground-truth it is the best leakage-free deterministic
method (mean person MPJPE 64.05 mm, better than every other leakage-free method
on 69-100% of the 137 people). It replaces the earlier baseline
`sim3_face_stable_smooth_kpt`, which the regenerated GT ranks sixth; runs
recorded against that older baseline are not directly comparable to ones scored
against `avg_body_current`. Note that `sim3_face_stable_joint_weight` scores
marginally lower still but derives its per-joint weights from the triangulated
GT it is then evaluated against, so it is excluded as a biased baseline. The
rotation-aware route never overwrites or trains inside that directory.

Time alignment is strict: the adapter reads only
`logs/split_cycle/person_<id>/alignment_record_<id>.json` and its
`offset_side_to_face`. A missing record or offset is an error; no keypoint
DTW or other fallback alignment is performed.

Training, synthetic corruption targets, validation score, and checkpoint
selection use no triangulated data. Triangulated pseudo-GT is imported only by
`fuse/rotation_aware/evaluation.py` for the external post-training
evaluation layer. It is not an input, pseudo-target, fusion weight, or model
selection criterion.

## Commands

All commands use the `gymnastic` environment. The default config is
`configs/fuse/rotation_aware.yaml`; pass `--config` to use a copied,
machine-specific YAML.

```bash
conda run -n gymnastic python -m fuse.rotation_aware prepare --config configs/fuse/rotation_aware.yaml
conda run -n gymnastic python -m fuse.rotation_aware train --config configs/fuse/rotation_aware.yaml --run-id paper_a6 --ablation A6
conda run -n gymnastic python -m fuse.rotation_aware infer --config configs/fuse/rotation_aware.yaml --run-id paper_a6
conda run -n gymnastic python -m fuse.rotation_aware evaluate --config configs/fuse/rotation_aware.yaml --run-id paper_a6
```

`prepare` may be scoped with `--person <id>`; it builds compact cache files
from split-cycle trials. `train` and `infer` require an explicit
`--run-id`. Each command also accepts `--person`, `--fold`, and
`--output-root` for a scoped reproducible run. `infer` accepts
`--checkpoint` to use a checkpoint outside the default run location.

## Batch-64 Throughput Protocol

`configs/fuse/rotation_aware_batch64.yaml` defines a new FP32 batch-64
protocol. It is not directly comparable to historical batch-32 runs: A4 and
A5 train for 200 epochs, while A6 trains for 100 epochs. All three use a
learning rate of `0.001` and retain the existing sample ordering, loss
definitions, optimizer semantics, and validation/checkpoint-selection rules.
Its training, inference, and evaluation artifacts are isolated below
`logs/fuse_rotation_aware/batch64`, while it reads the prepared cache at
`logs/fuse_rotation_aware/cache`.

Batch-64 run IDs must include their ablation, batch size, and resolved epoch
count. Use a fresh ID for every training attempt; the command rejects a
non-empty batch-64 run directory rather than overwriting it.

```bash
conda run -n gymnastic python -m fuse.rotation_aware prepare --config configs/fuse/rotation_aware_batch64.yaml
conda run -n gymnastic python -m fuse.rotation_aware train --config configs/fuse/rotation_aware_batch64.yaml --run-id paper_a4_b64_e200 --ablation A4
conda run -n gymnastic python -m fuse.rotation_aware train --config configs/fuse/rotation_aware_batch64.yaml --run-id paper_a5_b64_e200 --ablation A5
conda run -n gymnastic python -m fuse.rotation_aware train --config configs/fuse/rotation_aware_batch64.yaml --run-id paper_a6_b64_e100 --ablation A6
conda run -n gymnastic python -m fuse.rotation_aware infer --config configs/fuse/rotation_aware_batch64.yaml --run-id paper_a4_b64_e200
conda run -n gymnastic python -m fuse.rotation_aware infer --config configs/fuse/rotation_aware_batch64.yaml --run-id paper_a5_b64_e200
conda run -n gymnastic python -m fuse.rotation_aware infer --config configs/fuse/rotation_aware_batch64.yaml --run-id paper_a6_b64_e100
conda run -n gymnastic python -m fuse.rotation_aware evaluate --config configs/fuse/rotation_aware_batch64.yaml --run-id paper_a4_b64_e200 --run-id paper_a5_b64_e200 --run-id paper_a6_b64_e100
```

Benchmark an already prepared cache without writing a training run:

```bash
conda run --no-capture-output -n gymnastic python analysis/benchmark_rotation_aware_training.py \
  --config configs/fuse/rotation_aware_batch64.yaml \
  --ablation A6 --device cuda:0 --warmup-epochs 1 --measured-epochs 3 \
  --output /tmp/rotation_aware_a6_batch64_benchmark.json
```

The JSON report records the resolved configuration and device, warmup count,
source and effective training-device settings, per-epoch timings, median epoch
time, the end-to-end effective train-window rate, workload counts for train
windows/cycles and validation windows/cycles, measured losses, and peak CUDA
allocation. The effective train-window rate is train windows divided by the
complete epoch wall time, which includes the configured complete-cycle and
validation work; it is not raw per-loader throughput. It requires at least two
measured epochs and synchronizes CUDA at each timing boundary.

Before constructing the timed workload, the benchmark builds two independent
workloads from the same resolved ablation config and seed. Their initial model
parameters and Adam states must be identical. One executes a true synchronous
one-epoch reference with no ordered prefetch, pinning, nonblocking transfer,
or validation cache; the other executes the optimized training and input-cache
path. Both use the original scalar validation model forward. They remain FP32
and use the same preparation, forward, losses, loader seeds, batch shape, and
update protocol.

The `training_equivalence` gate records exact phase/window order, per-sample
SHA-256 digests for every corruption tensor before pinning or transfer, and
optimizer-step counts before each batch. A4 must perform one step per fixed
window batch. A6 additionally records every complete cycle in loader order and
must perform one step per cycle. Train metrics and all validation losses use
`1e-6` relative/absolute tolerance. Model parameters and Adam tensor state use
`1e-6` relative and `1e-7` absolute tolerance; non-floating state is exact.
Validation membership is exact, validation score uses `1e-7`, and both paths
must make the same `score >= best_score` checkpoint decision. Resolved
ablation, batch size, epoch count, seed, learning rate, hidden width, loss and
corruption settings, device, precision, and workload counts are included as
protocol/provenance evidence. Any failed exact, finite, metric, state, or
checkpoint gate aborts the benchmark before a report can be accepted.

Warmup and measured epochs always disable stage profiling so timing and peak
memory acceptance exclude profiling instrumentation. The report separately
records the configured `performance.profile_stages` value and captures stage
timings in one labeled, untimed diagnostic profiled epoch after the timing and
peak-memory windows.

After every warmup and measured trained epoch, the benchmark also validates the
same model state through uncached-reference and optimized-input paths. Both use
the retained scalar model forward. It records absolute and relative deltas
for every loss at `1e-6` relative/absolute tolerance and every score component
and score at `1e-7`. Each path independently replays the training command's
evolving checkpoint rule, `score >= best_score`, starting from no checkpoint
(`best = -inf`) and selecting the latest epoch on ties. The JSON includes every
prior-best, decision, and next-best value, both final selected epochs, and
their agreement. The benchmark fails on any per-epoch equivalence or decision
disagreement, or when the final selected checkpoint epoch differs.

With an independent validation split, the optimized-input path consumes the
materialized validation cache while the reference recomputes the same inputs.
For a train-only fallback the production command intentionally disables that
cache to preserve shared-generator order; the report labels cache equivalence
as not applicable instead of presenting the two uncached paths as a cache test.

The one-epoch training probe runs before warmup and before the measured workload
is created. Its models, optimizers, traces, and validation caches are released;
CUDA is synchronized, its allocator cache is emptied, and peak statistics are
reset before benchmark setup continues. Scalar/history diagnostics run after
each epoch's timing boundary. CUDA peak allocation is reset and captured per
measured epoch before the diagnostic work, so reported measured peak memory
excludes the equivalence probe, scalar/history checks, and the optional untimed
profiler epoch.

Treat median epoch time and samples per second as the throughput acceptance
metrics, not GPU utilization alone. Before accepting an optimized batch-64
path, require finite losses and timings, confirm an improved median epoch time,
and keep peak CUDA memory below 22 GiB.

### Batched Validation Status

Batched validation is intentionally disabled. A production-shape RTX 3090
probe at `B=64`, `J=70`, and `C=128` found scalar-versus-batched residual
coordinate differences of `1.42e-6` to `2.62e-6` and acceleration-loss
differences of `0.015625` to `0.0390625`. This fails the locked `1e-6`
validation-equivalence requirement. Fixed-order matrix multiplications also
did not isolate all shape-dependent FP32 reductions and were slower than stock
`Conv1d` in the CPU probe.

The production training command therefore keeps scalar validation and the
original checkpoint semantics. The accepted throughput changes are the
batch-64 schedule, ordered CPU prefetch, pinned/nonblocking transfer, cached
validation inputs, complete-cycle ROM synchronization removal, zero-weight
gradient pruning, and opt-in stage profiling. A future batched validation path
must pass the production-shape CUDA equivalence and end-to-end epoch-time gates
before it can be enabled.

## Ablations And Unified Evaluation

The evaluation registry uses these labels:

| Label | Method |
| --- | --- |
| A0 | face-only output |
| A1 | side-only output |
| A2 | canonical arithmetic mean |
| A3 | deterministic quality-weighted mean |
| A4 | learned spatial objectives |
| A5 | A4 plus rotation and adaptive temporal objectives |
| A6 | `rotation_aware_self_supervised`, including complete-cycle ROM preservation |

A0-A3 are emitted alongside every learned inference run. A4, A5, and A6 are
separate trained runs, so evaluate them together by repeating `--run-id`:

```bash
conda run -n gymnastic python -m fuse.rotation_aware evaluate \
  --config configs/fuse/rotation_aware.yaml \
  --run-id paper_a4 --run-id paper_a5 --run-id paper_a6
```

This creates one combined evaluation directory named
`evaluation/paper_a4+paper_a5+paper_a6/`. The evaluator can also read old
comparison outputs from the configured `old_fuse_root` without writing to
them.

## Output Layout

```text
logs/fuse_rotation_aware/
  cache/                         compact split-cycle trial inputs
  batch64/
    runs/<run_id>/               training-only artifacts and checkpoints
    inference/<run_id>/person_<id>/cycle_<n>/
    evaluation/<run_id-or-ids>/  person/joint/diagnostic CSV reports
  runs/<run_id>/                 legacy/default training artifacts
  inference/<run_id>/person_<id>/cycle_<n>/
                                 per-cycle model outputs
  evaluation/<run_id-or-ids>/    person/joint/diagnostic CSV reports
```

`cache/` holds read-optimized prepared trials and their source/config
metadata. `runs/<run_id>/` holds `config_resolved.yaml`,
`split_manifest.json`, `corruption_manifest.json`, checkpoints,
`train_metrics.csv`, and `run_metadata.json`. The latter records
`no_pseudo_gt_training: true` and the split/config/corruption provenance used
to select checkpoints. Checkpoint and run provenance also bind every selected
person to the exact consumed cache layout, generation, source/config hashes,
trial list, and manifest hash.

Prepared person caches use immutable internal generations:
`cache/person_<id>/.generations/<generation>/` contains that generation's
cycle NPZ files and manifest, while `cache/person_<id>/manifest.json`
atomically points to the active generation. Readers resolve the pointer once,
then read only that immutable generation. Older generations are retained, so a
reader that has already resolved a path remains valid during a later prepare.
The top-level manifest and the public person-cache API remain compatible with
legacy caches whose cycle NPZ files live directly under `person_<id>/`.

On Linux, `person_<id>/.publishing.lock` is a permanent POSIX `flock` guard,
not a publication marker: a writer holds an exclusive lock across staging,
generation publication, and pointer replacement. A first-time reader with no
pointer takes a shared lock and rechecks the pointer before declaring the cache
missing; it retries only while an exclusive writer is active. Readers with an
existing pointer immediately use its immutable generation, even while a newer
prepare is publishing.

Each inference cycle writes `fused_sequence.npz`, `config.json`, and
`metadata.json`. Important NPZ fields are:

- `kpts_world [T,70,3]`: face-reference compatibility output; metadata says
  `coordinate_system=face_reference_uncalibrated`.
- `kpts_body [T,70,3]`: existing unscaled body-frame convention.
- `kpts_fused_canonical [T,70,3]` and `kpts_base_canonical [T,70,3]`:
  trial-scale canonical outputs.
- `theta_fused_rad [T]`, `omega_fused_rad_s [T]`,
  `quality_face [T]`, `quality_side [T]`, and `frame_valid [T]`.
- `joint_valid [T,70]`, `face_map [T]`, `side_map [T]`, timestamps, and
  the face/side/arithmetic/base compatibility sequences.

Inference metadata includes the person/trial/run identifiers, checkpoint path
and SHA-256, split/config/corruption-manifest hashes, model configuration,
ablation, seed, and nested checkpoint provenance. It also records that
training used no pseudo-GT. Inference accepts a subset of checkpoint people,
but rejects people from a different split and rejects caches republished after
training; each cycle records its verified `consumed_cache_manifest` identity.

`evaluation/<run_id>/` contains `metrics_by_person.csv`,
`metrics_by_joint.csv`, `corruption_metrics.csv`,
`rotation_metrics.csv`, and `report.json`. Final rows are aggregated by
person; cycles are only trial/window units.

The diagnostic availability columns distinguish a measured value from why it
cannot exist. `unsupported_deterministic_baseline` means recovery is not
defined for A0-A3, `unsupported_legacy_output` means an old comparison NPZ
did not emit the diagnostic, and `unavailable_*` identifies missing manifest
windows, common valid points, or a reference. These are not zero-valued
measurements.

## External Evaluation

When `paths.triangulated_root` is configured, `evaluate` joins
triangulated cycles through `face_map` and `side_map`, then reports
root-normalized MPJPE, median, P95, and joint-level errors externally. It also
reports no-GT structural and motion metrics such as bone CV, rigidity, jerk,
trunk angular jerk, ROM/peak-velocity retention, swap error, and fixed
corruption recovery. Interpret smoothness together with ROM and peak-velocity
retention rather than treating any single metric as the conclusion.
