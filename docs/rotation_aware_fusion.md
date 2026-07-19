# Rotation-Aware Self-Supervised Fusion

## Research Boundary

`rotation_aware_self_supervised` is the paper mainline for rotation-aware,
self-supervised face/side SAM3D-Body 3D-keypoint fusion. It consumes only the
two SAM3D 3D keypoint streams and produces complete 70-joint sequences.

The existing `python -m fuse` command, its nine methods, and
`logs/fuse_experiments` remain deterministic comparison experiments. In
particular, `sim3_face_stable_smooth_kpt` remains the established comparison
baseline. The rotation-aware route never overwrites or trains inside that
directory.

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
  runs/<run_id>/                 training-only artifacts and checkpoints
  inference/<run_id>/person_<id>/cycle_<n>/
                                 per-cycle model outputs
  evaluation/<run_id-or-ids>/    person/joint/diagnostic CSV reports
```

`cache/` holds read-optimized prepared trials and their source/config
metadata. `runs/<run_id>/` holds `config_resolved.yaml`,
`split_manifest.json`, `corruption_manifest.json`, checkpoints,
`train_metrics.csv`, and `run_metadata.json`. The latter records
`no_pseudo_gt_training: true` and the split/config/corruption provenance used
to select checkpoints.

Prepared person caches use immutable internal generations:
`cache/person_<id>/.generations/<generation>/` contains that generation's
cycle NPZ files and manifest, while `cache/person_<id>/manifest.json`
atomically points to the active generation. Readers resolve the pointer once,
then read only that immutable generation. Older generations are retained, so a
reader that has already resolved a path remains valid during a later prepare.
The top-level manifest and the public person-cache API remain compatible with
legacy caches whose cycle NPZ files live directly under `person_<id>/`.

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
training used no pseudo-GT.

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
