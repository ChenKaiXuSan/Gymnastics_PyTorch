# FreeMan Dual-GPU Runner Design

## Goal

Run the full FreeMan benchmark concurrently on both local GPUs without
duplicating subjects, racing on shared state, corrupting subject artifacts, or
changing the single-GPU workflow.

The user-facing command will be:

```bash
gymnastics benchmark freeman run --devices 0 1
```

## Scheduling

The parent process performs the shared inspect, download, and annotation
preparation stages exactly once. It then partitions the sorted requested
subjects round-robin across the requested devices:

```text
GPU 0: subjects 1, 3, 5, ..., 39
GPU 1: subjects 2, 4, 6, ..., 40
```

Round-robin partitioning is preferred to contiguous halves because it is less
sensitive to systematic differences in subject archive or session size. A
resumed run removes already-complete subjects before partitioning so both GPUs
receive only outstanding work.

Each GPU worker is a separate process. Its environment exposes only its
assigned physical device through `CUDA_VISIBLE_DEVICES`, while the SAM3D config
uses logical device `0` inside that process. This avoids the multi-GPU device
enumeration failure already observed in the SAM3D environment.

## Output And State Isolation

All benchmark artifacts remain under the canonical configured output root.
Subject artifacts are already isolated by `subject_XX`, so workers may safely
write different subjects concurrently:

```text
sam3d/subject_XX/
manifests/subject_XX_sessions.json
fusion/methods/*/subject_XX/
evaluation/session_metrics/subject_XX.json
```

Workers must not write the shared `run_state.json` or generate aggregate
reports. Each worker writes a dedicated state file:

```text
workers/device_0/run_state.json
workers/device_1/run_state.json
```

After both workers exit, the parent merges their per-subject terminal states
into the canonical `run_state.json` using the existing atomic JSON writer.
Only the parent writes shared stage state and aggregate reports.

The subject workspaces under the dataset work root are also subject-scoped.
Because the partition is disjoint, extraction and cleanup cannot target the
same subject concurrently. Shared annotations are prepared before worker
launch and are read-only while workers run.

## Failure And Resume Behavior

If one worker fails, the other worker is allowed to continue its assigned
subjects. The parent waits for both workers, merges all completed and failed
subject states, and exits with an error after preserving the successful work.
It does not publish a final aggregate report unless every requested subject is
complete.

On resume:

1. Existing identity-valid SAM3D caches are reused.
2. Canonically complete subjects are skipped.
3. Failed or interrupted subjects are repartitioned across the available
   devices.
4. Stale worker state describes history only and cannot override a newer
   canonical complete state.

The current single-device invocation remains supported and follows the
existing sequential code path.

## CLI And Components

The `run` subcommand gains:

```text
--devices DEVICE [DEVICE ...]
```

The implementation is divided into small units:

- a deterministic subject partition helper;
- a worker entry point that processes only its assigned subjects;
- per-worker state paths and atomic state publication;
- a parent coordinator that prepares shared data, launches workers, merges
  state, and generates the report once;
- existing single-subject processing reused unchanged.

No training behavior, benchmark metric, camera pairing, fusion method, cache
format, or dataset layout changes.

## Verification

Tests will cover:

- deterministic odd/even round-robin partitioning;
- exclusion of already-complete subjects on resume;
- distinct worker state paths and disjoint subject assignments;
- physical-to-logical GPU environment mapping;
- successful state merge and single report publication;
- one-worker failure while the other worker completes;
- preservation of the existing single-device run path.

The full `tests/freeman_benchmark` suite must pass before the dual-GPU run is
started. A live smoke check must then confirm two worker processes, one compute
process on each physical GPU, disjoint current subjects, and intact canonical
state.
