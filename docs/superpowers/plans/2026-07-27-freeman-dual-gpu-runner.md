# FreeMan Dual-GPU Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable `gymnastics benchmark freeman run --devices 0 1`
coordinator that processes disjoint subjects concurrently on both GPUs and
publishes one canonical state and report.

**Architecture:** The existing parent run path retains sole ownership of shared
dataset preparation and aggregate reporting. For two or more requested devices,
it partitions outstanding subjects round-robin, starts one spawned process per
device, gives each process a private state file and a single visible physical
GPU, then merges terminal subject states before reporting.

**Tech Stack:** Python 3.11, `argparse`, `multiprocessing` spawn context,
atomic JSON files, pytest, tmux, CUDA/SAM3D-Body.

## Global Constraints

- Keep the existing single-device run path unchanged when `--devices` is absent.
- GPU workers share only subject-scoped outputs; workers never write the
  canonical `run_state.json` or aggregate report.
- Set each worker's `CUDA_VISIBLE_DEVICES` to one physical device and use SAM3D
  logical device `0`.
- Continue the healthy worker if its peer fails; preserve all successful
  artifacts and report failure only after every worker exits.
- Reuse identity-valid caches and never stage unrelated user changes.

---

### Task 1: CLI Contract And Deterministic Partitioning

**Files:**
- Modify: `src/gymnastics/benchmarks/freeman/cli.py`
- Test: `tests/freeman_benchmark/test_cli.py`

**Interfaces:**
- Produces: `partition_subjects(subjects: Sequence[int], worker_count: int) -> tuple[tuple[int, ...], ...]`
- Produces: the `run --devices DEVICE [DEVICE ...]` CLI option.
- Changes: `StageOperations.run(..., devices: Sequence[int] | None = None)`.

- [ ] **Step 1: Write failing CLI and partition tests**

```python
def test_round_robin_partition_is_disjoint_and_complete():
    assert partition_subjects([1, 2, 3, 4, 5], 2) == ((1, 3, 5), (2, 4))

def test_run_cli_forwards_validated_devices():
    operations = RecordingOperations()
    main(["run", "--devices", "0", "1"], operations=operations)
    assert operations.devices == (0, 1)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/freeman_benchmark/test_cli.py::test_round_robin_partition_is_disjoint_and_complete \
  tests/freeman_benchmark/test_cli.py::test_run_cli_forwards_validated_devices -q
```

Expected: failure because the helper and CLI option do not exist.

- [ ] **Step 3: Implement validation and partitioning**

Add the helper, add `--devices` only to `run`, reject negative or duplicate
device IDs, and pass the immutable device tuple into `StageOperations.run`.

- [ ] **Step 4: Run focused tests**

Run the command from Step 2. Expected: both tests pass.

### Task 2: Isolated GPU Workers And State Merge

**Files:**
- Modify: `src/gymnastics/benchmarks/freeman/cli.py`
- Test: `tests/freeman_benchmark/test_cli.py`

**Interfaces:**
- Produces: `_worker_state_path(output_root: Path, device: int) -> Path`.
- Produces: `_run_device_worker(config: Mapping[str, Any], device: int, subjects: Sequence[int], state_path: Path, keep_workspace: bool) -> None`.
- Produces: `_merge_worker_states(canonical_path: Path, worker_paths: Sequence[Path]) -> dict[str, Any]`.
- Consumes: `partition_subjects` from Task 1 and existing `run_subjects`.

- [ ] **Step 1: Write failing state and environment tests**

```python
def test_worker_uses_private_state_and_logical_device_zero(monkeypatch, tmp_path):
    observed = {}
    monkeypatch.setattr(freeman_cli, "run_subjects", capture_run(observed))
    _run_device_worker(config(tmp_path), 1, [2, 4], tmp_path / "worker.json", False)
    assert observed["cuda_visible_devices"] == "1"
    assert observed["sam3d_device"] == 0
    assert observed["subjects"] == [2, 4]

def test_merge_preserves_disjoint_worker_terminal_states(tmp_path):
    merged = _merge_worker_states(canonical, [worker0, worker1])
    assert merged["subjects"]["1"]["status"] == "complete"
    assert merged["subjects"]["2"]["status"] == "failed"
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
conda run -n gymnastic python -m pytest tests/freeman_benchmark/test_cli.py \
  -k "worker_uses_private or merge_preserves" -q
```

Expected: failure because worker and merge helpers do not exist.

- [ ] **Step 3: Implement worker isolation and atomic merge**

The worker copies the config, sets `os.environ["CUDA_VISIBLE_DEVICES"]`, sets
`config["sam3d"]["device"] = 0`, creates a new `DefaultStageOperations`, and
calls `run_subjects` with its private state path. The merge reloads canonical
state, copies only each worker's assigned subject terminal records, and writes
once through `_atomic_json`.

- [ ] **Step 4: Run focused tests**

Run the command from Step 2. Expected: all selected tests pass.

### Task 3: Parent Coordinator, Failure Semantics, And Resume

**Files:**
- Modify: `src/gymnastics/benchmarks/freeman/cli.py`
- Test: `tests/freeman_benchmark/test_cli.py`

**Interfaces:**
- Consumes: Task 1 partitioning and Task 2 worker/merge helpers.
- Produces: `_run_parallel_subjects(config, canonical_state_path, devices, keep_workspace) -> None`.
- Changes: `DefaultStageOperations.run` launches workers after shared
  preparation and invokes `report` only when every worker succeeds.

- [ ] **Step 1: Write failing coordinator tests**

Use a fake spawn context whose processes record target arguments and exit codes.
Assert that:

```python
assignments == {0: (1, 3), 1: (2, 4)}
```

Assert completed canonical subjects are excluded before partitioning. Simulate
one worker failing and verify the peer is joined, both state files are merged,
and `report` is not called.

- [ ] **Step 2: Run coordinator tests and verify failure**

Run:

```bash
conda run -n gymnastic python -m pytest tests/freeman_benchmark/test_cli.py \
  -k "parallel or completed_subjects" -q
```

Expected: failure because the coordinator is not implemented.

- [ ] **Step 3: Implement parent coordination**

Use `multiprocessing.get_context("spawn")`. Seed each worker state from the
current canonical records for its assignment, start all non-empty assignments,
join every process, merge worker states, and raise a `RuntimeError` listing
failed devices after all joins. Skip aggregate report in the existing `run`
method when this exception is raised.

- [ ] **Step 4: Run coordinator and legacy CLI tests**

Run:

```bash
conda run -n gymnastic python -m pytest tests/freeman_benchmark/test_cli.py -q
```

Expected: all CLI tests pass, including the unchanged sequential path.

### Task 4: Full Verification, Publication, And Live Cutover

**Files:**
- Verify: `src/gymnastics/benchmarks/freeman/cli.py`
- Verify: `tests/freeman_benchmark/test_cli.py`

**Interfaces:**
- Consumes the completed dual-GPU CLI.
- Produces a pushed master commit and a live two-worker benchmark.

- [ ] **Step 1: Run the full FreeMan suite**

```bash
conda run -n gymnastic python -m pytest tests/freeman_benchmark -q
```

Expected: all tests pass.

- [ ] **Step 2: Review scoped diff and commit**

Stage only the CLI, its tests, and this implementation plan. Do not stage
existing paper or result-summary edits.

- [ ] **Step 3: Push master**

```bash
git push origin master
```

- [ ] **Step 4: Cut over the background run**

Stop only the current `freeman_benchmark` tmux session, then launch:

```bash
tmux new-session -d -s freeman_benchmark \
  "cd /home/workspace/kaixu/code/Gymnastics_PyTorch && \
   exec conda run -n sam_3d_body gymnastics benchmark freeman run \
   --devices 0 1 >> local/runs/freeman_benchmark/runner_dual_gpu.log 2>&1"
```

- [ ] **Step 5: Verify the live run**

Confirm one worker process per device, compute memory on both physical GPUs,
disjoint worker subject states, canonical subject 1 remains resumable, and no
new traceback appears in the dual-GPU log.
