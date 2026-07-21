# Rotation-Aware Training Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify the batch-64 rotation-aware training protocol while reducing epoch time through deterministic prefetch, batched validation, ROM synchronization removal, and opt-in stage profiling.

**Architecture:** Keep the existing model, losses, FP32 precision, sample ordering, and optimizer semantics. Add method-specific schedule resolution in the CLI, ordered CPU preparation in a focused prefetch module, stage timing in a focused profiling module, and optimized validation/ROM execution with scalar reference paths retained for equivalence tests.

**Tech Stack:** Python 3.10, PyTorch, `torch.utils.data.DataLoader`, `concurrent.futures`, pytest, YAML, and two NVIDIA RTX 3090 GPUs.

## Global Constraints

- Use `conda run -n gymnastic ...` for project Python commands.
- A4 uses batch 64, 200 epochs, and learning rate 0.001.
- A5 uses batch 64, 200 epochs, and learning rate 0.001.
- A6 uses batch 64, 100 epochs, and learning rate 0.001.
- Keep FP32; do not enable AMP, FP16, or BF16.
- Preserve stable corruption seeds, fold membership, sample order, and one optimizer step per existing training batch.
- Preserve one optimizer step per complete training cycle in A6.
- Preserve all nine loss metrics, validation aggregation, validation score, and `score >= best` checkpoint selection.
- Do not use triangulated pseudo-GT during training or checkpoint selection.
- Do not interrupt or overwrite the active batch-32 all-137-person run.

## File Structure

- Create `fuse/rotation_aware/prefetch.py`: ordered background preparation, tensor pinning, and runtime options.
- Create `fuse/rotation_aware/profiling.py`: opt-in CPU/CUDA stage timing without checkpoint state.
- Create `configs/fuse/rotation_aware_batch64.yaml`: batch-64 method schedules and runtime options.
- Create `analysis/benchmark_rotation_aware_training.py`: warm-epoch GPU throughput benchmark.
- Modify `fuse/rotation_aware/cli.py`: schedule resolution, validation cache lifetime, and profiler output.
- Modify `fuse/rotation_aware/training.py`: prepared forwarding, prefetch integration, and batched validation.
- Modify `fuse/rotation_aware/losses.py`: synchronization-light ROM and zero-weight gradient pruning.
- Modify `tests/rotation_aware/test_cli.py`, `test_training.py`, `test_losses.py`, and `test_end_to_end.py`.

---

### Task 1: Declare And Resolve The Batch-64 Protocol

**Files:**
- Create: `configs/fuse/rotation_aware_batch64.yaml`
- Modify: `fuse/rotation_aware/cli.py`
- Test: `tests/rotation_aware/test_cli.py`

**Interfaces:**
- Produces: `_training_config_for_ablation(config: Mapping[str, Any], ablation: str) -> dict[str, Any]`.
- Consumes: CLI ablation labels `A4`, `A5`, and `A6`.

- [ ] **Step 1: Write the failing schedule test**

```python
def test_training_schedule_resolves_batch64_method_epochs() -> None:
    config = {
        "training": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "epochs_by_ablation": {"A4": 200, "A5": 200, "A6": 100},
        }
    }
    assert _training_config_for_ablation(config, "A4")["epochs"] == 200
    assert _training_config_for_ablation(config, "A5")["epochs"] == 200
    assert _training_config_for_ablation(config, "A6")["epochs"] == 100
    assert _training_config_for_ablation(config, "A6")["batch_size"] == 64
```

Also reject a missing ablation key, non-positive epochs, and unknown schedule labels.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_cli.py -k training_schedule -q
```

Expected: FAIL because `_training_config_for_ablation` is absent.

- [ ] **Step 3: Implement schedule resolution**

```python
def _training_config_for_ablation(
    config: Mapping[str, Any], ablation: str
) -> dict[str, Any]:
    training = dict(config.get("training", {}))
    schedule = training.pop("epochs_by_ablation", None)
    if schedule is not None:
        if not isinstance(schedule, Mapping) or set(schedule) != {"A4", "A5", "A6"}:
            raise ValueError("training.epochs_by_ablation must define exactly A4, A5, and A6")
        training["epochs"] = int(schedule[ablation])
    epochs = int(training.get("epochs", 1))
    if epochs < 1:
        raise ValueError("training epochs must be positive")
    training["epochs"] = epochs
    training["ablation"] = ablation
    return training
```

Call the helper from `_cmd_train` before loader/provenance construction. Create `rotation_aware_batch64.yaml` with:

```yaml
training:
  epochs_by_ablation: {A4: 200, A5: 200, A6: 100}
  batch_size: 64
  learning_rate: 0.001
  hidden_channels: 128
  seed: 0
  device: cuda:0
performance:
  prefetch_batches: 2
  pin_memory: true
  non_blocking_transfer: true
  cache_validation_batches: true
  profile_stages: false
```

- [ ] **Step 4: Run CLI tests and verify GREEN**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_cli.py -q
```

Expected: all tests pass; configs without a schedule retain old behavior.

- [ ] **Step 5: Commit**

```bash
git add configs/fuse/rotation_aware_batch64.yaml fuse/rotation_aware/cli.py tests/rotation_aware/test_cli.py
git commit -m "feat: declare batch 64 rotation-aware schedule"
```

### Task 2: Add Opt-In Stage Profiling

**Files:**
- Create: `fuse/rotation_aware/profiling.py`
- Modify: `fuse/rotation_aware/training.py`
- Test: `tests/rotation_aware/test_training.py`

**Interfaces:**
- Produces: `StageProfiler(enabled: bool, device: torch.device)`, `.stage(name)`, and `.summary()`.
- Consumed by: window train/validation and complete-cycle train/validation loops.

- [ ] **Step 1: Write the failing profiler test**

```python
def test_stage_profiler_collects_cpu_wall_time() -> None:
    profiler = StageProfiler(enabled=True, device=torch.device("cpu"))
    with profiler.stage("corruption"):
        torch.arange(1024).sum()
    summary = profiler.summary()
    assert summary["corruption"]["calls"] == 1
    assert summary["corruption"]["wall_seconds"] >= 0
    json.dumps(summary)
```

Also assert a disabled profiler returns `{}`.

- [ ] **Step 2: Run the test and verify RED**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_training.py -k stage_profiler -q
```

Expected: FAIL because `StageProfiler` is absent.

- [ ] **Step 3: Implement CPU/CUDA deferred timing**

```python
@contextmanager
def stage(self, name: str) -> Iterator[None]:
    if not self.enabled:
        yield
        return
    started = time.perf_counter()
    events = None
    if self.device.type == "cuda":
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        events = (begin, end)
    try:
        yield
    finally:
        if events is not None:
            events[1].record()
            self._events.setdefault(name, []).append(events)
        self._wall.setdefault(name, []).append(time.perf_counter() - started)
```

Synchronize CUDA only in `summary()`. Thread a default-disabled profiler through training and validation without changing existing return mappings.

- [ ] **Step 4: Run training tests and verify GREEN**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_training.py -q
```

- [ ] **Step 5: Commit**

```bash
git add fuse/rotation_aware/profiling.py fuse/rotation_aware/training.py tests/rotation_aware/test_training.py
git commit -m "feat: profile rotation-aware training stages"
```

### Task 3: Remove ROM Synchronization And Zero-Weight Backward Work

**Files:**
- Modify: `fuse/rotation_aware/losses.py`
- Test: `tests/rotation_aware/test_losses.py`

**Interfaces:**
- Produces: `_contiguous_true_runs(mask: Tensor) -> list[tuple[int, int, int]]` and optimized `_rom_loss`.
- Preserves: `LossConfig`, `LossBreakdown`, and public loss function signatures.

- [ ] **Step 1: Write failing ROM value/gradient equivalence tests**

Retain the current implementation as `_rom_loss_reference`. Parameterize no run, one run, separated runs, padding, wrapped angles, and two batch members:

```python
actual = _rom_loss(prediction, target, valid, complete)
expected = _rom_loss_reference(prediction_ref, target, valid, complete)
torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
actual.backward()
expected.backward()
torch.testing.assert_close(prediction.grad, prediction_ref.grad, rtol=1e-6, atol=1e-6)
```

Monkeypatch `Tensor.__bool__` to raise during optimized multi-frame ROM execution.

- [ ] **Step 2: Write a failing zero-weight gradient test**

Compare totals, all reported components, and fused-keypoint gradients with rotation, temporal, and ROM weights zero. Use `1e-7` tolerance.

- [ ] **Step 3: Run focused tests and verify RED**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_losses.py -k "rom or zero_weight" -q
```

- [ ] **Step 4: Implement one-sync run detection and vectorized unwrapping**

```python
def _unwrap_valid_run(values: Tensor) -> Tensor:
    if values.shape[0] <= 1:
        return values
    steps = circular_diff(values[1:], values[:-1])
    return torch.cat((values[:1], values[:1] + torch.cumsum(steps, dim=0)))

def _contiguous_true_runs(mask: Tensor) -> list[tuple[int, int, int]]:
    padded = torch.nn.functional.pad(mask.bool(), (1, 1), value=False)
    changes = (padded[:, 1:] ^ padded[:, :-1]).nonzero(as_tuple=False).cpu()
    boundaries: dict[int, list[int]] = {}
    for batch_index, frame_index in changes.tolist():
        boundaries.setdefault(batch_index, []).append(frame_index)
    return [
        (batch_index, points[offset], points[offset + 1])
        for batch_index, points in boundaries.items()
        for offset in range(0, len(points), 2)
    ]
```

Use the run list in `_rom_loss`, preserving run order and the same ROM formula.

- [ ] **Step 5: Exclude exact-zero weights from `total` autograd**

```python
total = fused.new_zeros(())
for name, value in values.items():
    weight = config.weights[name]
    if weight > 0:
        total = total + weight * value
```

Continue calculating and reporting all components.

- [ ] **Step 6: Run all loss tests and verify GREEN**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_losses.py -q
```

- [ ] **Step 7: Commit**

```bash
git add fuse/rotation_aware/losses.py tests/rotation_aware/test_losses.py
git commit -m "perf: remove complete-cycle ROM synchronizations"
```

### Task 4: Add Ordered Corruption Prefetch And Async Transfer

**Files:**
- Create: `fuse/rotation_aware/prefetch.py`
- Modify: `fuse/rotation_aware/training.py`
- Test: `tests/rotation_aware/test_training.py`

**Interfaces:**
- Produces: `ThroughputConfig`, `ordered_prefetch(source, prepare, depth)`, `pin_tensor_batch(batch)`, `_prepare_window`, and `_forward_prepared`.
- Preserves: `_forward_window` as direct composition for reference tests.

- [ ] **Step 1: Write failing order, exception, and exact-corruption tests**

```python
def test_ordered_prefetch_preserves_source_order() -> None:
    values = list(ordered_prefetch(range(8), lambda value: value * 2, depth=3))
    assert values == [value * 2 for value in range(8)]
```

Add an exception propagation test and tensor-exact direct/prefetched corruption comparisons over two epochs.

- [ ] **Step 2: Write failing pinned-transfer tests**

Assert pinning does not mutate inputs, keeps metadata unchanged, and preserves tensor values. On CUDA hosts, assert CPU tensors are pinned.

- [ ] **Step 3: Run tests and verify RED**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_training.py -k "prefetch or pinned or non_blocking" -q
```

- [ ] **Step 4: Implement bounded ordered prefetch**

Use one `ThreadPoolExecutor` and a deque of at most `depth` futures. Consume from the left, submit one replacement, and always close the executor in `finally`.

```python
@dataclass(frozen=True)
class ThroughputConfig:
    prefetch_batches: int = 0
    pin_memory: bool = False
    non_blocking_transfer: bool = False
    cache_validation_batches: bool = False
    profile_stages: bool = False
```

- [ ] **Step 5: Split CPU preparation from GPU forwarding**

```python
def _forward_window(...):
    prepared = _prepare_window(batch, seed=seed, skeleton=skeleton,
                               corruption_config=corruption_config, epoch=epoch)
    return _forward_prepared(model, prepared, skeleton, device=device)
```

Add `non_blocking` to `_tensor_batch` and use `Tensor.to(device, non_blocking=non_blocking)`.

- [ ] **Step 6: Integrate prefetch without changing optimizer order**

The preparation closure captures the exact epoch and seed. Optimizer stepping stays on the consuming thread in source order. Complete-cycle training remains batch one with one update per cycle.

- [ ] **Step 7: Run training tests and verify GREEN**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_training.py -q
```

- [ ] **Step 8: Commit**

```bash
git add fuse/rotation_aware/prefetch.py fuse/rotation_aware/training.py tests/rotation_aware/test_training.py
git commit -m "perf: prefetch deterministic training batches"
```

### Task 5: Batch Validation Forward And Cache Fixed Corruptions

**Files:**
- Modify: `fuse/rotation_aware/training.py`
- Modify: `fuse/rotation_aware/cli.py`
- Test: `tests/rotation_aware/test_training.py`
- Test: `tests/rotation_aware/test_cli.py`

**Interfaces:**
- Produces: `_single_output`, `prepare_validation_batches`, and optional `prepared_loader`/`scalar_forward` validation arguments.
- Consumes: `ThroughputConfig`, `_prepare_window`, `_forward_prepared`, and `StageProfiler`.

- [ ] **Step 1: Write failing batched-versus-scalar validation tests**

```python
optimized = validate(model, loader4, spec, seed=17, scalar_forward=False)
reference = validate(model, loader4, spec, seed=17, scalar_forward=True)
for name in optimized["losses"]:
    assert optimized["losses"][name] == pytest.approx(
        reference["losses"][name], rel=1e-6, abs=1e-6
    )
assert optimized["score"] == pytest.approx(reference["score"], rel=1e-7, abs=1e-7)
```

Compare all score components and checkpoint decisions.

- [ ] **Step 2: Write failing validation-cache tests**

Assert repeated materialization is tensor-exact, metadata is stable, source batches are not mutated, and two validations reusing the cache return equal results.

- [ ] **Step 3: Run focused tests and verify RED**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_training.py -k "validation and (batch or cache or scalar)" -q
```

- [ ] **Step 4: Implement one forward per validation batch**

```python
output, prepared = _forward_prepared(model, prepared_batch, skeleton, device=target_device)
for sample_index in range(output.fused_kpts.shape[0]):
    sample_output = _single_output(output, sample_index)
    sample_prepared = _single_sample(prepared, sample_index)
    losses = compute_self_supervised_losses(sample_output, sample_prepared, window_config, skeleton)
```

Retain the current scalar loop behind `scalar_forward=True` until equivalence and GPU benchmarks pass.

- [ ] **Step 5: Cache fixed epoch-zero validation corruptions**

Prepare validation-window and validation-cycle batches once in `_cmd_train` when configured. Reuse prepared input tensors across epochs; never cache model outputs or model-dependent features.

- [ ] **Step 6: Connect performance configuration and stage output**

Parse the top-level `performance` mapping into `ThroughputConfig`. Write stage summaries to `runs/<run_id>/stage_profile.jsonl` only when profiling is enabled; exclude timing from checkpoints and scores.

- [ ] **Step 7: Run CLI and training tests and verify GREEN**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_training.py tests/rotation_aware/test_cli.py -q
```

- [ ] **Step 8: Commit**

```bash
git add fuse/rotation_aware/training.py fuse/rotation_aware/cli.py tests/rotation_aware/test_training.py tests/rotation_aware/test_cli.py
git commit -m "perf: batch deterministic validation forwards"
```

### Task 6: Integration Verification And GPU Benchmark

**Files:**
- Create: `analysis/benchmark_rotation_aware_training.py`
- Modify: `docs/rotation_aware_fusion.md`
- Test: `tests/rotation_aware/test_end_to_end.py`

**Interfaces:**
- Produces: benchmark JSON with config, device, warmups, epoch timings, stage timings, samples/second, and peak CUDA memory.
- Consumes: batch-64 config, prepared cache, scalar reference path, and optimized path.

- [ ] **Step 1: Add a failing batch-64 integration test**

Use the tiny cache fixture and assert A4 resolves to 200 epochs, A6 to 100, batch size to 64, a one-epoch override has the expected optimizer-step count, and checkpoint provenance records resolved settings.

- [ ] **Step 2: Run the integration test and verify RED**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_end_to_end.py -k batch64 -q
```

- [ ] **Step 3: Implement the benchmark script**

Accept `--config`, `--ablation`, `--device`, `--warmup-epochs`, `--measured-epochs`, and `--output`. Reject fewer than two measured epochs, synchronize CUDA around measurements, and report median time rather than the fastest value.

```bash
conda run --no-capture-output -n gymnastic python analysis/benchmark_rotation_aware_training.py \
  --config configs/fuse/rotation_aware_batch64.yaml \
  --ablation A6 --device cuda:0 --warmup-epochs 1 --measured-epochs 3 \
  --output /tmp/rotation_aware_a6_batch64_benchmark.json
```

- [ ] **Step 4: Document the new protocol**

Document A4/A5 200 epochs and A6 100 epochs. State that batch-64 runs are a new protocol and use epoch time/samples-per-second, not utilization alone, as acceptance metrics.

- [ ] **Step 5: Run the full suite**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware -q
```

- [ ] **Step 6: Run static checks**

```bash
conda run -n gymnastic python -m py_compile fuse/rotation_aware/cli.py fuse/rotation_aware/training.py fuse/rotation_aware/losses.py fuse/rotation_aware/prefetch.py fuse/rotation_aware/profiling.py analysis/benchmark_rotation_aware_training.py
git diff --check
```

- [ ] **Step 7: Run GPU benchmarks after the active run releases a GPU**

Accept the optimized path only when equivalence assertions pass, checkpoint selection matches the batch-64 scalar reference, median epoch time improves, peak memory stays below 22 GiB, and every loss/timing is finite.

- [ ] **Step 8: Commit**

```bash
git add analysis/benchmark_rotation_aware_training.py docs/rotation_aware_fusion.md tests/rotation_aware/test_end_to_end.py
git commit -m "test: benchmark batch 64 rotation-aware training"
```

### Task 7: Final Review

**Files:**
- Review: all files changed by Tasks 1-6.

**Interfaces:**
- Produces: verified code ready for a new isolated batch-64 run.

- [ ] **Step 1: Inspect the diff for protocol drift**

Confirm no AMP, loss-weight, fold, pseudo-GT, or complete-cycle batching change entered the implementation.

- [ ] **Step 2: Confirm active run isolation**

Verify no implementation task wrote to existing `all137_*_e100_seed0` configs, logs, runs, or checkpoints.

- [ ] **Step 3: Record evidence**

Report test count, A4/A6 equivalence deltas, median old/new epoch time, samples/second, peak memory, and remaining limits.

- [ ] **Step 4: Commit review corrections only when needed**

Stage only implementation files. Do not stage `.superpowers/` or `paper/`.

### Implementation Note

The batched-validation task was rejected after a production-shape RTX 3090
probe violated the `1e-6` equivalence gate. Production CLI and benchmark runs
use scalar validation. The remaining throughput tasks stay valid and batched
validation requires a future fixed-reduction CUDA implementation plus a
measured end-to-end speedup before activation.
