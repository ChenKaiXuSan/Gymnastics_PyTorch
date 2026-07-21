# Rotation-Aware Training Throughput Design

## Goal

Reduce end-to-end training time for rotation-aware A4, A5, and A6 while adopting
an explicitly versioned higher-throughput training protocol. The new protocol
uses batch size 64, 200 epochs for A4/A5, 100 epochs for A6, and learning rate
0.001 for every method. Implementation optimizations must retain the same model,
FP32 precision, losses, corruption seeds, sample order, validation definition,
and checkpoint selection rule within this new protocol.

The current all-137-person run remains untouched. The optimized route is for a
subsequent run after equivalence and GPU throughput benchmarks pass.

This is not a protocol-identical continuation of the batch-32 run. A4/A5 see
twice as many epochs and batch twice as many windows per update. Results must be
identified by their batch and epoch configuration in all run IDs and reports.

## Baseline

The current configuration uses 1,555 training windows and 361 validation
windows per epoch at batch size 32. Validation executes every sample as an
independent model forward. A6 additionally executes 654 training complete
cycles and 181 validation complete cycles one at a time.

Observed baseline throughput on two RTX 3090 GPUs is:

- A4: about 2.5 minutes per epoch and 23-24 epochs per hour.
- A6: about 5.8 minutes per epoch and 10-11 epochs per hour.
- GPU SM utilization: typically 20-56 percent despite about 8.5 GiB allocated
  per process.

The dominant causes are synchronous CPU corruption preparation, pageable and
blocking host-to-device copies, validation model forwards forced to batch size
one, and per-frame Python boolean checks in complete-cycle ROM computation that
synchronize the CPU and GPU repeatedly.

## New Training Protocol

The fixed configuration is:

| Method | Batch size | Epochs | Learning rate |
| --- | ---: | ---: | ---: |
| A4 | 64 | 200 | 0.001 |
| A5 | 64 | 200 | 0.001 |
| A6 | 64 | 100 | 0.001 |

With 1,555 training windows, the old fixed-window schedule performed 49 updates
per epoch. The new schedule performs 25. A4/A5 therefore move from about 4,900
to 5,000 optimizer updates. A6 retains 100 epochs because its 654 one-cycle
updates dominate each epoch: its approximate total moves from 70,300 updates
to 67,900. These counts explain the asymmetric epoch choice; they do not make
the batch-64 and batch-32 gradients scientifically identical.

## Protocol Invariants

The optimization must preserve all of the following:

- A4, A5, and A6 model architecture and hidden width.
- FP32 training; AMP, FP16, and BF16 remain disabled.
- Batch size 64 for fixed windows.
- 200 epochs for A4/A5 and 100 epochs for A6.
- One optimizer step per fixed-window batch.
- One optimizer step per complete training cycle in A6.
- Existing Adam configuration and the declared method-specific learning rate
  and epoch counts.
- Fold membership and deterministic sample order for a fixed seed.
- Stable corruption seed derivation and exact corruption tensors.
- All nine reported loss components and their configured weights.
- Validation sample membership, per-sample metric aggregation, validation score,
  and `score >= best` checkpoint selection.
- No triangulated pseudo-GT in training or checkpoint selection.

Implementation equivalence is measured against an unoptimized reference running
the same new method-specific configuration. It requires exact integer, boolean,
seed, ordering, and step-count identity, plus FP32 floating-point agreement
within explicit tolerances. Batched CUDA kernels can change floating-point
reduction order, so bitwise identity is not required. Any difference large
enough to change checkpoint selection fails the equivalence gate.

## Design

### Stage Timing

Add opt-in structured timing for CPU batch acquisition, corruption generation,
host-to-device transfer, window forward/backward, window validation,
complete-cycle training, and complete-cycle validation. CUDA events measure GPU
work; wall-clock timers measure end-to-end latency. Timing is diagnostic only
and is excluded from checkpoint state.

### Ordered CPU Prefetch

Introduce an ordered prepared-batch iterator that computes deterministic
corruptions for the next batch while the GPU processes the current batch. It
uses a bounded queue and emits batches in the original DataLoader order. Each
sample continues to use `stable_window_seed(epoch_seed, window_id)`, so worker
scheduling cannot affect generated tensors.

Prepared tensor batches are pinned before transfer. `_tensor_batch` accepts a
non-blocking transfer mode and preserves all non-tensor metadata. The training
loop synchronizes only where loss values or diagnostics must return to the CPU.

Validation corruptions are fixed at epoch zero. They may be prepared once and
reused across epochs, provided a test proves every cached tensor is exactly
equal to the uncached path and the cached values are never mutated.

### Batched Validation Forward

Replace the validation loop's one-sample model forwards with one forward per
existing validation batch. Corruption generation remains per sample with the
same stable seed. After the batched forward, outputs and prepared tensors are
split in original order and each sample's losses and diagnostics are computed
individually. Sorting, averaging, trial-level bone CV, complete-cycle ROM, and
validation score formulas remain unchanged.

This changes only CUDA execution shape. Equivalence tests compare every loss
component and final validation score against the scalar-forward reference.

### Complete-Cycle ROM Synchronization Removal

Keep complete-cycle training at batch size one and retain one optimizer step per
cycle. Replace `_rom_loss` frame-by-frame `bool(tensor)` checks with a vectorized
contiguous-run descriptor. Run boundaries are derived once from the effective
mask, then the existing circular unwrapping and per-run max/min ROM expression
are evaluated in the same run order.

The objective remains the mean squared difference between predicted and target
ROM over valid contiguous runs. Tests cover no valid run, one run, multiple runs,
gaps, padding, wrapped angles, and finite gradients.

### Zero-Weight Gradient Pruning

Continue computing and reporting every loss component. When a configured loss
weight is exactly zero, exclude that component from the autograd total instead
of multiplying its live graph by zero. This preserves the scalar total and all
reported metrics while avoiding backward traversal through objectives that
cannot contribute a gradient.

The implementation must not skip zero-weight validation metrics because the
validation score intentionally reports rotation and ROM behavior for every
ablation.

### Optional Compilation

`torch.compile` is a separate, default-off optimization. It is considered only
after the preceding changes pass equivalence and benchmark gates. Dynamic
complete-cycle lengths and recompilation counts must be measured. Compilation
is rejected if it changes checkpoint selection, increases end-to-end epoch
time, or requires relaxing the protocol invariants.

## Failure Handling

- A prefetch worker exception is re-raised on the training thread with its
  original traceback and stops the run.
- Prefetch queues are bounded and shut down on normal completion or exception.
- CUDA timing is disabled automatically on CPU runs.
- Pinned-memory and non-blocking transfer remain configurable so CPU-only tests
  and environments continue to work.
- The reference scalar validation and ROM implementations remain available to
  equivalence tests until the optimized path is proven.

## Verification

Focused tests must prove:

- Prefetched and direct corruptions are tensor-exact for several epochs and
  worker completion orders.
- Prefetch preserves batch and window order.
- Blocking and non-blocking transfer preserve tensor values and metadata.
- Batched and scalar validation produce the same ordered sample set, every loss
  component within `1e-6` absolute and relative tolerance, validation score
  within `1e-7`, and the same checkpoint-selection decision.
- Vectorized and reference ROM values and gradients agree within `1e-6` over
  all mask and wrap cases.
- Zero-weight pruning leaves total loss and model gradients unchanged within
  `1e-7` while preserving reported component values.
- Seeded one-epoch integration runs for A4 and A6 have optimizer-step counts
  matching the unoptimized batch-64 reference, identical corruption manifests,
  identical sample ordering, and complete checkpoint provenance.
- The full `tests/rotation_aware` suite passes.

GPU benchmarks use the existing prepared cache and report median times over
multiple warm epochs. A change is accepted only if it reduces end-to-end epoch
time without violating equivalence. GPU utilization is reported as supporting
evidence, not as the acceptance criterion.

## Implementation Outcome

The batched-validation proposal did not pass the locked GPU equivalence gate
and is not part of the production protocol. On an RTX 3090 at the production
`B=64`, `J=70`, `C=128` shape, the fixed-order batched experiment still
produced residual-coordinate deltas of `1.42e-6` to `2.62e-6` and
acceleration-loss deltas of `0.015625` to `0.0390625`. It also left
shape-dependent reductions in learned projections and did not establish a
throughput improvement.

Production training consequently retains the original scalar validation
forward and checkpoint selection. Ordered prefetch, pinned/nonblocking
transfer, validation-input caching, ROM synchronization removal, zero-weight
gradient pruning, stage profiling, and the isolated batch-64 schedule remain
in scope. Batched validation may be reconsidered only with a production-shape
CUDA test that satisfies all numerical and end-to-end timing acceptance gates.

## Expected Outcome

The target, not a guarantee, is 1.4-1.8 minutes per A4/A5 epoch and 3-4 minutes
per A6 epoch. Sustained 100 percent GPU utilization is not expected because A6
must retain one update per variable-length cycle. The primary success measure
is higher samples per second and lower epoch duration while faithfully executing
the newly declared batch-64 protocol.
