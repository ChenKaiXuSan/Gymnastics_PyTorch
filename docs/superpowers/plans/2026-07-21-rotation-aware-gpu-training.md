# Rotation-Aware GPU Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train rotation-aware ablations A4, A5, and A6 for 100 epochs on all 137 people with two RTX 3090 GPUs, then infer and evaluate the resulting methods.

**Architecture:** Add one configuration-driven device handoff at the CLI-to-training boundary while preserving CPU defaults. Build a deterministic demographic-stratified person fold, prepare one shared immutable cache, run A6 on GPU0 and A4 then A5 on GPU1, and perform combined inference/evaluation only after all checkpoints pass validation.

**Tech Stack:** Python 3.10, PyTorch, pytest, YAML/JSON, tmux, existing `fuse.rotation_aware` CLI.

## Global Constraints

- Use all 137 people: 80 elderly and 57 students; ID135/S55 is intentionally absent.
- Use person-disjoint train/val/test splits with 96/27/14 people.
- Train A4, A5, and A6 for exactly 100 epochs each.
- Use triangulated data only for external evaluation, never training or checkpoint selection.
- Preserve existing deterministic Fuse outputs and existing classification folds.
- Run orchestration in tmux and keep per-run logs and completion metadata.

---

### Task 1: Configuration-Driven Training Device

**Files:**
- Modify: `fuse/rotation_aware/cli.py`
- Test: `tests/rotation_aware/test_cli.py`

**Interfaces:**
- Consumes: `training.device` as a PyTorch device string, defaulting to `cpu`.
- Produces: `train_one_epoch(..., device=device)` and `validate(..., device=device)` calls.

- [ ] **Step 1: Write a failing test**

Add a focused test that configures `training.device: cuda:1`, replaces the CLI module's `train_one_epoch` and `validate` functions with recording fakes, runs one tiny training command, and asserts both fakes received `device="cuda:1"`.

- [ ] **Step 2: Verify the test fails**

Run:

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_cli.py -k configured_training_device -q
```

Expected: failure because `_cmd_train` does not pass `device`.

- [ ] **Step 3: Implement the minimum device handoff**

In `_cmd_train`, after resolving `training`, add:

```python
device = str(training.get("device", "cpu"))
```

Pass `device=device` to both `train_one_epoch` and `validate`. Do not change the CPU default.

- [ ] **Step 4: Verify focused and full tests**

Run:

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_cli.py -k configured_training_device -q
conda run -n gymnastic python -m pytest tests/rotation_aware -q
```

Expected: focused test passes; full suite reports 162 passing tests.

### Task 2: Full-Data Training Inputs And GPU Smoke Test

**Files:**
- Create at runtime: `logs/fuse_rotation_aware/folds/fold_00_all137.json`
- Create at runtime: `logs/fuse_rotation_aware/configs/all137_100ep_gpu0.yaml`
- Create at runtime: `logs/fuse_rotation_aware/configs/all137_100ep_gpu1.yaml`

**Interfaces:**
- Consumes: current SAM3D and split-cycle outputs for 137 people.
- Produces: one fold with 96 train, 27 val, and 14 test people; two identical configs differing only in `training.device`.

- [ ] **Step 1: Generate deterministic demographic-stratified membership**

Shuffle elderly IDs 1-80 and student IDs 81-134,136-138 independently with seed 20260721. Allocate elderly 56/16/8 and students 40/11/6 across train/val/test.

- [ ] **Step 2: Write 100-epoch configs**

Use absolute paths for SAM3D, split-cycle, skeleton, folds, deterministic Fuse outputs, triangulated references, and `logs/fuse_rotation_aware`. Set epochs=100, batch_size=4, hidden_channels=128, seed=0, and device to `cuda:0` or `cuda:1`.

- [ ] **Step 3: Run a one-person, one-epoch GPU smoke test**

Prepare one aligned person into a temporary output, train A6 for one epoch on `cuda:0`, and verify the checkpoint exists and `nvidia-smi` reports GPU memory use during training.

### Task 3: Tmux Training, Inference, And Evaluation

**Files:**
- Create at runtime: `logs/fuse_rotation_aware/logs/*.log`
- Create at runtime: `logs/fuse_rotation_aware/runs/{all137_a4_e100_seed0,all137_a5_e100_seed0,all137_a6_e100_seed0}`
- Create at runtime: `logs/fuse_rotation_aware/inference/*`
- Create at runtime: `logs/fuse_rotation_aware/evaluation/all137_a4_e100_seed0+all137_a5_e100_seed0+all137_a6_e100_seed0`

**Interfaces:**
- Consumes: shared cache, fold, GPU configs, and three trained checkpoints.
- Produces: A0-A6 evaluation rows plus external triangulated metrics and diagnostics.

- [ ] **Step 1: Prepare and validate the shared cache**

Run `prepare` once and require 137 prepared people, zero failures, and 928 cached cycles.

- [ ] **Step 2: Train three runs on two GPUs**

Start A6 on GPU0. On GPU1, train A4 and then A5. Require exactly 100 finite metric rows and a best checkpoint for every run.

- [ ] **Step 3: Infer all three runs**

Run `infer` for A4, A5, and A6 across all fold people and require 137 person directories and 928 cycle outputs per run.

- [ ] **Step 4: Run combined evaluation**

Evaluate all three run IDs together so A0-A6 and existing deterministic methods share one report. Require finite external MPJPE for learned methods and `no_pseudo_gt_training: true`.

- [ ] **Step 5: Write completion metadata**

Record fold counts, epoch counts, checkpoint hashes, inference coverage, evaluation row counts, best validation epochs, and final paths in `logs/fuse_rotation_aware/RUN_COMPLETE.json`.
