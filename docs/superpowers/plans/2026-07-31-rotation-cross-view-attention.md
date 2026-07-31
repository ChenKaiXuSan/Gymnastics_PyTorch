# 身体回旋交叉视角注意力实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 A4--A9 行为和旧 checkpoint 的前提下，实现 A10（身体回旋引导的双向 cross-view attention）与 A11（无显式回旋条件的 cross-view-only 对照），完成端到端训练、推理和评价准备。

**Architecture:** 新增一个独立的逐帧双向交叉注意力模块，以共享参数分别完成 face 查询 side 和 side 查询 face，并把结果送入现有交换不变的均值/绝对差融合。A10 保留 A6 的回旋输入与全部自监督损失；A11 保持相同网络容量，但将显式回旋输入置零并关闭三个回旋损失。旧方法默认不创建 attention 参数，保证历史 state dict 可继续严格加载。

**Tech Stack:** Python 3、PyTorch `nn.MultiheadAttention`、PyYAML、pytest、现有 `gymnastics.fusion.rotation_aware` CLI；所有项目命令通过 `conda run -n gymnastic` 前缀执行。

## Global Constraints

- A6 默认模型路径必须保持修改前的数值行为，A4--A9 旧 checkpoint 必须继续严格加载。
- A10 使用显式回旋输入、A6 完整自监督目标和双向 cross-view attention。
- A11 使用同构 attention 网络，但显式回旋输入置零，并关闭 circular、SO(3) 和 complete-cycle ROM 三项损失。
- A6、A10、A11 共享同一身体坐标规范化，不把预处理变化混入结构消融。
- attention 为逐帧关节级、一层、四头、dropout 0、`need_weights=False`。
- 两个方向共享参数，最终仍使用均值与绝对差，保持 view-swap invariance。
- 三角化 pseudo-GT 仅供评价，不参与训练、模型选择或超参数选择。
- 新训练从头开始；A6 checkpoint 不能作为 A10/A11 主实验的 warm start。
- 不修改或提交工作区中与本功能无关的论文和文档改动。

---

### Task 1: 独立的双向 Cross-View Attention 模块

**Files:**
- Create: `src/gymnastics/fusion/rotation_aware/cross_attention.py`
- Create: `tests/rotation_aware/test_cross_attention.py`

**Interfaces:**
- Consumes: `face`, `side` tensors of shape `[B,T,J,C]`; `valid_face`, `valid_side` masks of shape `[B,T,J]`.
- Produces: `BidirectionalCrossViewAttention(hidden_channels: int, num_heads: int = 4)` and `forward(face: Tensor, side: Tensor, valid_face: Tensor, valid_side: Tensor) -> tuple[Tensor, Tensor]`.
- Guarantees: finite outputs for empty key frames, zero invalid query tokens, shared directional parameters, swap-equivariant pair output.

- [ ] **Step 1: Write failing constructor and shape tests**

```python
def test_attention_rejects_non_divisible_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        BidirectionalCrossViewAttention(hidden_channels=10, num_heads=4)


def test_attention_preserves_shape_and_swaps_directionally() -> None:
    torch.manual_seed(3)
    face = torch.randn(2, 3, 5, 8)
    side = torch.randn(2, 3, 5, 8)
    valid_face = torch.ones(2, 3, 5, dtype=torch.bool)
    valid_side = torch.ones(2, 3, 5, dtype=torch.bool)
    block = BidirectionalCrossViewAttention(8, num_heads=2).eval()

    face_out, side_out = block(face, side, valid_face, valid_side)
    swapped_side, swapped_face = block(side, face, valid_side, valid_face)

    assert face_out.shape == face.shape
    assert side_out.shape == side.shape
    torch.testing.assert_close(face_out, swapped_face, atol=1e-6, rtol=0)
    torch.testing.assert_close(side_out, swapped_side, atol=1e-6, rtol=0)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_cross_attention.py -q
```

Expected: collection fails because `cross_attention.py` and `BidirectionalCrossViewAttention` do not exist.

- [ ] **Step 3: Implement the minimal attention block**

```python
class BidirectionalCrossViewAttention(nn.Module):
    def __init__(self, hidden_channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if num_heads <= 0 or hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.norm = nn.LayerNorm(hidden_channels)
        self.attention = nn.MultiheadAttention(
            hidden_channels,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(
        self,
        face: Tensor,
        side: Tensor,
        valid_face: Tensor,
        valid_side: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self._validate(face, side, valid_face, valid_side)
        return (
            self._exchange(face, side, valid_face, valid_side),
            self._exchange(side, face, valid_side, valid_face),
        )
```

`_exchange` 必须将 `[B,T,J,C]` 展平为 `[B*T,J,C]`，用 source mask 构造 `key_padding_mask`。全空 source 行临时开放第一个全零 key，attention 后再按 `source_valid.any(-1)` 清零 context；最后按 query mask 清零无效 token。

- [ ] **Step 4: Add failing invalid-mask and gradient tests**

```python
def test_attention_handles_empty_source_frames_without_nan() -> None:
    face = torch.randn(1, 2, 3, 8, requires_grad=True)
    side = torch.randn(1, 2, 3, 8, requires_grad=True)
    valid_face = torch.tensor([[[True, True, False], [True, False, False]]])
    valid_side = torch.tensor([[[True, False, True], [False, False, False]]])
    block = BidirectionalCrossViewAttention(8, num_heads=2)

    face_out, side_out = block(face, side, valid_face, valid_side)
    assert torch.isfinite(face_out).all()
    assert torch.isfinite(side_out).all()
    assert torch.equal(face_out[~valid_face], torch.zeros_like(face_out[~valid_face]))
    assert torch.equal(side_out[~valid_side], torch.zeros_like(side_out[~valid_side]))
    (face_out.square().sum() + side_out.square().sum()).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in block.parameters()
    )
```

- [ ] **Step 5: Run attention tests and verify GREEN**

Run:

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_cross_attention.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/gymnastics/fusion/rotation_aware/cross_attention.py tests/rotation_aware/test_cross_attention.py
git commit -m "feat: add bidirectional cross-view attention"
```

---

### Task 2: 将 Attention 与回旋条件开关接入融合模型

**Files:**
- Modify: `src/gymnastics/fusion/rotation_aware/model.py`
- Modify: `tests/rotation_aware/test_model.py`

**Interfaces:**
- Consumes: Task 1 的 `BidirectionalCrossViewAttention`.
- Extends: `RotationAwareFusionModel.__init__(spec: SkeletonSpec, *, hidden_channels: int = 128, max_delta_by_joint: float | Tensor | Sequence[float] = 0.05, twist_residual: bool = False, max_twist: float = 1.0, twist_gate_sharpness: float = 8.0, cross_attention: bool = False, attention_heads: int = 4, rotation_conditioning: bool = True)`.
- Extends: `SharedViewEncoder.forward(points: Tensor, valid: Tensor, effective_mask: Tensor, features: FeatureBundle, trunk_rotation: Tensor, trunk_angle: Tensor, trunk_omega: Tensor, trunk_alpha: Tensor, trunk_valid: Tensor, *, rotation_conditioning: bool = True)`.
- Extends: `_cross_features(cross: DisagreementFeatures, shape: tuple[int, int, int], dtype: torch.dtype, *, rotation_conditioning: bool = True)`.
- Guarantees: `cross_attention=False` 走原始 A6 字节兼容路径；A10/A11 参数量一致。

- [ ] **Step 1: Write failing model-mode tests**

```python
def test_cross_attention_model_is_view_swap_invariant() -> None:
    torch.manual_seed(17)
    face, side, ff, sf, cross, vf, vs = _inputs()
    model = RotationAwareFusionModel(
        SPEC,
        hidden_channels=16,
        cross_attention=True,
        attention_heads=4,
    ).eval()
    left = model(face, side, ff, sf, cross, vf, vs)
    swapped_cross = compute_disagreement_features(
        side, face,
        extract_trunk_features(side, vs, SPEC, dt=1.0),
        extract_trunk_features(face, vf, SPEC, dt=1.0),
        vs, vf,
    )
    right = model(side, face, sf, ff, swapped_cross, vs, vf)
    torch.testing.assert_close(left.fused_kpts, right.fused_kpts, atol=1e-5, rtol=0)


def test_rotation_conditioned_and_unconditioned_attention_have_equal_parameter_counts() -> None:
    conditioned = RotationAwareFusionModel(
        SPEC, hidden_channels=16, cross_attention=True, rotation_conditioning=True
    )
    unconditioned = RotationAwareFusionModel(
        SPEC, hidden_channels=16, cross_attention=True, rotation_conditioning=False
    )
    assert sum(p.numel() for p in conditioned.parameters()) == sum(
        p.numel() for p in unconditioned.parameters()
    )
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/rotation_aware/test_model.py::test_cross_attention_model_is_view_swap_invariant \
  tests/rotation_aware/test_model.py::test_rotation_conditioned_and_unconditioned_attention_have_equal_parameter_counts -q
```

Expected: constructor rejects unknown `cross_attention` and `rotation_conditioning`.

- [ ] **Step 3: Add constructor flags and conditional attention path**

Add these defaults without changing the existing default state dict:

```python
self.cross_attention = bool(cross_attention)
self.rotation_conditioning = bool(rotation_conditioning)
self.cross_view_attention = (
    BidirectionalCrossViewAttention(hidden_channels, attention_heads)
    if self.cross_attention
    else None
)
```

When attention is disabled, retain the current calls and masks exactly. When enabled:

1. encode face with `valid_face` and side with `valid_side`;
2. run `self.cross_view_attention`;
3. form mean/absolute-difference features;
4. mask the symmetric result by `effective_mask`;
5. continue through unchanged projection, TCN and residual head.

- [ ] **Step 4: Write failing no-rotation input tests**

The tests must show that disabling rotation conditioning makes encoder output independent of trunk rotation tensors and makes cross encoding independent of angle/rotation disagreement fields:

```python
def test_unconditioned_view_encoder_ignores_rotation_inputs() -> None:
    face, _, face_features, _, _, valid_face, _ = _inputs()
    batch, frames = face.shape[:2]
    encoder = SharedViewEncoder(hidden_channels=16).eval()
    frame_valid = valid_face.any(dim=-1)
    identity = torch.eye(3).expand(batch, frames, 3, 3).clone()
    zeros = torch.zeros(batch, frames)
    first = encoder(
        face, valid_face, valid_face, face_features,
        identity, zeros, zeros, zeros, frame_valid,
        rotation_conditioning=False,
    )
    second = encoder(
        face, valid_face, valid_face, face_features,
        -identity, zeros + 1.0, zeros + 2.0, zeros + 3.0, ~frame_valid,
        rotation_conditioning=False,
    )
    torch.testing.assert_close(first, second)


def test_unconditioned_cross_features_zero_angle_and_rotation_channels() -> None:
    original = RotationAwareFusionModel._cross_features(
        cross, shape, torch.float32, rotation_conditioning=False
    )
    changed = replace(
        cross,
        angle_abs_delta=cross.angle_abs_delta + 20,
        rotation_distance=cross.rotation_distance + 30,
    )
    actual = RotationAwareFusionModel._cross_features(
        changed, shape, torch.float32, rotation_conditioning=False
    )
    torch.testing.assert_close(original, actual)
```

- [ ] **Step 5: Implement rotation input masking**

In `SharedViewEncoder.forward`, build the existing 14-channel trunk vector, then replace it by zeros when `rotation_conditioning=False`. This retains the same trunk MLP and parameter count.

In `_cross_features`, retain all 13 channels but zero indices corresponding to:

- angle difference and validity;
- SO(3) rotation distance and validity.

Coordinate difference, trunk displacement and validity-difference channels remain active.

- [ ] **Step 6: Add an A6 state-dict compatibility regression**

```python
def test_default_model_does_not_create_cross_attention_state() -> None:
    model = RotationAwareFusionModel(SPEC, hidden_channels=16)
    assert not any(name.startswith("cross_view_attention.") for name in model.state_dict())
    assert model.cross_attention is False
    assert model.rotation_conditioning is True
```

This catches accidentally creating new required parameters in legacy A4--A9 models.

- [ ] **Step 7: Run model suite and verify GREEN**

Run:

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_model.py -q
```

Expected: all model tests pass, including existing A6 swap and masking tests.

- [ ] **Step 8: Commit Task 2**

```bash
git add src/gymnastics/fusion/rotation_aware/model.py tests/rotation_aware/test_model.py
git commit -m "feat: integrate rotation-conditioned view exchange"
```

---

### Task 3: 注册 A10/A11、损失边界和生产配置

**Files:**
- Modify: `src/gymnastics/fusion/rotation_aware/cli.py`
- Modify: `tests/rotation_aware/test_cli.py`
- Create: `configs/fusion/rotation_aware_cross_attention.yaml`

**Interfaces:**
- Extends: `LEARNED_ABLATIONS` with `"A10"` and `"A11"`.
- Produces: `CROSS_ATTENTION_ABLATIONS = frozenset({"A10", "A11"})`.
- Produces: `ROTATION_UNCONDITIONED_ABLATIONS = frozenset({"A11"})`.
- Produces: `model_kwargs_for_training(training: Mapping[str, Any]) -> dict[str, Any]`.
- Extends: `loss_config_for_ablation`.

- [ ] **Step 1: Write failing ablation and loss tests**

```python
def test_cross_attention_ablation_flags_and_losses_are_isolated() -> None:
    assert CROSS_ATTENTION_ABLATIONS == {"A10", "A11"}
    a10 = loss_config_for_ablation("A10")
    a11 = loss_config_for_ablation("A11")
    assert a10 == LossConfig()
    assert a11.circular_axial_rotation_weight == 0.0
    assert a11.so3_rotation_weight == 0.0
    assert a11.complete_cycle_rom_weight == 0.0
    assert a11.adaptive_temporal_acceleration_weight == 1.0


def test_model_kwargs_distinguish_a10_and_a11_without_capacity_change() -> None:
    common = {"hidden_channels": 128, "attention_heads": 4}
    assert model_kwargs_for_training({**common, "ablation": "A10"}) == {
        "hidden_channels": 128,
        "twist_residual": False,
        "cross_attention": True,
        "attention_heads": 4,
        "rotation_conditioning": True,
    }
    assert model_kwargs_for_training({**common, "ablation": "A11"}) == {
        "hidden_channels": 128,
        "twist_residual": False,
        "cross_attention": True,
        "attention_heads": 4,
        "rotation_conditioning": False,
    }
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/rotation_aware/test_cli.py::test_cross_attention_ablation_flags_and_losses_are_isolated \
  tests/rotation_aware/test_cli.py::test_model_kwargs_distinguish_a10_and_a11_without_capacity_change -q
```

Expected: imports or assertions fail because A10/A11 registrations do not exist.

- [ ] **Step 3: Implement ablation constants, loss mapping and model kwargs**

`A10` returns the unchanged full `LossConfig`. `A11` uses:

```python
replace(
    full,
    circular_axial_rotation_weight=0.0,
    so3_rotation_weight=0.0,
    complete_cycle_rom_weight=0.0,
)
```

Both `_cmd_train` and `_cmd_infer` must construct the model through `model_kwargs_for_training` so architecture resolution cannot drift.

- [ ] **Step 4: Write failing schedule/config test**

```python
def test_cross_attention_production_config_declares_equal_budgets() -> None:
    config = load_config("configs/fusion/rotation_aware_cross_attention.yaml")
    assert _training_config_for_ablation(config, "A10")["epochs"] == 100
    assert _training_config_for_ablation(config, "A11")["epochs"] == 100
    assert config["training"]["attention_heads"] == 4
    assert config["training"]["hidden_channels"] == 128
```

- [ ] **Step 5: Add isolated production config**

Create a config matching the existing batch-64 data/window/performance paths, with:

```yaml
training:
  epochs_by_ablation: {A10: 100, A11: 100}
  batch_size: 64
  learning_rate: 0.001
  hidden_channels: 128
  attention_heads: 4
  seed: 0
  device: cuda:0
  protocol:
    run_id_token_template: "{ablation_lower}_b{batch_size}_e{epochs}_s{seed}"
```

- [ ] **Step 6: Run CLI tests and verify GREEN**

Run:

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware/test_cli.py -q
```

Expected: all CLI tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/gymnastics/fusion/rotation_aware/cli.py tests/rotation_aware/test_cli.py configs/fusion/rotation_aware_cross_attention.yaml
git commit -m "feat: register A10 and A11 training protocols"
```

---

### Task 4: Checkpoint、推理元数据和评价发现

**Files:**
- Modify: `src/gymnastics/fusion/rotation_aware/cli.py`
- Modify: `src/gymnastics/fusion/rotation_aware/evaluation.py`
- Modify: `tests/rotation_aware/test_inference.py`
- Modify: `tests/rotation_aware/test_evaluation.py`
- Modify: `tests/rotation_aware/test_cli.py`

**Interfaces:**
- Consumes: Task 3 的 `model_kwargs_for_training`.
- Extends: checkpoint `training_config` and inference `model_config` with `cross_attention`, `attention_heads`, `rotation_conditioning`.
- Extends: `ABLATION_REGISTRY` with A10/A11.
- Guarantees: inference rebuilds the architecture encoded by the checkpoint rather than relying on CLI defaults.

- [ ] **Step 1: Write failing checkpoint reconstruction test**

Create an A11 model through `model_kwargs_for_training`, save it with `save_checkpoint`, then invoke the existing infer reconstruction boundary and assert:

```python
assert metadata["ablation"] == "A11"
assert metadata["model_config"] == {
    "hidden_channels": 8,
    "cross_attention": True,
    "attention_heads": 2,
    "rotation_conditioning": False,
}
```

The production change this catches is an infer path that silently rebuilds A11 as legacy A6.

- [ ] **Step 2: Run focused inference/CLI tests and verify RED**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/rotation_aware/test_inference.py \
  tests/rotation_aware/test_cli.py -q
```

Expected: new metadata assertions fail because only `hidden_channels` is currently recorded.

- [ ] **Step 3: Persist and validate architecture metadata**

- resolve `attention_heads` into `training` before checkpoint save;
- add it to protected protocol fields;
- build train and infer models with `model_kwargs_for_training`;
- emit all four architecture fields under inference provenance `model_config`;
- preserve old checkpoint defaults: no attention, four heads, rotation conditioning enabled;
- reject an explicit CLI ablation that differs from the saved ablation.

- [ ] **Step 4: Write failing evaluation discovery tests**

```python
def test_registry_names_cross_attention_ablations() -> None:
    assert ABLATION_REGISTRY["A10"] == "rotation_conditioned_cross_view_attention"
    assert ABLATION_REGISTRY["A11"] == "cross_view_attention_without_rotation"
```

Extend the existing synthetic NPZ discovery fixture with A10 and A11 payloads and assert both are returned with their exact method labels and diagnostics.

- [ ] **Step 5: Extend all learned-method allowlists**

Add A10/A11 to:

- `ABLATION_REGISTRY`;
- learned checkpoint output discovery;
- diagnostic availability handling;
- evaluation row generation;
- any CLI `choices=list(LEARNED_ABLATIONS)` path via the central constant.

Do not change the mappings for A0--A9.

- [ ] **Step 6: Run inference and evaluation suites**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/rotation_aware/test_inference.py \
  tests/rotation_aware/test_evaluation.py \
  tests/rotation_aware/test_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add \
  src/gymnastics/fusion/rotation_aware/cli.py \
  src/gymnastics/fusion/rotation_aware/evaluation.py \
  tests/rotation_aware/test_inference.py \
  tests/rotation_aware/test_evaluation.py \
  tests/rotation_aware/test_cli.py
git commit -m "feat: persist and evaluate attention ablations"
```

---

### Task 5: A10/A11 端到端训练契约和文档

**Files:**
- Modify: `tests/rotation_aware/test_end_to_end.py`
- Modify: `docs/rotation_aware_fusion.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1--4 的完整 CLI。
- Produces: CPU tiny A10/A11 train-infer-evaluate regression and user-facing commands.

- [ ] **Step 1: Write parameterized failing end-to-end test**

Refactor现有 synthetic fixture 生成逻辑为测试文件内 helper，并增加：

```python
@pytest.mark.parametrize(
    ("ablation", "rotation_conditioning"),
    [("A10", True), ("A11", False)],
)
def test_cross_attention_ablations_train_infer_and_evaluate(
    tmp_path: Path,
    ablation: str,
    rotation_conditioning: bool,
) -> None:
    # 16-frame tiny cycle, hidden_channels=8, attention_heads=2, CPU, 1 epoch.
    assert main(["prepare", "--config", str(config), "--person", "1"]) == 0
    assert main(["train", "--config", str(config), "--run-id", run_id, "--ablation", ablation]) == 0
    assert main(["infer", "--config", str(config), "--run-id", run_id, "--person", "1"]) == 0
    assert main(["evaluate", "--config", str(config), "--run-id", run_id, "--person", "1"]) == 0
    assert metadata["ablation"] == ablation
    assert metadata["model_config"]["cross_attention"] is True
    assert metadata["model_config"]["rotation_conditioning"] is rotation_conditioning
```

- [ ] **Step 2: Run E2E test and verify RED**

Run:

```bash
conda run -n gymnastic python -m pytest \
  tests/rotation_aware/test_end_to_end.py::test_cross_attention_ablations_train_infer_and_evaluate -q
```

Expected: fail before complete A10/A11 CLI integration.

- [ ] **Step 3: Make only integration fixes required by the E2E test**

Fix actual boundary mismatches revealed by the test. Every unexpected bug receives its own smallest failing regression before production changes.

- [ ] **Step 4: Document exact commands and interpretation**

Add to `docs/rotation_aware_fusion.md` and `README.md`:

```bash
conda run -n gymnastic gymnastics fuse rotation-aware train \
  --config configs/fusion/rotation_aware_cross_attention.yaml \
  --run-id paper_a10_b64_e100_s0 --ablation A10

conda run -n gymnastic gymnastics fuse rotation-aware train \
  --config configs/fusion/rotation_aware_cross_attention.yaml \
  --run-id paper_a11_b64_e100_s0 --ablation A11
```

State that A11 has no explicit rotation conditioning but retains body-frame canonicalization. State that triangulated 3D remains evaluation-only.

- [ ] **Step 5: Run full rotation-aware suite**

Run:

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware -q
```

Expected: zero failures.

- [ ] **Step 6: Commit Task 5**

```bash
git add tests/rotation_aware/test_end_to_end.py docs/rotation_aware_fusion.md README.md
git commit -m "test: cover attention ablations end to end"
```

---

### Task 6: 真实数据 Smoke Test 与训练启动准备

**Files:**
- Generated only: runtime artifacts below `local/runs/fuse_rotation_aware/`
- No tracked source edits unless a smoke failure first receives a regression test.

**Interfaces:**
- Consumes: production config and existing prepared cache.
- Produces: one-epoch A10/A11 smoke artifacts, resource measurements, and validated 100-epoch launch commands.

- [ ] **Step 1: Inspect GPU and active process state**

Run:

```bash
nvidia-smi
ps -eo pid,ppid,stat,etime,pcpu,pmem,args
```

Record available GPU memory and ensure an existing user training process is not displaced.

- [ ] **Step 2: Create runtime-only smoke configs**

Copy the resolved production settings into ignored runtime configs under `local/runs/fuse_rotation_aware/smoke_configs/`, changing only:

- `epochs_by_ablation` to one epoch for A10/A11;
- batch size to a measured safe micro-batch if batch 64 does not fit;
- run IDs to `smoke_a10_*` and `smoke_a11_*`.

Do not commit runtime configs or outputs.

- [ ] **Step 3: Run A10 one-epoch smoke**

Run:

```bash
conda run -n gymnastic gymnastics fuse rotation-aware train \
  --config local/runs/fuse_rotation_aware/smoke_configs/a10.yaml \
  --run-id smoke_a10_b64_e1_s0 --ablation A10
```

Expected: finite training loss and validation score, checkpoint and metadata written.

- [ ] **Step 4: Run A11 one-epoch smoke**

Run:

```bash
conda run -n gymnastic gymnastics fuse rotation-aware train \
  --config local/runs/fuse_rotation_aware/smoke_configs/a11.yaml \
  --run-id smoke_a11_b64_e1_s0 --ablation A11
```

Expected: finite training loss and validation score, checkpoint and metadata written.

- [ ] **Step 5: Infer and evaluate smoke checkpoints**

For each smoke run, execute `infer` and `evaluate` with the same config and run ID. Verify:

- finite fused keypoints;
- metadata records the correct architecture;
- view-swap error is finite and within the existing numerical tolerance;
- triangulated roots are accessed only during `evaluate`.

- [ ] **Step 6: Report production launch readiness**

Report:

- safe micro-batch and effective batch strategy;
- peak GPU memory;
- A10/A11 seconds per epoch;
- smoke loss/validation score;
- exact 100-epoch commands;
- whether production training has been launched.

Do not claim the method improves accuracy until held-out evaluation is complete.

---

### Task 7: Final Verification

**Files:**
- Verify all files changed by Tasks 1--5.

**Interfaces:**
- Produces: evidence that implementation, legacy compatibility and documentation are complete.

- [ ] **Step 1: Run focused and project-required tests**

```bash
conda run -n gymnastic python -m pytest tests/rotation_aware -q
conda run -n gymnastic python -m pytest tests/test_fuse_experiment_matrix.py -q
conda run -n gymnastic python -m pytest \
  tests/test_sam3d_triangulation.py \
  tests/test_compare_fused_triangulated.py -q
```

- [ ] **Step 2: Check formatting and unintended changes**

```bash
git diff --check
git status --short
git diff --stat
```

Review every changed tracked file and exclude unrelated user modifications from commits.

- [ ] **Step 3: Verify requirements one by one**

Confirm from fresh outputs:

- A6 default state dict has no attention parameters;
- old A6 checkpoint loads;
- A10/A11 parameter counts match;
- A10/A11 swap tests pass;
- A11 rotation inputs and losses are disabled;
- checkpoint inference rebuilds the right architecture;
- E2E and real smoke tests are finite;
- no training path reads triangulated pseudo-GT.

- [ ] **Step 4: Confirm task commits contain only intended files**

Run `git show --stat --oneline HEAD~5..HEAD` and compare the file list with Tasks 1--5. If verification exposes a new defect, return to the owning task, add a failing regression, implement the minimal fix, rerun that task's tests, and commit only that task's explicitly listed files.
