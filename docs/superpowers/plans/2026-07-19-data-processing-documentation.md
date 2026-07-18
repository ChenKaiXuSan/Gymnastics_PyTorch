# 中文数据处理流程文档 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将仓库现有流程概览和运行手册更新为准确、可执行、易于长期查询的中文文档，并从 README 提供一致入口。

**Architecture:** `docs/current_pipeline.md` 只负责解释当前双视角流水线、数据流和模块边界；`docs/runbook.md` 负责单人及全数据集的实际执行步骤、结果检查和故障排查；`README.md` 只保留简短流程及两份权威文档的入口。三个文件共享同一处理顺序，但不复制大段说明。

**Tech Stack:** Markdown、Python 模块命令、Hydra 配置、Conda `gymnastic` 环境、Git 静态检查。

## Global Constraints

- 文档正文使用中文；代码标识、命令、配置键和路径保持原样。
- 当前处理顺序固定为 `SAM3D-Body -> split_cycle -> triangulation -> fuse -> analysis / optional train`。
- 完整流程必须明确要求同一人的 `face + side` 两个视角。
- 所有项目命令默认写成 `conda run -n gymnastic ...`。
- `fuse` 必须使用 `logs/split_cycle/person_<id>/alignment_record_<id>.json` 中的 `offset_side_to_face`，不能描述为重新估计 DTW 偏移。
- 当前推荐融合方法固定为 `sim3_face_stable_smooth_kpt`。
- 不修改 Python 代码、配置、实验参数或 `legacy/prepare_dataset` 行为。
- 不运行耗时的推理、三角化、融合或训练任务，只进行文档及命令接口的静态验证。
- 保留用户现有的 `AGENTS.md` 修改，不暂存、不提交。

---

### Task 1: 重写当前流水线概览

**Files:**
- Modify: `docs/current_pipeline.md:1-29`
- Reference: `docs/superpowers/specs/2026-07-18-data-processing-documentation-design.md`
- Reference: `AGENTS.md`

**Interfaces:**
- Consumes: 当前入口模块、默认数据目录、切分对齐记录、推荐融合方法。
- Produces: `docs/current_pipeline.md`，作为 README 和运行手册引用的权威概念说明。

- [ ] **Step 1: 记录旧流程检查的失败基线**

Run:

```bash
rg -n "fuse|split_cycle|drivefusion" docs/current_pipeline.md
```

Expected: 输出显示旧文档在流水线中把 `fuse` 放在 `split_cycle` 前面，证明概览需要更新。

- [ ] **Step 2: 用中文完整替换流程概览**

使用 `apply_patch` 重写 `docs/current_pipeline.md`，并按以下准确结构组织：

````markdown
# 当前数据处理流程

## 适用范围

完整流水线以同一人的 `face` 和 `side` 双视角视频为输入。只有一个视角时，只能执行 SAM3D-Body 单视角推理，不能完成双视角对齐、三角化或融合。

## 端到端流程

```text
/home/data/xchen/gymnastics/raw/person/<id>/ID<id>_{face,side}.MOV
  -> SAM3D-Body 逐帧提取 2D/3D 关键点
  -> split_cycle 时间对齐和动作周期切分
  -> triangulation 使用 2D 关键点生成 3D 伪真值
  -> fuse 对齐并融合 face/side 的 3D 关键点
  -> analysis 与三角化伪真值比较
  -> project/train（可选分类任务）
```

## 各阶段说明

| 阶段 | 输入 | 主要处理 | 输出 |
|---|---|---|---|
| SAM3D-Body | face/side 原始视频 | 逐帧人体推理 | `sam3d_body_results/person/<id>/<view>/*.npz` |
| split_cycle | 原始视频与 SAM3D 关键点 | 时间对齐、偏移选择、周期切分 | `logs/split_cycle/person_<id>/alignment_record_<id>.json` 和周期视频 |
| triangulation | 对齐记录与两个视角的 2D 关键点 | 多视角三角化 | `sam3d_triangulated/person/person_<id>/cycle_<idx>/` |
| fuse | 对齐记录与两个视角的 3D 关键点 | Sim3 对齐、融合、平滑 | `logs/fuse_experiments/<method>/person_<id>/fused_sequence.npz` |
| analysis | 融合结果与三角化伪真值 | MPJPE 等指标及报告 | `logs/analysis/` 和融合指标 CSV |
| project/train | 已准备动作数据与人员级划分 | 分类训练与评估 | `logs/train/` 等训练输出 |
````

随后必须包含以下独立小节：

- `## 关键规则`：说明 `split_cycle` 的对齐记录是 triangulation 和 fuse 的共同时间基准；fuse 不回退到新估计的关键点 DTW 偏移；face 是参考视角。
- `## 推荐融合方法`：准确解释 `sim3_face_stable_smooth_kpt` 的“稳定关节估计 Sim3 -> side 对齐到 face -> 平均 -> 时间平滑”。
- `## 主要目录`：列出 raw、SAM3D、split-cycle、triangulated、fuse experiments、person folds。
- `## 入口命令`：按正确顺序列出六条 `conda run -n gymnastic ...` 命令，并将分类标为可选。
- `## 旧流程`：说明 `legacy/prepare_dataset` 的 DPT/RAFT/YOLO/Detectron2 路径不属于当前活动流程。
- 文末链接 `[运行手册](runbook.md)`、`[模块职责](modules.md)` 和 `../triangulation/README.md`。

- [ ] **Step 3: 验证概览中的顺序和关键事实**

Run:

```bash
rg -n "SAM3D-Body|split_cycle|triangulation|sim3_face_stable_smooth_kpt|offset_side_to_face|gymnastic" docs/current_pipeline.md
```

Expected: 每个关键词都有清晰命中，流程图中 `split_cycle` 位于 `triangulation` 和 `fuse` 之前。

Run:

```bash
rg -n "drivefusion|fuse.*->.*split_cycle|重新估计.*DTW" docs/current_pipeline.md
```

Expected: 无输出，退出码为 `1`。

Run:

```bash
git diff --check -- docs/current_pipeline.md
```

Expected: 无输出，退出码为 `0`。

- [ ] **Step 4: 提交流程概览**

```bash
git add docs/current_pipeline.md
git commit -m "docs: 更新中文数据处理流程"
```

提交前运行 `git diff --cached --name-only`，预期只显示 `docs/current_pipeline.md`。

---

### Task 2: 重写可执行运行手册

**Files:**
- Modify: `docs/runbook.md:1-255`
- Reference: `configs/sam3d_body.yaml`
- Reference: `configs/sam3d_triangulation.yaml`
- Reference: `split_cycle/main.py:840-921`
- Reference: `triangulation/sam3d_from_split_cycle.py:395-454`
- Reference: `fuse/experiment_matrix.py:776-840`

**Interfaces:**
- Consumes: Task 1 定义的处理顺序和术语。
- Produces: 可复制执行的单人、全数据集及自定义数据根目录命令；每一步都有预期输出和检查方法。

- [ ] **Step 1: 记录旧运行手册的失败基线**

Run:

```bash
rg -n "drivefusion|Run the default multi-view fusion|Run cycle segmentation|One-Person" docs/runbook.md
```

Expected: 命中旧环境、英文标题以及 fusion 先于 split-cycle 的旧顺序。

- [ ] **Step 2: 建立中文手册头部和运行前说明**

使用 `apply_patch` 重写 `docs/runbook.md`。文件开头使用以下内容和顺序：

````markdown
# 数据处理运行手册

本文档说明如何处理同一人的 `face + side` 双视角视频。流程原理见[当前数据处理流程](current_pipeline.md)。

## 运行环境

所有项目命令默认使用：

```bash
conda run -n gymnastic ...
```

默认数据根目录为 `/home/data/xchen/gymnastics`。`GYMNASTICS_DATA_ROOT` 会被 SAM3D-Body 和三角化的 Hydra 配置读取；`split_cycle` 和 `fuse` 使用独立命令行路径参数，自定义数据根目录时也必须显式传入对应参数。
````

继续加入 `## 单人完整处理示例`，统一使用 person `46`，并明确“将所有 `46` 替换为目标人物编号”。

- [ ] **Step 3: 写入单人处理的准确命令和检查点**

按以下顺序写入小节、命令和预期结果：

1. `### 1. 检查原始视频`

```bash
ls /home/data/xchen/gymnastics/raw/person/46/ID46_face.MOV /home/data/xchen/gymnastics/raw/person/46/ID46_side.MOV
```

预期两个文件都存在；缺少任一视角时停止完整双视角流程。

2. `### 2. 运行 SAM3D-Body`

```bash
conda run -n gymnastic python -m SAM3Dbody.main infer.person_list=[46] infer.gpu=[0] infer.workers_per_gpu=1
```

记录 face/side `*_sam3d_body.npz`、可视化目录和 `person_logs/46.log`，并保留两个 `find ... | wc -l` 计数命令。

3. `### 3. 时间对齐和周期切分`

```bash
conda run -n gymnastic python -m split_cycle.main --person 46 --threads 1
```

记录 `alignment_record_46.json`、`theta_unwrap.png`、face/side 周期视频，并说明后续重点读取 `offset_side_to_face`。

4. `### 4. 三角化生成 3D 伪真值`

先写冒烟命令：

```bash
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle --person 46 --max-cycles 1 --max-frames 2
```

再写正式命令：

```bash
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle --person 46
```

记录 `summary.json`、`joints_3d_sequence.npz`、可视化和 3D 视频；解释 `missing_pairs` 应尽量为 `0`，并检查 face/side 重投影误差。

5. `### 5. 运行推荐融合方法`

```bash
conda run -n gymnastic python -m fuse --person 46 --methods sim3_face_stable_smooth_kpt
```

记录以下输出：

```text
logs/fuse_experiments/sim3_face_stable_smooth_kpt/person_46/fused_sequence.npz
logs/fuse_experiments/sim3_face_stable_smooth_kpt/person_46/config.json
logs/fuse_experiments/metrics_by_person.csv
logs/fuse_experiments/metrics_by_joint.csv
```

说明缺少 `alignment_record_46.json` 时融合必须失败；缺少 triangulated cycle 时可以生成融合序列，但不会得到有效伪真值评估指标。

6. `### 6. 生成或检查分析结果`

```bash
conda run -n gymnastic python triangulation/tools/generate_results_report.py
```

记录 `logs/analysis/triangulated_results/` 下的 Markdown 和 CSV 报告。

7. `### 7. 可选：分类训练`

```bash
conda run -n gymnastic python -m project.train.train
```

明确这不是生成融合关键点的必需步骤。

- [ ] **Step 4: 写入全数据集、自定义路径和故障排查**

加入 `## 全数据集处理`，按照相同顺序列出：

```bash
conda run -n gymnastic python -m SAM3Dbody.main
conda run -n gymnastic python -m split_cycle.main
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle
conda run -n gymnastic python -m fuse --methods sim3_face_stable_smooth_kpt
```

另列完整九方法实验矩阵命令，并标注耗时和存储开销高于单推荐方法：

```bash
conda run -n gymnastic python -m fuse
```

加入 `## 自定义数据根目录`，准确说明各入口的不同路径机制，并给出：

```bash
export GYMNASTICS_DATA_ROOT=/path/to/gymnastics
conda run -n gymnastic python -m SAM3Dbody.main
conda run -n gymnastic python -m split_cycle.main --raw-root /path/to/gymnastics/raw --kpt-root /path/to/gymnastics/sam3d_body_results
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle
conda run -n gymnastic python -m fuse --sam3d-root /path/to/gymnastics/sam3d_body_results --triangulated-root /path/to/gymnastics/sam3d_triangulated/person
```

加入 `## 常见故障` 表格，至少覆盖：原始视频缺失、SAM3D 没有 `.npz`、对齐记录缺失、相机标定文件缺失、冒烟结果被误当完整结果、融合指标为空、错误 Conda 环境。

最后加入 `## 旧流程`，只链接 `legacy/prepare_dataset/` 和 `configs/legacy/prepare_dataset.yaml`，明确它们不属于当前流程。

- [ ] **Step 5: 验证运行手册的命令接口和静态内容**

Run:

```bash
conda run -n gymnastic python -m split_cycle.main --help
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle --help
conda run -n gymnastic python -m fuse --help
```

Expected: 三条命令退出码均为 `0`；帮助信息分别包含 `--person`，并包含 `--max-cycles`/`--max-frames` 或 `--methods` 等文档使用的参数。

Run:

```bash
rg -n "drivefusion|Runbook|One-Person|Run classifier" docs/runbook.md
```

Expected: 无输出，退出码为 `1`。

Run:

```bash
rg -n "gymnastic|offset_side_to_face|sim3_face_stable_smooth_kpt|--max-cycles|--methods" docs/runbook.md
```

Expected: 所有关键词都有命中。

Run:

```bash
git diff --check -- docs/runbook.md
```

Expected: 无输出，退出码为 `0`。

- [ ] **Step 6: 提交运行手册**

```bash
git add docs/runbook.md
git commit -m "docs: 添加中文数据处理运行手册"
```

提交前运行 `git diff --cached --name-only`，预期只显示 `docs/runbook.md`。

---

### Task 3: 更新 README 入口并进行跨文档验证

**Files:**
- Modify: `README.md:22-32`
- Modify: `README.md:56-120`
- Verify: `docs/current_pipeline.md`
- Verify: `docs/runbook.md`

**Interfaces:**
- Consumes: Task 1 的流程概览和 Task 2 的运行手册。
- Produces: README 中正确的流程顺序、中文文档入口及全局一致性验证结果。

- [ ] **Step 1: 记录 README 旧描述的失败基线**

Run:

```bash
rg -n "### 2. Fuse|### 3. Split|rebuilds face/side temporal alignment" README.md
```

Expected: 命中旧顺序和“fuse 重新建立时间对齐”的过时描述。

- [ ] **Step 2: 修正 README 的流程概览和使用顺序**

使用 `apply_patch` 完成以下精确改动：

- `Pipeline Overview` 的活动顺序改为 `SAM3Dbody`、`split_cycle`、`triangulation`、`fuse`、`analysis`、`project/train`；`camera_calibration` 保留为支持模块。
- 在 `Usage` 开头加入中文入口：

```markdown
中文文档：

- [当前数据处理流程](docs/current_pipeline.md)
- [数据处理运行手册](docs/runbook.md)
- [模块职责](docs/modules.md)
```

- `Usage` 的命令顺序改为：

```bash
conda run -n gymnastic python -m SAM3Dbody.main
conda run -n gymnastic python -m split_cycle.main
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle
conda run -n gymnastic python -m fuse --methods sim3_face_stable_smooth_kpt
conda run -n gymnastic python -m project.train.train
```

- 将 fuse 说明改为“读取 split-cycle 保存的时间偏移，运行融合方法，并与三角化伪真值比较”；不得保留“rebuilds temporal alignment”。
- 将 triangulation 从“optional support module”移入活动数据流程；camera calibration 仍保留在支持模块。
- README 其余英文介绍、安装说明和项目结构保持不变，避免无关翻译或重排。

- [ ] **Step 3: 执行跨文档一致性检查**

Run:

```bash
rg -n "drivefusion|rebuilds face/side temporal alignment|fuse[[:space:]]*->[[:space:]]*split_cycle" README.md docs/current_pipeline.md docs/runbook.md
```

Expected: 无输出，退出码为 `1`。

Run:

```bash
rg --pcre2 -n "conda run -n (?!gymnastic)" README.md docs/current_pipeline.md docs/runbook.md
```

Expected: 无输出，退出码为 `1`。

Run:

```bash
test -f docs/current_pipeline.md
test -f docs/runbook.md
test -f docs/modules.md
test -f triangulation/README.md
test -f SAM3Dbody/main.py
test -f split_cycle/main.py
test -f triangulation/sam3d_from_split_cycle.py
test -f fuse/__main__.py
test -f project/train/train.py
test -f configs/sam3d_body.yaml
test -f configs/sam3d_triangulation.yaml
```

Expected: 所有命令均无输出，退出码为 `0`。

Run:

```bash
rg -n "SAM3D-Body|split_cycle|triangulation|sim3_face_stable_smooth_kpt" README.md docs/current_pipeline.md docs/runbook.md
```

Expected: 三个目标文件均覆盖正确流程和推荐方法，README 链接到两份中文文档。

Run:

```bash
git diff --check -- README.md docs/current_pipeline.md docs/runbook.md
```

Expected: 无输出，退出码为 `0`。

- [ ] **Step 4: 检查提交范围并提交 README**

```bash
git status --short
git diff -- README.md
git add README.md
git diff --cached --name-only
git commit -m "docs: 更新数据流程文档入口"
```

Expected: 暂存列表只包含 `README.md`；`AGENTS.md` 仍保持用户原有的未暂存修改。

- [ ] **Step 5: 最终只读验收**

```bash
git log -4 --oneline
git status --short
```

Expected: 日志依次包含本计划的三次文档提交；工作区只显示用户原有的 ` M AGENTS.md`，没有本任务遗留的未提交文件。
