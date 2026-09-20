# 数据处理流程与运行手册

本文档说明当前活动流水线的原理、数据流、目录约定，以及如何从头处理同一人的
`face + side` 双视角视频。模块划分见 [modules.md](modules.md)，三角化细节见
[triangulation.md](triangulation.md)，模型训练见 [cycle_aware_fusion.md](cycle_aware_fusion.md)。

## 适用范围

流水线以同一人的 `face` 和 `side` 双视角视频为输入。两个视角需要属于同一次动作采集，
并使用相同的人物编号。`python -m pose_estimation run` 会检查两个视角是否齐全；缺少任一
视角时该人物会被跳过。单视角关键点无法完成后续的时间对齐、三角化或融合。

## 端到端流程

```text
$GYMNASTICS_DATA_ROOT/raw/person/<id>/ID<id>_{face,side}.MOV
  -> ① pose_estimation   SAM3D-Body 逐帧提取 2D/3D 关键点
  -> ② cycle_alignment   对齐 face/side 时间轴并切分动作周期
  -> ③ pseudo_gt         先估计逐人相机外参，再用两个视角的 2D 关键点三角化出 3D 伪真值
  -> ④ fusion            训练 cycle-aware 双视角融合模型；确定性方法作为对照矩阵
  -> fusion.analysis     将融合结果与三角化伪真值比较，并做队列/重复周期统计
```

| 阶段 | 输入 | 主要处理 | 输出 |
|---|---|---|---|
| `pose_estimation` | `face`/`side` 原始视频 | 逐帧人体推理，提取 2D/3D 关键点 | `sam3d_body_results/person/<id>/<view>/*_sam3d_body.npz` |
| `cycle_alignment` | 原始视频和 SAM3D 关键点 | 时间对齐、偏移选择、周期切分、周期中点标注 | `local/runs/split_cycle/person_<id>/alignment_record_<id>.json`、周期视频、`local/runs/cycle_records/` |
| `pseudo_gt` | 对齐记录、两个视角的 2D 关键点、逐人外参 | 按周期进行两视角三角化 | `sam3d_triangulated/person/person_<id>/cycle_<idx>/` |
| `fusion` | 对齐记录、周期记录和两个视角的 3D 关键点 | cycle-aware 模型训练/推理；确定性对照方法 | `local/runs/cycle_aware/`、`local/runs/fuse_experiments/<method>/person_<id>/fused_sequence.npz` |
| `fusion.analysis` | 融合结果和三角化伪真值 | 计算 MPJPE 等指标并生成报告 | `local/runs/analysis/` 和融合指标 CSV |

## 关键规则

- `cycle_alignment` 生成的对齐记录是三角化和融合共同使用的时间基准。融合必须读取记录中的
  `offset_side_to_face`，不会回退到新的关键点 DTW 偏移估算。
- 融合以 `face` 为参考视角，将 `side` 的 3D 关键点变换到 `face` 坐标系。
- 三角化结果只用于评估。模型训练不得读取三角化数据（伪目标、权重、checkpoint 选择、损失
  都不能用），只有评估层可以读取。
- 三角化的相机内参来自棋盘格标定，外参由 `python -m pseudo_gt estimate-extrinsics` 从数据
  估计。两视角几何是无尺度的，重投影误差无法反映基线长度是否正确；伪真值的米制尺度来自
  SAM3D 单目 3D，未经器械标定。
- 融合指标默认使用 `similarity` 对齐（逐序列拟合含尺度的 Sim3），因此方法排序和相对比较对
  伪真值的尺度误差免疫；只有绝对毫米值与尺度误差成正比。报告绝对精度时应同时给出
  `fusion.analysis.normalize_by_body_scale` 产出的无量纲指标（误差占体长百分比）。

## 确定性对照方法

确定性实验矩阵（`python -m fusion deterministic`）是所有模型的对照基线。其中推荐方法是
`avg_body_current`：

```text
将 face 和 side 关键点各自转换到身体坐标系（骨盆居中、朝向归一化）
  -> 在身体坐标系下平均两个视角
  -> 用 face 的骨盆位置和朝向变换回世界坐标系
```

该方法在重新生成的三角化伪真值上逐人 MPJPE 均值 64.05 mm，在 137 人中的 69-100% 上优于
其余每一种无泄漏方法（Wilcoxon 配对检验，Holm 校正后 p < 1e-4）。
`sim3_face_stable_joint_weight` 的数值更低（63.48 mm），但它的逐关节权重来自三角化伪真值
本身，属于在评估目标上拟合，不能作为推荐依据。

## 主要目录

| 数据或结果 | 默认路径 |
|---|---|
| 原始双视角视频 | `$GYMNASTICS_DATA_ROOT/raw/person` |
| SAM3D-Body 逐帧结果 | `$GYMNASTICS_DATA_ROOT/sam3d_body_results/person` |
| 时间对齐和周期切分 | `local/runs/split_cycle` |
| 周期记录（含中点） | `local/runs/cycle_records` |
| 三角化 3D 伪真值 | `$GYMNASTICS_DATA_ROOT/sam3d_triangulated/person` |
| 确定性融合结果 | `local/runs/fuse_experiments` |
| cycle-aware 训练输出 | `local/runs/cycle_aware` |

## 运行环境

从仓库根目录运行命令，并使用项目的 conda 环境（见 [README](../README.md#environment)）。
所有 `python -m ...` 命令都需要 `PYTHONPATH=src`（或 `pip install -e .`）。

数据根目录只在 `src/common/paths.py` 定义一次：设置 `GYMNASTICS_DATA_ROOT` 即可覆盖，
否则使用第一个存在的已知机器路径。所有阶段的默认路径（Hydra 配置和 argparse 默认值）
都由它派生，因此切换数据根目录只需：

```bash
export GYMNASTICS_DATA_ROOT=/path/to/gymnastics
```

自定义 `cycle_alignment --log-root` 时，后续三角化配置中的 `paths.split_cycle_root` 和融合
命令的 `--split-root` 必须指向同一目录。

## 单人完整处理示例

以下命令以 person `46` 为例。处理其他人物时，将所有 `46` 替换为目标人物编号。

### 1. 检查原始视频

```bash
ls $GYMNASTICS_DATA_ROOT/raw/person/46/ID46_face.MOV $GYMNASTICS_DATA_ROOT/raw/person/46/ID46_side.MOV
```

两个文件都应存在。

### 2. 运行 SAM3D-Body

```bash
python -m pose_estimation run infer.person_list=[46] infer.gpu=[0] infer.workers_per_gpu=1
```

默认配置 `infer.person_list=[-1]` 表示处理全部人物。主要输出：

```text
$GYMNASTICS_DATA_ROOT/sam3d_body_results/person/46/{face,side}/*_sam3d_body.npz
local/runs/sam3d/46/{face,side}/visualization/
local/runs/sam3d/person_logs/46.log
```

分别统计两个视角的逐帧结果，两个计数都应大于 `0`；计数差异较大时先查看人物日志以及两个
原始视频的帧数：

```bash
find $GYMNASTICS_DATA_ROOT/sam3d_body_results/person/46/face -name '*_sam3d_body.npz' | wc -l
find $GYMNASTICS_DATA_ROOT/sam3d_body_results/person/46/side -name '*_sam3d_body.npz' | wc -l
```

### 3. 时间对齐和周期切分

```bash
python -m cycle_alignment align --person 46 --threads 1
```

主要输出：

```text
local/runs/split_cycle/person_46/alignment_record_46.json
local/runs/split_cycle/person_46/theta_unwrap.png
local/runs/split_cycle/person_46/{face,side}/cycle_*.mp4
```

关键字段 `offset_side_to_face` 表示侧面序列相对于正面序列的帧偏移。模型训练还需要带
周期中点（转身帧）的周期记录：

```bash
python -m cycle_alignment cycles private
python -m cycle_alignment cycles index     # local/runs/cycle_records/index.json
```

### 4. 三角化生成 3D 伪真值

三角化依赖两个视角的 SAM3D 2D 关键点、`alignment_record_46.json` 中的周期帧记录和时间偏移、
`src/configs/pseudo_gt/sam3d_triangulation.yaml` 指向的 face/side 内参标定文件，以及
`local/runs/analysis/extrinsics/estimated_extrinsics.json` 中的逐人外参。

相机在不同拍摄场次之间被重新摆放过，配置里的 `camera_position` 合成布局对所有人共用一套
位姿，留出帧重投影误差中位约 21 px、最差 57 px；逐人估计后降到约 6 px。若外参文件不存在，
三角化会直接报错：

```bash
python -m pseudo_gt estimate-extrinsics
python -m fusion.analysis.reports.compare_extrinsics    # 可选：对比新旧外参的三角化质量
```

先处理一个周期的两个帧做冒烟检查，确认无误后再运行完整命令：

```bash
python -m pseudo_gt triangulate --person 46 --max-cycles 1 --max-frames 2
python -m pseudo_gt triangulate --person 46
```

主要输出：

```text
$GYMNASTICS_DATA_ROOT/sam3d_triangulated/person/person_46/summary.json
$GYMNASTICS_DATA_ROOT/sam3d_triangulated/person/person_46/cycle_000/summary.json
$GYMNASTICS_DATA_ROOT/sam3d_triangulated/person/person_46/cycle_000/joints_3d/*.json
$GYMNASTICS_DATA_ROOT/sam3d_triangulated/person/person_46/cycle_000/joints_3d_sequence.npz
$GYMNASTICS_DATA_ROOT/sam3d_triangulated/person/person_46/cycle_000/visualization/*.png
$GYMNASTICS_DATA_ROOT/sam3d_triangulated/person/person_46/cycle_000/cycle_000_3d.mp4
```

每个周期重点检查 `processed_frames`、`missing_pairs`（应尽量为 `0`）、
`face_reprojection_error_mean_px` 和 `side_reprojection_error_mean_px`。重投影误差较大说明
三维骨架投回 face/side 图像时一致性差，即使骨架动画看起来连续，也应继续检查对齐偏移或
相机标定。

### 5. 运行确定性对照方法

```bash
python -m fusion deterministic --person 46 --methods avg_body_current
```

主要输出：

```text
local/runs/fuse_experiments/avg_body_current/person_46/fused_sequence.npz
local/runs/fuse_experiments/avg_body_current/person_46/config.json
local/runs/fuse_experiments/metrics_by_person.csv
local/runs/fuse_experiments/metrics_by_joint.csv
```

融合必须找到 `local/runs/split_cycle/person_46/alignment_record_46.json`。如果缺少三角化
周期，程序仍可保存 `fused_sequence.npz`，但不会产生该人物的有效伪真值评估指标。

### 6. 训练 cycle-aware 模型

```bash
python -m fusion train experiment=smoke                  # 冒烟
python -m fusion train data=gymnastics data.fold_json=src/configs/fusion/folds/gymnastics/fold_01.json
```

5 折的集群提交脚本在 `pegasus/`，详见 [cycle_aware_fusion.md](cycle_aware_fusion.md)。

### 7. 生成或检查分析结果

```bash
python -m fusion.analysis.reports.generate_results_report   # 三角化质量报告
python -m fusion analyze                                    # 融合与伪真值比较
```

报告输出在 `local/runs/analysis/triangulated_results/` 和 `local/runs/analysis/`；融合评估的
MPJPE、median、p95 和最大误差位于 `local/runs/fuse_experiments/metrics_by_person.csv` 和
`metrics_by_joint.csv`。

## 全数据集处理

以下命令会处理数据目录中发现的全部人物，占用较长 GPU/CPU 时间和存储空间。运行前先用单人
命令验证数据和标定。

```bash
python -m pose_estimation run
python -m cycle_alignment align
python -m cycle_alignment cycles private && python -m cycle_alignment cycles index
python -m pseudo_gt estimate-extrinsics
python -m pseudo_gt triangulate
python -m fusion deterministic                 # 全部方法；加 --methods avg_body_current 只跑推荐方法
```

## 常见故障

| 现象 | 检查内容 |
|---|---|
| SAM3D-Body 跳过人物 | `raw/person/<id>/` 下是否同时存在文件名包含 `face` 和 `side` 的视频。 |
| SAM3D-Body 运行后没有 `.npz` | `local/runs/sam3d/person_logs/<id>.log`，以及 GPU、模型 checkpoint 和输入视频。 |
| `cycle_alignment` 找不到人物 | `sam3d_body_results/person/<id>/face` 和 `side` 是否存在逐帧结果。 |
| 三角化跳过人物 | `local/runs/split_cycle/person_<id>/alignment_record_<id>.json` 是否存在。 |
| 三角化无法加载相机 | `src/configs/pseudo_gt/sam3d_triangulation.yaml` 中 face/side 标定文件是否存在；是否已运行 `estimate-extrinsics`。 |
| 三角化结果只有少量帧 | 是否只运行了 `--max-cycles 1 --max-frames 2`；正式使用前重新运行完整命令。 |
| 融合报对齐记录缺失 | 先运行 `cycle_alignment align`，并确认 `--split-root` 指向正确目录。 |
| 融合指标为空 | 对应人物是否存在三角化 `cycle_*` 目录，以及周期帧是否能与融合序列匹配。 |
| 训练报缺少周期记录 | 先运行 `cycle_alignment cycles private` 与 `cycles index`。 |
| 找不到模块或依赖 | 是否从仓库根目录运行、`PYTHONPATH=src` 是否设置、conda 环境是否正确。 |

## 旧流程

旧的 DPT、RAFT、YOLO 和 Detectron2 数据准备代码与动作分类代码已于 2026-09 从仓库移除；
需要时从 git 历史（提交 7286a02 之前）恢复。旧的 rotation-aware 模型已归档到
`src/fusion/archive/rotation_aware/`，只用于复现论文表格，见 [archive/](archive/)。
