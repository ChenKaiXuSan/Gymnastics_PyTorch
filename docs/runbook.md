# 数据处理运行手册

本文档说明如何处理同一人的 `face + side` 双视角视频。流水线原理、数据流和模块职责见[当前数据处理流程](current_pipeline.md)。

## 运行环境

从仓库根目录运行命令，并统一使用 `gymnastic` Conda 环境：

```bash
conda run -n gymnastic ...
```

默认数据根目录是：

```text
/home/data/xchen/gymnastics
```

`GYMNASTICS_DATA_ROOT` 会被 SAM3D-Body 和三角化的 Hydra 配置读取。`split_cycle` 和 `fuse` 使用独立的命令行路径参数；使用自定义数据根目录时，还需要显式传入它们的输入路径。具体示例见[自定义数据根目录](#自定义数据根目录)。

## 单人完整处理示例

以下命令以 person `46` 为例。处理其他人物时，需要将路径和参数中的所有 `46` 替换为目标人物编号。

### 1. 检查原始视频

完整流程需要同一人的两个视频：

```text
/home/data/xchen/gymnastics/raw/person/46/ID46_face.MOV
/home/data/xchen/gymnastics/raw/person/46/ID46_side.MOV
```

检查文件：

```bash
ls /home/data/xchen/gymnastics/raw/person/46/ID46_face.MOV /home/data/xchen/gymnastics/raw/person/46/ID46_side.MOV
```

两个文件都应存在。当前 `SAM3Dbody.main` 会检查 `face` 和 `side` 是否齐全，缺少任一视角时会跳过该人物。

### 2. 运行 SAM3D-Body

为 person `46` 的两个视角生成逐帧关键点：

```bash
conda run -n gymnastic python -m SAM3Dbody.main infer.person_list=[46] infer.gpu=[0] infer.workers_per_gpu=1
```

可通过 `infer.gpu=[0]` 选择 GPU。默认配置 `infer.person_list=[-1]` 表示处理全部人物。

主要输出：

```text
/home/data/xchen/gymnastics/sam3d_body_results/person/46/face/*_sam3d_body.npz
/home/data/xchen/gymnastics/sam3d_body_results/person/46/side/*_sam3d_body.npz
/home/data/xchen/gymnastics/sam3d_body_results/logs/46/face/visualization/
/home/data/xchen/gymnastics/sam3d_body_results/logs/46/side/visualization/
/home/data/xchen/gymnastics/sam3d_body_results/logs/person_logs/46.log
```

分别统计两个视角的逐帧结果：

```bash
find /home/data/xchen/gymnastics/sam3d_body_results/person/46/face -type f -name '*_sam3d_body.npz' | wc -l
find /home/data/xchen/gymnastics/sam3d_body_results/person/46/side -type f -name '*_sam3d_body.npz' | wc -l
```

两个计数都应大于 `0`。计数差异较大时，应先查看人物日志以及两个原始视频的帧数。

### 3. 时间对齐和周期切分

对齐 person `46` 的 face/side 时间轴并切分动作周期：

```bash
conda run -n gymnastic python -m split_cycle.main --person 46 --threads 1
```

主要输出：

```text
logs/split_cycle/person_46/alignment_record_46.json
logs/split_cycle/person_46/theta_unwrap.png
logs/split_cycle/person_46/face/cycle_*.mp4
logs/split_cycle/person_46/side/cycle_*.mp4
```

检查对齐记录：

```bash
sed -n '1,220p' logs/split_cycle/person_46/alignment_record_46.json
```

后续三角化和融合都使用该记录。关键字段 `offset_side_to_face` 表示侧面序列相对于正面序列的帧偏移；融合不会改用新估算的关键点 DTW 偏移。

### 4. 三角化生成 3D 伪真值

三角化依赖以下内容：

- 两个视角的 SAM3D 2D 关键点。
- `alignment_record_46.json` 中的周期帧记录和时间偏移。
- `configs/sam3d_triangulation.yaml` 指向的 face/side 相机标定文件。

可以先处理一个周期的两个帧，确认加载和标定流程可用：

```bash
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle --person 46 --max-cycles 1 --max-frames 2
```

冒烟检查只生成部分结果，不能作为最终数据。确认无误后必须运行完整命令：

```bash
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle --person 46
```

主要输出：

```text
/home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/summary.json
/home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/cycle_000/summary.json
/home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/cycle_000/joints_3d/*.json
/home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/cycle_000/joints_3d_sequence.npz
/home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/cycle_000/visualization/*.png
/home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/cycle_000/cycle_000_3d.mp4
```

查看人物摘要：

```bash
sed -n '1,220p' /home/data/xchen/gymnastics/sam3d_triangulated/person/person_46/summary.json
```

每个周期应重点检查：

```text
processed_frames
missing_pairs
face_reprojection_error_mean_px
side_reprojection_error_mean_px
```

`missing_pairs` 应尽量为 `0`。重投影误差较大说明三维骨架投回 face/side 图像时的一致性较差，即使骨架动画看起来连续，也应继续检查对齐偏移或相机标定。

### 5. 运行推荐融合方法

以 face 为参考，用推荐方法处理 person `46`：

```bash
conda run -n gymnastic python -m fuse --person 46 --methods sim3_face_stable_smooth_kpt
```

主要输出：

```text
logs/fuse_experiments/sim3_face_stable_smooth_kpt/person_46/fused_sequence.npz
logs/fuse_experiments/sim3_face_stable_smooth_kpt/person_46/config.json
logs/fuse_experiments/metrics_by_person.csv
logs/fuse_experiments/metrics_by_joint.csv
```

检查人物级指标：

```bash
sed -n '1,40p' logs/fuse_experiments/metrics_by_person.csv
```

融合必须找到 `logs/split_cycle/person_46/alignment_record_46.json`。如果缺少三角化周期，程序仍可保存 `fused_sequence.npz`，但不会产生该人物的有效伪真值评估指标。

### 6. 生成或检查分析结果

完成正式三角化后，刷新三角化质量报告：

```bash
conda run -n gymnastic python triangulation/tools/generate_results_report.py
```

报告输出：

```text
logs/analysis/triangulated_results/triangulated_results_report.md
logs/analysis/triangulated_results/triangulated_person_summary.csv
logs/analysis/triangulated_results/triangulated_cycle_details.csv
```

融合评估的 MPJPE、median、p95 和最大误差位于 `logs/fuse_experiments/metrics_by_person.csv` 和 `metrics_by_joint.csv`。

### 7. 可选：分类训练

需要进行动作分类实验时运行：

```bash
conda run -n gymnastic python -m project.train.train
```

分类训练使用已准备的动作数据、标签和人员级折叠映射。它不是生成三角化或融合 3D 关键点的必要步骤。

## 全数据集处理

以下命令会处理配置和数据目录中发现的全部人物，可能占用较长 GPU 时间、CPU 时间和存储空间。运行前应先用单人命令验证数据和标定。

按顺序执行：

```bash
conda run -n gymnastic python -m SAM3Dbody.main
conda run -n gymnastic python -m split_cycle.main
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle
conda run -n gymnastic python -m fuse --methods sim3_face_stable_smooth_kpt
```

如需运行全部九种融合方法的实验矩阵，使用：

```bash
conda run -n gymnastic python -m fuse
```

完整实验矩阵的耗时和存储开销高于只运行推荐方法。

## 自定义数据根目录

SAM3D-Body 和三角化配置读取 `GYMNASTICS_DATA_ROOT`：

```bash
export GYMNASTICS_DATA_ROOT=/path/to/gymnastics
conda run -n gymnastic python -m SAM3Dbody.main
conda run -n gymnastic python -m triangulation.sam3d_from_split_cycle
```

`split_cycle` 不读取该环境变量，需要显式指定 raw 和 SAM3D 结果目录：

```bash
conda run -n gymnastic python -m split_cycle.main --raw-root /path/to/gymnastics/raw --kpt-root /path/to/gymnastics/sam3d_body_results --log-root logs/split_cycle
```

`fuse` 也需要显式指定 SAM3D、三角化和切分记录目录：

```bash
conda run -n gymnastic python -m fuse --sam3d-root /path/to/gymnastics/sam3d_body_results --triangulated-root /path/to/gymnastics/sam3d_triangulated/person --split-root logs/split_cycle --methods sim3_face_stable_smooth_kpt
```

如果同时自定义 `--log-root`，后续三角化配置中的 `paths.split_cycle_root` 和融合命令的 `--split-root` 必须指向同一目录。

## 常见故障

| 现象 | 检查内容 |
|---|---|
| SAM3D-Body 跳过人物 | 检查 `raw/person/<id>/` 下是否同时存在文件名包含 `face` 和 `side` 的视频。 |
| SAM3D-Body 运行后没有 `.npz` | 查看 `sam3d_body_results/logs/person_logs/<id>.log`，并检查 GPU、模型 checkpoint 和输入视频。 |
| `split_cycle` 找不到人物 | 检查 `sam3d_body_results/person/<id>/face` 和 `side` 是否存在逐帧结果。 |
| 三角化跳过人物 | 检查 `logs/split_cycle/person_<id>/alignment_record_<id>.json`。 |
| 三角化无法加载相机 | 检查 `configs/sam3d_triangulation.yaml` 中 face/side 标定文件是否存在。 |
| 三角化结果只有少量帧 | 确认是否只运行了 `--max-cycles 1 --max-frames 2`；正式使用前重新运行完整命令。 |
| 融合报对齐记录缺失 | 先运行 `split_cycle`，并确认 `--split-root` 指向正确目录。 |
| 融合指标为空 | 检查对应人物是否存在三角化 `cycle_*` 目录，以及周期帧是否能与融合序列匹配。 |
| 找不到模块或依赖 | 确认从仓库根目录运行，并使用 `conda run -n gymnastic ...`。 |

## 旧流程

旧的 DPT、RAFT、YOLO 和 Detectron2 数据准备代码位于 `legacy/prepare_dataset/`，配置位于 `configs/legacy/prepare_dataset.yaml`。它们仅供参考，不属于当前活动流程。

更多三角化细节见[三角化说明](../triangulation/README.md)。
