# 当前数据处理流程

## 适用范围

当前活动流水线以同一人的 `face` 和 `side` 双视角视频为输入。两个视角需要属于同一次动作采集，并使用相同的人物编号。

当前 `gymnastics sam3d` 数据集入口会检查两个视角是否齐全；缺少任一视角时，该人物会被跳过。即使通过其他方式取得单视角关键点，也不能完成本流水线的双视角时间对齐、三角化或融合。

## 端到端流程

```text
/home/data/xchen/gymnastics/raw/person/<id>/ID<id>_{face,side}.MOV
  -> SAM3D-Body 逐帧提取 2D/3D 关键点
  -> alignment 对齐 face/side 时间轴并切分动作周期
  -> triangulation 先估计逐人相机外参，再用两个视角的 2D 关键点生成 3D 伪真值
  -> fusion 对齐并融合两个视角的 3D 关键点
  -> analysis 将融合结果与三角化伪真值比较
  -> classification（可选的动作分类训练与评估）
```

## 各阶段说明

| 阶段 | 输入 | 主要处理 | 输出 |
|---|---|---|---|
| SAM3D-Body | `face`/`side` 原始视频 | 逐帧人体推理，提取 2D/3D 关键点 | `sam3d_body_results/person/<id>/<view>/*_sam3d_body.npz` |
| `alignment` | 原始视频和 SAM3D 关键点 | 时间对齐、偏移选择、周期切分 | `local/runs/split_cycle/person_<id>/alignment_record_<id>.json` 和周期视频 |
| `triangulation` | 对齐记录、两个视角的 2D 关键点、逐人外参 | 按周期进行两视角三角化 | `sam3d_triangulated/person/person_<id>/cycle_<idx>/` |
| `fusion` | 对齐记录和两个视角的 3D 关键点 | Sim3 对齐、双视角融合、时间平滑 | `local/runs/fuse_experiments/<method>/person_<id>/fused_sequence.npz` |
| `analysis` | 融合结果和三角化伪真值 | 计算 MPJPE 等指标并生成报告 | `local/runs/analysis/` 和融合指标 CSV |
| `classification` | 已准备的动作数据和人员级划分 | 分类训练与评估 | `local/runs/train/` 等训练输出 |

## 关键规则

- `alignment` 生成的对齐记录是三角化和融合共同使用的时间基准。
- `fusion` 必须读取对齐记录中的 `offset_side_to_face`，不会回退到新的关键点 DTW 偏移估算。
- 当前融合以 `face` 为参考视角，将 `side` 的 3D 关键点变换到 `face` 坐标系。
- 三角化结果是融合实验的 3D 伪真值。没有三角化结果时仍可生成融合序列，但无法得到有效的伪真值误差指标。
- 三角化的相机内参来自棋盘格标定，外参由 `gymnastics triangulate estimate-extrinsics` 从数据估计。两视角几何是无尺度的，重投影误差无法反映基线长度是否正确；伪真值的米制尺度来自 SAM3D 单目 3D，未经器械标定。
- 融合指标默认使用 `similarity` 对齐（逐序列拟合含尺度的 Sim3），因此**方法排序和相对比较对伪真值的尺度误差完全免疫**；只有绝对毫米值与尺度误差成正比。报告绝对精度时应同时给出 `gymnastics.analysis.normalize_by_body_scale` 产出的无量纲指标（误差占体长百分比），该指标与尺度无关。
- 分类训练是可选下游任务，不是生成三角化或融合关键点的必要步骤。

## 推荐融合方法

当前推荐方法是 `avg_body_current`，处理顺序为：

```text
将 face 和 side 关键点各自转换到身体坐标系（骨盆居中、朝向归一化）
  -> 在身体坐标系下平均两个视角
  -> 用 face 的骨盆位置和朝向变换回世界坐标系
```

该方法在身体坐标系下融合，避免了两视角世界坐标系尺度和朝向差异带来的配准误差。

该推荐基于重新生成的三角化伪真值：逐人 MPJPE 均值 64.05 mm，在 137 人中的 69-100%
上优于其余每一种无泄漏方法（Wilcoxon 配对检验，Holm 校正后 p < 1e-4）。

`sim3_face_stable_joint_weight` 的数值更低（63.48 mm），但它的逐关节权重是从三角化
伪真值本身估计的，而评估用的又是同一批帧，属于在评估目标上拟合，指标偏乐观，不能作为
推荐依据。

## 主要目录

| 数据或结果 | 默认路径 |
|---|---|
| 原始双视角视频 | `/home/data/xchen/gymnastics/raw/person` |
| SAM3D-Body 逐帧结果 | `/home/data/xchen/gymnastics/sam3d_body_results/person` |
| 时间对齐和周期切分 | `local/runs/split_cycle` |
| 三角化 3D 伪真值 | `/home/data/xchen/gymnastics/sam3d_triangulated/person` |
| 融合实验结果 | `local/runs/fuse_experiments` |
| 人员级交叉验证划分 | `/home/data/xchen/gymnastics/index_mapping/camera_pairs_by_person_folds` |
| 训练结果 | `local/runs/train` |

## 入口命令

所有项目命令默认使用 `gymnastic` Conda 环境，并从仓库根目录运行。

1. 提取 SAM3D-Body 关键点：

   ```bash
   conda run -n gymnastic gymnastics sam3d
   ```

2. 对齐双视角并切分周期：

   ```bash
   conda run -n gymnastic gymnastics align
   ```

3. 估计逐人相机外参（三角化的前置步骤，相机在不同场次间被重新摆放过）：

   ```bash
   conda run -n gymnastic gymnastics triangulate estimate-extrinsics
   ```

4. 生成三角化 3D 伪真值：

   ```bash
   conda run -n gymnastic gymnastics triangulate
   ```

5. 运行推荐融合方法：

   ```bash
   conda run -n gymnastic gymnastics fuse deterministic --methods avg_body_current
   ```

6. 刷新三角化质量报告：

   ```bash
   conda run -n gymnastic python -m gymnastics.analysis.reports.generate_results_report
   ```

7. 可选：训练和评估分类模型：

   ```bash
   conda run -n gymnastic gymnastics classify
   ```

单个人物的完整命令、预期输出和故障排查见[数据处理运行手册](runbook.md)。

## 旧流程

旧的 DPT、RAFT、YOLO 和 Detectron2 数据准备代码保存在 `legacy/prepare_dataset/`，配置位于 `configs/legacy/prepare_dataset.yaml`。它们仅供参考，不属于当前 SAM3D-Body 优先的活动流程。

## 相关文档

- [数据处理运行手册](runbook.md)
- [模块职责](modules.md)
- [三角化说明](triangulation.md)
