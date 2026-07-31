# 外参对照与逐关节精度论文扩展设计

## 目标

在不改变 calibration-free 主线和既有训练结果的前提下，将已完成的显式相机外参融合纳入 Sports Engineering 稿件，并增加可复现的逐关节精度分析。

## 证据边界

- A6 继续作为 calibration-free 自监督主方法。
- `extrinsic_r_average` 和 `extrinsic_r_quality_average` 使用估计的 side-to-face 相机旋转，单列为 camera-assisted comparators，不并入 A0--A6 主消融。
- 外参确定性矩阵使用全部 137 人和 928 个 cycle；它没有可训练参数，因此允许进行全体配对描述性比较。
- 学习式逐关节主表固定使用 `paper_137_a6_split.json` 中的 14 名测试参与者。
- 三角化 pseudo-reference 只用于评价。外参估计与 pseudo-reference 共享上游双视角视频证据，因此结果只能解释为 same-video agreement，不能解释为独立绝对 3D 精度。

## 外参对照矩阵

正文新增独立的 camera-assisted comparison：

1. `avg_body_current`：无外参 body-frame average。
2. `extrinsic_r_average`：用估计的外参旋转将 side pose 映射至 face axes 后等权平均。
3. `extrinsic_r_quality_average`：相同外参旋转，加固定质量权重。

表格报告 137 人的 mean person MPJPE、标准差、相对 `avg_body_current` 的配对差值、participant-bootstrap 95% CI、Holm 校正 Wilcoxon p 值和改善人数。正文解释 calibration quality 与误差的关系，并保留逐人外参 129 人、cluster-consensus fallback 8 人的来源说明。

已完成的 G0--G5 相机特征分支不作为正向外参证据。它在 Online Resource 中作为负结果简述：正确相机参数未优于冻结 A6，且与错误 30 度相机参数几乎等价。

## 逐关节精度表

### 正文

正文列出分类和动作分析共用的 20 个主要 MHR70 关节：nose、双侧 shoulder、elbow、wrist、hip、knee、ankle、三类 foot landmarks，以及 neck。

列为：

- Face only (A0)
- Side only (A1)
- Body-frame average (A2)
- Complete self-supervised model (A6)
- Extrinsic-R average

每个单元格是先在每名测试参与者内部计算的 joint MPJPE，再对同一 14 名参与者求均值，单位 mm。每行加粗最低误差，最后一行给出这 20 个关节的宏平均。正文只做描述性解读，不对 20 个关节分别运行未预注册的显著性检验。

### Online Resource

补充材料使用同一 14 人、同一评价口径列出完整 70 个 MHR70 关节，并增加 `extrinsic_r_quality_average` 列。长表允许跨页，所有数值由生成脚本写入 LaTeX，避免手工抄录。

## 实现边界

- 新增一个聚焦于论文表格生成的分析模块，从现有 `metrics_by_joint.csv`、`metrics_by_person.csv` 和固定 split JSON 读取数据。
- 输出可审计 CSV 和 LaTeX：正文 20 关节表、Online Resource 70 关节表、外参汇总表。
- 生成器必须检查参与者集合严格等于 14 人、方法列完整、每个方法每名参与者有 70 个关节、所有 MPJPE 有限且单位从 repository coordinates 转成 mm。
- 论文只读取生成的 `.tex` 表格文件；正文中同步修改 Methods、Results、Discussion、Limitations、abstract 和 article counts 中受影响的陈述。
- 不重新训练模型，不覆盖已有实验产物，不修改原始参与者数据。

## 验证

- 测试先覆盖固定测试集过滤、20/70 关节顺序、参与者优先聚合、单位转换、缺失方法/关节拒绝和 LaTeX 最小值加粗。
- 重新生成表格后，将正文数字与源 CSV 交叉核对。
- 编译 `manuscript.tex` 与 `online_resource_1.tex`，检查未定义引用、LaTeX 错误和表格溢出警告。
- 运行现有融合矩阵测试，确认没有改变原有方法行为。

## 论文结论约束

允许的结论是：估计的相机旋转在同视频 pseudo-reference 评价中，相对无外参 body-frame average 带来小而一致的改善；这种改善在若干关节上可见。禁止的结论包括独立运动学准确性、标定优于 A6 的普遍性，以及由共享视频证据推导出的绝对三维有效性。
