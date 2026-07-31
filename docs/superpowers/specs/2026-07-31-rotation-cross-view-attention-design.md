# 身体回旋引导的双视角交叉注意力自监督融合设计

**日期：** 2026-07-31  
**状态：** 待作者确认  
**范围：** 在现有 A6 自监督旋转感知融合主线上增加双向 cross-view attention，并增加一个不使用显式身体回旋信息的干净对照。

## 1. 研究目标

本扩展研究下列问题：

> 身体轴向回旋特征能否引导正面与侧面视角之间的特征交换，从而在不使用三角化三维标签训练的条件下改善双视角三维姿态融合？

需要分别回答两个子问题：

1. 在身体回旋建模保持不变时，cross-view attention 是否优于 A6 的对称 MLP 融合？
2. 在 cross-view attention 结构保持不变时，显式身体回旋特征及其自监督约束是否带来额外收益？

本研究不预设新增模型一定优于 A6。A10 或 A11 的负结果也必须保留并解释。

## 2. 方法与实验编号

| 方法 | 显式回旋输入 | 回旋自监督损失 | 双向 cross-view attention | 作用 |
|---|---:|---:|---:|---|
| A6 | 是 | 是 | 否 | 当前完整自监督主线 |
| A10 | 是 | 是 | 是 | 目标方法：身体回旋引导的跨视角特征交换 |
| A11 | 否 | 否 | 是 | 干净对照：无显式身体回旋条件的跨视角特征交换 |

A7--A9 已用于已有的躯干回旋扩展，不能覆盖或重新解释。A10 和 A11 必须作为新模型保存到独立 run ID 下。

## 3. A10 架构

### 3.1 输入与规范化

沿用现有数据流程：

```text
对齐后的 face/side SAM3D 3D 关键点
  -> 身体坐标系规范化
  -> 每视角姿态与质量特征
  -> 共享视角编码器
  -> 双向 cross-view attention
  -> 视角交换不变的对称融合
  -> 现有 TCN
  -> bounded residual
  -> 融合三维关键点
```

身体坐标系规范化对 A6、A10 和 A11 保持一致，以免把坐标系变化与 attention 效果混在一起。

### 3.2 回旋条件化的视角编码

A10 的每个关节 token 继续使用 A6 的观测信息：

- 三维关节坐标；
- 三维速度；
- 关节及速度有效性；
- 每帧视角质量分数；
- 躯干旋转矩阵；
- 轴向回旋角的正弦与余弦；
- 轴向角速度；
- 轴向角加速度；
- 躯干旋转有效性。

face 和 side 必须使用同一个 `SharedViewEncoder`，不能引入视角专属参数。

### 3.3 双向 cross-view attention

每一帧的关节作为 token。对于隐藏维度 \(C=128\)，使用一层四头注意力：

\[
C_{f\leftarrow s}
=\operatorname{MHA}
\left(
Q=\operatorname{LN}(E_f),
K=\operatorname{LN}(E_s),
V=\operatorname{LN}(E_s)
\right),
\]

\[
C_{s\leftarrow f}
=\operatorname{MHA}
\left(
Q=\operatorname{LN}(E_s),
K=\operatorname{LN}(E_f),
V=\operatorname{LN}(E_f)
\right).
\]

两个方向共用同一个 attention 模块、LayerNorm 和输出参数。attention dropout 固定为 0，以避免随机 dropout 破坏训练时的结构对称性。交换后的表示为：

\[
E'_f=E_f+C_{f\leftarrow s},\qquad
E'_s=E_s+C_{s\leftarrow f}.
\]

实现时将 \([B,T,J,C]\) 重排为 \([B T,J,C]\)，仅在同一帧的关节集合内交换信息，不在 attention 中同时建模时间。时间关系继续由现有 TCN 负责。

调用 attention 时必须设置 `need_weights=False`，以避免保存完整的注意力矩阵。论文诊断若需要注意力分布，只能在独立的小批量推理路径中显式开启。

### 3.4 有效性掩码

- face 查询由 `valid_face` 控制；
- side 查询由 `valid_side` 控制；
- face 到 side 的 key/value padding mask 使用 `valid_side`；
- side 到 face 的 key/value padding mask 使用 `valid_face`；
- 最终可学习残差仍只作用于两视角共同有效的目标关节；
- 若某帧不存在任何有效 key，必须使用安全的零占位避免 softmax 产生 NaN，并在 attention 后把该帧输出清零。

attention 模式下，每个视角的编码器先保留该视角自身有效的 token，使一个共同有效的目标关节能够读取另一视角中其他有效关节的信息。A6 的原有 `effective_mask` 路径保持不变。

### 3.5 对称融合与输出

attention 后继续使用交换不变的统计量：

\[
E_{\mathrm{sym}}=
\left[
\frac{E'_f+E'_s}{2},
\left|E'_f-E'_s\right|,
E_{\mathrm{cross}}
\right].
\]

其中 \(E_{\mathrm{cross}}\) 是现有跨视角差异编码，包括坐标、角度、旋转、躯干位移和有效性差异。随后沿用现有 `fuse_projection`、逐关节 TCN、`delta_head` 和每关节有界残差。

只要双向 attention 共享参数、后续只使用对称统计量，A10 在交换 face/side 输入后应保持相同输出，允许浮点误差范围内的差异。

## 4. A11：仅 cross-view 引导的干净对照

A11 与 A10 使用完全相同的：

- attention 层数；
- attention heads；
- 隐藏维度；
- 参数量；
- TCN 和残差头；
- 数据划分、随机种子、batch size、学习率和训练轮数。

为了做到“无显式身体回旋引导”，A11 将下列输入置零，同时保留对应网络维度和参数，以控制模型容量：

- 躯干旋转矩阵；
- 回旋角正弦与余弦；
- 轴向角速度和角加速度；
- 躯干旋转有效性；
- 跨视角角度差及其有效性；
- 跨视角旋转距离及其有效性。

A11 禁用下列回旋相关损失：

- circular axial-rotation consistency；
- SO(3) rotation consistency；
- complete-cycle rotation-ROM consistency。

A11 保留通用自监督目标：

- corruption recovery；
- high-consensus identity；
- trial-level bone length；
- local rigidity；
- adaptive temporal acceleration；
- minimal residual。

身体坐标系规范化仍然保留，因为它是三个方法共享的几何预处理。论文中 A11 必须称为：

> cross-view attention without explicit body-rotation conditioning

不能称为“完全不使用任何旋转操作”。

A10 与 A11 的差异同时包含“显式回旋输入”和“回旋自监督损失”。因此 \(A10-A11\) 表示完整回旋引导的总体贡献，不能被解释成单独某一个回旋特征的贡献。

## 5. 自监督与信息隔离

A10 继续使用 A6 的完整自监督目标。A11 使用第 4 节声明的通用目标子集。

三角化三维姿态只能由评价层读取，不能用于：

- 模型输入或伪目标；
- attention 权重或掩码；
- 损失权重；
- checkpoint 选择；
- early stopping；
- 超参数选择。

训练、验证和测试必须按 person 划分。checkpoint 只依据验证集自监督分数选择。评价阶段才加载三角化 pseudo-GT。

## 6. 兼容性与配置

实现必须满足：

1. `RotationAwareFusionModel` 默认关闭 cross-attention，使现有 A4--A9 架构和 checkpoint 保持兼容；
2. A6 在相同输入和权重下的输出与修改前一致；
3. checkpoint 元数据记录 `cross_attention`、`attention_heads` 和 `rotation_conditioning`；
4. 推理时从 checkpoint 元数据重建准确架构，命令行指定的 ablation 必须与 checkpoint 一致；
5. A10/A11 纳入训练、推理、评价和结果发现注册表；
6. 新增独立配置 `configs/fusion/rotation_aware_cross_attention.yaml`，不覆盖已有 A6 生产配置；
7. A10/A11 的 run ID 必须包含 ablation、batch size、epoch 和 seed。

## 7. 测试设计

实现遵循测试驱动流程，至少覆盖：

- attention heads 必须整除隐藏维度；
- 双向 attention 的输入输出形状；
- 部分缺失关节和全空 key 帧不会产生 NaN；
- A10 和 A11 的 view-swap error 接近零；
- attention 参数能接收到有限且非零的梯度；
- A11 的回旋输入通道确实不影响输出；
- A11 的回旋损失权重确实为零；
- A10 与 A11 参数量相同；
- 默认 A6 输出回归测试；
- 旧 A6 checkpoint 加载兼容性；
- A10/A11 checkpoint 能被推理路径准确重建；
- CPU 上一轮 tiny end-to-end 训练、推理和评价可完成。

## 8. 训练实验

### 8.1 阶段一：功能 smoke test

- 使用 tiny fixture 和 CPU 完成一轮端到端测试；
- 使用少量真实 trial、GPU、1 epoch 检查显存、吞吐、NaN、梯度和输出文件；
- 不把 smoke test 结果写入论文。

### 8.2 阶段二：主筛选实验

- split：现有 `fold_00`；
- seed：0；
- epochs：100；
- batch size：优先沿用 A6 的 64；若显存不足，只允许通过梯度累积保持等效 batch size，并记录实际 micro-batch；
- optimizer、学习率和 checkpoint 规则与 A6 一致；
- A10 与 A11 从头训练，不能以 A6 warm start 作为主结果；
- A6 使用已有结果，但必须核对其 split、seed、训练预算和评价协议完全匹配。

### 8.3 阶段三：重复与 OOF

若阶段二完成且训练稳定：

1. 在 `fold_00` 上为 A10/A11 运行 seeds 0、1、2；
2. 以 person 为单位汇总每个 seed 的结果，不把 cycle 或 seed 当作独立受试者；
3. 为老人/学生追加分析生成三折 out-of-fold 推理结果，使每个人只由未见过该人的模型预测；
4. 使用 OOF A10/A11 重复 person-level 组间分析和 person 内 cycle 变异分析。

## 9. 评价与统计

主要融合评价：

- held-out person-level MPJPE；
- 躯干轴向角误差；
- cycle ROM retention；
- peak angular-speed retention；
- bone-length coefficient of variation；
- temporal acceleration；
- fixed-corruption recovery；
- view-swap error；
- 参数量、峰值显存和推理时间。

主要配对比较：

\[
A10-A6:
\quad\text{在回旋建模固定时检验 cross-view attention 的增量价值},
\]

\[
A10-A11:
\quad\text{在 attention 固定时检验完整显式回旋引导的总体价值}.
\]

统计单位必须是 person。报告 person-level 配对差、bootstrap 置信区间、Wilcoxon 配对检验以及同一指标族内的 Holm 校正。多 seed 结果先在 person 内汇总，不能通过重复 seed 人为增加样本量。

年龄组分析继续区分：

- 老人标签组与学生标签组之间的 person-level 差异；
- 每个人内部不同 cycle 的变异。

年龄组结果是探索性、非因果和表征敏感的。不能把标签组差异解释为年龄造成的生理变化。

## 10. 结果判定

实验完成的判据不是“A10 必须胜出”，而是：

- 三个方法在相同信息边界和评价协议下可比较；
- A10/A11 无 NaN、无数据泄漏且保持视角交换不变；
- 主要及负面结果均有 person-level 统计；
- 能分别回答 attention 的增量价值与完整回旋引导的总体价值；
- 下游年龄组和 cycle 分析使用 OOF 表征，不使用训练内预测冒充泛化结果。

只有当 A10 在 held-out person-level 位置误差上优于或不劣于 A6，同时没有明显损害回旋保持、骨长稳定性和时间平滑性时，才考虑将 A10 升级为论文主方法。否则保留 A6 为主线，并把 A10/A11 作为有信息量的结构消融或负结果。
