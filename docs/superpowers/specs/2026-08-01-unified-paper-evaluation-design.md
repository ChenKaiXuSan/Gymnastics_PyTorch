# Sports Engineering 统一评估口径设计

## 目标

消除主文 Table 1、Table 2 和逐关节 Table 3 之间的评分协议与评估人群差异。
三个主文表格都只使用固定的14人 held-out test set。所有与 triangulated
pseudo-reference 比较的论文结果统一使用：每个 cycle 拟合一次 similarity
transform，随后对候选姿势和参考姿势做逐帧髋中心化，再计算关节欧氏距离；
先在每位参与者内部汇总，再跨参与者统计。

## 方案选择

采用“14人主比较 + 137人补充验证”的方案。Table 2 从 compact fused
sequence 统一重评估，并在主文中只汇总与 Table 1 相同的14人。完整137人结果
移到 Online Resource，作为无训练确定性方法的全队列描述性分析。该方案比复用旧
`metrics_by_person.csv` 更慢，但能够保证 `avg_body_current`、
`extrinsic_r_average` 和 `extrinsic_r_quality_average` 经过完全相同的帧匹配、
有效点掩码、cycle 对齐和髋中心化逻辑。旧的 similarity-only 人级结果不再进入
论文表格。

未采用两个替代方案。把 Table 1 扩到137人会把训练和验证参与者计入学习方法
性能，破坏独立测试；只保留14人并删除137人结果则会丢失确定性方法的全队列
稳定性信息。

## 数据流

1. 从固定的 split 文件读取137人集合和14人测试集合，并验证两者的包含关系。
2. 对每个方法和参与者读取 `fused_sequence.npz`，使用 face/side frame map 与
   triangulated cycle 配对。
3. 共享评估函数同时产生人级 pooled MPJPE 和70关节指标，并在每一行写入
   `similarity_plus_hip_centering` 协议标签。
4. 主文 Table 2 只接受固定14人且协议标签一致的人级数据；Table 3 使用同一
   14人和相同评估函数产生的关节数据。
5. Online Resource 的相机辅助表接受完整137人数据，并明确标为 secondary
   all-participant analysis，不与 held-out 学习结果直接比较。
6. 分别从14人与137人的统一人级结果重算 bootstrap 95% CI、配对 Wilcoxon、
   Holm 校正和改善人数。

## 防错约束

- 任何缺失协议标签、混合协议、缺人、重复 person-method 或非有限 MPJPE 都使
  表格生成失败。
- 主文 Table 2 的基线和两个外参方法必须覆盖与 Table 1 完全相同的14人；补充
  表必须覆盖完整137人。两层汇总都来自同一次统一重评估。
- Table 1 的 A2 与 Table 2 的 deterministic body-frame baseline 是两个独立
  物化输出，不强制逐点相等；稿件不得把它们写成同一行结果。两者使用同一测试
  人群、相同 cycle 匹配和相同评分协议。
- 论文 caption 明确写出 per-cycle similarity alignment 和 framewise hip
  centering；不允许再以笼统的“same-video pseudo-reference”代替协议说明。
- 旧 CSV 保留为历史实验产物，但不再作为论文生成器输入。

## 验证

- 单元测试证明共享函数同时返回正确的人级 pooled error 和关节级 error。
- 单元测试证明主文 Table 2 只汇总14人、补充表汇总137人，并拒绝
  similarity-only、混合协议或错误参与者集合。
- 重新生成14人与137人结果两次并比较 CSV，确定确定性输出完全一致。
- 编译主文和 Online Resource，检查引用、溢出、数字锚点和提交压缩包。

