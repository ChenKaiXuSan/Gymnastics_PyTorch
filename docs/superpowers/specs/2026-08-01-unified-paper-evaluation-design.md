# Sports Engineering 统一评估口径设计

## 目标

消除主文 Table 1、Table 2 和逐关节 Table 3 之间的评分协议差异。所有与
triangulated pseudo-reference 比较的论文结果统一使用：每个 cycle 拟合一次
similarity transform，随后对候选姿势和参考姿势做逐帧髋中心化，再计算关节
欧氏距离；先在每位参与者内部汇总，再跨参与者统计。

## 方案选择

采用“从 compact fused sequence 统一重评估”的方案。它比复用旧
`metrics_by_person.csv` 更慢，但能够保证 `avg_body_current`、
`extrinsic_r_average` 和 `extrinsic_r_quality_average` 经过完全相同的帧匹配、
有效点掩码、cycle 对齐和髋中心化逻辑。旧的 similarity-only 人级结果不再进入
论文表格。

## 数据流

1. 从固定的 split 文件读取137人集合和14人测试集合。
2. 对每个方法和参与者读取 `fused_sequence.npz`，使用 face/side frame map 与
   triangulated cycle 配对。
3. 共享评估函数同时产生人级 pooled MPJPE 和70关节指标，并在每一行写入
   `similarity_plus_hip_centering` 协议标签。
4. Table 2 只接受完整137人且协议标签一致的人级数据；Table 3 使用相同函数
   产生的14人关节数据。
5. 从统一人级结果重算 bootstrap 95% CI、配对 Wilcoxon、Holm 校正和改善人数。

## 防错约束

- 任何缺失协议标签、混合协议、缺人、重复 person-method 或非有限 MPJPE 都使
  表格生成失败。
- 137人 Table 2 的基线和两个外参方法必须来自同一次统一重评估。
- 论文 caption 明确写出 per-cycle similarity alignment 和 framewise hip
  centering；不允许再以笼统的“same-video pseudo-reference”代替协议说明。
- 旧 CSV 保留为历史实验产物，但不再作为论文生成器输入。

## 验证

- 单元测试证明共享函数同时返回正确的人级 pooled error 和关节级 error。
- 单元测试证明 Table 2 汇总拒绝 similarity-only 或混合协议。
- 重新生成137人结果两次并比较 CSV，确定确定性输出完全一致。
- 编译主文和 Online Resource，检查引用、溢出、数字锚点和提交压缩包。

