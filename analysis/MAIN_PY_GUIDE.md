# analysis/main.py 详细说明文档（最新版）

本文档说明 [main.py](main.py) 当前版本实际做了什么、用了什么方法、每个结果有什么意义，供研发、实验复现实验与后处理分析使用。

---

## 1. 代码职责概览

[main.py](main.py) 是整套 3D 体操分析的主控脚本，核心职责是：

1. 读取每位人物的融合 3D 关键点序列。
2. 计算三项基础运动指标：
  - ツイスト（Twist）
  - 姿勢（Trunk Tilt）
  - 脱力（Wrist Lead）
3. 对三条时序做“去不平滑”稳健过滤（异常点检测 + 插值 + 平滑）。
4. 计算三项业务派生指标：
  - Twist 峰差指标（含新版周期振幅均值）
  - 姿勢绝对逸脱均值
  - 肩部正面时的手腕均值位置
5. 产出单人 JSON、单人派生指标 JSON、可视化图片。
6. 汇总多人比较 JSON 与全体派生指标 CSV。

---

## 2. 输入、依赖、输出

### 2.1 主要依赖

- [analysis/metrics.py](metrics.py)
  - `compute_twist`
  - `compute_trunk_tilt`
  - `compute_wrist_lead_angle`
  - `MHR70_INDEX`
- [analysis/visualize.py](visualize.py)
  - `plot_all_visualizations`
  - `plot_derived_metrics_summary`
- [fuse/load.py](../fuse/load.py)
  - `load_fused_sequence`

### 2.2 默认路径

- 输入目录：`/workspace/code/logs/fuse/person_*`
- 输出目录：`/workspace/code/logs/analysis`

---

## 3. 计算了什么（指标与定义）

### 3.1 基础时序指标

1. Twist（ねじれ）
  - 来源：肩部与髋部旋转角差。
  - 处理：`unwrap` 后转为角度序列。

2. Trunk Tilt（姿勢）
  - 来源：体干相对垂直方向的倾角。
  - 处理：转为角度序列。

3. Wrist Lead（脱力）
  - 来源：滞后手手腕相对躯干的角度。
  - 处理：`unwrap` 后转为角度序列。

### 3.2 三项业务派生指标

1. ツイスト（Twist）
  - 旧定义：所有局部最大值均值 - 所有局部最小值均值。
  - 新定义（当前主用）：相邻极值配对的周期振幅均值（`cycle_amp_mean_deg`）。

2. 姿勢（Posture）
  - 定义：`|tilt_deg|` 的均值（绝对逸脱度均值）。

3. 脱力（Relaxation）
  - 定义：肩部正面帧下，滞后手手腕位置 `XYZ` 的均值。
  - 比较默认使用 `Y` 分量（`wrist_frontal_mean_y`）。

---

## 4. 用了什么方法（算法细节）

### 4.1 不平滑过滤方法

函数：`_filter_unsmooth_signal(values, median_window=9, z_thresh=3.5, smooth_window=5)`

处理流程：

1. 中值滤波基线：`_median_filter`
2. 残差：`residual = values - baseline`
3. 鲁棒异常分数：MAD 版 z-score
  - $z = \frac{|r - \text{median}(r)|}{\text{MAD} + \epsilon}$
4. 异常点判定：`z > z_thresh`
5. 异常点设为 NaN 后线性插值：`_interpolate_nan`
6. 轻度滑动平均：`_moving_average`

作用：

- 抑制单帧突刺和离群跳变；
- 保留主趋势；
- 提高峰值与统计稳定性。

### 4.2 Twist 峰值指标方法

函数：`_compute_twist_peak_gap_deg(twist_deg)`

内部做两套计算：

1. 全局极值均值差（legacy）
  - 找局部极大值、极小值。
  - `peak_gap_legacy_deg = mean(local_max) - mean(local_min)`

2. 周期振幅均值（adjacent extrema）
  - 带索引提取局部极值点。
  - 按时间排序后对“相邻且类型不同”的极值配对。
  - 每对振幅：`abs(v_{i+1} - v_i)`
  - `cycle_amp_mean_deg = mean(all_pair_amplitudes)`

作用：

- legacy 更偏全局；
- cycle 版本更贴近“每个周期振幅”的直觉，当前比较主要使用该值。

### 4.3 肩部正面帧与手腕位置

1. `_compute_shoulder_frontal_mask`：
  - 取肩线向量（`rsho - lsho`），投影水平面。
  - 与参考轴 `[1,0,0]` 的夹角接近 `0` 或 `π` 视为“正面”。
2. `_compute_frontal_wrist_position`：
  - 依据 `lagging_hand` 选左右手腕。
  - 在正面帧上求 `XYZ` 均值和帧数。

作用：

- 将“脱力”落到可解释的空间位置指标上，支持动作时机分析。

---

## 5. 主流程（main + analyze_sequence）

### 5.1 单人流程 `analyze_sequence`

1. 读取 `kpts_world` 和 `fps`。
2. 计算 twist / tilt / lead 三条原始序列。
3. 对三条序列执行不平滑过滤。
4. 计算派生指标（twist_peak、tilt_abs_mean、wrist_frontal_mean_xyz）。
5. 打印统计和过滤数量。
6. 保存单人主结果 JSON：`analysis_{id}.json`。
7. 保存单人派生值 JSON：`analysis_{id}_derived_values.json`。
8. 生成常规图 + 派生指标总结图。

### 5.2 多人流程 `main`

1. 扫描 `person_*` 目录。
2. 逐人执行 `analyze_sequence`。
3. 聚合多人指标写入：
  - `comparison_summary.json`
  - `derived_metrics_all_persons.csv`

异常策略：单人异常不会中断全体，直接跳过继续。

---

## 6. 输出文件说明

### 6.1 单人输出（person_{id}）

1. 主结果：`analysis_{id}.json`
  - 含过滤后时序、统计、派生指标等。

2. 派生数值：`analysis_{id}_derived_values.json`
  - 关键字段：
    - `twist_cycle_amp_mean_deg`
    - `twist_peak_gap_legacy_deg`
    - `tilt_abs_mean_deg`
    - `wrist_frontal_mean_x/y/z`
    - `shoulder_frontal_frame_count`

3. 可视化图：
  - 常规四图：timeseries / distributions / cdf / correlation
  - 派生指标图：`analysis_{id}_derived_metrics.png`
    - 包含时序 + box plot（均值、分布与离散程度）

### 6.2 全体输出（analysis 根目录）

1. `comparison_summary.json`
  - `comparisons` 中包含：
    - `twist_cycle_amp_mean_deg`（主用）
    - `twist_peak_gap_deg`（legacy）
    - `tilt_abs_mean_deg`
    - `wrist_frontal_mean_y`

2. `derived_metrics_all_persons.csv`
  - 每行一人，便于直接进 Excel / pandas 做统计与建模。

---

## 7. 为什么这些指标有用（作用解释）

1. Twist 周期振幅均值
  - 反映动作回旋幅度是否充分、节奏是否稳定。

2. 姿勢绝对逸脱均值
  - 量化躯干偏离竖直的总体程度，反映姿态控制。

3. 肩正面时手腕均值位置
  - 把“脱力时机”映射到空间位置，帮助比较动作技术差异。

4. Box Plot
  - 展示中位数、四分位、离散程度与均值，便于识别稳定性和异常。

---

## 8. 参数与调参建议

默认参数（当前）：

- `median_window=9`
- `z_thresh=3.5`
- `smooth_window=5`

调参方向：

1. 过滤太强（细节被抹平）
  - 增大 `z_thresh` 或减小窗口。
2. 过滤太弱（尖刺仍多）
  - 减小 `z_thresh` 或增大窗口。

建议预设：

- 保守：`z_thresh=4.5, median_window=7, smooth_window=3`
- 平衡：`z_thresh=3.5, median_window=9, smooth_window=5`
- 强过滤：`z_thresh=2.8, median_window=11, smooth_window=7`

---

## 9. 已知边界与注意事项

1. 局部极值法在平台段/高噪声段可能出现较多极值点。
2. `unwrap` 会改变角度绝对区间表达，但有利于连续性分析。
3. `frontal_mask`（来自 wrist lead 算法）与 `shoulder_frontal_mask`（来自肩线）语义不同。
4. `shoulder_frontal_frame_count` 太小时，手腕均值稳定性下降。

---

## 10. 运行方式

```bash
python code/analysis/main.py
```

运行后建议检查：

1. 单人目录下是否有 `analysis_{id}_derived_values.json`。
2. 根目录是否有 `comparison_summary.json` 和 `derived_metrics_all_persons.csv`。
3. 派生图是否包含 box plot 且统计值合理。

---

## 11. 快速定位（函数索引）

- 过滤：`_filter_unsmooth_signal`
- Twist 派生：`_compute_twist_peak_gap_deg`
- 正面帧：`_compute_shoulder_frontal_mask`
- 手腕均值：`_compute_frontal_wrist_position`
- 单人保存：`_save_person_derived_values`
- 全体保存：`_save_all_persons_derived_csv`
- 全体比较：`_save_comparison_summary`
- 主流程：`analyze_sequence` / `main`
