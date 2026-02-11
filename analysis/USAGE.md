# 体操动作分析模块

## 概述

本模块用于分析体操动作的3D关键点数据，计算三个关键指标：

1. **ねじれ (Twist)**: 肩部和髋部的旋转差异
2. **姿勢 (Trunk Tilt)**: 躯干相对于垂直方向的倾斜角度
3. **脱力 (Wrist Lead Angle)**: 手腕在正面位置时的先行角度

## 模块结构

```
analysis/
├── __init__.py           # 模块初始化
├── metrics.py            # 指标计算函数（核心算法）
├── visualize.py          # 可视化函数（多种图表）
├── load.py               # 数据加载工具
├── main.py               # 主分析流程
├── batch_analyze.py      # 批量分析脚本
├── compare.py            # 多人对比脚本
├── example_single.py     # 单人分析示例
└── README.md             # 本文档
```

## 核心文件说明

### metrics.py - 指标计算

包含所有指标计算的核心函数：

```python
from analysis.metrics import (
    compute_twist,           # 计算扭转
    compute_trunk_tilt,      # 计算躯干倾角
    compute_wrist_lead_angle,# 计算手腕先行角
    MHR70_INDEX,            # MHR70骨架的关键点索引
)

# 使用示例
twist = compute_twist(kpts_world, MHR70_INDEX)  # 返回弧度
twist_deg = np.degrees(twist)  # 转换为角度
```

### visualize.py - 可视化

提供丰富的可视化功能：

- **时间序列图**: 三个指标随时间的变化
- **分布图**: 直方图 + 箱线图
- **累积分布函数 (CDF)**: 数值的累积概率分布
- **相关性图**: 指标之间的相关性散点图

```python
from analysis.visualize import plot_all_visualizations

# 生成所有可视化图表
plot_all_visualizations(
    time, twist_deg, tilt_deg, lead_angle_deg, 
    frontal_mask, output_dir, person_id
)
```

### main.py - 主分析流程

提供端到端的分析流程：

```python
from analysis.main import analyze_sequence

# 分析单个人物
analyze_sequence(person_id="1")

# 结果保存到: logs/analysis/person_1/
# - analysis_1.json (数值结果)
# - analysis_1_timeseries.png (时间序列图)
# - analysis_1_distributions.png (分布图)
# - analysis_1_cdf.png (累积分布)
# - analysis_1_correlation.png (相关性)
```

## 使用方法

### 1. 单人分析

```bash
# 方法1: 直接运行main.py
python code/analysis/main.py

# 方法2: 使用示例脚本
python code/analysis/example_single.py
```

### 2. 批量分析

```bash
# 分析所有人物
python code/analysis/batch_analyze.py
```

### 3. 多人对比

```bash
# 生成对比图表
python code/analysis/compare.py
```

## 输出说明

### JSON结果文件

```json
{
  "person_id": "1",
  "fps": 30.0,
  "n_frames": 300,
  "twist_deg": [...],        // 每帧的扭转角度
  "tilt_deg": [...],         // 每帧的倾角
  "lead_angle_deg": [...],   // 每帧的手腕先行角
  "frontal_mask": [...],     // 正面位置标记
  "lagging_hand": "right",   // 滞后的手
  "statistics": {
    "twist": {
      "mean": 15.3,
      "std": 8.2,
      "min": -5.1,
      "max": 45.6
    },
    ...
  }
}
```

### 可视化图表

1. **timeseries.png**: 三个指标的时间序列图
2. **distributions.png**: 直方图和箱线图（带统计信息）
3. **cdf.png**: 累积分布函数
4. **correlation.png**: 指标间的相关性散点图

## 代码示例

### 基础使用

```python
import numpy as np
from analysis.load import load_fused_sequence
from analysis.metrics import (
    compute_twist, 
    compute_trunk_tilt, 
    compute_wrist_lead_angle,
    MHR70_INDEX
)

# 加载数据
kpts_world, kpts_body, metadata = load_fused_sequence("1")

# 计算指标
twist = compute_twist(kpts_world, MHR70_INDEX)
tilt = compute_trunk_tilt(kpts_world, MHR70_INDEX)
lead_angle, frontal_mask, lagging_hand = compute_wrist_lead_angle(
    kpts_world, MHR70_INDEX
)

# 转换为角度
twist_deg = np.degrees(twist)
tilt_deg = np.degrees(tilt)
lead_angle_deg = np.degrees(lead_angle)

print(f"平均扭转: {twist_deg.mean():.1f}°")
print(f"平均倾角: {tilt_deg.mean():.1f}°")
```

### 自定义可视化

```python
from analysis.visualize import (
    plot_time_series,
    plot_distributions,
    plot_cdf,
    plot_correlation_matrix
)

# 只生成时间序列图
plot_time_series(
    time, twist_deg, tilt_deg, lead_angle_deg, 
    frontal_mask, output_path
)

# 只生成分布图
plot_distributions(
    twist_deg, tilt_deg, lead_angle_deg, 
    frontal_mask, output_path
)
```

## 指标详细说明

### 1. ねじれ (Twist)

**定义**: 肩部和髋部在水平面内的旋转角度差

**计算方法**:
- 投影肩部连线和髋部连线到水平面
- 计算各自相对于参考轴（左右方向）的旋转角度
- Twist = 肩部角度 - 髋部角度

**物理意义**: 
- 正值：上半身相对下半身逆时针旋转
- 负值：上半身相对下半身顺时针旋转
- 绝对值越大，扭转越明显

### 2. 姿勢 (Trunk Tilt)

**定义**: 躯干（肩部中心到髋部中心）相对于垂直方向的倾斜角度

**计算方法**:
- 计算肩部中心和髋部中心
- 躯干向量 = 肩部中心 - 髋部中心
- 计算躯干向量与垂直轴的夹角

**物理意义**:
- 0°：完全垂直
- 正值：向前倾
- 负值：向后仰

### 3. 脱力 (Wrist Lead Angle)

**定义**: 在正面位置时，滞后手的手腕相对于躯干的先行角度

**计算方法**:
1. 检测旋转方向（顺时针/逆时针）
2. 确定滞后的手（旋转方向的对侧）
3. 在正面位置时，计算手腕相对于躯干的角度

**物理意义**:
- 正值：手腕超前（更放松）
- 负值：手腕滞后（较紧张）
- 只在正面位置时测量最准确

## 数据要求

### 输入数据格式

```python
kpts_world: np.ndarray  # (T, J, 3) - 世界坐标系
kpts_body: np.ndarray   # (T, J, 3) - 身体坐标系
```

其中:
- T: 帧数
- J: 关键点数量（MHR70骨架为70+个关键点）
- 3: (x, y, z) 坐标

### 关键点索引

使用MHR70骨架格式，关键点索引定义在`metrics.py`中：

```python
MHR70_INDEX = {
    "lhip": 9,      # 左髋
    "rhip": 10,     # 右髋
    "lsho": 5,      # 左肩
    "rsho": 6,      # 右肩
    "rwrist": 41,   # 右手腕
    "lwrist": 40,   # 左手腕
    ...
}
```

## 性能说明

- 单人分析时间: ~2-5秒（取决于帧数）
- 内存使用: ~500MB（300帧序列）
- 可视化生成: ~1-2秒/图

## 注意事项

1. **坐标系**: 确保输入数据使用正确的坐标系（世界坐标系）
2. **单位**: 所有角度计算结果为弧度，需要使用`np.degrees()`转换为度
3. **正面检测**: Lead angle的计算依赖正面位置检测，阈值可在`metrics.py`中调整
4. **数据完整性**: 确保所有必需的关键点都存在且有效

## 扩展开发

### 添加新指标

1. 在`metrics.py`中添加计算函数
2. 在`main.py`中调用新函数
3. 在`visualize.py`中添加可视化（可选）
4. 更新`__init__.py`导出新函数

### 自定义可视化

参考`visualize.py`中的现有函数，创建新的绘图函数。

## 故障排查

### 常见错误

1. **FileNotFoundError**: 检查`logs/fuse/person_X`目录是否存在
2. **KeyError**: 检查MHR70关键点索引是否正确
3. **MemoryError**: 减少处理的帧数范围

### 调试技巧

```python
# 启用详细输出
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据形状
print(f"kpts_world shape: {kpts_world.shape}")
print(f"Expected: (T, 70+, 3)")

# 检查关键点是否有效
assert not np.any(np.isnan(kpts_world)), "NaN detected in keypoints"
```

## 参考

- MHR70骨架定义: 见SAM-3D-Body文档
- 坐标系说明: 见fusion模块文档

## 更新日志

- 2026-02-11: 
  - 提取指标计算到`metrics.py`
  - 创建丰富的可视化模块`visualize.py`
  - 重构代码结构，提高模块化
