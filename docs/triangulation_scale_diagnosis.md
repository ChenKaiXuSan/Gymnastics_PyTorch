# 三角化尺度诊断

诊断日期：2026-07-23。本文只记录诊断结论，**未对三角化流程做任何修改**。

## 结论

三角化伪真值的绝对尺度不可信，逐人尺度也不一致。根因是**双相机外参从未标定**，
而是由 `configs/sam3d_triangulation.yaml` 里写死的名义几何合成出来的。

## 证据

### 1. 外参是合成的，不是标定的

`configs/sam3d_triangulation.yaml` 中：

```yaml
camera_position:
  T: [0.0, 0.0, 1.5]   # 假定被试站在原点
  r: 3.5               # 假定相机在半径 3.5 的圆周上
  z: 1.5               # 假定相机高度
  yaws: {1: -90.0, 2: -45.0, 3: 0.0, 4: -135.0}
view_camera: {face: 3, side: 1}
```

`triangulation/camera_position_mapping.py::prepare_camera_position` 直接由这些数值
构造外参：相机放在半径 `r`、高度 `z` 的圆周上，朝向 `T`。face（yaw 0°）与
side（yaw −90°）被假定为**恰好相隔 90°**，基线 `3.5 * sqrt(2) ≈ 4.95`。

标定文件（`logs/calibration_vis/IMG_*/calibration_parameters.npz`）里只有
**内参**：`camera_matrix`、`dist_coeffs`，以及每张棋盘图各自的 `rvecs`/`tvecs`。
四段标定视频是各相机**独立**拍摄的（帧号互不对应，`square_size = 25.0` mm），
没有双相机同时看到同一棋盘的画面，因此**现有数据无法做立体外参标定**。

### 2. 内参很好，三角化很差

| 项目 | 重投影误差 |
|---|---:|
| 内参标定 IMG_2420 | 0.173 px |
| 内参标定 IMG_2676 | 0.226 px |
| 三角化 face（928 cycle） | 均值 25.62 px，中位 20.55，p95 54.71 |
| 三角化 side（928 cycle） | 均值 26.48 px，中位 21.46，p95 51.88 |

三角化误差是内参标定误差的约 **128 倍**。镜头模型没问题，错的是几何假设。

### 3. 重建身高不合理

对 137 人各取前 3 个 cycle 的逐帧最大轴向跨度中位数：

| 来源 | 均值 | 标准差 | CV | 范围 |
|---|---:|---:|---:|---|
| SAM3D 融合关键点 | 1.582 | 0.038 | **2.4%** | [1.47, 1.67] |
| 三角化伪真值 | 1.955 | 0.197 | **10.1%** | [1.62, **2.55**] |

- 三角化的逐人离散度是 SAM3D 的 4 倍，真实成人身高 CV 约 5–6%。
- 最大值 2.55 对人体不成立。
- 全局尺度比 `median(三角化 / SAM3D) = 1.206`。若 SAM3D 尺度可信，
  则真实相机半径约 `3.5 / 1.206 ≈ 2.90`，而非配置里的 3.5。

逐人尺度比与重投影误差的相关性只有 −0.25，说明逐人偏差不是简单的噪声大小问题，
而是几何假设本身对不同被试/不同场次都不成立。

## 影响范围

- **绝对尺度不可用**：任何以三角化结果换算毫米的结论都不成立。
- **逐人可比性受损**：10% 的尺度离散会直接进入逐人 MPJPE 对比。
- 融合评测已通过 per-sequence Sim3 对齐吸收掉了尺度与朝向失配
  （见 `fuse/experiment_matrix.py::align_candidate`），所以**方法间排名不受影响**；
  受影响的是任何依赖三角化绝对尺度或跨人尺度一致性的结论。

## 可选的修复路径（未实施）

1. **重新采集标定数据做立体外参标定** —— 最彻底，需要双相机同时拍摄同一棋盘。
2. **从现有 2D 对应自标定外参** —— 用 928 个 cycle 的双视角 2D 关键点估计本质矩阵，
   恢复相对位姿，再用体高或骨长定尺度。不需要新数据。
3. **仅改全局尺度** —— 把 `r` 从 3.5 改为约 2.90，可消除 1.206 倍的系统性偏差，
   但无法解决 10% 的逐人离散。

## 复现

```bash
conda run -n gymnastic python - <<'PY'
import json, numpy as np
from pathlib import Path
s = json.load(open('/home/data/xchen/gymnastics/sam3d_triangulated/person/summary.json'))
err = [c['face_reprojection_error_mean_px']
       for p in s['persons'] for c in p.get('cycles', [])
       if c.get('face_reprojection_error_mean_px') is not None]
print('face reproj err mean/median/p95:',
      np.mean(err), np.median(err), np.percentile(err, 95))
PY
```
