# Split-Cycle 轻量 SAM3D 加载器设计

## 背景

`split_cycle.main` 只使用每帧的 `pred_keypoints_3d` 和帧号来完成双视角时间对齐、身体坐标融合与周期切分。然而，当前 `split_cycle/load.py` 会把每个 NPZ 中的完整 `output` 字典保存在返回值中，其中包含一张 `1920×1080×3` 图像和人体网格等大数组；加载器还会在循环中反复追加同一个可变字典对象。

单个人的正面和侧面序列约有 2,500 帧，这种实现会让一个工作线程长期持有十几 GB 数据。当前机器有 62 GiB 内存，因此默认 11 线程批处理存在明显的内存耗尽和交换分区抖动风险。

## 目标

- 保持 `load_sam3d_body_sequence()` 的公开返回形式 `(all_info, kpts3d)` 不变。
- 返回的 `all_info` 只包含周期切分所需的轻量逐帧元数据。
- 不让返回值继续引用输入 NPZ 中的图像、人体网格或完整输出字典。
- 修复所有列表元素指向同一个字典对象的问题。
- 保持 `split_cycle.main` 的调用方式和关键点数值行为不变。
- 为后续多人物并发处理降低内存风险。

## 非目标

- 不修改 SAM3D-Body NPZ 的磁盘格式。
- 不修改 `fuse/load.py`；它是独立模块，不是本次 `split_cycle` 批处理的调用路径。
- 不改变 DTW、音频对齐、周期检测或视频切分算法。
- 不在源代码中固定 Numba 缓存路径；批处理命令会显式提供可写的 `NUMBA_CACHE_DIR`。
- 不清理或改写人物 1–68 的既有切分结果。

## 方案比较

### 方案 A：兼容式轻量加载（采用）

保留现有函数和返回接口，但每帧只保存 `frame_idx` 与 `pred_keypoints_3d`。优点是改动范围小、调用方无需迁移、容易用单元测试锁定行为。缺点是 `all_info` 仍保留在接口中，尽管其体积已经很小。

### 方案 B：新增关键点专用加载函数

新增只返回关键点和帧号的接口，并迁移 `split_cycle.main`。边界更清晰，但会形成两个相近加载接口，并扩大本次变更和测试范围。

### 方案 C：保持代码不变并使用单线程

无需修改代码，但单人仍会长期占用大量内存，批处理时间显著增加，也没有解决重复字典和无关图像保留的根因。

## 详细设计

### 加载流程

对排序后的每个 `*_sam3d_body.npz`：

1. 使用 `with np.load(path, allow_pickle=True)` 打开归档，确保文件及时关闭。
2. 读取顶层 `output` 字典。
3. 读取并规范化 `frame_idx` 为 Python `int`。
4. 将 `pred_keypoints_3d` 转成独立的 NumPy 数组，避免返回值依赖已关闭的归档对象。
5. 为该帧新建独立字典：

   ```python
   {
       "frame_idx": frame_idx,
       "pred_keypoints_3d": pred_keypoints_3d,
   }
   ```

6. 将轻量字典追加到 `all_info`，将关键点追加到 `all_kpts`。
7. 完成后将关键点堆叠为 `(T, J, 3)` 并返回 `(all_info, kpts3d)`。

返回值不得包含 `frame`、`pred_vertices` 或原始 `output` 字典。

### 错误行为

- 输入目录不存在或没有匹配文件时，保持现有 `FileNotFoundError` 行为。
- NPZ 缺少 `output`、`frame_idx` 或 `pred_keypoints_3d` 时，保留明确异常，不静默跳帧。
- 关键点无法堆叠时，让 NumPy 抛出形状不一致错误，避免生成长度或关节数不一致的序列。

### 音频缓存

ID69 冒烟处理证明：未设置缓存目录时，`librosa/numba` 因 Conda 包目录不可写而回退到纯关键点对齐；设置

```bash
NUMBA_CACHE_DIR=/tmp/gymnastics_numba_cache
```

后，音频偏移可正常计算。此变量将在重新处理 ID69 和批量处理新人物时由运行命令显式设置，不纳入本次源代码修改。

## 测试设计

新增 `tests/test_split_cycle_load.py`，使用临时目录生成两个小型 SAM3D NPZ：

- 每帧包含不同的 `frame_idx`、3D 关键点和模拟大图像。
- 验证返回的关键点形状和数值顺序正确。
- 验证 `all_info` 中每个元素是不同字典对象。
- 验证每个元素只包含 `frame_idx` 与 `pred_keypoints_3d`。
- 验证返回值中不存在 `frame` 和完整 `output`。
- 验证修改测试中的一个元数据字典不会影响另一帧。

回归验证运行：

```bash
conda run -n gymnastic python -m pytest tests/test_split_cycle_load.py -q
conda run -n gymnastic python -m pytest tests/test_split_cycle_audio_alignment.py tests/test_split_cycle_cli.py -q
```

测试驱动顺序为：先新增测试并观察其在旧实现上失败，再修改加载器并观察测试通过，最后运行现有 split-cycle 回归测试。

## 运行与验收

1. 将本轮已生成的 ID69 纯关键点回退结果移动到 `/tmp` 备份，避免旧周期视频残留在正式目录。
2. 设置可写的 `NUMBA_CACHE_DIR`，重新完整处理 ID69。
3. 验证 ID69 的对齐记录包含音频偏移、置信度、最终偏移和非空周期。
4. 使用保守并发批量处理人物 70–134、136–138。
5. 全量解析人物 1–138（排除缺少输入的 135）的对齐记录，检查：
   - 每个可用人物恰有一份记录；
   - `offset_side_to_face` 存在且为整数；
   - 周期列表非空；
   - 正面和侧面周期数量一致；
   - 周期范围单调、非重叠且位于原视频帧数内；
   - 汇总所有失败人物与低重叠率警告。

成功标准是轻量加载单元测试和现有回归测试全部通过，ID69 音频辅助对齐可用，并且所有可用人物生成结构有效的 split-cycle 对齐记录。
