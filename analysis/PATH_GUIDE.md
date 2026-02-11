# 路径配置说明

## 更新内容

所有分析模块的脚本现在都使用**绝对路径**而非相对路径，确保无论在哪个目录下运行都能正确找到文件。

## 主要变更

### 1. main.py
- `analyze_sequence()` 函数现在接收完整的 `person_root` 路径参数
- 使用 `.resolve()` 确保所有路径都是规范化的绝对路径
- 默认输出目录：`/workspace/code/logs/analysis/person_{id}`

```python
# 旧版本（相对路径）
analyze_sequence(person_id="1")

# 新版本（绝对路径）
person_root = Path("/workspace/code/logs/fuse/person_1")
output_dir = Path("/workspace/code/logs/analysis/person_1")
analyze_sequence(person_id="1", person_root=person_root, output_dir=output_dir)
```

### 2. batch_analyze.py
- 使用绝对路径：`/workspace/code/logs/fuse` 和 `/workspace/code/logs/analysis`
- 传递完整的 `person_dir` 和 `output_dir` 路径给 `analyze_sequence()`

### 3. example_single.py
- 工作区根目录：`/workspace/code`
- 输入目录：`/workspace/code/logs/fuse/person_{id}`
- 输出目录：`/workspace/code/logs/analysis/person_{id}`

### 4. compare.py
- 分析结果目录：`/workspace/code/logs/analysis`

## 优势

1. **可靠性**：无论在哪个目录运行脚本，都能正确找到文件
2. **清晰性**：路径明确，不会产生歧义
3. **可维护性**：路径集中管理，易于修改

## 使用示例

### 运行单人分析
```bash
cd /workspace/code
python analysis/example_single.py
```

### 批量分析
```bash
cd /workspace/code
python analysis/batch_analyze.py
```

### 主分析程序
```bash
cd /workspace/code
python analysis/main.py
```

### 结果对比
```bash
cd /workspace/code
python analysis/compare.py
```

## 路径结构

```
/workspace/code/
├── logs/
│   ├── fuse/              # 输入：融合后的3D关键点
│   │   ├── person_1/
│   │   ├── person_2/
│   │   └── ...
│   └── analysis/          # 输出：分析结果
│       ├── person_1/
│       │   ├── analysis_1.json
│       │   ├── analysis_1_timeseries.png
│       │   ├── analysis_1_distributions.png
│       │   ├── analysis_1_cdf.png
│       │   └── analysis_1_correlation.png
│       ├── person_2/
│       └── ...
└── analysis/              # 分析代码
    ├── main.py
    ├── batch_analyze.py
    ├── example_single.py
    ├── compare.py
    └── ...
```

## 自定义路径

如果需要修改默认路径，可以在调用函数时传入自定义路径：

```python
from pathlib import Path
from analysis.main import analyze_sequence

person_id = "1"
person_root = Path("/custom/path/to/fuse/person_1")
output_dir = Path("/custom/path/to/output/person_1")

analyze_sequence(person_id, person_root, output_dir)
```

## 测试路径配置

运行测试脚本检查路径配置：

```bash
cd /workspace/code
python analysis/test_paths.py
```

该脚本会显示：
- 工作区根目录
- Fuse输出目录及所有人物目录
- Analysis输出目录及所有结果目录
