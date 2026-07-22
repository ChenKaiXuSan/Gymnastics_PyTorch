# Gymnastics Multiview Flowchart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an editable Japanese draw.io flowchart that summarizes the gymnastics pipeline and expands the deterministic and rotation-aware multiview fusion branches.

**Architecture:** Use one left-to-right 16:9 page with a shared preprocessing trunk, a separate triangulated pseudo-reference evaluation path, and two parallel fusion lanes. Hand-author deterministic draw.io XML so the layout, colors, Japanese labels, and the separation between training and evaluation paths remain explicit.

**Tech Stack:** draw.io `mxGraphModel` XML, bundled draw.io structural validator, draw.io desktop CLI when available, SVG fallback when the CLI is unavailable.

## Global Constraints

- Use Japanese labels throughout the figure.
- Use a horizontal 16:9 layout with left-to-right data flow.
- Use neutral colors for shared processing, blue for the deterministic baseline, orange for the proposed method, and green for evaluation.
- Call triangulated output `3D疑似参照`; do not call it ground truth.
- Do not connect the triangulated pseudo-reference to the proposed method's training path.
- Treat action analysis and classification as optional downstream processing.
- Preserve unrelated modifications already present in the worktree.

---

### Task 1: Author and structurally validate the draw.io source

**Files:**
- Create: `docs/figures/gymnastics_multiview_pipeline.drawio`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-07-22-gymnastics-multiview-flowchart-design.md`
- Produces: one uncompressed `mxGraphModel` page named `マルチビュー体操動作解析フロー`, sized for a 1600 × 900 canvas.

- [ ] **Step 1: Create the figure directory**

Run:

```bash
mkdir -p docs/figures
```

Expected: `docs/figures/` exists without changing any existing files.

- [ ] **Step 2: Author the draw.io XML with fixed sections**

Create the source with these vertex groups and labels:

| Group | Required labels, in order |
|---|---|
| Shared trunk | `正面・側面映像`, `SAM3D-Body`, `2D / 3D関節点抽出`, `時間同期・動作周期分割` |
| Evaluation reference | `2D関節点`, `三角測量`, `3D疑似参照`, `評価時のみ使用` |
| Baseline lane | `ベースライン手法`, `安定関節の選択`, `side → face Sim3位置合わせ`, `平均融合`, `時間平滑化`, `融合3D姿勢` |
| Proposed lane | `提案手法`, `身体中心座標への正規化`, `回転・視点間差分特徴`, `視点交換不変の残差時系列融合モデル`, `融合3D姿勢` |
| Self-supervision | `自己教師あり学習`, `破損・合意性・幾何・時間的一貫性・周期回転` |
| Output | `3D疑似参照との比較`, `MPJPE`, `動作解析・分類（任意）` |

Use rounded rectangles for processing steps, a document/data shape for the pseudo-reference, dashed orange connectors from the self-supervision block to the proposed model, and solid orthogonal connectors for data flow. Place the pseudo-reference path above the two fusion lanes and route it only to the evaluation block.

- [ ] **Step 3: Run structural validation**

Run:

```bash
python3 /home/workspace/kaixu/.agents/skills/drawio-skill/skills/drawio-skill/scripts/validate.py docs/figures/gymnastics_multiview_pipeline.drawio --score
```

Expected: exit code `0`, no duplicate or reserved IDs, no dangling edges, and no overlapping vertices.

- [ ] **Step 4: Check the source diff**

Run:

```bash
git diff --check -- docs/figures/gymnastics_multiview_pipeline.drawio
git status --short -- docs/figures/gymnastics_multiview_pipeline.drawio
```

Expected: no whitespace errors and exactly one new draw.io source file.

### Task 2: Render, inspect, and deliver the preview

**Files:**
- Create: `docs/figures/gymnastics_multiview_pipeline.svg`
- Create when draw.io CLI is available: `docs/figures/gymnastics_multiview_pipeline.png`
- Create after user approval when draw.io CLI is available: `docs/figures/gymnastics_multiview_pipeline.drawio.png`

**Interfaces:**
- Consumes: validated `docs/figures/gymnastics_multiview_pipeline.drawio`
- Produces: a locally viewable preview plus an editable final source.

- [ ] **Step 1: Resolve the draw.io renderer**

Run in order until one command prints a version:

```bash
drawio --version
draw.io --version
```

Expected on the current host: neither binary is installed. Continue with the SVG fallback and keep the draw.io source as the editable deliverable.

- [ ] **Step 2: Create a matching SVG fallback**

Create an SVG using the same 1600 × 900 coordinates, Japanese labels, palette, node dimensions, and connector routes as the draw.io page. Include a small legend for `共通処理`, `ベースライン`, `提案手法`, and `評価`.

Expected: the SVG opens directly in the Codex app and visually matches the draw.io source.

- [ ] **Step 3: Inspect the SVG preview**

Open `docs/figures/gymnastics_multiview_pipeline.svg` with the local image viewer and verify:

- no clipped Japanese labels;
- no overlapping boxes or connectors;
- the baseline and proposed lanes remain visually distinct;
- the pseudo-reference has no training connection to the proposed method;
- both fused outputs connect to MPJPE evaluation.

Expected: all five checks pass after at most two targeted layout adjustments.

- [ ] **Step 4: Export PNG when draw.io becomes available**

Preview command:

```bash
drawio -x -f png --width 2000 -o docs/figures/gymnastics_multiview_pipeline.png docs/figures/gymnastics_multiview_pipeline.drawio
```

After user approval, final editable PNG command:

```bash
drawio -x -f png -e -s 2 -o docs/figures/gymnastics_multiview_pipeline.drawio.png docs/figures/gymnastics_multiview_pipeline.drawio
python3 /home/workspace/kaixu/.agents/skills/drawio-skill/skills/drawio-skill/scripts/repair_png.py docs/figures/gymnastics_multiview_pipeline.drawio.png
```

Expected: both PNG files open successfully; the final double-extension PNG contains embedded diagram XML.

- [ ] **Step 5: Commit only the diagram deliverables**

```bash
git add docs/figures/gymnastics_multiview_pipeline.drawio docs/figures/gymnastics_multiview_pipeline.svg
git commit -m "docs: add multiview fusion flowchart"
```

Expected: one commit containing only the editable flowchart and its SVG preview. PNG files remain optional until the renderer is installed.
