"""Collect the strict external baselines' ``summary_*.json`` files into one table.

    python -m fusion external-published report [--markdown out.md] [--csv out.csv]

Rows are (method, variant), columns the datasets; cells are the fold mean
+- sd of the per-frame PA-MPJPE in mm with the number of scored joints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common.paths import PROJECT_ROOT

OUTPUT_ROOT = PROJECT_ROOT / "local" / "runs" / "external_published"
DATASETS = ("gymnastics", "freeman", "fit3d")
ZERO_SHOT = {("videopose3d", "procrustes_average"), ("videopose3d", "per_view"), ("metapose", "s2_released")}


def collect(root: Path = OUTPUT_ROOT) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(root.glob("*/*/summary_*.json")) + sorted(root.glob("*/*/*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        parts = path.relative_to(root).parts
        method, dataset = parts[0], parts[1]
        variant = path.stem[len("summary_"):] if path.stem.startswith("summary_") else parts[2]
        if dataset not in DATASETS:
            continue
        summary = payload["summary"]
        row = rows.setdefault((method, variant), {"method": method, "variant": variant, "zero_shot": (method, variant) in ZERO_SHOT})
        row[dataset] = {"mean_mm": 1000 * summary["pa_mpjpe_mean"], "sd_mm": 1000 * summary["pa_mpjpe_sd"], "folds": summary["folds"], "joints": len(summary["joint_names"]), "per_fold_mm": [1000 * f["pa_mpjpe"] for f in payload["folds"]], "method_info": payload.get("method", {})}
    return [rows[k] for k in sorted(rows)]


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["| Method | Variant | " + " | ".join(DATASETS) + " |", "|---|---|" + "|".join("---:" for _ in DATASETS) + "|"]
    for row in rows:
        cells = []
        for dataset in DATASETS:
            cell = row.get(dataset)
            cells.append(f"{cell['mean_mm']:.1f} ± {cell['sd_mm']:.1f} ({cell['joints']} j, {cell['folds']} folds)" if cell else "--")
        label = row["method"] + (" (zero-shot, appendix)" if row["zero_shot"] else "")
        lines.append(f"| {label} | {row['variant']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = collect(args.root)
    table = markdown(rows)
    print(table)
    if args.markdown:
        args.markdown.write_text(table + "\n", encoding="utf-8")
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["method", "variant", "zero_shot", "dataset", "pa_mpjpe_mean_mm", "pa_mpjpe_sd_mm", "folds", "joints", "per_fold_mm"])
            for row in rows:
                for dataset in DATASETS:
                    cell = row.get(dataset)
                    if cell:
                        writer.writerow([row["method"], row["variant"], row["zero_shot"], dataset, f"{cell['mean_mm']:.2f}", f"{cell['sd_mm']:.2f}", cell["folds"], cell["joints"], " ".join(f"{v:.1f}" for v in cell["per_fold_mm"])])
    return 0
