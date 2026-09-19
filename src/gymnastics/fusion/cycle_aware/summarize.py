"""Summarise a cross-validation sweep whose folds ran as separate jobs.

``gymnastics fuse cycle-aware folds_dir=...`` runs folds sequentially and
writes the summary itself.  On the cluster each fold is a separate job
(``pegasus/cycle_aware_fold_qsub.sh``), so the per-fold ``result.json`` files
land under ``<sweep_dir>/fold_NN/`` independently; this module collects them::

    python -m gymnastics.fusion.cycle_aware.summarize local/runs/cycle_aware/gym_v1_5fold

It writes ``summary.json`` (per-fold test metrics, mean and sd) and
``summary.csv`` (one row per fold plus mean / sd rows) into the sweep
directory and prints the table.  Missing folds are reported, not invented.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


def summarise(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and sd of every ``test/*`` metric over the folds."""
    metrics: dict[str, list[float]] = {}
    for fold in results.values():
        for name, value in fold.get("test_metrics", {}).items():
            metrics.setdefault(name, []).append(float(value))
    return {name: {"mean": statistics.fmean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "folds": len(v)} for name, v in metrics.items()}


def summarize_sweep(sweep_dir: Path) -> dict[str, Any]:
    """Collect ``fold_*/result.json`` below ``sweep_dir`` and write the summary files."""
    sweep_dir = Path(sweep_dir)
    results: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for fold_dir in sorted(sweep_dir.glob("fold_*")):
        result = fold_dir / "result.json"
        if result.is_file():
            results[fold_dir.name] = json.loads(result.read_text(encoding="utf-8"))
        else:
            missing.append(fold_dir.name)
    summary = summarise(results)
    payload = {"sweep_dir": str(sweep_dir), "folds": results, "missing": missing, "summary": summary}
    (sweep_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    names = sorted({name for fold in results.values() for name in fold.get("test_metrics", {})})
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", *names])
        for fold_name, fold in results.items():
            writer.writerow([fold_name, *[fold.get("test_metrics", {}).get(name, "") for name in names]])
        if summary:
            writer.writerow(["mean", *[summary[name]["mean"] for name in names]])
            writer.writerow(["sd", *[summary[name]["sd"] for name in names]])
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise per-fold cycle-aware results.")
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = summarize_sweep(args.sweep_dir)
    print(f"folds: {list(payload['folds'])}  missing: {payload['missing']}")
    for name, stats in sorted(payload["summary"].items()):
        if name.startswith("test/") and ("mpjpe" in name or name.endswith("total")):
            print(f"  {name:28s} {stats['mean']:.5f} +- {stats['sd']:.5f}  (n={stats['folds']})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
