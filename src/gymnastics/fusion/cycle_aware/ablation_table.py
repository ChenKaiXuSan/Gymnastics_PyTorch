"""Architecture-ablation table over several 5-fold sweeps.

Each sweep directory (``local/runs/cycle_aware/<sweep>/fold_NN/result.json``)
is one row.  Two metric groups are reported because the learned components
of the model behave differently on the two evaluation conditions:

* ``clean``      -- ``test/*`` metrics: the held-out fold on unmodified inputs
  (the measurement-accuracy question).  Procrustes-aligned MPJPE is reported
  for the fused pose and for the reliability-weighted base pose.
* ``corrupted``  -- ``val/*`` metrics: the validation fold with the fixed
  replayed corruption (the robustness question).  ``corrupted_error`` is the
  distance to the clean pseudo-target on damaged joints, for the fused pose
  and for the base pose.

Values are in canonical units (one unit = the sequence's median torso
length, about 0.5 m on the private data) unless ``--scale-mm`` is given, in
which case they are multiplied by that constant for display.

Differences against the baseline sweep are paired by fold (same subjects in
every sweep) and summarised as mean and per-fold sign count; with five folds
a formal test is not meaningful, so none is printed.

Usage::

    python -m gymnastics.fusion.cycle_aware.ablation_table \
        --baseline local/runs/cycle_aware/gymnastics_v1_5fold_seed0 \
        local/runs/cycle_aware/gymnastics_v1_no_film_5fold_seed0 ... \
        --out local/runs/cycle_aware/ablation_gymnastics_seed0
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

CLEAN = ("test/pa_mpjpe", "test/pa_mpjpe_base", "test/weight_entropy")
CORRUPTED = ("val/corrupted_error", "val/corrupted_error_base", "val/pa_mpjpe", "val/weight_entropy", "val/residual")
COLUMNS = CLEAN + CORRUPTED


def load_sweep(sweep_dir: Path) -> dict[str, dict[str, float]]:
    """``{fold: {metric: value}}`` for every finished fold of ``sweep_dir``."""
    folds: dict[str, dict[str, float]] = {}
    for result in sorted(Path(sweep_dir).glob("fold_*/result.json")):
        payload = json.loads(result.read_text(encoding="utf-8"))
        metrics = {**payload.get("fit_metrics", {}), **payload.get("test_metrics", {})}
        folds[result.parent.name] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    return folds


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def build_table(baseline: Path, sweeps: Sequence[Path]) -> list[dict[str, Any]]:
    """One row per sweep with fold mean/sd per metric and paired deltas vs baseline."""
    base = load_sweep(baseline)
    rows: list[dict[str, Any]] = []
    for sweep in (baseline, *sweeps):
        folds = load_sweep(sweep)
        row: dict[str, Any] = {"sweep": Path(sweep).name, "folds": len(folds), "missing": 5 - len(folds)}
        for metric in COLUMNS:
            values = [f[metric] for f in folds.values() if metric in f]
            mean, sd = _mean_sd(values)
            row[f"{metric}:mean"], row[f"{metric}:sd"] = mean, sd
            if sweep != baseline:
                paired = [(folds[k][metric] - base[k][metric]) for k in folds if k in base and metric in folds[k] and metric in base[k]]
                row[f"{metric}:delta"] = statistics.fmean(paired) if paired else float("nan")
                row[f"{metric}:worse_folds"] = sum(d > 0 for d in paired)
        rows.append(row)
    return rows


def render(rows: Sequence[dict[str, Any]], *, scale: float = 1.0, unit: str = "cu") -> str:
    """Markdown table: clean-test and corrupted-validation groups side by side."""
    head = (
        f"| sweep | folds | clean PA fused ({unit}) | clean PA base | Δ vs base (%) | "
        f"corrupted err fused ({unit}) | corrupted err base | Δ vs base (%) | val weight entropy | Δ fused vs baseline sweep ({unit}) |"
    )
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        pa, pab = row["test/pa_mpjpe:mean"] * scale, row["test/pa_mpjpe_base:mean"] * scale
        ce, ceb = row["val/corrupted_error:mean"] * scale, row["val/corrupted_error_base:mean"] * scale
        rel_pa = 100.0 * (pa - pab) / pab if pab else float("nan")
        rel_ce = 100.0 * (ce - ceb) / ceb if ceb else float("nan")
        delta = row.get("test/pa_mpjpe:delta")
        delta_txt = "—" if delta is None else f"{delta * scale:+.3f} ({row['test/pa_mpjpe:worse_folds']}/{row['folds']} worse)"
        lines.append(
            f"| {row['sweep']} | {row['folds']} | {pa:.3f} ± {row['test/pa_mpjpe:sd'] * scale:.3f} | {pab:.3f} | {rel_pa:+.2f} | "
            f"{ce:.3f} ± {row['val/corrupted_error:sd'] * scale:.3f} | {ceb:.3f} | {rel_ce:+.1f} | {row['val/weight_entropy:mean']:.3f} | {delta_txt} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sweeps", type=Path, nargs="*")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="write <out>.md and <out>.json")
    parser.add_argument("--scale-mm", type=float, default=None, help="multiply canonical units by this constant (e.g. 500 for ~0.5 m torso) for display")
    args = parser.parse_args(list(argv) if argv is not None else None)
    rows = build_table(args.baseline, args.sweeps)
    scale, unit = (args.scale_mm, "mm≈") if args.scale_mm else (1.0, "cu")
    text = render(rows, scale=scale, unit=unit)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
        args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
