"""Paired statistics between two result files, over subjects rather than folds.

A 5-fold comparison can never be significant: the Wilcoxon signed-rank test
floors at p = 0.0625 for n = 5. The cross-validation folds are subject-
disjoint, so every subject is a held-out measurement of both methods and the
natural unit is the subject (137 private participants, 37 FreeMan subjects).

    python -m fusion external-published compare \
        --a local/runs/external_published/model/<run>/summary_model_12joints.json \
        --b local/runs/external_published/canonpose/gymnastics/summary_canonical_average_12joints.json

Reported per comparison: the mean paired difference (B - A, positive = A is
better), a bootstrap 95 % CI of that mean, the share of subjects on which A
wins, the Wilcoxon signed-rank p-value and, when several comparisons are
given at once, Holm-corrected p-values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def load_subjects(path: Path) -> dict[str, float]:
    """``{subject: PA-MPJPE in mm}`` from a summary written by :mod:`evaluate`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "per_subject" not in payload:
        raise ValueError(f"{path} has no per-subject errors; re-run the evaluation with the current code")
    return {subject: 1000.0 * values["pa_mpjpe"] for subject, values in payload["per_subject"].items()}


def paired(a: dict[str, float], b: dict[str, float], *, bootstrap: int = 10000, seed: int = 0) -> dict[str, Any]:
    """Paired comparison of A (ours) against B over the subjects both cover."""
    from scipy import stats

    subjects = sorted(set(a) & set(b), key=lambda s: (len(s), s))
    if not subjects:
        raise ValueError("the two files share no subjects")
    x = np.array([a[s] for s in subjects])
    y = np.array([b[s] for s in subjects])
    diff = y - x  # positive: B is worse, i.e. A is better
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(bootstrap, len(diff)))].mean(axis=1)
    wilcoxon = stats.wilcoxon(y, x) if len(subjects) > 1 and np.any(diff != 0) else None
    return {
        "subjects": len(subjects),
        "a_mean_mm": float(x.mean()),
        "b_mean_mm": float(y.mean()),
        "diff_mean_mm": float(diff.mean()),
        "diff_ci95_mm": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "diff_median_mm": float(np.median(diff)),
        "a_better_subjects": int((diff > 0).sum()),
        "wilcoxon_p": float(wilcoxon.pvalue) if wilcoxon is not None else float("nan"),
        "cohens_dz": float(diff.mean() / diff.std(ddof=1)) if len(subjects) > 1 and diff.std(ddof=1) > 0 else float("nan"),
    }


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return [float(v) for v in adjusted]


def compare_many(a: Path, others: Sequence[tuple[str, Path]], *, bootstrap: int = 10000, seed: int = 0) -> dict[str, Any]:
    ours = load_subjects(a)
    rows = [{"method": name, **paired(ours, load_subjects(path), bootstrap=bootstrap, seed=seed), "file": str(path)} for name, path in others]
    for row, adjusted in zip(rows, holm([row["wilcoxon_p"] for row in rows])):
        row["holm_p"] = adjusted
    return {"reference": str(a), "comparisons": rows}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, required=True, help="our summary (the reference of every comparison)")
    parser.add_argument("--b", type=Path, nargs="+", required=True, help="one or more summaries to compare against")
    parser.add_argument("--labels", nargs="*", default=None, help="names for the --b files (default: their directories)")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    labels = args.labels or [f"{p.parts[-3]}/{p.stem}" for p in args.b]
    payload = compare_many(args.a, list(zip(labels, args.b)), bootstrap=args.bootstrap, seed=args.seed)
    print(f"reference: {payload['reference']}")
    print(f"{'method':34s} {'n':>4s} {'theirs':>7s} {'ours':>6s} {'diff':>7s}  {'95% CI':>18s} {'ours better':>12s} {'p':>8s} {'p_Holm':>8s}")
    for row in payload["comparisons"]:
        ci = f"[{row['diff_ci95_mm'][0]:+.2f}, {row['diff_ci95_mm'][1]:+.2f}]"
        print(f"{row['method']:34s} {row['subjects']:4d} {row['b_mean_mm']:7.1f} {row['a_mean_mm']:6.1f} {row['diff_mean_mm']:+7.2f}  {ci:>18s} {row['a_better_subjects']:6d}/{row['subjects']:<5d} {row['wilcoxon_p']:8.1e} {row['holm_p']:8.1e}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"-> {args.json}")
    return 0
