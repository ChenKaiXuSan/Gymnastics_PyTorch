"""Subject-disjoint fold files for cross-validation.

A fold file is a JSON document with three subject-id lists::

    {"name": "fold_01", "dataset": "gymnastics", "protocol": "subject_disjoint",
     "train": [...], "val": [...], "test": [...]}

It is the same schema the rotation-aware FreeMan protocol uses
(``src/configs/shared/folds/freeman/fold_0N.json``), so those files can be passed
directly as ``data.fold_json``.  ``make_subject_folds`` builds *k* such folds
whose test sets partition the subjects (every subject is tested exactly
once); the validation set of fold *k* is the test set of fold *k + 1*, and
the remaining subjects train.  An optional stratification key (e.g. the
elderly / student cohort of the private data) keeps the cohort proportions
equal across folds; optional per-subject weights (e.g. session counts on
FreeMan, where subjects have 1 to 117 sessions) balance the amount of data
per group instead of the number of subjects.

Command line::

    python -m fusion.data.folds --dataset gymnastics \\
        --subjects-from local/runs/split_cycle --student-mapping <csv> \\
        --out src/configs/fusion/folds/gymnastics
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


def make_subject_folds(
    subjects: Sequence[str],
    *,
    k: int = 5,
    seed: int = 0,
    stratify: Callable[[str], str] | None = None,
    weights: Mapping[str, float] | None = None,
) -> list[dict[str, list[str]]]:
    """Build ``k`` subject-disjoint folds.

    Args:
        subjects: Subject identifiers.
        k: Number of folds.
        seed: Shuffling seed.
        stratify: Optional function mapping a subject to a stratum label;
            each stratum is split into ``k`` groups separately.
        weights: Optional per-subject weight (e.g. number of sessions).  When
            given, subjects are dealt in descending weight order to the group
            with the smallest total weight so far (ties broken by the seeded
            shuffle), which balances data volume rather than subject count.

    Returns:
        List of ``{"train", "val", "test"}`` dictionaries (sorted id lists).
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    subjects = [str(s) for s in subjects]
    if len(set(subjects)) != len(subjects):
        raise ValueError("subjects must be unique")
    rng = np.random.default_rng(seed)
    strata: dict[str, list[str]] = {}
    for subject in subjects:
        strata.setdefault(stratify(subject) if stratify else "all", []).append(subject)
    groups: list[list[str]] = [[] for _ in range(k)]
    load = [0.0] * k
    for label in sorted(strata):
        members = sorted(strata[label], key=lambda s: (len(s), s))
        order = rng.permutation(len(members))
        shuffled = [members[i] for i in order]
        if weights is None:
            for position, subject in enumerate(shuffled):
                groups[position % k].append(subject)
            continue
        # Greedy balancing: heaviest subjects first, each to the lightest group.
        for subject in sorted(shuffled, key=lambda s: -float(weights.get(s, 0.0))):
            target = min(range(k), key=lambda g: (load[g], len(groups[g])))
            groups[target].append(subject)
            load[target] += float(weights.get(subject, 0.0))
    folds = []
    for fold in range(k):
        test = groups[fold]
        val = groups[(fold + 1) % k]
        train = [s for g in range(k) if g not in (fold, (fold + 1) % k) for s in groups[g]]
        folds.append({"train": sorted(train, key=lambda s: (len(s), s)), "val": sorted(val, key=lambda s: (len(s), s)), "test": sorted(test, key=lambda s: (len(s), s))})
    return folds


def write_fold_files(folds: Sequence[Mapping[str, Sequence[str]]], out_dir: Path, *, dataset: str, extra: Mapping[str, object] | None = None) -> list[Path]:
    """Write ``fold_01.json`` ... below ``out_dir`` and return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, fold in enumerate(folds, start=1):
        payload = {"name": f"fold_{index:02d}", "dataset": dataset, "protocol": "subject_disjoint", **dict(extra or {}), "train": list(fold["train"]), "val": list(fold["val"]), "test": list(fold["test"])}
        path = out_dir / f"fold_{index:02d}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def read_fold_file(path: Path) -> dict[str, list[str]]:
    """Read a fold file and return its ``train`` / ``val`` / ``test`` lists as strings."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {split: [str(s) for s in payload.get(split, [])] for split in ("train", "val", "test")}


def _student_ids(mapping_csv: Path) -> set[str]:
    with Path(mapping_csv).open(encoding="utf-8") as handle:
        return {str(row["person_id"]) for row in csv.DictReader(handle)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate subject-disjoint fold files.")
    parser.add_argument("--dataset", default="gymnastics")
    parser.add_argument("--subjects-from", type=Path, default=Path("local/runs/split_cycle"), help="directory with person_<id> sub-directories")
    parser.add_argument("--student-mapping", type=Path, default=None, help="student_id_mapping.csv for cohort stratification")
    parser.add_argument("--freeman-manifests", type=Path, default=None, help="FreeMan benchmark manifests dir: subjects = every subject_NN_sessions.json, weights = session counts")
    parser.add_argument("--out", type=Path, default=Path("src/configs/fusion/folds/gymnastics"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    weights: dict[str, float] | None = None
    if args.freeman_manifests is not None:
        subjects, weights = [], {}
        for manifest in sorted(args.freeman_manifests.glob("subject_*_sessions.json")):
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            subject = f"{int(payload['subject_id']):02d}"
            subjects.append(subject)
            weights[subject] = float(len(payload.get("sessions", [])))
    else:
        subjects = sorted((p.name.split("_", 1)[1] for p in args.subjects_from.glob("person_*") if p.is_dir()), key=lambda s: (len(s), s))
    stratify = None
    extra: dict[str, object] = {"seed": args.seed}
    if weights is not None:
        extra["weights"] = "sessions"
        extra["sessions_per_subject"] = {s: int(w) for s, w in weights.items()}
    if args.student_mapping is not None:
        students = _student_ids(args.student_mapping)
        stratify = lambda s: "student" if s in students else "elderly"  # noqa: E731
        extra["strata"] = {"student": sum(s in students for s in subjects), "elderly": sum(s not in students for s in subjects)}
    folds = make_subject_folds(subjects, k=args.k, seed=args.seed, stratify=stratify, weights=weights)
    for path in write_fold_files(folds, args.out, dataset=args.dataset, extra=extra):
        fold = read_fold_file(path)
        if weights:
            volume = {split: int(sum(weights.get(s, 0) for s in fold[split])) for split in ("train", "val", "test")}
            print(f"{path}: train {len(fold['train'])} val {len(fold['val'])} test {len(fold['test'])} subjects; sessions {volume}")
        else:
            print(f"{path}: train {len(fold['train'])} val {len(fold['val'])} test {len(fold['test'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
