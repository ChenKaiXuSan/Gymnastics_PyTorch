"""Outer-fold construction contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping

from .cohorts import CohortRecord


@dataclass(frozen=True)
class FoldSplit:
    """Person-disjoint roles for one outer cross-fitting fold."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


def build_crossfit_folds(
    records: list[CohortRecord],
    fold0: FoldSplit,
    *,
    seed: int,
) -> dict[int, FoldSplit]:
    """Construct ten cohort-stratified outer folds."""
    cohort_by_person = _validate_records(records)
    all_people = set(cohort_by_person)
    _validate_split(fold0, all_people, fold_index=0)

    fold0_test = set(fold0.test)
    remaining_elderly = sorted(
        (
            person_id
            for person_id, cohort in cohort_by_person.items()
            if cohort == "elderly" and person_id not in fold0_test
        ),
        key=int,
    )
    remaining_students = sorted(
        (
            person_id
            for person_id, cohort in cohort_by_person.items()
            if cohort == "student" and person_id not in fold0_test
        ),
        key=int,
    )
    rng = random.Random(seed)
    rng.shuffle(remaining_elderly)
    rng.shuffle(remaining_students)

    elderly_tests = [
        remaining_elderly[offset : offset + 8]
        for offset in range(0, len(remaining_elderly), 8)
    ]
    student_sizes = (6, 6, 6, 6, 6, 6, 5, 5, 5)
    student_tests: list[list[str]] = []
    cursor = 0
    for size in student_sizes:
        student_tests.append(remaining_students[cursor : cursor + size])
        cursor += size

    if len(elderly_tests) != 9 or cursor != len(remaining_students):
        raise ValueError("cohort inventory cannot satisfy the declared folds")

    folds: dict[int, FoldSplit] = {0: fold0}
    for fold_index in range(1, 10):
        test = set(elderly_tests[fold_index - 1])
        test.update(student_tests[fold_index - 1])
        available_elderly = sorted(
            (
                person_id
                for person_id, cohort in cohort_by_person.items()
                if cohort == "elderly" and person_id not in test
            ),
            key=int,
        )
        available_students = sorted(
            (
                person_id
                for person_id, cohort in cohort_by_person.items()
                if cohort == "student" and person_id not in test
            ),
            key=int,
        )
        validation_rng = random.Random(seed + fold_index)
        validation_rng.shuffle(available_elderly)
        validation_rng.shuffle(available_students)
        val = set(available_elderly[:16] + available_students[:11])
        train = all_people - test - val
        split = FoldSplit(
            train=tuple(sorted(train, key=int)),
            val=tuple(sorted(val, key=int)),
            test=tuple(sorted(test, key=int)),
        )
        _validate_split(split, all_people, fold_index=fold_index)
        folds[fold_index] = split

    published = [
        person_id
        for fold_index in range(10)
        for person_id in folds[fold_index].test
    ]
    if Counter(published) != Counter(all_people):
        raise ValueError("outer test folds must cover every person exactly once")
    return folds


def write_crossfit_artifacts(
    folds: Mapping[int, FoldSplit],
    records: list[CohortRecord],
    output_dir: str | Path,
    *,
    seed: int,
    source_hashes: Mapping[str, str],
) -> None:
    """Write fold JSON files and their cross-fit provenance manifest."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cohort_counts = Counter(record.cohort for record in records)
    cohort_by_person = {
        record.person_id: record.cohort
        for record in sorted(records, key=lambda item: int(item.person_id))
    }

    fold_entries: dict[str, object] = {}
    registry_entries: dict[str, object] = {}
    for index in range(10):
        split = folds[index]
        payload = {
            "test": list(split.test),
            "train": list(split.train),
            "val": list(split.val),
        }
        filename = f"fold_{index:02d}.json"
        encoded = _canonical_json(payload)
        _atomic_write(target / filename, encoded)
        run_id = (
            "all137_a6_e100_seed0"
            if index == 0
            else f"cohort_oof_f{index:02d}_a6_e100_s0"
        )
        fold_entries[f"{index:02d}"] = {
            "cohort_counts": {
                "test": dict(
                    sorted(
                        Counter(
                            cohort_by_person[person_id]
                            for person_id in split.test
                        ).items()
                    )
                ),
                "train": dict(
                    sorted(
                        Counter(
                            cohort_by_person[person_id]
                            for person_id in split.train
                        ).items()
                    )
                ),
                "val": dict(
                    sorted(
                        Counter(
                            cohort_by_person[person_id]
                            for person_id in split.val
                        ).items()
                    )
                ),
            },
            "run_id": run_id,
            "split_file": filename,
            "split_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        registry_entries[f"{index:02d}"] = {
            "outer_fold": index,
            "run_id": run_id,
            "seed": 0,
            "split_file": filename,
        }

    manifest = {
        "schema_version": 1,
        "split_seed": seed,
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "cohorts": cohort_by_person,
        "source_hashes": dict(sorted(source_hashes.items())),
        "folds": fold_entries,
    }
    registry = {
        "schema_version": 1,
        "primary_seed": 0,
        "runs": registry_entries,
    }
    _atomic_write(
        target / "crossfit_manifest.json",
        _canonical_json(manifest),
    )
    _atomic_write(
        target / "run_registry.json",
        _canonical_json(registry),
    )


def load_fold_split(path: str | Path) -> FoldSplit:
    """Load one existing rotation-aware split manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fold manifest must be a JSON object")
    try:
        return FoldSplit(
            train=tuple(str(value) for value in payload["train"]),
            val=tuple(str(value) for value in payload["val"]),
            test=tuple(str(value) for value in payload["test"]),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("fold manifest requires train, val, and test lists") from error


def _validate_records(records: list[CohortRecord]) -> dict[str, str]:
    by_person = {record.person_id: record.cohort for record in records}
    if len(by_person) != len(records):
        raise ValueError("cohort records contain duplicate people")
    counts = Counter(by_person.values())
    if counts != Counter({"elderly": 80, "student": 57}):
        raise ValueError(
            "cohort records must contain 80 elderly and 57 student people"
        )
    return by_person


def _validate_split(
    split: FoldSplit,
    all_people: set[str],
    *,
    fold_index: int,
) -> None:
    train = set(split.train)
    val = set(split.val)
    test = set(split.test)
    if len(train) != len(split.train) or len(val) != len(split.val) or len(
        test
    ) != len(split.test):
        raise ValueError(f"fold {fold_index} contains duplicate role members")
    if train & val or train & test or val & test:
        raise ValueError(f"fold {fold_index} roles are not person-disjoint")
    if train | val | test != all_people:
        raise ValueError(f"fold {fold_index} does not cover the cohort inventory")
    if len(val) != 27:
        raise ValueError(f"fold {fold_index} must contain 27 validation people")


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
