from __future__ import annotations

import json
from pathlib import Path

from gymnastics.analysis.cohort_cycle.cohorts import CohortRecord
from gymnastics.analysis.cohort_cycle.folds import (
    FoldSplit,
    build_crossfit_folds,
    write_crossfit_artifacts,
)


FOLD0_TEST = (
    "1",
    "106",
    "116",
    "117",
    "130",
    "136",
    "24",
    "36",
    "49",
    "51",
    "52",
    "60",
    "79",
    "85",
)


def _records() -> list[CohortRecord]:
    elderly = [
        CohortRecord(str(person_id), "elderly")
        for person_id in range(1, 81)
    ]
    students = [
        CohortRecord(str(person_id), "student")
        for person_id in range(81, 139)
        if person_id != 135
    ]
    return elderly + students


def _fold0() -> FoldSplit:
    elderly = [
        record.person_id
        for record in _records()
        if record.cohort == "elderly" and record.person_id not in FOLD0_TEST
    ]
    students = [
        record.person_id
        for record in _records()
        if record.cohort == "student" and record.person_id not in FOLD0_TEST
    ]
    val = elderly[:16] + students[:11]
    train = elderly[16:] + students[11:]
    return FoldSplit(
        train=tuple(train),
        val=tuple(val),
        test=FOLD0_TEST,
    )


def test_crossfit_folds_preserve_fold0_and_cover_each_person_once():
    """A moved fold-0 person or duplicate OOF assignment is leakage."""
    folds = build_crossfit_folds(_records(), _fold0(), seed=20260728)

    assert folds[0] == _fold0()
    assert [len(folds[index].test) for index in range(10)] == [
        14,
        14,
        14,
        14,
        14,
        14,
        14,
        13,
        13,
        13,
    ]
    test_people = [
        person_id
        for index in range(10)
        for person_id in folds[index].test
    ]
    assert len(test_people) == 137
    assert len(set(test_people)) == 137
    assert set(test_people) == {
        record.person_id for record in _records()
    }


def test_crossfit_folds_have_declared_cohort_and_role_counts():
    """Wrong stratification would confound cohort with outer-fold size."""
    folds = build_crossfit_folds(_records(), _fold0(), seed=20260728)
    cohort = {record.person_id: record.cohort for record in _records()}

    for index, split in folds.items():
        assert not (set(split.train) & set(split.val))
        assert not (set(split.train) & set(split.test))
        assert not (set(split.val) & set(split.test))
        assert len(split.val) == 27
        assert len(split.train) == (96 if index < 7 else 97)
        assert sum(cohort[p] == "elderly" for p in split.test) == 8
        expected_students = 6 if index < 7 else 5
        assert sum(cohort[p] == "student" for p in split.test) == expected_students
        assert sum(cohort[p] == "elderly" for p in split.val) == 16
        assert sum(cohort[p] == "student" for p in split.val) == 11


def test_crossfit_generation_is_byte_deterministic(tmp_path: Path):
    """A rerun with identical sources and seed must not change any manifest."""
    folds = build_crossfit_folds(_records(), _fold0(), seed=20260728)
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_crossfit_artifacts(
        folds,
        _records(),
        first,
        seed=20260728,
        source_hashes={"students": "abc", "organization": "def"},
    )
    write_crossfit_artifacts(
        folds,
        _records(),
        second,
        seed=20260728,
        source_hashes={"students": "abc", "organization": "def"},
    )

    first_files = {
        path.name: path.read_bytes() for path in sorted(first.glob("*.json"))
    }
    second_files = {
        path.name: path.read_bytes() for path in sorted(second.glob("*.json"))
    }
    assert first_files == second_files

    manifest = json.loads(first_files["crossfit_manifest.json"])
    assert manifest["schema_version"] == 1
    assert manifest["split_seed"] == 20260728
    assert manifest["cohort_counts"] == {"elderly": 80, "student": 57}
    assert manifest["folds"]["00"]["run_id"] == "all137_a6_e100_seed0"
    assert manifest["folds"]["09"]["run_id"] == "cohort_oof_f09_a6_e100_s0"
