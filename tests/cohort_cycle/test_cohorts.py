from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gymnastics.analysis.cohort_cycle.cohorts import (
    load_cohort_records,
    sha256_file,
)


def _write_mapping_fixtures(root: Path) -> tuple[Path, Path]:
    student_path = root / "student_id_mapping.csv"
    with student_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("student_id", "person_id", "complete"),
        )
        writer.writeheader()
        for person_id in range(81, 139):
            writer.writerow(
                {
                    "student_id": f"S{person_id - 80}",
                    "person_id": str(person_id),
                    "complete": str(person_id != 135),
                }
            )

    organization_path = root / "organization.csv"
    with organization_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("group", "original_id", "person_id", "view", "status"),
        )
        writer.writeheader()
        for person_id in range(69, 81):
            for view in ("face", "side"):
                writer.writerow(
                    {
                        "group": "elderly",
                        "original_id": f"ID{person_id}",
                        "person_id": str(person_id),
                        "view": view,
                        "status": "copied",
                    }
                )
        for person_id in range(81, 139):
            for view in ("face", "side"):
                writer.writerow(
                    {
                        "group": "student",
                        "original_id": f"S{person_id - 80}",
                        "person_id": str(person_id),
                        "view": view,
                        "status": (
                            "missing_source" if person_id == 135 else "copied"
                        ),
                    }
                )
    return student_path, organization_path


def test_load_cohorts_excludes_incomplete_student_and_assigns_declared_groups(
    tmp_path: Path,
):
    """Including ID135 or assigning an ID to the wrong cohort is a data bug."""
    student_path, organization_path = _write_mapping_fixtures(tmp_path)

    records = load_cohort_records(student_path, organization_path)

    by_id = {record.person_id: record.cohort for record in records}
    assert len(records) == 137
    assert list(by_id.values()).count("elderly") == 80
    assert list(by_id.values()).count("student") == 57
    assert "135" not in by_id
    assert by_id["1"] == "elderly"
    assert by_id["80"] == "elderly"
    assert by_id["81"] == "student"
    assert by_id["138"] == "student"


def test_load_cohorts_rejects_organization_disagreement(tmp_path: Path):
    """Silently accepting a contradictory source mapping would corrupt folds."""
    student_path, organization_path = _write_mapping_fixtures(tmp_path)
    text = organization_path.read_text(encoding="utf-8")
    organization_path.write_text(
        text.replace("student,S1,81,face", "elderly,S1,81,face"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cohort disagreement"):
        load_cohort_records(student_path, organization_path)


def test_sha256_file_changes_when_mapping_changes(tmp_path: Path):
    """Source provenance must detect a changed cohort mapping."""
    student_path, _ = _write_mapping_fixtures(tmp_path)
    original_hash = sha256_file(student_path)

    with student_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    assert sha256_file(student_path) != original_hash
