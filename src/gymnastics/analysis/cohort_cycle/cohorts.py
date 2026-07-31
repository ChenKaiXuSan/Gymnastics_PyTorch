"""Cohort mapping contracts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal


Cohort = Literal["elderly", "student"]


@dataclass(frozen=True)
class CohortRecord:
    """Canonical cohort assignment for one available participant."""

    person_id: str
    cohort: Cohort


def sha256_file(path: str | Path) -> str:
    """Return the content identity of a cohort source file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cohort_records(
    student_mapping: str | Path,
    organization_mapping: str | Path,
) -> list[CohortRecord]:
    """Load and reconcile the two authoritative cohort mapping sources."""
    student_path = Path(student_mapping)
    organization_path = Path(organization_mapping)

    with student_path.open(encoding="utf-8-sig", newline="") as handle:
        student_rows = list(csv.DictReader(handle))
    if not student_rows:
        raise ValueError(f"student mapping is empty: {student_path}")

    complete_students: set[str] = set()
    incomplete_students: set[str] = set()
    for row in student_rows:
        person_id = _canonical_person_id(row.get("person_id"))
        is_complete = str(row.get("complete", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        target = complete_students if is_complete else incomplete_students
        target.add(person_id)

    expected_students = {
        str(person_id)
        for person_id in range(81, 139)
        if person_id != 135
    }
    if complete_students != expected_students or incomplete_students != {"135"}:
        raise ValueError(
            "student mapping must contain complete IDs 81-134 and 136-138 "
            "with ID135 as the sole incomplete record"
        )

    expected_cohort: dict[str, Cohort] = {
        str(person_id): "elderly" for person_id in range(1, 81)
    }
    expected_cohort.update(
        {person_id: "student" for person_id in complete_students}
    )

    with organization_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        organization_rows = list(csv.DictReader(handle))
    if not organization_rows:
        raise ValueError(f"organization mapping is empty: {organization_path}")

    for row in organization_rows:
        person_id = _canonical_person_id(row.get("person_id"))
        declared = str(row.get("group", "")).strip().lower()
        if declared not in {"elderly", "student"}:
            raise ValueError(
                f"unknown cohort {declared!r} for person {person_id}"
            )
        expected = expected_cohort.get(person_id)
        if person_id == "135":
            expected = "student"
        if expected != declared:
            raise ValueError(
                "cohort disagreement for person "
                f"{person_id}: organization={declared}, expected={expected}"
            )

    return [
        CohortRecord(person_id, expected_cohort[person_id])
        for person_id in sorted(expected_cohort, key=int)
    ]


def _canonical_person_id(value: object) -> str:
    text = str(value or "").strip()
    if not text.isdigit():
        raise ValueError(f"invalid numeric person ID: {value!r}")
    return str(int(text))
