#!/usr/bin/env python3
"""Validate SAM3D triangulated outputs against split-cycle expectations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Set

import numpy as np


DEFAULT_SPLIT_ROOT = Path("local/runs/split_cycle")
DEFAULT_OUTPUT_ROOT = Path("/home/data/xchen/gymnastics/sam3d_triangulated/person")
DEFAULT_REPORT = Path("local/runs/analysis/triangulated_results/validation_summary.json")


def _numeric_id(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 2**31 - 1, value


def _person_id(path: Path) -> str:
    return path.name.removeprefix("person_")


def _cycle_index(path: Path) -> int:
    return int(path.name.removeprefix("cycle_"))


def _error(
    errors: List[Dict[str, Any]],
    code: str,
    person_id: str | None = None,
    cycle_index: int | None = None,
    **details: Any,
) -> None:
    row: Dict[str, Any] = {"code": code}
    if person_id is not None:
        row["person_id"] = person_id
    if cycle_index is not None:
        row["cycle_index"] = cycle_index
    row.update(details)
    errors.append(row)


def _load_expected_cycles(split_root: Path) -> Dict[str, Set[int]]:
    expected: Dict[str, Set[int]] = {}
    records = sorted(
        split_root.glob("person_*/alignment_record_*.json"),
        key=lambda path: _numeric_id(_person_id(path.parent)),
    )
    for record_path in records:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        person_id = str(
            record.get("metadata", {}).get(
                "person_id", _person_id(record_path.parent)
            )
        )
        expected[person_id] = {
            int(cycle["cycle_index"]) for cycle in record.get("cycles", [])
        }
    return expected


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(mean(values)) if values else None


def validate_dataset(
    split_root: Path,
    output_root: Path,
    warning_threshold_px: float = 60.0,
    excluded_person_ids: set[str] | None = None,
) -> Dict[str, Any]:
    """Validate every split-cycle expectation against triangulated outputs."""
    split_root = Path(split_root)
    output_root = Path(output_root)
    excluded = {str(value) for value in (excluded_person_ids or set())}
    expected = _load_expected_cycles(split_root)
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    people: List[Dict[str, Any]] = []
    aggregate_face_errors: List[float] = []
    aggregate_side_errors: List[float] = []
    aggregate_cycles = 0
    validated_cycles = 0
    validated_persons = 0

    actual_person_ids = {
        _person_id(path)
        for path in output_root.glob("person_*")
        if path.is_dir()
    }
    if actual_person_ids != set(expected):
        _error(
            errors,
            "person_inventory_mismatch",
            expected=sorted(expected, key=_numeric_id),
            actual=sorted(actual_person_ids, key=_numeric_id),
        )

    for person_id in sorted(expected, key=_numeric_id):
        expected_indices = expected[person_id]
        person_root = output_root / f"person_{person_id}"
        person_error_start = len(errors)
        person_warning_start = len(warnings)

        if not person_root.is_dir():
            _error(errors, "missing_person_directory", person_id)
            actual_indices: Set[int] = set()
        else:
            actual_indices = {
                _cycle_index(path)
                for path in person_root.glob("cycle_*")
                if path.is_dir()
            }

        person_summary_path = person_root / "summary.json"
        if not person_summary_path.is_file():
            _error(errors, "missing_person_summary", person_id)
        else:
            validated_persons += 1

        if actual_indices != expected_indices:
            _error(
                errors,
                "cycle_inventory_mismatch",
                person_id,
                expected=sorted(expected_indices),
                actual=sorted(actual_indices),
            )

        for cycle_index in sorted(expected_indices):
            cycle_root = person_root / f"cycle_{cycle_index:03d}"
            if not cycle_root.is_dir():
                _error(errors, "missing_cycle_directory", person_id, cycle_index)
                continue

            summary_path = cycle_root / "summary.json"
            sequence_path = cycle_root / "joints_3d_sequence.npz"
            if not summary_path.is_file():
                _error(errors, "missing_cycle_summary", person_id, cycle_index)
                continue
            if not sequence_path.is_file():
                _error(errors, "missing_sequence", person_id, cycle_index)
                continue

            try:
                summary = _load_json(summary_path)
            except (OSError, json.JSONDecodeError) as exc:
                _error(
                    errors,
                    "invalid_cycle_summary",
                    person_id,
                    cycle_index,
                    detail=str(exc),
                )
                continue

            try:
                with np.load(sequence_path, allow_pickle=False) as archive:
                    sequence = np.asarray(archive["joints_3d"])
            except (OSError, KeyError, ValueError) as exc:
                _error(
                    errors,
                    "invalid_sequence_archive",
                    person_id,
                    cycle_index,
                    detail=str(exc),
                )
                continue

            validated_cycles += 1
            frame_count = int(sequence.shape[0]) if sequence.ndim >= 1 else 0
            if (
                sequence.ndim != 3
                or sequence.shape[1:] != (70, 3)
                or frame_count == 0
            ):
                _error(
                    errors,
                    "invalid_sequence_shape",
                    person_id,
                    cycle_index,
                    shape=list(sequence.shape),
                )
            if not np.isfinite(sequence).all():
                _error(
                    errors,
                    "non_finite_sequence",
                    person_id,
                    cycle_index,
                    non_finite_values=int((~np.isfinite(sequence)).sum()),
                )

            processed_frames = int(summary.get("processed_frames", 0) or 0)
            if processed_frames != frame_count:
                _error(
                    errors,
                    "processed_frames_mismatch",
                    person_id,
                    cycle_index,
                    expected=frame_count,
                    actual=processed_frames,
                )

            missing_pairs = int(summary.get("missing_pairs", 0) or 0)
            if missing_pairs != 0:
                _error(
                    errors,
                    "missing_pairs",
                    person_id,
                    cycle_index,
                    actual=missing_pairs,
                )

            frame_json_count = len(
                list((cycle_root / "joints_3d").glob("*_joints_3d.json"))
            )
            if frame_json_count != frame_count:
                _error(
                    errors,
                    "frame_json_count_mismatch",
                    person_id,
                    cycle_index,
                    expected=frame_count,
                    actual=frame_json_count,
                )

            cycle_errors: Dict[str, float] = {}
            for view in ("face", "side"):
                value = summary.get(f"{view}_reprojection_error_mean_px")
                if value is None:
                    continue
                numeric_value = float(value)
                cycle_errors[view] = numeric_value
                if numeric_value > warning_threshold_px:
                    warnings.append(
                        {
                            "code": "high_reprojection_error",
                            "person_id": person_id,
                            "cycle_index": cycle_index,
                            "view": view,
                            "actual": numeric_value,
                            "threshold": warning_threshold_px,
                        }
                    )

            if person_id not in excluded:
                aggregate_cycles += 1
                if "face" in cycle_errors:
                    aggregate_face_errors.append(cycle_errors["face"])
                if "side" in cycle_errors:
                    aggregate_side_errors.append(cycle_errors["side"])

        person_error_count = len(errors) - person_error_start
        person_warning_count = len(warnings) - person_warning_start
        if person_error_count:
            quality_status = "error"
        elif person_id in excluded:
            quality_status = "excluded_low_quality"
        elif person_warning_count:
            quality_status = "warning"
        else:
            quality_status = "ok"
        people.append(
            {
                "person_id": person_id,
                "expected_cycles": len(expected_indices),
                "output_cycles": len(actual_indices),
                "error_count": person_error_count,
                "warning_count": person_warning_count,
                "quality_status": quality_status,
                "excluded_from_aggregate": person_id in excluded,
            }
        )

    root_summary_path = output_root / "summary.json"
    if not root_summary_path.is_file():
        _error(errors, "missing_root_summary")
    else:
        try:
            root_summary = _load_json(root_summary_path)
            root_person_ids = [
                str(item["person_id"]) for item in root_summary.get("persons", [])
            ]
            expected_person_ids = sorted(expected, key=_numeric_id)
            if (
                int(root_summary.get("num_persons", -1)) != len(expected_person_ids)
                or root_person_ids != expected_person_ids
            ):
                _error(
                    errors,
                    "root_summary_inventory_mismatch",
                    expected=expected_person_ids,
                    actual=root_person_ids,
                    declared_count=root_summary.get("num_persons"),
                )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            _error(errors, "invalid_root_summary", detail=str(exc))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_root": str(split_root),
        "output_root": str(output_root),
        "warning_threshold_px": float(warning_threshold_px),
        "excluded_person_ids": sorted(excluded, key=_numeric_id),
        "passed": not errors,
        "counts": {
            "expected_persons": len(expected),
            "validated_persons": validated_persons,
            "expected_cycles": sum(len(indices) for indices in expected.values()),
            "validated_cycles": validated_cycles,
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "aggregate_metrics": {
            "included_cycles": aggregate_cycles,
            "face_reprojection_error_mean_px": _mean_or_none(
                aggregate_face_errors
            ),
            "side_reprojection_error_mean_px": _mean_or_none(
                aggregate_side_errors
            ),
        },
        "persons": people,
        "errors": errors,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SAM3D triangulated outputs against split-cycle records."
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--warning-threshold-px", type=float, default=60.0)
    parser.add_argument("--exclude-person", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_dataset(
        split_root=args.split_root,
        output_root=args.output_root,
        warning_threshold_px=args.warning_threshold_px,
        excluded_person_ids=set(args.exclude_person),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.report.with_suffix(args.report.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp_path.replace(args.report)
    print(json.dumps(report["counts"], ensure_ascii=False))
    print(f"[INFO] Validation report: {args.report}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
