import json

import numpy as np


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_split_record(split_root, person_id, cycle_indices):
    _write_json(
        split_root
        / f"person_{person_id}"
        / f"alignment_record_{person_id}.json",
        {
            "metadata": {"person_id": str(person_id)},
            "cycles": [
                {"cycle_index": cycle_index} for cycle_index in cycle_indices
            ],
        },
    )


def _write_cycle(
    output_root,
    person_id,
    cycle_index,
    sequence,
    *,
    processed_frames=None,
    missing_pairs=0,
    frame_json_count=None,
    face_error=10.0,
    side_error=12.0,
):
    cycle_root = (
        output_root / f"person_{person_id}" / f"cycle_{cycle_index:03d}"
    )
    cycle_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cycle_root / "joints_3d_sequence.npz", joints_3d=sequence)
    _write_json(
        cycle_root / "summary.json",
        {
            "person_id": str(person_id),
            "cycle_index": cycle_index,
            "processed_frames": (
                int(sequence.shape[0])
                if processed_frames is None
                else processed_frames
            ),
            "missing_pairs": missing_pairs,
            "num_joints": int(sequence.shape[1]),
            "face_reprojection_error_mean_px": face_error,
            "side_reprojection_error_mean_px": side_error,
        },
    )
    if frame_json_count is None:
        frame_json_count = int(sequence.shape[0])
    for frame_index in range(frame_json_count):
        _write_json(
            cycle_root
            / "joints_3d"
            / f"{frame_index:06d}_joints_3d.json",
            {"cycle_frame_index": frame_index},
        )
    return cycle_root


def _write_dataset_summaries(output_root, person_ids):
    people = []
    for person_id in person_ids:
        summary = {"person_id": str(person_id), "cycles": []}
        _write_json(output_root / f"person_{person_id}" / "summary.json", summary)
        people.append(summary)
    _write_json(
        output_root / "summary.json",
        {"num_persons": len(people), "persons": people},
    )


def test_collect_person_summaries_reads_all_people_in_numeric_order(tmp_path):
    from triangulation.sam3d_from_split_cycle import collect_person_summaries

    _write_json(tmp_path / "person_10" / "summary.json", {"person_id": "10"})
    _write_json(tmp_path / "person_2" / "summary.json", {"person_id": "2"})
    (tmp_path / "_camera").mkdir()

    summaries = collect_person_summaries(tmp_path)

    assert [item["person_id"] for item in summaries] == ["2", "10"]


def test_validate_dataset_accepts_complete_data_and_excludes_person_119(tmp_path):
    from triangulation.tools.validate_sam3d_triangulated import validate_dataset

    split_root = tmp_path / "split"
    output_root = tmp_path / "triangulated"
    for person_id in (2, 119):
        _write_split_record(split_root, person_id, [0])
        _write_cycle(
            output_root,
            person_id,
            0,
            np.ones((2, 70, 3), dtype=np.float32),
        )
    _write_dataset_summaries(output_root, [2, 119])

    report = validate_dataset(
        split_root,
        output_root,
        excluded_person_ids={"119"},
    )

    assert report["passed"] is True
    assert report["counts"]["expected_persons"] == 2
    assert report["counts"]["validated_persons"] == 2
    assert report["counts"]["expected_cycles"] == 2
    assert report["counts"]["validated_cycles"] == 2
    assert report["excluded_person_ids"] == ["119"]
    assert report["aggregate_metrics"]["included_cycles"] == 1
    person_rows = {row["person_id"]: row for row in report["persons"]}
    assert person_rows["2"]["quality_status"] == "ok"
    assert person_rows["119"]["quality_status"] == "excluded_low_quality"


def test_validate_dataset_reports_sequence_and_frame_integrity_errors(tmp_path):
    from triangulation.tools.validate_sam3d_triangulated import validate_dataset

    split_root = tmp_path / "split"
    output_root = tmp_path / "triangulated"
    _write_split_record(split_root, 2, [0])
    malformed = np.ones((1, 69, 3), dtype=np.float32)
    malformed[0, 0, 0] = np.nan
    _write_cycle(
        output_root,
        2,
        0,
        malformed,
        processed_frames=2,
        missing_pairs=1,
        frame_json_count=0,
    )
    _write_dataset_summaries(output_root, [2])

    report = validate_dataset(split_root, output_root)

    assert report["passed"] is False
    codes = {error["code"] for error in report["errors"]}
    assert {
        "invalid_sequence_shape",
        "non_finite_sequence",
        "processed_frames_mismatch",
        "missing_pairs",
        "frame_json_count_mismatch",
    } <= codes
