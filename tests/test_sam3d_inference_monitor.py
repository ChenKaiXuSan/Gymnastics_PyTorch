import pytest

from scripts import monitor_sam3d_inference as monitor


def test_parse_person_ids_supports_ranges_and_gap():
    ids = monitor.parse_person_ids("69-134,136-138")

    assert len(ids) == 69
    assert ids[0] == 69
    assert ids[-1] == 138
    assert 135 not in ids


def test_parse_person_ids_rejects_descending_range():
    with pytest.raises(ValueError, match="descending range"):
        monitor.parse_person_ids("10-8")


def test_collect_progress_requires_finished_log_and_both_views(tmp_path):
    result_root = tmp_path / "person"
    log_root = tmp_path / "person_logs"
    (result_root / "69" / "face").mkdir(parents=True)
    (result_root / "69" / "side").mkdir(parents=True)
    log_root.mkdir()
    (result_root / "69" / "face" / "000000_sam3d_body.npz").write_bytes(b"face")
    (result_root / "69" / "side" / "000000_sam3d_body.npz").write_bytes(b"side")
    (log_root / "69.log").write_text(
        "==== Finished Person: 69 ====\n", encoding="utf-8"
    )

    snapshot = monitor.collect_progress([69, 70], result_root, log_root)

    assert snapshot.completed_ids == (69,)
    assert snapshot.incomplete_ids == (70,)
    assert snapshot.npz_count == 2
    assert snapshot.latest_npz_mtime > 0


def test_collect_progress_keeps_partial_person_incomplete(tmp_path):
    result_root = tmp_path / "person"
    log_root = tmp_path / "person_logs"
    (result_root / "69" / "face").mkdir(parents=True)
    log_root.mkdir()
    (result_root / "69" / "face" / "000000_sam3d_body.npz").write_bytes(b"face")
    (log_root / "69.log").write_text(
        "==== Finished Person: 69 ====\n", encoding="utf-8"
    )

    snapshot = monitor.collect_progress([69], result_root, log_root)

    assert snapshot.completed_ids == ()
    assert snapshot.incomplete_ids == (69,)
