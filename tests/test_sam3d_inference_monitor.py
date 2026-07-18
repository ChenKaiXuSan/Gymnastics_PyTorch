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


def test_scan_new_fatal_errors_uses_offset_and_ignores_frame_warning(tmp_path):
    run_log = tmp_path / "run.log"
    run_log.write_text("No person detected in frame 3\n", encoding="utf-8")
    errors, offset = monitor.scan_new_fatal_errors(run_log, 0)
    assert errors == []

    with run_log.open("a", encoding="utf-8") as handle:
        handle.write("GPU 0 \u5904\u7406 83 \u65f6\u51fa\u9519: CUDA out of memory\n")
    errors, new_offset = monitor.scan_new_fatal_errors(run_log, offset)

    assert errors == ["GPU 0 \u5904\u7406 83 \u65f6\u51fa\u9519: CUDA out of memory"]
    assert new_offset > offset


def test_scan_resets_offset_when_log_is_truncated(tmp_path):
    run_log = tmp_path / "run.log"
    run_log.write_text("CUDA error\n", encoding="utf-8")

    errors, offset = monitor.scan_new_fatal_errors(run_log, 10_000)

    assert errors == ["CUDA error"]
    assert offset == run_log.stat().st_size


def test_process_is_active_requires_main_and_run_match():
    commands = [
        "python -m SAM3Dbody.main infer.workers_per_gpu=2",
        "python scripts/monitor_sam3d_inference.py",
    ]

    assert monitor.process_is_active(commands, "infer.workers_per_gpu=2")
    assert not monitor.process_is_active(commands, "infer.workers_per_gpu=3")


def test_state_round_trip_and_atomic_save(tmp_path):
    state_path = tmp_path / "state.json"
    state = monitor.MonitorState(
        started_at=100.0,
        last_progress_at=120.0,
        last_npz_count=10,
        latest_npz_mtime=110.0,
        log_offset=42,
        sent_fingerprints=["abc"],
        stalled=True,
    )

    monitor.save_state(state_path, state)

    assert monitor.load_state(state_path, now=999.0) == state
    assert not state_path.with_suffix(".json.tmp").exists()


def test_update_progress_state_detects_progress_and_stall_age():
    state = monitor.MonitorState.new(now=100.0)
    snapshot = monitor.ProgressSnapshot((), (69,), 3, 90.0)

    progressed = monitor.update_progress_state(state, snapshot, now=120.0)

    assert progressed
    assert state.last_progress_at == 120.0
    assert not monitor.is_stalled(state, now=1_919.0, stall_seconds=1_800)
    assert monitor.is_stalled(state, now=1_920.0, stall_seconds=1_800)


def test_load_smtp_settings_requires_private_file(tmp_path):
    config = tmp_path / "smtp.env"
    config.write_text(
        "SMTP_HOST=smtp.gmail.com\n"
        "SMTP_PORT=465\n"
        "SMTP_USER=chenkaixusan@gmail.com\n"
        "SMTP_APP_PASSWORD=abcdefghijklmnop\n"
        "EMAIL_TO=chenkaixusan@gmail.com\n",
        encoding="utf-8",
    )
    config.chmod(0o644)
    with pytest.raises(PermissionError, match="0600"):
        monitor.load_smtp_settings(config)

    config.chmod(0o600)
    settings = monitor.load_smtp_settings(config)

    assert settings.user == "chenkaixusan@gmail.com"
    assert settings.port == 465
    assert "abcdefghijklmnop" not in repr(settings)


def test_send_email_retries_without_exposing_password():
    attempts = []

    class FakeSmtp:
        def __init__(self, host, port, timeout):
            attempts.append((host, port, timeout))

        def __enter__(self):
            if len(attempts) < 3:
                raise OSError("temporary network failure")
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def login(self, user, password):
            assert user == "chenkaixusan@gmail.com"
            assert password == "secret-app-password"

        def send_message(self, message):
            assert message["To"] == "chenkaixusan@gmail.com"
            assert "secret-app-password" not in message.as_string()

    settings = monitor.SmtpSettings(
        "smtp.gmail.com",
        465,
        "chenkaixusan@gmail.com",
        "secret-app-password",
        "chenkaixusan@gmail.com",
    )
    notification = monitor.Notification("ERROR", "fp", "subject", "body")

    monitor.send_email(
        settings, notification, smtp_factory=FakeSmtp, sleep_fn=lambda _: None
    )

    assert len(attempts) == 3


def test_send_email_raises_after_three_failed_attempts():
    attempts = []

    class AlwaysFailsSmtp:
        def __init__(self, host, port, timeout):
            attempts.append((host, port, timeout))

        def __enter__(self):
            raise OSError("SMTP unavailable")

        def __exit__(self, exc_type, exc, traceback):
            return False

    settings = monitor.SmtpSettings(
        "smtp.gmail.com",
        465,
        "chenkaixusan@gmail.com",
        "secret-app-password",
        "chenkaixusan@gmail.com",
    )
    notification = monitor.Notification("ERROR", "fp", "subject", "body")

    with pytest.raises(OSError, match="SMTP unavailable"):
        monitor.send_email(
            settings,
            notification,
            smtp_factory=AlwaysFailsSmtp,
            sleep_fn=lambda _: None,
        )

    assert len(attempts) == 3


def test_make_notification_includes_progress_and_stable_fingerprint():
    snapshot = monitor.ProgressSnapshot((69,), (70, 71), 42, 123.0)

    first = monitor.make_notification("STOPPED", "worker exited", snapshot)
    second = monitor.make_notification("STOPPED", "worker exited", snapshot)

    assert first.subject == "[Gymnastics][SAM3D-Body] STOPPED"
    assert "Completed persons: 1" in first.body
    assert "Incomplete persons: 70,71" in first.body
    assert "NPZ files: 42" in first.body
    assert first.fingerprint == second.fingerprint


def test_send_once_records_only_successful_delivery():
    state = monitor.MonitorState.new(now=100.0)
    notification = monitor.Notification("ERROR", "same", "subject", "body")
    deliveries = []

    def sender(item):
        deliveries.append(item)
        return True

    assert monitor.send_once(notification, state, sender)
    assert monitor.send_once(notification, state, sender)
    assert deliveries == [notification]
    assert state.sent_fingerprints == ["same"]


def test_send_once_does_not_record_failed_delivery():
    state = monitor.MonitorState.new(now=100.0)
    notification = monitor.Notification("ERROR", "retry-me", "subject", "body")

    assert not monitor.send_once(notification, state, lambda _: False)
    assert state.sent_fingerprints == []
