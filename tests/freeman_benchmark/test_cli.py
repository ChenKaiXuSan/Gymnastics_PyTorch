from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import gymnastics.benchmarks.freeman.cli as freeman_cli
from gymnastics.benchmarks.freeman.cli import (
    DefaultStageOperations,
    StageOperations,
    main,
    partition_subjects,
    reset_forced_stage,
    run_subjects,
)


class RecordingOperations(StageOperations):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def inspect(self, config, *, dry_run=False):
        self.calls.append(("inspect", dict(config)))

    def download(self, config):
        self.calls.append(("download", dict(config)))

    def infer(self, config):
        self.calls.append(("infer", dict(config)))

    def fuse(self, config):
        self.calls.append(("fuse", dict(config)))

    def evaluate(self, config):
        self.calls.append(("evaluate", dict(config)))

    def report(self, config):
        self.calls.append(("report", dict(config)))

    def run(
        self,
        config,
        *,
        force_stage=None,
        keep_workspace=False,
        dry_run=False,
        devices=None,
    ):
        self.calls.append(("run", dict(config)))
        self.force_stage = force_stage
        self.keep_workspace = keep_workspace
        self.dry_run = dry_run
        self.devices = devices


@pytest.mark.parametrize("stage", ["inspect", "infer", "fuse", "evaluate", "report"])
def test_staged_commands_only_call_their_stage(stage: str) -> None:
    operations = RecordingOperations()

    assert main([stage], operations=operations) == 0

    assert [name for name, _ in operations.calls] == [stage]


def test_download_calls_inspect_before_mutating_download() -> None:
    operations = RecordingOperations()

    assert main(["download"], operations=operations) == 0

    assert [name for name, _ in operations.calls] == ["inspect", "download"]


def test_cli_overrides_only_documented_benchmark_scope() -> None:
    operations = RecordingOperations()

    assert (
        main(
            [
                "run",
                "--subject",
                "7",
                "3",
                "--fps",
                "30",
                "--frame-stride",
                "4",
                "--force-stage",
                "infer",
                "--keep-workspace",
                "--dry-run",
            ],
            operations=operations,
        )
        == 0
    )

    config = operations.calls[0][1]
    assert config["dataset"]["subjects"] == [3, 7]
    assert config["dataset"]["fps_subsets"] == [30]
    assert config["dataset"]["frame_stride"] == 4
    assert config["evaluation"]["headline_eligible"] is False
    assert config["evaluation"]["diagnostic_reason"] == "frame_stride_not_one"
    assert operations.force_stage == "infer"
    assert operations.keep_workspace is True
    assert operations.dry_run is True


def test_round_robin_partition_is_disjoint_and_complete() -> None:
    assert partition_subjects([5, 1, 4, 3, 2], 2) == (
        (1, 3, 5),
        (2, 4),
    )


def test_run_cli_forwards_validated_devices() -> None:
    operations = RecordingOperations()

    assert main(["run", "--devices", "0", "1"], operations=operations) == 0

    assert operations.devices == (0, 1)


@pytest.mark.parametrize("devices", [("0", "0"), ("-1", "0")])
def test_run_cli_rejects_duplicate_or_negative_devices(devices) -> None:
    with pytest.raises(SystemExit):
        main(["run", "--devices", *devices], operations=RecordingOperations())


def test_worker_uses_private_state_and_logical_device_zero(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed = {}
    config = {
        "paths": {
            "output_root": tmp_path / "output",
            "work_root": tmp_path / "work",
        },
        "sam3d": {"device": 7},
    }

    def process(_self, worker_config, subject):
        observed["sam3d_device"] = worker_config["sam3d"]["device"]
        observed["subject"] = subject

    def capture_run(
        subjects,
        *,
        state_path,
        process,
        cleanup,
        keep_workspace,
    ):
        del cleanup
        observed["subjects"] = tuple(subjects)
        observed["state_path"] = state_path
        observed["keep_workspace"] = keep_workspace
        observed["cuda_visible_devices"] = freeman_cli.os.environ[
            "CUDA_VISIBLE_DEVICES"
        ]
        process(subjects[0])

    monkeypatch.setattr(DefaultStageOperations, "_process_subject", process)
    monkeypatch.setattr(freeman_cli, "run_subjects", capture_run)
    state_path = tmp_path / "worker.json"

    freeman_cli._run_device_worker(
        config,
        1,
        [2, 4],
        state_path,
        False,
    )

    assert observed == {
        "subjects": (2, 4),
        "state_path": state_path,
        "keep_workspace": False,
        "cuda_visible_devices": "1",
        "sam3d_device": 0,
        "subject": 2,
    }
    assert config["sam3d"]["device"] == 7


def test_merge_preserves_worker_terminal_states_and_canonical_completions(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "run_state.json"
    canonical.write_text(
        json.dumps(
            {
                "stages": {"download": {"status": "complete"}},
                "subjects": {"3": {"status": "complete"}},
            }
        ),
        encoding="utf-8",
    )
    worker0 = tmp_path / "worker0.json"
    worker0.write_text(
        json.dumps({"subjects": {"1": {"status": "complete"}}}),
        encoding="utf-8",
    )
    worker1 = tmp_path / "worker1.json"
    worker1.write_text(
        json.dumps(
            {
                "subjects": {
                    "2": {
                        "status": "failed",
                        "error_type": "RuntimeError",
                        "error_message": "broken",
                    },
                    "3": {"status": "failed"},
                }
            }
        ),
        encoding="utf-8",
    )

    merged = freeman_cli._merge_worker_states(
        canonical,
        [worker0, worker1],
    )

    assert merged["stages"]["download"]["status"] == "complete"
    assert merged["subjects"]["1"]["status"] == "complete"
    assert merged["subjects"]["2"]["status"] == "failed"
    assert merged["subjects"]["3"]["status"] == "complete"
    assert json.loads(canonical.read_text(encoding="utf-8")) == merged


def test_parallel_coordinator_skips_complete_and_merges_disjoint_assignments(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    canonical = output / "run_state.json"
    canonical.write_text(
        json.dumps(
            {
                "stages": {},
                "subjects": {"1": {"status": "complete"}},
            }
        ),
        encoding="utf-8",
    )
    started = []
    joined = []

    class FakeProcess:
        def __init__(self, *, target, args):
            del target
            self.args = args
            self.exitcode = 0

        def start(self):
            _, device, subjects, state_path, _ = self.args
            started.append((device, tuple(subjects)))
            Path(state_path).write_text(
                json.dumps(
                    {
                        "subjects": {
                            str(subject): {"status": "complete"}
                            for subject in subjects
                        }
                    }
                ),
                encoding="utf-8",
            )

        def join(self):
            joined.append(self.args[1])

    class FakeContext:
        Process = FakeProcess

    monkeypatch.setattr(
        freeman_cli.multiprocessing,
        "get_context",
        lambda method: FakeContext(),
    )
    config = {
        "paths": {
            "output_root": output,
            "work_root": tmp_path / "work",
        },
        "dataset": {"subjects": [1, 2, 3, 4]},
        "sam3d": {"device": 0},
    }

    freeman_cli._run_parallel_subjects(
        config,
        canonical,
        (0, 1),
        False,
    )

    assert started == [(0, (2, 4)), (1, (3,))]
    assert joined == [0, 1]
    state = json.loads(canonical.read_text(encoding="utf-8"))
    assert {
        subject: details["status"]
        for subject, details in state["subjects"].items()
    } == {"1": "complete", "2": "complete", "3": "complete", "4": "complete"}


def test_parallel_coordinator_waits_for_peer_and_records_crashed_worker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    canonical = output / "run_state.json"
    canonical.write_text(
        json.dumps({"stages": {}, "subjects": {}}),
        encoding="utf-8",
    )
    joined = []

    class FakeProcess:
        def __init__(self, *, target, args):
            del target
            self.args = args
            self.exitcode = 9 if args[1] == 0 else 0

        def start(self):
            _, device, subjects, state_path, _ = self.args
            status = "running" if device == 0 else "complete"
            Path(state_path).write_text(
                json.dumps(
                    {
                        "subjects": {
                            str(subject): {"status": status}
                            for subject in subjects
                        }
                    }
                ),
                encoding="utf-8",
            )

        def join(self):
            joined.append(self.args[1])

    class FakeContext:
        Process = FakeProcess

    monkeypatch.setattr(
        freeman_cli.multiprocessing,
        "get_context",
        lambda method: FakeContext(),
    )
    config = {
        "paths": {
            "output_root": output,
            "work_root": tmp_path / "work",
        },
        "dataset": {"subjects": [1, 2]},
        "sam3d": {"device": 0},
    }

    with pytest.raises(RuntimeError, match="device 0"):
        freeman_cli._run_parallel_subjects(
            config,
            canonical,
            (0, 1),
            False,
        )

    assert joined == [0, 1]
    state = json.loads(canonical.read_text(encoding="utf-8"))
    assert state["subjects"]["1"]["status"] == "failed"
    assert state["subjects"]["1"]["error_type"] == "WorkerProcessError"
    assert state["subjects"]["2"]["status"] == "complete"


def test_run_subjects_cleans_each_success_before_starting_next(tmp_path: Path) -> None:
    events: list[str] = []

    run_subjects(
        [7, 3],
        state_path=tmp_path / "run_state.json",
        process=lambda subject: events.append(f"process:{subject}"),
        cleanup=lambda subject: events.append(f"cleanup:{subject}"),
        keep_workspace=False,
    )

    assert events == [
        "process:3",
        "cleanup:3",
        "process:7",
        "cleanup:7",
    ]
    state = json.loads((tmp_path / "run_state.json").read_text(encoding="utf-8"))
    assert state["subjects"]["3"]["status"] == "complete"
    assert state["subjects"]["7"]["status"] == "complete"


def test_run_subjects_preserves_failed_workspace_and_records_error(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def process(subject: int) -> None:
        events.append(f"process:{subject}")
        if subject == 3:
            raise RuntimeError("broken inference")

    with pytest.raises(RuntimeError, match="broken inference"):
        run_subjects(
            [3, 7],
            state_path=tmp_path / "run_state.json",
            process=process,
            cleanup=lambda subject: events.append(f"cleanup:{subject}"),
            keep_workspace=False,
        )

    assert events == ["process:3"]
    state = json.loads((tmp_path / "run_state.json").read_text(encoding="utf-8"))
    assert state["subjects"]["3"] == {
        "status": "failed",
        "error_type": "RuntimeError",
        "error_message": "broken inference",
    }


def test_run_resumes_after_completed_dataset_preparation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    work = tmp_path / "work"
    shared = work / "shared"
    for relative in (
        "30FPS/cameras",
        "30FPS/keypoints2d",
        "30FPS/keypoints3d",
    ):
        (shared / relative).mkdir(parents=True, exist_ok=True)
    (shared / "session_list.txt").write_text("session_subj01\n", encoding="utf-8")
    (shared / "30FPS/cameras/session_subj01.json").write_text(
        "[]",
        encoding="utf-8",
    )
    (shared / "30FPS/keypoints2d/session_subj01.npy").write_bytes(b"2d")
    (shared / "30FPS/keypoints3d/session_subj01.npy").write_bytes(b"3d")
    (shared / "extraction_manifest.json").write_text("{}", encoding="utf-8")
    output.mkdir()
    state_path = output / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "stages": {
                    "inspect": {"status": "complete"},
                    "download": {"status": "complete"},
                    "shared_annotations": {"status": "complete"},
                },
                "subjects": {"1": {"status": "failed"}},
            }
        ),
        encoding="utf-8",
    )
    results_json = output / "results.json"
    markdown = output / "report.md"
    results_json.write_text("{}", encoding="utf-8")
    markdown.write_text("report", encoding="utf-8")
    calls: list[str] = []

    class ResumeOperations(DefaultStageOperations):
        def inspect(self, config, *, dry_run=False):
            calls.append("inspect")
            return SimpleNamespace(
                required_bytes=0,
                entries=(),
                archive_root=tmp_path / "archives",
            )

        def report(self, config):
            calls.append("report")
            return SimpleNamespace(
                results_json=results_json,
                markdown=markdown,
            )

    monkeypatch.setattr(
        freeman_cli,
        "validate_downloads",
        lambda *args, **kwargs: calls.append("validate"),
    )
    monkeypatch.setattr(
        freeman_cli,
        "extract_shared_annotations",
        lambda *args, **kwargs: calls.append("extract"),
    )
    monkeypatch.setattr(
        freeman_cli,
        "run_subjects",
        lambda *args, **kwargs: calls.append("subjects"),
    )
    config = {
        "paths": {
            "archive_root": tmp_path / "archives",
            "work_root": work,
            "output_root": output,
        },
        "dataset": {"subjects": [1], "frame_stride": 1},
    }

    ResumeOperations().run(config)

    assert calls == ["subjects", "report"]


def test_parallel_run_does_not_publish_report_when_worker_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    work = tmp_path / "work"
    shared = work / "shared"
    output.mkdir()
    shared.mkdir(parents=True)
    (shared / "extraction_manifest.json").write_text("{}", encoding="utf-8")
    (output / "run_state.json").write_text(
        json.dumps(
            {
                "stages": {
                    "inspect": {"status": "complete"},
                    "download": {"status": "complete"},
                    "shared_annotations": {"status": "complete"},
                },
                "subjects": {},
            }
        ),
        encoding="utf-8",
    )
    calls = []

    class ParallelOperations(DefaultStageOperations):
        def report(self, config):
            calls.append("report")
            raise AssertionError("failed workers must suppress reporting")

    monkeypatch.setattr(freeman_cli, "_shared_tree_valid", lambda path: True)

    def fail_parallel(config, state_path, devices, keep_workspace):
        calls.append((tuple(devices), keep_workspace))
        raise RuntimeError("device 0 failed")

    monkeypatch.setattr(
        freeman_cli,
        "_run_parallel_subjects",
        fail_parallel,
    )
    config = {
        "paths": {
            "archive_root": tmp_path / "archives",
            "work_root": work,
            "output_root": output,
        },
        "dataset": {"subjects": [1, 2], "frame_stride": 1},
    }

    with pytest.raises(RuntimeError, match="device 0 failed"):
        ParallelOperations().run(config, devices=(0, 1))

    assert calls == [((0, 1), False)]


def test_force_infer_invalidates_only_selected_subject_and_downstream(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    selected = [
        output / "sam3d" / "subject_03" / "cache",
        output / "fusion" / "methods" / "candidate" / "subject_03",
        output / "evaluation" / "session_metrics",
    ]
    for path in selected:
        path.mkdir(parents=True, exist_ok=True)
    (selected[2] / "subject_03.json").write_text("{}", encoding="utf-8")
    untouched = output / "sam3d" / "subject_07" / "cache"
    untouched.mkdir(parents=True)
    config = {
        "paths": {"output_root": output},
        "dataset": {"subjects": [3]},
    }

    reset_forced_stage(config, "infer")

    assert not (output / "sam3d" / "subject_03").exists()
    assert not (
        output / "fusion" / "methods" / "candidate" / "subject_03"
    ).exists()
    assert not (selected[2] / "subject_03.json").exists()
    assert untouched.is_dir()
