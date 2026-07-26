from __future__ import annotations

import json
from pathlib import Path

import pytest

from gymnastics.benchmarks.freeman.cli import (
    StageOperations,
    main,
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

    def run(self, config, *, force_stage=None, keep_workspace=False, dry_run=False):
        self.calls.append(("run", dict(config)))
        self.force_stage = force_stage
        self.keep_workspace = keep_workspace
        self.dry_run = dry_run


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
