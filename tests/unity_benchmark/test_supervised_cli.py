from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from gymnastics.benchmarks.unity import cli as unity_cli


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "gymnastics", *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_unity_cli_exposes_supervised_stages() -> None:
    help_result = _run("benchmark", "unity", "--help")

    assert help_result.returncode == 0, help_result.stderr
    for stage in (
        "finetune",
        "finetune-matrix",
        "evaluate-finetuned",
        "report-finetuned",
        "train-extrinsic",
        "train-extrinsic-matrix",
        "evaluate-extrinsic",
        "report-extrinsic",
    ):
        assert stage in help_result.stdout


def test_supervised_stage_arguments_parse() -> None:
    parser = unity_cli._parser()

    one = parser.parse_args(
        [
            "finetune",
            "--ablation",
            "A4",
            "--fold",
            "left_to_right",
            "--seed",
            "0",
            "--device",
            "cpu",
        ]
    )
    matrix = parser.parse_args(["finetune-matrix", "--device", "cuda"])
    evaluate = parser.parse_args(["evaluate-finetuned"])
    report = parser.parse_args(["report-finetuned"])

    assert (one.ablation, one.fold, one.seed, one.device) == (
        "A4",
        "left_to_right",
        0,
        "cpu",
    )
    assert matrix.device == "cuda"
    assert evaluate.stage == "evaluate-finetuned"
    assert report.stage == "report-finetuned"


def test_extrinsic_stage_arguments_parse() -> None:
    parser = unity_cli._parser()
    one = parser.parse_args(
        [
            "train-extrinsic",
            "--method",
            "extrinsic_gate",
            "--fold",
            "right_to_left",
            "--seed",
            "1",
            "--device",
            "cpu",
        ]
    )
    matrix = parser.parse_args(["train-extrinsic-matrix", "--device", "cpu"])
    evaluate = parser.parse_args(["evaluate-extrinsic"])
    report = parser.parse_args(["report-extrinsic"])

    assert (one.method, one.fold, one.seed, one.device) == (
        "extrinsic_gate",
        "right_to_left",
        1,
        "cpu",
    )
    assert matrix.device == "cpu"
    assert evaluate.stage == "evaluate-extrinsic"
    assert report.stage == "report-extrinsic"


def test_matrix_dispatch_skips_two_completed_cells(
    monkeypatch,
) -> None:
    cells = unity_cli._supervised_matrix_cells()
    expected = {
        (ablation, fold, seed)
        for ablation in ("A4", "A5", "A6", "A7", "A8", "A9")
        for fold in ("left_to_right", "right_to_left")
        for seed in (0, 1, 2)
    }
    complete = {
        ("A4", "left_to_right", 0),
        ("A9", "right_to_left", 2),
    }
    invoked = []
    monkeypatch.setattr(
        unity_cli,
        "_supervised_cell_is_complete",
        lambda cell: cell in complete,
    )
    monkeypatch.setattr(
        unity_cli,
        "_run_supervised_cell",
        lambda cell: invoked.append(cell),
    )

    counts = unity_cli._dispatch_supervised_matrix(cells)

    assert set(cells) == expected
    assert set(invoked) == expected - complete
    assert counts == {"completed": 34, "reused": 2, "failed": 0}
