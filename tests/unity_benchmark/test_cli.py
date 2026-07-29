from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import yaml

import numpy as np

from gymnastics.benchmarks.unity.cli import (
    _load_method_sequence,
    main as unity_main,
)

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


def test_unified_cli_exposes_all_unity_stages() -> None:
    top = _run("--help")
    nested = _run("benchmark", "unity", "--help")

    assert top.returncode == 0, top.stderr
    assert "benchmark" in top.stdout
    assert nested.returncode == 0, nested.stderr
    for stage in (
        "inspect",
        "infer",
        "triangulate",
        "fuse",
        "evaluate",
        "report",
        "run",
    ):
        assert stage in nested.stdout


def test_inspect_reports_real_dataset_inventory() -> None:
    result = _run(
        "benchmark",
        "unity",
        "inspect",
        "--config",
        "configs/benchmarks/unity.yaml",
    )

    assert result.returncode == 0, result.stderr
    assert "samples: 199" in result.stdout
    assert "images: 398" in result.stdout
    assert "joints: 22" in result.stdout
    assert "evaluation_sequences: 3" in result.stdout


def test_oracle_triangulation_stage_writes_three_sequences(
    tmp_path: Path,
) -> None:
    config = {
        "paths": {
            "dataset_root": "/home/data/xchen/gymnastics/unity_benchmark",
            "output_root": str(tmp_path / "run"),
            "sam3d_config": "configs/sam3d/sam3d_body.yaml",
            "skeleton": "configs/fusion/skeleton_mhr70.yaml",
        },
        "checkpoints": {},
        "data": {"fps": 60.0},
        "evaluation": {"alignment": "similarity", "camera_reference": "cam0"},
    }
    config_path = tmp_path / "unity.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    status = unity_main(
        ["triangulate", "--config", str(config_path), "--oracle-only"]
    )

    assert status == 0
    oracle = tmp_path / "run/triangulation/oracle2d"
    assert sorted(path.stem for path in oracle.glob("*.npz")) == [
        "continuous_left_060_r00",
        "continuous_right_060_r00",
        "static_sweep",
    ]


def test_loads_saved_method_sequence_contract(tmp_path: Path) -> None:
    path = tmp_path / "sequence.npz"
    np.savez_compressed(
        path,
        method=np.asarray("method"),
        sequence_id=np.asarray("sequence"),
        sample_ids=np.asarray([3, 4]),
        points=np.ones((2, 16, 3), dtype=np.float32),
        valid=np.ones((2, 16), dtype=bool),
        joint_names=np.asarray([f"joint_{i}" for i in range(16)]),
        metadata=np.asarray('{"ranking_group": "valid"}'),
    )

    sequence = _load_method_sequence(path)

    assert sequence.method == "method"
    assert sequence.sequence_id == "sequence"
    assert sequence.sample_ids.tolist() == [3, 4]
    assert sequence.metadata["ranking_group"] == "valid"
