import os
from pathlib import Path
import subprocess
import sys

import gymnastics.cli as unified_cli


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_unified_cli_lists_pipeline_commands():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-m", "gymnastics", "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for command in (
        "sam3d",
        "align",
        "triangulate",
        "fuse",
        "classify",
        "analyze",
        "calibrate",
        "benchmark",
    ):
        assert command in result.stdout


def test_unified_cli_exposes_freeman_benchmark():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gymnastics",
            "benchmark",
            "freeman",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for stage in (
        "inspect",
        "download",
        "infer",
        "fuse",
        "evaluate",
        "report",
        "run",
    ):
        assert stage in result.stdout


def test_runtime_cache_preserves_default_huggingface_credentials(
    monkeypatch,
    tmp_path,
):
    home = tmp_path / "home"
    local_root = tmp_path / "local"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(unified_cli, "LOCAL_ROOT", local_root)

    unified_cli._configure_runtime_cache()

    assert os.environ["HF_HOME"] == str(home / ".cache" / "huggingface")
    assert os.environ["XDG_CACHE_HOME"] == str(local_root / "cache" / "xdg")
