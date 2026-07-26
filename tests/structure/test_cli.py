import os
from pathlib import Path
import subprocess
import sys


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
    ):
        assert command in result.stdout
