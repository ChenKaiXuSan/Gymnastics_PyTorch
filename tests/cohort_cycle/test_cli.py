from pathlib import Path
import os
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_cohort_cycle_help_lists_pipeline_stages():
    """Removing the route or any public stage must break the CLI contract."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gymnastics",
            "cohort-cycle",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for stage in ("folds", "audit", "features", "analyze", "assets"):
        assert stage in result.stdout
