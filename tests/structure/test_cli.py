"""The four stage packages are the only entry points under src/."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

import common.cli as shared_cli
from fusion.cli import COMMANDS as FUSION_COMMANDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGES = ("pose_estimation", "cycle_alignment", "pseudo_gt", "fusion")


def _run(*argv: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run([sys.executable, "-m", *argv], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=False)


def test_src_exposes_exactly_the_four_stage_entry_points():
    entry_points = sorted(p.parent.name for p in (PROJECT_ROOT / "src").glob("*/__main__.py"))
    assert entry_points == sorted(STAGES)
    assert not (PROJECT_ROOT / "src" / "gymnastics").exists()


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_prints_help(stage: str):
    result = _run(stage, "--help")
    assert result.returncode == 0, result.stderr
    assert stage in result.stdout


def test_pseudo_gt_lists_its_three_commands():
    result = _run("pseudo_gt", "--help")
    for command in ("calibrate", "estimate-extrinsics", "triangulate"):
        assert command in result.stdout


def test_fusion_lists_model_baselines_benchmarks_and_analysis():
    result = _run("fusion", "--help")
    for command in ("train", "deterministic", "benchmark-freeman", "benchmark-unity", "analyze", "cohort-cycle", "rotation-aware"):
        assert command in result.stdout
    assert FUSION_COMMANDS["train"][0] == "fusion.train"


def test_fusion_benchmark_freeman_forwards_its_stages():
    result = _run("fusion", "benchmark-freeman", "--help")
    assert result.returncode == 0, result.stderr
    for stage in ("inspect", "download", "infer", "fuse", "evaluate", "report", "run"):
        assert stage in result.stdout


def test_runtime_cache_preserves_default_huggingface_credentials(monkeypatch, tmp_path):
    home = tmp_path / "home"
    local_root = tmp_path / "local"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(shared_cli, "LOCAL_ROOT", local_root)

    shared_cli.configure_runtime_cache()

    assert os.environ["HF_HOME"] == str(home / ".cache" / "huggingface")
    assert os.environ["XDG_CACHE_HOME"] == str(local_root / "cache" / "xdg")
