"""The private data root is defined once, in common.paths, and injected into every config."""

from pathlib import Path
import subprocess

import common.paths as paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Split so this file does not match its own markers.
MACHINE_SPECIFIC_ROOTS = ("/home/data/" + "xchen", "/workspace/" + "data")


def _tracked(prefix: str) -> list[Path]:
    out = subprocess.run(["git", "ls-files", prefix], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.split()
    return [PROJECT_ROOT / f for f in out if "third_party" not in f]


def test_no_machine_specific_data_root_outside_common_paths():
    offenders = []
    for path in _tracked("src") + _tracked("tests") + _tracked("pegasus"):
        if path.suffix not in {".py", ".yaml", ".sh"} or path == PROJECT_ROOT / "src/common/paths.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(marker in text for marker in MACHINE_SPECIFIC_ROOTS):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


def test_configs_interpolate_the_data_root_from_the_environment():
    for path in _tracked("src/configs"):
        if path.suffix != ".yaml":
            continue
        text = path.read_text(encoding="utf-8")
        if "GYMNASTICS_DATA_ROOT" in text:
            assert "${oc.env:GYMNASTICS_DATA_ROOT}" in text, path
            assert "${oc.env:GYMNASTICS_DATA_ROOT," not in text, path


def test_common_paths_exports_the_resolved_root():
    import os

    assert os.environ["GYMNASTICS_DATA_ROOT"] == str(paths.DATA_ROOT)
    assert paths.SAM3D_RESULTS_ROOT == paths.DATA_ROOT / "sam3d_body_results"
