from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pytest


PAPER_ROOT = Path(__file__).resolve().parent.parent


def _load_checker():
    path = PAPER_ROOT / "scripts" / "check_manuscript.py"
    spec = importlib.util.spec_from_file_location("check_manuscript", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_generator():
    path = PAPER_ROOT / "scripts" / "generate_paper_assets.py"
    spec = importlib.util.spec_from_file_location(
        "generate_paper_assets",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manuscript_text() -> str:
    paths = [PAPER_ROOT / "manuscript.tex"]
    paths.extend(sorted((PAPER_ROOT / "sections").glob("*.tex")))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_checker_rejects_a_changed_cohort_result(tmp_path: Path):
    copied = tmp_path / "paper"
    shutil.copytree(
        PAPER_ROOT,
        copied,
        ignore=shutil.ignore_patterns("build", ".mplconfig", "__pycache__"),
    )
    table_path = copied / "tables" / "cohort_cycle_results.tex"
    table_path.write_text(
        table_path.read_text(encoding="utf-8").replace(
            "0.0114",
            "0.9999",
            1,
        ),
        encoding="utf-8",
    )
    checker = _load_checker()
    checker.ROOT = copied

    with pytest.raises(SystemExit, match="cohort table"):
        checker.main()


def test_checker_accepts_the_source_matched_cohort_tables():
    checker = _load_checker()

    checker.main()


def test_learned_table_uses_held_out_test_rows(tmp_path: Path):
    generator = _load_generator()
    evidence = generator.load_learned_evidence()
    output = tmp_path / "learned_results.tex"

    generator.write_learned_table(output, evidence)

    table = output.read_text(encoding="utf-8")
    assert "held-out test set" in table
    assert "$N=14$" in table
    assert "60.78" in table
    assert "A6" in table


def test_deterministic_generator_resolves_current_run_root():
    generator = _load_generator()

    path = generator.default_metrics_path()

    assert path == (
        generator.project_root()
        / "local/runs/fuse_experiments/metrics_by_person.csv"
    )


def test_unity_table_preserves_input_regime_boundaries(tmp_path: Path):
    generator = _load_generator()
    evidence = generator.load_unity_evidence()
    output = tmp_path / "unity_benchmark.tex"

    generator.write_unity_table(output, evidence)

    table = output.read_text(encoding="utf-8")
    assert "Unity native 3D" in table
    assert "A6" in table
    assert "178.506" in table
    assert "Triangulation" in table
    assert "calibrated 2D" in table
    assert "uncalibrated direct 3D" in table


def test_cohort_assets_require_complete_centered_mixed_models(
    tmp_path: Path,
):
    generator = _load_generator()

    summary = generator.publish_cohort_assets(tmp_path)

    assert summary == {"core_outcomes": 8, "sensitivity_models": 32}
    table = (
        tmp_path / "tables" / "cohort_cycle_results.tex"
    ).read_text(encoding="utf-8")
    assert "mid-repetition reference" in table
    sensitivity = (
        tmp_path / "tables" / "cohort_cycle_sensitivity.tex"
    ).read_text(encoding="utf-8")
    assert "same centered cycle-level mixed model" in sensitivity
    assert (tmp_path / "figures" / "cohort_cycle_analysis.pdf").is_file()


def test_primary_learned_and_unity_claims_name_population_and_reference():
    manuscript = _manuscript_text()

    assert "held-out test set ($N=14$)" in manuscript
    assert "descriptive all-person" in manuscript
    assert "Unity native 3D" in manuscript
    assert "one sequence-level Sim3" in manuscript
    assert "A3-relative retention" in manuscript
