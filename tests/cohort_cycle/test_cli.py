from pathlib import Path
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from gymnastics.analysis.cohort_cycle.cli import main as cohort_cycle_main
from gymnastics.fusion.rotation_aware.cli import (
    _paths as rotation_paths,
    load_config as load_rotation_config,
)

from .test_features import _upright_pose
from .test_oof import _make_run
from .test_report import _write_finalized_inputs
from .test_statistics import _write_synthetic_feature_artifacts


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


def test_a6_crossfit_config_resolves_shared_cache_environment(
    tmp_path: Path,
    monkeypatch,
):
    """A nested unresolved interpolation makes every cross-fit run miss cache."""
    monkeypatch.setenv("GYMNASTICS_SHARED_RUN_ROOT", str(tmp_path))
    config = load_rotation_config(
        PROJECT_ROOT / "configs/analysis/cohort_cycle_a6_train.yaml"
    )

    paths = rotation_paths(config, None)

    assert paths["cache"] == tmp_path / "fuse_rotation_aware" / "cache"


def test_folds_stage_writes_crossfit_manifest_from_config(tmp_path: Path):
    """A parsed `folds` command that does no work is a broken public stage."""
    student_path = tmp_path / "students.csv"
    with student_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("student_id", "person_id", "complete"),
        )
        writer.writeheader()
        for person_id in range(81, 139):
            writer.writerow(
                {
                    "student_id": f"S{person_id - 80}",
                    "person_id": person_id,
                    "complete": person_id != 135,
                }
            )

    organization_path = tmp_path / "organization.csv"
    with organization_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("group", "original_id", "person_id", "view", "status"),
        )
        writer.writeheader()
        for person_id in range(69, 81):
            writer.writerow(
                {
                    "group": "elderly",
                    "original_id": f"ID{person_id}",
                    "person_id": person_id,
                    "view": "face",
                    "status": "copied",
                }
            )
        for person_id in range(81, 139):
            writer.writerow(
                {
                    "group": "student",
                    "original_id": f"S{person_id - 80}",
                    "person_id": person_id,
                    "view": "face",
                    "status": (
                        "missing_source" if person_id == 135 else "copied"
                    ),
                }
            )

    fold0_test = {
        "1",
        "24",
        "36",
        "49",
        "51",
        "52",
        "60",
        "79",
        "85",
        "106",
        "116",
        "117",
        "130",
        "136",
    }
    elderly = [
        str(person_id)
        for person_id in range(1, 81)
        if str(person_id) not in fold0_test
    ]
    students = [
        str(person_id)
        for person_id in range(81, 139)
        if person_id != 135 and str(person_id) not in fold0_test
    ]
    val = elderly[:16] + students[:11]
    fold0 = tmp_path / "fold0.json"
    fold0.write_text(
        json.dumps(
            {
                "train": elderly[16:] + students[11:],
                "val": val,
                "test": sorted(fold0_test, key=int),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "student_mapping": str(student_path),
                    "organization_mapping": str(organization_path),
                    "fold0_split": str(fold0),
                    "fold_output": str(output),
                },
                "crossfit": {"split_seed": 20260728},
            }
        ),
        encoding="utf-8",
    )

    assert cohort_cycle_main(["folds", "--config", str(config)]) == 0
    manifest = json.loads(
        (output / "crossfit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["cohort_counts"] == {"elderly": 80, "student": 57}


def test_audit_stage_check_only_validates_configured_test_run(
    tmp_path: Path,
    capsys,
):
    """The public audit command must execute provenance checks, not just parse."""
    run = _make_run(
        tmp_path,
        fold=0,
        person_id="1",
        run_id="run0",
    )
    alternate = _make_run(
        tmp_path,
        fold=0,
        person_id="1",
        run_id="run_alternate",
    )
    (tmp_path / "run_registry.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "primary_seed": 0,
                "runs": {
                    "00": {
                        "outer_fold": 0,
                        "run_id": run.run_id,
                        "seed": 0,
                        "split_file": run.split_manifest.name,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "crossfit_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cohorts": {"1": "elderly"},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "audit.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "fold_output": str(tmp_path),
                    "cohort_output": str(tmp_path / "cohort"),
                    "rotation_aware_root": str(tmp_path),
                },
                "crossfit": {
                    "expected_people": 1,
                    "expected_cycles": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        cohort_cycle_main(
            [
                "audit",
                "--config",
                str(config),
                "--fold",
                "0",
                "--check-only",
            ]
        )
        == 0
    )
    audit = json.loads(capsys.readouterr().out)
    assert audit["valid"] is True
    assert audit["people"] == 1
    assert audit["cycles"] == 1

    alternate_output = tmp_path / "alternate_oof"
    assert (
        cohort_cycle_main(
            [
                "audit",
                "--config",
                str(config),
                "--fold",
                "0",
                "--run-id",
                alternate.run_id,
                "--seed",
                "0",
                "--publication-root",
                str(alternate_output),
            ]
        )
        == 0
    )
    alternate_rows = list(
        csv.DictReader(
            (alternate_output / "oof_provenance.csv").open(
                encoding="utf-8"
            )
        )
    )
    assert alternate_rows[0]["run_id"] == "run_alternate"


def test_features_stage_extracts_an_explicit_publication(tmp_path: Path):
    """The public features command must materialize tidy analysis tables."""
    publication = tmp_path / "oof"
    provenance_rows = []
    timestamps = np.linspace(0.0, 2.0, 101)
    theta = np.sin(np.pi * timestamps)
    for cycle_index in range(4):
        cycle_id = f"cycle_{cycle_index:03d}"
        cycle_root = publication / "person_1" / cycle_id
        cycle_root.mkdir(parents=True)
        np.savez_compressed(
            cycle_root / "prediction.npz",
            kpts_body=_upright_pose(101),
            theta_fused_rad=theta,
            omega_fused_rad_s=np.gradient(theta, timestamps),
            timestamps=timestamps,
            frame_valid=np.ones(101, dtype=bool),
            joint_valid=np.ones((101, 70), dtype=bool),
        )
        provenance_rows.append(
            {
                "person_id": "1",
                "cohort": "elderly",
                "outer_fold": "0",
                "cycle_id": cycle_id,
                "prediction_path": (
                    f"person_1/{cycle_id}/prediction.npz"
                ),
            }
        )
    with (publication / "oof_provenance.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(provenance_rows[0]),
        )
        writer.writeheader()
        writer.writerows(provenance_rows)
    output = tmp_path / "features"
    config = tmp_path / "features.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {"cohort_output": str(tmp_path / "cohort")},
                "quality_control": {
                    "phase_points": 101,
                    "minimum_person_cycles": 4,
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        cohort_cycle_main(
            [
                "features",
                "--config",
                str(config),
                "--publication-root",
                str(publication),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "cycle_features.csv").is_file()


def test_analyze_stage_writes_corrected_core_results(tmp_path: Path):
    """The public analyze stage must execute the finalized statistical pipeline."""
    feature_root = tmp_path / "features"
    _write_synthetic_feature_artifacts(feature_root)
    output = tmp_path / "analysis"
    config = tmp_path / "analysis.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {"cohort_output": str(tmp_path / "cohort")},
                "statistics": {
                    "permutations": 99,
                    "permutation_seed": 5,
                    "log_transform": [],
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        cohort_cycle_main(
            [
                "analyze",
                "--config",
                str(config),
                "--feature-root",
                str(feature_root),
                "--output-root",
                str(output),
                "--no-random-slope",
            ]
        )
        == 0
    )
    core = pd.read_csv(output / "core_mixed_models.csv")
    assert len(core) == 8


def test_assets_stage_renders_an_explicit_finalized_analysis(tmp_path: Path):
    """The public assets stage must preserve the finalized analysis boundary."""
    features, statistics = _write_finalized_inputs(tmp_path)
    output = tmp_path / "report"
    config = tmp_path / "assets.yaml"
    config.write_text(
        yaml.safe_dump(
            {"paths": {"cohort_output": str(tmp_path / "cohort")}}
        ),
        encoding="utf-8",
    )

    assert (
        cohort_cycle_main(
            [
                "assets",
                "--config",
                str(config),
                "--feature-root",
                str(features),
                "--statistics-root",
                str(statistics),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "cohort_cycle_analysis.pdf").is_file()
