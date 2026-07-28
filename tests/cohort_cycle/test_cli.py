from pathlib import Path
import csv
import json
import os
import subprocess
import sys

import yaml

from gymnastics.analysis.cohort_cycle.cli import main as cohort_cycle_main

from .test_oof import _make_run


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
