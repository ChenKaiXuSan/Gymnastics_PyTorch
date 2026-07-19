from pathlib import Path

import numpy as np

from fuse.metadata.mhr70 import mhr_names
from fuse.rotation_aware.config import load_skeleton_spec
from fuse.rotation_aware.evaluation import (
    MethodSequence,
    discover_method_sequences,
    evaluate_person_trials,
    external_metrics_from_reference,
)


SPEC = load_skeleton_spec(Path("configs/fuse/skeleton_mhr70.yaml"))


def _sequence(offset: float = 0.0) -> MethodSequence:
    values = np.ones((8, len(mhr_names), 3), dtype=np.float32) * offset
    values[:, 9, 0], values[:, 10, 0] = -1, 1
    values[:, 5, 1], values[:, 6, 1], values[:, 2, 1] = 2, 2, 3
    return MethodSequence(
        method="rotation_aware_self_supervised",
        kpts_world=values,
        timestamps=np.arange(8) / 60.0,
    )


def test_evaluation_aggregates_cycles_by_person() -> None:
    report = evaluate_person_trials("1", [_sequence(), _sequence(0.1)], SPEC)

    assert len(report.person_metrics) == 1
    row = report.person_metrics[0]
    assert row["person_id"] == "1"
    assert row["method"] == "rotation_aware_self_supervised"
    assert {
        "bone_cv",
        "rigidity",
        "joint_jerk",
        "trunk_angular_jerk",
        "rom_retention",
        "peak_angular_velocity_retention",
        "swap_error",
        "fixed_corruption_recovery",
    } <= set(row)
    assert np.isnan(row["swap_error"])
    assert np.isnan(row["fixed_corruption_recovery"])


def test_retention_uses_matched_trial_reference_not_a_perfect_placeholder() -> None:
    sequence = _sequence()
    reference = _sequence()
    reference.kpts_world[:, 5, 0] = np.linspace(0, 4, len(reference.kpts_world))
    sequence = MethodSequence(
        sequence.method,
        sequence.kpts_world,
        sequence.timestamps,
        reference_kpts=reference.kpts_world,
    )
    report = evaluate_person_trials("1", [sequence], SPEC)

    assert report.person_metrics[0]["rom_retention"] != 1.0


def test_external_reference_matching_rejects_unmatched_or_wrong_length_trials() -> None:
    sequence = _sequence()
    sequence = MethodSequence(
        sequence.method, sequence.kpts_world, sequence.timestamps, trial_id="cycle_000"
    )
    with np.testing.assert_raises(ValueError):
        evaluate_person_trials(
            "1", [sequence], SPEC, references={"cycle_001": sequence.kpts_world}
        )
    with np.testing.assert_raises(ValueError):
        evaluate_person_trials(
            "1", [sequence], SPEC, references={"cycle_000": sequence.kpts_world[:-1]}
        )


def test_external_gt_metrics_are_optional_and_root_normalized() -> None:
    candidate = _sequence().kpts_world
    reference = candidate + np.array([10.0, 0.0, 0.0], dtype=np.float32)
    metrics, joints = external_metrics_from_reference(candidate, reference)

    assert metrics["mpjpe"] < 1e-5
    assert len(joints) == len(mhr_names)


def test_external_evaluation_imports_are_isolated() -> None:
    source = Path("fuse/rotation_aware/evaluation.py").read_text(encoding="utf-8")
    for path in ("inference.py", "training.py", "cli.py"):
        assert "triangulation" not in Path("fuse/rotation_aware", path).read_text(
            encoding="utf-8"
        )
    assert "triangulation" in source


def test_discovery_reports_only_saved_new_baselines_as_available(
    tmp_path: Path,
) -> None:
    root = tmp_path / "inference" / "person_1" / "cycle_000"
    root.mkdir(parents=True)
    values = _sequence().kpts_world
    np.savez_compressed(
        root / "fused_sequence.npz",
        kpts_world=values,
        kpts_face_world=values,
        kpts_side_world=values,
        kpts_arithmetic_world=values,
        kpts_base_world=values,
        frame_valid=np.ones(len(values), dtype=bool),
        joint_valid=np.ones(values.shape[:2], dtype=bool),
        face_map=np.arange(len(values)),
        side_map=np.arange(len(values)),
    )

    sequences, status = discover_method_sequences(
        tmp_path / "inference", tmp_path / "old", "1"
    )

    assert {sequence.method for sequence in sequences} >= {
        "face_only",
        "side_only",
        "canonical_arithmetic",
        "quality_mean",
        "rotation_aware_self_supervised",
    }
    assert status["A0"] == "absent"
