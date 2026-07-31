import json
from pathlib import Path

import numpy as np

from gymnastics.common.skeletons.mhr70 import mhr_names
from gymnastics.fusion.rotation_aware.config import load_skeleton_spec
from gymnastics.fusion.rotation_aware.evaluation import (
    ABLATION_REGISTRY,
    MethodSequence,
    _circular_rom,
    _derivative,
    _trunk,
    discover_method_sequences,
    evaluate_person_trials,
    external_metrics_from_reference,
    load_triangulated_references,
)
from tests.rotation_aware.test_geometry import synthetic_mhr70_pose


SPEC = load_skeleton_spec(Path("configs/fusion/skeleton_mhr70.yaml"))


def test_registry_names_cross_attention_ablations() -> None:
    assert (
        ABLATION_REGISTRY["A10"]
        == "rotation_conditioned_cross_view_attention"
    )
    assert ABLATION_REGISTRY["A11"] == "cross_view_attention_without_rotation"


def _sequence(offset: float = 0.0) -> MethodSequence:
    values = np.ones((8, len(mhr_names), 3), dtype=np.float32) * offset
    values[:, 9, 0], values[:, 10, 0] = -1, 1
    values[:, 5, 1], values[:, 6, 1], values[:, 2, 1] = 2, 2, 3
    return MethodSequence(
        method="rotation_aware_self_supervised",
        kpts_world=values,
        timestamps=np.arange(8) / 60.0,
    )


def _rotation_sequence(
    angles: list[float], trial_id: str, *, reference_angles: list[float] | None = None
) -> MethodSequence:
    def points(values: list[float]) -> np.ndarray:
        frames = [
            synthetic_mhr70_pose(theta_deg=angle, frames=1)[0][0, 0].numpy()
            for angle in values
        ]
        return np.stack(frames)

    candidate = points(angles)
    reference = points(reference_angles) if reference_angles is not None else None
    return MethodSequence(
        method="rotation_aware_self_supervised",
        kpts_world=candidate,
        timestamps=np.arange(len(angles), dtype=np.float64) / 60.0,
        trial_id=trial_id,
        reference_kpts=reference,
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


def test_external_reference_matching_requires_matching_shape_but_allows_missing_trials() -> (
    None
):
    sequence = _sequence()
    sequence = MethodSequence(
        sequence.method, sequence.kpts_world, sequence.timestamps, trial_id="cycle_000"
    )
    report = evaluate_person_trials(
        "1", [sequence], SPEC, references={"cycle_001": sequence.kpts_world}
    )
    assert np.isnan(report.person_metrics[0]["mpjpe"])
    with np.testing.assert_raises(ValueError):
        evaluate_person_trials(
            "1", [sequence], SPEC, references={"cycle_000": sequence.kpts_world[:-1]}
        )


def test_external_gt_metrics_are_optional_and_root_normalized() -> None:
    candidate = _sequence().kpts_world
    reference = candidate + np.array([10.0, 0.0, 0.0], dtype=np.float32)
    metrics, joints = external_metrics_from_reference(candidate, reference, SPEC)

    assert metrics["mpjpe"] < 1e-5
    assert len(joints) == len(mhr_names)


def test_external_metrics_exclude_frames_with_zero_reference_or_invalid_candidate_hips() -> (
    None
):
    candidate = _sequence().kpts_world
    reference = candidate.copy()
    reference[0, 9:11] = 0
    candidate_valid = np.ones(candidate.shape[:2], dtype=bool)
    candidate_valid[1, 9] = False

    metrics, _ = external_metrics_from_reference(
        candidate, reference, SPEC, candidate_valid
    )

    assert metrics["matched_frames"] == len(candidate) - 2


def test_missing_triangulated_cycle_is_not_prepopulated_as_nan_reference(
    tmp_path: Path,
) -> None:
    sequence = MethodSequence(
        "A6", _sequence().kpts_world, _sequence().timestamps, trial_id="cycle_404"
    )

    assert load_triangulated_references(tmp_path, "1", [sequence]) == {}


def test_external_evaluation_imports_are_isolated() -> None:
    source = Path("src/gymnastics/fusion/rotation_aware/evaluation.py").read_text(encoding="utf-8")
    for path in ("inference.py", "training.py", "cli.py"):
        assert "triangulation" not in Path(
            "src/gymnastics/fusion/rotation_aware", path
        ).read_text(
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
        metadata=np.asarray(json.dumps({"ablation": "A4"})),
        diagnostics=np.asarray(json.dumps({"A4": {"swap_error": 3.0}})),
    )

    sequences, status = discover_method_sequences(
        tmp_path / "inference", tmp_path / "old", "1"
    )

    assert {sequence.method for sequence in sequences} >= {
        "face_only",
        "side_only",
        "canonical_arithmetic",
        "quality_mean",
        "A4",
    }
    assert status["A0"] == "available"
    assert status["A4"] == "available"
    assert next(item for item in sequences if item.method == "A4").swap_error == 3.0
    assert next(item for item in sequences if item.method == "A0").swap_error is None


def test_discovery_preserves_cross_attention_ablation_label(
    tmp_path: Path,
) -> None:
    root = tmp_path / "inference" / "person_1" / "cycle_000"
    root.mkdir(parents=True)
    values = _sequence().kpts_world
    np.savez_compressed(
        root / "fused_sequence.npz",
        kpts_world=values,
        frame_valid=np.ones(len(values), dtype=bool),
        joint_valid=np.ones(values.shape[:2], dtype=bool),
        face_map=np.arange(len(values)),
        side_map=np.arange(len(values)),
        metadata=np.asarray(json.dumps({"ablation": "A11"})),
        diagnostics=np.asarray(json.dumps({"A11": {"swap_error": 0.0}})),
    )

    sequences, status = discover_method_sequences(
        tmp_path / "inference", tmp_path / "old", "1"
    )

    assert {sequence.method for sequence in sequences} == {"A11"}
    assert status["A11"] == "available"
    assert sequences[0].swap_error == 0.0


def test_person_metrics_are_weighted_by_valid_points_and_masked_static_joints_have_no_jerk() -> (
    None
):
    short = _sequence()
    long = _sequence()
    long = MethodSequence(
        long.method,
        np.repeat(long.kpts_world[:1], 40, axis=0),
        np.arange(40) / 50.0,
        joint_valid=np.ones((40, len(mhr_names)), dtype=bool),
        trial_id="cycle_001",
    )
    short = MethodSequence(
        short.method,
        short.kpts_world,
        short.timestamps,
        joint_valid=np.ones(short.kpts_world.shape[:2], dtype=bool),
        trial_id="cycle_000",
    )
    report = evaluate_person_trials("1", [short, long], SPEC)

    assert report.person_metrics[0]["joint_jerk"] == 0.0
    assert report.person_metrics[0]["valid_points"] == 5 * (8 + 40)


def test_person_nonlinear_metrics_are_computed_after_pooling_cycles() -> None:
    short = _rotation_sequence(
        [0.0, 10.0], "cycle_000", reference_angles=[0.0, 20.0]
    )
    long = _rotation_sequence(
        [0.0, 60.0, 0.0, 60.0],
        "cycle_001",
        reference_angles=[0.0, 30.0, 0.0, 30.0],
    )
    short.kpts_world[:, [9, 10], 0] *= 0.5
    long.kpts_world[:, [9, 10], 0] *= 2.0

    row = evaluate_person_trials("1", [short, long], SPEC).person_metrics[0]

    assert row["bone_cv"] > 0.1
    np.testing.assert_allclose(row["theta_rom"], np.deg2rad(60.0), atol=1e-5)
    np.testing.assert_allclose(row["rom_retention"], 2.0, atol=1e-5)


def test_external_root_normalization_resolves_hip_roles_from_skeleton() -> None:
    names = list(SPEC.joint_names)
    names[0], names[9] = names[9], names[0]
    names[1], names[10] = names[10], names[1]
    reordered = SPEC.__class__(
        SPEC.name,
        tuple(names),
        SPEC.bones,
        SPEC.roles,
        SPEC.required_roles,
        SPEC.joint_groups,
    )
    candidate = np.ones((3, len(names), 3), dtype=np.float32)
    candidate[:, reordered.joint_index("left-hip")] = [-1.0, 0.0, 0.0]
    candidate[:, reordered.joint_index("right-hip")] = [1.0, 0.0, 0.0]
    reference = candidate + np.array([10.0, 0.0, 0.0], dtype=np.float32)
    candidate_valid = np.ones(candidate.shape[:2], dtype=bool)
    candidate_valid[:, [9, 10]] = False

    metrics, _ = external_metrics_from_reference(
        candidate, reference, reordered, candidate_valid
    )

    assert metrics["matched_frames"] == 3
    assert metrics["mpjpe"] < 1e-5


def test_circular_rom_and_unavailable_diagnostics_are_not_fabricated() -> None:
    sequence = _sequence()
    sequence.kpts_world[:, 5, 2] = np.array(
        [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
    )
    report = evaluate_person_trials("1", [sequence], SPEC)

    row = report.person_metrics[0]
    assert row["theta_rom"] < np.pi
    assert np.isnan(row["swap_error"])
    assert np.isnan(row["fixed_corruption_recovery"])


def test_circular_rom_does_not_unwrap_across_missing_frame_gaps() -> None:
    theta = np.array([0.0, 0.1, 0.0, 0.0, 3.0, -3.0])
    valid = np.array([True, True, False, False, True, True])

    assert _circular_rom(theta, valid) < 0.5


def test_old_full_sequence_is_sliced_by_new_trial_frame_maps(tmp_path: Path) -> None:
    inference = tmp_path / "inference" / "person_1"
    values = _sequence().kpts_world
    for trial_id, frame_ids in (("cycle_000", [0, 2]), ("cycle_001", [4, 6])):
        root = inference / trial_id
        root.mkdir(parents=True)
        np.savez_compressed(
            root / "fused_sequence.npz",
            kpts_world=values[:2],
            kpts_face_world=values[:2],
            kpts_side_world=values[:2],
            kpts_arithmetic_world=values[:2],
            kpts_base_world=values[:2],
            frame_valid=np.ones(2, dtype=bool),
            joint_valid=np.ones(values[:2].shape[:2], dtype=bool),
            face_map=np.array(frame_ids),
            side_map=np.array(frame_ids),
        )
    old = tmp_path / "old" / "legacy_method" / "person_1"
    old.mkdir(parents=True)
    np.savez_compressed(
        old / "fused_sequence.npz",
        kpts_world=values,
        face_map=np.arange(len(values)),
        side_map=np.arange(len(values)),
        fps=np.array(50.0),
    )

    sequences, _ = discover_method_sequences(
        tmp_path / "inference", tmp_path / "old", "1"
    )
    legacy = [sequence for sequence in sequences if sequence.method == "legacy_method"]

    assert [(sequence.trial_id, len(sequence.kpts_world)) for sequence in legacy] == [
        ("cycle_000", 2),
        ("cycle_001", 2),
    ]
    assert legacy[0].timestamps[1] - legacy[0].timestamps[0] == 2 / 50.0


def test_legacy_retention_metrics_have_explicit_unsupported_availability() -> None:
    sequence = _sequence()
    legacy = MethodSequence(
        "legacy",
        sequence.kpts_world,
        sequence.timestamps,
        diagnostic_status={"swap_error": "unsupported_legacy_output"},
    )

    row = evaluate_person_trials("1", [legacy], SPEC).person_metrics[0]

    assert row["rom_retention_availability"] == "unsupported_legacy_output"
    assert (
        row["peak_angular_velocity_retention_availability"]
        == "unsupported_legacy_output"
    )


def test_external_percentiles_are_computed_from_all_valid_points_not_cycle_means() -> (
    None
):
    short, long = _sequence(), _sequence()
    short = MethodSequence(
        short.method,
        short.kpts_world[:2],
        short.timestamps[:2],
        trial_id="cycle_000",
    )
    long = MethodSequence(
        long.method,
        np.repeat(long.kpts_world[:1], 30, axis=0),
        np.arange(30, dtype=np.float64) / 60.0,
        trial_id="cycle_001",
    )
    short_reference, long_reference = short.kpts_world.copy(), long.kpts_world.copy()
    for points, error in ((short_reference, 1.0), (long_reference, 10.0)):
        points[:, [2, 5, 6], 0] += error

    # Pinned to root alignment so the injected 1.0/10.0 offsets stay exact; this
    # test is about pooling percentiles across cycles, not about frame alignment.
    report = evaluate_person_trials(
        "1",
        [short, long],
        SPEC,
        references={"cycle_000": short_reference, "cycle_001": long_reference},
        alignment="root",
    )

    row = report.person_metrics[0]
    assert row["median"] == 10.0
    assert row["p95"] == 10.0


def test_derivatives_return_validity_masks_so_missing_intervals_are_not_averaged() -> (
    None
):
    values = np.array([[0.0], [1.0], [20.0], [3.0], [4.0]])
    derivative, valid = _derivative(
        values,
        np.arange(len(values), dtype=np.float64),
        1,
        np.array([[True], [True], [False], [True], [True]]),
    )

    assert valid[:, 0].tolist() == [True, False, False, True]
    assert np.mean(np.abs(derivative[valid])) == 1.0


def test_trunk_uses_median_first_dt_and_handles_single_frame() -> None:
    sequence = _sequence()
    _, _, _, omega_valid = _trunk(
        sequence.kpts_world[:1],
        np.ones(sequence.kpts_world[:1].shape[:2], dtype=bool),
        np.array([0.0]),
        SPEC,
    )

    assert not omega_valid.any()
