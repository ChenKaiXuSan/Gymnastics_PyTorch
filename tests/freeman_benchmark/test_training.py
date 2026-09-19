from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gymnastics.benchmarks.freeman.evaluation import SessionMetrics
from gymnastics.benchmarks.freeman.fusion import RotationRuntime
from gymnastics.benchmarks.freeman.schema import PosePairInput, ViewPrediction
from gymnastics.benchmarks.freeman.training import (
    FoldRun,
    METHOD_PREFIX,
    TRAINING_SOURCE,
    assert_subject_disjoint,
    build_training_cache,
    build_training_trial,
    fold_run_for_subject,
    fold_runs_from_checkpoints,
    fuse_rotation_aware_trained,
    load_manifest_pair,
    load_manifest_sessions,
    make_subject_disjoint_folds,
    write_subject_disjoint_folds,
)
from gymnastics.benchmarks.freeman.training_cli import paired_subject_comparison
from gymnastics.common.skeletons.mhr70 import MHR70_NAMES
from gymnastics.fusion.rotation_aware.data import load_cached_trial
from gymnastics.fusion.rotation_aware.dataset import build_split_manifest

PAIR = {
    "session_id": "20220101_fixture_subj05",
    "view_a": "c01",
    "view_b": "c07",
    "reference_view": "c01",
    "separation_deg": 89.0,
    "target_error_deg": 1.0,
    "height_difference": 10.0,
}


def _write_view(root: Path, subject: int, session: str, view: str, frames: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    target = root / "sam3d" / f"subject_{subject:02d}" / session / view
    target.mkdir(parents=True)
    points = rng.normal(size=(frames, 70, 3)).astype(np.float32)
    points[..., 2] += 3.0
    valid = np.ones((frames, 70), dtype=bool)
    np.savez(
        target / "prediction.npz",
        frame_ids=np.arange(frames, dtype=np.int64),
        points3d=points,
        points2d=np.ones((frames, 70, 2), dtype=np.float32),
        valid3d=valid,
        valid2d=valid,
    )
    (target / "metadata.json").write_text(
        json.dumps(
            {
                "session_id": session,
                "subject_id": subject,
                "fps": 25.0,
                "view_id": view,
                "identity": {"view_id": view, "source_video_sha256": view * 32},
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def benchmark_root(tmp_path: Path) -> Path:
    root = tmp_path / "benchmark"
    session = PAIR["session_id"]
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    reference = tmp_path / f"{session}.npy"
    np.save(
        reference,
        {"keypoints3d_optim": np.random.default_rng(7).normal(size=(6, 17, 3)).astype(np.float32)},
    )
    manifests.joinpath("subject_05_sessions.json").write_text(
        json.dumps(
            {
                "subject_id": 5,
                "reference_scale_to_m": 0.01,
                "sessions": [
                    {
                        "session_id": session,
                        "fps": 25.0,
                        "split": "test",
                        "scenario": None,
                        "action": None,
                        "frames": 6,
                        "excluded_trailing_frames": {},
                        "keypoints3d_path": str(reference),
                        "pair": PAIR,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_view(root, 5, session, "c01", 6, seed=1)
    _write_view(root, 5, session, "c07", 6, seed=2)
    return root


def test_manifest_pair_matches_selected_views(benchmark_root: Path) -> None:
    sessions = load_manifest_sessions(benchmark_root, 5)
    assert [s.session_id for s in sessions] == [PAIR["session_id"]]
    assert sessions[0].frame_ids.tolist() == list(range(6))
    pair = load_manifest_pair(benchmark_root, sessions[0])
    assert (pair.view_a.view_id, pair.view_b.view_id) == ("c01", "c07")
    assert pair.subject_id == 5 and pair.fps == 25.0


def test_training_trial_marks_training_role_without_reference(benchmark_root: Path) -> None:
    session = load_manifest_sessions(benchmark_root, 5)[0]
    trial = build_training_trial(load_manifest_pair(benchmark_root, session))
    assert trial.person_id == "05" and trial.trial_id == session.session_id
    assert trial.joint_names == tuple(MHR70_NAMES)
    assert trial.source_metadata["zero_shot"] is False
    assert trial.source_metadata["training_source"] == TRAINING_SOURCE
    assert trial.source_metadata["reference_3d_consumed"] is False
    assert np.array_equal(trial.face_map, trial.side_map)


def test_training_cache_round_trips_through_rotation_aware_loader(
    benchmark_root: Path, tmp_path: Path
) -> None:
    cache_root = tmp_path / "cache"
    written = build_training_cache(
        benchmark_root, [5], cache_root, config_metadata={"window": {"length": 128}}
    )
    assert set(written) == {"05"}
    manifest = json.loads((cache_root / "person_05" / "manifest.json").read_text())
    assert manifest["source"]["offset_side_to_face"] == 0
    assert manifest["source"]["person_id"] == "05"
    assert manifest["source"]["reference_3d_consumed"] is False
    trial, metadata = load_cached_trial(cache_root / "person_05", PAIR["session_id"])
    assert trial.trial_id == PAIR["session_id"]
    assert trial.face.shape == (6, 70, 3)
    assert metadata["person_id"] == "05"


def test_subject_disjoint_folds_partition_cohort_and_share_validation(tmp_path: Path) -> None:
    folds = make_subject_disjoint_folds(
        evaluation_subjects=[1, 7, 9, 12],
        test_groups=[[1, 12], [7, 9]],
        val_subjects=[2, 4],
    )
    assert [fold.test for fold in folds] == [(1, 12), (7, 9)]
    assert folds[0].train == (7, 9) and folds[1].train == (1, 12)
    assert all(fold.val == (2, 4) for fold in folds)
    paths = write_subject_disjoint_folds(folds, tmp_path / "folds")
    manifest = build_split_manifest(paths[0])
    assert manifest.test == ("01", "12")
    assert manifest.val == ("02", "04")
    assert manifest.train == ("07", "09")


def test_subject_disjoint_folds_reject_leaky_designs() -> None:
    with pytest.raises(ValueError):
        make_subject_disjoint_folds(
            evaluation_subjects=[1, 7], test_groups=[[1]], val_subjects=[2]
        )
    with pytest.raises(ValueError):
        make_subject_disjoint_folds(
            evaluation_subjects=[1, 7], test_groups=[[1], [7]], val_subjects=[7]
        )
    with pytest.raises(ValueError):
        make_subject_disjoint_folds(
            evaluation_subjects=[1, 7], test_groups=[[1], [1, 7]], val_subjects=[2]
        )


def _fake_run(root: Path, run_id: str, *, train: list[str], val: list[str], test: list[str]) -> Path:
    run = root / run_id
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints" / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    (run / "split_manifest.json").write_text(
        json.dumps({"train": train, "val": val, "test": test}), encoding="utf-8"
    )
    return checkpoint


def test_fold_runs_recover_test_subjects_and_reject_duplicates(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _fake_run(runs_root, "f1", train=["07"], val=["02"], test=["01", "12"])
    _fake_run(runs_root, "f2", train=["01"], val=["02"], test=["07"])
    runs = fold_runs_from_checkpoints(runs_root, ["f1", "f2"])
    assert runs[0].test_subjects == (1, 12) and runs[1].test_subjects == (7,)
    assert fold_run_for_subject(runs, 12).run_id == "f1"
    with pytest.raises(ValueError):
        fold_run_for_subject(runs, 9)
    _fake_run(runs_root, "f3", train=["01"], val=["02"], test=["07"])
    with pytest.raises(ValueError):
        fold_runs_from_checkpoints(runs_root, ["f1", "f2", "f3"])


def test_assert_subject_disjoint_blocks_training_and_validation_subjects(tmp_path: Path) -> None:
    checkpoint = _fake_run(tmp_path, "f1", train=["07"], val=["02"], test=["05"])
    assert assert_subject_disjoint(checkpoint, 5)["test"] == ("05",)
    with pytest.raises(ValueError, match="training/validation"):
        assert_subject_disjoint(checkpoint, 7)
    with pytest.raises(ValueError, match="training/validation"):
        assert_subject_disjoint(checkpoint, 2)
    with pytest.raises(ValueError, match="not a declared test subject"):
        assert_subject_disjoint(checkpoint, 9)


def test_trained_fusion_runs_only_on_declared_test_subject(
    benchmark_root: Path, tmp_path: Path
) -> None:
    session = load_manifest_sessions(benchmark_root, 5)[0]
    pair = load_manifest_pair(benchmark_root, session)
    checkpoint = _fake_run(tmp_path / "runs", "f1", train=["07"], val=["02"], test=["05"])
    output_root = tmp_path / "out"

    def loader(path: Path, config: dict) -> RotationRuntime:
        return RotationRuntime(
            model=object(),
            skeleton=SimpleNamespace(joint_names=tuple(MHR70_NAMES)),
            provenance={"ablation": "A6", "checkpoint_path": str(path)},
            resolved_config={"window": {"length": 128, "eval_stride": 64}},
        )

    def runner(model, trial, skeleton, *, output_root, run_id, **kwargs):
        target = Path(output_root) / f"{run_id}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            kpts_world=(trial.face + trial.side) / 2,
            joint_valid=trial.valid_face & trial.valid_side,
            face_map=trial.face_map,
        )
        return SimpleNamespace(sequence_path=target)

    config = {"paths": {"output_root": str(output_root)}, "rotation_aware": {}}
    prediction = fuse_rotation_aware_trained(
        pair, checkpoint, "f1", config, family="a6", runtime_loader=loader, inference_runner=runner
    )
    assert prediction.method == f"{METHOD_PREFIX}:a6"
    assert prediction.metadata["zero_shot"] is False
    assert prediction.metadata["training_source"] == TRAINING_SOURCE
    assert prediction.metadata["fold_run_id"] == "f1"
    assert prediction.metadata["fold_test_subjects"] == ["05"]
    assert prediction.points.shape == (6, 70, 3)

    leaky = _fake_run(tmp_path / "runs", "f2", train=["05"], val=["02"], test=["07"])
    with pytest.raises(ValueError, match="training/validation"):
        fuse_rotation_aware_trained(
            pair, leaky, "f2", config, family="a6", runtime_loader=loader, inference_runner=runner
        )


def _row(subject: int, session: str, method: str, sim3: float, pa: float) -> SessionMetrics:
    return SessionMetrics(
        subject_id=subject,
        session_id=session,
        fps=25.0,
        split="test",
        scenario=None,
        action=None,
        method=method,
        classification="VALID",
        frames_total=10,
        frames_valid=10,
        valid_points=170,
        sim3_mpjpe_mm=sim3,
        median_mpjpe_mm=sim3,
        p95_mpjpe_mm=sim3,
        max_mpjpe_mm=sim3,
        root_mpjpe_mm=sim3,
        pa_mpjpe_mm=pa,
        velocity_error_mm_s=1.0,
        acceleration_error_mm_s2=1.0,
        pck={50: 0.5},
        auc=0.5,
        coverage=1.0,
        per_joint_mpjpe_mm=tuple([sim3] * 17),
    )


def test_paired_subject_comparison_uses_subject_means() -> None:
    rows = []
    for subject in range(1, 7):
        for session in ("a", "b"):
            base = 100.0 + subject
            rows.append(_row(subject, session, "ref", base, base))
            rows.append(_row(subject, session, "cand", base - 2.0, base - 1.0))
    table = paired_subject_comparison(
        rows, candidate="cand", references=["ref"], seed=1, bootstrap_samples=200
    )
    assert set(table["metric"]) == {"sim3_mpjpe_mm", "pa_mpjpe_mm"}
    sim3 = table[table["metric"] == "sim3_mpjpe_mm"].iloc[0]
    assert sim3["subjects"] == 6
    assert sim3["difference"] == pytest.approx(-2.0)
    assert sim3["improved_subjects"] == 6
    assert sim3["ci_high"] <= -2.0 + 1e-9
    assert 0.0 < sim3["p_holm"] <= 1.0


def test_deterministic_baselines_write_session_metrics(benchmark_root: Path, tmp_path: Path) -> None:
    from gymnastics.benchmarks.freeman.training import (
        evaluate_deterministic_methods,
        load_session_metric_rows,
    )

    rows = evaluate_deterministic_methods(
        benchmark_root=benchmark_root,
        output_root=tmp_path / "baselines",
        methods=("avg_body_current", "kalman_body_fusion"),
        subjects=[5],
        thresholds_mm=(50.0, 100.0, 150.0),
        reference_scale_to_m=0.01,
    )
    assert [row.method for row in rows] == ["avg_body_current", "kalman_body_fusion"]
    loaded = load_session_metric_rows(
        tmp_path / "baselines" / "evaluation" / "session_metrics", [5]
    )
    assert {row.method for row in loaded} == {"avg_body_current", "kalman_body_fusion"}
    assert (tmp_path / "baselines/fusion/methods/kalman_body_fusion/subject_05").is_dir()
