from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gymnastics.benchmarks.freeman.fusion import (
    METHOD_CLASSIFICATION,
    RotationRuntime,
    build_rotation_aware_trial,
    fuse_deterministic,
    fuse_rotation_aware,
    load_method_prediction,
    save_method_prediction,
)
from gymnastics.benchmarks.freeman.schema import PosePairInput, ViewPrediction
from gymnastics.common.skeletons.mhr70 import MHR70_NAMES
from gymnastics.fusion.deterministic.experiment_matrix import ALL_METHODS


def _view(
    *,
    view_id: str,
    points: np.ndarray,
    frame_ids: np.ndarray | None = None,
) -> ViewPrediction:
    frames = points.shape[0]
    ids = np.arange(frames, dtype=np.int64) if frame_ids is None else frame_ids
    valid = np.ones((frames, 70), dtype=bool)
    return ViewPrediction(
        session_id="fixture_subj01",
        subject_id=1,
        fps=30.0,
        view_id=view_id,
        frame_ids=ids,
        points3d=points,
        points2d=np.ones((frames, 70, 2), dtype=np.float32),
        valid3d=valid,
        valid2d=valid,
        metadata={
            "identity": {
                "view_id": view_id,
                "source_video_sha256": view_id * 32,
            }
        },
    )


@pytest.fixture
def pose_pair() -> PosePairInput:
    rng = np.random.default_rng(20260726)
    face = rng.normal(size=(6, 70, 3)).astype(np.float32)
    face[..., 2] += 3.0
    side = (face * 1.15 + np.array([0.4, -0.2, 0.7])).astype(np.float32)
    return PosePairInput(
        session_id="fixture_subj01",
        subject_id=1,
        fps=30.0,
        view_a=_view(view_id="c01", points=face),
        view_b=_view(view_id="c03", points=side),
    )


def test_runs_all_nine_registered_methods_without_reference_3d(
    pose_pair: PosePairInput,
) -> None:
    outputs = fuse_deterministic(pose_pair)

    assert tuple(item.method for item in outputs) == ALL_METHODS
    assert not hasattr(pose_pair, "reference")
    assert all(item.points.shape == (6, 70, 3) for item in outputs)
    assert all(item.valid.shape == (6, 70) for item in outputs)
    assert all(np.isfinite(item.points).all() for item in outputs)


def test_gt_weight_method_is_diagnostic_equal_fallback(
    pose_pair: PosePairInput,
) -> None:
    outputs = {item.method: item for item in fuse_deterministic(pose_pair)}
    diagnostic = outputs["sim3_face_stable_joint_weight"]

    assert (
        METHOD_CLASSIFICATION["sim3_face_stable_joint_weight"]
        == "GT_LEAKY_DIAGNOSTIC"
    )
    assert diagnostic.metadata["excluded_from_ranking"] is True
    assert (
        diagnostic.metadata["joint_weight_source"]
        == "unavailable_external_reference_equal_fallback"
    )
    assert diagnostic.metadata["reference_3d_consumed"] is False


def test_rejects_nonidentical_native_frame_ids(pose_pair: PosePairInput) -> None:
    mismatched = PosePairInput(
        session_id=pose_pair.session_id,
        subject_id=pose_pair.subject_id,
        fps=pose_pair.fps,
        view_a=pose_pair.view_a,
        view_b=_view(
            view_id="c03",
            points=np.array(pose_pair.view_b.points3d, copy=True),
            frame_ids=np.arange(1, 7, dtype=np.int64),
        ),
    )

    with pytest.raises(ValueError, match="exact synchronized frame IDs"):
        fuse_deterministic(mismatched)


def test_saves_compact_method_prediction_atomically(
    pose_pair: PosePairInput,
    tmp_path: Path,
) -> None:
    prediction = fuse_deterministic(
        pose_pair,
        methods=("avg_body_current",),
    )[0]

    path = save_method_prediction(prediction, tmp_path)

    assert path == (
        tmp_path
        / "avg_body_current"
        / "subject_01"
        / "fixture_subj01"
        / "fused_sequence.npz"
    )
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["frame_ids"], np.arange(6))
        assert data["points"].shape == (6, 70, 3)
        assert data["valid"].shape == (6, 70)
        assert "reference_3d_consumed" in str(data["metadata"].item())
    assert not path.with_suffix(".npz.tmp").exists()

    loaded = load_method_prediction(path)
    assert loaded.method == prediction.method
    assert loaded.metadata == prediction.metadata
    np.testing.assert_array_equal(loaded.points, prediction.points)


def test_builds_exact_zero_offset_mhr70_trial(pose_pair: PosePairInput) -> None:
    trial = build_rotation_aware_trial(pose_pair)

    np.testing.assert_array_equal(trial.face_map, pose_pair.view_a.frame_ids)
    np.testing.assert_array_equal(trial.side_map, pose_pair.view_b.frame_ids)
    np.testing.assert_allclose(
        trial.timestamps,
        pose_pair.view_a.frame_ids / pose_pair.fps,
    )
    assert trial.joint_names == tuple(MHR70_NAMES)
    assert trial.source_metadata["temporal_alignment"] == "native_zero_offset"
    assert trial.source_metadata["reference_3d_consumed"] is False


def test_rotation_aware_adapter_returns_zero_shot_prediction(
    pose_pair: PosePairInput,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    skeleton = SimpleNamespace(joint_names=tuple(MHR70_NAMES))

    def runtime_loader(path, config):
        return RotationRuntime(
            model=object(),
            skeleton=skeleton,
            provenance={
                "training_dataset": "private_gymnastics",
                "ablation": "A6",
                "checkpoint_path": str(path),
                "checkpoint_sha256": "a" * 64,
            },
            resolved_config={"window": {"length": 4, "eval_stride": 2}},
        )

    def inference_runner(model, trial, skeleton_spec, **kwargs):
        target = tmp_path / "native" / "fused_sequence.npz"
        target.parent.mkdir(parents=True)
        np.savez_compressed(
            target,
            kpts_world=np.array(trial.face, copy=True),
            joint_valid=np.array(trial.valid_face, copy=True),
            face_map=np.array(trial.face_map, copy=True),
        )
        return SimpleNamespace(sequence_path=target, metadata={"run_id": kwargs["run_id"]})

    prediction = fuse_rotation_aware(
        pose_pair,
        checkpoint,
        "paper_a6",
        {"paths": {"output_root": tmp_path / "runs"}},
        runtime_loader=runtime_loader,
        inference_runner=inference_runner,
    )

    assert prediction.method == "rotation_aware:paper_a6"
    np.testing.assert_array_equal(prediction.points, pose_pair.view_a.points3d)
    assert prediction.metadata["zero_shot"] is True
    assert prediction.metadata["reference_3d_consumed"] is False
    assert prediction.metadata["checkpoint_sha256"] == "a" * 64


def test_rotation_aware_rejects_freeman_training_provenance_before_inference(
    pose_pair: PosePairInput,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    called = False

    def runtime_loader(path, config):
        return RotationRuntime(
            model=object(),
            skeleton=SimpleNamespace(joint_names=tuple(MHR70_NAMES)),
            provenance={"training_dataset": "FreeMan", "ablation": "A6"},
            resolved_config={"window": {"length": 4, "eval_stride": 2}},
        )

    def inference_runner(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("leaky checkpoint must be rejected before inference")

    with pytest.raises(ValueError, match="FreeMan training provenance"):
        fuse_rotation_aware(
            pose_pair,
            checkpoint,
            "paper_a6",
            {"paths": {"output_root": tmp_path / "runs"}},
            runtime_loader=runtime_loader,
            inference_runner=inference_runner,
        )

    assert called is False
