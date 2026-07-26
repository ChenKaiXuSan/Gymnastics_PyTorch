from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np
import pytest

from gymnastics.benchmarks.freeman.dataset import (
    load_session_reference,
    load_subject_sessions,
)
from gymnastics.benchmarks.freeman.mapping import FREEMAN_COCO17_NAMES
from gymnastics.benchmarks.freeman.pairing import select_camera_pair
from gymnastics.benchmarks.freeman.schema import FreeManCamera, FreeManSession


def test_loads_both_fps_subsets_and_exact_subject(freeman_fixture) -> None:
    sessions = load_subject_sessions(
        freeman_fixture.subject_root,
        freeman_fixture.shared_root,
        fps_values=(30, 60),
    )

    assert {(item.fps, item.subject_id) for item in sessions} == {(30, 1), (60, 1)}
    assert all(
        tuple(item.video_paths) == tuple(f"c{view:02d}" for view in range(1, 9))
        for item in sessions
    )
    assert all(item.split == "train" for item in sessions)
    assert all(item.scenario is None and item.action is None for item in sessions)
    assert all(not item.frame_ids.flags.writeable for item in sessions)


def test_records_trailing_video_exclusion_without_shifting_frames(
    freeman_fixture,
) -> None:
    freeman_fixture.rewrite_video(30, "c03", frames=2)

    session = load_subject_sessions(
        freeman_fixture.subject_root,
        freeman_fixture.shared_root,
        fps_values=(30,),
    )[0]

    np.testing.assert_array_equal(session.frame_ids, np.array([0, 1]))
    assert session.excluded_trailing_frames["c03"] == 0
    assert session.excluded_trailing_frames["c01"] == 1
    assert session.excluded_trailing_frames["keypoints2d"] == 1
    assert session.excluded_trailing_frames["keypoints3d"] == 1


def test_rejects_duplicate_session_ids(freeman_fixture) -> None:
    subset = freeman_fixture.shared_root / "30FPS"
    session = freeman_fixture.session_ids[30]
    (subset / "session_list.txt").write_text(
        f"{session}\n{session}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate session"):
        load_subject_sessions(
            freeman_fixture.subject_root,
            freeman_fixture.shared_root,
            fps_values=(30,),
        )


def test_rejects_ambiguous_validation_split_filename(freeman_fixture) -> None:
    subset = freeman_fixture.shared_root / "30FPS"
    (subset / "validation.txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="validation split"):
        load_subject_sessions(
            freeman_fixture.subject_root,
            freeman_fixture.shared_root,
            fps_values=(30,),
        )


def test_rejects_non_coco17_annotation_shape(freeman_fixture) -> None:
    subset = freeman_fixture.shared_root / "30FPS"
    session = freeman_fixture.session_ids[30]
    wrong = np.zeros((8, 3, 16, 3), dtype=np.float32)
    np.save(
        subset / "keypoints2d" / f"{session}.npy",
        np.asarray([{"keypoints2d": wrong}], dtype=object),
        allow_pickle=True,
    )

    with pytest.raises(ValueError, match=r"\[8,F,17,3\]"):
        load_subject_sessions(
            freeman_fixture.subject_root,
            freeman_fixture.shared_root,
            fps_values=(30,),
        )


def test_reference_uses_only_optimized_keypoints(freeman_fixture) -> None:
    session = load_subject_sessions(
        freeman_fixture.subject_root,
        freeman_fixture.shared_root,
        fps_values=(30,),
    )[0]

    reference = load_session_reference(session)

    expected = np.arange(3 * 17 * 3, dtype=np.float32).reshape(3, 17, 3)
    np.testing.assert_array_equal(reference.points_m, expected)
    assert reference.joint_names == FREEMAN_COCO17_NAMES
    assert not reference.points_m.flags.writeable


def _rvec_for_world_axis(axis: np.ndarray) -> np.ndarray:
    forward = np.asarray(axis, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, forward))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    first = np.cross(helper, forward)
    first /= np.linalg.norm(first)
    second = np.cross(forward, first)
    rotation = np.stack([first, second, forward], axis=0)
    return cv2.Rodrigues(rotation)[0].reshape(3)


def _camera(name: str, axis: tuple[float, float, float], height: float) -> FreeManCamera:
    rotation = _rvec_for_world_axis(np.asarray(axis, dtype=np.float64))
    matrix = np.eye(3, dtype=np.float64)
    world_to_camera = cv2.Rodrigues(rotation)[0]
    translation = -world_to_camera @ np.array([0.0, 0.0, height])
    return FreeManCamera(
        name=name,
        size=(64, 48),
        matrix=matrix,
        rotation=rotation,
        translation=translation,
        distortions=np.zeros(5, dtype=np.float64),
    )


def _pairing_session(cameras: dict[str, FreeManCamera]) -> FreeManSession:
    return FreeManSession(
        session_id="pairing_subj01",
        subject_id=1,
        fps=30,
        split="test",
        scenario=None,
        action=None,
        video_paths=MappingProxyType(
            {name: Path(f"/fixture/{name}.mp4") for name in cameras}
        ),
        cameras=MappingProxyType(cameras),
        keypoints2d_path=Path("/fixture/keypoints2d.npy"),
        keypoints3d_path=Path("/fixture/keypoints3d.npy"),
        frame_ids=np.arange(3, dtype=np.int64),
        excluded_trailing_frames=MappingProxyType({}),
    )


def test_selects_pair_closest_to_ninety_degrees() -> None:
    session = _pairing_session(
        {
            "c01": _camera("c01", (1.0, 0.0, 0.0), 1.0),
            "c02": _camera("c02", (1.0, 1.0, 0.0), 1.0),
            "c03": _camera("c03", (0.0, 1.0, 0.0), 1.0),
        }
    )

    pair = select_camera_pair(
        session,
        target_angle_deg=90.0,
        world_up=np.array([0.0, 0.0, 1.0]),
    )

    assert (pair.view_a, pair.view_b) == ("c01", "c03")
    assert pair.reference_view == "c01"
    assert pair.separation_deg == pytest.approx(90.0)
    assert pair.target_error_deg == pytest.approx(0.0)


def test_pair_tie_prefers_smaller_height_difference() -> None:
    session = _pairing_session(
        {
            "c01": _camera("c01", (1.0, 0.0, 0.0), 1.0),
            "c02": _camera("c02", (0.0, 1.0, 0.0), 3.0),
            "c03": _camera("c03", (0.0, -1.0, 0.0), 1.1),
        }
    )

    pair = select_camera_pair(
        session,
        target_angle_deg=90.0,
        world_up=np.array([0.0, 0.0, 1.0]),
    )

    assert (pair.view_a, pair.view_b) == ("c01", "c03")
    assert pair.height_difference == pytest.approx(0.1)


def test_pair_tie_uses_lexical_view_ids_last() -> None:
    session = _pairing_session(
        {
            "c01": _camera("c01", (1.0, 0.0, 0.0), 1.0),
            "c02": _camera("c02", (0.0, 1.0, 0.0), 1.0),
            "c03": _camera("c03", (0.0, -1.0, 0.0), 1.0),
        }
    )

    pair = select_camera_pair(
        session,
        target_angle_deg=90.0,
        world_up=np.array([0.0, 0.0, 1.0]),
    )

    assert (pair.view_a, pair.view_b) == ("c01", "c02")


def test_pairing_rejects_zero_world_up() -> None:
    session = _pairing_session(
        {
            "c01": _camera("c01", (1.0, 0.0, 0.0), 1.0),
            "c02": _camera("c02", (0.0, 1.0, 0.0), 1.0),
        }
    )

    with pytest.raises(ValueError, match="world_up"):
        select_camera_pair(
            session,
            target_angle_deg=90.0,
            world_up=np.zeros(3),
        )
