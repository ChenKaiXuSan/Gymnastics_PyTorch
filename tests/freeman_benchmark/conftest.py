from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


@dataclass(frozen=True)
class FreeManFixture:
    subject_root: Path
    shared_root: Path
    session_ids: dict[int, str]

    def video_path(self, fps: int, view: str) -> Path:
        session = self.session_ids[fps]
        return (
            self.subject_root
            / f"{fps}FPS"
            / "videos"
            / session
            / "vframes"
            / f"{view}.mp4"
        )

    def rewrite_video(self, fps: int, view: str, frames: int) -> None:
        _write_video(self.video_path(fps, view), fps=fps, frames=frames)


def _write_video(path: Path, *, fps: int, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV test fixture could not create MP4")
    try:
        for frame in range(frames):
            image = np.full((48, 64, 3), frame * 30, dtype=np.uint8)
            writer.write(image)
    finally:
        writer.release()


def _camera_payload() -> list[dict]:
    return [
        {
            "name": f"c{view:02d}",
            "size": [64, 48],
            "matrix": [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]],
            "rotation": [0.0, 0.0, 0.0],
            "translation": [float(view), 0.0, 1.0],
            "distortions": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        for view in range(1, 9)
    ]


@pytest.fixture
def freeman_fixture(tmp_path: Path) -> FreeManFixture:
    subject_root = tmp_path / "subject_01"
    shared_root = tmp_path / "shared"
    session_ids = {
        30: "20260726_fixture30_subj01",
        60: "20260726_fixture60_subj01",
    }
    for fps, session in session_ids.items():
        subset = shared_root / f"{fps}FPS"
        (subset / "cameras").mkdir(parents=True)
        (subset / "keypoints2d").mkdir()
        (subset / "keypoints3d").mkdir()
        (subset / "session_list.txt").write_text(session + "\n", encoding="utf-8")
        (subset / "train.txt").write_text(session + "\n", encoding="utf-8")
        (subset / "valid.txt").write_text("", encoding="utf-8")
        (subset / "test.txt").write_text("", encoding="utf-8")
        (subset / "cameras" / f"{session}.json").write_text(
            json.dumps(_camera_payload()),
            encoding="utf-8",
        )
        keypoints2d = np.zeros((8, 3, 17, 3), dtype=np.float32)
        keypoints2d[..., :2] = 10.0
        keypoints2d[..., 2] = 1.0
        np.save(
            subset / "keypoints2d" / f"{session}.npy",
            np.asarray(
                [{"keypoints2d": keypoints2d, "center": None, "scale": None}],
                dtype=object,
            ),
            allow_pickle=True,
        )
        keypoints3d = np.arange(3 * 17 * 3, dtype=np.float32).reshape(3, 17, 3)
        np.save(
            subset / "keypoints3d" / f"{session}.npy",
            np.asarray(
                [
                    {
                        "keypoints3d_optim": keypoints3d,
                        "keypoints3d": keypoints3d + 1.0,
                    }
                ],
                dtype=object,
            ),
            allow_pickle=True,
        )
        for view in range(1, 9):
            _write_video(
                subject_root
                / f"{fps}FPS"
                / "videos"
                / session
                / "vframes"
                / f"c{view:02d}.mp4",
                fps=fps,
                frames=3,
            )
    return FreeManFixture(subject_root, shared_root, session_ids)
