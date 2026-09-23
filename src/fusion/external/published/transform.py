"""Trial transforms that replace the SAM3D views with a published method's poses.

A :class:`ViewSource` knows where each dataset keeps the 2D keypoints of the
two views of a trial; :class:`LiftedTrialTransform` lifts them with a
monocular lifter, maps the result into the MHR70 layout and returns a trial
on the same frames, either with the two monocular poses in the two view
slots (``mode="per_view"``, evaluated as single views) or with their
Procrustes-averaged fusion in both slots (``mode="procrustes_average"``,
evaluated as the two-view result). Each video is lifted once as a whole (the
authors' inference setting, full temporal context) and cached by its name,
then sampled at the trial's frames.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from common.paths import DERIVED_ROOT, FIT3D_ROOT, PROJECT_ROOT, SAM3D_RESULTS_ROOT
from fusion.keypoints.schema import PosePairTrial

from .fuse import procrustes_average
from .keypoints2d import DEFAULT_CACHE_ROOT, View2D, fit3d_view, freeman_view, gymnastics_view
from .mapping import fill_missing_joints, h36m17_to_mhr70, mhr70_to_coco17_2d

Lifter = Callable[[np.ndarray, int, int], np.ndarray]
"""``(coco17 [T, 17, 2], width, height) -> h36m17 [T, 17, 3]``."""


class ViewSource(Protocol):
    def views(self, trial: PosePairTrial) -> tuple[View2D, View2D]:
        """2D keypoints of view A and view B covering the trial's frames."""

    def rotations(self, trial: PosePairTrial) -> tuple[np.ndarray, np.ndarray]:
        """World-to-camera rotation matrices of view A and view B (supervised
        methods express their targets in the camera frame); raises for
        datasets without calibration."""


class GymnasticsViewSource:
    """Private recordings: person id + face/side per-frame SAM3D files (cached)."""

    def __init__(self, *, cache_root: Path = DEFAULT_CACHE_ROOT, sam3d_root: Path = SAM3D_RESULTS_ROOT) -> None:
        self.cache_root, self.sam3d_root = Path(cache_root), Path(sam3d_root)

    def views(self, trial: PosePairTrial) -> tuple[View2D, View2D]:
        return gymnastics_view(trial.person_id, "face", cache_root=self.cache_root, sam3d_root=self.sam3d_root), gymnastics_view(trial.person_id, "side", cache_root=self.cache_root, sam3d_root=self.sam3d_root)

    def rotations(self, trial: PosePairTrial) -> tuple[np.ndarray, np.ndarray]:
        raise ValueError("the private recordings have no independent 3D reference: supervised methods cannot be trained on them")


class FreeManViewSource:
    """FreeMan benchmark cache: the two views the benchmark selected per session."""

    def __init__(self, benchmark_root: Path) -> None:
        self.benchmark_root = Path(benchmark_root)
        self._pairs: dict[str, tuple[int, str, str]] = {}

    def _pair(self, session_id: str, subject_id: int) -> tuple[int, str, str]:
        if session_id not in self._pairs:
            from fusion.benchmarks.freeman.training import load_manifest_sessions

            for session in load_manifest_sessions(self.benchmark_root, int(subject_id)):
                self._pairs[session.session_id] = (int(subject_id), session.pair.view_a, session.pair.view_b)
        return self._pairs[session_id]

    def views(self, trial: PosePairTrial) -> tuple[View2D, View2D]:
        subject, view_a, view_b = self._pair(trial.trial_id, int(trial.source_metadata.get("subject_id", trial.person_id)))
        return freeman_view(self.benchmark_root, subject, trial.trial_id, view_a), freeman_view(self.benchmark_root, subject, trial.trial_id, view_b)

    def rotations(self, trial: PosePairTrial) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        from fusion.benchmarks.freeman.dataset import _load_cameras
        from fusion.benchmarks.freeman.training import load_manifest_sessions

        subject, view_a, view_b = self._pair(trial.trial_id, int(trial.source_metadata.get("subject_id", trial.person_id)))
        session = next(s for s in load_manifest_sessions(self.benchmark_root, subject) if s.session_id == trial.trial_id)
        # The release keeps cameras/, keypoints2d/ and keypoints3d/ side by side.
        cameras = _load_cameras(Path(session.keypoints3d_path).parents[1] / "cameras" / f"{trial.trial_id}.json")
        return tuple(cv2.Rodrigues(np.asarray(cameras[v].rotation, dtype=np.float64))[0] for v in (view_a, view_b))  # type: ignore[return-value]


class Fit3DViewSource:
    """Fit3D: the external per-video cache on the sequence's two selected cameras."""

    def __init__(self, *, derived_root: Path | None = None, dataset_root: Path | None = None, split: str = "train") -> None:
        self.derived_root = Path(derived_root) if derived_root is not None else DERIVED_ROOT / "sam3d_fit3d"
        self.dataset_root = Path(dataset_root) if dataset_root is not None else FIT3D_ROOT
        self.split = str(split)

    def _sequence(self, trial: PosePairTrial):
        from fusion.benchmarks.fit3d.dataset import discover_sequences

        action = str(trial.source_metadata["action"])
        sequences = discover_sequences(self.dataset_root, split=self.split, subjects=[trial.person_id], actions=[action])
        if not sequences:
            raise FileNotFoundError(f"fit3d sequence {trial.person_id}/{action} not found below {self.dataset_root}")
        return sequences[0]

    def views(self, trial: PosePairTrial) -> tuple[View2D, View2D]:
        sequence = self._sequence(trial)
        meta = trial.source_metadata
        frame_ids = np.asarray(trial.face_map, dtype=np.int64)
        return tuple(fit3d_view(self.derived_root, sequence, str(meta[key]), frame_ids, split=self.split) for key in ("view_a", "view_b"))  # type: ignore[return-value]

    def rotations(self, trial: PosePairTrial) -> tuple[np.ndarray, np.ndarray]:
        from fusion.benchmarks.fit3d.dataset import load_camera

        sequence = self._sequence(trial)
        meta = trial.source_metadata
        return tuple(np.asarray(load_camera(sequence, str(meta[key])).rotation, dtype=np.float64) for key in ("view_a", "view_b"))  # type: ignore[return-value]


def view_source(dataset: str, **options) -> ViewSource:
    if dataset == "gymnastics":
        return GymnasticsViewSource(**options)
    if dataset == "freeman":
        return FreeManViewSource(Path(options.get("benchmark_root", PROJECT_ROOT / "local/runs/freeman_benchmark_cluster")))
    if dataset == "fit3d":
        return Fit3DViewSource(**options)
    raise ValueError(f"no published-method view source for dataset {dataset!r}")


class LiftedTrialTransform:
    """Replace both views of a trial with a monocular lifter's 3D poses.

    Args:
        lifter: Monocular 2D -> 3D lifter (COCO17 in, H36M17 out).
        source: Where the trial's 2D keypoints come from.
        mode: ``"per_view"`` or ``"procrustes_average"`` (see module docstring).
        cache_dir: Directory for per-video lifted caches (``None`` = no cache).
        method: Cache namespace (e.g. ``"videopose3d"``).
    """

    def __init__(self, lifter: Lifter, source: ViewSource, *, mode: str = "procrustes_average", cache_dir: Path | None = None, method: str = "lifter") -> None:
        if mode not in {"per_view", "procrustes_average"}:
            raise ValueError("mode must be 'per_view' or 'procrustes_average'")
        self.lifter, self.source, self.mode, self.cache_dir, self.method = lifter, source, mode, cache_dir, method

    def _cache_path(self, view: View2D) -> Path | None:
        if self.cache_dir is None or not view.name:
            return None
        key = hashlib.sha1(json.dumps({"name": view.name, "frames": [int(view.frame_ids[0]), int(view.frame_ids[-1]), int(len(view.frame_ids))]}, sort_keys=True).encode()).hexdigest()[:16]
        return Path(self.cache_dir) / self.method / (view.name.replace("/", "__") + f"_{key}.npz")

    def lift_view_h36m(self, view: View2D) -> tuple[np.ndarray, np.ndarray]:
        """Lift a whole video in the lifter's own layout: ``(h36m [N, 17, 3], frame_valid [N])``."""
        path = self._cache_path(view)
        if path is not None and path.is_file():
            payload = np.load(path)
            if "h36m" in payload:
                return payload["h36m"], payload["frame_valid"]
        coco, coco_valid = mhr70_to_coco17_2d(view.points, view.valid)
        lifted = self.lifter(fill_missing_joints(coco, coco_valid), view.width, view.height)
        # A frame whose 2D input was missing has no meaningful lift.
        frame_valid = coco_valid.all(axis=1)
        if path is not None:
            pose, pose_valid = h36m17_to_mhr70(lifted)
            pose_valid &= frame_valid[:, None]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, pose=pose, valid=pose_valid, frame_ids=view.frame_ids, h36m=lifted.astype(np.float32), frame_valid=frame_valid)
        return lifted.astype(np.float32), frame_valid

    def lift_view(self, view: View2D) -> tuple[np.ndarray, np.ndarray]:
        """Lift a whole video: ``(pose [N, 70, 3], valid [N, 70])`` on ``view.frame_ids``."""
        path = self._cache_path(view)
        if path is not None and path.is_file():
            payload = np.load(path)
            return payload["pose"], payload["valid"]
        lifted, frame_valid = self.lift_view_h36m(view)
        pose, pose_valid = h36m17_to_mhr70(lifted)
        pose_valid &= frame_valid[:, None]
        return pose, pose_valid

    def lifted_views(self, trial: PosePairTrial) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(pose_a, valid_a, pose_b, valid_b)`` in MHR70 layout on the trial's frames."""
        view_a, view_b = self.source.views(trial)
        output = []
        for view, frame_map in ((view_a, trial.face_map), (view_b, trial.side_map)):
            pose, valid = self.lift_view(view)
            wanted = np.asarray(frame_map, dtype=np.int64)
            position = np.clip(np.searchsorted(view.frame_ids, wanted), 0, len(view.frame_ids) - 1)
            hit = view.frame_ids[position] == wanted
            output.append(np.where(hit[:, None, None], pose[position], 0.0).astype(np.float32))
            output.append(valid[position] & hit[:, None])
        return output[0], output[1], output[2], output[3]

    def __call__(self, trial: PosePairTrial) -> PosePairTrial:
        pose_a, valid_a, pose_b, valid_b = self.lifted_views(trial)
        if self.mode == "procrustes_average":
            fused, valid = procrustes_average(pose_a, valid_a, pose_b, valid_b)
            pose_a = pose_b = fused
            valid_a = valid_b = valid
        metadata = {**dict(trial.source_metadata), "external_method": self.method, "external_mode": self.mode}
        return replace(trial, face=pose_a, side=pose_b, valid_face=valid_a, valid_side=valid_b, source_metadata=metadata)
