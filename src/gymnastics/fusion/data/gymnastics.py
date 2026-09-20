"""Private gymnastics dataset adapter (Dataset A).

Source dataset:
    Two-camera (face / side) recordings of 137 participants performing a
    repeated trunk-rotation exercise, processed by SAM3D-Body.  Inputs come
    either from the rotation-aware person cache
    (``local/runs/fuse_rotation_aware/cache/person_<id>``) or directly from
    the SAM3D result root plus the split-cycle alignment records.

Expected directory structure (cache mode, default):
    <cache_root>/person_<id>/manifest.json            (generation pointer)
    <cache_root>/person_<id>/.generations/<gen>/cycle_NNN.npz
  raw mode:
    <sam3d_root>/person_<id>/{face,side}/*.npz        SAM3D outputs
    <split_cycle_root>/person_<id>/alignment_record_<id>.json

Original skeleton:
    MHR70 (70 joints) for both views.

Original coordinate system:
    Each view is in its own camera-centred SAM3D world frame (unknown relative
    pose).  Views are canonicalised independently into the pelvis body frame.

Camera / view information:
    View A = face camera, View B = side camera.  No calibration is used.

Sequence synchronisation:
    The side view is shifted by the split-cycle ``offset_side_to_face``
    (integer frames, chosen by audio/keypoint alignment); the repository's
    ``build_aligned_timeline`` provides frame-exact pairs.

Ground truth:
    No marker-based ground truth.  A triangulated pseudo-reference (MHR70,
    calibrated world frame) exists per cycle under ``<triangulated_root>``;
    it is attached to validation/test samples only when
    ``attach_reference`` is on and is never read during training.

Cycle information:
    Exact, read from ``alignment_record_<id>.json`` (``gymnastics align``
    writes the boundaries, ``gymnastics align cycles private`` adds the
    turn-around middles).  Consecutive cycles of a person are concatenated
    into one sample so the long-term branch can see cycle-to-cycle
    recurrence.  With ``options.require_cycle_mids`` (default true) a record
    without middles is an error.

Split:
    The fixed paper protocol ``configs/fusion/folds/paper_137_a6_split.json``
    (96 / 27 / 14 people) is the default.

Options (``data.options``):
    source: "cache" | "raw"
    cache_root, sam3d_root, split_cycle_root, triangulated_root, fold_json,
    persons (optional explicit list), require_cycle_mids
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from gymnastics.common.paths import PROJECT_ROOT
from gymnastics.keypoints.schema import PosePairTrial, valid_from_points

from ..sample import DualViewSample, sample_from_pose_pair_trial
from .base import DualViewDataModule, SplitSpec
from .cycle_records import private_cycles_for_trials

TrialLoader = Callable[[str], Sequence[PosePairTrial]]
ReferenceLoader = Callable[[str, str], tuple[np.ndarray, list[tuple[int, int]]] | None]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def concatenate_cycles(trials: Sequence[PosePairTrial]) -> tuple[PosePairTrial, tuple[tuple[int, int], ...]]:
    """Join consecutive cycle trials of one person into one trial with cycle bounds.

    Timestamps are rebuilt from the face video frame index so gaps between
    cycles keep their physical duration.

    Args:
        trials: Cycle trials of one person (any order; sorted by face frame).

    Returns:
        Tuple ``(joined_trial, cycle_bounds)``.
    """
    if not trials:
        raise ValueError("no trials to concatenate")
    ordered = sorted(trials, key=lambda trial: int(trial.face_map[0]))
    person = ordered[0].person_id
    fps = float(ordered[0].fps)
    if any(t.person_id != person or t.fps != fps for t in ordered):
        raise ValueError("all trials must belong to one person with one fps")
    bounds: list[tuple[int, int]] = []
    offset = 0
    for trial in ordered:
        bounds.append((offset, offset + trial.face.shape[0]))
        offset += trial.face.shape[0]
    face = np.concatenate([t.face for t in ordered])
    side = np.concatenate([t.side for t in ordered])
    face_map = np.concatenate([t.face_map for t in ordered])
    side_map = np.concatenate([t.side_map for t in ordered])
    if np.any(np.diff(face_map) <= 0):
        raise ValueError(f"cycles of person {person} overlap in face frames")
    timestamps = (face_map - face_map[0]).astype(np.float64) / fps
    joined = PosePairTrial(
        face=face,
        side=side,
        valid_face=np.concatenate([t.valid_face for t in ordered]),
        valid_side=np.concatenate([t.valid_side for t in ordered]),
        timestamps=timestamps,
        face_map=face_map,
        side_map=side_map,
        joint_names=ordered[0].joint_names,
        person_id=person,
        trial_id="all_cycles",
        fps=fps,
        source_metadata={"cycles": [t.trial_id for t in ordered], **dict(ordered[0].source_metadata)},
    )
    return joined, tuple(bounds)


def reference_from_triangulation(
    trial: PosePairTrial,
    cycle_ids: Sequence[str],
    loader: ReferenceLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill a ``[T, 70, 3]`` reference from per-cycle triangulated sequences.

    Frames are matched by ``(face_frame, side_frame)`` pairs, exactly like the
    rotation-aware evaluation layer.

    Args:
        trial: The concatenated trial.
        cycle_ids: Cycle identifiers (``cycle_NNN``) to look up.
        loader: ``loader(person_id, cycle_id) -> (joints [F, 70, 3], pairs)`` or ``None``.

    Returns:
        Tuple ``(reference, reference_valid)``.
    """
    frames = trial.face.shape[0]
    reference = np.zeros((frames, trial.face.shape[1], 3), dtype=np.float32)
    valid = np.zeros((frames, trial.face.shape[1]), dtype=bool)
    index = {(int(f), int(s)): i for i, (f, s) in enumerate(zip(trial.face_map, trial.side_map))}
    for cycle_id in cycle_ids:
        loaded = loader(trial.person_id, cycle_id)
        if loaded is None:
            continue
        joints, pairs = loaded
        for row, pair in enumerate(pairs):
            target = index.get((int(pair[0]), int(pair[1])))
            if target is None:
                continue
            points = np.asarray(joints[row], dtype=np.float32)
            finite = np.isfinite(points).all(axis=-1)
            reference[target] = np.where(finite[:, None], points, 0.0)
            valid[target] = finite
    return reference, valid


class GymnasticsDataModule(DualViewDataModule):
    """Private gymnastics recordings (see module docstring).

    Args:
        config: Data configuration.
        trial_loader: Optional override returning the cycle trials of a
            person id (used by tests); defaults to the cache or raw loaders.
        reference_loader: Optional override returning ``(joints, pairs)`` for
            ``(person_id, cycle_id)``; defaults to the triangulation loader.
    """

    def __init__(self, config, *, trial_loader: TrialLoader | None = None, reference_loader: ReferenceLoader | None = None) -> None:
        super().__init__(config)
        self._trial_loader = trial_loader or self._default_trial_loader()
        self._reference_loader = reference_loader or self._default_reference_loader()

    # ----- default loaders -------------------------------------------------
    def _default_trial_loader(self) -> TrialLoader:
        options = dict(self.config.options)
        source = str(options.get("source", "cache"))
        if source == "cache":
            cache_root = _resolve(options.get("cache_root", "local/runs/fuse_rotation_aware/cache"))

            def load_from_cache(person_id: str) -> Sequence[PosePairTrial]:
                from gymnastics.keypoints.data import load_cached_trial, resolve_cache_manifest

                person_dir = cache_root / f"person_{person_id}"
                _, manifest = resolve_cache_manifest(person_dir)
                return [load_cached_trial(person_dir, trial_id)[0] for trial_id in manifest["trials"]]

            return load_from_cache
        if source == "raw":
            sam3d_root = _resolve(options.get("sam3d_root", "/home/data/xchen/gymnastics/sam3d_body_results"))
            split_root = _resolve(options.get("split_cycle_root", "local/runs/split_cycle"))

            def load_raw(person_id: str) -> Sequence[PosePairTrial]:
                from gymnastics.keypoints.config import load_skeleton_spec
                from gymnastics.keypoints.data import load_person_trials

                spec = load_skeleton_spec(PROJECT_ROOT / "configs" / "fusion" / "skeleton_mhr70.yaml")
                return load_person_trials(person_id, sam3d_root, split_root, spec)

            return load_raw
        raise ValueError("gymnastics options.source must be 'cache' or 'raw'")

    def _default_reference_loader(self) -> ReferenceLoader:
        root = _resolve(self.config.options.get("triangulated_root", "/home/data/xchen/gymnastics/sam3d_triangulated/person"))

        def load(person_id: str, cycle_id: str):
            from gymnastics.analysis.compare_fused_triangulated import load_triangulated_sequence

            cycle_root = root / f"person_{person_id}" / cycle_id
            if not (cycle_root / "joints_3d_sequence.npz").is_file():
                return None
            return load_triangulated_sequence(cycle_root)

        return load

    def _person_ids(self) -> list[str]:
        persons = self.config.options.get("persons")
        if persons:
            return [str(p) for p in persons]
        fold_subjects = self.fold_subjects()
        if fold_subjects is not None:
            return list(fold_subjects)
        fold = json.loads(_resolve(self.config.options.get("fold_json", "configs/fusion/folds/paper_137_a6_split.json")).read_text(encoding="utf-8"))
        return sorted({str(p) for split in ("train", "val", "test") for p in fold.get(split, [])}, key=lambda s: (len(s), s))

    # ----- DualViewDataModule ------------------------------------------------
    @property
    def reference_allowed_in_training(self) -> bool:
        """The triangulated pseudo-reference comes from the same two views: never a training target."""
        return False

    def load_samples(self) -> Sequence[DualViewSample]:
        samples: list[DualViewSample] = []
        split_root = _resolve(self.config.options.get("split_cycle_root", "local/runs/split_cycle"))
        require_mids = bool(self.config.options.get("require_cycle_mids", True))
        for person_id in self._person_ids():
            trials = list(self._trial_loader(person_id))
            if not trials:
                continue
            joined, bounds = concatenate_cycles(trials)
            ordered = sorted(trials, key=lambda t: int(t.face_map[0]))
            record_path = split_root / f"person_{person_id}" / f"alignment_record_{person_id}.json"
            record_bounds, mids = private_cycles_for_trials(record_path, ordered, require_mids=require_mids)
            if record_bounds != bounds:
                raise ValueError(f"{record_path}: cycle boundaries differ from the cached trials")
            reference = reference_valid = None
            if self.config.attach_reference:
                reference, reference_valid = reference_from_triangulation(joined, [t.trial_id for t in sorted(trials, key=lambda t: int(t.face_map[0]))], self._reference_loader)
                if not reference_valid.any():
                    reference = reference_valid = None
            samples.append(
                sample_from_pose_pair_trial(
                    joined,
                    self.skeleton,
                    dataset="gymnastics",
                    cycle_bounds=bounds,
                    cycle_mids=mids,
                    reference=reference,
                    reference_valid=reference_valid,
                    subject_id=person_id,
                    sequence_id="all_cycles",
                    metadata={"cycles": len(bounds)},
                )
            )
        return samples

    def default_split(self, samples: Sequence[DualViewSample]) -> SplitSpec:
        fold = json.loads(_resolve(self.config.options.get("fold_json", "configs/fusion/folds/paper_137_a6_split.json")).read_text(encoding="utf-8"))
        known = {sample.subject_id for sample in samples}

        def members(name: str) -> tuple[str, ...]:
            return tuple(sorted({str(p) for p in fold.get(name, []) if str(p) in known}, key=lambda s: (len(s), s)))

        return SplitSpec(train=members("train"), val=members("val"), test=members("test"))
