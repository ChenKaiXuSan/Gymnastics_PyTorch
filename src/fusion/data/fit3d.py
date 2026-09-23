"""Fit3D public dataset adapter (Dataset C).

Source dataset:
    Fit3D (Fieraru et al., CVPR 2021, "AIFit"): eight subjects with 3D
    ground truth perform 47 fitness exercises (squats, lunges, presses,
    curls, warm-ups) in front of four synchronised calibrated cameras at
    50 fps. The release annotates every repetition, so the movement cycles
    of this dataset are ground truth rather than detector output -- the
    closest public analogue of the private recordings.

Expected directory structure:
    <dataset_root>/train/<subject>/{joints3d_25,camera_parameters,rep_ann.json}
    <benchmark_root>/selected_views.json                       (``benchmark-fit3d select-views``)
    <sam3d_derived_root>/train/<subject>/<camera>/<action>.npz (external per-video SAM3D cache)

Original skeleton:
    Inputs are MHR70 from SAM3D-Body. The reference is ``joints3d_25``
    (Human3.6M-17 core plus two foot and two hand joints per side); the 14
    unambiguous joints are placed at their MHR70 positions and everything
    else stays invalid (Fit3D has no heel, eyes or ears, and its foot joints
    cannot be told apart reliably).

Original coordinate system:
    Each SAM3D view is in its own camera frame; both views are canonicalised
    into the pelvis body frame. The reference is in the calibration world
    frame (metres).

Camera / view information:
    View A is the camera closest to the direction the subject faces and view
    B the camera closest to 90 degrees of azimuth from it; the rig only
    offers separations of about 46, 132 and 178 degrees, so the selected
    pair is normally 132 degrees apart (a weaker baseline than FreeMan's 90
    degrees). Calibration is used for this choice only.

Sequence synchronisation:
    Native: hardware-synchronised cameras, frame ids match (zero offset).
    The cameras record one or two frames more than the reference, so the
    common frame grid is the shortest of the three.

Ground truth:
    Multi-view fitted reference, sampled at the cached frames; attached to
    validation / test samples only when ``attach_reference`` is on.

Cycle information:
    From ``rep_ann.json``: consecutive repetition marks are the cycles, and
    the turn-around middles are written by
    ``python -m cycle_alignment cycles fit3d``. 296 of the 376 sequences are
    annotated (1526 repetitions); ``require_cycle_records`` keeps the rest
    out of training.

Split:
    Subject-disjoint over the eight reference subjects; folds via
    ``data.fold_json`` (``src/configs/fusion/folds/fit3d``).

Options (``data.options``):
    dataset_root, benchmark_root, sam3d_derived_root, split, frame_stride,
    subjects, actions, min_cycles, cycle_records_root, require_cycle_records
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from common.paths import FIT3D_ROOT, PROJECT_ROOT
from fusion.keypoints.schema import PosePairTrial

from ..sample import DualViewSample, sample_from_pose_pair_trial
from .base import DualViewDataModule, SplitSpec
from .cycle_records import public_cycles_for_sequence

SequenceLoader = Callable[[], Sequence[tuple[PosePairTrial, np.ndarray | None]]]
"""Returns ``(trial, reference [T, 25, 3] or None)`` per subject/exercise sequence."""


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def joints3d_25_to_mhr70(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scatter ``[T, 25, 3]`` Fit3D reference joints into ``[T, 70, 3]`` MHR70 slots.

    Only the joints whose correspondence is unambiguous are filled (nose,
    neck, shoulders, elbows, wrists, hips, knees, ankles); the rest of the
    validity mask stays false.
    """
    from common.skeletons.mhr70 import MHR70_INDEX
    from fusion.benchmarks.fit3d.schema import JOINTS3D_25_TO_MHR70

    source = np.asarray(points, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (25, 3):
        raise ValueError("Fit3D reference joints must have shape [T, 25, 3]")
    target = np.zeros((source.shape[0], 70, 3), dtype=np.float32)
    valid = np.zeros((source.shape[0], 70), dtype=bool)
    for name, column in JOINTS3D_25_TO_MHR70.items():
        if name not in MHR70_INDEX:
            continue  # informational entries such as "pelvis-approx"
        joint = MHR70_INDEX[name]
        finite = np.isfinite(source[:, column]).all(axis=-1)
        target[:, joint] = np.where(finite[:, None], source[:, column], 0.0)
        valid[:, joint] = finite
    return target, valid


class Fit3DDataModule(DualViewDataModule):
    """Fit3D sequences from the external SAM3D cache (see module docstring).

    Args:
        config: Data configuration.
        sequence_loader: Optional override returning ``(trial, reference)``
            per sequence; defaults to the release plus the cache.
    """

    def __init__(self, config, *, sequence_loader: SequenceLoader | None = None, trial_transform=None) -> None:
        super().__init__(config, trial_transform=trial_transform)
        self._sequence_loader = sequence_loader or self._default_sequence_loader()

    def _wanted_subjects(self) -> set[str] | None:
        subjects = self.config.options.get("subjects")
        if subjects:
            return {str(s) for s in subjects}
        fold_subjects = self.fold_subjects()
        return {str(s) for s in fold_subjects} if fold_subjects is not None else None

    def _default_sequence_loader(self) -> SequenceLoader:
        options = dict(self.config.options)
        dataset_root = _resolve(options.get("dataset_root", FIT3D_ROOT))
        benchmark_root = _resolve(options.get("benchmark_root", "local/runs/fit3d_benchmark"))
        derived_root = _resolve(options.get("sam3d_derived_root", "local/runs/fit3d_benchmark/sam3d"))
        split = str(options.get("split", "train"))
        frame_stride = int(options.get("frame_stride", 1))
        wanted_actions = options.get("actions")
        attach = bool(self.config.attach_reference)

        def load() -> Sequence[tuple[PosePairTrial, np.ndarray | None]]:
            from fusion.benchmarks.fit3d.cli import read_selected_views
            from fusion.benchmarks.fit3d.dataset import discover_sequences
            from fusion.benchmarks.fit3d.trials import load_sequence

            views = read_selected_views(benchmark_root / "selected_views.json")
            wanted_subjects = self._wanted_subjects()
            result: list[tuple[PosePairTrial, np.ndarray | None]] = []
            for sequence in discover_sequences(dataset_root, split=split, subjects=sorted(wanted_subjects) if wanted_subjects else None, actions=wanted_actions):
                selected = views.get((sequence.subject_key, sequence.sequence_key))
                if selected is None:
                    continue
                try:
                    trial, reference = load_sequence(sequence, selected, derived_root, frame_stride=frame_stride, split=split, reference=attach)
                except FileNotFoundError:
                    continue  # the SAM3D cache does not cover this sequence yet
                result.append((trial, reference))
            return result

        return load

    def load_samples(self) -> Sequence[DualViewSample]:
        options = dict(self.config.options)
        records_root = _resolve(options.get("cycle_records_root", "local/runs/cycle_records/fit3d"))
        require_record = bool(options.get("require_cycle_records", True))
        min_cycles = int(options.get("min_cycles", 0))
        samples: list[DualViewSample] = []
        for trial, reference_points in self._sequence_loader():
            trial = self._transform(trial)
            reference = reference_valid = None
            if self.config.attach_reference and reference_points is not None:
                reference, reference_valid = joints3d_25_to_mhr70(reference_points)
            bounds, mids, record = public_cycles_for_sequence(records_root, trial.person_id, trial.trial_id, trial, require_record=require_record)
            if len(bounds) < min_cycles:
                continue
            samples.append(
                sample_from_pose_pair_trial(
                    trial,
                    self.skeleton,
                    dataset="fit3d",
                    cycle_bounds=bounds,
                    cycle_mids=mids,
                    reference=reference,
                    reference_valid=reference_valid,
                    subject_id=trial.person_id,
                    sequence_id=trial.trial_id,
                    metadata={
                        "action": trial.source_metadata.get("action"),
                        "separation_deg": trial.source_metadata.get("separation_deg"),
                        "cycle_record": record is not None,
                        "cycle_source": "rep_ann",
                        "cycle_detection": dict(record.detection) if record is not None else {},
                    },
                )
            )
        return samples

    def default_split(self, samples: Sequence[DualViewSample]) -> SplitSpec:
        subjects = sorted({sample.subject_id for sample in samples})
        if len(subjects) < 3:
            return SplitSpec(train=tuple(subjects[:1]), val=tuple(subjects[1:2]), test=tuple(subjects[2:]))
        n_test = max(1, round(0.2 * len(subjects)))
        n_val = max(1, round(0.2 * len(subjects)))
        return SplitSpec(train=tuple(subjects[: len(subjects) - n_val - n_test]), val=tuple(subjects[len(subjects) - n_val - n_test : len(subjects) - n_test]), test=tuple(subjects[len(subjects) - n_test :]))
