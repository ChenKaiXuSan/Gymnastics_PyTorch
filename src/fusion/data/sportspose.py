"""SportsPose public dataset adapter (Dataset D).

Source dataset:
    SportsPose (Ingwersen et al., 2023): 26 subjects (22 indoors + 4
    outdoors), 7 calibrated cameras at 90 fps, markerless multi-view 3D
    reference (COCO17, metres). Five actions (jump, soccer, tennis,
    throw_baseball, volley), about five 3-second trials per subject and
    action. This adapter does not read videos; it consumes the SAM3D-Body
    predictions cached by ``python -m fusion benchmark-sportspose infer`` on
    the two views chosen by ``select-views``.

Expected directory structure (``options.benchmark_root``):
    <root>/selected_views.json                              view A/B per subject and sequence
    <root>/sam3d/<day>/<S>/<activity>/<clip>/<view>.npz (+ .json)
    or, when ``options.sam3d_derived_root`` points to an existing directory,
    the external per-video cache ``<derived>/<day>/<S>/<Video_dir>/CAM<k>.npz``
    (every camera and frame) thinned to every ``options.frame_stride``-th frame.

Original skeleton:
    Inputs are MHR70 from SAM3D-Body. The reference is COCO17 and is placed
    at the 17 homologous MHR70 joints with a validity mask.

Original coordinate system:
    Each SAM3D view is in its own camera frame; both views are canonicalised
    into the pelvis body frame. The reference is in the calibration world
    frame (metres).

Camera / view information:
    View A is the camera closest to the subject's facing direction, view B
    the camera closest to 90 degrees of azimuth from it (the private
    face/side roles), chosen per subject and sequence; see
    ``fusion.benchmarks.sportspose.dataset``.

Sequence synchronisation:
    Native: hardware-synchronised cameras, frame ids match (zero offset).

Ground truth:
    Markerless multi-view reference, sampled at the cached frames; attached
    to validation/test samples only when ``attach_reference`` is on.

Cycle information:
    Trial-as-cycle. One sample per subject, day and activity
    (``<day>_<activity>``) concatenates that subject's trials of the action (``fusion.benchmarks.sportspose.trials``);
    every trial is one cycle whose middle is the extremum of the wrist
    azimuth, read from the records written by
    ``python -m cycle_alignment cycles sportspose``.

Split:
    Subject-disjoint by S-id (S00 and S10 were recorded indoors and
    outdoors; both days stay on the same side of the split). Default: deterministic
    70/15/15 split of the loaded subjects; folds via ``data.fold_json``
    (``src/configs/fusion/folds/sportspose``).

Options (``data.options``):
    benchmark_root, dataset_root, sam3d_derived_root, frame_stride,
    subjects (list of S-ids), activities, days,
    cycle_records_root, require_cycle_records
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from common.paths import PROJECT_ROOT, SPORTSPOSE_ROOT
from fusion.keypoints.schema import PosePairTrial

from ..sample import DualViewSample, sample_from_pose_pair_trial
from .base import DualViewDataModule, SplitSpec
from .cycle_records import public_cycles_for_sequence
from .freeman import coco17_to_mhr70

SequenceLoader = Callable[[], Sequence[tuple[PosePairTrial, np.ndarray | None]]]
"""Returns ``(trial, reference [T, 17, 3] or None)`` per subject/activity sequence."""


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


class SportsPoseDataModule(DualViewDataModule):
    """SportsPose sequences from the benchmark cache (see module docstring).

    Args:
        config: Data configuration.
        sequence_loader: Optional override returning ``(trial, reference)``
            per sequence; defaults to the benchmark cache.
    """

    def __init__(self, config, *, sequence_loader: SequenceLoader | None = None) -> None:
        super().__init__(config)
        self._sequence_loader = sequence_loader or self._default_sequence_loader()

    def _wanted_subjects(self) -> set[str] | None:
        subjects = self.config.options.get("subjects")
        if subjects:
            return {str(s) for s in subjects}
        fold_subjects = self.fold_subjects()
        return {str(s) for s in fold_subjects} if fold_subjects is not None else None

    def _default_sequence_loader(self) -> SequenceLoader:
        options = dict(self.config.options)
        benchmark_root = _resolve(options.get("benchmark_root", "local/runs/sportspose_benchmark"))
        dataset_root = _resolve(options.get("dataset_root", SPORTSPOSE_ROOT))
        wanted_subjects = self._wanted_subjects()
        wanted_activities = options.get("activities")
        wanted_days = options.get("days")
        attach = bool(self.config.attach_reference)
        derived_root = _resolve(options["sam3d_derived_root"]) if options.get("sam3d_derived_root") else None
        if derived_root is not None and not derived_root.is_dir():
            derived_root = None
        frame_stride = int(options.get("frame_stride", 3))

        def load() -> Sequence[tuple[PosePairTrial, np.ndarray | None]]:
            from fusion.benchmarks.sportspose.cli import read_selected_views
            from fusion.benchmarks.sportspose.dataset import discover_clips, group_clips
            from fusion.benchmarks.sportspose.trials import build_sequence_trial, load_clip_predictions

            views = read_selected_views(benchmark_root / "selected_views.json")
            clips = discover_clips(dataset_root, days=wanted_days, activities=wanted_activities)
            result = []
            for (subject_key, sequence_key), group in group_clips(clips).items():
                if wanted_subjects is not None and subject_key not in wanted_subjects:
                    continue
                selected = views[(subject_key, sequence_key)]
                predictions = [load_clip_predictions(clip, selected, cache_root=benchmark_root / "sam3d", derived_root=derived_root, frame_stride=frame_stride) for clip in group]
                trial, _, reference = build_sequence_trial(group, predictions, selected, reference=attach)
                result.append((trial, reference))
            return result

        return load

    def load_samples(self) -> Sequence[DualViewSample]:
        options = dict(self.config.options)
        records_root = _resolve(options.get("cycle_records_root", "local/runs/cycle_records/sportspose"))
        require_record = bool(options.get("require_cycle_records", True))
        samples: list[DualViewSample] = []
        for trial, reference_points in self._sequence_loader():
            reference = reference_valid = None
            if self.config.attach_reference and reference_points is not None:
                reference, reference_valid = coco17_to_mhr70(reference_points)
            bounds, mids, record = public_cycles_for_sequence(records_root, trial.person_id, trial.trial_id, trial, require_record=require_record)
            samples.append(
                sample_from_pose_pair_trial(
                    trial,
                    self.skeleton,
                    dataset="sportspose",
                    cycle_bounds=bounds,
                    cycle_mids=mids,
                    reference=reference,
                    reference_valid=reference_valid,
                    subject_id=trial.person_id,
                    sequence_id=trial.trial_id,
                    metadata={"action": trial.source_metadata.get("activity"), "day": trial.source_metadata.get("day"), "clips": list(trial.source_metadata.get("clips", ())), "cycle_record": record is not None, "cycle_detection": dict(record.detection) if record is not None else {}},
                )
            )
        return samples

    def default_split(self, samples: Sequence[DualViewSample]) -> SplitSpec:
        subjects = sorted({sample.subject_id for sample in samples})
        n_val = max(1, int(round(0.15 * len(subjects)))) if len(subjects) >= 3 else (1 if len(subjects) == 2 else 0)
        n_test = max(1, int(round(0.15 * len(subjects)))) if len(subjects) >= 3 else 0
        n_train = len(subjects) - n_val - n_test
        return SplitSpec(train=tuple(subjects[:n_train]), val=tuple(subjects[n_train : n_train + n_val]), test=tuple(subjects[n_train + n_val :]))
