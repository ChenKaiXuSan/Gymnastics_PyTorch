"""Unity synthetic benchmark adapter (Dataset C).

Source dataset:
    The project's Unity benchmark: an animated humanoid recorded by two
    virtual cameras (``cam0``, ``cam1``) with exact 3D joint positions.
    SAM3D-Body is run on the rendered images by ``python -m fusion benchmark-unity``
    and cached per camera and sample id.

Expected directory structure:
    <benchmark_root>/skeleton.json, cameras.json, manifest.jsonl, images/...
    <sam3d_cache_root>/<camera_id>/<sample_id:08d>.npz   (+ summary.json)

Original skeleton:
    Inputs are MHR70 from SAM3D-Body.  Ground truth uses the 22-joint Unity
    humanoid; the 13 joints with a direct MHR70 homologue (see
    ``fusion.benchmarks.unity.mapping``) are scattered into MHR70 positions
    with a validity mask.

Original coordinate system:
    SAM3D views are in their own camera frames and are canonicalised into the
    pelvis body frame.  Ground truth is in Unity world metres.

Camera / view information:
    View A = ``cam0``, View B = ``cam1``; extrinsics are known but unused.

Sequence synchronisation:
    Exact: both cameras render the same simulation frame.

Ground truth:
    Native 3D, attached to validation/test samples when ``attach_reference``
    is on.

Cycle information:
    None in the manifest.  Cycles and middles are detected offline by
    ``python -m cycle_alignment cycles unity`` and read from
    ``options.cycle_records_root``; missing records are an error when
    ``options.require_cycle_records`` is true (default).

Split:
    Sequence-level.  Unity has a single "subject", so each evaluation
    sequence is treated as its own subject id for the split; the default
    reproduces the repository's strict direction-transfer fold
    ``left_to_right`` (train on the left sweep, test on the right sweep).

Options (``data.options``):
    benchmark_root, sam3d_cache_root, fps, fold ("left_to_right" |
    "right_to_left"), sequences (explicit list), cycle_records_root,
    require_cycle_records
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from common.paths import PROJECT_ROOT
from fusion.keypoints.schema import PosePairTrial

from ..sample import DualViewSample, sample_from_pose_pair_trial
from .base import DualViewDataModule, SplitSpec
from .cycle_records import public_cycles_for_sequence

SequenceLoader = Callable[[], Sequence[tuple[PosePairTrial, np.ndarray | None, np.ndarray | None]]]
"""Returns ``(trial, gt_world_m [T, 22, 3], gt_available [T, 22])`` per sequence."""


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def unity22_to_mhr70(points: np.ndarray, available: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scatter the directly homologous Unity joints into ``[T, 70, 3]`` MHR70 positions."""
    from fusion.benchmarks.unity.mapping import UNITY_JOINT_INDICES, _DIRECT

    source = np.asarray(points, dtype=np.float32)
    avail = np.asarray(available, dtype=bool)
    if source.ndim != 3 or source.shape[1] != len(UNITY_JOINT_INDICES) or source.shape[-1] != 3:
        raise ValueError("Unity points must have shape [T, 22, 3]")
    target = np.zeros((source.shape[0], 70, 3), dtype=np.float32)
    target_valid = np.zeros((source.shape[0], 70), dtype=bool)
    for name, mhr_index in _DIRECT.items():
        unity_index = UNITY_JOINT_INDICES[name]
        ok = avail[:, unity_index] & np.isfinite(source[:, unity_index]).all(axis=-1)
        target[:, mhr_index] = np.where(ok[:, None], source[:, unity_index], 0.0)
        target_valid[:, mhr_index] = ok
    return target, target_valid


class UnityDataModule(DualViewDataModule):
    """Unity benchmark sequences (see module docstring).

    Args:
        config: Data configuration.
        sequence_loader: Optional override returning the sequences; defaults
            to the benchmark manifest plus the SAM3D camera cache.
    """

    def __init__(self, config, *, sequence_loader: SequenceLoader | None = None) -> None:
        super().__init__(config)
        self._sequence_loader = sequence_loader or self._default_sequence_loader()

    def _default_sequence_loader(self) -> SequenceLoader:
        options = dict(self.config.options)
        benchmark_root = _resolve(options.get("benchmark_root", "/home/data/xchen/gymnastics/unity_benchmark"))
        cache_root = _resolve(options.get("sam3d_cache_root", "local/runs/unity_benchmark/sam3d"))
        fps = float(options.get("fps", 60.0))
        wanted = options.get("sequences")

        def load() -> Sequence[tuple[PosePairTrial, np.ndarray | None, np.ndarray | None]]:
            from fusion.benchmarks.unity.dataset import group_evaluation_sequences, load_unity_benchmark
            from fusion.benchmarks.unity.fusion import build_pose_pair_trial
            from fusion.benchmarks.unity.sam3d import load_sam3d_camera_cache

            benchmark = load_unity_benchmark(benchmark_root)
            result = []
            for sequence_id, frames in group_evaluation_sequences(benchmark).items():
                if sequence_id == "static_sweep" or (wanted and sequence_id not in set(wanted)):
                    continue
                sample_ids = np.asarray([frame.sample_id for frame in frames], dtype=np.int64)
                cam0 = load_sam3d_camera_cache(cache_root, "cam0", sample_ids)
                cam1 = load_sam3d_camera_cache(cache_root, "cam1", sample_ids)
                trial = build_pose_pair_trial(sequence_id, sample_ids, cam0.points_3d, cam1.points_3d, cam0.valid_3d, cam1.valid_3d, fps=fps)
                gt = np.stack([frame.gt_world_m for frame in frames])
                available = np.stack([frame.gt_available for frame in frames])
                result.append((trial, gt, available))
            return result

        return load

    def load_samples(self) -> Sequence[DualViewSample]:
        options = dict(self.config.options)
        records_root = _resolve(options.get("cycle_records_root", "local/runs/cycle_records/unity"))
        require_record = bool(options.get("require_cycle_records", True))
        samples: list[DualViewSample] = []
        for trial, gt, available in self._sequence_loader():
            reference = reference_valid = None
            if self.config.attach_reference and gt is not None and available is not None:
                reference, reference_valid = unity22_to_mhr70(gt, available)
            bounds, mids, record = public_cycles_for_sequence(records_root, trial.trial_id, trial.trial_id, trial, require_record=require_record)
            samples.append(
                sample_from_pose_pair_trial(
                    trial,
                    self.skeleton,
                    dataset="unity",
                    cycle_bounds=bounds,
                    cycle_mids=mids,
                    reference=reference,
                    reference_valid=reference_valid,
                    subject_id=trial.trial_id,
                    sequence_id=trial.trial_id,
                    metadata={"cycle_record": record is not None, "cycle_detection": dict(record.detection) if record is not None else {}},
                )
            )
        return samples

    def default_split(self, samples: Sequence[DualViewSample]) -> SplitSpec:
        from fusion.benchmarks.unity.supervised_data import UNITY_SUPERVISED_FOLDS

        ids = sorted({sample.subject_id for sample in samples})
        fold_name = str(self.config.options.get("fold", "left_to_right"))
        fold = UNITY_SUPERVISED_FOLDS.get(fold_name)
        if fold is not None and fold.train_sequence in ids and fold.test_sequence in ids:
            train = (fold.train_sequence,)
            test = (fold.test_sequence,)
            val = tuple(i for i in ids if i not in train + test)[:1] or ()
            return SplitSpec(train=train, val=val, test=test)
        n_test = 1 if len(ids) > 1 else 0
        n_val = 1 if len(ids) > 2 else 0
        return SplitSpec(train=tuple(ids[: len(ids) - n_val - n_test]), val=tuple(ids[len(ids) - n_val - n_test : len(ids) - n_test]), test=tuple(ids[len(ids) - n_test :]))
