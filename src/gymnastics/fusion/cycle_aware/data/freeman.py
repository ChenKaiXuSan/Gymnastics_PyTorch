"""FreeMan public dataset adapter (Dataset B).

Source dataset:
    FreeMan (Wang et al., 2023): 40 subjects, 8 synchronised cameras,
    markerless multi-view 3D reference.  This adapter does not read videos;
    it consumes the SAM3D-Body predictions cached by the zero-shot benchmark
    (``gymnastics benchmark freeman``), which selected two views per session.

Expected directory structure (``options.benchmark_root``):
    <root>/manifests/subject_NN_sessions.json         per-subject session list
    <root>/sam3d/subject_NN/<session>/<view>/prediction.npz  (+ sidecar JSON)
    and, for references, the official release files named in the manifest
    (``keypoints3d_path`` -> ``keypoints3d_optim`` [F, 17, 3]).

Original skeleton:
    Inputs are MHR70 from SAM3D-Body.  The reference is COCO17; it is placed
    at the 17 homologous MHR70 joint positions with a validity mask.

Original coordinate system:
    Each SAM3D view is in its own camera frame; both views are canonicalised
    into the pelvis body frame.  The reference is in the FreeMan world frame
    (centimetres in the release, scaled to metres with
    ``options.reference_scale_to_m``, default 0.01).

Camera / view information:
    ``view_a`` and ``view_b`` are the two views the benchmark selected
    (``pair`` entry of the manifest).

Sequence synchronisation:
    Native: FreeMan cameras are hardware-synchronised, so frame ids match
    exactly (zero offset).

Ground truth:
    Markerless multi-view optimisation (``keypoints3d_optim``); attached to
    validation/test samples only when ``attach_reference`` is on.

Cycle information:
    None in the release.  Cycles and middles are detected offline by
    ``gymnastics align cycles freeman`` and read from
    ``options.cycle_records_root`` (one ``cycle_record_v1`` file per
    session).  A session whose record lists no cycles is trained without
    phase; a missing record is an error when ``options.require_cycle_records``
    is true (default).

Split:
    Subject-disjoint by construction.  Default: a deterministic 70/15/15
    split of the loaded subjects (sorted ids); explicit lists via
    ``data.split``.

Options (``data.options``):
    benchmark_root, subjects (list of ints), reference_scale_to_m,
    cycle_records_root, require_cycle_records
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from gymnastics.common.paths import PROJECT_ROOT
from gymnastics.fusion.rotation_aware.schema import PosePairTrial

from ..sample import DualViewSample, sample_from_pose_pair_trial
from .base import DualViewDataModule, SplitSpec
from .cycle_records import public_cycles_for_sequence

SessionLoader = Callable[[int], Sequence[tuple[PosePairTrial, Path | None]]]
"""Returns ``(trial, keypoints3d_path)`` per session of one subject."""


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def coco17_to_mhr70(points: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Scatter ``[F, 17, 3]`` COCO17 points into ``[F, 70, 3]`` MHR70 positions."""
    from gymnastics.benchmarks.freeman.mapping import _MHR_INDICES

    source = np.asarray(points, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (17, 3):
        raise ValueError("COCO17 points must have shape [F, 17, 3]")
    finite = np.isfinite(source).all(axis=-1)
    if valid is not None:
        finite &= np.asarray(valid, dtype=bool)
    target = np.zeros((source.shape[0], 70, 3), dtype=np.float32)
    target_valid = np.zeros((source.shape[0], 70), dtype=bool)
    target[:, list(_MHR_INDICES)] = np.where(finite[..., None], source, 0.0)
    target_valid[:, list(_MHR_INDICES)] = finite
    return target, target_valid


class FreeManDataModule(DualViewDataModule):
    """FreeMan sessions from the zero-shot benchmark cache (see module docstring).

    Args:
        config: Data configuration.
        session_loader: Optional override returning ``(trial, keypoints3d_path)``
            per session of a subject id; defaults to the benchmark cache.
    """

    def __init__(self, config, *, session_loader: SessionLoader | None = None) -> None:
        super().__init__(config)
        self._session_loader = session_loader or self._default_session_loader()

    def _default_session_loader(self) -> SessionLoader:
        root = _resolve(self.config.options.get("benchmark_root", "local/runs/freeman_benchmark_cluster"))

        def load(subject: int) -> Sequence[tuple[PosePairTrial, Path | None]]:
            from gymnastics.benchmarks.freeman.fusion import build_rotation_aware_trial
            from gymnastics.benchmarks.freeman.training import load_manifest_pair, load_manifest_sessions

            result = []
            for session in load_manifest_sessions(root, subject):
                trial = build_rotation_aware_trial(load_manifest_pair(root, session))
                result.append((trial, Path(session.keypoints3d_path)))
            return result

        return load

    def _subjects(self) -> list[int]:
        subjects = self.config.options.get("subjects")
        if subjects:
            return sorted({int(s) for s in subjects})
        fold_subjects = self.fold_subjects()
        if fold_subjects is not None:
            return sorted({int(s) for s in fold_subjects})
        root = _resolve(self.config.options.get("benchmark_root", "local/runs/freeman_benchmark_cluster"))
        return sorted(int(path.stem.split("_")[1]) for path in (root / "manifests").glob("subject_*_sessions.json"))

    def _load_reference(self, path: Path | None, frames: int) -> tuple[np.ndarray, np.ndarray] | None:
        if path is None or not Path(path).is_file():
            return None
        scale = float(self.config.options.get("reference_scale_to_m", 0.01))
        payload = np.load(path, allow_pickle=True)
        payload = payload.item() if payload.shape == () else payload[0]
        points = np.asarray(payload["keypoints3d_optim"], dtype=np.float32)[:frames] * scale
        if points.shape[0] != frames:
            return None
        return coco17_to_mhr70(points)

    def load_samples(self) -> Sequence[DualViewSample]:
        options = dict(self.config.options)
        records_root = _resolve(options.get("cycle_records_root", "local/runs/cycle_records/freeman"))
        require_record = bool(options.get("require_cycle_records", True))
        samples: list[DualViewSample] = []
        for subject in self._subjects():
            subject_id = f"{int(subject):02d}"
            for trial, reference_path in self._session_loader(subject):
                reference = reference_valid = None
                if self.config.attach_reference:
                    loaded = self._load_reference(reference_path, trial.face.shape[0])
                    if loaded is not None:
                        reference, reference_valid = loaded
                bounds, mids, record = public_cycles_for_sequence(records_root, subject_id, trial.trial_id, trial, require_record=require_record)
                samples.append(
                    sample_from_pose_pair_trial(
                        trial,
                        self.skeleton,
                        dataset="freeman",
                        cycle_bounds=bounds,
                        cycle_mids=mids,
                        reference=reference,
                        reference_valid=reference_valid,
                        subject_id=subject_id,
                        sequence_id=trial.trial_id,
                        metadata={"cycle_record": record is not None, "cycle_detection": dict(record.detection) if record is not None else {}},
                    )
                )
        return samples

    def default_split(self, samples: Sequence[DualViewSample]) -> SplitSpec:
        subjects = sorted({sample.subject_id for sample in samples})
        n_val = max(1, int(round(0.15 * len(subjects)))) if len(subjects) >= 3 else (1 if len(subjects) == 2 else 0)
        n_test = max(1, int(round(0.15 * len(subjects)))) if len(subjects) >= 3 else 0
        n_train = len(subjects) - n_val - n_test
        return SplitSpec(train=tuple(subjects[:n_train]), val=tuple(subjects[n_train : n_train + n_val]), test=tuple(subjects[n_train + n_val :]))
