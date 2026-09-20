"""Subject-disjoint FreeMan training support for the rotation-aware model.

The zero-shot benchmark applies private-data checkpoints to FreeMan without
any FreeMan training. This module adds the complementary experiment: the same
self-supervised model is trained on FreeMan itself, with subjects held out so
that every evaluated session comes from a subject the checkpoint never saw.

Three pieces are provided.

* ``build_training_cache`` turns the cached per-view SAM3D predictions of the
  zero-shot benchmark into rotation-aware person caches (one "person" per
  FreeMan subject, one trial per session). The FreeMan 3D reference is never
  read here, so training stays label-free exactly as on the private data.
* ``write_subject_disjoint_folds`` writes the fold JSONs consumed by
  ``gymnastics fuse rotation-aware train``.
* ``fuse_rotation_aware_trained`` and ``evaluate_trained_family`` run one
  FreeMan-trained checkpoint per session out-of-fold and score it with the
  session evaluator of the zero-shot benchmark, so the rows are directly
  comparable with the existing FreeMan tables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gymnastics.keypoints.data import write_person_cache
from gymnastics.keypoints.schema import PosePairTrial

from .dataset import load_session_reference
from .evaluation import SessionMetrics, evaluate_session
from .fusion import (
    RotationRuntime,
    _default_runtime_loader,
    _file_sha256,
    build_rotation_aware_trial,
    fuse_deterministic,
    fuse_rotation_aware,
    save_method_prediction,
)
from .sam3d import load_inference
from .schema import MethodPrediction, PosePairInput, SelectedPair

TRAINING_SOURCE = "freeman_subject_disjoint"
METHOD_PREFIX = "rotation_aware_trained"
ALIGNMENT_RECORD = "freeman_native_zero_offset"

RuntimeLoader = Callable[[Path, Mapping[str, Any]], RotationRuntime]


def person_id_for_subject(subject: int) -> str:
    """Rotation-aware person id used for one FreeMan subject."""
    subject = int(subject)
    if subject < 1 or subject > 40:
        raise ValueError("FreeMan subject must be within 1..40")
    return f"{subject:02d}"


@dataclass(frozen=True)
class ManifestSession:
    """Session metadata recovered from the zero-shot benchmark manifest.

    Only the fields consumed by ``load_session_reference`` and the evaluator
    are kept, so per-subject video workspaces are not required.
    """

    session_id: str
    subject_id: int
    fps: float
    split: str
    scenario: str | None
    action: str | None
    frames: int
    keypoints3d_path: Path
    pair: SelectedPair

    def __post_init__(self) -> None:
        if not self.session_id or self.frames <= 0 or self.fps <= 0:
            raise ValueError("manifest session requires id, frames, and positive FPS")
        object.__setattr__(self, "keypoints3d_path", Path(self.keypoints3d_path))
        object.__setattr__(self, "fps", float(self.fps))

    @property
    def frame_ids(self) -> np.ndarray:
        return np.arange(self.frames, dtype=np.int64)


def manifest_path(benchmark_root: Path, subject: int) -> Path:
    return (
        Path(benchmark_root)
        / "manifests"
        / f"subject_{int(subject):02d}_sessions.json"
    )


def load_manifest_sessions(
    benchmark_root: Path,
    subject: int,
) -> tuple[ManifestSession, ...]:
    """Load the sessions that the zero-shot benchmark processed for one subject."""
    path = manifest_path(benchmark_root, subject)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or int(payload.get("subject_id", -1)) != int(subject):
        raise ValueError(f"manifest subject mismatch: {path}")
    sessions: list[ManifestSession] = []
    for entry in payload.get("sessions", ()):
        pair = SelectedPair(**dict(entry["pair"]))
        sessions.append(
            ManifestSession(
                session_id=str(entry["session_id"]),
                subject_id=int(subject),
                fps=float(entry["fps"]),
                split=str(entry.get("split", "unassigned")),
                scenario=entry.get("scenario"),
                action=entry.get("action"),
                frames=int(entry["frames"]),
                keypoints3d_path=Path(entry["keypoints3d_path"]),
                pair=pair,
            )
        )
    if not sessions:
        raise ValueError(f"manifest lists no sessions: {path}")
    return tuple(sorted(sessions, key=lambda item: item.session_id))


def _prediction_path(benchmark_root: Path, session: ManifestSession, view: str) -> Path:
    return (
        Path(benchmark_root)
        / "sam3d"
        / f"subject_{session.subject_id:02d}"
        / session.session_id
        / view
        / "prediction.npz"
    )


def load_manifest_pair(benchmark_root: Path, session: ManifestSession) -> PosePairInput:
    """Load the selected two-view SAM3D pair of one session from the cache."""
    views = []
    for view in (session.pair.view_a, session.pair.view_b):
        path = _prediction_path(benchmark_root, session, view)
        prediction = load_inference(path)
        if (
            prediction.session_id != session.session_id
            or prediction.subject_id != session.subject_id
            or prediction.view_id != view
        ):
            raise RuntimeError(f"SAM3D cache identity mismatch: {path}")
        views.append(prediction)
    return PosePairInput(
        session_id=session.session_id,
        subject_id=session.subject_id,
        fps=session.fps,
        view_a=views[0],
        view_b=views[1],
    )


def build_training_trial(pair: PosePairInput) -> PosePairTrial:
    """Build one training trial; identical geometry to the zero-shot trial."""
    trial = build_rotation_aware_trial(pair)
    return replace(
        trial,
        source_metadata={
            **dict(trial.source_metadata),
            "zero_shot": False,
            "role": "freeman_training_cache",
            "training_source": TRAINING_SOURCE,
            "reference_3d_consumed": False,
        },
    )


def build_training_cache(
    benchmark_root: Path,
    subjects: Sequence[int],
    cache_root: Path,
    *,
    config_metadata: Mapping[str, Any],
) -> dict[str, Path]:
    """Write one rotation-aware person cache per FreeMan subject.

    Returns the manifest path per person id. The FreeMan reference files are
    never opened.
    """
    written: dict[str, Path] = {}
    for subject in sorted({int(value) for value in subjects}):
        sessions = load_manifest_sessions(benchmark_root, subject)
        trials = [
            build_training_trial(load_manifest_pair(benchmark_root, session))
            for session in sessions
        ]
        fps_values = sorted({float(trial.fps) for trial in trials})
        person_id = person_id_for_subject(subject)
        source_metadata = {
            "dataset": "FreeMan",
            "alignment_record": ALIGNMENT_RECORD,
            "offset_side_to_face": 0,
            "fps": fps_values[0] if len(fps_values) == 1 else fps_values,
            "person_id": person_id,
            "subject_id": subject,
            "benchmark_root": str(Path(benchmark_root).resolve()),
            "sessions": [session.session_id for session in sessions],
            "reference_3d_consumed": False,
            "training_source": TRAINING_SOURCE,
        }
        written[person_id] = write_person_cache(
            trials,
            cache_root,
            source_metadata=source_metadata,
            config_metadata=dict(config_metadata),
        )
    return written


@dataclass(frozen=True)
class SubjectDisjointFold:
    name: str
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = {"train": set(self.train), "val": set(self.val), "test": set(self.test)}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(f"fold {self.name}: {left}/{right} overlap {sorted(overlap)}")
        if not self.train or not self.test:
            raise ValueError(f"fold {self.name} requires train and test subjects")

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": "FreeMan",
            "protocol": "subject_disjoint",
            "train": [person_id_for_subject(s) for s in sorted(self.train)],
            "val": [person_id_for_subject(s) for s in sorted(self.val)],
            "test": [person_id_for_subject(s) for s in sorted(self.test)],
        }


def make_subject_disjoint_folds(
    *,
    evaluation_subjects: Sequence[int],
    test_groups: Sequence[Sequence[int]],
    val_subjects: Sequence[int],
    name_template: str = "fold_{index:02d}",
) -> tuple[SubjectDisjointFold, ...]:
    """Cross-validation folds whose test groups partition the evaluation cohort.

    Training uses the evaluation subjects not in the fold's test group;
    validation (checkpoint selection) uses a fixed disjoint subject set so
    every fold selects its checkpoint on identical data.
    """
    evaluation = tuple(sorted({int(s) for s in evaluation_subjects}))
    validation = tuple(sorted({int(s) for s in val_subjects}))
    if set(validation) & set(evaluation):
        raise ValueError("validation subjects must not overlap the evaluation cohort")
    covered: list[int] = []
    folds: list[SubjectDisjointFold] = []
    for index, group in enumerate(test_groups, start=1):
        test = tuple(sorted({int(s) for s in group}))
        if not set(test).issubset(evaluation):
            raise ValueError(f"test group {test} is not inside the evaluation cohort")
        train = tuple(s for s in evaluation if s not in test)
        folds.append(
            SubjectDisjointFold(
                name=name_template.format(index=index),
                train=train,
                val=validation,
                test=test,
            )
        )
        covered.extend(test)
    if sorted(covered) != list(evaluation):
        raise ValueError("test groups must partition the evaluation cohort exactly once")
    return tuple(folds)


def write_subject_disjoint_folds(
    folds: Sequence[SubjectDisjointFold],
    fold_root: Path,
) -> tuple[Path, ...]:
    root = Path(fold_root)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fold in folds:
        path = root / f"{fold.name}.json"
        path.write_text(json.dumps(fold.as_json(), indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def _run_split_manifest(checkpoint: Path) -> dict[str, tuple[str, ...]]:
    """Read the person split recorded next to a rotation-aware checkpoint."""
    run_root = Path(checkpoint).resolve().parent.parent
    manifest = run_root / "split_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"trained checkpoint lacks split_manifest.json: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("split_manifest.json must be a mapping")
    return {
        name: tuple(str(value) for value in payload.get(name, ()))
        for name in ("train", "val", "test")
    }


def assert_subject_disjoint(checkpoint: Path, subject: int) -> dict[str, tuple[str, ...]]:
    """Refuse to evaluate a subject the checkpoint trained or validated on."""
    split = _run_split_manifest(checkpoint)
    person = person_id_for_subject(subject)
    if person in split["train"] or person in split["val"]:
        raise ValueError(
            f"subject {subject} was used for training/validation of {checkpoint}"
        )
    if person not in split["test"]:
        raise ValueError(
            f"subject {subject} is not a declared test subject of {checkpoint}"
        )
    return split


def fuse_rotation_aware_trained(
    pair: PosePairInput,
    checkpoint: Path,
    run_id: str,
    config: Mapping[str, Any],
    *,
    family: str,
    runtime_loader: RuntimeLoader | None = None,
    inference_runner: Callable[..., Any] | None = None,
) -> MethodPrediction:
    """Run one FreeMan-trained checkpoint on a pair from one of its test subjects."""
    if not run_id or not family:
        raise ValueError("rotation-aware run_id and family are required")
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    split = assert_subject_disjoint(checkpoint_path, pair.subject_id)
    trial = build_rotation_aware_trial(pair)
    loader = runtime_loader or _default_runtime_loader
    runtime = loader(checkpoint_path, config)
    if inference_runner is None:
        from gymnastics.archive.rotation_aware.inference import run_inference

        inference_runner = run_inference
    window = runtime.resolved_config.get("window", {})
    if not isinstance(window, Mapping):
        raise ValueError("rotation-aware resolved config requires window mapping")
    output_root = (
        Path(config["paths"]["output_root"]).resolve()
        / "fusion"
        / METHOD_PREFIX
        / run_id
        / "native"
    )
    result = inference_runner(
        runtime.model,
        trial,
        runtime.skeleton,
        output_root=output_root,
        run_id=run_id,
        window_length=int(window.get("length", 128)),
        stride=int(window.get("eval_stride", 64)),
        provenance=dict(runtime.provenance),
        resolved_config=dict(runtime.resolved_config),
    )
    sequence_path = Path(result.sequence_path).resolve()
    with np.load(sequence_path, allow_pickle=False) as data:
        points = np.asarray(data["kpts_world"], dtype=np.float32)
        valid = np.asarray(data["joint_valid"], dtype=bool)
        frame_ids = (
            np.asarray(data["face_map"], dtype=np.int64)
            if "face_map" in data
            else np.array(pair.view_a.frame_ids, copy=True)
        )
    if points.shape != pair.view_a.points3d.shape or valid.shape != points.shape[:2]:
        raise ValueError("rotation-aware output does not match MHR70 pair shape")
    if not np.array_equal(frame_ids, pair.view_a.frame_ids):
        raise ValueError("rotation-aware output changed native frame identity")
    valid &= np.isfinite(points).all(axis=-1)
    points = np.where(valid[..., None], points, 0)
    provenance = dict(runtime.provenance)
    method = f"{METHOD_PREFIX}:{family}"
    return MethodPrediction(
        method=method,
        session_id=pair.session_id,
        subject_id=pair.subject_id,
        fps=pair.fps,
        points=points,
        valid=valid,
        frame_ids=frame_ids,
        metadata={
            "dataset": "FreeMan",
            "method": method,
            "classification": "VALID",
            "excluded_from_ranking": False,
            "zero_shot": False,
            "reference_3d_consumed": False,
            "training_source": TRAINING_SOURCE,
            "family": family,
            "fold_run_id": run_id,
            "fold_test_subjects": list(split["test"]),
            "ablation": provenance.get("ablation"),
            "checkpoint_path": provenance.get("checkpoint_path", str(checkpoint_path)),
            "checkpoint_sha256": provenance.get(
                "checkpoint_sha256",
                _file_sha256(checkpoint_path),
            ),
        },
    )


@dataclass(frozen=True)
class FoldRun:
    """One trained fold: the run id, its checkpoint, and its test subjects."""

    run_id: str
    checkpoint: Path
    test_subjects: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "test_subjects", tuple(int(s) for s in self.test_subjects))
        if not self.run_id or not self.test_subjects:
            raise ValueError("fold run requires a run id and test subjects")


def fold_runs_from_checkpoints(
    run_root: Path,
    run_ids: Sequence[str],
) -> tuple[FoldRun, ...]:
    """Recover each fold's test subjects from its recorded split manifest."""
    runs: list[FoldRun] = []
    seen: set[int] = set()
    for run_id in run_ids:
        checkpoint = Path(run_root) / run_id / "checkpoints" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing trained checkpoint: {checkpoint}")
        split = _run_split_manifest(checkpoint)
        subjects = tuple(int(person) for person in split["test"])
        duplicate = seen & set(subjects)
        if duplicate:
            raise ValueError(f"test subjects {sorted(duplicate)} appear in more than one fold")
        seen.update(subjects)
        runs.append(FoldRun(run_id=run_id, checkpoint=checkpoint, test_subjects=subjects))
    return tuple(runs)


def fold_run_for_subject(runs: Sequence[FoldRun], subject: int) -> FoldRun:
    matches = [run for run in runs if int(subject) in run.test_subjects]
    if len(matches) != 1:
        raise ValueError(f"subject {subject} must belong to exactly one fold, found {len(matches)}")
    return matches[0]


def evaluate_trained_family(
    *,
    benchmark_root: Path,
    output_root: Path,
    rotation_config: Path,
    family: str,
    runs: Sequence[FoldRun],
    subjects: Sequence[int],
    thresholds_mm: Sequence[float],
    reference_scale_to_m: float,
    runtime_loader: RuntimeLoader | None = None,
    inference_runner: Callable[..., Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[SessionMetrics, ...]:
    """Fuse and score every session of ``subjects`` with its out-of-fold checkpoint.

    Session metric rows are written per subject under
    ``<output_root>/evaluation/session_metrics`` in the same JSON layout as the
    zero-shot benchmark so the two can be merged.
    """
    output = Path(output_root).resolve()
    config = {
        "paths": {"output_root": str(output)},
        "rotation_aware": {"config": str(Path(rotation_config).resolve())},
    }
    method_root = output / "fusion" / "methods"
    metrics_root = output / "evaluation" / "session_metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[SessionMetrics] = []
    for subject in sorted({int(value) for value in subjects}):
        run = fold_run_for_subject(runs, subject)
        rows: list[SessionMetrics] = []
        for session in load_manifest_sessions(benchmark_root, subject):
            pair = load_manifest_pair(benchmark_root, session)
            prediction = fuse_rotation_aware_trained(
                pair,
                run.checkpoint,
                run.run_id,
                config,
                family=family,
                runtime_loader=runtime_loader,
                inference_runner=inference_runner,
            )
            save_method_prediction(prediction, method_root)
            reference = load_session_reference(
                session,
                reference_scale_to_m=reference_scale_to_m,
            )
            rows.append(evaluate_session(prediction, reference, thresholds_mm))
            if progress is not None:
                progress(f"subject={subject:02d} session={session.session_id} run={run.run_id}")
        payload = {"subject_id": subject, "rows": [asdict(row) for row in rows]}
        target = metrics_root / f"subject_{subject:02d}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
        all_rows.extend(rows)
    return tuple(all_rows)


def load_session_metric_rows(
    metrics_root: Path,
    subjects: Sequence[int],
    *,
    methods: Sequence[str] | None = None,
) -> tuple[SessionMetrics, ...]:
    """Read session metric JSON files written by either benchmark path."""
    wanted = set(methods) if methods else None
    rows: list[SessionMetrics] = []
    for subject in sorted({int(value) for value in subjects}):
        path = Path(metrics_root) / f"subject_{subject:02d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for value in payload["rows"]:
            if wanted is not None and value["method"] not in wanted:
                continue
            rows.append(
                SessionMetrics(
                    **{
                        **dict(value),
                        "pck": {
                            int(float(key)): float(item)
                            for key, item in value["pck"].items()
                        },
                        "per_joint_mpjpe_mm": tuple(value["per_joint_mpjpe_mm"]),
                    }
                )
            )
    return tuple(rows)


def evaluate_deterministic_methods(
    *,
    benchmark_root: Path,
    output_root: Path,
    methods: Sequence[str],
    subjects: Sequence[int],
    thresholds_mm: Sequence[float],
    reference_scale_to_m: float,
    progress: Callable[[str], None] | None = None,
) -> tuple[SessionMetrics, ...]:
    """Fuse and score additional deterministic methods on cached SAM3D pairs.

    Mirrors ``evaluate_trained_family`` so new label-free baselines can be
    added to the FreeMan tables without re-running SAM3D inference or the
    already-published methods.
    """
    output = Path(output_root).resolve()
    method_root = output / "fusion" / "methods"
    metrics_root = output / "evaluation" / "session_metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[SessionMetrics] = []
    for subject in sorted({int(value) for value in subjects}):
        rows: list[SessionMetrics] = []
        for session in load_manifest_sessions(benchmark_root, subject):
            pair = load_manifest_pair(benchmark_root, session)
            reference = load_session_reference(
                session,
                reference_scale_to_m=reference_scale_to_m,
            )
            for prediction in fuse_deterministic(pair, methods=tuple(methods)):
                save_method_prediction(prediction, method_root)
                rows.append(evaluate_session(prediction, reference, thresholds_mm))
            if progress is not None:
                progress(f"subject={subject:02d} session={session.session_id}")
        payload = {"subject_id": subject, "rows": [asdict(row) for row in rows]}
        target = metrics_root / f"subject_{subject:02d}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
        all_rows.extend(rows)
    return tuple(all_rows)


def evaluate_zero_shot_checkpoint(
    *,
    benchmark_root: Path,
    output_root: Path,
    rotation_config: Path,
    run_id: str,
    checkpoint: Path,
    subjects: Sequence[int],
    thresholds_mm: Sequence[float],
    reference_scale_to_m: float,
    progress: Callable[[str], None] | None = None,
) -> tuple[SessionMetrics, ...]:
    """Score one additional private-data checkpoint zero-shot on cached pairs.

    Uses the same ``fuse_rotation_aware`` adapter (including its FreeMan
    training-provenance guard) as the published zero-shot rows, so a new
    learned baseline such as B1 lands in the FreeMan tables without re-running
    the whole benchmark.
    """
    output = Path(output_root).resolve()
    config = {
        "paths": {"output_root": str(output)},
        "rotation_aware": {"config": str(Path(rotation_config).resolve())},
    }
    method_root = output / "fusion" / "methods"
    metrics_root = output / "evaluation" / "session_metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[SessionMetrics] = []
    for subject in sorted({int(value) for value in subjects}):
        rows: list[SessionMetrics] = []
        for session in load_manifest_sessions(benchmark_root, subject):
            pair = load_manifest_pair(benchmark_root, session)
            prediction = fuse_rotation_aware(pair, Path(checkpoint), run_id, config)
            save_method_prediction(prediction, method_root)
            reference = load_session_reference(
                session,
                reference_scale_to_m=reference_scale_to_m,
            )
            rows.append(evaluate_session(prediction, reference, thresholds_mm))
            if progress is not None:
                progress(f"subject={subject:02d} session={session.session_id} run={run_id}")
        target = metrics_root / f"subject_{subject:02d}.json"
        existing: list[dict[str, Any]] = []
        if target.exists():
            existing = [
                row
                for row in json.loads(target.read_text(encoding="utf-8"))["rows"]
                if row["method"] != f"rotation_aware:{run_id}"
            ]
        payload = {"subject_id": subject, "rows": existing + [asdict(row) for row in rows]}
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
        all_rows.extend(rows)
    return tuple(all_rows)
