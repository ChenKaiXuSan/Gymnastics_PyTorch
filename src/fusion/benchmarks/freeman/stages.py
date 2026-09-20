"""Per-subject building blocks of the FreeMan stages: artefact paths, camera
pair selection, SAM3D inference artefacts, fusion, evaluation and cached
metrics."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import (
    load_session_reference,
    load_subject_sessions,
)
from .evaluation import (
    EvaluationTables,
    SessionMetrics,
    aggregate_metrics,
    evaluate_session,
)
from .fusion import (
    fuse_deterministic,
    fuse_rotation_aware,
    load_method_prediction,
    save_method_prediction,
)
from .pairing import select_camera_pair
from .sam3d import (
    infer_subject_sessions,
    load_inference,
)
from .schema import (
    FreeManSession,
    MethodPrediction,
    PosePairInput,
    SelectedPair,
    ViewPrediction,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"stages": {}, "subjects": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("FreeMan run state must be a mapping")
    value.setdefault("stages", {})
    value.setdefault("subjects", {})
    return value


def _pair_path(config: Mapping[str, Any], subject: int) -> Path:
    return (
        Path(config["paths"]["output_root"])
        / "manifests"
        / f"subject_{subject:02d}_sessions.json"
    )


def _metric_path(config: Mapping[str, Any], subject: int) -> Path:
    return (
        Path(config["paths"]["output_root"])
        / "evaluation"
        / "session_metrics"
        / f"subject_{subject:02d}.json"
    )


def _select_pairs(
    sessions: Sequence[FreeManSession],
    config: Mapping[str, Any],
) -> dict[str, SelectedPair]:
    pairing = config["pairing"]
    return {
        session.session_id: select_camera_pair(
            session,
            target_angle_deg=float(pairing["target_angle_deg"]),
            world_up=np.asarray(pairing["world_up_axis"], dtype=np.float64),
            minimum_axis_norm=float(pairing.get("minimum_axis_norm", 1e-8)),
        )
        for session in sessions
    }


def _write_session_manifest(
    config: Mapping[str, Any],
    subject: int,
    sessions: Sequence[FreeManSession],
    pairs: Mapping[str, SelectedPair],
) -> None:
    payload = {
        "subject_id": subject,
        "reference_scale_to_m": float(
            config["dataset"]["reference_scale_to_m"]
        ),
        "sessions": [
            {
                "session_id": session.session_id,
                "fps": session.fps,
                "split": session.split,
                "scenario": session.scenario,
                "action": session.action,
                "frames": len(session.frame_ids),
                "excluded_trailing_frames": dict(
                    session.excluded_trailing_frames
                ),
                "keypoints3d_path": str(session.keypoints3d_path),
                "pair": asdict(pairs[session.session_id]),
            }
            for session in sessions
        ],
    }
    _atomic_json(_pair_path(config, subject), payload)


def _subject_sessions(
    config: Mapping[str, Any],
    subject: int,
) -> tuple[FreeManSession, ...]:
    work_root = Path(config["paths"]["work_root"])
    subject_root = work_root / f"subject_{subject:02d}"
    shared_root = work_root / "shared"
    return load_subject_sessions(
        subject_root,
        shared_root,
        fps_values=config["dataset"]["fps_subsets"],
    )


def _pose_pairs(
    sessions: Sequence[FreeManSession],
    pairs: Mapping[str, SelectedPair],
    artifacts_by_identity: Mapping[tuple[str, str], Path],
) -> dict[str, PosePairInput]:
    result: dict[str, PosePairInput] = {}
    for session in sessions:
        pair = pairs[session.session_id]
        view_a = load_inference(
            artifacts_by_identity[(session.session_id, pair.view_a)]
        )
        view_b = load_inference(
            artifacts_by_identity[(session.session_id, pair.view_b)]
        )
        result[session.session_id] = PosePairInput(
            session_id=session.session_id,
            subject_id=session.subject_id,
            fps=float(session.fps),
            view_a=view_a,
            view_b=view_b,
        )
    return result


def _inference_artifacts(
    sessions: Sequence[FreeManSession],
    pairs: Mapping[str, SelectedPair],
    config: Mapping[str, Any],
) -> dict[tuple[str, str], Path]:
    artifacts = infer_subject_sessions(sessions, pairs, config)
    return {
        (artifact.session_id, artifact.view_id): artifact.path
        for artifact in artifacts
    }


def _existing_inference_artifacts(
    sessions: Sequence[FreeManSession],
    pairs: Mapping[str, SelectedPair],
    config: Mapping[str, Any],
) -> dict[tuple[str, str], Path]:
    root = Path(config["paths"]["output_root"]) / "sam3d"
    artifacts: dict[tuple[str, str], Path] = {}
    for session in sessions:
        pair = pairs[session.session_id]
        for view in (pair.view_a, pair.view_b):
            path = (
                root
                / f"subject_{session.subject_id:02d}"
                / session.session_id
                / view
                / "prediction.npz"
            )
            try:
                prediction = load_inference(path)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"fuse stage requires a valid SAM3D cache: {path}"
                ) from error
            if (
                prediction.session_id != session.session_id
                or prediction.subject_id != session.subject_id
                or prediction.view_id != view
            ):
                raise RuntimeError(f"SAM3D cache identity mismatch: {path}")
            artifacts[(session.session_id, view)] = path
    return artifacts


def _view_baseline(view: ViewPrediction, method: str) -> MethodPrediction:
    return MethodPrediction(
        method=method,
        session_id=view.session_id,
        subject_id=view.subject_id,
        fps=view.fps,
        points=view.points3d,
        valid=view.valid3d,
        frame_ids=view.frame_ids,
        metadata={
            "dataset": "FreeMan",
            "method": method,
            "classification": "VALID",
            "reference_3d_consumed": False,
            "source_view": view.view_id,
        },
    )


def _rotation_checkpoint(config: Mapping[str, Any], run_id: str) -> Path:
    rotation_config_path = Path(config["rotation_aware"]["config"])
    if not rotation_config_path.is_absolute():
        from common.paths import PROJECT_ROOT

        rotation_config_path = PROJECT_ROOT / rotation_config_path
    import yaml

    raw = yaml.safe_load(rotation_config_path.read_text(encoding="utf-8"))
    output = Path(raw["paths"]["output_root"])
    if not output.is_absolute():
        from common.paths import PROJECT_ROOT

        output = PROJECT_ROOT / output
    checkpoint = output / "runs" / run_id / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"required zero-shot rotation-aware checkpoint is missing: {checkpoint}"
        )
    return checkpoint.resolve()


def _apply_protocol_classification(
    prediction: MethodPrediction,
    config: Mapping[str, Any],
) -> MethodPrediction:
    if int(config["dataset"]["frame_stride"]) == 1:
        return prediction
    metadata = {
        **dict(prediction.metadata),
        "classification": "DIAGNOSTIC_FRAME_STRIDE",
        "excluded_from_ranking": True,
        "diagnostic_reason": "frame_stride_not_one",
    }
    return MethodPrediction(
        method=prediction.method,
        session_id=prediction.session_id,
        subject_id=prediction.subject_id,
        fps=prediction.fps,
        points=prediction.points,
        valid=prediction.valid,
        frame_ids=prediction.frame_ids,
        metadata=metadata,
    )


def _fuse_pairs(
    pose_pairs: Mapping[str, PosePairInput],
    config: Mapping[str, Any],
) -> dict[str, tuple[MethodPrediction, ...]]:
    output = Path(config["paths"]["output_root"]) / "fusion" / "methods"
    checkpoints = {
        str(run_id): _rotation_checkpoint(config, str(run_id))
        for run_id in config["rotation_aware"].get("run_ids", ())
    }
    all_predictions: dict[str, tuple[MethodPrediction, ...]] = {}
    for session_id in sorted(pose_pairs):
        pair = pose_pairs[session_id]
        predictions: list[MethodPrediction] = [
            _view_baseline(pair.view_a, "view_a"),
            _view_baseline(pair.view_b, "view_b"),
            *fuse_deterministic(pair),
        ]
        for run_id, checkpoint in checkpoints.items():
            predictions.append(
                fuse_rotation_aware(
                    pair,
                    checkpoint,
                    run_id,
                    config,
                )
            )
        predictions = [
            _apply_protocol_classification(prediction, config)
            for prediction in predictions
        ]
        for prediction in predictions:
            save_method_prediction(prediction, output)
        all_predictions[session_id] = tuple(predictions)
    return all_predictions


def _load_fused_subject(
    config: Mapping[str, Any],
    subject: int,
) -> dict[str, tuple[MethodPrediction, ...]]:
    root = Path(config["paths"]["output_root"]) / "fusion" / "methods"
    predictions: dict[str, list[MethodPrediction]] = {}
    for path in sorted(root.glob(f"*/subject_{subject:02d}/*/fused_sequence.npz")):
        loaded = load_method_prediction(path)
        predictions.setdefault(loaded.session_id, []).append(loaded)
    return {
        session: tuple(sorted(items, key=lambda item: item.method))
        for session, items in predictions.items()
    }


def _evaluate_subject(
    sessions: Sequence[FreeManSession],
    predictions: Mapping[str, Sequence[MethodPrediction]],
    config: Mapping[str, Any],
) -> tuple[SessionMetrics, ...]:
    thresholds = tuple(
        float(value) for value in config["evaluation"]["pck_thresholds_mm"]
    )
    scale = float(config["dataset"]["reference_scale_to_m"])
    rows: list[SessionMetrics] = []
    for session in sessions:
        reference = load_session_reference(
            session,
            reference_scale_to_m=scale,
        )
        session_predictions = predictions.get(session.session_id)
        if not session_predictions:
            raise RuntimeError(
                f"no fused predictions available for {session.session_id}"
            )
        rows.extend(
            evaluate_session(prediction, reference, thresholds)
            for prediction in session_predictions
        )
    return tuple(rows)


def _write_subject_metrics(
    config: Mapping[str, Any],
    subject: int,
    rows: Sequence[SessionMetrics],
) -> None:
    _atomic_json(
        _metric_path(config, subject),
        {
            "subject_id": subject,
            "rows": [asdict(row) for row in rows],
        },
    )


def _metric_from_json(value: Mapping[str, Any]) -> SessionMetrics:
    return SessionMetrics(
        **{
            **dict(value),
            "pck": {
                int(float(key)): float(item)
                for key, item in value["pck"].items()
            },
            "per_joint_mpjpe_mm": tuple(value["per_joint_mpjpe_mm"]),
        }
    )


def _cached_metrics(config: Mapping[str, Any]) -> tuple[SessionMetrics, ...]:
    root = Path(config["paths"]["output_root"]) / "evaluation" / "session_metrics"
    rows: list[SessionMetrics] = []
    for path in sorted(root.glob("subject_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(_metric_from_json(item) for item in payload["rows"])
    if not rows:
        raise RuntimeError("no cached FreeMan session metrics are available")
    return tuple(rows)


def _tables_with_failures(
    rows: Sequence[SessionMetrics],
    state_path: Path,
) -> EvaluationTables:
    tables = aggregate_metrics(rows)
    if not state_path.is_file():
        return tables
    state = _load_state(state_path)
    failures = [
        {
            "subject_id": int(subject),
            "session_id": None,
            "stage": "subject",
            "reason": details.get("error_message"),
        }
        for subject, details in state["subjects"].items()
        if details.get("status") == "failed"
    ]
    return EvaluationTables(
        by_session=tables.by_session,
        by_subject=tables.by_subject,
        by_method=tables.by_method,
        by_joint=tables.by_joint,
        by_split=tables.by_split,
        by_scenario=tables.by_scenario,
        paired_statistics=tables.paired_statistics,
        failures=pd.DataFrame(
            failures,
            columns=["subject_id", "session_id", "stage", "reason"],
        ),
    )


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

