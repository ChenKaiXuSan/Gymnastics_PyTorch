"""Staged CLI and resumable one-subject-at-a-time FreeMan orchestration."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .dataset import load_session_reference, load_subject_sessions
from .download import (
    cleanup_subject_workspace,
    download_release,
    extract_shared_annotations,
    extract_subject,
    load_config,
    run_preflight,
    validate_downloads,
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
from .report import ReportContext, write_report
from .sam3d import infer_subject_sessions, load_inference
from .schema import (
    FreeManSession,
    MethodPrediction,
    PosePairInput,
    PreflightReport,
    SelectedPair,
    ViewPrediction,
)


DEFAULT_CONFIG = Path("configs/benchmarks/freeman.yaml")
_STAGES = ("inspect", "download", "infer", "fuse", "evaluate", "report", "run")
_FORCE_STAGES = ("inspect", "infer", "fuse", "evaluate", "report")


class StageOperations:
    """Replaceable stage boundary used by the CLI and its tests."""

    def inspect(self, config: Mapping[str, Any], *, dry_run: bool = False) -> Any:
        raise NotImplementedError

    def download(self, config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def infer(self, config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def fuse(self, config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def evaluate(self, config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def report(self, config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def run(
        self,
        config: Mapping[str, Any],
        *,
        force_stage: str | None = None,
        keep_workspace: bool = False,
        dry_run: bool = False,
    ) -> Any:
        raise NotImplementedError


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


def run_subjects(
    subjects: Sequence[int],
    *,
    state_path: Path,
    process: Callable[[int], Any],
    cleanup: Callable[[int], Any],
    keep_workspace: bool,
) -> None:
    """Run subjects in numeric order, publishing state after every transition."""
    state_file = Path(state_path)
    state = _load_state(state_file)
    subject_state = state["subjects"]
    for subject in sorted({int(value) for value in subjects}):
        if subject < 1 or subject > 40:
            raise ValueError("FreeMan subjects must be within 1..40")
        key = str(subject)
        if subject_state.get(key, {}).get("status") == "complete":
            continue
        subject_state[key] = {"status": "running"}
        _atomic_json(state_file, state)
        try:
            artifacts = process(subject)
            if not keep_workspace:
                cleanup(subject)
        except Exception as error:
            subject_state[key] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            _atomic_json(state_file, state)
            raise
        subject_state[key] = {"status": "complete"}
        if isinstance(artifacts, Mapping):
            subject_state[key]["artifacts"] = dict(artifacts)
        _atomic_json(state_file, state)


def _remove_scoped_tree(root: Path, target: Path) -> None:
    resolved_root = root.resolve()
    resolved_target = target.resolve(strict=False)
    if (
        resolved_target == resolved_root
        or resolved_root not in resolved_target.parents
    ):
        raise ValueError(f"refusing to remove unscoped benchmark path: {target}")
    if resolved_target.exists():
        shutil.rmtree(resolved_target)


def _remove_aggregate_outputs(output_root: Path) -> None:
    evaluation = output_root / "evaluation"
    if evaluation.is_dir():
        for path in evaluation.iterdir():
            if path.name != "session_metrics":
                if path.is_dir():
                    _remove_scoped_tree(output_root, path)
                else:
                    path.unlink()
    _remove_scoped_tree(output_root, output_root / "report")


def reset_forced_stage(
    config: Mapping[str, Any],
    stage: str,
) -> None:
    """Invalidate exactly the selected stage and its downstream artifacts."""
    if stage not in _FORCE_STAGES:
        raise ValueError(f"force stage must be one of {_FORCE_STAGES}")
    if stage == "inspect":
        return
    output = Path(config["paths"]["output_root"]).resolve()
    subjects = sorted({int(value) for value in config["dataset"]["subjects"]})
    if stage == "infer":
        for subject in subjects:
            _remove_scoped_tree(output, output / "sam3d" / f"subject_{subject:02d}")
    if stage in {"infer", "fuse"}:
        methods_root = output / "fusion" / "methods"
        if methods_root.is_dir():
            for method_root in methods_root.iterdir():
                if method_root.is_dir():
                    for subject in subjects:
                        _remove_scoped_tree(
                            output,
                            method_root / f"subject_{subject:02d}",
                        )
    if stage in {"infer", "fuse", "evaluate"}:
        for subject in subjects:
            _metric_path(config, subject).unlink(missing_ok=True)
    _remove_aggregate_outputs(output)


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
        from gymnastics.common.paths import PROJECT_ROOT

        rotation_config_path = PROJECT_ROOT / rotation_config_path
    import yaml

    raw = yaml.safe_load(rotation_config_path.read_text(encoding="utf-8"))
    output = Path(raw["paths"]["output_root"])
    if not output.is_absolute():
        from gymnastics.common.paths import PROJECT_ROOT

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


class DefaultStageOperations(StageOperations):
    """Production implementation of every FreeMan stage."""

    def __init__(self) -> None:
        self._preflight: PreflightReport | None = None

    def _state_path(self, config: Mapping[str, Any]) -> Path:
        return Path(config["paths"]["output_root"]) / "run_state.json"

    def inspect(
        self,
        config: Mapping[str, Any],
        *,
        dry_run: bool = False,
    ) -> PreflightReport:
        del dry_run
        report = run_preflight(config)
        self._preflight = report
        checkpoint_status = {
            str(run_id): str(_rotation_checkpoint(config, str(run_id)))
            for run_id in config["rotation_aware"].get("run_ids", ())
        }
        _atomic_json(
            Path(config["paths"]["output_root"])
            / "inspect"
            / "preflight.json",
            {
                "repository": {
                    "repo_id": report.repo_id,
                    "revision": report.revision,
                },
                "authenticated_user": report.authenticated_user,
                "required_bytes": report.required_bytes,
                "free_bytes": report.free_bytes,
                "reserve_bytes": report.reserve_bytes,
                "inventory_entries": len(report.entries),
                "inventory_sha256": _sha256_json(
                    [asdict(entry) for entry in report.entries]
                ),
                "rotation_aware_checkpoints": checkpoint_status,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return report

    def download(self, config: Mapping[str, Any]) -> Path:
        report = self._preflight or run_preflight(config)
        return download_release(config, report)

    def infer(self, config: Mapping[str, Any]) -> None:
        for subject in config["dataset"]["subjects"]:
            sessions = _subject_sessions(config, int(subject))
            pairs = _select_pairs(sessions, config)
            _write_session_manifest(config, int(subject), sessions, pairs)
            _inference_artifacts(sessions, pairs, config)

    def fuse(self, config: Mapping[str, Any]) -> None:
        for subject in config["dataset"]["subjects"]:
            sessions = _subject_sessions(config, int(subject))
            pairs = _select_pairs(sessions, config)
            artifacts = _existing_inference_artifacts(sessions, pairs, config)
            _fuse_pairs(_pose_pairs(sessions, pairs, artifacts), config)

    def evaluate(self, config: Mapping[str, Any]) -> EvaluationTables:
        for subject in config["dataset"]["subjects"]:
            subject_id = int(subject)
            sessions = _subject_sessions(config, subject_id)
            predictions = _load_fused_subject(config, subject_id)
            rows = _evaluate_subject(sessions, predictions, config)
            _write_subject_metrics(config, subject_id, rows)
        return _tables_with_failures(
            _cached_metrics(config),
            self._state_path(config),
        )

    def _camera_pairs(self, config: Mapping[str, Any]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        manifest_root = Path(config["paths"]["output_root"]) / "manifests"
        for path in sorted(manifest_root.glob("subject_*_sessions.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for session in payload["sessions"]:
                rows.append(
                    {
                        "subject_id": payload["subject_id"],
                        "session_id": session["session_id"],
                        "fps": session["fps"],
                        **session["pair"],
                    }
                )
        return pd.DataFrame(rows)

    def _context(self, config: Mapping[str, Any]) -> ReportContext:
        metric_root = (
            Path(config["paths"]["output_root"]) / "evaluation" / "session_metrics"
        )
        processed = [
            int(path.stem.split("_")[-1])
            for path in sorted(metric_root.glob("subject_*.json"))
        ]
        rows = _cached_metrics(config)
        fps_counts: dict[str, int] = {}
        sessions = {(row.subject_id, row.session_id, row.fps) for row in rows}
        for _, _, fps in sessions:
            key = str(fps)
            fps_counts[key] = fps_counts.get(key, 0) + 1
        download_state = Path(config["paths"]["manifest_root"]) / "download_state.json"
        download_manifest = (
            json.loads(download_state.read_text(encoding="utf-8"))
            if download_state.is_file()
            else {}
        )
        inventory = download_manifest.get("files", ())
        download_manifest["inventory_sha256"] = _sha256_json(inventory)
        checkpoint_metadata = {
            "sam3d": {"checkpoint_id": config["sam3d"]["checkpoint_id"]},
            "rotation_aware": {
                str(run_id): {
                    "checkpoint": str(_rotation_checkpoint(config, str(run_id)))
                }
                for run_id in config["rotation_aware"].get("run_ids", ())
            },
        }
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return ReportContext(
            resolved_config=config,
            dataset_manifest={
                "processed_subjects": processed,
                "processed_sessions": len(sessions),
                "fps_session_counts": fps_counts,
            },
            download_manifest=download_manifest,
            camera_pairs=self._camera_pairs(config),
            checkpoint_metadata=checkpoint_metadata,
            code_commit=commit,
        )

    def report(self, config: Mapping[str, Any]) -> Any:
        tables = _tables_with_failures(
            _cached_metrics(config),
            self._state_path(config),
        )
        return write_report(
            tables,
            self._context(config),
            Path(config["paths"]["output_root"]),
        )

    def _process_subject(
        self,
        config: Mapping[str, Any],
        subject: int,
    ) -> Mapping[str, str]:
        extract_subject(
            subject,
            Path(config["paths"]["archive_root"]),
            Path(config["paths"]["work_root"]),
        )
        sessions = _subject_sessions(config, subject)
        if not sessions:
            raise RuntimeError(f"subject {subject:02d} has no requested FreeMan sessions")
        pairs = _select_pairs(sessions, config)
        _write_session_manifest(config, subject, sessions, pairs)
        artifacts = _inference_artifacts(sessions, pairs, config)
        fused = _fuse_pairs(_pose_pairs(sessions, pairs, artifacts), config)
        rows = _evaluate_subject(sessions, fused, config)
        _write_subject_metrics(config, subject, rows)
        manifest_path = _pair_path(config, subject)
        metric_path = _metric_path(config, subject)
        return {
            "session_manifest_sha256": _file_sha256(manifest_path),
            "session_metrics_sha256": _file_sha256(metric_path),
        }

    def run(
        self,
        config: Mapping[str, Any],
        *,
        force_stage: str | None = None,
        keep_workspace: bool = False,
        dry_run: bool = False,
    ) -> Any:
        state_path = self._state_path(config)
        state = _load_state(state_path)
        state["force_stage"] = force_stage
        state["frame_stride"] = config["dataset"]["frame_stride"]
        if force_stage is not None:
            reset_forced_stage(config, force_stage)
            if force_stage in {"infer", "fuse", "evaluate"}:
                for subject in config["dataset"]["subjects"]:
                    state["subjects"].pop(str(int(subject)), None)
        _atomic_json(state_path, state)
        state["stages"]["inspect"] = {"status": "running"}
        _atomic_json(state_path, state)
        try:
            report = self.inspect(config, dry_run=dry_run)
        except Exception as error:
            state["stages"]["inspect"] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            _atomic_json(state_path, state)
            raise
        state["stages"]["inspect"] = {"status": "complete"}
        _atomic_json(state_path, state)
        if dry_run:
            return report
        state["stages"]["download"] = {"status": "running"}
        _atomic_json(state_path, state)
        if report.required_bytes:
            self.download(config)
        else:
            validate_downloads(report.entries, report.archive_root)
        state["stages"]["download"] = {"status": "complete"}
        _atomic_json(state_path, state)
        state["stages"]["shared_annotations"] = {"status": "running"}
        _atomic_json(state_path, state)
        extract_shared_annotations(
            report.entries,
            report.archive_root,
            Path(config["paths"]["work_root"]),
        )
        state["stages"]["shared_annotations"] = {"status": "complete"}
        _atomic_json(state_path, state)
        run_subjects(
            config["dataset"]["subjects"],
            state_path=state_path,
            process=lambda subject: self._process_subject(config, subject),
            cleanup=lambda subject: cleanup_subject_workspace(
                subject,
                Path(config["paths"]["work_root"])
                / f"subject_{subject:02d}",
                Path(config["paths"]["work_root"]),
            ),
            keep_workspace=keep_workspace,
        )
        state = _load_state(state_path)
        state["stages"]["report"] = {"status": "running"}
        _atomic_json(state_path, state)
        outputs = self.report(config)
        state["stages"]["report"] = {
            "status": "complete",
            "results_json_sha256": _file_sha256(outputs.results_json),
            "markdown_sha256": _file_sha256(outputs.markdown),
        }
        _atomic_json(state_path, state)
        return outputs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gymnastics benchmark freeman",
        description="Full-release FreeMan zero-shot multi-view benchmark",
    )
    commands = parser.add_subparsers(dest="stage")
    for stage in _STAGES:
        child = commands.add_parser(stage)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if stage != "download":
            child.add_argument("--subject", type=int, nargs="+")
            child.add_argument("--fps", type=int, choices=(30, 60), nargs="+")
            child.add_argument("--frame-stride", type=int)
        if stage == "inspect":
            child.add_argument("--dry-run", action="store_true")
        if stage == "run":
            child.add_argument("--force-stage", choices=_FORCE_STAGES)
            child.add_argument("--keep-workspace", action="store_true")
            child.add_argument("--dry-run", action="store_true")
    return parser


def _overrides(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    if getattr(args, "subject", None):
        subjects = sorted(set(int(value) for value in args.subject))
        if any(value < 1 or value > 40 for value in subjects):
            raise ValueError("--subject values must be within 1..40")
        resolved["dataset"]["subjects"] = subjects
    if getattr(args, "fps", None):
        resolved["dataset"]["fps_subsets"] = sorted(set(args.fps))
    if getattr(args, "frame_stride", None) is not None:
        if args.frame_stride < 1:
            raise ValueError("--frame-stride must be positive")
        resolved["dataset"]["frame_stride"] = args.frame_stride
    stride = int(resolved["dataset"]["frame_stride"])
    resolved.setdefault("evaluation", {})
    resolved["evaluation"]["headline_eligible"] = stride == 1
    if stride != 1:
        resolved["evaluation"]["diagnostic_reason"] = "frame_stride_not_one"
    else:
        resolved["evaluation"].pop("diagnostic_reason", None)
    return resolved


def main(
    argv: Sequence[str] | None = None,
    *,
    operations: StageOperations | None = None,
) -> int:
    parser = make_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage is None:
        parser.print_help()
        return 0
    config = _overrides(load_config(args.config), args)
    stages = operations or DefaultStageOperations()
    if args.stage == "download":
        stages.inspect(config)
        stages.download(config)
    elif args.stage == "inspect":
        stages.inspect(config, dry_run=args.dry_run)
    elif args.stage == "run":
        stages.run(
            config,
            force_stage=args.force_stage,
            keep_workspace=args.keep_workspace,
            dry_run=args.dry_run,
        )
    else:
        getattr(stages, args.stage)(config)
    return 0
