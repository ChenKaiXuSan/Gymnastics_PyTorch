"""Staged command-line interface for the FreeMan public benchmark.

``DefaultStageOperations`` is the production implementation of every stage;
the orchestration lives in :mod:`.runner` and the per-subject helpers in
:mod:`.stages`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from copy import deepcopy
from dataclasses import asdict
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any

import pandas as pd

from .download import (
    _shared_tree_valid,
    cleanup_subject_workspace,
    download_release,
    extract_shared_annotations,
    extract_subject,
    load_config,
    run_preflight,
    validate_downloads,
)
from .evaluation import EvaluationTables
from .report import (
    ReportContext,
    write_report,
)
from .schema import PreflightReport
from fusion.benchmarks.freeman.runner import (
    StageOperations,
    _run_parallel_subjects,
    run_subjects,
)
from fusion.benchmarks.freeman.stages import (
    _atomic_json,
    _cached_metrics,
    _evaluate_subject,
    _existing_inference_artifacts,
    _file_sha256,
    _fuse_pairs,
    _inference_artifacts,
    _load_fused_subject,
    _load_state,
    _metric_path,
    _pair_path,
    _pose_pairs,
    _rotation_checkpoint,
    _select_pairs,
    _sha256_json,
    _subject_sessions,
    _tables_with_failures,
    _write_session_manifest,
    _write_subject_metrics,
)


DEFAULT_CONFIG = Path("src/configs/benchmarks/freeman.yaml")


_STAGES = ("inspect", "download", "infer", "fuse", "evaluate", "report", "run")


_FORCE_STAGES = ("inspect", "infer", "fuse", "evaluate", "report")


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
        devices: Sequence[int] | None = None,
    ) -> Any:
        state_path = self._state_path(config)

        def publish(mutate: Callable[[dict], None]) -> dict:
            # Fresh read-modify-write for every publication so concurrent
            # per-subject requests never resurrect a stale snapshot.
            state = _load_state(state_path)
            mutate(state)
            _atomic_json(state_path, state)
            return state

        def set_stage(name: str, value: Mapping[str, Any]) -> None:
            publish(lambda s: s["stages"].__setitem__(name, dict(value)))

        def startup(state: dict) -> None:
            state["force_stage"] = force_stage
            state["frame_stride"] = config["dataset"]["frame_stride"]
            if force_stage in {"infer", "fuse", "evaluate"}:
                for subject in config["dataset"]["subjects"]:
                    state["subjects"].pop(str(int(subject)), None)

        if force_stage is not None:
            reset_forced_stage(config, force_stage)
        state = publish(startup)
        shared_root = Path(config["paths"]["work_root"]) / "shared"
        prepared = (
            not dry_run
            and force_stage != "inspect"
            and all(
                state["stages"].get(stage, {}).get("status") == "complete"
                for stage in ("inspect", "download", "shared_annotations")
            )
            and (shared_root / "extraction_manifest.json").is_file()
            and _shared_tree_valid(shared_root)
        )
        if not prepared:
            set_stage("inspect", {"status": "running"})
            try:
                report = self.inspect(config, dry_run=dry_run)
            except Exception as error:
                set_stage(
                    "inspect",
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )
                raise
            set_stage("inspect", {"status": "complete"})
            if dry_run:
                return report
            set_stage("download", {"status": "running"})
            if report.required_bytes:
                self.download(config)
            else:
                validate_downloads(report.entries, report.archive_root)
            set_stage("download", {"status": "complete"})
            set_stage("shared_annotations", {"status": "running"})
            extract_shared_annotations(
                report.entries,
                report.archive_root,
                Path(config["paths"]["work_root"]),
            )
            set_stage("shared_annotations", {"status": "complete"})
        if devices:
            _run_parallel_subjects(
                config,
                state_path,
                devices,
                keep_workspace,
            )
        else:
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
        set_stage("report", {"status": "running"})
        outputs = self.report(config)
        set_stage(
            "report",
            {
                "status": "complete",
                "results_json_sha256": _file_sha256(outputs.results_json),
                "markdown_sha256": _file_sha256(outputs.markdown),
            },
        )
        return outputs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fusion benchmark-freeman",
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
            child.add_argument("--devices", type=int, nargs="+")
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
    devices = (
        tuple(int(value) for value in args.devices)
        if getattr(args, "devices", None)
        else None
    )
    if devices is not None and (
        len(set(devices)) != len(devices) or any(value < 0 for value in devices)
    ):
        parser.error("--devices values must be unique non-negative integers")
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
            devices=devices,
        )
    else:
        getattr(stages, args.stage)(config)
    return 0

