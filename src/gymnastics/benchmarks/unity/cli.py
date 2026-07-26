"""Staged command-line interface for the Unity external benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np
import yaml

from .dataset import group_evaluation_sequences, load_unity_benchmark
from .evaluation import (
    angular_residual_deg,
    build_reference_sequence,
    evaluate_method_sequence,
    summarize_results,
    to_evaluation_sequence,
    trunk_rotation_deg,
)
from .fusion import run_deterministic_fusion, run_rotation_aware_fusion
from .geometry import run_oracle_triangulation, run_sam3d_triangulation
from .mapping import EVALUATION_JOINT_NAMES, UNITY_JOINT_INDICES
from .report import write_report
from .sam3d import load_sam3d_camera_cache, run_sam3d_inference
from .schema import MethodSequence, UnityBenchmark
from gymnastics.common.skeletons.mhr70 import mhr_names
from gymnastics.fusion.deterministic.experiment_matrix import ALL_METHODS


DEFAULT_CONFIG = Path("configs/benchmarks/unity.yaml")


def _load_config(path: Path) -> Mapping[str, object]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Unity benchmark config must be a mapping")
    return payload


def _paths(config: Mapping[str, object]) -> tuple[Path, Path]:
    raw = config.get("paths")
    if not isinstance(raw, Mapping):
        raise ValueError("Unity benchmark config requires paths")
    return Path(str(raw["dataset_root"])), Path(str(raw["output_root"]))


def _path_value(config: Mapping[str, object], name: str) -> Path:
    raw = config.get("paths")
    if not isinstance(raw, Mapping) or name not in raw:
        raise ValueError(f"Unity benchmark config requires paths.{name}")
    return Path(str(raw[name]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gymnastics benchmark unity",
        description="Run the Unity external pose and fusion benchmark",
    )
    stages = parser.add_subparsers(dest="stage")
    for name, help_text in (
        ("inspect", "inspect dataset or verify generated artifacts"),
        ("infer", "run cached SAM3D inference for one camera"),
        ("triangulate", "run oracle and SAM3D-2D triangulation"),
        ("fuse", "run deterministic and zero-shot learned fusion"),
        ("evaluate", "evaluate every available method against Unity 3D"),
        ("report", "regenerate the benchmark report"),
        ("run", "run all post-inference stages"),
    ):
        child = stages.add_parser(name, help=help_text)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if name == "inspect":
            child.add_argument("--verify-results", action="store_true")
        elif name == "infer":
            child.add_argument("--camera", choices=("cam0", "cam1"), required=True)
            child.add_argument("--device", default="cuda:0")
            child.add_argument("--sample-id", type=int, action="append")
            child.add_argument("--force", action="store_true")
        elif name == "triangulate":
            child.add_argument("--oracle-only", action="store_true")
        elif name == "fuse":
            child.add_argument("--method", action="append")
            child.add_argument("--ablation", action="append")
            child.add_argument("--device", default="cpu")
    return parser


def _inspect(args: argparse.Namespace, config: Mapping[str, object]) -> int:
    dataset_root, output_root = _paths(config)
    benchmark = load_unity_benchmark(dataset_root)
    groups = group_evaluation_sequences(benchmark)
    print(f"samples: {len(benchmark.frames)}")
    print(f"images: {len(benchmark.frames) * len(benchmark.cameras)}")
    print(f"joints: {len(benchmark.joint_names)}")
    print(f"evaluation_sequences: {len(groups)}")
    for sequence_id, frames in groups.items():
        print(f"  {sequence_id}: {len(frames)}")
    if args.verify_results:
        report = output_root / "report/unity_benchmark_report.md"
        results = output_root / "report/results.json"
        if not report.is_file() or not results.is_file():
            raise FileNotFoundError(
                f"missing benchmark report artifacts below {output_root}"
            )
        print(f"report: {report}")
        print(f"results: {results}")
    return 0


def _infer(args: argparse.Namespace, config: Mapping[str, object]) -> int:
    dataset_root, output_root = _paths(config)
    benchmark = load_unity_benchmark(dataset_root)
    inference = config.get("inference", {})
    if not isinstance(inference, Mapping):
        raise ValueError("Unity benchmark config inference must be a mapping")
    fallback_bbox = inference.get("fallback_bbox_xyxy")
    summary = run_sam3d_inference(
        benchmark,
        args.camera,
        output_root / "sam3d",
        _path_value(config, "sam3d_config"),
        args.device,
        force=args.force,
        sample_ids=args.sample_id,
        fallback_bbox_xyxy=fallback_bbox,
    )
    print(
        f"{summary.camera_id}: expected={summary.expected} "
        f"completed={summary.completed} reused={summary.reused} "
        f"failed={len(summary.failed)}"
    )
    return 0 if summary.completed + len(summary.failed) == summary.expected else 1


def _triangulate(args: argparse.Namespace, config: Mapping[str, object]) -> int:
    dataset_root, output_root = _paths(config)
    benchmark = load_unity_benchmark(dataset_root)
    oracle = run_oracle_triangulation(
        benchmark, output_root / "triangulation"
    )
    raw_max = max(
        float(sequence.metadata["raw_max_error_m"]) for sequence in oracle
    )
    print(
        f"triangulation_oracle2d: sequences={len(oracle)} "
        f"raw_max_error_mm={raw_max * 1000.0:.6f}"
    )
    if not args.oracle_only:
        sam3d = run_sam3d_triangulation(
            benchmark,
            output_root / "sam3d",
            output_root / "triangulation",
        )
        print(f"triangulation_sam3d2d: sequences={len(sam3d)}")
    return 0


def _load_method_sequence(path: Path) -> MethodSequence:
    with np.load(path, allow_pickle=False) as data:
        metadata = (
            json.loads(str(data["metadata"].item()))
            if "metadata" in data.files
            else {}
        )
        return MethodSequence(
            method=str(data["method"].item()),
            sequence_id=str(data["sequence_id"].item()),
            sample_ids=np.asarray(data["sample_ids"], dtype=np.int64),
            points=np.asarray(data["points"], dtype=np.float32),
            valid=np.asarray(data["valid"], dtype=bool),
            joint_names=tuple(str(value) for value in data["joint_names"].tolist()),
            metadata=metadata,
        )


def _checkpoints(
    config: Mapping[str, object], wanted: Sequence[str] | None
) -> dict[str, Path]:
    raw = config.get("checkpoints", {})
    if not isinstance(raw, Mapping):
        raise ValueError("Unity benchmark checkpoints must be a mapping")
    selected = tuple(wanted) if wanted else tuple(str(key) for key in raw)
    missing = [name for name in selected if name not in raw]
    if missing:
        raise ValueError(f"unknown rotation-aware ablations: {missing}")
    return {name: Path(str(raw[name])) for name in selected}


def _data_fps(config: Mapping[str, object]) -> float:
    raw = config.get("data", {})
    if not isinstance(raw, Mapping):
        raise ValueError("Unity benchmark data config must be a mapping")
    return float(raw.get("fps", 60.0))


def _fuse(args: argparse.Namespace, config: Mapping[str, object]) -> int:
    dataset_root, output_root = _paths(config)
    benchmark = load_unity_benchmark(dataset_root)
    methods = tuple(args.method) if args.method else ALL_METHODS
    deterministic = run_deterministic_fusion(
        benchmark,
        output_root / "sam3d",
        output_root / "fusion",
        methods=methods,
    )
    checkpoints = _checkpoints(config, args.ablation)
    learned = (
        run_rotation_aware_fusion(
            benchmark,
            output_root / "sam3d",
            output_root / "fusion",
            checkpoints,
            skeleton_path=_path_value(config, "skeleton"),
            fps=_data_fps(config),
            device=args.device,
        )
        if checkpoints
        else ()
    )
    print(
        f"deterministic_sequences={len(deterministic)} "
        f"rotation_aware_sequences={len(learned)}"
    )
    return 0


def _single_view_sequences(
    benchmark: UnityBenchmark, output_root: Path
) -> tuple[MethodSequence, ...]:
    outputs: list[MethodSequence] = []
    for sequence_id, frames in group_evaluation_sequences(benchmark).items():
        sample_ids = tuple(frame.sample_id for frame in frames)
        for camera_id in ("cam0", "cam1"):
            cached = load_sam3d_camera_cache(
                output_root / "sam3d", camera_id, sample_ids
            )
            outputs.append(
                MethodSequence(
                    method=camera_id,
                    sequence_id=sequence_id,
                    sample_ids=cached.sample_ids,
                    points=cached.points_3d,
                    valid=cached.valid_3d,
                    joint_names=tuple(mhr_names),
                    metadata={
                        "ranking_group": "valid",
                        "source": "sam3d_single_view",
                        "camera": camera_id,
                    },
                )
            )
    return tuple(outputs)


def _disk_candidates(
    benchmark: UnityBenchmark,
    output_root: Path,
    checkpoints: Mapping[str, Path],
) -> tuple[tuple[MethodSequence, ...], tuple[Mapping[str, object], ...]]:
    candidates: list[MethodSequence] = []
    failures: list[Mapping[str, object]] = []
    groups = group_evaluation_sequences(benchmark)
    roots: dict[str, Path] = {
        "triangulation_sam3d2d": output_root / "triangulation/sam3d2d",
        "triangulation_oracle2d": output_root / "triangulation/oracle2d",
        **{
            method: output_root / "fusion/deterministic" / method
            for method in ALL_METHODS
        },
        **{
            ablation: output_root / "fusion/rotation_aware" / ablation
            for ablation in checkpoints
        },
    }
    for method, root in roots.items():
        for sequence_id in groups:
            path = root / f"{sequence_id}.npz"
            if not path.is_file():
                failures.append(
                    {
                        "stage": "evaluate",
                        "method": method,
                        "sequence_id": sequence_id,
                        "reason": f"missing_artifact:{path}",
                    }
                )
                continue
            sequence = _load_method_sequence(path)
            if sequence.method != method or sequence.sequence_id != sequence_id:
                raise ValueError(f"method artifact identity mismatch: {path}")
            candidates.append(sequence)
    return tuple(candidates), tuple(failures)


def _angle_offset(
    references: Mapping[str, MethodSequence],
    groups,
) -> float:
    residuals: list[np.ndarray] = []
    for sequence_id, reference in references.items():
        angles, valid = trunk_rotation_deg(reference.points, reference.valid)
        actual = np.asarray(
            [frame.actual_angle_deg for frame in groups[sequence_id]],
            dtype=np.float32,
        )
        neutral = valid & (np.abs(actual) <= 1.0)
        if neutral.any():
            residuals.append(angular_residual_deg(angles[neutral], actual[neutral]))
    if not residuals:
        return 0.0
    radians = np.deg2rad(np.concatenate(residuals))
    return float(np.rad2deg(np.angle(np.mean(np.exp(1j * radians)))))


def _visibility(frames) -> dict[str, np.ndarray]:
    indices = [UNITY_JOINT_INDICES[name] for name in EVALUATION_JOINT_NAMES]
    return {
        camera_id: np.stack(
            [frame.visible[camera_id][indices] for frame in frames]
        )
        for camera_id in ("cam0", "cam1")
    }


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _evaluate(config: Mapping[str, object]) -> int:
    dataset_root, output_root = _paths(config)
    benchmark = load_unity_benchmark(dataset_root)
    groups = group_evaluation_sequences(benchmark)
    references = {
        sequence_id: build_reference_sequence(sequence_id, frames)
        for sequence_id, frames in groups.items()
    }
    candidates = list(_single_view_sequences(benchmark, output_root))
    checkpoints = _checkpoints(config, None)
    disk, failures = _disk_candidates(benchmark, output_root, checkpoints)
    candidates.extend(disk)
    offset = _angle_offset(references, groups)
    results = []
    for raw_candidate in candidates:
        candidate = to_evaluation_sequence(raw_candidate)
        frames = groups[candidate.sequence_id]
        results.append(
            evaluate_method_sequence(
                candidate,
                references[candidate.sequence_id],
                visibility=_visibility(frames),
                actual_angles_deg=np.asarray(
                    [frame.actual_angle_deg for frame in frames],
                    dtype=np.float32,
                ),
                angle_offset_deg=offset,
            )
        )
    provenance = {
        "git_commit": _git_commit(),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "joint_subset": list(EVALUATION_JOINT_NAMES),
        "alignment": "one_sim3_per_sequence",
        "angle_offset_deg": offset,
        "expected_samples": len(benchmark.frames),
        "expected_sequences": len(groups),
    }
    bundle = summarize_results(
        results, failures=failures, provenance=provenance
    )
    report_path = write_report(
        bundle, output_root, provenance=provenance
    )
    print(f"evaluated_method_sequences={len(results)}")
    print(f"explicit_failures={len(failures)}")
    print(f"report={report_path}")
    if bundle.valid_ranking:
        best = bundle.valid_ranking[0]
        print(
            f"best_valid={best['method']} "
            f"mpjpe_mm={float(best['mpjpe_mm']):.3f}"
        )
    return 0


def _run(config: Mapping[str, object]) -> int:
    _triangulate(argparse.Namespace(oracle_only=False), config)
    _fuse(
        argparse.Namespace(method=None, ablation=None, device="cpu"),
        config,
    )
    return _evaluate(config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage is None:
        parser.print_help()
        return 0
    config = _load_config(args.config)
    if args.stage == "inspect":
        return _inspect(args, config)
    if args.stage == "infer":
        return _infer(args, config)
    if args.stage == "triangulate":
        return _triangulate(args, config)
    if args.stage == "fuse":
        return _fuse(args, config)
    if args.stage in {"evaluate", "report"}:
        return _evaluate(config)
    if args.stage == "run":
        return _run(config)
    raise NotImplementedError(f"Unity benchmark stage not implemented: {args.stage}")
