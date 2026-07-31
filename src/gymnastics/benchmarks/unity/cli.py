"""Staged command-line interface for the Unity external benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np
import torch
import yaml

from .camera_guided_data import (
    CAMERA_GUIDED_ABLATIONS,
    build_camera_guided_sequences,
)
from .camera_guided_evaluation import (
    evaluate_camera_guided_runs,
    write_camera_guided_report,
)
from .camera_guided_training import (
    CameraGuidedRun,
    CameraGuidedTrainingConfig,
    _run_contract as _camera_run_contract,
    run_camera_guided_inference,
    train_camera_guided_run,
)
from .extrinsic_evaluation import (
    evaluate_extrinsic_runs,
    write_extrinsic_report,
)
from .extrinsic_training import (
    EXTRINSIC_METHODS,
    ExtrinsicTrainingConfig,
    _run_contract as _extrinsic_run_contract,
    build_extrinsic_sequences,
    run_extrinsic_inference,
    train_extrinsic_run,
    validate_extrinsic_run,
)
from .supervised import (
    UnityFineTuneConfig,
    _resolved_config,
    _run_contract,
    _sha256_file,
    discover_completed_runs,
    run_finetuned_inference,
    run_supervised_finetune,
    validate_completed_run,
)
from .supervised_data import (
    UNITY_SUPERVISED_FOLDS,
    UnitySupervisedSequence,
    build_supervised_sequence,
    build_supervised_sequences,
)
from .supervised_evaluation import (
    build_finetuned_bundle,
    evaluate_finetuned_runs,
    write_finetuned_report,
)
from .supervised_loss import UnitySupervisedLossConfig
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
from gymnastics.fusion.rotation_aware.corruptions import CorruptionConfig
from gymnastics.fusion.rotation_aware.losses import LossConfig


DEFAULT_CONFIG = Path("configs/benchmarks/unity.yaml")
DEFAULT_SUPERVISED_CONFIG = Path("configs/benchmarks/unity_supervised.yaml")


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
        ("finetune", "fine-tune one model with Unity native 3D"),
        ("finetune-matrix", "run the complete resumable 36-cell fine-tuning matrix"),
        ("evaluate-finetuned", "infer and evaluate all completed fine-tuned runs"),
        ("report-finetuned", "regenerate the fine-tuned report from saved inference"),
        ("train-extrinsic", "train one calibrated learned Unity baseline"),
        (
            "train-extrinsic-matrix",
            "run the complete 18-cell calibrated-learning matrix",
        ),
        (
            "evaluate-extrinsic",
            "infer and evaluate calibrated learned baselines",
        ),
        (
            "report-extrinsic",
            "regenerate calibrated learned baseline results",
        ),
        (
            "train-camera-feature",
            "train one fitted-camera self-supervised G-series cell",
        ),
        (
            "train-camera-feature-matrix",
            "run the complete 36-cell fitted-camera G-series matrix",
        ),
        (
            "evaluate-camera-feature",
            "evaluate fitted-camera G-series runs against Unity native 3D",
        ),
        (
            "report-camera-feature",
            "regenerate the fitted-camera feature report",
        ),
    ):
        child = stages.add_parser(name, help=help_text)
        supervised_stage = name in {
            "finetune",
            "finetune-matrix",
            "evaluate-finetuned",
            "report-finetuned",
            "train-extrinsic",
            "train-extrinsic-matrix",
            "evaluate-extrinsic",
            "report-extrinsic",
            "train-camera-feature",
            "train-camera-feature-matrix",
            "evaluate-camera-feature",
            "report-camera-feature",
        }
        child.add_argument(
            "--config",
            type=Path,
            default=(
                DEFAULT_SUPERVISED_CONFIG
                if supervised_stage
                else DEFAULT_CONFIG
            ),
        )
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
        elif name == "finetune":
            child.add_argument(
                "--ablation",
                choices=("A4", "A5", "A6", "A7", "A8", "A9"),
                required=True,
            )
            child.add_argument(
                "--fold",
                choices=("left_to_right", "right_to_left"),
                required=True,
            )
            child.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
            child.add_argument("--device")
        elif name == "finetune-matrix":
            child.add_argument("--device")
        elif name == "train-extrinsic":
            child.add_argument(
                "--method",
                choices=EXTRINSIC_METHODS,
                required=True,
            )
            child.add_argument(
                "--fold",
                choices=("left_to_right", "right_to_left"),
                required=True,
            )
            child.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
            child.add_argument("--device")
        elif name == "train-extrinsic-matrix":
            child.add_argument("--device")
        elif name == "train-camera-feature":
            child.add_argument(
                "--ablation",
                choices=CAMERA_GUIDED_ABLATIONS,
                required=True,
            )
            child.add_argument(
                "--fold",
                choices=("left_to_right", "right_to_left"),
                required=True,
            )
            child.add_argument(
                "--seed", type=int, choices=(0, 1, 2), required=True
            )
            child.add_argument("--device")
        elif name == "train-camera-feature-matrix":
            child.add_argument("--device")
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


def _sam3d_inference_coverage(output_root: Path) -> Mapping[str, object]:
    coverage: dict[str, object] = {}
    for camera_id in ("cam0", "cam1"):
        camera_root = output_root / "sam3d" / camera_id
        summary_path = camera_root / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else {}
        )
        proposal_sources: dict[str, int] = {}
        for path in camera_root.glob("*.npz"):
            with np.load(path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata"].item()))
            source = str(metadata.get("proposal_source", "person_detector"))
            proposal_sources[source] = proposal_sources.get(source, 0) + 1
        coverage[camera_id] = {
            "completed": int(summary.get("completed", 0)),
            "failed": len(summary.get("failed", [])),
            "proposal_sources": proposal_sources,
        }
    return coverage


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
        "sam3d_inference": _sam3d_inference_coverage(output_root),
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


def _required_mapping(
    config: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Unity supervised config requires {name}")
    return value


def _supervised_context(
    config: Mapping[str, object],
) -> tuple[Mapping[str, object], Path, Path, Path, Path]:
    base_path = config.get("base_config")
    if not isinstance(base_path, str) or not base_path:
        raise ValueError("Unity supervised config requires base_config")
    base = _load_config(Path(base_path))
    dataset_root, zero_shot_root = _paths(base)
    supervised_paths = _required_mapping(config, "paths")
    output_root = Path(str(supervised_paths["output_root"]))
    skeleton_path = Path(str(supervised_paths["skeleton"]))
    return base, dataset_root, zero_shot_root, output_root, skeleton_path


def _supervised_matrix_cells(
    config: Mapping[str, object] | None = None,
) -> tuple[tuple[str, str, int], ...]:
    if config is None:
        ablations = ("A4", "A5", "A6", "A7", "A8", "A9")
        folds = ("left_to_right", "right_to_left")
        seeds = (0, 1, 2)
    else:
        matrix = _required_mapping(config, "matrix")
        ablations = tuple(str(value) for value in matrix.get("ablations", ()))
        folds = tuple(str(value) for value in matrix.get("folds", ()))
        seeds = tuple(int(value) for value in matrix.get("seeds", ()))
        if set(ablations) != {"A4", "A5", "A6", "A7", "A8", "A9"} or len(
            ablations
        ) != 6:
            raise ValueError("supervised matrix must contain exactly A4--A9")
        if set(folds) != {"left_to_right", "right_to_left"} or len(folds) != 2:
            raise ValueError(
                "supervised matrix must contain exactly the two direction folds"
            )
        if set(seeds) != {0, 1, 2} or len(seeds) != 3:
            raise ValueError("supervised matrix must contain exactly seeds 0,1,2")
    return tuple(
        (ablation, fold, seed)
        for fold in folds
        for ablation in ablations
        for seed in seeds
    )


def _supervised_cell_is_complete(cell: tuple[str, str, int]) -> bool:
    """Default test seam; real matrix execution supplies a strict validator."""
    return False


def _run_supervised_cell(cell: tuple[str, str, int]) -> None:
    """Default test seam; real matrix execution supplies the cell runner."""
    raise RuntimeError(f"no Unity supervised cell runner configured for {cell}")


def _dispatch_supervised_matrix(
    cells: Sequence[tuple[str, str, int]],
    *,
    is_complete=None,
    run_cell=None,
) -> dict[str, int]:
    validator = is_complete or _supervised_cell_is_complete
    runner = run_cell or _run_supervised_cell
    counts = {"completed": 0, "reused": 0, "failed": 0}
    for cell in cells:
        if validator(cell):
            counts["reused"] += 1
            continue
        try:
            runner(cell)
        except Exception as error:
            counts["failed"] += 1
            print(
                f"failed={cell[0]}/{cell[1]}/seed_{cell[2]}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        counts["completed"] += 1
    return counts


def _fine_tune_config(
    config: Mapping[str, object],
    *,
    device: str | None,
) -> UnityFineTuneConfig:
    training = _required_mapping(config, "training")
    window = _required_mapping(config, "window")
    if str(training.get("optimizer", "")).lower() != "adamw":
        raise ValueError("Unity supervised optimizer must be adamw")
    return UnityFineTuneConfig(
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        window_length=int(window["length"]),
        train_stride=int(window["train_stride"]),
        device=str(device or training["device"]),
    )


def _unity_loss_config(
    config: Mapping[str, object],
) -> UnitySupervisedLossConfig:
    loss = _required_mapping(config, "loss")
    return UnitySupervisedLossConfig(
        unity_3d_weight=float(loss["unity_3d_weight"]),
        self_supervised_weight=float(loss["self_supervised_weight"]),
        smooth_l1_beta_m=float(loss["smooth_l1_beta_m"]),
    )


def _source_training_configs(
    checkpoint: Path,
) -> tuple[LossConfig, CorruptionConfig]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_loss = payload.get("loss_config")
    raw_corruption = payload.get("corruption_config")
    if not isinstance(raw_loss, Mapping) or not isinstance(
        raw_corruption, Mapping
    ):
        raise ValueError(f"source checkpoint lacks training configs: {checkpoint}")
    return LossConfig(**dict(raw_loss)), CorruptionConfig(**dict(raw_corruption))


def _training_sequence(
    *,
    dataset_root: Path,
    sam3d_root: Path,
    skeleton_path: Path,
    sequence_id: str,
    fps: float,
) -> UnitySupervisedSequence:
    benchmark = load_unity_benchmark(
        dataset_root,
        sequence_ids=(sequence_id,),
    )
    return build_supervised_sequence(
        benchmark,
        sam3d_root,
        sequence_id,
        skeleton_path=skeleton_path,
        fps=fps,
    )


def _run_one_finetune(
    config: Mapping[str, object],
    *,
    ablation: str,
    fold_name: str,
    seed: int,
    device: str | None,
    prepared_sequence: UnitySupervisedSequence | None = None,
):
    base, dataset_root, zero_shot_root, output_root, skeleton_path = (
        _supervised_context(config)
    )
    _supervised_matrix_cells(config)
    fold = UNITY_SUPERVISED_FOLDS[fold_name]
    checkpoints = _checkpoints(base, (ablation,))
    source_checkpoint = checkpoints[ablation]
    fine_config = _fine_tune_config(config, device=device)
    sequence = prepared_sequence or _training_sequence(
        dataset_root=dataset_root,
        sam3d_root=zero_shot_root / "sam3d",
        skeleton_path=skeleton_path,
        sequence_id=fold.train_sequence,
        fps=_data_fps(base),
    )
    self_config, corruption_config = _source_training_configs(source_checkpoint)
    return run_supervised_finetune(
        sequence,
        ablation=ablation,
        fold=fold,
        seed=seed,
        source_checkpoint=source_checkpoint,
        skeleton_path=skeleton_path,
        output_root=output_root,
        config=fine_config,
        loss_config=_unity_loss_config(config),
        self_supervised_config=self_config,
        corruption_config=corruption_config,
    )


def _finetune(
    args: argparse.Namespace,
    config: Mapping[str, object],
) -> int:
    run = _run_one_finetune(
        config,
        ablation=args.ablation,
        fold_name=args.fold,
        seed=args.seed,
        device=args.device,
    )
    provenance = json.loads(run.provenance_path.read_text(encoding="utf-8"))
    history = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    final = history[-1]
    print(f"run_root={run.run_root}")
    print(f"source_checkpoint_sha256={provenance['source_checkpoint_sha256']}")
    print(f"final_checkpoint_sha256={provenance['final_checkpoint_sha256']}")
    print(
        "final_losses="
        f"unity_3d:{float(final['unity_3d_loss']):.6f},"
        f"self_supervised:{float(final['self_supervised_loss']):.6f},"
        f"total:{float(final['total_loss']):.6f}"
    )
    return 0


def _finetune_matrix(
    args: argparse.Namespace,
    config: Mapping[str, object],
) -> int:
    base, dataset_root, zero_shot_root, output_root, skeleton_path = (
        _supervised_context(config)
    )
    cells = _supervised_matrix_cells(config)
    fine_config = _fine_tune_config(config, device=args.device)
    unity_loss = _unity_loss_config(config)
    sequences = {
        fold_name: _training_sequence(
            dataset_root=dataset_root,
            sam3d_root=zero_shot_root / "sam3d",
            skeleton_path=skeleton_path,
            sequence_id=fold.train_sequence,
            fps=_data_fps(base),
        )
        for fold_name, fold in UNITY_SUPERVISED_FOLDS.items()
    }

    def cell_contract(cell: tuple[str, str, int]):
        ablation, fold_name, seed = cell
        fold = UNITY_SUPERVISED_FOLDS[fold_name]
        checkpoint = _checkpoints(base, (ablation,))[ablation]
        self_config, corruption = _source_training_configs(checkpoint)
        resolved = _resolved_config(
            ablation=ablation,
            fold=fold,
            seed=seed,
            source_checkpoint=checkpoint,
            skeleton_path=skeleton_path,
            config=fine_config,
            loss_config=unity_loss,
            self_supervised_config=self_config,
            corruption_config=corruption,
        )
        run = _run_contract(
            ablation=ablation,
            fold=fold,
            seed=seed,
            output_root=output_root,
        )
        return run, checkpoint, resolved

    def completed(cell: tuple[str, str, int]) -> bool:
        run, checkpoint, resolved = cell_contract(cell)
        sequence = sequences[cell[1]]
        return validate_completed_run(
            run,
            source_checkpoint_sha256=_sha256_file(checkpoint),
            resolved_config=resolved,
            unity_manifest_sha256=str(
                sequence.raw_trial.source_metadata["unity_manifest_sha256"]
            ),
        )

    def execute(cell: tuple[str, str, int]) -> None:
        ablation, fold_name, seed = cell
        _run_one_finetune(
            config,
            ablation=ablation,
            fold_name=fold_name,
            seed=seed,
            device=args.device,
            prepared_sequence=sequences[fold_name],
        )
        print(f"completed={ablation}/{fold_name}/seed_{seed}")

    counts = _dispatch_supervised_matrix(
        cells,
        is_complete=completed,
        run_cell=execute,
    )
    print(
        f"completed={counts['completed']} reused={counts['reused']} "
        f"failed={counts['failed']}"
    )
    return 1 if counts["failed"] else 0


def _evaluate_supervised_artifacts(
    config: Mapping[str, object],
    *,
    run_missing_inference: bool,
) -> Path:
    base, dataset_root, zero_shot_root, output_root, skeleton_path = (
        _supervised_context(config)
    )
    cells = _supervised_matrix_cells(config)
    runs = discover_completed_runs(
        output_root,
        expected_cells=cells,
        resolved_config={},
    )
    if len(runs) != len(cells):
        raise RuntimeError(
            f"expected {len(cells)} valid fine-tuned runs, found {len(runs)}"
        )
    benchmark = load_unity_benchmark(dataset_root)
    sequences = build_supervised_sequences(
        benchmark,
        zero_shot_root / "sam3d",
        skeleton_path=skeleton_path,
        fps=_data_fps(base),
    )
    evaluation = _required_mapping(config, "evaluation")
    window_length = int(evaluation["window_length"])
    stride = int(evaluation["stride"])
    for run in runs:
        expected = tuple(
            run.run_root / "inference" / f"{sequence_id}.npz"
            for sequence_id in (run.test_sequence, "static_sweep")
        )
        if all(path.is_file() for path in expected):
            continue
        if not run_missing_inference:
            raise FileNotFoundError(
                f"missing fine-tuned inference artifacts below {run.run_root}"
            )
        run_finetuned_inference(
            run,
            sequences,
            skeleton_path=skeleton_path,
            window_length=window_length,
            stride=stride,
            device="cpu",
        )
    results = evaluate_finetuned_runs(benchmark, runs, sequences)
    provenance = {
        "git_commit": _git_commit(),
        "protocol": "direction-held-out-2x3",
        "runs": len(runs),
        "alignment": str(evaluation["alignment"]),
        "unity_gt_supervision": True,
        "static_evaluation_only": True,
    }
    bundle = build_finetuned_bundle(
        results,
        failures=(),
        provenance=provenance,
    )
    baseline_path = zero_shot_root / "report/results.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    return write_finetuned_report(
        bundle,
        output_root,
        baseline_results=baseline,
    )


def _evaluate_finetuned(config: Mapping[str, object]) -> int:
    report = _evaluate_supervised_artifacts(
        config, run_missing_inference=True
    )
    print(f"report={report}")
    return 0


def _report_finetuned(config: Mapping[str, object]) -> int:
    report = _evaluate_supervised_artifacts(
        config, run_missing_inference=False
    )
    print(f"report={report}")
    return 0


def _extrinsic_settings(
    config: Mapping[str, object],
    *,
    device: str | None = None,
) -> tuple[
    Mapping[str, object],
    Path,
    Path,
    Path,
    ExtrinsicTrainingConfig,
]:
    base, dataset_root, zero_shot_root, _, _ = _supervised_context(config)
    paths = _required_mapping(config, "paths")
    if "extrinsic_output_root" not in paths:
        raise ValueError(
            "Unity supervised config requires paths.extrinsic_output_root"
        )
    output_root = Path(str(paths["extrinsic_output_root"]))
    raw = _required_mapping(config, "extrinsic")
    methods = tuple(str(value) for value in raw.get("methods", ()))
    folds = tuple(str(value) for value in raw.get("folds", ()))
    seeds = tuple(int(value) for value in raw.get("seeds", ()))
    if set(methods) != set(EXTRINSIC_METHODS) or len(methods) != 3:
        raise ValueError("extrinsic matrix must contain exactly three methods")
    if set(folds) != set(UNITY_SUPERVISED_FOLDS) or len(folds) != 2:
        raise ValueError("extrinsic matrix must contain exactly two folds")
    if set(seeds) != {0, 1, 2} or len(seeds) != 3:
        raise ValueError("extrinsic matrix must contain exactly seeds 0,1,2")
    training = ExtrinsicTrainingConfig(
        epochs=int(raw["epochs"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        hidden_channels=int(raw["hidden_channels"]),
        max_delta_m=float(raw["max_delta_m"]),
        smooth_l1_beta_m=float(raw["smooth_l1_beta_m"]),
        device=str(device or raw["device"]),
    )
    return base, dataset_root, zero_shot_root, output_root, training


def _extrinsic_cells(
    config: Mapping[str, object],
) -> tuple[tuple[str, str, int], ...]:
    raw = _required_mapping(config, "extrinsic")
    return tuple(
        (str(method), str(fold), int(seed))
        for fold in raw["folds"]
        for method in raw["methods"]
        for seed in raw["seeds"]
    )


def _train_one_extrinsic(
    config: Mapping[str, object],
    *,
    method: str,
    fold_name: str,
    seed: int,
    device: str | None,
    sequences=None,
):
    base, dataset_root, zero_shot_root, output_root, training = (
        _extrinsic_settings(config, device=device)
    )
    benchmark = load_unity_benchmark(dataset_root)
    prepared = sequences or build_extrinsic_sequences(
        benchmark, zero_shot_root / "sam3d"
    )
    fold = UNITY_SUPERVISED_FOLDS[fold_name]
    return train_extrinsic_run(
        prepared[fold.train_sequence],
        method=method,
        fold=fold,
        seed=seed,
        output_root=output_root,
        config=training,
    )


def _train_extrinsic(
    args: argparse.Namespace,
    config: Mapping[str, object],
) -> int:
    run = _train_one_extrinsic(
        config,
        method=args.method,
        fold_name=args.fold,
        seed=args.seed,
        device=args.device,
    )
    print(f"run_root={run.run_root}")
    print(f"valid={validate_extrinsic_run(run)}")
    return 0


def _train_extrinsic_matrix(
    args: argparse.Namespace,
    config: Mapping[str, object],
) -> int:
    _, dataset_root, zero_shot_root, output_root, training = (
        _extrinsic_settings(config, device=args.device)
    )
    benchmark = load_unity_benchmark(dataset_root)
    sequences = build_extrinsic_sequences(
        benchmark, zero_shot_root / "sam3d"
    )
    counts = {"completed": 0, "reused": 0, "failed": 0}
    for method, fold_name, seed in _extrinsic_cells(config):
        fold = UNITY_SUPERVISED_FOLDS[fold_name]
        run = _extrinsic_run_contract(
            output_root, method, fold, seed
        )
        if validate_extrinsic_run(run):
            counts["reused"] += 1
            continue
        try:
            train_extrinsic_run(
                sequences[fold.train_sequence],
                method=method,
                fold=fold,
                seed=seed,
                output_root=output_root,
                config=training,
            )
        except Exception as error:
            counts["failed"] += 1
            print(
                f"failed={method}/{fold_name}/seed_{seed}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        counts["completed"] += 1
        print(f"completed={method}/{fold_name}/seed_{seed}")
    print(
        f"completed={counts['completed']} reused={counts['reused']} "
        f"failed={counts['failed']}"
    )
    return 1 if counts["failed"] else 0


def _evaluate_extrinsic_artifacts(
    config: Mapping[str, object],
    *,
    run_missing_inference: bool,
) -> Path:
    _, dataset_root, zero_shot_root, output_root, _ = _extrinsic_settings(
        config
    )
    benchmark = load_unity_benchmark(dataset_root)
    sequences = build_extrinsic_sequences(
        benchmark, zero_shot_root / "sam3d"
    )
    runs = []
    for method, fold_name, seed in _extrinsic_cells(config):
        fold = UNITY_SUPERVISED_FOLDS[fold_name]
        run = _extrinsic_run_contract(output_root, method, fold, seed)
        if not validate_extrinsic_run(run):
            raise RuntimeError(f"invalid or incomplete extrinsic run: {run.run_root}")
        expected = tuple(
            run.run_root / "inference" / f"{sequence_id}.npz"
            for sequence_id in (run.test_sequence, "static_sweep")
        )
        if not all(path.is_file() for path in expected):
            if not run_missing_inference:
                raise FileNotFoundError(
                    f"missing extrinsic inference below {run.run_root}"
                )
            run_extrinsic_inference(run, sequences, device="cpu")
        runs.append(run)
    heldout, static = evaluate_extrinsic_runs(benchmark, runs)
    return write_extrinsic_report(
        heldout,
        static_rows=static,
        output_root=output_root,
        provenance={
            "git_commit": _git_commit(),
            "protocol": "direction-held-out-2x3",
            "runs": len(runs),
            "alignment": "one_sim3_per_sequence",
            "unity_gt_supervision": True,
            "exact_camera_geometry": True,
        },
        baseline_results=json.loads(
            (zero_shot_root / "report/results.json").read_text(
                encoding="utf-8"
            )
        ),
    )


def _evaluate_extrinsic(config: Mapping[str, object]) -> int:
    report = _evaluate_extrinsic_artifacts(
        config, run_missing_inference=True
    )
    print(f"report={report}")
    return 0


def _report_extrinsic(config: Mapping[str, object]) -> int:
    report = _evaluate_extrinsic_artifacts(
        config, run_missing_inference=False
    )
    print(f"report={report}")
    return 0


def _camera_feature_context(
    config: Mapping[str, object],
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Path,
    Path,
    Path,
    Path,
    float,
]:
    base_path = config.get("base_config")
    if not isinstance(base_path, str) or not base_path:
        raise ValueError("Unity camera-feature config requires base_config")
    base = _load_config(Path(base_path))
    dataset_root, benchmark_output_root = _paths(base)
    paths = _required_mapping(config, "paths")
    settings = _required_mapping(config, "camera_feature")
    output_root = Path(str(paths["camera_feature_output_root"]))
    skeleton_path = Path(str(paths["skeleton"]))
    checkpoints = _required_mapping(base, "checkpoints")
    source_checkpoint = Path(str(checkpoints["A6"]))
    fps = float(_required_mapping(base, "data")["fps"])
    return (
        base,
        settings,
        dataset_root,
        benchmark_output_root / "sam3d",
        output_root,
        skeleton_path,
        fps,
    )


def _camera_feature_cells(
    settings: Mapping[str, object],
) -> tuple[tuple[str, str, int], ...]:
    ablations = tuple(str(value) for value in settings["ablations"])
    folds = tuple(str(value) for value in settings["folds"])
    seeds = tuple(int(value) for value in settings["seeds"])
    if set(ablations) != set(CAMERA_GUIDED_ABLATIONS) or len(ablations) != 6:
        raise ValueError("camera-feature matrix must contain exactly G0--G5")
    if set(folds) != {"left_to_right", "right_to_left"} or len(folds) != 2:
        raise ValueError("camera-feature matrix requires two direction folds")
    if set(seeds) != {0, 1, 2} or len(seeds) != 3:
        raise ValueError("camera-feature matrix requires seeds 0,1,2")
    return tuple(
        (ablation, fold, seed)
        for fold in folds
        for ablation in ablations
        for seed in seeds
    )


def _camera_training_config(
    settings: Mapping[str, object], device: str | None
) -> CameraGuidedTrainingConfig:
    return CameraGuidedTrainingConfig(
        epochs=int(settings["epochs"]),
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        window_length=int(settings["window_length"]),
        train_stride=int(settings["train_stride"]),
        batch_size=int(settings["batch_size"]),
        device=str(device or settings["device"]),
    )


def _camera_run_for_cell(
    output_root: Path, ablation: str, fold_name: str, seed: int
) -> CameraGuidedRun:
    return _camera_run_contract(
        output_root=output_root,
        ablation=ablation,
        fold=UNITY_SUPERVISED_FOLDS[fold_name],
        seed=seed,
    )


def _camera_cell_complete(run: CameraGuidedRun) -> bool:
    if not all(
        path.is_file()
        for path in (
            run.final_checkpoint,
            run.history_path,
            run.provenance_path,
            run.run_root / "inference" / f"{run.test_sequence}.npz",
            run.run_root / "inference" / "static_sweep.npz",
        )
    ):
        return False
    provenance = json.loads(
        run.provenance_path.read_text(encoding="utf-8")
    )
    return (
        provenance.get("ablation") == run.ablation
        and provenance.get("fold") == run.fold
        and provenance.get("seed") == run.seed
        and provenance.get("unity_native_3d_available_to_training") is False
    )


def _run_one_camera_feature(
    config: Mapping[str, object],
    *,
    ablation: str,
    fold_name: str,
    seed: int,
    device: str | None,
) -> CameraGuidedRun:
    (
        _,
        settings,
        dataset_root,
        sam3d_root,
        output_root,
        skeleton_path,
        fps,
    ) = _camera_feature_context(config)
    fold = UNITY_SUPERVISED_FOLDS[fold_name]
    benchmark = load_unity_benchmark(dataset_root)
    sequences = build_camera_guided_sequences(
        benchmark,
        sam3d_root,
        skeleton_path=skeleton_path,
        fps=fps,
        fold=fold,
        ablation=ablation,
        threshold_px=float(settings["threshold_px"]),
    )
    base_path = Path(str(config["base_config"]))
    base = _load_config(base_path)
    source_checkpoint = Path(
        str(_required_mapping(base, "checkpoints")["A6"])
    )
    run = train_camera_guided_run(
        sequences[fold.train_sequence],
        ablation=ablation,
        fold=fold,
        seed=seed,
        source_checkpoint=source_checkpoint,
        skeleton_path=skeleton_path,
        output_root=output_root,
        config=_camera_training_config(settings, device),
    )
    run_camera_guided_inference(
        run,
        sequences,
        skeleton_path=skeleton_path,
        window_length=int(settings["window_length"]),
        stride=int(settings["inference_stride"]),
        device=str(device or settings["device"]),
    )
    return run


def _train_camera_feature(
    args: argparse.Namespace, config: Mapping[str, object]
) -> int:
    run = _run_one_camera_feature(
        config,
        ablation=str(args.ablation),
        fold_name=str(args.fold),
        seed=int(args.seed),
        device=args.device,
    )
    print(f"camera_feature_run={run.run_root}")
    return 0


def _train_camera_feature_matrix(
    args: argparse.Namespace, config: Mapping[str, object]
) -> int:
    _, settings, _, _, output_root, _, _ = _camera_feature_context(config)
    counts = {"completed": 0, "reused": 0, "failed": 0}
    for ablation, fold_name, seed in _camera_feature_cells(settings):
        run = _camera_run_for_cell(
            output_root, ablation, fold_name, seed
        )
        if _camera_cell_complete(run):
            counts["reused"] += 1
            continue
        try:
            _run_one_camera_feature(
                config,
                ablation=ablation,
                fold_name=fold_name,
                seed=seed,
                device=args.device,
            )
        except Exception as error:
            counts["failed"] += 1
            print(
                f"failed={ablation}/{fold_name}/seed_{seed}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        counts["completed"] += 1
    print(f"camera_feature_matrix={counts}")
    return 1 if counts["failed"] else 0


def _camera_feature_runs(
    config: Mapping[str, object],
) -> tuple[CameraGuidedRun, ...]:
    _, settings, _, _, output_root, _, _ = _camera_feature_context(config)
    runs = tuple(
        _camera_run_for_cell(output_root, ablation, fold, seed)
        for ablation, fold, seed in _camera_feature_cells(settings)
    )
    incomplete = [run.run_root for run in runs if not _camera_cell_complete(run)]
    if incomplete:
        raise ValueError(
            f"camera-feature matrix has {len(incomplete)} incomplete cells"
        )
    return runs


def _evaluate_camera_feature(config: Mapping[str, object]) -> int:
    (
        _,
        _,
        dataset_root,
        _,
        output_root,
        _,
        _,
    ) = _camera_feature_context(config)
    runs = _camera_feature_runs(config)
    benchmark = load_unity_benchmark(dataset_root)
    rows = evaluate_camera_guided_runs(benchmark, runs)
    outputs = write_camera_guided_report(
        output_root / "evaluation",
        run_rows=rows,
        provenance={
            "runs": len(runs),
            "unity_native_3d_loaded_after_training": True,
        },
    )
    print(f"camera_feature_report={outputs['report']}")
    return 0


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
    if args.stage == "finetune":
        return _finetune(args, config)
    if args.stage == "finetune-matrix":
        return _finetune_matrix(args, config)
    if args.stage == "evaluate-finetuned":
        return _evaluate_finetuned(config)
    if args.stage == "report-finetuned":
        return _report_finetuned(config)
    if args.stage == "train-extrinsic":
        return _train_extrinsic(args, config)
    if args.stage == "train-extrinsic-matrix":
        return _train_extrinsic_matrix(args, config)
    if args.stage == "evaluate-extrinsic":
        return _evaluate_extrinsic(config)
    if args.stage == "report-extrinsic":
        return _report_extrinsic(config)
    if args.stage == "train-camera-feature":
        return _train_camera_feature(args, config)
    if args.stage == "train-camera-feature-matrix":
        return _train_camera_feature_matrix(args, config)
    if args.stage in {
        "evaluate-camera-feature",
        "report-camera-feature",
    }:
        return _evaluate_camera_feature(config)
    raise NotImplementedError(f"Unity benchmark stage not implemented: {args.stage}")
