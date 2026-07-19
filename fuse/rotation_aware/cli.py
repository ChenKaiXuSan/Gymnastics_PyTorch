"""Command line entrypoint for the isolated rotation-aware fusion route."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml
from torch.utils.data import DataLoader

from .config import load_skeleton_spec
from .corruptions import CorruptionConfig, write_corruption_manifest
from .data import load_cached_trial, load_person_trials, write_person_cache
from .dataset import (
    PosePairWindowDataset,
    WindowConfig,
    build_split_manifest,
    collate_pose_pair_windows,
)
from .evaluation import (
    discover_method_sequences,
    evaluate_person_trials,
    load_triangulated_references,
)
from .inference import canonicalize_trial, run_inference
from .losses import LossConfig
from .model import RotationAwareFusionModel
from .training import load_checkpoint, save_checkpoint, train_one_epoch, validate


_ENV = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")
_PATH = re.compile(r"\$\{([^{}]+)\}")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load project YAML with the small env/path interpolation subset we use."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("rotation-aware config must be a mapping")

    def resolve(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if not isinstance(value, str):
            return value

        def env(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), match.group(2) or "")

        value = _ENV.sub(env, value)

        def path_ref(match: re.Match[str]) -> str:
            keys = match.group(1).split(".")
            current: Any = data
            for key in keys:
                if not isinstance(current, Mapping) or key not in current:
                    return match.group(0)
                current = current[key]
            return str(current)

        return _PATH.sub(path_ref, value)

    return resolve(data)


def _paths(config: Mapping[str, Any], output_override: str | None) -> dict[str, Path]:
    paths = dict(config.get("paths", {}))
    root = Path(output_override or paths.get("output_root", "logs/fuse_rotation_aware"))
    return {
        "sam3d": Path(paths["sam3d_root"]),
        "split": Path(paths["split_cycle_root"]),
        "output": root,
        "cache": root / "cache",
        "skeleton": Path(paths.get("skeleton", "configs/fuse/skeleton_mhr70.yaml")),
    }


def resolve_fold(config: Mapping[str, Any], value: str | None) -> Path:
    """Resolve an explicit JSON path or numeric fold index without a cwd fallback."""
    paths = config.get("paths", {})
    if not isinstance(paths, Mapping):
        raise ValueError("rotation-aware config paths must be a mapping")
    root_value = paths.get("fold_root")
    default_value = paths.get("default_fold", "fold_00.json")
    if value is None:
        if isinstance(paths.get("fold_json"), str):
            candidate = Path(str(paths["fold_json"]))
        elif isinstance(root_value, (str, Path)) and isinstance(default_value, str):
            candidate = Path(root_value) / default_value
        else:
            raise ValueError(
                "fold configuration requires paths.fold_root and paths.default_fold"
            )
    elif value.isdigit():
        if not isinstance(root_value, (str, Path)):
            raise ValueError("numeric --fold requires paths.fold_root")
        candidate = Path(root_value) / f"fold_{int(value):02d}.json"
    else:
        candidate = Path(value)
    if candidate == Path(".") or candidate.suffix != ".json" or not candidate.is_file():
        raise ValueError(f"fold JSON does not exist: {candidate}")
    return candidate


def loss_config_for_ablation(ablation: str) -> LossConfig:
    """Return the declared self-supervised objective set for A4--A6."""
    full = LossConfig()
    if ablation == "A4":
        return replace(
            full,
            circular_axial_rotation_weight=0.0,
            so3_rotation_weight=0.0,
            adaptive_temporal_acceleration_weight=0.0,
            complete_cycle_rom_weight=0.0,
        )
    if ablation == "A5":
        return replace(full, complete_cycle_rom_weight=0.0)
    if ablation == "A6":
        return full
    raise ValueError(f"learned ablation must be A4, A5, or A6: {ablation}")


def _people(paths: Mapping[str, Path], wanted: Iterable[str] | None) -> list[str]:
    if wanted:
        return sorted({str(person) for person in wanted})
    root = (
        paths["sam3d"] / "person"
        if paths["sam3d"].name == "sam3d_body_results"
        else paths["sam3d"] / "sam3d_body_results" / "person"
    )
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and entry.name.isdigit()
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _skeleton_metadata(skeleton) -> dict[str, object]:
    return {
        "name": skeleton.name,
        "joint_names": list(skeleton.joint_names),
        "bones": [list(bone) for bone in skeleton.bones],
        "roles": {
            name: {
                "kind": role.kind,
                "joints": list(role.joints),
                "fallback": list(role.fallback),
            }
            for name, role in skeleton.roles.items()
        },
        "required_roles": list(skeleton.required_roles),
    }


def _validate_checkpoint_skeleton(payload: Mapping[str, object], skeleton) -> None:
    if payload.get("skeleton") != _skeleton_metadata(skeleton):
        raise ValueError("checkpoint skeleton does not match the current SkeletonSpec")


def _cached_trials(cache: Path, people: Iterable[str], skeleton) -> list:
    trials = []
    for person, paths in _cache_trial_paths(cache, people).items():
        for path in paths:
            trial, _ = load_cached_trial(path)
            trials.append(canonicalize_trial(trial, skeleton).trial)
    if not trials:
        raise FileNotFoundError(f"No cached trials found under {cache}")
    return trials


def _cache_trial_paths(cache: Path, people: Iterable[str]) -> dict[str, list[Path]]:
    """Require every selected person cache to match its declared cycle manifest."""
    cached: dict[str, list[Path]] = {}
    missing: list[str] = []
    for person in sorted({str(value) for value in people}):
        person_dir = cache / f"person_{person}"
        manifest_path = person_dir / "manifest.json"
        if not person_dir.is_dir() or not manifest_path.is_file():
            missing.append(f"person_{person}")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = manifest.get("trials") if isinstance(manifest, dict) else None
            if (
                not isinstance(expected, list)
                or not expected
                or manifest.get("person_id") != person
                or not all(isinstance(trial_id, str) for trial_id in expected)
            ):
                raise ValueError("invalid person cache manifest")
            paths = [person_dir / f"{trial_id}.npz" for trial_id in expected]
            if not all(path.is_file() for path in paths):
                raise ValueError("missing declared cache cycle")
        except (OSError, ValueError, json.JSONDecodeError):
            missing.append(f"person_{person}")
            continue
        cached[person] = paths
    if missing:
        raise FileNotFoundError(
            f"missing complete cache for selected people: {', '.join(missing)}"
        )
    return cached


def _manifest_people(manifest, subset: Iterable[str] | None) -> tuple[str, ...]:
    allowed = set(manifest.train) | set(manifest.val) | set(manifest.test)
    selected = set(str(value) for value in subset) if subset else allowed
    unknown = sorted(selected - allowed)
    if unknown:
        raise ValueError(f"--person is not present in the selected fold: {unknown}")
    return tuple(sorted(selected))


def _cmd_prepare(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    paths = _paths(config, args.output_root)
    skeleton = load_skeleton_spec(paths["skeleton"])
    fold = resolve_fold(config, args.fold)
    manifest = build_split_manifest(fold)
    discovered = set(_people(paths, None))
    explicit = set(str(person) for person in args.person) if args.person else None
    requested = (explicit & discovered) if explicit is not None else discovered
    aligned_people = []
    prepared_people = []
    failures: dict[str, str] = {}
    excluded: dict[str, str] = {}
    if explicit is not None:
        for person in sorted(explicit - discovered):
            failures[person] = "missing SAM3D person directory"
    for person in sorted(requested):
        record = paths["split"] / f"person_{person}" / f"alignment_record_{person}.json"
        if record.is_file():
            aligned_people.append(person)
        else:
            message = f"missing split alignment record: {record}"
            if explicit is None:
                excluded[person] = message
            else:
                failures[person] = message
    for person in aligned_people:
        try:
            trials = load_person_trials(
                person, paths["sam3d"], paths["split"], skeleton
            )
            write_person_cache(
                trials,
                paths["cache"],
                source_metadata=trials[0].source_metadata,
                config_metadata={"config": config, "skeleton": skeleton.name},
            )
            prepared_people.append(person)
        except (FileNotFoundError, KeyError, ValueError) as error:
            failures[person] = str(error)
    target = paths["output"] / "split_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                **asdict(manifest),
                "fold_path": str(fold),
                "selected_people": sorted(aligned_people),
                "aligned_people": aligned_people,
                "prepared_people": prepared_people,
                "failures": failures,
                "excluded_sam3d_people": excluded,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 1 if args.person and failures else 0


def _cmd_train(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    if not args.run_id:
        raise ValueError("train requires explicit --run-id")
    paths = _paths(config, args.output_root)
    skeleton = load_skeleton_spec(paths["skeleton"])
    fold = resolve_fold(config, args.fold)
    manifest = build_split_manifest(fold)
    selected = _manifest_people(manifest, args.person)
    trials = _cached_trials(paths["cache"], selected, skeleton)
    actual_people = tuple(sorted({trial.person_id for trial in trials}))
    if actual_people != selected:
        raise ValueError(
            "cached trial people do not exactly match the selected fold people"
        )
    training = dict(config.get("training", {}))
    training["ablation"] = args.ablation or "A6"
    loss_config = loss_config_for_ablation(str(training["ablation"]))
    training["loss_config"] = asdict(loss_config)
    window = WindowConfig(**dict(config.get("window", {})))
    train_trials = [trial for trial in trials if trial.person_id in manifest.train]
    val_trials = [trial for trial in trials if trial.person_id in manifest.val]
    if not train_trials:
        raise ValueError("no selected canonical trials belong to the training split")
    train_set = PosePairWindowDataset(
        train_trials, skeleton=skeleton, manifest=manifest, split="train", config=window
    )
    val_set = (
        PosePairWindowDataset(
            val_trials, skeleton=skeleton, manifest=manifest, split="val", config=window
        )
        if val_trials
        else None
    )
    batch_size = int(training.get("batch_size", 4))
    generator = torch.Generator().manual_seed(int(training.get("seed", 0)))
    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_pose_pair_windows,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_pose_pair_windows,
        )
        if val_set
        else loader
    )
    model = RotationAwareFusionModel(
        skeleton, hidden_channels=int(training.get("hidden_channels", 128))
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training.get("learning_rate", 1e-3))
    )
    run = paths["output"] / "runs" / args.run_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "config_resolved.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=True), encoding="utf-8"
    )
    (run / "split_manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    corruption = CorruptionConfig()
    corrupt_manifest = write_corruption_manifest(
        run / "corruption_manifest.json",
        [val_set[index]["window_id"] for index in range(len(val_set))]
        if val_set
        else [train_set[index]["window_id"] for index in range(len(train_set))],
        seed=int(training.get("seed", 0)),
        config=corruption,
        window_length=window.length,
        stride=window.eval_stride,
    )
    provenance = {
        "split_hash": _hash(asdict(manifest)),
        "training_config_hash": _hash(config),
        "corruption_manifest_hash": _hash(corrupt_manifest),
        "corruption_seed": str(int(training.get("seed", 0))),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
        ).stdout.strip()
        or "unknown",
        "selected_people": list(selected),
        "prepared_people": list(actual_people),
    }
    history = []
    best = float("-inf")
    for epoch in range(int(training.get("epochs", 1))):
        row = {
            "epoch": epoch,
            **train_one_epoch(
                model,
                loader,
                optimizer,
                skeleton,
                loss_config=loss_config,
                corruption_config=corruption,
                seed=int(training.get("seed", 0)),
                epoch=epoch,
            ),
        }
        score = validate(
            model,
            val_loader,
            skeleton,
            loss_config=loss_config,
            corruption_config=corruption,
            seed=int(training.get("seed", 0)),
        )["score"]
        row["val_score"] = score
        history.append(row)
        if score >= best:
            best = score
            save_checkpoint(
                run / "checkpoints" / "best.pt",
                model,
                optimizer,
                loss_config=loss_config,
                skeleton=skeleton,
                provenance=provenance,
                training_config=training or {"epochs": 1},
                corruption_config=corruption,
                score=score,
            )
    with (run / "train_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (run / "run_metadata.json").write_text(
        json.dumps({"provenance": provenance, "no_pseudo_gt_training": True}, indent=2),
        encoding="utf-8",
    )
    return 0


def _cmd_infer(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    if not args.run_id:
        raise ValueError("infer requires explicit --run-id")
    paths = _paths(config, args.output_root)
    skeleton = load_skeleton_spec(paths["skeleton"])
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else paths["output"] / "runs" / args.run_id / "checkpoints" / "best.pt"
    )
    raw_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw_payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    _validate_checkpoint_skeleton(raw_payload, skeleton)
    saved_training = raw_payload.get("training_config", {})
    if not isinstance(saved_training, Mapping):
        raise ValueError("checkpoint training_config must be a mapping")
    model = RotationAwareFusionModel(
        skeleton, hidden_channels=int(saved_training.get("hidden_channels", 128))
    )
    payload = load_checkpoint(checkpoint, model)
    saved_ablation = str(saved_training.get("ablation", "A6"))
    if args.ablation and args.ablation != saved_ablation:
        raise ValueError(
            f"--ablation {args.ablation} does not match checkpoint ablation {saved_ablation}"
        )
    raw_corruption_config = payload.get("corruption_config", {})
    if not isinstance(raw_corruption_config, Mapping):
        raise ValueError("checkpoint corruption_config must be a mapping")
    corruption_config = CorruptionConfig(**dict(raw_corruption_config))
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint provenance must be a mapping")
    manifest_path = checkpoint.parents[1] / "corruption_manifest.json"
    corruption_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    expected_manifest_hash = str(provenance.get("corruption_manifest_hash", ""))
    if not corruption_manifest or _hash(corruption_manifest) != expected_manifest_hash:
        raise ValueError(
            "run corruption manifest is missing or does not match checkpoint provenance"
        )
    window = WindowConfig(**dict(config.get("window", {})))
    manifest_window = corruption_manifest.get("window")
    if manifest_window != {"length": window.length, "stride": window.eval_stride}:
        raise ValueError(
            "run corruption manifest window contract does not match inference config"
        )
    if args.fold:
        manifest = build_split_manifest(resolve_fold(config, args.fold))
        allowed = _manifest_people(manifest, args.person)
    else:
        allowed = tuple(args.person or [])
    if allowed:
        people = list(allowed)
    else:
        prepared_manifest = paths["output"] / "split_manifest.json"
        if not prepared_manifest.is_file():
            raise FileNotFoundError(
                "default infer requires prepare split_manifest.json"
            )
        prepared = json.loads(prepared_manifest.read_text(encoding="utf-8")).get(
            "prepared_people"
        )
        if not isinstance(prepared, list) or not all(
            isinstance(item, str) for item in prepared
        ):
            raise ValueError("prepare split manifest has no valid prepared_people list")
        people = sorted(set(prepared))
    cache_paths = _cache_trial_paths(paths["cache"], people)
    inference_provenance = {
        **dict(provenance),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _file_hash(checkpoint),
        "ablation": saved_ablation,
        "model_config": {
            "hidden_channels": int(saved_training.get("hidden_channels", 128))
        },
        "training_config_hash": str(provenance.get("training_config_hash", "")),
        "inference_config_hash": _hash(config),
    }
    for person in people:
        for raw_path in cache_paths[person]:
            raw, _ = load_cached_trial(raw_path)
            run_inference(
                model,
                raw,
                skeleton,
                output_root=paths["output"] / "inference" / args.run_id,
                run_id=args.run_id,
                window_length=window.length,
                stride=window.eval_stride,
                provenance=inference_provenance,
                corruption_config=corruption_config,
                resolved_config=config,
                corruption_manifest=corruption_manifest,
            )
    return 0


def _cmd_evaluate(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    if not args.run_id:
        raise ValueError("evaluate requires explicit --run-id")
    paths = _paths(config, args.output_root)
    skeleton = load_skeleton_spec(paths["skeleton"])
    run_ids = [args.run_id] if isinstance(args.run_id, str) else list(args.run_id)
    roots = [paths["output"] / "inference" / run_id for run_id in run_ids]
    target = paths["output"] / "evaluation" / "+".join(run_ids)
    target.mkdir(parents=True, exist_ok=True)
    if args.fold:
        manifest = build_split_manifest(resolve_fold(config, args.fold))
        people = list(_manifest_people(manifest, args.person))
    else:
        people = args.person or [
            path.name.removeprefix("person_")
            for root in roots
            for path in root.glob("person_*")
        ]
    people = sorted({str(person) for person in people})
    rows = []
    joints = []
    availability: dict[str, dict[str, str]] = {}
    configured_paths = config.get("paths", {})
    old_root = (
        Path(configured_paths.get("old_fuse_root", "logs/fuse_experiments"))
        if isinstance(configured_paths, Mapping)
        else Path("logs/fuse_experiments")
    )
    external_root = (
        configured_paths.get("triangulated_root")
        if isinstance(configured_paths, Mapping)
        else None
    )
    for person in people:
        sequences = []
        status: dict[str, str] = {}
        seen: set[tuple[str, str]] = set()
        for root in roots:
            discovered, discovered_status = discover_method_sequences(
                root, old_root, str(person)
            )
            for method, value in discovered_status.items():
                if value == "available" or method not in status:
                    status[method] = value
            for sequence in discovered:
                key = (sequence.method, sequence.trial_id)
                if key not in seen:
                    seen.add(key)
                    sequences.append(sequence)
        availability[str(person)] = status
        if sequences:
            aliases = {
                "face_only": "A0",
                "side_only": "A1",
                "canonical_arithmetic": "A2",
                "quality_mean": "A3",
            }
            available_methods = {sequence.method for sequence in sequences}
            sequences = [
                sequence
                for sequence in sequences
                if not (
                    sequence.method in aliases
                    and aliases[sequence.method] in available_methods
                )
            ]
            new_sequences = [
                sequence
                for sequence in sequences
                if sequence.trial_id != "full_sequence"
            ]
            references = (
                load_triangulated_references(external_root, str(person), new_sequences)
                if external_root and new_sequences
                else None
            )
            report = evaluate_person_trials(
                str(person), new_sequences, skeleton, references=references
            )
            rows.extend(report.person_metrics)
            joints.extend(report.joint_metrics)
            old_sequences = [
                sequence
                for sequence in sequences
                if sequence.trial_id == "full_sequence"
            ]
            if old_sequences:
                report = evaluate_person_trials(str(person), old_sequences, skeleton)
                rows.extend(report.person_metrics)
                joints.extend(report.joint_metrics)
    for name, values in (
        ("metrics_by_person.csv", rows),
        ("metrics_by_joint.csv", joints),
        (
            "corruption_metrics.csv",
            [
                {
                    key: row.get(key)
                    for key in (
                        "person_id",
                        "method",
                        "fixed_corruption_recovery",
                        "fixed_corruption_recovery_availability",
                        "valid_points",
                    )
                }
                for row in rows
            ],
        ),
        (
            "rotation_metrics.csv",
            [
                {
                    key: row.get(key)
                    for key in (
                        "person_id",
                        "method",
                        "theta_rom",
                        "peak_omega",
                        "trunk_angular_jerk",
                        "rom_retention",
                        "rom_retention_availability",
                        "peak_angular_velocity_retention",
                        "peak_angular_velocity_retention_availability",
                        "swap_error",
                        "swap_error_availability",
                    )
                }
                for row in rows
            ],
        ),
    ):
        with (target / name).open("w", newline="", encoding="utf-8") as handle:
            if values:
                writer = csv.DictWriter(handle, fieldnames=values[0].keys())
                writer.writeheader()
                writer.writerows(values)
    (target / "report.json").write_text(
        json.dumps(
            {
                "person_metrics": rows,
                "joint_metrics": joints,
                "method_availability": availability,
                "no_pseudo_gt_training": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fuse.rotation_aware")
    parser.add_argument("--config", default="configs/fuse/rotation_aware.yaml")
    parser.add_argument("--person", action="append")
    parser.add_argument("--fold")
    parser.add_argument("--output-root")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "infer", "evaluate"):
        child = commands.add_parser(name)
        child.add_argument("--config", default=argparse.SUPPRESS)
        child.add_argument("--person", action="append", default=argparse.SUPPRESS)
        child.add_argument("--fold", default=argparse.SUPPRESS)
        child.add_argument("--output-root", default=argparse.SUPPRESS)
        if name in {"train", "infer"}:
            child.add_argument("--run-id")
            child.add_argument(
                "--ablation", choices=[f"A{index}" for index in range(4, 7)]
            )
        if name == "evaluate":
            child.add_argument("--run-id", action="append")
        if name == "infer":
            child.add_argument("--checkpoint")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    config = load_config(args.config)
    return {
        "prepare": _cmd_prepare,
        "train": _cmd_train,
        "infer": _cmd_infer,
        "evaluate": _cmd_evaluate,
    }[args.command](args, config)
