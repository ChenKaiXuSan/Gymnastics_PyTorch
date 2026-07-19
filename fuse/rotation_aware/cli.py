"""Command line entrypoint for the isolated rotation-aware fusion route."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict
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
    fold_value = paths.get("fold_json", paths.get("fold_root"))
    if not isinstance(fold_value, (str, Path)):
        fold_value = ""
    return {
        "sam3d": Path(paths["sam3d_root"]),
        "split": Path(paths["split_cycle_root"]),
        "output": root,
        "cache": root / "cache",
        "skeleton": Path(paths.get("skeleton", "configs/fuse/skeleton_mhr70.yaml")),
        "fold": Path(fold_value),
    }


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


def _cached_trials(cache: Path, people: Iterable[str], skeleton) -> list:
    trials = []
    for person in people:
        person_dir = cache / f"person_{person}"
        for path in sorted(person_dir.glob("cycle_*.npz")):
            trial, _ = load_cached_trial(path)
            trials.append(canonicalize_trial(trial, skeleton).trial)
    if not trials:
        raise FileNotFoundError(f"No cached trials found under {cache}")
    return trials


def _cmd_prepare(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    paths = _paths(config, args.output_root)
    skeleton = load_skeleton_spec(paths["skeleton"])
    people = _people(paths, args.person)
    for person in people:
        trials = load_person_trials(person, paths["sam3d"], paths["split"], skeleton)
        write_person_cache(
            trials,
            paths["cache"],
            source_metadata=trials[0].source_metadata,
            config_metadata={"config": config, "skeleton": skeleton.name},
        )
    fold = Path(args.fold) if args.fold else paths["fold"]
    if fold and fold.exists():
        manifest = build_split_manifest(fold)
        (paths["output"] / "split_manifest.json").parent.mkdir(
            parents=True, exist_ok=True
        )
        (paths["output"] / "split_manifest.json").write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8"
        )
    return 0


def _cmd_train(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    if not args.run_id:
        raise ValueError("train requires explicit --run-id")
    paths = _paths(config, args.output_root)
    skeleton = load_skeleton_spec(paths["skeleton"])
    fold = Path(args.fold) if args.fold else paths["fold"]
    manifest = build_split_manifest(fold)
    trials = _cached_trials(
        paths["cache"], manifest.train + manifest.val + manifest.test, skeleton
    )
    training = dict(config.get("training", {}))
    window = WindowConfig(**dict(config.get("window", {})))
    train_set = PosePairWindowDataset(
        trials, skeleton=skeleton, manifest=manifest, split="train", config=window
    )
    val_set = (
        PosePairWindowDataset(
            trials, skeleton=skeleton, manifest=manifest, split="val", config=window
        )
        if manifest.val
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
        [train_set[index]["window_id"] for index in range(len(train_set))],
        seed=int(training.get("seed", 0)),
        config=corruption,
    )
    provenance = {
        "split_hash": _hash(asdict(manifest)),
        "corruption_manifest_hash": _hash(corrupt_manifest),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
        ).stdout.strip()
        or "unknown",
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
                loss_config=LossConfig(),
                corruption_config=corruption,
                seed=int(training.get("seed", 0)),
                epoch=epoch,
            ),
        }
        score = validate(
            model,
            val_loader,
            skeleton,
            loss_config=LossConfig(),
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
                loss_config=LossConfig(),
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
    training = dict(config.get("training", {}))
    model = RotationAwareFusionModel(
        skeleton, hidden_channels=int(training.get("hidden_channels", 128))
    )
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else paths["output"] / "runs" / args.run_id / "checkpoints" / "best.pt"
    )
    payload = load_checkpoint(checkpoint, model)
    people = (
        _people(paths, args.person)
        if args.person
        else [
            path.name.removeprefix("person_")
            for path in paths["cache"].glob("person_*")
        ]
    )
    for trial in _cached_trials(paths["cache"], people, skeleton):
        raw_path = (
            paths["cache"] / f"person_{trial.person_id}" / f"{trial.trial_id}.npz"
        )
        raw, _ = load_cached_trial(raw_path)
        provenance = payload.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("checkpoint provenance must be a mapping")
        run_inference(
            model,
            raw,
            skeleton,
            output_root=paths["output"] / "inference" / args.run_id,
            run_id=args.run_id,
            provenance=dict(provenance),
        )
    return 0


def _cmd_evaluate(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    if not args.run_id:
        raise ValueError("evaluate requires explicit --run-id")
    paths = _paths(config, args.output_root)
    skeleton = load_skeleton_spec(paths["skeleton"])
    root = paths["output"] / "inference" / args.run_id
    target = paths["output"] / "evaluation" / args.run_id
    target.mkdir(parents=True, exist_ok=True)
    people = args.person or [
        path.name.removeprefix("person_") for path in root.glob("person_*")
    ]
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
        sequences, status = discover_method_sequences(root, old_root, str(person))
        availability[str(person)] = status
        if sequences:
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
        if name in {"train", "infer", "evaluate"}:
            child.add_argument("--run-id")
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
