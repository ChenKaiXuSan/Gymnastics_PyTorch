"""Command line entrypoint for the isolated rotation-aware fusion route."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml
from torch.utils.data import DataLoader

from .config import load_skeleton_spec
from .corruptions import CorruptionConfig, write_corruption_manifest
from .data import (
    CACHE_MANIFEST_FILENAME,
    CACHE_PUBLICATION_LOCK,
    cache_manifest_identity,
    load_cached_trial,
    load_person_trials,
    resolve_cache_manifest,
    validate_cache_manifest_identities,
    verify_cache_manifest_identities,
    write_person_cache,
)
from .dataset import (
    PosePairCompleteCycleDataset,
    PosePairWindowDataset,
    WindowConfig,
    build_split_manifest,
    collate_pose_pair_windows,
)
from .evaluation import (
    DEFAULT_EXTERNAL_ALIGNMENT,
    EXTERNAL_ALIGNMENT_MODES,
    discover_method_sequences,
    evaluate_person_trials,
    load_triangulated_references,
)
from .inference import canonicalize_trial, run_inference
from .losses import LossConfig
from .model import RotationAwareFusionModel
from .prefetch import ThroughputConfig
from .profiling import StageProfiler
from .training import (
    load_checkpoint,
    prepare_validation_batches,
    save_checkpoint,
    train_one_epoch,
    validate,
)


_ENV = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")
_PATH = re.compile(r"\$\{([^{}]+)\}")
_CACHE_PUBLICATION_MAX_ATTEMPTS = 20
_CACHE_PUBLICATION_SLEEP_SECONDS = 0.05


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
    root = Path(output_override or paths.get("output_root", "local/runs/fuse_rotation_aware"))
    data = config.get("data", {})
    cache_root = (
        root / "cache"
        if output_override is not None
        else Path(data.get("cache_dir", root / "cache"))
        if isinstance(data, Mapping)
        else root / "cache"
    )
    return {
        "sam3d": Path(paths["sam3d_root"]),
        "split": Path(paths["split_cycle_root"]),
        "output": root,
        "cache": cache_root,
        "skeleton": Path(paths.get("skeleton", "configs/fusion/skeleton_mhr70.yaml")),
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


# The learned (network) ablations. A4/A5/A6 are the original objective ladder;
# A7/A8 are the twist-fusion matrix: A7 = A6 + per-view-peak ROM anchor (改法4),
# A8 = A7 + rotation-parameterised trunk-twist residual (改法2). Kept additive so
# A4/A5/A6 are byte-identical.
LEARNED_ABLATIONS = ("A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11")
# Ablations whose model uses the opt-in twist residual (改法2).
TWIST_ABLATIONS = frozenset({"A8", "A9"})
# Ablations that exchange same-frame joint tokens between the two views.
CROSS_ATTENTION_ABLATIONS = frozenset({"A10", "A11"})
# A11 keeps the attention capacity but removes explicit rotation conditioning.
ROTATION_UNCONDITIONED_ABLATIONS = frozenset({"A11"})


def loss_config_for_ablation(ablation: str) -> LossConfig:
    """Return the declared self-supervised objective set for A4--A11."""
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
    if ablation in ("A6", "A10"):
        return full
    if ablation in ("A7", "A8"):
        # 改法4: full A6 objective plus the per-view-peak ROM anchor. A8 also flips
        # on the twist residual in the model; the loss weights are identical.
        return replace(full, complete_cycle_rom_peak_weight=1.0)
    if ablation == "A9":
        # 改法3: A8 + the observed twist-rate anchor that restrains the twist
        # residual from over-rotating.
        return replace(
            full, complete_cycle_rom_peak_weight=1.0, observed_twist_rate_weight=1.0
        )
    if ablation == "A11":
        return replace(
            full,
            circular_axial_rotation_weight=0.0,
            so3_rotation_weight=0.0,
            complete_cycle_rom_weight=0.0,
        )
    raise ValueError(f"learned ablation must be one of {LEARNED_ABLATIONS}: {ablation}")


def model_kwargs_for_training(training: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one checkpoint-stable model architecture from training metadata."""
    ablation = str(training.get("ablation", "A6"))
    return {
        "hidden_channels": int(training.get("hidden_channels", 128)),
        "twist_residual": ablation in TWIST_ABLATIONS,
        "cross_attention": ablation in CROSS_ATTENTION_ABLATIONS,
        "attention_heads": int(training.get("attention_heads", 4)),
        "rotation_conditioning": ablation not in ROTATION_UNCONDITIONED_ABLATIONS,
    }


def model_metadata_for_training(training: Mapping[str, Any]) -> dict[str, Any]:
    """Return the architecture fields required to reproduce inference."""
    kwargs = model_kwargs_for_training(training)
    return {
        "hidden_channels": kwargs["hidden_channels"],
        "cross_attention": kwargs["cross_attention"],
        "attention_heads": kwargs["attention_heads"],
        "rotation_conditioning": kwargs["rotation_conditioning"],
    }


def _training_config_for_ablation(
    config: Mapping[str, Any], ablation: str
) -> dict[str, Any]:
    """Resolve the training schedule and objective label for one ablation."""
    training = dict(config.get("training", {}))
    schedule = training.pop("epochs_by_ablation", None)
    if schedule is not None:
        if (
            not isinstance(schedule, Mapping)
            or not schedule
            or not set(schedule).issubset(LEARNED_ABLATIONS)
        ):
            raise ValueError(
                f"training.epochs_by_ablation keys must be learned ablations {LEARNED_ABLATIONS}"
            )
        if ablation not in schedule:
            raise ValueError(f"epochs_by_ablation must define epochs for {ablation}")
        training["epochs"] = int(schedule[ablation])
    epochs = int(training.get("epochs", 1))
    if epochs < 1:
        raise ValueError("training epochs must be positive")
    training["epochs"] = epochs
    training["ablation"] = ablation
    if ablation in CROSS_ATTENTION_ABLATIONS:
        training["attention_heads"] = int(training.get("attention_heads", 4))
    return training


def _validate_safe_run_id_component(run_id: str) -> None:
    """Reject path-like IDs before they can escape a protocol output root."""
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or ".." in run_id
        or Path(run_id).is_absolute()
        or "/" in run_id
        or "\\" in run_id
    ):
        raise ValueError(f"run ID must be a safe run-ID component: {run_id!r}")


def _config_has_protocol(config: Mapping[str, Any]) -> bool:
    training = config.get("training", {})
    return isinstance(training, Mapping) and training.get("protocol") is not None


def _protocol_run_id_token(training: Mapping[str, Any]) -> str | None:
    protocol = training.get("protocol")
    if protocol is None:
        return None
    if not isinstance(protocol, Mapping):
        raise ValueError("training.protocol must be a mapping")
    template = protocol.get("run_id_token_template")
    if not isinstance(template, str) or not template:
        raise ValueError("training.protocol requires run_id_token_template")
    try:
        token = template.format(
            ablation=str(training["ablation"]),
            ablation_lower=str(training["ablation"]).lower(),
            batch_size=int(training["batch_size"]),
            epochs=int(training["epochs"]),
        )
    except (KeyError, ValueError) as error:
        raise ValueError("training.protocol run_id_token_template is invalid") from error
    return token


def _run_id_has_protocol_token(run_id: str, token: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", run_id))


def _validate_protocol_run_id(run_id: str, training: Mapping[str, Any]) -> bool:
    """Validate a protocol-qualified run ID and report whether its directory is protected."""
    token = _protocol_run_id_token(training)
    if token is None:
        return False
    _validate_safe_run_id_component(run_id)
    if not _run_id_has_protocol_token(run_id, token):
        raise ValueError(
            f"run ID for this protocol must include token {token!r}: {run_id!r}"
        )
    return True


def _validate_config_protocol_run_id(run_id: str, config: Mapping[str, Any]) -> None:
    """Require one protected A4/A5/A6 schedule token before output-path access."""
    _validate_safe_run_id_component(run_id)
    matches = []
    for ablation in ("A4", "A5", "A6"):
        training = _training_config_for_ablation(config, ablation)
        token = _protocol_run_id_token(training)
        if token is not None and _run_id_has_protocol_token(run_id, token):
            matches.append(token)
    if len(matches) != 1:
        raise ValueError(
            f"protected run ID must match exactly one resolved protocol token: {run_id!r}"
        )


def _build_training_loaders(
    train_set: object,
    val_set: object | None,
    train_cycles: object,
    val_cycles: object,
    *,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, bool]:
    """Build the historical shared-generator loader topology for train-only folds."""
    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_pose_pair_windows,
    )
    uses_training_validation = val_set is None
    val_loader = (
        loader
        if uses_training_validation
        else DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_pose_pair_windows,
        )
    )
    complete_cycle_loader = DataLoader(
        train_cycles,
        batch_size=1,
        shuffle=True,
        generator=generator,
        collate_fn=collate_pose_pair_windows,
    )
    val_complete_cycle_loader = DataLoader(
        val_cycles,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_pose_pair_windows,
    )
    return (
        loader,
        val_loader,
        complete_cycle_loader,
        val_complete_cycle_loader,
        uses_training_validation,
    )


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
        "joint_groups": {
            name: list(joints) for name, joints in skeleton.joint_groups.items()
        },
    }


def _validate_checkpoint_skeleton(payload: Mapping[str, object], skeleton) -> None:
    if payload.get("skeleton") != _skeleton_metadata(skeleton):
        raise ValueError("checkpoint skeleton does not match the current SkeletonSpec")


def _cached_trials_with_provenance(
    cache: Path, people: Iterable[str], skeleton
) -> tuple[list, dict[str, dict[str, Any]]]:
    trials = []
    paths_by_person, identities = _resolve_cache_selection(cache, people)
    for person, paths in paths_by_person.items():
        for path in paths:
            trial, manifest = load_cached_trial(path)
            if cache_manifest_identity(manifest) != identities[person]:
                raise ValueError(
                    f"loaded cache manifest identity changed for person {person}"
                )
            trials.append(canonicalize_trial(trial, skeleton).trial)
    if not trials:
        raise FileNotFoundError(f"No cached trials found under {cache}")
    return trials, identities


def _cached_trials(cache: Path, people: Iterable[str], skeleton) -> list:
    trials, _ = _cached_trials_with_provenance(cache, people, skeleton)
    return trials


def _try_shared_publication_guard(person_dir: Path) -> int | None:
    """Return a shared flock descriptor, or None while a writer is active."""
    lock_path = person_dir / CACHE_PUBLICATION_LOCK
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


def _release_shared_publication_guard(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _resolve_cache_selection(
    cache: Path, people: Iterable[str]
) -> tuple[dict[str, list[Path]], dict[str, dict[str, Any]]]:
    """Resolve immutable generation paths, waiting only for a first publication."""
    cached: dict[str, list[Path]] = {}
    identities: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for person in sorted({str(value) for value in people}):
        person_dir = cache / f"person_{person}"
        manifest_path = person_dir / CACHE_MANIFEST_FILENAME
        for attempt in range(_CACHE_PUBLICATION_MAX_ATTEMPTS + 1):
            if manifest_path.is_file():
                try:
                    payload_dir, manifest = resolve_cache_manifest(person_dir)
                    expected = (
                        manifest.get("trials") if isinstance(manifest, dict) else None
                    )
                    if (
                        not isinstance(expected, list)
                        or not expected
                        or manifest.get("person_id") != person
                        or not all(isinstance(trial_id, str) for trial_id in expected)
                    ):
                        raise ValueError("invalid person cache manifest")
                    paths = [payload_dir / f"{trial_id}.npz" for trial_id in expected]
                    if not all(path.is_file() for path in paths):
                        raise ValueError("missing declared cache cycle")
                except (OSError, ValueError, json.JSONDecodeError):
                    missing.append(f"person_{person}")
                    break
                cached[person] = paths
                identities[person] = cache_manifest_identity(manifest)
                break
            if not person_dir.is_dir():
                missing.append(f"person_{person}")
                break
            descriptor = _try_shared_publication_guard(person_dir)
            if descriptor is None:
                if attempt == _CACHE_PUBLICATION_MAX_ATTEMPTS:
                    missing.append(
                        f"person_{person} (cache publication timed out)"
                    )
                    break
                time.sleep(_CACHE_PUBLICATION_SLEEP_SECONDS)
                continue
            try:
                pointer_published = manifest_path.is_file()
            finally:
                _release_shared_publication_guard(descriptor)
            if pointer_published:
                continue
            missing.append(f"person_{person}")
            break
    if missing:
        raise FileNotFoundError(
            f"missing complete cache for selected people: {', '.join(missing)}"
        )
    return cached, identities


def _cache_trial_paths(cache: Path, people: Iterable[str]) -> dict[str, list[Path]]:
    paths, _ = _resolve_cache_selection(cache, people)
    return paths


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
            if not trials:
                raise ValueError(f"alignment record has no cycles for person {person}")
            write_person_cache(
                trials,
                paths["cache"],
                source_metadata=trials[0].source_metadata,
                config_metadata={"config": config, "skeleton": skeleton.name},
            )
            prepared_people.append(person)
        except (KeyError, OSError, ValueError) as error:
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
    return 1 if failures else 0


def _cmd_train(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    if not args.run_id:
        raise ValueError("train requires explicit --run-id")
    training = _training_config_for_ablation(config, args.ablation or "A6")
    protected_run = _validate_protocol_run_id(args.run_id, training)
    performance = config.get("performance", {})
    if not isinstance(performance, Mapping):
        raise ValueError("rotation-aware performance config must be a mapping")
    throughput = ThroughputConfig(**dict(performance))
    paths = _paths(config, args.output_root)
    run = paths["output"] / "runs" / args.run_id
    if protected_run and run.exists() and any(run.iterdir()):
        raise FileExistsError(f"protected batch-64 run directory is not empty: {run}")
    skeleton = load_skeleton_spec(paths["skeleton"])
    fold = resolve_fold(config, args.fold)
    manifest = build_split_manifest(fold)
    selected = _manifest_people(manifest, args.person)
    trials, cache_manifests = _cached_trials_with_provenance(
        paths["cache"], selected, skeleton
    )
    actual_people = tuple(sorted({trial.person_id for trial in trials}))
    if actual_people != selected:
        raise ValueError(
            "cached trial people do not exactly match the selected fold people"
        )
    loss_config = loss_config_for_ablation(str(training["ablation"]))
    training["loss_config"] = asdict(loss_config)
    device = str(training.get("device", "cpu"))
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
    train_cycles = PosePairCompleteCycleDataset(
        train_trials, skeleton=skeleton, manifest=manifest, split="train"
    )
    validation_cycle_trials = val_trials or train_trials
    validation_cycle_split = "val" if val_trials else "train"
    val_cycles = PosePairCompleteCycleDataset(
        validation_cycle_trials,
        skeleton=skeleton,
        manifest=manifest,
        split=validation_cycle_split,
    )
    batch_size = int(training.get("batch_size", 4))
    generator = torch.Generator().manual_seed(int(training.get("seed", 0)))
    (
        loader,
        val_loader,
        complete_cycle_loader,
        val_complete_cycle_loader,
        uses_training_validation,
    ) = _build_training_loaders(
        train_set,
        val_set,
        train_cycles,
        val_cycles,
        batch_size=batch_size,
        generator=generator,
    )
    model = RotationAwareFusionModel(skeleton, **model_kwargs_for_training(training))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training.get("learning_rate", 1e-3))
    )
    run.mkdir(parents=True, exist_ok=True)
    profile_path = run / "stage_profile.jsonl"
    profile_path.unlink(missing_ok=True)
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
        "cache_manifests": cache_manifests,
    }
    history = []
    best = float("-inf")
    epochs = int(training.get("epochs", 1))
    prepared_val_loader = None
    prepared_val_complete_cycle_loader = None
    if throughput.cache_validation_batches and not uses_training_validation:
        prepared_val_loader = prepare_validation_batches(
            val_loader,
            skeleton,
            seed=int(training.get("seed", 0)),
            corruption_config=corruption,
            throughput_config=throughput,
        )
        prepared_val_complete_cycle_loader = prepare_validation_batches(
            val_complete_cycle_loader,
            skeleton,
            seed=int(training.get("seed", 0)),
            corruption_config=corruption,
            throughput_config=throughput,
    )
    for epoch in range(epochs):
        profiler = StageProfiler(
            enabled=throughput.profile_stages, device=torch.device(device)
        )
        row = {
            "epoch": epoch,
            **train_one_epoch(
                model,
                loader,
                optimizer,
                skeleton,
                loss_config=loss_config,
                corruption_config=corruption,
                complete_cycle_loader=complete_cycle_loader,
                seed=int(training.get("seed", 0)),
                epoch=epoch,
                device=device,
                profiler=profiler,
                throughput_config=throughput,
            ),
        }
        score = validate(
            model,
            val_loader,
            skeleton,
            loss_config=loss_config,
            corruption_config=corruption,
            complete_cycle_loader=val_complete_cycle_loader,
            seed=int(training.get("seed", 0)),
            device=device,
            profiler=profiler,
            throughput_config=throughput,
            prepared_loader=prepared_val_loader,
            prepared_complete_cycle_loader=prepared_val_complete_cycle_loader,
            scalar_forward=True,
        )["score"]
        row["val_score"] = score
        history.append(row)
        print(
            f"[epoch] run_id={args.run_id} epoch={epoch + 1}/{epochs} "
            f"loss={float(row['loss']):.6g} val_score={float(score):.6g}",
            flush=True,
        )
        # Save on improvement (finite scores behave exactly as before). If the
        # validation score is non-finite (e.g. nan), still persist the latest
        # weights so a completed run never ends up with no checkpoint at all.
        if not math.isfinite(score) or score >= best:
            if math.isfinite(score):
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
        if throughput.profile_stages:
            with profile_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"epoch": epoch, "stages": profiler.summary()}, sort_keys=True)
                    + "\n"
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
    if _config_has_protocol(config):
        _validate_safe_run_id_component(args.run_id)
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
    if _config_has_protocol(config):
        saved_ablation = str(saved_training.get("ablation", ""))
        active_training = _training_config_for_ablation(config, saved_ablation)
        protected_fields = (
            "ablation",
            "batch_size",
            "epochs",
            "learning_rate",
            "hidden_channels",
            "attention_heads",
            "seed",
            "protocol",
        )
        mismatched_fields = [
            name
            for name in protected_fields
            if name in active_training and saved_training.get(name) != active_training[name]
        ]
        if mismatched_fields:
            raise ValueError(
                "checkpoint training protocol does not match active config: "
                + ", ".join(mismatched_fields)
            )
        _validate_protocol_run_id(args.run_id, active_training)
        if not _validate_protocol_run_id(args.run_id, saved_training):
            raise ValueError("protected infer checkpoint is missing training.protocol")
    saved_ablation = str(saved_training.get("ablation", "A6"))
    model = RotationAwareFusionModel(
        skeleton, **model_kwargs_for_training(saved_training)
    )
    payload = load_checkpoint(checkpoint, model)
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
    cache_paths, consumed_cache_manifests = _resolve_cache_selection(
        paths["cache"], people
    )
    checkpoint_cache_manifests = validate_cache_manifest_identities(
        provenance.get("cache_manifests"), field_name="checkpoint cache_manifests"
    )
    verify_cache_manifest_identities(
        checkpoint_cache_manifests, consumed_cache_manifests
    )
    inference_provenance = {
        **dict(provenance),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _file_hash(checkpoint),
        "ablation": saved_ablation,
        "model_config": model_metadata_for_training(saved_training),
        "training_config_hash": str(provenance.get("training_config_hash", "")),
        "inference_config_hash": _hash(config),
        "inference_cache_manifests": consumed_cache_manifests,
    }
    for person in people:
        for raw_path in cache_paths[person]:
            raw, cache_manifest = load_cached_trial(raw_path)
            consumed_identity = cache_manifest_identity(cache_manifest)
            if consumed_identity != consumed_cache_manifests[person]:
                raise ValueError(
                    f"loaded cache manifest identity changed for person {person}"
                )
            run_inference(
                model,
                raw,
                skeleton,
                output_root=paths["output"] / "inference" / args.run_id,
                run_id=args.run_id,
                window_length=window.length,
                stride=window.eval_stride,
                provenance={
                    **inference_provenance,
                    "consumed_cache_manifest": consumed_identity,
                },
                corruption_config=corruption_config,
                resolved_config=config,
                corruption_manifest=corruption_manifest,
            )
    return 0


def _cmd_evaluate(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    if not args.run_id:
        raise ValueError("evaluate requires explicit --run-id")
    run_ids = [args.run_id] if isinstance(args.run_id, str) else list(args.run_id)
    alignment = str(getattr(args, "alignment", DEFAULT_EXTERNAL_ALIGNMENT))
    if _config_has_protocol(config):
        for run_id in run_ids:
            _validate_config_protocol_run_id(run_id, config)
    paths = _paths(config, args.output_root)
    skeleton = load_skeleton_spec(paths["skeleton"])
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
        Path(configured_paths.get("old_fuse_root", "local/runs/fuse_experiments"))
        if isinstance(configured_paths, Mapping)
        else Path("local/runs/fuse_experiments")
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
                str(person),
                new_sequences,
                skeleton,
                references=references,
                alignment=alignment,
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
                "external_alignment": alignment,
                "no_pseudo_gt_training": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gymnastics fuse rotation-aware")
    parser.add_argument("--config", default="configs/fusion/rotation_aware.yaml")
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
                "--ablation", choices=list(LEARNED_ABLATIONS)
            )
        if name == "evaluate":
            child.add_argument("--run-id", action="append")
            child.add_argument(
                "--alignment",
                default=DEFAULT_EXTERNAL_ALIGNMENT,
                choices=list(EXTERNAL_ALIGNMENT_MODES),
                help=(
                    "Alignment applied before comparing against the triangulated "
                    "pseudo-reference. 'similarity' (default) removes the static "
                    "world-frame and scale mismatch per sequence; 'root' is the "
                    "legacy pelvis-only behaviour."
                ),
            )
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
