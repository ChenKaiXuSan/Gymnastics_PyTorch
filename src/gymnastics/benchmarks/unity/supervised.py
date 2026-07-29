"""Deterministic, leakage-safe Unity-supervised fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from gymnastics.common.skeletons.mhr70 import mhr_names
from gymnastics.fusion.rotation_aware.config import SkeletonSpec
from gymnastics.fusion.rotation_aware.corruptions import CorruptionConfig
from gymnastics.fusion.rotation_aware.dataset import (
    collate_pose_pair_windows,
)
from gymnastics.fusion.rotation_aware.losses import (
    LossConfig,
    compute_self_supervised_losses,
)
from gymnastics.fusion.rotation_aware.inference import run_inference
from gymnastics.fusion.rotation_aware.model import RotationAwareFusionModel
from gymnastics.fusion.rotation_aware.training import _forward_window

from .fusion import _save_sequence, load_rotation_aware_model
from .schema import MethodSequence
from .supervised_data import (
    UnityFold,
    UnitySupervisedSequence,
    UnitySupervisedWindowDataset,
)
from .supervised_loss import (
    UnitySupervisedLossConfig,
    compute_unity_supervised_loss,
)


@dataclass(frozen=True)
class UnityFineTuneConfig:
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    window_length: int = 32
    train_stride: int = 8
    device: str = "cuda"

    def __post_init__(self) -> None:
        if (
            self.epochs < 1
            or self.batch_size < 1
            or self.window_length < 1
            or self.train_stride < 1
        ):
            raise ValueError("epochs, batch size, and window settings must be positive")
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("optimizer settings are invalid")
        if not self.device:
            raise ValueError("device is required")


@dataclass(frozen=True)
class UnityFineTuneRun:
    ablation: str
    fold: str
    seed: int
    train_sequence: str
    test_sequence: str
    run_root: Path
    final_checkpoint: Path
    metrics_path: Path
    provenance_path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _plain(value: object) -> object:
    return json.loads(json.dumps(value))


def _resolved_config(
    *,
    ablation: str,
    fold: UnityFold,
    seed: int,
    source_checkpoint: Path,
    skeleton_path: Path,
    config: UnityFineTuneConfig,
    loss_config: UnitySupervisedLossConfig,
    self_supervised_config: LossConfig,
    corruption_config: CorruptionConfig,
) -> dict[str, object]:
    return _plain(
        {
            "ablation": ablation,
            "fold": asdict(fold),
            "seed": seed,
            "source_checkpoint": str(Path(source_checkpoint).resolve()),
            "skeleton_path": str(Path(skeleton_path).resolve()),
            "training": asdict(config),
            "unity_loss": asdict(loss_config),
            "self_supervised_loss": asdict(self_supervised_config),
            "corruption": asdict(corruption_config),
        }
    )


def train_supervised_epoch(
    model: RotationAwareFusionModel,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    skeleton: SkeletonSpec,
    *,
    loss_config: UnitySupervisedLossConfig,
    self_supervised_config: LossConfig,
    corruption_config: CorruptionConfig,
    seed: int,
    epoch: int,
    device: str | torch.device,
) -> dict[str, float]:
    """Run one full-supervision epoch and reject non-finite gradients."""
    target_device = torch.device(device)
    model.to(target_device).train()
    torch.manual_seed(int(seed) + int(epoch))
    metrics: list[dict[str, float]] = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        output, prepared = _forward_window(
            model,
            batch,
            skeleton,
            seed=int(seed),
            corruption_config=corruption_config,
            device=target_device,
            epoch=epoch,
            phase="unity_supervised_train",
        )
        self_supervised = compute_self_supervised_losses(
            output,
            prepared,
            self_supervised_config,
            skeleton,
        ).total
        losses = compute_unity_supervised_loss(
            output,
            prepared,
            loss_config,
            self_supervised=self_supervised,
        )
        components = (
            losses.unity_3d,
            losses.self_supervised,
            losses.total,
        )
        if not all(torch.isfinite(value).item() for value in components):
            raise FloatingPointError("Unity supervised loss component is non-finite")
        losses.total.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(
            torch.isfinite(gradient).all().item() for gradient in gradients
        ):
            raise FloatingPointError("Unity supervised gradient is non-finite")
        optimizer.step()
        metrics.append(
            {
                "unity_3d_loss": float(losses.unity_3d.detach().cpu()),
                "self_supervised_loss": float(
                    losses.self_supervised.detach().cpu()
                ),
                "total_loss": float(losses.total.detach().cpu()),
            }
        )
    if not metrics:
        raise ValueError("Unity supervised loader produced no batches")
    return {
        key: float(np.mean([item[key] for item in metrics]))
        for key in metrics[0]
    }


def _run_contract(
    *,
    ablation: str,
    fold: UnityFold,
    seed: int,
    output_root: Path,
) -> UnityFineTuneRun:
    run_root = (
        Path(output_root)
        / f"fold_{fold.name}"
        / ablation
        / f"seed_{seed}"
    )
    return UnityFineTuneRun(
        ablation=ablation,
        fold=fold.name,
        seed=seed,
        train_sequence=fold.train_sequence,
        test_sequence=fold.test_sequence,
        run_root=run_root,
        final_checkpoint=run_root / "final.pt",
        metrics_path=run_root / "history.json",
        provenance_path=run_root / "provenance.json",
    )


def run_supervised_finetune(
    train_sequence: UnitySupervisedSequence,
    *,
    ablation: str,
    fold: UnityFold,
    seed: int,
    source_checkpoint: Path,
    skeleton_path: Path,
    output_root: Path,
    config: UnityFineTuneConfig,
    loss_config: UnitySupervisedLossConfig,
    self_supervised_config: LossConfig,
    corruption_config: CorruptionConfig,
) -> UnityFineTuneRun:
    """Fine-tune one cell without ever receiving held-out or static data."""
    if train_sequence.sequence_id != fold.train_sequence:
        raise ValueError("training sequence does not match fold")
    if len(train_sequence.sample_ids) != 97:
        raise ValueError("training sequence must contain exactly 97 frames")
    if train_sequence.sequence_id == "static_sweep":
        raise ValueError("static sequence cannot be used for training")
    source_checkpoint = Path(source_checkpoint)
    source_sha256 = _sha256_file(source_checkpoint)
    loaded = load_rotation_aware_model(
        source_checkpoint, skeleton_path, config.device
    )
    if loaded.ablation != ablation:
        raise ValueError(
            "checkpoint ablation mismatch: "
            f"requested {ablation}, source is {loaded.ablation}"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    skeleton = loaded.skeleton
    dataset = UnitySupervisedWindowDataset(
        train_sequence,
        skeleton_path=skeleton_path,
        length=config.window_length,
        stride=config.train_stride,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_pose_pair_windows,
    )
    optimizer = torch.optim.AdamW(
        loaded.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    run = _run_contract(
        ablation=ablation,
        fold=fold,
        seed=seed,
        output_root=output_root,
    )
    resolved = _resolved_config(
        ablation=ablation,
        fold=fold,
        seed=seed,
        source_checkpoint=source_checkpoint,
        skeleton_path=skeleton_path,
        config=config,
        loss_config=loss_config,
        self_supervised_config=self_supervised_config,
        corruption_config=corruption_config,
    )
    _atomic_yaml(run.run_root / "resolved_config.yaml", resolved)
    history: list[dict[str, object]] = []
    for epoch in range(config.epochs):
        values = train_supervised_epoch(
            loaded.model,
            loader,
            optimizer,
            skeleton,
            loss_config=loss_config,
            self_supervised_config=self_supervised_config,
            corruption_config=corruption_config,
            seed=seed,
            epoch=epoch,
            device=config.device,
        )
        history.append({"epoch": epoch + 1, **values})
        _atomic_json(run.metrics_path, history)
    source_payload = torch.load(
        source_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_provenance = {
        **dict(loaded.provenance),
        "unity_gt_supervision": True,
        "static_excluded_from_training": True,
        "fold": fold.name,
        "train_sequence": fold.train_sequence,
        "test_sequence": fold.test_sequence,
        "seed": seed,
        "ablation": ablation,
        "source_checkpoint_sha256": source_sha256,
    }
    payload = {
        **dict(source_payload),
        "model": loaded.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_config": asdict(self_supervised_config),
        "corruption_config": asdict(corruption_config),
        "training_config": {
            **dict(source_payload["training_config"]),
            "ablation": ablation,
            "hidden_channels": loaded.hidden_channels,
            "unity_supervised": True,
            "epochs": config.epochs,
            "seed": seed,
            "fold": fold.name,
        },
        "provenance": checkpoint_provenance,
        "score": float(history[-1]["total_loss"]),
    }
    _atomic_torch_save(run.final_checkpoint, payload)
    metadata = train_sequence.raw_trial.source_metadata
    unity_manifest_sha256 = str(metadata.get("unity_manifest_sha256", ""))
    sam3d_cache_identity = str(metadata.get("sam3d_cache_identity", ""))
    if len(unity_manifest_sha256) != 64 or not sam3d_cache_identity:
        raise ValueError("training sequence lacks Unity/SAM3D source identity")
    provenance = {
        **checkpoint_provenance,
        "final_checkpoint_sha256": _sha256_file(run.final_checkpoint),
        "git_commit": _git_commit(),
        "unity_manifest_sha256": unity_manifest_sha256,
        "sam3d_cache_identity": sam3d_cache_identity,
        "resolved_config": resolved,
        "history_epochs": len(history),
    }
    _atomic_json(run.provenance_path, provenance)
    return run


def validate_completed_run(
    run: UnityFineTuneRun,
    *,
    source_checkpoint_sha256: str,
    resolved_config: Mapping[str, object],
    unity_manifest_sha256: str,
) -> bool:
    """Return true only for a complete, identity-matching final-epoch run."""
    resolved_path = run.run_root / "resolved_config.yaml"
    required = (
        run.final_checkpoint,
        run.metrics_path,
        run.provenance_path,
        resolved_path,
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        provenance = json.loads(
            run.provenance_path.read_text(encoding="utf-8")
        )
        history = json.loads(run.metrics_path.read_text(encoding="utf-8"))
        stored_resolved = yaml.safe_load(
            resolved_path.read_text(encoding="utf-8")
        )
        epochs = int(stored_resolved["training"]["epochs"])
        checkpoint = torch.load(
            run.final_checkpoint, map_location="cpu", weights_only=False
        )
        training = checkpoint["training_config"]
        return bool(
            provenance["ablation"] == run.ablation
            and provenance["fold"] == run.fold
            and provenance["seed"] == run.seed
            and provenance["train_sequence"] == run.train_sequence
            and provenance["test_sequence"] == run.test_sequence
            and provenance["static_excluded_from_training"] is True
            and provenance["unity_gt_supervision"] is True
            and provenance["source_checkpoint_sha256"]
            == source_checkpoint_sha256
            and provenance["unity_manifest_sha256"]
            == unity_manifest_sha256
            and provenance["resolved_config"] == dict(resolved_config)
            and stored_resolved == dict(resolved_config)
            and provenance["history_epochs"] == epochs
            and len(history) == epochs
            and provenance["final_checkpoint_sha256"]
            == _sha256_file(run.final_checkpoint)
            and training["ablation"] == run.ablation
            and training["fold"] == run.fold
            and int(training["seed"]) == run.seed
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ):
        return False


def discover_completed_runs(
    output_root: Path,
    *,
    expected_cells: Sequence[tuple[str, str, int]],
    resolved_config: Mapping[str, object],
) -> tuple[UnityFineTuneRun, ...]:
    """Discover only configured matrix cells in stable fold/ablation/seed order."""
    unique = set(expected_cells)
    if len(unique) != len(expected_cells):
        raise ValueError("duplicate expected Unity supervised matrix cell")
    runs: list[UnityFineTuneRun] = []
    expected_roots: set[Path] = set()
    for ablation, fold_name, seed in sorted(
        unique, key=lambda item: (item[1], item[0], item[2])
    ):
        if fold_name not in {"left_to_right", "right_to_left"}:
            raise ValueError(f"unexpected fold identity: {fold_name}")
        if ablation not in {"A4", "A5", "A6", "A7", "A8", "A9"}:
            raise ValueError(f"unexpected ablation identity: {ablation}")
        if seed not in {0, 1, 2}:
            raise ValueError(f"unexpected seed identity: {seed}")
        train_sequence = (
            "continuous_left_060_r00"
            if fold_name == "left_to_right"
            else "continuous_right_060_r00"
        )
        test_sequence = (
            "continuous_right_060_r00"
            if fold_name == "left_to_right"
            else "continuous_left_060_r00"
        )
        run_root = (
            Path(output_root)
            / f"fold_{fold_name}"
            / ablation
            / f"seed_{seed}"
        )
        expected_roots.add(run_root.resolve())
        run = UnityFineTuneRun(
            ablation=ablation,
            fold=fold_name,
            seed=seed,
            train_sequence=train_sequence,
            test_sequence=test_sequence,
            run_root=run_root,
            final_checkpoint=run_root / "final.pt",
            metrics_path=run_root / "history.json",
            provenance_path=run_root / "provenance.json",
        )
        if not run.provenance_path.is_file():
            continue
        provenance = json.loads(
            run.provenance_path.read_text(encoding="utf-8")
        )
        stored_resolved = provenance.get("resolved_config")
        if resolved_config and isinstance(stored_resolved, Mapping):
            for key, value in resolved_config.items():
                if stored_resolved.get(key) != value:
                    raise ValueError(
                        f"resolved config mismatch for {fold_name}/{ablation}/{seed}"
                    )
        if not validate_completed_run(
            run,
            source_checkpoint_sha256=str(
                provenance.get("source_checkpoint_sha256", "")
            ),
            resolved_config=dict(stored_resolved or {}),
            unity_manifest_sha256=str(
                provenance.get("unity_manifest_sha256", "")
            ),
        ):
            continue
        runs.append(run)
    for final_path in Path(output_root).glob(
        "fold_*/*/seed_*/final.pt"
    ):
        if final_path.parent.resolve() not in expected_roots:
            raise ValueError(f"unexpected completed Unity run: {final_path.parent}")
    return tuple(runs)


def run_finetuned_inference(
    run: UnityFineTuneRun,
    sequences: Mapping[str, UnitySupervisedSequence],
    *,
    skeleton_path: Path,
    window_length: int,
    stride: int,
    device: str = "cpu",
) -> tuple[MethodSequence, ...]:
    """Infer only the held-out direction and static OOD diagnostic."""
    if device != "cpu":
        raise ValueError("fine-tuned Unity inference currently requires device='cpu'")
    if run.train_sequence in {
        run.test_sequence,
        "static_sweep",
    }:
        raise ValueError("training identity overlaps an evaluation sequence")
    selected = (run.test_sequence, "static_sweep")
    if any(sequence_id not in sequences for sequence_id in selected):
        raise ValueError("held-out or static Unity sequence is unavailable")
    provenance = json.loads(
        run.provenance_path.read_text(encoding="utf-8")
    )
    if provenance.get("fold") != run.fold:
        raise ValueError("completed run provenance does not match fold")
    loaded = load_rotation_aware_model(
        run.final_checkpoint, skeleton_path, device
    )
    if loaded.ablation != run.ablation:
        raise ValueError("fine-tuned checkpoint ablation mismatch")
    outputs: list[MethodSequence] = []
    for sequence_id in selected:
        sequence = sequences[sequence_id]
        with torch.inference_mode():
            result = run_inference(
                loaded.model,
                sequence.raw_trial,
                loaded.skeleton,
                output_root=run.run_root
                / "inference"
                / "_runtime"
                / sequence_id,
                run_id=(
                    f"unity_supervised_{run.fold}_"
                    f"{run.ablation.lower()}_seed{run.seed}"
                ),
                window_length=window_length,
                stride=stride,
                provenance={
                    **dict(loaded.provenance),
                    "ablation": run.ablation,
                    "checkpoint_path": str(run.final_checkpoint),
                    "checkpoint_sha256": loaded.checkpoint_sha256,
                    "model_config": {
                        "hidden_channels": loaded.hidden_channels
                    },
                },
                resolved_config=provenance["resolved_config"],
            )
        with np.load(result.sequence_path, allow_pickle=False) as data:
            points = np.asarray(data["kpts_world"], dtype=np.float32)
            valid = np.asarray(data["joint_valid"], dtype=bool)
        metadata = {
            "ranking_group": "unity_supervised",
            "unity_gt_used_for_training": True,
            "evaluation_gt_loaded_after_training": True,
            "fold": run.fold,
            "seed": run.seed,
            "ablation": run.ablation,
            "train_sequence": run.train_sequence,
            "test_sequence": run.test_sequence,
            "source_checkpoint_sha256": provenance[
                "source_checkpoint_sha256"
            ],
            "final_checkpoint_sha256": provenance[
                "final_checkpoint_sha256"
            ],
        }
        method_sequence = MethodSequence(
            method=run.ablation,
            sequence_id=sequence_id,
            sample_ids=sequence.sample_ids,
            points=points,
            valid=valid,
            joint_names=tuple(mhr_names),
            metadata=metadata,
        )
        _save_sequence(
            run.run_root / "inference" / f"{sequence_id}.npz",
            method_sequence,
        )
        outputs.append(method_sequence)
    return tuple(outputs)
