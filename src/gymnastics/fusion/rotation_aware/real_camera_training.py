"""Frozen-backbone fitted-camera training for the collected dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .camera import CameraConditioningConfig, CameraFeatureBundle
from .config import SkeletonSpec, load_skeleton_spec
from .corruptions import CorruptionConfig
from .dataset import (
    PosePairWindowDataset,
    SplitManifest,
    WindowConfig,
    collate_pose_pair_windows,
)
from .inference import _forward, _starts, overlap_taper
from .losses import LossConfig
from .model import RotationAwareFusionModel
from .real_camera_data import (
    CAMERA_ABLATIONS,
    CameraWindowDataset,
    RealCameraTrial,
)
from .training import train_one_epoch, validate


CAMERA_PARAMETER_PREFIXES = ("camera_conditioner.", "camera_delta_head.")


def camera_conditioning_config(
    ablation: str,
) -> CameraConditioningConfig | None:
    if ablation not in CAMERA_ABLATIONS:
        raise ValueError(f"Unsupported camera ablation: {ablation}")
    if ablation == "G0":
        return None
    return CameraConditioningConfig(
        global_channels=19,
        joint_channels=8,
        mode="film" if ablation in {"G4", "G5"} else "additive",
    )


@dataclass(frozen=True)
class RealCameraTrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    window_length: int = 128
    train_stride: int = 32
    eval_stride: int = 64
    batch_size: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(
            self.epochs,
            self.window_length,
            self.train_stride,
            self.eval_stride,
            self.batch_size,
        ) < 1:
            raise ValueError("Training counts must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative and finite")
        if not self.device:
            raise ValueError("device is required")


@dataclass(frozen=True)
class ExpandedFrozenModel:
    model: RotationAwareFusionModel
    skeleton: SkeletonSpec
    source_payload: Mapping[str, Any]
    loss_config: LossConfig
    corruption_config: CorruptionConfig
    hidden_channels: int
    trainable_parameter_prefixes: tuple[str, ...]


@dataclass(frozen=True)
class RealCameraRun:
    ablation: str
    seed: int
    run_root: Path
    checkpoint: Path
    history_path: Path
    provenance_path: Path


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _load_source(
    source_checkpoint: str | Path,
) -> tuple[dict[str, Any], LossConfig, CorruptionConfig]:
    payload = torch.load(
        Path(source_checkpoint), map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict):
        raise ValueError("A6 source checkpoint must be a mapping")
    training = payload.get("training_config")
    if not isinstance(training, Mapping) or training.get("ablation") != "A6":
        raise ValueError("Real camera training requires an A6 source checkpoint")
    loss = payload.get("loss_config")
    corruption = payload.get("corruption_config")
    if not isinstance(loss, Mapping) or not isinstance(corruption, Mapping):
        raise ValueError("A6 source objective metadata is incomplete")
    return payload, LossConfig(**dict(loss)), CorruptionConfig(**dict(corruption))


def expand_and_freeze_camera_model(
    source_checkpoint: str | Path,
    *,
    skeleton_path: str | Path,
    ablation: str,
    seed: int,
) -> ExpandedFrozenModel:
    """Expand A6 with a deterministic camera branch and freeze the backbone."""

    if ablation not in CAMERA_ABLATIONS:
        raise ValueError(f"Unsupported camera ablation: {ablation}")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    source, loss, corruption = _load_source(source_checkpoint)
    training = source["training_config"]
    hidden_channels = int(training.get("hidden_channels", 128))
    skeleton = load_skeleton_spec(Path(skeleton_path))
    model = RotationAwareFusionModel(
        skeleton,
        hidden_channels=hidden_channels,
        camera_config=camera_conditioning_config(ablation),
    )
    state = source.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("A6 source checkpoint contains no model state")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(
        not name.startswith(CAMERA_PARAMETER_PREFIXES) for name in missing
    ):
        raise ValueError("A6 source is incompatible with camera expansion")
    if ablation == "G0" and missing:
        raise ValueError("G0 must exactly match the source A6 model")
    if ablation != "G0" and not missing:
        raise ValueError("Camera expansion created no camera parameters")

    prefixes = () if ablation == "G0" else CAMERA_PARAMETER_PREFIXES
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    return ExpandedFrozenModel(
        model=model,
        skeleton=skeleton,
        source_payload=MappingProxyType(source),
        loss_config=loss,
        corruption_config=corruption,
        hidden_channels=hidden_channels,
        trainable_parameter_prefixes=prefixes,
    )


def _run_contract(
    output_root: str | Path, ablation: str, seed: int
) -> RealCameraRun:
    run_root = Path(output_root) / ablation / f"seed_{int(seed)}"
    return RealCameraRun(
        ablation=ablation,
        seed=int(seed),
        run_root=run_root,
        checkpoint=run_root / "best.pt",
        history_path=run_root / "history.json",
        provenance_path=run_root / "provenance.json",
    )


def _people(trials: Sequence[RealCameraTrial]) -> tuple[str, ...]:
    return tuple(
        sorted({str(trial.canonical_trial.trial.person_id) for trial in trials})
    )


def _validate_trial_partition(
    train_trials: Sequence[RealCameraTrial],
    val_trials: Sequence[RealCameraTrial],
    ablation: str,
) -> None:
    if not train_trials or not val_trials:
        raise ValueError("Training and validation trials must be non-empty")
    if any(trial.ablation != ablation for trial in (*train_trials, *val_trials)):
        raise ValueError("Trial camera ablation does not match the run")
    overlap = set(_people(train_trials)) & set(_people(val_trials))
    if overlap:
        raise ValueError(f"Train/validation people overlap: {sorted(overlap)}")


def _loaders(
    train_trials: Sequence[RealCameraTrial],
    val_trials: Sequence[RealCameraTrial],
    expanded: ExpandedFrozenModel,
    config: RealCameraTrainingConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_people = _people(train_trials)
    val_people = _people(val_trials)
    window_config = WindowConfig(
        length=config.window_length,
        train_stride=config.train_stride,
        eval_stride=config.eval_stride,
    )
    train_base = PosePairWindowDataset(
        [trial.canonical_trial.trial for trial in train_trials],
        skeleton=expanded.skeleton,
        manifest=SplitManifest(train=train_people, val=(), test=()),
        split="train",
        config=window_config,
    )
    val_base = PosePairWindowDataset(
        [trial.canonical_trial.trial for trial in val_trials],
        skeleton=expanded.skeleton,
        manifest=SplitManifest(train=(), val=val_people, test=()),
        split="val",
        config=window_config,
    )
    generator = torch.Generator().manual_seed(int(seed))
    common = {"collate_fn": collate_pose_pair_windows, "num_workers": 0}
    return (
        DataLoader(
            CameraWindowDataset(train_base, train_trials),
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            **common,
        ),
        DataLoader(
            CameraWindowDataset(val_base, val_trials),
            batch_size=1,
            shuffle=False,
            **common,
        ),
    )


def train_real_camera_cell(
    *,
    train_trials: Sequence[RealCameraTrial],
    val_trials: Sequence[RealCameraTrial],
    ablation: str,
    seed: int,
    source_checkpoint: str | Path,
    skeleton_path: str | Path,
    output_root: str | Path,
    config: RealCameraTrainingConfig,
) -> RealCameraRun:
    """Train only the fitted-camera branch; no test data or pseudo-GT is accepted."""

    _validate_trial_partition(train_trials, val_trials, ablation)
    expanded = expand_and_freeze_camera_model(
        source_checkpoint,
        skeleton_path=skeleton_path,
        ablation=ablation,
        seed=seed,
    )
    run = _run_contract(output_root, ablation, seed)
    source_checkpoint = Path(source_checkpoint)
    provenance: dict[str, Any] = {
        "ablation": ablation,
        "seed": int(seed),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "train_people": list(_people(train_trials)),
        "validation_people": list(_people(val_trials)),
        "test_people_available_to_training": False,
        "triangulated_3d_available_to_training": False,
        "camera_fit_scope": "per_person_transductive_input_operation",
        "training_inputs": ["sam3d_3d", "sam3d_2d", "fitted_camera"],
        "camera_adaptation_objective": (
            "window_self_supervision_with_frozen_A6_rotation_prior"
        ),
        "trainable_parameter_prefixes": list(
            expanded.trainable_parameter_prefixes
        ),
        "resolved_training_config": asdict(config),
    }
    history: list[dict[str, Any]] = []

    if ablation == "G0":
        checkpoint = {
            "model": _cpu_state_dict(expanded.model),
            "training_config": {
                "ablation": ablation,
                "hidden_channels": expanded.hidden_channels,
                "camera_config": None,
            },
            "loss_config": asdict(expanded.loss_config),
            "corruption_config": asdict(expanded.corruption_config),
            "provenance": provenance,
            "trainable_parameter_prefixes": [],
            "optimizer_steps": 0,
            "history": history,
            "score": expanded.source_payload.get("score"),
        }
        _atomic_torch(run.checkpoint, checkpoint)
        provenance["checkpoint_sha256"] = _sha256(run.checkpoint)
        _atomic_json(run.history_path, history)
        _atomic_json(run.provenance_path, provenance)
        return run

    train_loader, val_loader = _loaders(
        train_trials, val_trials, expanded, config, seed
    )
    adaptation_loss = replace(
        expanded.loss_config, complete_cycle_rom_weight=0.0
    )
    trainable = [
        parameter
        for parameter in expanded.model.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Camera cell has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    optimizer_steps = 0
    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(
            expanded.model,
            train_loader,
            optimizer,
            expanded.skeleton,
            loss_config=adaptation_loss,
            corruption_config=expanded.corruption_config,
            seed=seed,
            epoch=epoch,
            device=config.device,
        )
        validation = validate(
            expanded.model,
            val_loader,
            expanded.skeleton,
            loss_config=adaptation_loss,
            corruption_config=expanded.corruption_config,
            seed=seed,
            device=config.device,
        )
        optimizer_steps += len(train_loader)
        score = float(validation["score"])
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation,
            }
        )
        _atomic_json(run.history_path, history)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_state = _cpu_state_dict(expanded.model)
    if best_state is None:
        raise FloatingPointError("No finite validation checkpoint was produced")

    checkpoint = {
        "model": best_state,
        "training_config": {
            "ablation": ablation,
            "hidden_channels": expanded.hidden_channels,
            "camera_config": asdict(camera_conditioning_config(ablation)),
        },
        "loss_config": asdict(adaptation_loss),
        "corruption_config": asdict(expanded.corruption_config),
        "provenance": provenance,
        "trainable_parameter_prefixes": list(
            expanded.trainable_parameter_prefixes
        ),
        "optimizer_steps": optimizer_steps,
        "history": history,
        "score": best_score,
    }
    _atomic_torch(run.checkpoint, checkpoint)
    provenance["checkpoint_sha256"] = _sha256(run.checkpoint)
    provenance["best_validation_score"] = best_score
    _atomic_json(run.provenance_path, provenance)
    return run


@dataclass(frozen=True)
class LoadedRealCameraModel:
    model: RotationAwareFusionModel
    skeleton: SkeletonSpec
    ablation: str
    provenance: Mapping[str, Any]


def load_real_camera_model(
    checkpoint: str | Path,
    *,
    skeleton_path: str | Path,
    device: str,
) -> LoadedRealCameraModel:
    payload = torch.load(
        Path(checkpoint), map_location="cpu", weights_only=False
    )
    training = payload.get("training_config")
    provenance = payload.get("provenance")
    if not isinstance(training, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("Real camera checkpoint metadata is incomplete")
    ablation = str(training.get("ablation", ""))
    skeleton = load_skeleton_spec(Path(skeleton_path))
    model = RotationAwareFusionModel(
        skeleton,
        hidden_channels=int(training.get("hidden_channels", 128)),
        camera_config=camera_conditioning_config(ablation),
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return LoadedRealCameraModel(
        model=model,
        skeleton=skeleton,
        ablation=ablation,
        provenance=MappingProxyType(dict(provenance)),
    )


def _predict_trial(
    loaded: LoadedRealCameraModel,
    trial: RealCameraTrial,
    *,
    window_length: int,
    stride: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    source = trial.canonical_trial.trial
    frames, joints = source.face.shape[:2]
    taper = overlap_taper(window_length)
    fused_sum = np.zeros((frames, joints, 3), dtype=np.float64)
    point_weight = np.zeros((frames, joints), dtype=np.float64)
    for start in _starts(frames, window_length, stride):
        count = min(window_length, frames - start)
        face = torch.zeros(
            (1, window_length, joints, 3), dtype=torch.float32, device=device
        )
        side = torch.zeros_like(face)
        valid_face = torch.zeros(
            (1, window_length, joints), dtype=torch.bool, device=device
        )
        valid_side = torch.zeros_like(valid_face)
        face[:, :count] = torch.from_numpy(
            np.array(source.face[start : start + count], copy=True)
        ).to(device)
        side[:, :count] = torch.from_numpy(
            np.array(source.side[start : start + count], copy=True)
        ).to(device)
        valid_face[:, :count] = torch.from_numpy(
            np.array(source.valid_face[start : start + count], copy=True)
        ).to(device)
        valid_side[:, :count] = torch.from_numpy(
            np.array(source.valid_side[start : start + count], copy=True)
        ).to(device)
        dt = torch.zeros(
            (1, window_length), dtype=torch.float32, device=device
        )
        dt[:, :count] = 1.0 / source.fps
        camera_bundle = None
        if trial.camera_features is not None:
            features = trial.camera_features
            joint = torch.zeros(
                (1, window_length, joints, 8),
                dtype=torch.float32,
                device=device,
            )
            camera_valid = torch.zeros(
                (1, window_length, joints), dtype=torch.bool, device=device
            )
            joint[:, :count] = torch.from_numpy(
                np.array(
                    features.joint_features[start : start + count], copy=True
                )
            ).to(device)
            camera_valid[:, :count] = torch.from_numpy(
                np.array(features.valid[start : start + count], copy=True)
            ).to(device)
            camera_bundle = CameraFeatureBundle(
                global_features=torch.from_numpy(
                    np.array(features.global_features[None], copy=True)
                ).to(device),
                joint_features=joint,
                valid=camera_valid,
            )
        with torch.inference_mode():
            output, _, _ = _forward(
                loaded.model,
                face,
                side,
                valid_face,
                valid_side,
                loaded.skeleton,
                dt,
                camera_features=camera_bundle,
            )
        usable = output.valid[0, :count].cpu().numpy()
        weights = taper[:count, None] * usable
        fused_sum[start : start + count] += (
            output.fused_kpts[0, :count].cpu().numpy() * weights[..., None]
        )
        point_weight[start : start + count] += weights
    fused = (
        fused_sum / np.maximum(point_weight[..., None], 1e-12)
    ).astype(np.float32)
    valid = point_weight > 0
    fused[~valid] = 0.0
    world = trial.canonical_trial.restore_face(fused)
    world[~valid] = 0.0
    return world, valid


def infer_real_camera_cell(
    run: RealCameraRun,
    *,
    test_trials: Sequence[RealCameraTrial],
    skeleton_path: str | Path,
    window_length: int = 128,
    stride: int = 64,
    device: str = "cpu",
) -> tuple[Path, ...]:
    """Infer held-out people and write one target-free NPZ per cycle."""

    loaded = load_real_camera_model(
        run.checkpoint, skeleton_path=skeleton_path, device=device
    )
    if loaded.ablation != run.ablation:
        raise ValueError("Run/checkpoint ablation mismatch")
    if any(trial.ablation != run.ablation for trial in test_trials):
        raise ValueError("Test feature ablation does not match checkpoint")
    train_people = set(loaded.provenance.get("train_people", ()))
    val_people = set(loaded.provenance.get("validation_people", ()))
    test_people = {
        trial.canonical_trial.trial.person_id for trial in test_trials
    }
    if test_people & (train_people | val_people):
        raise ValueError("Inference people overlap the training partition")

    outputs: list[Path] = []
    for trial in test_trials:
        source = trial.canonical_trial.trial
        points, valid = _predict_trial(
            loaded,
            trial,
            window_length=window_length,
            stride=stride,
            device=device,
        )
        metadata = {
            "ablation": run.ablation,
            "seed": run.seed,
            "person_id": source.person_id,
            "trial_id": source.trial_id,
            "no_pseudo_gt_training": True,
            "camera_fit_scope": loaded.provenance.get("camera_fit_scope"),
            "checkpoint_sha256": _sha256(run.checkpoint),
        }
        target = (
            run.run_root
            / "inference"
            / f"person_{source.person_id}"
            / source.trial_id
            / "fused_sequence.npz"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            kpts_world=points,
            joint_valid=valid,
            frame_valid=valid.any(axis=-1),
            face_map=source.face_map,
            side_map=source.side_map,
            timestamps=source.timestamps,
            fps=np.asarray(source.fps, dtype=np.float64),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        outputs.append(target)
    return tuple(outputs)
