"""Self-supervised G-series training with fitted-camera input features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import subprocess
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from gymnastics.common.skeletons.mhr70 import mhr_names
from gymnastics.fusion.rotation_aware.camera import CameraFeatureBundle
from gymnastics.fusion.rotation_aware.config import (
    SkeletonSpec,
    load_skeleton_spec,
)
from gymnastics.fusion.rotation_aware.corruptions import CorruptionConfig
from gymnastics.fusion.rotation_aware.dataset import collate_pose_pair_windows
from gymnastics.fusion.rotation_aware.inference import (
    _forward,
    _starts,
    overlap_taper,
)
from gymnastics.fusion.rotation_aware.losses import LossConfig
from gymnastics.fusion.rotation_aware.model import RotationAwareFusionModel
from gymnastics.fusion.rotation_aware.training import train_one_epoch

from .camera_guided_data import (
    UnityCameraGuidedSequence,
    UnityCameraGuidedWindowDataset,
    camera_conditioning_config,
)
from .fusion import _save_sequence
from .schema import MethodSequence
from .supervised_data import UnityFold


@dataclass(frozen=True)
class CameraGuidedTrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    window_length: int = 128
    train_stride: int = 32
    batch_size: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            self.epochs < 1
            or self.window_length < 1
            or self.train_stride < 1
            or self.batch_size < 1
        ):
            raise ValueError("camera-guided counts must be positive")
        if (
            not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not np.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("camera-guided optimizer settings are invalid")
        if not self.device:
            raise ValueError("camera-guided device is required")


@dataclass(frozen=True)
class CameraGuidedRun:
    ablation: str
    fold: str
    seed: int
    train_sequence: str
    test_sequence: str
    run_root: Path
    final_checkpoint: Path
    history_path: Path
    provenance_path: Path


@dataclass(frozen=True)
class LoadedCameraGuided:
    model: RotationAwareFusionModel
    skeleton: SkeletonSpec
    ablation: str
    hidden_channels: int
    provenance: Mapping[str, object]


def _sha256(path: Path) -> str:
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


def _run_contract(
    *,
    output_root: Path,
    ablation: str,
    fold: UnityFold,
    seed: int,
) -> CameraGuidedRun:
    run_root = (
        Path(output_root)
        / f"fold_{fold.name}"
        / ablation
        / f"seed_{seed}"
    )
    return CameraGuidedRun(
        ablation=ablation,
        fold=fold.name,
        seed=seed,
        train_sequence=fold.train_sequence,
        test_sequence=fold.test_sequence,
        run_root=run_root,
        final_checkpoint=run_root / "final.pt",
        history_path=run_root / "history.json",
        provenance_path=run_root / "provenance.json",
    )


def _source_configs(
    source_checkpoint: Path,
) -> tuple[dict[str, object], LossConfig, CorruptionConfig]:
    payload = torch.load(
        source_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict):
        raise ValueError("A6 source checkpoint must be a mapping")
    training = payload.get("training_config")
    if not isinstance(training, Mapping) or training.get("ablation") != "A6":
        raise ValueError("camera-guided source checkpoint must be A6")
    loss = payload.get("loss_config")
    corruption = payload.get("corruption_config")
    if not isinstance(loss, Mapping) or not isinstance(corruption, Mapping):
        raise ValueError("A6 source checkpoint has incomplete objective metadata")
    return payload, LossConfig(**dict(loss)), CorruptionConfig(**dict(corruption))


def _expanded_model(
    source_payload: Mapping[str, object],
    skeleton: SkeletonSpec,
    ablation: str,
) -> tuple[RotationAwareFusionModel, int]:
    training = source_payload["training_config"]
    if not isinstance(training, Mapping):
        raise ValueError("source training configuration is unavailable")
    hidden_channels = int(training.get("hidden_channels", 128))
    model = RotationAwareFusionModel(
        skeleton,
        hidden_channels=hidden_channels,
        camera_config=camera_conditioning_config(ablation),
    )
    state = source_payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("A6 source checkpoint has no model state")
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected_prefix = "camera_conditioner."
    if unexpected or any(
        not name.startswith(expected_prefix) for name in missing
    ):
        raise ValueError(
            "A6 source state is incompatible with camera-guided expansion"
        )
    if ablation == "G0" and missing:
        raise ValueError("G0 must exactly load the A6 source state")
    if ablation != "G0" and not missing:
        raise ValueError("camera-guided model did not create camera parameters")
    return model, hidden_channels


def train_camera_guided_run(
    train_sequence: UnityCameraGuidedSequence,
    *,
    ablation: str,
    fold: UnityFold,
    seed: int,
    source_checkpoint: Path,
    skeleton_path: Path,
    output_root: Path,
    config: CameraGuidedTrainingConfig,
) -> CameraGuidedRun:
    """Train one G cell without accepting any 3D reference argument."""
    if train_sequence.sequence_id != fold.train_sequence:
        raise ValueError("camera-guided training sequence does not match fold")
    if train_sequence.ablation != ablation:
        raise ValueError("camera-guided sequence ablation does not match run")
    source_checkpoint = Path(source_checkpoint)
    source_payload, loss_config, corruption_config = _source_configs(
        source_checkpoint
    )
    skeleton = load_skeleton_spec(Path(skeleton_path))
    model, hidden_channels = _expanded_model(
        source_payload, skeleton, ablation
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = UnityCameraGuidedWindowDataset(
        train_sequence,
        skeleton_path=Path(skeleton_path),
        length=config.window_length,
        stride=config.train_stride,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_pose_pair_windows,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    run = _run_contract(
        output_root=Path(output_root),
        ablation=ablation,
        fold=fold,
        seed=seed,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            skeleton,
            loss_config=loss_config,
            corruption_config=corruption_config,
            seed=seed,
            epoch=epoch,
            device=config.device,
        )
        history.append({"epoch": epoch + 1, **metrics})
        _atomic_json(run.history_path, history)
    camera = train_sequence.fitted_camera
    camera_payload = (
        None
        if camera is None
        else {
            "rotation_face_to_side": camera.rotation_face_to_side.tolist(),
            "translation_direction_face_to_side": (
                camera.translation_direction_face_to_side.tolist()
            ),
            "inlier_ratio": camera.inlier_ratio,
            "holdout_reprojection_px": camera.holdout_reprojection_px,
            "fit_sample_ids": camera.fit_sample_ids.tolist(),
        }
    )
    resolved_config = {
        "ablation": ablation,
        "fold": asdict(fold),
        "seed": int(seed),
        "training": asdict(config),
        "hidden_channels": hidden_channels,
        "camera_conditioning": (
            None
            if model.camera_config is None
            else asdict(model.camera_config)
        ),
        "self_supervised_loss": asdict(loss_config),
        "corruption": asdict(corruption_config),
    }
    provenance: dict[str, object] = {
        "ablation": ablation,
        "fold": fold.name,
        "seed": int(seed),
        "train_sequence": fold.train_sequence,
        "test_sequence": fold.test_sequence,
        "train_sample_ids": train_sequence.sample_ids.tolist(),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "git_commit": _git_commit(),
        "unity_native_3d_available_to_training": False,
        "triangulated_3d_available_to_training": False,
        "training_inputs": ["sam3d_3d", "sam3d_2d", "fitted_camera"],
        "fitted_camera": camera_payload,
        "resolved_config": resolved_config,
    }
    checkpoint = {
        "model": model.state_dict(),
        "training_config": {
            "ablation": ablation,
            "hidden_channels": hidden_channels,
            "camera_config": (
                None
                if model.camera_config is None
                else asdict(model.camera_config)
            ),
        },
        "loss_config": asdict(loss_config),
        "corruption_config": asdict(corruption_config),
        "provenance": provenance,
        "source_checkpoint_sha256": provenance["source_checkpoint_sha256"],
        "history": history,
    }
    _atomic_torch(run.final_checkpoint, checkpoint)
    provenance["final_checkpoint_sha256"] = _sha256(run.final_checkpoint)
    _atomic_json(run.provenance_path, provenance)
    return run


def load_camera_guided_model(
    checkpoint: Path,
    skeleton_path: Path,
    *,
    device: str,
) -> LoadedCameraGuided:
    payload = torch.load(
        Path(checkpoint), map_location="cpu", weights_only=False
    )
    if not isinstance(payload, Mapping):
        raise ValueError("camera-guided checkpoint must be a mapping")
    training = payload.get("training_config")
    provenance = payload.get("provenance")
    if not isinstance(training, Mapping) or not isinstance(
        provenance, Mapping
    ):
        raise ValueError("camera-guided checkpoint metadata is incomplete")
    ablation = str(training.get("ablation", ""))
    hidden_channels = int(training.get("hidden_channels", 0))
    expected = camera_conditioning_config(ablation)
    stored_camera = training.get("camera_config")
    expected_camera = None if expected is None else asdict(expected)
    if stored_camera != expected_camera:
        raise ValueError("camera-guided checkpoint feature schema mismatch")
    skeleton = load_skeleton_spec(Path(skeleton_path))
    model = RotationAwareFusionModel(
        skeleton,
        hidden_channels=hidden_channels,
        camera_config=expected,
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return LoadedCameraGuided(
        model=model,
        skeleton=skeleton,
        ablation=ablation,
        hidden_channels=hidden_channels,
        provenance=MappingProxyType(dict(provenance)),
    )


def _predict_sequence(
    model: RotationAwareFusionModel,
    sequence: UnityCameraGuidedSequence,
    skeleton: SkeletonSpec,
    *,
    window_length: int,
    stride: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    source = sequence.canonical_trial.trial
    frames, joints = source.face.shape[:2]
    weights = overlap_taper(window_length)
    fused_sum = np.zeros((frames, joints, 3), dtype=np.float64)
    point_weight = np.zeros((frames, joints), dtype=np.float64)
    for start in _starts(frames, window_length, stride):
        count = min(window_length, frames - start)
        face = torch.zeros(
            (1, window_length, joints, 3),
            dtype=torch.float32,
            device=device,
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
        if sequence.camera_features is not None:
            features = sequence.camera_features
            joint = torch.zeros(
                (1, window_length, joints, 8),
                dtype=torch.float32,
                device=device,
            )
            camera_valid = torch.zeros(
                (1, window_length, joints),
                dtype=torch.bool,
                device=device,
            )
            joint[:, :count] = torch.from_numpy(
                np.array(
                    features.joint_features[start : start + count],
                    copy=True,
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
                model,
                face,
                side,
                valid_face,
                valid_side,
                skeleton,
                dt,
                camera_features=camera_bundle,
            )
        usable = output.valid[0, :count].cpu().numpy()
        weight = weights[:count, None] * usable
        fused_sum[start : start + count] += (
            output.fused_kpts[0, :count].cpu().numpy() * weight[..., None]
        )
        point_weight[start : start + count] += weight
    fused = (
        fused_sum / np.maximum(point_weight[..., None], 1e-12)
    ).astype(np.float32)
    valid = point_weight > 0
    fused[~valid] = 0.0
    world = sequence.canonical_trial.restore_face(fused)
    world[~valid] = 0.0
    return world, valid


def run_camera_guided_inference(
    run: CameraGuidedRun,
    sequences: Mapping[str, UnityCameraGuidedSequence],
    *,
    skeleton_path: Path,
    window_length: int,
    stride: int,
    device: str = "cpu",
) -> tuple[MethodSequence, ...]:
    """Infer the held-out direction and static diagnostic after training."""
    if run.train_sequence in {run.test_sequence, "static_sweep"}:
        raise ValueError("camera-guided run has evaluation leakage")
    loaded = load_camera_guided_model(
        run.final_checkpoint, skeleton_path, device=device
    )
    if loaded.ablation != run.ablation:
        raise ValueError("camera-guided checkpoint ablation mismatch")
    outputs: list[MethodSequence] = []
    for sequence_id in (run.test_sequence, "static_sweep"):
        sequence = sequences[sequence_id]
        if sequence.ablation != run.ablation:
            raise ValueError("inference feature ablation mismatch")
        points, valid = _predict_sequence(
            loaded.model,
            sequence,
            loaded.skeleton,
            window_length=window_length,
            stride=stride,
            device=device,
        )
        method = MethodSequence(
            method=run.ablation,
            sequence_id=sequence_id,
            sample_ids=sequence.sample_ids,
            points=points,
            valid=valid,
            joint_names=tuple(mhr_names),
            metadata={
                "ranking_group": "camera_feature_self_supervised",
                "fold": run.fold,
                "seed": run.seed,
                "train_sequence": run.train_sequence,
                "test_sequence": run.test_sequence,
                "unity_gt_used_for_training": False,
                "fitted_camera_used_at_inference": run.ablation != "G0",
                "checkpoint_sha256": _sha256(run.final_checkpoint),
            },
        )
        _save_sequence(
            run.run_root / "inference" / f"{sequence_id}.npz", method
        )
        outputs.append(method)
    return tuple(outputs)
