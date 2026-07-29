"""Leakage-safe training for calibrated learned Unity baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gymnastics.common.skeletons.mhr70 import mhr_names

from .dataset import group_evaluation_sequences
from .extrinsic_models import (
    CalibratedPrediction,
    ExtrinsicGateModel,
    ExtrinsicResidualTCN,
    LearnableTriangulationModel,
    relative_camera_rotation,
)
from .fusion import _save_sequence
from .geometry import pixel_projection
from .mapping import select_unity_evaluation_joints
from .sam3d import load_sam3d_camera_cache
from .schema import MethodSequence, UnityBenchmark
from .supervised_data import UnityFold
from .supervised_loss import (
    apply_torch_sim3,
    masked_window_sim3,
    torch_map_mhr70_to_unity16,
)


EXTRINSIC_METHODS = (
    "extrinsic_gate",
    "extrinsic_residual_tcn",
    "learnable_triangulation",
)
FUSION_3D_METHODS = ("extrinsic_gate", "extrinsic_residual_tcn")


def _readonly(value: np.ndarray, *, dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ExtrinsicSequence:
    """One synchronized calibrated input sequence with Unity16 targets."""

    sequence_id: str
    sample_ids: np.ndarray
    face_3d: np.ndarray
    side_3d: np.ndarray
    valid_face_3d: np.ndarray
    valid_side_3d: np.ndarray
    pixels_2d: np.ndarray
    valid_2d: np.ndarray
    gt_unity16_m: np.ndarray
    gt_valid: np.ndarray
    relative_rotation: np.ndarray
    projection: np.ndarray
    image_size: np.ndarray
    source_identity: Mapping[str, str]

    def __post_init__(self) -> None:
        sample_ids = np.asarray(self.sample_ids, dtype=np.int64)
        frames = len(sample_ids)
        expected = {
            "face_3d": ((frames, 70, 3), np.float32),
            "side_3d": ((frames, 70, 3), np.float32),
            "valid_face_3d": ((frames, 70), bool),
            "valid_side_3d": ((frames, 70), bool),
            "pixels_2d": ((frames, 2, 70, 2), np.float32),
            "valid_2d": ((frames, 2, 70), bool),
            "gt_unity16_m": ((frames, 16, 3), np.float32),
            "gt_valid": ((frames, 16), bool),
            "relative_rotation": ((3, 3), np.float32),
            "projection": ((2, 3, 4), np.float32),
            "image_size": ((2, 2), np.float32),
        }
        if not self.sequence_id:
            raise ValueError("sequence_id is required")
        if sample_ids.shape != (frames,) or len(set(sample_ids.tolist())) != frames:
            raise ValueError("sample_ids must be one-dimensional and unique")
        for name, (shape, dtype) in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if np.issubdtype(np.dtype(dtype), np.floating) and not np.isfinite(
                value
            ).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, _readonly(value, dtype=dtype))
        identity = {str(key): str(value) for key, value in self.source_identity.items()}
        if any(len(identity.get(key, "")) != 64 for key in ("manifest_sha256", "sam3d_sha256")):
            raise ValueError("source identity must contain two SHA-256 digests")
        object.__setattr__(self, "sample_ids", _readonly(sample_ids, dtype=np.int64))
        object.__setattr__(self, "source_identity", MappingProxyType(identity))

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence_id": self.sequence_id,
            "sample_ids": self.sample_ids,
            "face_3d": self.face_3d,
            "side_3d": self.side_3d,
            "valid_face_3d": self.valid_face_3d,
            "valid_side_3d": self.valid_side_3d,
            "pixels_2d": self.pixels_2d,
            "valid_2d": self.valid_2d,
            "gt_unity16_m": self.gt_unity16_m,
            "gt_valid": self.gt_valid,
            "relative_rotation": self.relative_rotation,
            "projection": self.projection,
            "image_size": self.image_size,
            "source_identity": self.source_identity,
        }


@dataclass(frozen=True)
class ExtrinsicTrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_channels: int = 32
    max_delta_m: float = 0.05
    smooth_l1_beta_m: float = 0.02
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.hidden_channels < 1:
            raise ValueError("epochs and hidden_channels must be positive")
        values = (
            self.learning_rate,
            self.weight_decay,
            self.max_delta_m,
            self.smooth_l1_beta_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("training values must be finite")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or self.max_delta_m < 0
            or self.smooth_l1_beta_m <= 0
        ):
            raise ValueError("training values are outside their valid ranges")
        if not self.device:
            raise ValueError("device is required")


@dataclass(frozen=True)
class ExtrinsicRun:
    method: str
    fold: str
    seed: int
    train_sequence: str
    test_sequence: str
    run_root: Path
    checkpoint_path: Path
    history_path: Path
    config_path: Path
    provenance_path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_paths(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(value) for value in paths):
        digest.update(str(path).encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_extrinsic_sequences(
    benchmark: UnityBenchmark,
    cache_root: Path,
) -> Mapping[str, ExtrinsicSequence]:
    """Join cached SAM3D inputs, exact cameras, and Unity-native targets."""
    rotation = relative_camera_rotation(
        benchmark.cameras["cam0"], benchmark.cameras["cam1"]
    )
    projection = np.stack(
        (
            pixel_projection(benchmark.cameras["cam0"]),
            pixel_projection(benchmark.cameras["cam1"]),
        )
    ).astype(np.float32)
    image_size = np.asarray(
        (
            benchmark.cameras["cam0"].image_size,
            benchmark.cameras["cam1"].image_size,
        ),
        dtype=np.float32,
    )
    manifest_sha256 = _sha256_file(benchmark.root / "manifest.jsonl")
    output: dict[str, ExtrinsicSequence] = {}
    for sequence_id, frames in group_evaluation_sequences(benchmark).items():
        sample_ids = np.asarray([frame.sample_id for frame in frames], dtype=np.int64)
        cam0 = load_sam3d_camera_cache(cache_root, "cam0", sample_ids)
        cam1 = load_sam3d_camera_cache(cache_root, "cam1", sample_ids)
        cache_paths = [
            Path(cache_root) / camera_id / f"{int(sample_id):08d}.npz"
            for camera_id in ("cam0", "cam1")
            for sample_id in sample_ids
        ]
        gt = select_unity_evaluation_joints(
            np.stack([frame.gt_world_m for frame in frames]),
            np.stack([frame.gt_available for frame in frames]),
        )
        output[sequence_id] = ExtrinsicSequence(
            sequence_id=sequence_id,
            sample_ids=sample_ids,
            face_3d=cam0.points_3d,
            side_3d=cam1.points_3d,
            valid_face_3d=cam0.valid_3d,
            valid_side_3d=cam1.valid_3d,
            pixels_2d=np.stack((cam0.points_2d, cam1.points_2d), axis=1),
            valid_2d=np.stack((cam0.valid_2d, cam1.valid_2d), axis=1),
            gt_unity16_m=gt.points,
            gt_valid=gt.valid,
            relative_rotation=rotation,
            projection=projection,
            image_size=image_size,
            source_identity={
                "manifest_sha256": manifest_sha256,
                "sam3d_sha256": _sha256_paths(cache_paths),
            },
        )
    return MappingProxyType(output)


def make_extrinsic_model(
    method: str,
    *,
    hidden_channels: int = 32,
    max_delta_m: float = 0.05,
) -> nn.Module:
    if method == "extrinsic_gate":
        return ExtrinsicGateModel(
            joint_count=70,
            pelvis_indices=(9, 10),
            hidden_channels=hidden_channels,
        )
    if method == "extrinsic_residual_tcn":
        return ExtrinsicResidualTCN(
            joint_count=70,
            pelvis_indices=(9, 10),
            hidden_channels=hidden_channels,
            max_delta_m=max_delta_m,
        )
    if method == "learnable_triangulation":
        return LearnableTriangulationModel(hidden_channels=hidden_channels)
    raise ValueError(f"unsupported extrinsic method: {method}")


def _tensor(value: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(np.array(value, copy=True)).unsqueeze(0).to(device)


def _predict(
    model: nn.Module,
    method: str,
    sequence: ExtrinsicSequence,
    device: torch.device,
) -> CalibratedPrediction:
    if method in FUSION_3D_METHODS:
        return model(
            _tensor(sequence.face_3d, device),
            _tensor(sequence.side_3d, device),
            _tensor(sequence.valid_face_3d, device),
            _tensor(sequence.valid_side_3d, device),
            torch.from_numpy(np.array(sequence.relative_rotation, copy=True)).to(
                device
            ),
        )
    if method == "learnable_triangulation":
        return model(
            _tensor(sequence.pixels_2d, device),
            _tensor(sequence.valid_2d, device),
            torch.from_numpy(np.array(sequence.projection, copy=True)).to(device),
            image_size=torch.from_numpy(
                np.array(sequence.image_size, copy=True)
            ).to(device),
        )
    raise ValueError(f"unsupported extrinsic method: {method}")


def calibrated_supervised_loss(
    model: nn.Module,
    method: str,
    sequence: ExtrinsicSequence,
    device: str | torch.device,
    *,
    smooth_l1_beta_m: float = 0.02,
) -> tuple[Tensor, Mapping[str, float]]:
    """Compute the method-appropriate masked Unity16 training objective."""
    target_device = torch.device(device)
    prediction = _predict(model, method, sequence, target_device)
    mapped, mapped_valid = torch_map_mhr70_to_unity16(
        prediction.points, prediction.valid
    )
    target = _tensor(sequence.gt_unity16_m, target_device)
    target_valid = _tensor(sequence.gt_valid, target_device).bool()
    common = mapped_valid & target_valid
    if common.sum() < 3:
        raise ValueError("fewer than three valid Unity supervision points")
    if method in FUSION_3D_METHODS:
        transform = masked_window_sim3(mapped, target, common)
        compared = apply_torch_sim3(mapped, transform)
    elif method == "learnable_triangulation":
        compared = mapped
    else:
        raise ValueError(f"unsupported extrinsic method: {method}")
    safe_target = torch.where(
        common[..., None], target, torch.zeros_like(target)
    )
    point_loss = F.smooth_l1_loss(
        compared,
        safe_target,
        beta=float(smooth_l1_beta_m),
        reduction="none",
    ).sum(dim=-1)
    loss = point_loss[common].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("calibrated supervised loss is non-finite")
    return loss, MappingProxyType(
        {
            "supervised_loss": float(loss.detach().cpu()),
            "valid_points": float(common.sum().detach().cpu()),
        }
    )


def _run_contract(
    output_root: Path,
    method: str,
    fold: UnityFold,
    seed: int,
) -> ExtrinsicRun:
    run_root = (
        Path(output_root)
        / f"fold_{fold.name}"
        / method
        / f"seed_{seed}"
    )
    return ExtrinsicRun(
        method=method,
        fold=fold.name,
        seed=int(seed),
        train_sequence=fold.train_sequence,
        test_sequence=fold.test_sequence,
        run_root=run_root,
        checkpoint_path=run_root / "final.pt",
        history_path=run_root / "history.json",
        config_path=run_root / "resolved_config.json",
        provenance_path=run_root / "provenance.json",
    )


def train_extrinsic_run(
    train_sequence: ExtrinsicSequence,
    *,
    method: str,
    fold: UnityFold,
    seed: int,
    output_root: Path,
    config: ExtrinsicTrainingConfig,
) -> ExtrinsicRun:
    """Train one fixed-epoch cell while accepting only the declared direction."""
    if method not in EXTRINSIC_METHODS:
        raise ValueError(f"unsupported extrinsic method: {method}")
    if train_sequence.sequence_id != fold.train_sequence:
        raise ValueError("training sequence does not match fold")
    if train_sequence.sequence_id in {fold.test_sequence, "static_sweep"}:
        raise ValueError("training sequence overlaps evaluation")
    device = torch.device(config.device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_extrinsic_model(
        method,
        hidden_channels=config.hidden_channels,
        max_delta_m=config.max_delta_m,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    run = _run_contract(output_root, method, fold, seed)
    resolved_config = {
        "method": method,
        "fold": asdict(fold),
        "seed": int(seed),
        "training": asdict(config),
    }
    _atomic_json(run.config_path, resolved_config)
    history: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = calibrated_supervised_loss(
            model,
            method,
            train_sequence,
            device,
            smooth_l1_beta_m=config.smooth_l1_beta_m,
        )
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(
            torch.isfinite(gradient).all().item() for gradient in gradients
        ):
            raise FloatingPointError("calibrated supervised gradient is non-finite")
        optimizer.step()
        history.append({"epoch": epoch + 1, **metrics})
        _atomic_json(run.history_path, history)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "method": method,
        "fold": fold.name,
        "seed": int(seed),
        "train_sequence": fold.train_sequence,
        "test_sequence": fold.test_sequence,
        "resolved_config": resolved_config,
        "source_identity": dict(train_sequence.source_identity),
    }
    _atomic_torch(run.checkpoint_path, checkpoint)
    provenance = {
        "method": method,
        "input_regime": (
            "calibrated_3d_to_3d"
            if method in FUSION_3D_METHODS
            else "calibrated_2d_to_3d"
        ),
        "uses_exact_camera_extrinsics": True,
        "unity_gt_supervision": True,
        "fixed_epoch_final_checkpoint": True,
        "fold": fold.name,
        "seed": int(seed),
        "train_sequence": fold.train_sequence,
        "test_sequence": fold.test_sequence,
        "static_excluded_from_training": True,
        "train_sample_ids": train_sequence.sample_ids.tolist(),
        "source_identity": dict(train_sequence.source_identity),
        "resolved_config": resolved_config,
        "history_epochs": len(history),
        "checkpoint_sha256": _sha256_file(run.checkpoint_path),
    }
    _atomic_json(run.provenance_path, provenance)
    return run


def validate_extrinsic_run(run: ExtrinsicRun) -> bool:
    """Validate artifact identity, fixed-epoch completeness, and hashes."""
    required = (
        run.checkpoint_path,
        run.history_path,
        run.config_path,
        run.provenance_path,
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        history = json.loads(run.history_path.read_text(encoding="utf-8"))
        config = json.loads(run.config_path.read_text(encoding="utf-8"))
        provenance = json.loads(
            run.provenance_path.read_text(encoding="utf-8")
        )
        checkpoint = torch.load(
            run.checkpoint_path, map_location="cpu", weights_only=False
        )
        epochs = int(config["training"]["epochs"])
        return bool(
            run.method in EXTRINSIC_METHODS
            and provenance["method"] == run.method == checkpoint["method"]
            and provenance["fold"] == run.fold == checkpoint["fold"]
            and int(provenance["seed"]) == run.seed == int(checkpoint["seed"])
            and provenance["train_sequence"]
            == run.train_sequence
            == checkpoint["train_sequence"]
            and provenance["test_sequence"]
            == run.test_sequence
            == checkpoint["test_sequence"]
            and provenance["static_excluded_from_training"] is True
            and provenance["fixed_epoch_final_checkpoint"] is True
            and provenance["resolved_config"] == config
            and checkpoint["resolved_config"] == config
            and len(history) == epochs
            and provenance["history_epochs"] == epochs
            and [int(row["epoch"]) for row in history]
            == list(range(1, epochs + 1))
            and provenance["checkpoint_sha256"]
            == _sha256_file(run.checkpoint_path)
            and provenance["source_identity"] == checkpoint["source_identity"]
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def load_extrinsic_model(
    run: ExtrinsicRun, device: str | torch.device = "cpu"
) -> nn.Module:
    if not validate_extrinsic_run(run):
        raise ValueError(f"invalid or incomplete extrinsic run: {run.run_root}")
    checkpoint = torch.load(
        run.checkpoint_path, map_location="cpu", weights_only=False
    )
    training = checkpoint["resolved_config"]["training"]
    model = make_extrinsic_model(
        run.method,
        hidden_channels=int(training["hidden_channels"]),
        max_delta_m=float(training["max_delta_m"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def run_extrinsic_inference(
    run: ExtrinsicRun,
    sequences: Mapping[str, ExtrinsicSequence],
    *,
    device: str = "cpu",
) -> tuple[MethodSequence, ...]:
    """Infer the held-out direction and static OOD sequence only."""
    if run.train_sequence in {run.test_sequence, "static_sweep"}:
        raise ValueError("training identity overlaps evaluation")
    selected = (run.test_sequence, "static_sweep")
    if any(sequence_id not in sequences for sequence_id in selected):
        raise ValueError("held-out or static sequence is unavailable")
    model = load_extrinsic_model(run, device)
    outputs: list[MethodSequence] = []
    for sequence_id in selected:
        sequence = sequences[sequence_id]
        with torch.inference_mode():
            prediction = _predict(
                model, run.method, sequence, torch.device(device)
            )
        points = prediction.points[0].detach().cpu().numpy().astype(np.float32)
        valid = prediction.valid[0].detach().cpu().numpy().astype(bool)
        method_sequence = MethodSequence(
            method=run.method,
            sequence_id=sequence_id,
            sample_ids=sequence.sample_ids,
            points=points,
            valid=valid,
            joint_names=tuple(mhr_names),
            metadata={
                "ranking_group": "unity_supervised_extrinsic",
                "input_regime": (
                    "calibrated_3d_to_3d"
                    if run.method in FUSION_3D_METHODS
                    else "calibrated_2d_to_3d"
                ),
                "fold": run.fold,
                "seed": run.seed,
                "train_sequence": run.train_sequence,
                "uses_exact_camera_extrinsics": True,
                "unity_gt_used_for_training": True,
            },
        )
        _save_sequence(
            run.run_root / "inference" / f"{sequence_id}.npz",
            method_sequence,
        )
        outputs.append(method_sequence)
    return tuple(outputs)
