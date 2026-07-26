"""Leakage-isolated adapters for existing multi-view fusion methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from gymnastics.fusion.deterministic.experiment_matrix import (
    ALL_METHODS,
    STABLE_SIM3_JOINTS,
    bodypart_weights,
    current_body_average,
    fuse_weighted,
    root_align_to_reference,
    sim3_align_to_reference,
    smooth_sequence,
)
from gymnastics.common.skeletons.mhr70 import MHR70_NAMES
from gymnastics.fusion.rotation_aware.schema import PosePairTrial

from .schema import MethodPrediction, PosePairInput


METHOD_CLASSIFICATION = MappingProxyType(
    {
        method: (
            "GT_LEAKY_DIAGNOSTIC"
            if method == "sim3_face_stable_joint_weight"
            else "VALID"
        )
        for method in ALL_METHODS
    }
)


@dataclass(frozen=True)
class RotationRuntime:
    """Loaded zero-shot rotation-aware model and immutable provenance."""

    model: Any
    skeleton: Any
    provenance: Mapping[str, Any]
    resolved_config: Mapping[str, Any]


RuntimeLoader = Callable[[Path, Mapping[str, Any]], RotationRuntime]
InferenceRunner = Callable[..., Any]


def _prepared_views(pair: PosePairInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.array_equal(pair.view_a.frame_ids, pair.view_b.frame_ids):
        raise ValueError("fusion requires exact synchronized frame IDs")
    face = np.array(pair.view_a.points3d, dtype=np.float32, copy=True)
    side = np.array(pair.view_b.points3d, dtype=np.float32, copy=True)
    valid_face = np.asarray(pair.view_a.valid3d, dtype=bool)
    valid_side = np.asarray(pair.view_b.valid3d, dtype=bool)
    valid = valid_face | valid_side
    face = np.where(
        valid_face[..., None],
        face,
        np.where(valid_side[..., None], side, 0),
    )
    side = np.where(
        valid_side[..., None],
        side,
        np.where(valid_face[..., None], face, 0),
    )
    return face, side, valid


def _method_metadata(
    pair: PosePairInput,
    method: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "dataset": "FreeMan",
        "session_id": pair.session_id,
        "subject_id": pair.subject_id,
        "method": method,
        "classification": METHOD_CLASSIFICATION[method],
        "excluded_from_ranking": METHOD_CLASSIFICATION[method] != "VALID",
        "reference_view": pair.view_a.view_id,
        "other_view": pair.view_b.view_id,
        "temporal_alignment": "native_zero_offset",
        "reference_3d_consumed": False,
        "view_a_identity": dict(pair.view_a.metadata.get("identity", {})),
        "view_b_identity": dict(pair.view_b.metadata.get("identity", {})),
        **dict(extra or {}),
    }


def fuse_deterministic(
    pair: PosePairInput,
    methods: Sequence[str] = ALL_METHODS,
) -> tuple[MethodPrediction, ...]:
    """Run the existing deterministic matrix without accepting reference 3D."""
    requested = tuple(methods)
    unsupported = [method for method in requested if method not in ALL_METHODS]
    if unsupported:
        raise ValueError(f"unsupported deterministic methods: {unsupported}")
    face, side, source_valid = _prepared_views(pair)
    sim3_all: np.ndarray | None = None
    sim3_stable: np.ndarray | None = None
    stable_scales: np.ndarray | None = None
    smooth_stable: np.ndarray | None = None
    predictions: list[MethodPrediction] = []
    for method in requested:
        extra: dict[str, Any] = {}
        if method == "avg_body_current":
            fused = current_body_average(face, side)
        elif method == "avg_world_face_ref":
            fused = 0.5 * (face + side)
        elif method == "root_face_stable":
            aligned = root_align_to_reference(side, face)
            fused = 0.5 * (face + aligned)
        elif method == "sim3_face_all":
            if sim3_all is None:
                sim3_all, scales = sim3_align_to_reference(
                    side,
                    face,
                    tuple(range(face.shape[1])),
                )
                extra["scale_mean"] = float(np.mean(scales))
            fused = 0.5 * (face + sim3_all)
        elif method in {
            "sim3_face_stable",
            "sim3_face_stable_joint_weight",
            "sim3_face_stable_bodypart_weight",
            "sim3_face_stable_smooth_transform",
            "sim3_face_stable_smooth_kpt",
        }:
            if sim3_stable is None:
                sim3_stable, stable_scales = sim3_align_to_reference(
                    side,
                    face,
                    STABLE_SIM3_JOINTS,
                )
            extra["sim3_joints"] = list(STABLE_SIM3_JOINTS)
            extra["scale_mean"] = float(np.mean(stable_scales))
            if method == "sim3_face_stable":
                fused = 0.5 * (face + sim3_stable)
            elif method == "sim3_face_stable_joint_weight":
                weights = np.full((face.shape[1], 2), 0.5, dtype=np.float32)
                fused = fuse_weighted(face, sim3_stable, weights)
                extra.update(
                    {
                        "joint_weight_source": (
                            "unavailable_external_reference_equal_fallback"
                        ),
                        "joint_weights": weights.tolist(),
                    }
                )
            elif method == "sim3_face_stable_bodypart_weight":
                weights = bodypart_weights(face.shape[1])
                fused = fuse_weighted(face, sim3_stable, weights)
                extra["joint_weights"] = weights.tolist()
            elif method == "sim3_face_stable_smooth_transform":
                if smooth_stable is None:
                    smooth_stable = smooth_sequence(sim3_stable, win=5)
                fused = 0.5 * (face + smooth_stable)
                extra.update({"smooth_target": "side_after_sim3", "smooth_window": 5})
            else:
                fused = smooth_sequence(0.5 * (face + sim3_stable), win=5)
                extra.update({"smooth_target": "fused_world", "smooth_window": 5})
        else:
            raise AssertionError(f"unreachable method: {method}")
        fused = np.asarray(fused, dtype=np.float32)
        valid = source_valid & np.isfinite(fused).all(axis=-1)
        fused = np.where(valid[..., None], fused, 0)
        predictions.append(
            MethodPrediction(
                method=method,
                session_id=pair.session_id,
                subject_id=pair.subject_id,
                fps=pair.fps,
                points=fused,
                valid=valid,
                frame_ids=pair.view_a.frame_ids,
                metadata=_method_metadata(pair, method, extra),
            )
        )
    return tuple(predictions)


def save_method_prediction(
    prediction: MethodPrediction,
    output_root: Path,
) -> Path:
    """Atomically save one compact deterministic or learned prediction."""
    target = (
        Path(output_root).resolve()
        / prediction.method
        / f"subject_{prediction.subject_id:02d}"
        / prediction.session_id
        / "fused_sequence.npz"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            method=np.asarray(prediction.method),
            session_id=np.asarray(prediction.session_id),
            subject_id=np.asarray(prediction.subject_id, dtype=np.int32),
            fps=np.asarray(prediction.fps, dtype=np.float64),
            points=prediction.points,
            valid=prediction.valid,
            frame_ids=prediction.frame_ids,
            metadata=np.asarray(
                json.dumps(dict(prediction.metadata), sort_keys=True)
            ),
        )
    temporary.replace(target)
    return target


def load_method_prediction(path: Path) -> MethodPrediction:
    """Load one compact fused prediction and validate its immutable contract."""
    prediction_path = Path(path).resolve()
    with np.load(prediction_path, allow_pickle=False) as data:
        required = {
            "method",
            "session_id",
            "subject_id",
            "fps",
            "points",
            "valid",
            "frame_ids",
            "metadata",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"fused cache missing arrays: {sorted(missing)}")
        metadata = json.loads(str(data["metadata"].item()))
        if not isinstance(metadata, dict):
            raise ValueError("fused cache metadata must be a mapping")
        return MethodPrediction(
            method=str(data["method"].item()),
            session_id=str(data["session_id"].item()),
            subject_id=int(data["subject_id"].item()),
            fps=float(data["fps"].item()),
            points=np.asarray(data["points"]),
            valid=np.asarray(data["valid"]),
            frame_ids=np.asarray(data["frame_ids"]),
            metadata=metadata,
        )


def build_rotation_aware_trial(pair: PosePairInput) -> PosePairTrial:
    """Build the existing MHR70 trial contract with native zero offset."""
    if not np.array_equal(pair.view_a.frame_ids, pair.view_b.frame_ids):
        raise ValueError("rotation-aware fusion requires exact synchronized frame IDs")
    face = np.where(
        pair.view_a.valid3d[..., None],
        pair.view_a.points3d,
        0,
    )
    side = np.where(
        pair.view_b.valid3d[..., None],
        pair.view_b.points3d,
        0,
    )
    frame_ids = np.array(pair.view_a.frame_ids, dtype=np.int32, copy=True)
    return PosePairTrial(
        face=face,
        side=side,
        valid_face=pair.view_a.valid3d,
        valid_side=pair.view_b.valid3d,
        timestamps=frame_ids.astype(np.float64) / float(pair.fps),
        face_map=frame_ids,
        side_map=frame_ids,
        joint_names=tuple(MHR70_NAMES),
        person_id=f"{pair.subject_id:02d}",
        trial_id=pair.session_id,
        fps=float(pair.fps),
        source_metadata={
            "dataset": "FreeMan",
            "session_id": pair.session_id,
            "subject_id": pair.subject_id,
            "temporal_alignment": "native_zero_offset",
            "reference_view": pair.view_a.view_id,
            "zero_shot": True,
            "reference_3d_consumed": False,
        },
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_project_path(path: str | Path) -> Path:
    from gymnastics.common.paths import PROJECT_ROOT

    value = Path(path)
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def _default_runtime_loader(
    checkpoint: Path,
    benchmark_config: Mapping[str, Any],
) -> RotationRuntime:
    import torch

    from gymnastics.fusion.rotation_aware.cli import (
        TWIST_ABLATIONS,
        load_config as load_rotation_config,
    )
    from gymnastics.fusion.rotation_aware.config import load_skeleton_spec
    from gymnastics.fusion.rotation_aware.model import RotationAwareFusionModel
    from gymnastics.fusion.rotation_aware.training import load_checkpoint

    rotation_settings = benchmark_config.get("rotation_aware", {})
    rotation_config_path = _resolved_project_path(
        rotation_settings.get(
            "config",
            "configs/fusion/rotation_aware.yaml",
        )
    )
    resolved = load_rotation_config(rotation_config_path)
    paths = resolved.get("paths", {})
    if not isinstance(paths, Mapping) or "skeleton" not in paths:
        raise ValueError("rotation-aware config requires paths.skeleton")
    skeleton = load_skeleton_spec(_resolved_project_path(paths["skeleton"]))
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ValueError("rotation-aware checkpoint payload must be a mapping")
    skeleton_metadata = raw.get("skeleton")
    if not isinstance(skeleton_metadata, Mapping) or tuple(
        skeleton_metadata.get("joint_names", ())
    ) != tuple(MHR70_NAMES):
        raise ValueError("rotation-aware checkpoint skeleton is not exact MHR70")
    training = raw.get("training_config")
    if not isinstance(training, Mapping) or not training:
        raise ValueError("rotation-aware checkpoint requires training_config")
    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("rotation-aware checkpoint requires provenance")
    ablation = str(training.get("ablation", "A6"))
    model = RotationAwareFusionModel(
        skeleton,
        hidden_channels=int(training.get("hidden_channels", 128)),
        twist_residual=ablation in TWIST_ABLATIONS,
    )
    payload = load_checkpoint(checkpoint, model)
    loaded_provenance = payload.get("provenance", {})
    return RotationRuntime(
        model=model,
        skeleton=skeleton,
        provenance={
            **dict(loaded_provenance),
            "training_config": dict(training),
            "ablation": ablation,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _file_sha256(checkpoint),
        },
        resolved_config=resolved,
    )


def _validate_zero_shot_runtime(runtime: RotationRuntime) -> None:
    joint_names = tuple(getattr(runtime.skeleton, "joint_names", ()))
    if joint_names != tuple(MHR70_NAMES):
        raise ValueError("rotation-aware runtime skeleton is not exact MHR70")

    def contains_freeman_training(value: Any, key_path: tuple[str, ...] = ()) -> bool:
        if isinstance(value, Mapping):
            return any(
                contains_freeman_training(
                    child,
                    (*key_path, str(key).lower()),
                )
                for key, child in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(contains_freeman_training(child, key_path) for child in value)
        relevant = any(
            token in key
            for key in key_path
            for token in ("train", "dataset", "data_root", "cache_manifest")
        )
        return relevant and isinstance(value, str) and "freeman" in value.lower()

    if contains_freeman_training(runtime.provenance):
        raise ValueError("rotation-aware checkpoint has FreeMan training provenance")


def fuse_rotation_aware(
    pair: PosePairInput,
    checkpoint: Path,
    run_id: str,
    config: Mapping[str, Any],
    *,
    runtime_loader: RuntimeLoader | None = None,
    inference_runner: InferenceRunner | None = None,
) -> MethodPrediction:
    """Run an existing gymnastics checkpoint on one FreeMan pair zero-shot."""
    if not run_id:
        raise ValueError("rotation-aware run_id is required")
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    trial = build_rotation_aware_trial(pair)
    loader = runtime_loader or _default_runtime_loader
    runtime = loader(checkpoint_path, config)
    _validate_zero_shot_runtime(runtime)
    if inference_runner is None:
        from gymnastics.fusion.rotation_aware.inference import run_inference

        inference_runner = run_inference
    window = runtime.resolved_config.get("window", {})
    if not isinstance(window, Mapping):
        raise ValueError("rotation-aware resolved config requires window mapping")
    output_root = (
        Path(config["paths"]["output_root"]).resolve()
        / "fusion"
        / "rotation_aware"
        / run_id
        / "native"
    )
    result = inference_runner(
        runtime.model,
        trial,
        runtime.skeleton,
        output_root=output_root,
        run_id=run_id,
        window_length=int(window.get("length", 128)),
        stride=int(window.get("eval_stride", 64)),
        provenance=dict(runtime.provenance),
        resolved_config=dict(runtime.resolved_config),
    )
    sequence_path = Path(result.sequence_path).resolve()
    with np.load(sequence_path, allow_pickle=False) as data:
        points = np.asarray(data["kpts_world"], dtype=np.float32)
        valid = np.asarray(data["joint_valid"], dtype=bool)
        frame_ids = (
            np.asarray(data["face_map"], dtype=np.int64)
            if "face_map" in data
            else np.array(pair.view_a.frame_ids, copy=True)
        )
    if points.shape != pair.view_a.points3d.shape or valid.shape != points.shape[:2]:
        raise ValueError("rotation-aware output does not match MHR70 pair shape")
    if not np.array_equal(frame_ids, pair.view_a.frame_ids):
        raise ValueError("rotation-aware output changed native frame identity")
    valid &= np.isfinite(points).all(axis=-1)
    points = np.where(valid[..., None], points, 0)
    provenance = dict(runtime.provenance)
    return MethodPrediction(
        method=f"rotation_aware:{run_id}",
        session_id=pair.session_id,
        subject_id=pair.subject_id,
        fps=pair.fps,
        points=points,
        valid=valid,
        frame_ids=frame_ids,
        metadata={
            "dataset": "FreeMan",
            "method": f"rotation_aware:{run_id}",
            "classification": "VALID",
            "excluded_from_ranking": False,
            "zero_shot": True,
            "reference_3d_consumed": False,
            "training_source": "private_gymnastics",
            "run_id": run_id,
            "ablation": provenance.get("ablation"),
            "checkpoint_path": provenance.get("checkpoint_path", str(checkpoint_path)),
            "checkpoint_sha256": provenance.get(
                "checkpoint_sha256",
                _file_sha256(checkpoint_path),
            ),
        },
    )
