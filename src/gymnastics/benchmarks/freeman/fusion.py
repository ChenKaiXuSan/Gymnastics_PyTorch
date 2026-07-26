"""Leakage-isolated adapters for existing multi-view fusion methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

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
