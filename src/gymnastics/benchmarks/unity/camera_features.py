"""Leakage-safe fitted-camera features from synchronized SAM3D 2D joints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gymnastics.triangulation.estimate_extrinsics import (
    estimate_relative_pose,
    reprojection_error,
)


def _readonly(value: np.ndarray, *, dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _proper_rotation(value: np.ndarray) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
    ):
        raise ValueError("rotation_face_to_side must be a finite proper rotation")
    return rotation


@dataclass(frozen=True)
class FittedRelativeCamera:
    """Relative face-to-side pose fitted only from a declared training sequence."""

    rotation_face_to_side: np.ndarray
    translation_direction_face_to_side: np.ndarray
    inlier_ratio: float
    holdout_reprojection_px: float
    fit_sample_ids: np.ndarray

    def __post_init__(self) -> None:
        rotation = _proper_rotation(self.rotation_face_to_side)
        translation = np.asarray(
            self.translation_direction_face_to_side, dtype=np.float64
        )
        sample_ids = np.asarray(self.fit_sample_ids, dtype=np.int64)
        if (
            translation.shape != (3,)
            or not np.isfinite(translation).all()
            or not np.isclose(np.linalg.norm(translation), 1.0, atol=1e-4)
        ):
            raise ValueError("translation direction must be a finite unit vector")
        if (
            sample_ids.ndim != 1
            or not len(sample_ids)
            or len(set(sample_ids.tolist())) != len(sample_ids)
        ):
            raise ValueError("fit_sample_ids must be non-empty and unique")
        if (
            not np.isfinite(self.inlier_ratio)
            or not 0.0 <= self.inlier_ratio <= 1.0
            or not np.isfinite(self.holdout_reprojection_px)
            or self.holdout_reprojection_px < 0.0
        ):
            raise ValueError("camera fit diagnostics are invalid")
        object.__setattr__(
            self,
            "rotation_face_to_side",
            _readonly(rotation, dtype=np.float32),
        )
        object.__setattr__(
            self,
            "translation_direction_face_to_side",
            _readonly(translation, dtype=np.float32),
        )
        object.__setattr__(
            self, "fit_sample_ids", _readonly(sample_ids, dtype=np.int64)
        )


@dataclass(frozen=True)
class CameraFeatureSequence:
    """One fitted rig encoded as global and frame-joint camera features."""

    global_features: np.ndarray
    joint_features: np.ndarray
    valid: np.ndarray
    global_schema: tuple[str, ...]
    joint_schema: tuple[str, ...]

    def __post_init__(self) -> None:
        global_features = np.asarray(self.global_features, dtype=np.float32)
        joint_features = np.asarray(self.joint_features, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        if global_features.shape != (len(self.global_schema),):
            raise ValueError("global camera features do not match their schema")
        if (
            joint_features.ndim != 3
            or joint_features.shape[-1] != len(self.joint_schema)
            or valid.shape != joint_features.shape[:2]
        ):
            raise ValueError("joint camera features must have shape [T,J,C]")
        if not np.isfinite(global_features).all() or not np.isfinite(
            joint_features
        ).all():
            raise ValueError("camera features must be finite")
        if np.any(joint_features[~valid] != 0):
            raise ValueError("invalid joint camera features must be zero")
        object.__setattr__(
            self,
            "global_features",
            _readonly(global_features, dtype=np.float32),
        )
        object.__setattr__(
            self,
            "joint_features",
            _readonly(joint_features, dtype=np.float32),
        )
        object.__setattr__(self, "valid", _readonly(valid, dtype=bool))
        object.__setattr__(self, "global_schema", tuple(self.global_schema))
        object.__setattr__(self, "joint_schema", tuple(self.joint_schema))


def _validated_observations(
    pixels: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    sample_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    if pixels.ndim != 4 or pixels.shape[1] != 2 or pixels.shape[-1] != 2:
        raise ValueError("pixels must have shape [T,2,J,2]")
    if valid.shape != pixels.shape[:-1]:
        raise ValueError("valid must have shape [T,2,J]")
    if intrinsics.shape != (2, 3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("intrinsics must have finite shape [2,3,3]")
    if sample_ids.shape != (pixels.shape[0],):
        raise ValueError("sample_ids must have shape [T]")
    effective = valid & np.isfinite(pixels).all(axis=-1)
    safe = np.where(effective[..., None], pixels, 0.0)
    return safe, effective, intrinsics, sample_ids


def _calibration(intrinsics: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "K": np.asarray(intrinsics, dtype=np.float64),
        "dist": np.zeros(5, dtype=np.float64),
    }


def fit_relative_camera_from_training_2d(
    pixels: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    *,
    sample_ids: np.ndarray,
    threshold_px: float = 2.0,
) -> FittedRelativeCamera:
    """Fit and audit one relative pose without receiving evaluation frames or 3D."""
    safe, effective, intrinsics, sample_ids = _validated_observations(
        pixels, valid, intrinsics, sample_ids
    )
    if (
        not np.isfinite(threshold_px)
        or threshold_px <= 0.0
        or len(sample_ids) < 4
    ):
        raise ValueError("camera fitting needs four frames and a positive threshold")
    face_calibration = _calibration(intrinsics[0])
    side_calibration = _calibration(intrinsics[1])
    audit_fit = np.arange(len(sample_ids)) % 2 == 0
    audit_holdout = ~audit_fit
    audit = estimate_relative_pose(
        np.where(
            effective[audit_fit, 0, :, None],
            safe[audit_fit, 0],
            0.0,
        ),
        np.where(
            effective[audit_fit, 1, :, None],
            safe[audit_fit, 1],
            0.0,
        ),
        face_calibration,
        side_calibration,
        float(threshold_px),
    )
    if audit is None:
        raise ValueError("training 2D correspondences cannot fit an audit camera")
    audit_rotation, audit_translation, _ = audit
    holdout_px = reprojection_error(
        audit_rotation,
        audit_translation,
        np.where(
            effective[audit_holdout, 0, :, None],
            safe[audit_holdout, 0],
            0.0,
        ),
        np.where(
            effective[audit_holdout, 1, :, None],
            safe[audit_holdout, 1],
            0.0,
        ),
        face_calibration,
        side_calibration,
    )
    fitted = estimate_relative_pose(
        np.where(effective[:, 0, :, None], safe[:, 0], 0.0),
        np.where(effective[:, 1, :, None], safe[:, 1], 0.0),
        face_calibration,
        side_calibration,
        float(threshold_px),
    )
    if fitted is None or not np.isfinite(holdout_px):
        raise ValueError("training 2D correspondences cannot fit a finite camera")
    rotation, translation, inlier_ratio = fitted
    translation = np.asarray(translation, dtype=np.float64)
    translation /= max(float(np.linalg.norm(translation)), 1e-12)
    return FittedRelativeCamera(
        rotation_face_to_side=rotation,
        translation_direction_face_to_side=translation,
        inlier_ratio=float(inlier_ratio),
        holdout_reprojection_px=float(holdout_px),
        fit_sample_ids=sample_ids,
    )


GLOBAL_CAMERA_SCHEMA = (
    "rotation_col0_x",
    "rotation_col0_y",
    "rotation_col0_z",
    "rotation_col1_x",
    "rotation_col1_y",
    "rotation_col1_z",
    "translation_direction_x",
    "translation_direction_y",
    "translation_direction_z",
    "face_fx_over_width",
    "face_fy_over_height",
    "face_cx_over_width",
    "face_cy_over_height",
    "side_fx_over_width",
    "side_fy_over_height",
    "side_cx_over_width",
    "side_cy_over_height",
    "fit_inlier_ratio",
    "holdout_reprojection_log_quality",
)

JOINT_CAMERA_SCHEMA = (
    "face_x_normalized",
    "face_y_normalized",
    "side_x_normalized",
    "side_y_normalized",
    "symmetric_epipolar_residual",
    "ray_intersection_angle_over_pi",
    "face_valid",
    "side_valid",
)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _normalized_rays(
    pixels: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pixel_homogeneous = np.concatenate(
        (pixels, np.ones(pixels.shape[:-1] + (1,), dtype=np.float64)), axis=-1
    )
    normalized = np.einsum(
        "cij,tcqj->tcqi", np.linalg.inv(intrinsics), pixel_homogeneous
    )
    rays = normalized / np.linalg.norm(
        normalized, axis=-1, keepdims=True
    ).clip(min=1e-12)
    return normalized, rays


def build_camera_feature_sequence(
    fitted: FittedRelativeCamera,
    pixels: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    *,
    image_sizes: np.ndarray,
) -> CameraFeatureSequence:
    """Encode one fitted rig and observation geometry without using any 3D target."""
    frame_ids = np.arange(np.asarray(pixels).shape[0], dtype=np.int64)
    safe, effective, intrinsics, _ = _validated_observations(
        pixels, valid, intrinsics, frame_ids
    )
    image_sizes = np.asarray(image_sizes, dtype=np.float64)
    if (
        image_sizes.shape != (2, 2)
        or not np.isfinite(image_sizes).all()
        or np.any(image_sizes <= 0.0)
    ):
        raise ValueError("image_sizes must have positive finite shape [2,2]")
    rotation = np.asarray(fitted.rotation_face_to_side, dtype=np.float64)
    translation = np.asarray(
        fitted.translation_direction_face_to_side, dtype=np.float64
    )
    rotation_6d = rotation[:, :2].T.reshape(-1)
    intrinsic_features: list[float] = []
    for camera_index in range(2):
        width, height = image_sizes[camera_index]
        matrix = intrinsics[camera_index]
        intrinsic_features.extend(
            (
                matrix[0, 0] / width,
                matrix[1, 1] / height,
                matrix[0, 2] / width,
                matrix[1, 2] / height,
            )
        )
    reprojection_quality = np.log1p(
        min(float(fitted.holdout_reprojection_px), 100.0)
    ) / np.log(101.0)
    global_features = np.concatenate(
        (
            rotation_6d,
            translation,
            np.asarray(intrinsic_features),
            np.asarray((fitted.inlier_ratio, reprojection_quality)),
        )
    ).astype(np.float32)

    normalized_pixels = safe.copy()
    for camera_index in range(2):
        normalized_pixels[:, camera_index, :, 0] /= image_sizes[camera_index, 0]
        normalized_pixels[:, camera_index, :, 1] /= image_sizes[camera_index, 1]
    homogeneous, rays = _normalized_rays(safe, intrinsics)
    essential = _skew(translation) @ rotation
    face_h = homogeneous[:, 0]
    side_h = homogeneous[:, 1]
    face_epiline = np.einsum("ij,tqj->tqi", essential, face_h)
    side_epiline = np.einsum("ij,tqj->tqi", essential.T, side_h)
    numerator = np.abs(
        np.einsum("tqi,tqi->tq", side_h, face_epiline)
    )
    denominator = np.sqrt(
        np.square(face_epiline[..., :2]).sum(axis=-1)
        + np.square(side_epiline[..., :2]).sum(axis=-1)
    ).clip(min=1e-12)
    epipolar = numerator / denominator
    side_ray_in_face = np.einsum("ij,tqj->tqi", rotation.T, rays[:, 1])
    ray_cosine = np.einsum("tqi,tqi->tq", rays[:, 0], side_ray_in_face)
    ray_angle = np.arccos(np.clip(ray_cosine, -1.0, 1.0)) / np.pi
    common = effective[:, 0] & effective[:, 1]
    joint_features = np.concatenate(
        (
            normalized_pixels[:, 0],
            normalized_pixels[:, 1],
            epipolar[..., None],
            ray_angle[..., None],
            effective[:, 0, :, None].astype(np.float64),
            effective[:, 1, :, None].astype(np.float64),
        ),
        axis=-1,
    )
    joint_features = np.where(common[..., None], joint_features, 0.0)
    return CameraFeatureSequence(
        global_features=global_features,
        joint_features=joint_features.astype(np.float32),
        valid=common,
        global_schema=GLOBAL_CAMERA_SCHEMA,
        joint_schema=JOINT_CAMERA_SCHEMA,
    )
