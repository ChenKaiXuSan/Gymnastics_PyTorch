"""Fixed semantic mapping from MHR70 to the Unity evaluation subset."""

from __future__ import annotations

import numpy as np

from .schema import MappedPose


MHR70_MAPPING_VERSION = "unity16-v1"
EVALUATION_JOINT_NAMES = (
    "Hips",
    "Neck",
    "LeftUpperArm",
    "LeftLowerArm",
    "LeftHand",
    "RightUpperArm",
    "RightLowerArm",
    "RightHand",
    "LeftUpperLeg",
    "LeftLowerLeg",
    "LeftFoot",
    "LeftToes",
    "RightUpperLeg",
    "RightLowerLeg",
    "RightFoot",
    "RightToes",
)
UNITY_JOINT_INDICES = {
    name: index
    for index, name in enumerate(
        (
            "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
            "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
            "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
            "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
            "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
        )
    )
}

_DIRECT = {
    "Neck": 69,
    "LeftUpperArm": 5,
    "LeftLowerArm": 7,
    "LeftHand": 62,
    "RightUpperArm": 6,
    "RightLowerArm": 8,
    "RightHand": 41,
    "LeftUpperLeg": 9,
    "LeftLowerLeg": 11,
    "LeftFoot": 13,
    "RightUpperLeg": 10,
    "RightLowerLeg": 12,
    "RightFoot": 14,
}
_DERIVED = {
    "Hips": (9, 10),
    "LeftToes": (15, 16),
    "RightToes": (18, 19),
}
MHR70_EVALUATION_SOURCES = {
    name: (_DIRECT[name],) if name in _DIRECT else _DERIVED[name]
    for name in EVALUATION_JOINT_NAMES
}


def map_mhr70_to_unity(
    points: np.ndarray, valid: np.ndarray | None = None
) -> MappedPose:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim < 2 or array.shape[-2:] != (70, 3):
        raise ValueError("MHR70 points must end with shape [70,3]")
    inferred = np.isfinite(array).all(axis=-1) & np.any(array != 0, axis=-1)
    source_valid = inferred if valid is None else np.asarray(valid, dtype=bool) & inferred
    if source_valid.shape != array.shape[:-1]:
        raise ValueError("MHR70 validity must match points without xyz")
    output_shape = array.shape[:-2] + (len(EVALUATION_JOINT_NAMES), 3)
    mapped = np.zeros(output_shape, dtype=np.float32)
    mapped_valid = np.zeros(output_shape[:-1], dtype=bool)
    for target_index, name in enumerate(EVALUATION_JOINT_NAMES):
        if name in _DIRECT:
            source = _DIRECT[name]
            mapped[..., target_index, :] = array[..., source, :]
            mapped_valid[..., target_index] = source_valid[..., source]
        else:
            sources = _DERIVED[name]
            mapped[..., target_index, :] = np.mean(array[..., sources, :], axis=-2)
            mapped_valid[..., target_index] = np.all(source_valid[..., sources], axis=-1)
    mapped[~mapped_valid] = 0
    return MappedPose(mapped, mapped_valid, EVALUATION_JOINT_NAMES)


def select_unity_evaluation_joints(
    points: np.ndarray, valid: np.ndarray | None = None
) -> MappedPose:
    array = np.asarray(points, dtype=np.float32)
    indices = [UNITY_JOINT_INDICES[name] for name in EVALUATION_JOINT_NAMES]
    selected = array[..., indices, :]
    inferred = np.isfinite(selected).all(axis=-1)
    if valid is None:
        selected_valid = inferred
    else:
        selected_valid = np.asarray(valid, dtype=bool)[..., indices] & inferred
    selected = np.where(selected_valid[..., None], selected, 0)
    return MappedPose(selected, selected_valid, EVALUATION_JOINT_NAMES)
