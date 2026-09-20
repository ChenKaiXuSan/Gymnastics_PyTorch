"""Unified dual-view sample and batch contracts.

Every dataset adapter (private gymnastics recordings, FreeMan, Unity) converts
its raw material into :class:`DualViewSample`, and the shared windowing code in
:mod:`gymnastics.fusion.data.windows` turns samples into the
tensor batch described by :class:`FusionBatch`.  The fusion model therefore
never sees dataset-specific structure.

Research Motivation:
    The three data sources differ in almost everything (camera rigs, frame
    rates, availability of ground truth, availability of cycle annotations),
    but the *method* only needs synchronised dual-view 3D keypoints, physical
    timestamps, and, when available, cycle boundaries.  Fixing that contract
    here makes the "model is dataset-agnostic" requirement checkable: an
    adapter that produces a valid :class:`DualViewSample` can be trained and
    evaluated with the same Lightning module and the same Hydra experiment.

Common Coordinate System:
    ``view_a`` and ``view_b`` are expressed in the project's canonical
    *pelvis-centred body frame* (``gymnastics.keypoints.geometry``):

    * origin at the pelvis (midpoint of the hips) of the same view,
    * x-axis from the left hip to the right hip,
    * y-axis along the pelvis-to-thorax direction (orthogonalised),
    * z-axis completing a right-handed frame,
    * lengths divided by the per-sequence median pelvis-to-thorax distance
      (so one unit is approximately one torso length).

    Both views are canonicalised independently, which removes the unknown
    relative camera pose without calibration.  The transform of View A is
    kept (:attr:`DualViewSample.transform_a`) so a fused pose can be mapped
    back into View A's original world frame.  Reference poses (ground truth or
    pseudo ground truth) stay in their native frame, so accuracy against them
    is measured after Procrustes alignment.

Cycle Information:
    ``cycle_bounds`` lists complete motion cycles as half-open frame ranges
    ``[start, end)`` and ``cycle_mids`` the turn-around frame of each cycle
    (``start < mid < end``).  Both come from the record files written by
    ``gymnastics align`` / ``gymnastics align cycles``; the training code
    never detects cycles itself.  ``cycle_bounds`` may be empty when a
    sequence has no (detected) cycles; in that case phase features are marked
    invalid and the model falls back to plain temporal modelling.
    ``cycle_mids`` is either empty or has one entry per cycle.

Tensor shapes in :class:`FusionBatch` (``B`` windows, ``T`` samples, ``J`` joints):

    pose_a, pose_b         [B, T, J, 3]  float32
    valid_a, valid_b       [B, T, J]     bool
    frame_mask             [B, T]        bool   (false on padding)
    delta_t                [B, T]        float32 seconds since the previous sample
    timestamps             [B, T]        float64 seconds
    phase                  [B, T]        float32 in [0, 1)
    phase_valid            [B, T]        bool
    cycle_index            [B, T]        int64  (-1 outside known cycles)
    half_index             [B, T]        int64  (0 outward, 1 return, -1 unknown)
    clean_a, clean_b       [B, T, J, 3]  the uncorrupted inputs (training only)
    clean_valid_a/b        [B, T, J]     bool
    corruption_mask_a/b    [B, T, J]     bool   (true where the input was altered)
    reference              [B, T, J, 3]  float32 (optional, evaluation only)
    reference_valid        [B, T, J]     bool
    reference_canonical    [B]           bool   (reference shares the canonical frame)
    cycle_target           [B, T, J, 3]  float32 leave-one-cycle-out target (cycle_target.py)
    cycle_confidence       [B, T, J]     float32 in [0, 1], 0 where undefined
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TypedDict

import numpy as np
import torch

from gymnastics.common.paths import CONFIG_ROOT
from gymnastics.keypoints.schema import PosePairTrial

from .skeleton import CommonSkeleton


def _readonly(value: np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class CanonicalTransformRecord:
    """Per-frame rigid transform plus sequence scale that canonicalised a view.

    ``world = canonical * scale @ rotation^T + origin`` (applied per frame).

    Attributes:
        rotation: ``[T, 3, 3]`` local-to-world rotation of each frame.
        origin: ``[T, 3]`` world position of the pelvis of each frame.
        scale: Scalar sequence scale (median torso length in world units).
        valid: ``[T]`` true where the frame was observed directly.
    """

    rotation: np.ndarray
    origin: np.ndarray
    scale: float
    valid: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float32)
        origin = np.asarray(self.origin, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        frames = len(valid)
        if rotation.shape != (frames, 3, 3) or origin.shape != (frames, 3):
            raise ValueError("transform arrays must have shapes [T,3,3], [T,3] and [T]")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("transform scale must be positive and finite")
        object.__setattr__(self, "rotation", _readonly(rotation, dtype=np.float32))
        object.__setattr__(self, "origin", _readonly(origin, dtype=np.float32))
        object.__setattr__(self, "valid", _readonly(valid, dtype=bool))
        object.__setattr__(self, "scale", float(self.scale))

    def restore(self, points: np.ndarray) -> np.ndarray:
        """Map canonical ``[T, J, 3]`` points back to the original world frame."""
        canonical = np.asarray(points, dtype=np.float32)
        if canonical.ndim != 3 or canonical.shape[0] != len(self.valid) or canonical.shape[-1] != 3:
            raise ValueError("points must have shape [T, J, 3] matching the transform")
        return np.einsum("tjc,tdc->tjd", canonical * self.scale, self.rotation) + self.origin[:, None, :]


@dataclass(frozen=True)
class DualViewSample:
    """One synchronised dual-view 3D pose sequence in the common representation.

    Attributes:
        dataset: Short dataset identifier (``"gymnastics"``, ``"freeman"``,
            ``"unity"``, ``"synthetic"``).
        subject_id: Identifier used for subject-disjoint splitting.
        sequence_id: Identifier of the sequence within the subject.
        view_a: ``[T, J, 3]`` float32 canonical keypoints of View A.
        view_b: ``[T, J, 3]`` float32 canonical keypoints of View B.
        valid_a: ``[T, J]`` bool validity of View A joints.
        valid_b: ``[T, J]`` bool validity of View B joints.
        timestamps: ``[T]`` float64 strictly increasing seconds.
        joint_names: Names of the ``J`` joints (the common skeleton).
        cycle_bounds: Complete cycles as ``(start, end)`` frame ranges,
            increasing and non-overlapping.  Empty when unknown.
        cycle_mids: Turn-around frame of each cycle (``start < mid < end``);
            empty when unknown, otherwise one entry per cycle.
        reference: Optional ``[T, J, 3]`` reference pose in its native frame.
        reference_valid: Optional ``[T, J]`` validity of ``reference``.
        reference_canonical: True when ``reference`` is expressed in the same
            canonical body frame as ``view_a`` (synthetic data); false for
            world-frame references, for which only Procrustes-aligned errors
            are meaningful.
        transform_a: Optional canonical transform of View A.
        metadata: Free-form provenance (JSON-serialisable).
    """

    dataset: str
    subject_id: str
    sequence_id: str
    view_a: np.ndarray
    view_b: np.ndarray
    valid_a: np.ndarray
    valid_b: np.ndarray
    timestamps: np.ndarray
    joint_names: tuple[str, ...]
    cycle_bounds: tuple[tuple[int, int], ...] = ()
    cycle_mids: tuple[int, ...] = ()
    reference: np.ndarray | None = None
    reference_valid: np.ndarray | None = None
    reference_canonical: bool = False
    transform_a: CanonicalTransformRecord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.dataset or not self.subject_id or not self.sequence_id:
            raise ValueError("dataset, subject_id and sequence_id are required")
        view_a = np.asarray(self.view_a, dtype=np.float32)
        view_b = np.asarray(self.view_b, dtype=np.float32)
        if view_a.ndim != 3 or view_a.shape[-1] != 3 or view_a.shape != view_b.shape:
            raise ValueError("view_a and view_b must have equal shape [T, J, 3]")
        frames, joints = view_a.shape[:2]
        if frames == 0:
            raise ValueError("a sample needs at least one frame")
        if len(self.joint_names) != joints:
            raise ValueError("joint_names must have one entry per joint")
        valid_a = np.asarray(self.valid_a, dtype=bool)
        valid_b = np.asarray(self.valid_b, dtype=bool)
        if valid_a.shape != (frames, joints) or valid_b.shape != (frames, joints):
            raise ValueError("valid masks must have shape [T, J]")
        finite_a = np.isfinite(view_a).all(axis=-1)
        finite_b = np.isfinite(view_b).all(axis=-1)
        if np.any(valid_a & ~finite_a) or np.any(valid_b & ~finite_b):
            raise ValueError("valid masks cannot mark non-finite joints as valid")
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        if timestamps.shape != (frames,) or not np.isfinite(timestamps).all():
            raise ValueError("timestamps must be finite with shape [T]")
        if frames > 1 and not np.all(np.diff(timestamps) > 0):
            raise ValueError("timestamps must be strictly increasing")
        bounds = tuple((int(s), int(e)) for s, e in self.cycle_bounds)
        previous_end = 0
        for start, end in bounds:
            if start < previous_end or end <= start or end > frames:
                raise ValueError(f"cycle_bounds must be increasing, non-overlapping ranges within [0, {frames}): {bounds}")
            previous_end = end
        mids = tuple(int(m) for m in self.cycle_mids)
        if mids and len(mids) != len(bounds):
            raise ValueError("cycle_mids must be empty or have one entry per cycle")
        for (start, end), mid in zip(bounds, mids):
            if not start < mid < end:
                raise ValueError(f"cycle middle {mid} must lie strictly inside ({start}, {end})")
        if (self.reference is None) != (self.reference_valid is None):
            raise ValueError("reference and reference_valid must be provided together")
        if self.reference is not None:
            reference = np.asarray(self.reference, dtype=np.float32)
            reference_valid = np.asarray(self.reference_valid, dtype=bool)
            if reference.shape != view_a.shape or reference_valid.shape != (frames, joints):
                raise ValueError("reference must have shape [T, J, 3] and reference_valid [T, J]")
            if np.any(reference_valid & ~np.isfinite(reference).all(axis=-1)):
                raise ValueError("reference_valid cannot mark non-finite joints as valid")
            object.__setattr__(self, "reference", _readonly(np.where(reference_valid[..., None], reference, 0.0), dtype=np.float32))
            object.__setattr__(self, "reference_valid", _readonly(reference_valid, dtype=bool))
        if self.transform_a is not None and len(self.transform_a.valid) != frames:
            raise ValueError("transform_a must cover every frame")
        object.__setattr__(self, "view_a", _readonly(np.where(valid_a[..., None], view_a, 0.0), dtype=np.float32))
        object.__setattr__(self, "view_b", _readonly(np.where(valid_b[..., None], view_b, 0.0), dtype=np.float32))
        object.__setattr__(self, "valid_a", _readonly(valid_a, dtype=bool))
        object.__setattr__(self, "valid_b", _readonly(valid_b, dtype=bool))
        object.__setattr__(self, "timestamps", _readonly(timestamps, dtype=np.float64))
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(self, "cycle_bounds", bounds)
        object.__setattr__(self, "cycle_mids", mids)
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))

    @property
    def num_frames(self) -> int:
        """Number of frames ``T``."""
        return int(self.view_a.shape[0])

    @property
    def num_joints(self) -> int:
        """Number of joints ``J``."""
        return int(self.view_a.shape[1])

    @property
    def has_cycles(self) -> bool:
        """Whether at least one complete cycle is annotated."""
        return bool(self.cycle_bounds)

    @property
    def has_cycle_mids(self) -> bool:
        """Whether every annotated cycle carries a middle frame."""
        return bool(self.cycle_bounds) and len(self.cycle_mids) == len(self.cycle_bounds)

    @property
    def key(self) -> str:
        """Stable identifier ``dataset/subject/sequence``."""
        return f"{self.dataset}/{self.subject_id}/{self.sequence_id}"


class FusionBatch(TypedDict, total=False):
    """Collated tensor batch consumed by the model and the losses.

    See the module docstring for the shape of every field.  Fields marked
    optional are present only when the DataModule produced them.
    """

    pose_a: torch.Tensor
    pose_b: torch.Tensor
    valid_a: torch.Tensor
    valid_b: torch.Tensor
    frame_mask: torch.Tensor
    delta_t: torch.Tensor
    timestamps: torch.Tensor
    phase: torch.Tensor
    phase_valid: torch.Tensor
    cycle_index: torch.Tensor
    half_index: torch.Tensor
    clean_a: torch.Tensor
    clean_b: torch.Tensor
    clean_valid_a: torch.Tensor
    clean_valid_b: torch.Tensor
    corruption_mask_a: torch.Tensor
    corruption_mask_b: torch.Tensor
    reference: torch.Tensor
    reference_valid: torch.Tensor
    reference_canonical: torch.Tensor
    cycle_target: torch.Tensor
    cycle_confidence: torch.Tensor
    window_start: torch.Tensor
    dataset: list[str]
    subject_id: list[str]
    sequence_id: list[str]
    window_id: list[str]


@lru_cache(maxsize=1)
def _mhr70_spec():
    from gymnastics.keypoints.config import load_skeleton_spec

    return load_skeleton_spec(CONFIG_ROOT / "fusion" / "skeleton_mhr70.yaml")


def canonicalize_view(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, CanonicalTransformRecord]:
    """Canonicalise one ``[T, 70, 3]`` MHR70 view into the pelvis body frame.

    This is a thin NumPy wrapper around the differentiable canonicalisation of
    the rotation-aware package so both packages share one definition of the
    body frame.

    Args:
        points: ``[T, 70, 3]`` world-frame keypoints of one view.
        valid: ``[T, 70]`` joint validity.

    Returns:
        Tuple ``(canonical_points, canonical_valid, transform)``.  Frames in
        which the pelvis frame could not be observed are marked invalid.
    """
    from gymnastics.keypoints.geometry import canonicalize_pose

    tensor = torch.from_numpy(np.array(points, dtype=np.float32, copy=True)).unsqueeze(0)
    mask = torch.from_numpy(np.array(valid, dtype=bool, copy=True)).unsqueeze(0)
    canonical = canonicalize_pose(tensor, mask, _mhr70_spec())
    canonical_valid = canonical.valid & canonical.transform.valid[..., None]
    canonical_points = torch.where(canonical_valid[..., None], canonical.points, torch.zeros_like(canonical.points))
    transform = CanonicalTransformRecord(
        rotation=canonical.transform.rotation[0].numpy(),
        origin=canonical.transform.origin[0].numpy(),
        scale=float(canonical.transform.scale[0]),
        valid=canonical.transform.valid[0].numpy(),
    )
    return canonical_points[0].numpy(), canonical_valid[0].numpy(), transform


def sample_from_pose_pair_trial(
    trial: PosePairTrial,
    skeleton: CommonSkeleton,
    *,
    dataset: str,
    cycle_bounds: Sequence[tuple[int, int]] = (),
    cycle_mids: Sequence[int] = (),
    canonicalize: bool = True,
    reference: np.ndarray | None = None,
    reference_valid: np.ndarray | None = None,
    subject_id: str | None = None,
    sequence_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DualViewSample:
    """Convert a repository ``PosePairTrial`` into a :class:`DualViewSample`.

    The trial is the MHR70 contract shared by every SAM3D-based loader of the
    repository.  Conversion canonicalises each view (unless ``canonicalize``
    is false), then selects the common skeleton's joints.

    Args:
        trial: Synchronised face/side (View A/View B) MHR70 trial.
        skeleton: Target common skeleton.
        dataset: Dataset identifier stored in the sample.
        cycle_bounds: Complete cycles as ``(start, end)`` frame ranges.
        cycle_mids: Turn-around frame per cycle (empty when unknown).
        canonicalize: Whether to map each view into the pelvis body frame.
        reference: Optional ``[T, 70, 3]`` reference pose in its native frame.
        reference_valid: Optional ``[T, 70]`` validity of ``reference``.
        subject_id: Override for ``trial.person_id``.
        sequence_id: Override for ``trial.trial_id``.
        metadata: Extra provenance merged over ``trial.source_metadata``.

    Returns:
        The converted sample.
    """
    face, side = np.asarray(trial.face, dtype=np.float32), np.asarray(trial.side, dtype=np.float32)
    valid_face, valid_side = np.asarray(trial.valid_face, dtype=bool), np.asarray(trial.valid_side, dtype=bool)
    transform_a: CanonicalTransformRecord | None = None
    if canonicalize:
        face, valid_face, transform_a = canonicalize_view(face, valid_face)
        side, valid_side, _ = canonicalize_view(side, valid_side)
    select = list(skeleton.source_indices)
    provenance: dict[str, Any] = {k: v for k, v in dict(trial.source_metadata).items()}
    provenance.update({"canonicalized": bool(canonicalize), "skeleton": skeleton.name, "fps": float(trial.fps)})
    if metadata:
        provenance.update(dict(metadata))
    return DualViewSample(
        dataset=dataset,
        subject_id=subject_id or str(trial.person_id),
        sequence_id=sequence_id or str(trial.trial_id),
        view_a=face[:, select],
        view_b=side[:, select],
        valid_a=valid_face[:, select],
        valid_b=valid_side[:, select],
        timestamps=np.asarray(trial.timestamps, dtype=np.float64),
        joint_names=skeleton.joint_names,
        cycle_bounds=tuple(cycle_bounds),
        cycle_mids=tuple(cycle_mids),
        reference=None if reference is None else np.asarray(reference, dtype=np.float32)[:, select],
        reference_valid=None if reference_valid is None else np.asarray(reference_valid, dtype=bool)[:, select],
        transform_a=transform_a,
        metadata=provenance,
    )


def collate_fusion_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack tensor fields of window dictionaries; keep string fields as lists.

    Args:
        samples: Window dictionaries produced by the windowing dataset.

    Returns:
        A :class:`FusionBatch`-shaped dictionary.

    Raises:
        ValueError: If the batch is empty or the windows disagree on fields.
    """
    if not samples:
        raise ValueError("cannot collate an empty batch")
    keys = set(samples[0])
    for sample in samples[1:]:
        if set(sample) != keys:
            raise ValueError("all windows in a batch must expose the same fields")
    batch: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        batch[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else list(values)
    return batch
