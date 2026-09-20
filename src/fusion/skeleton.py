"""Common skeleton definition for cycle-aware dual-view fusion.

Every dataset adapter converts its input poses to one *common* skeleton so
that the fusion model is dataset-agnostic.  The model itself only needs the
number of joints ``J``; the loss functions additionally need the bone list
(for bilateral bone-length symmetry) and the left/right joint pairs.  The
corruption pipeline needs to know which joints are *distal* (hands and feet)
because those are the joints monocular estimators lose most often.

Research Motivation:
    All three data sources of this project (private gymnastics recordings,
    the public FreeMan release and the Unity synthetic benchmark) are
    processed by the same monocular estimator, SAM3D-Body, whose output is
    the 70-joint MHR70 skeleton.  Keeping MHR70 as the common representation
    therefore requires no re-targeting of the *inputs*; only the evaluation
    references (COCO17 for FreeMan, Unity16 for Unity, triangulated MHR70 for
    the private data) differ, and those are mapped by the dataset adapters
    into MHR70 index positions with a validity mask.

    Two variants are provided:

    * ``mhr70``        the full 70-joint set (fingers, face, feet).
    * ``mhr70_major``  the 20 major body joints that the repository already
                       uses for cohort statistics (``MHR70_MAJOR_JOINT_INDICES``).
                       It removes the 40 finger joints, which are extremely
                       noisy in whole-body video and dominate ``J`` while
                       carrying little information about trunk motion.

    The variant is chosen through Hydra (``model.skeleton``) and every module
    that consumes joint indices reads them from :class:`CommonSkeleton`.

Coordinate System:
    The skeleton does not fix a coordinate system.  Dataset adapters place
    both views in a pelvis-centred canonical body frame (see
    :mod:`fusion.sample`).

Notes:
    * Bones cover the head, torso, limbs and feet.  Hand-internal bones are
      intentionally excluded from the bone list because finger geometry is
      not a useful bilateral-symmetry constraint at whole-body scale.
    * ``left_right_pairs`` are derived from joint names (``left-*`` versus
      ``right-*``) so the mapping cannot silently drift from the joint list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from common.skeletons.mhr70 import MHR70_MAJOR_JOINT_INDICES, mhr_names

MHR70_JOINT_NAMES: tuple[str, ...] = tuple(mhr_names)

# Kinematic bones shared by both skeleton variants, expressed by joint name so
# the same list is valid for the full set and for any subset.
_MHR70_BONE_NAMES: tuple[tuple[str, str], ...] = (
    ("neck", "nose"),
    ("nose", "left-eye"),
    ("nose", "right-eye"),
    ("left-eye", "left-ear"),
    ("right-eye", "right-ear"),
    ("neck", "left-shoulder"),
    ("neck", "right-shoulder"),
    ("left-shoulder", "left-elbow"),
    ("left-elbow", "left-wrist"),
    ("right-shoulder", "right-elbow"),
    ("right-elbow", "right-wrist"),
    ("left-shoulder", "left-hip"),
    ("right-shoulder", "right-hip"),
    ("left-hip", "right-hip"),
    ("left-hip", "left-knee"),
    ("left-knee", "left-ankle"),
    ("right-hip", "right-knee"),
    ("right-knee", "right-ankle"),
    ("left-ankle", "left-heel"),
    ("left-ankle", "left-big-toe-tip"),
    ("left-ankle", "left-small-toe-tip"),
    ("right-ankle", "right-heel"),
    ("right-ankle", "right-big-toe-tip"),
    ("right-ankle", "right-small-toe-tip"),
)

# Distal segments: hands (wrist plus fingers) and feet (heel plus toes).  These
# are the joints most frequently occluded or truncated in monocular video.
_DISTAL_PREFIXES = ("left-thumb", "left-index", "left-middle", "left-ring", "left-pinky",
                    "right-thumb", "right-index", "right-middle", "right-ring", "right-pinky")
_DISTAL_EXACT = (
    "left-wrist", "right-wrist",
    "left-heel", "right-heel",
    "left-big-toe-tip", "left-small-toe-tip",
    "right-big-toe-tip", "right-small-toe-tip",
)


@dataclass(frozen=True)
class CommonSkeleton:
    """The joint set every dataset is converted to before fusion.

    Attributes:
        name: Identifier of the variant (``"mhr70"`` or ``"mhr70_major"``).
        joint_names: Ordered joint names; ``J = len(joint_names)``.
        source_indices: For every common joint, its index in the MHR70 joint
            list.  Dataset adapters use it to select the subset from the
            ``[T, 70, 3]`` SAM3D output.
        bones: Index pairs ``(parent, child)`` into ``joint_names``.
        left_right_pairs: Index pairs ``(left, right)`` of mirrored joints.
        distal_indices: Indices of hand and foot joints.
        left_hip_index: Index of the left hip joint (pelvis anchor).
        right_hip_index: Index of the right hip joint (pelvis anchor).

    Example:
        >>> skeleton = build_common_skeleton("mhr70_major")
        >>> skeleton.num_joints
        20
        >>> skeleton.joint_names[skeleton.left_hip_index]
        'left-hip'
    """

    name: str
    joint_names: tuple[str, ...]
    source_indices: tuple[int, ...]
    bones: tuple[tuple[int, int], ...]
    left_right_pairs: tuple[tuple[int, int], ...]
    distal_indices: tuple[int, ...]
    left_hip_index: int
    right_hip_index: int
    _index: Mapping[str, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CommonSkeleton requires a name")
        if len(self.joint_names) != len(set(self.joint_names)):
            raise ValueError("CommonSkeleton joint names must be unique")
        if len(self.source_indices) != len(self.joint_names):
            raise ValueError("source_indices must have one entry per joint")
        joints = len(self.joint_names)
        for start, end in self.bones:
            if not (0 <= start < joints and 0 <= end < joints) or start == end:
                raise ValueError(f"invalid bone ({start}, {end}) for {joints} joints")
        for left, right in self.left_right_pairs:
            if not (0 <= left < joints and 0 <= right < joints) or left == right:
                raise ValueError(f"invalid left/right pair ({left}, {right})")
        if any(not 0 <= index < joints for index in self.distal_indices):
            raise ValueError("distal_indices are outside the joint list")
        if not (0 <= self.left_hip_index < joints and 0 <= self.right_hip_index < joints):
            raise ValueError("hip indices are outside the joint list")
        object.__setattr__(self, "_index", {name: i for i, name in enumerate(self.joint_names)})

    @property
    def num_joints(self) -> int:
        """Number of joints ``J`` in the common representation."""
        return len(self.joint_names)

    def index(self, joint_name: str) -> int:
        """Return the index of ``joint_name``.

        Args:
            joint_name: Name from :attr:`joint_names`.

        Returns:
            Zero-based joint index.

        Raises:
            KeyError: If the joint is not part of this skeleton.
        """
        try:
            return self._index[joint_name]
        except KeyError as error:
            raise KeyError(f"unknown joint {joint_name!r} in skeleton {self.name}") from error

    def bilateral_bone_pairs(self) -> tuple[tuple[int, int], ...]:
        """Return pairs of bone indices ``(left_bone, right_bone)``.

        A bone belongs to the left side when at least one of its joints is a
        ``left-*`` joint and none is a ``right-*`` joint; its mirror bone is
        found by swapping the side prefix of both joint names.  Bones that
        touch the mid-line from both sides (``left-hip``/``right-hip``) are
        their own mirror and are excluded.

        Returns:
            Tuple of ``(left_bone_index, right_bone_index)`` pairs into
            :attr:`bones`.
        """
        by_names = {
            (self.joint_names[a], self.joint_names[b]): i for i, (a, b) in enumerate(self.bones)
        }
        pairs: list[tuple[int, int]] = []
        for (name_a, name_b), bone_index in by_names.items():
            if "right-" in (name_a + name_b):
                continue
            if "left-" not in (name_a + name_b):
                continue
            mirror = (name_a.replace("left-", "right-"), name_b.replace("left-", "right-"))
            if mirror in by_names:
                pairs.append((bone_index, by_names[mirror]))
        return tuple(pairs)


def _left_right_pairs(names: Iterable[str]) -> tuple[tuple[int, int], ...]:
    names = tuple(names)
    index = {name: i for i, name in enumerate(names)}
    pairs = []
    for name in names:
        if name.startswith("left-"):
            mirror = "right-" + name[len("left-"):]
            if mirror in index:
                pairs.append((index[name], index[mirror]))
    return tuple(pairs)


def _subset_skeleton(name: str, source_indices: Iterable[int]) -> CommonSkeleton:
    source = tuple(int(i) for i in source_indices)
    names = tuple(MHR70_JOINT_NAMES[i] for i in source)
    index = {joint: i for i, joint in enumerate(names)}
    bones = tuple(
        (index[a], index[b]) for a, b in _MHR70_BONE_NAMES if a in index and b in index
    )
    distal = tuple(
        i for i, joint in enumerate(names)
        if joint in _DISTAL_EXACT or joint.startswith(_DISTAL_PREFIXES)
    )
    return CommonSkeleton(
        name=name,
        joint_names=names,
        source_indices=source,
        bones=bones,
        left_right_pairs=_left_right_pairs(names),
        distal_indices=distal,
        left_hip_index=index["left-hip"],
        right_hip_index=index["right-hip"],
    )


SKELETON_VARIANTS: tuple[str, ...] = ("mhr70", "mhr70_major")


def build_common_skeleton(name: str = "mhr70") -> CommonSkeleton:
    """Build one of the supported common skeleton variants.

    Args:
        name: ``"mhr70"`` for all 70 SAM3D-Body joints or ``"mhr70_major"``
            for the 20 major body joints.

    Returns:
        The :class:`CommonSkeleton` for that variant.

    Raises:
        ValueError: If ``name`` is not a supported variant.
    """
    if name == "mhr70":
        return _subset_skeleton("mhr70", range(len(MHR70_JOINT_NAMES)))
    if name == "mhr70_major":
        return _subset_skeleton("mhr70_major", MHR70_MAJOR_JOINT_INDICES)
    raise ValueError(f"unknown skeleton variant {name!r}; expected one of {SKELETON_VARIANTS}")
