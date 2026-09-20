"""Synthetic input corruption for self-supervised recovery training.

The fusion model is trained without 3D ground truth.  Its main objective is
*recovery*: given two views of which one (or both) has been synthetically
damaged, reproduce the clean pseudo-target built from the undamaged views.
This module implements the damage.

Research Motivation:
    Each corruption family imitates a failure mode observed in monocular
    SAM3D-Body output on the project's data:

    ``joint_mask``          isolated missing joints (detector drop-outs);
    ``distal_mask``         hands or feet missing for a stretch of frames
                            (truncation at the image border, self-occlusion);
    ``gaussian_noise``      per-joint jitter;
    ``depth_perturbation``  a slowly drifting error along one direction, the
                            monocular depth ambiguity;
    ``temporal_dropout``    single frames lost (tracking failure);
    ``contiguous_dropout``  a block of frames lost.

    Because the corruption is applied to a *copy* of the input while the
    clean copy is kept for the target, the training signal is exact and
    label-free.  Each window uses a seed derived from its stable identifier
    and the epoch, so a training run is reproducible and a validation suite
    replays the same damage every epoch.

Depth direction:
    Inputs are in the canonical body frame, in which the camera's optical
    axis is not fixed.  The depth perturbation therefore uses one random unit
    direction per corrupted view, drawn from the same generator, which
    approximates an unknown camera axis.

Shapes (one view of one window):
    pose    [T, J, 3]   float32
    valid   [T, J]      bool
    mask    [T, J]      bool  true where the value or validity was altered
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from .skeleton import CommonSkeleton

CORRUPTION_FAMILIES: tuple[str, ...] = (
    "joint_mask",
    "distal_mask",
    "gaussian_noise",
    "depth_perturbation",
    "temporal_dropout",
    "contiguous_dropout",
)


@dataclass(frozen=True)
class CorruptionConfig:
    """Parameters of the corruption families (see ``src/configs/fusion/corruption``).

    Attributes:
        enabled: Master switch.
        families: Enabled family names.
        view_probability: Probability that a given view of a window is
            corrupted at all (each view decided independently).
        joint_mask_probability: Per-(frame, joint) drop probability.
        distal_mask_probability: Probability that a contiguous distal block
            occurs in the view.
        distal_block_length: Length in samples of the distal block.
        gaussian_noise_probability: Probability that a joint trajectory is
            jittered.
        gaussian_noise_std: Jitter standard deviation (canonical units).
        depth_probability: Probability of the depth perturbation.
        depth_shift_std: Standard deviation of the per-step random-walk
            increment of the depth offset (canonical units).
        depth_scale_std: Standard deviation of the multiplicative depth scale
            error (dimensionless).
        temporal_dropout_probability: Per-frame probability of losing every
            joint of the frame.
        contiguous_dropout_probability: Probability of a contiguous block of
            lost frames.
        contiguous_block_length: Length in samples of that block.
    """

    enabled: bool = True
    families: tuple[str, ...] = CORRUPTION_FAMILIES
    view_probability: float = 0.5
    joint_mask_probability: float = 0.05
    distal_mask_probability: float = 0.3
    distal_block_length: int = 8
    gaussian_noise_probability: float = 0.3
    gaussian_noise_std: float = 0.02
    depth_probability: float = 0.3
    depth_shift_std: float = 0.01
    depth_scale_std: float = 0.1
    temporal_dropout_probability: float = 0.03
    contiguous_dropout_probability: float = 0.2
    contiguous_block_length: int = 8

    def __post_init__(self) -> None:
        families = tuple(str(name) for name in self.families)
        unknown = sorted(set(families) - set(CORRUPTION_FAMILIES))
        if unknown:
            raise ValueError(f"unknown corruption families: {unknown}")
        object.__setattr__(self, "families", families)
        for name in (
            "view_probability",
            "joint_mask_probability",
            "distal_mask_probability",
            "gaussian_noise_probability",
            "depth_probability",
            "temporal_dropout_probability",
            "contiguous_dropout_probability",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.distal_block_length < 1 or self.contiguous_block_length < 1:
            raise ValueError("block lengths must be positive")
        if min(self.gaussian_noise_std, self.depth_shift_std, self.depth_scale_std) < 0:
            raise ValueError("noise scales must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "CorruptionConfig" | None) -> "CorruptionConfig":
        """Build from a mapping (e.g. an OmegaConf node); ``None`` gives defaults."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)  # type: ignore[assignment]
        except ImportError:  # pragma: no cover
            pass
        payload = dict(value)
        if "families" in payload and payload["families"] is not None:
            payload["families"] = tuple(payload["families"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary representation."""
        return asdict(self)


def stable_seed(seed: int, window_id: str, epoch: int = 0) -> int:
    """Derive a reproducible, order-independent seed for one window and epoch."""
    digest = hashlib.sha256(f"{int(seed)}:{int(epoch)}:{window_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _bernoulli(shape: tuple[int, ...], probability: float, generator: torch.Generator) -> torch.Tensor:
    return torch.rand(shape, generator=generator) < probability


def corrupt_view(
    pose: torch.Tensor,
    valid: torch.Tensor,
    generator: torch.Generator,
    config: CorruptionConfig,
    skeleton: CommonSkeleton,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrupt one view of one window.

    Args:
        pose: ``[T, J, 3]`` clean keypoints (CPU).
        valid: ``[T, J]`` bool validity.
        generator: Seeded CPU generator.
        config: Corruption parameters; every family in ``config.families``
            is sampled once.
        skeleton: Provides the distal joint indices.

    Returns:
        Tuple ``(corrupted_pose, corrupted_valid, mask)``; ``mask`` is true
        where either the value or the validity changed.
    """
    if pose.ndim != 3 or pose.shape[-1] != 3 or valid.shape != pose.shape[:2]:
        raise ValueError("pose must have shape [T, J, 3] and valid [T, J]")
    if pose.device.type != "cpu":
        raise ValueError("corruption runs on CPU tensors")
    frames, joints = valid.shape
    out, out_valid = pose.clone(), valid.clone()
    families = set(config.families)

    if "joint_mask" in families:
        drop = _bernoulli((frames, joints), config.joint_mask_probability, generator) & out_valid
        out[drop] = 0.0
        out_valid[drop] = False

    if "distal_mask" in families and skeleton.distal_indices and bool(_bernoulli((), config.distal_mask_probability, generator)):
        length = min(config.distal_block_length, frames)
        start = int(torch.randint(frames - length + 1, (), generator=generator))
        distal = torch.tensor(skeleton.distal_indices)
        # Drop a random subset (at least one) of the distal joints.
        chosen = distal[_bernoulli((len(distal),), 0.5, generator)]
        if chosen.numel() == 0:
            chosen = distal[int(torch.randint(len(distal), (), generator=generator))][None]
        block = torch.zeros_like(out_valid)
        block[start : start + length, chosen] = True
        block &= out_valid
        out[block] = 0.0
        out_valid[block] = False

    if "gaussian_noise" in families:
        selected = _bernoulli((joints,), config.gaussian_noise_probability, generator)
        noise = torch.randn(out.shape, generator=generator, dtype=out.dtype) * config.gaussian_noise_std
        out = out + noise * (selected[None, :, None] & out_valid[..., None])

    if "depth_perturbation" in families and bool(_bernoulli((), config.depth_probability, generator)):
        direction = torch.randn(3, generator=generator, dtype=out.dtype)
        direction = direction / direction.norm().clamp_min(1e-6)
        # Smooth drift: cumulative sum of small increments, plus a global
        # multiplicative error of the coordinate along the depth direction.
        increments = torch.randn(frames, generator=generator, dtype=out.dtype) * config.depth_shift_std
        offset = torch.cumsum(increments, dim=0)
        scale = torch.randn((), generator=generator, dtype=out.dtype) * config.depth_scale_std
        depth = (out * direction).sum(dim=-1, keepdim=True)
        shift = (offset[:, None, None] + scale * depth) * direction
        out = out + shift * out_valid[..., None]

    if "temporal_dropout" in families:
        frames_dropped = _bernoulli((frames,), config.temporal_dropout_probability, generator)
        drop = frames_dropped[:, None] & out_valid
        out[drop] = 0.0
        out_valid[drop] = False

    if "contiguous_dropout" in families and bool(_bernoulli((), config.contiguous_dropout_probability, generator)):
        length = min(config.contiguous_block_length, frames)
        start = int(torch.randint(frames - length + 1, (), generator=generator))
        drop = torch.zeros_like(out_valid)
        drop[start : start + length] = True
        drop &= out_valid
        out[drop] = 0.0
        out_valid[drop] = False

    mask = (out != pose).any(dim=-1) | (out_valid != valid)
    return out, out_valid, mask


def corrupt_window(
    pose_a: torch.Tensor,
    pose_b: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
    *,
    seed: int,
    config: CorruptionConfig,
    skeleton: CommonSkeleton,
) -> dict[str, torch.Tensor]:
    """Corrupt the two views of one window with a shared seed.

    Each view is corrupted with probability ``config.view_probability``;
    the decision and all random draws come from one generator seeded with
    ``seed`` so the outcome is a pure function of the seed.

    Args:
        pose_a: ``[T, J, 3]`` clean View A.
        pose_b: ``[T, J, 3]`` clean View B.
        valid_a: ``[T, J]`` validity of View A.
        valid_b: ``[T, J]`` validity of View B.
        seed: Generator seed (see :func:`stable_seed`).
        config: Corruption parameters.
        skeleton: Common skeleton.

    Returns:
        Dictionary with ``pose_a``, ``pose_b``, ``valid_a``, ``valid_b``
        (corrupted) and ``corruption_mask_a``, ``corruption_mask_b``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    result: dict[str, torch.Tensor] = {}
    for name, pose, valid in (("a", pose_a, valid_a), ("b", pose_b, valid_b)):
        if config.enabled and bool(_bernoulli((), config.view_probability, generator)):
            corrupted, corrupted_valid, mask = corrupt_view(pose, valid, generator, config, skeleton)
        else:
            corrupted, corrupted_valid, mask = pose.clone(), valid.clone(), torch.zeros_like(valid)
        result[f"pose_{name}"] = corrupted
        result[f"valid_{name}"] = corrupted_valid
        result[f"corruption_mask_{name}"] = mask
    return result
