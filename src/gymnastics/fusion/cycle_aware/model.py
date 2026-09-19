"""CycleAwareFusionModel: the complete dual-view fusion network (v1.0).

This module assembles the building blocks of :mod:`.modules` into the pure
PyTorch model.  It has no knowledge of datasets, losses or training loops;
those live in :mod:`.data`, :mod:`.losses` and :mod:`.lightning_module`.

Pipeline (one shared encoder applied to each view):

    P_A, P_B [B, T, J, 3], valid masks, delta_t, phase
        |
        |-- velocity: v = (P[t] - P[t-1]) / delta_t[t]                (velocity.py)
        |-- F_pose  = SpatialTransformer(P, valid)                    (spatial_transformer.py)
        |-- F_short = ShortMotionTransformer(P, v, valid)             (short_motion.py)
        |-- F_long  = LongMotionTransformer(P, v, valid, phase enc.)  (long_motion.py)
        |-- F_motion = MLP([F_short ; F_long])                        (motion_fusion.py)
        |-- H = (1 + gamma(F_motion)) * F_pose + beta(F_motion)       (film.py)
        |
    C_A, C_B = CrossViewAttention(H_A, H_B)                           (cross_view_attention.py)
    [w_A, w_B] = softmax(R(C_A, C_B))                                 (reliability.py)
    P_base = w_A * P_A + w_B * P_B                                    (weighted_fusion.py)
    Delta_P = Residual(C_A, C_B, w, P_base)                           (residual_refinement.py)
    P_hat = P_base + Delta_P

Ablation switches (all through :class:`CycleAwareModelConfig`, i.e. Hydra):
    short_motion.enabled   F_short := 0
    long_motion.enabled    F_long  := 0
    phase_encoding.enabled phase channels := 0
    film.enabled           H := F_pose (motion ignored)
    cross_view.enabled     C := H (no exchange)
    reliability.enabled    R := 0 (equal weights up to validity)
    residual.enabled       Delta_P := 0 (pure weighted fusion)

Assumptions:
    * Inputs are expressed in the canonical body frame produced by the
      dataset adapters (see :mod:`.sample`), so the two views are directly
      comparable without calibration.
    * ``delta_t`` carries physical seconds so velocities are physical even
      after phase normalisation.
    * The window length ``T`` is arbitrary at inference; during training it is
      ``long_motion.num_cycles * samples_per_cycle`` samples.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .modules import (
    BidirectionalCrossViewAttention,
    FiLMMotionGuidance,
    JointReliabilityHead,
    LongMotionTransformer,
    MotionFusion,
    ResidualRefinement,
    ShortMotionTransformer,
    SpatialTransformer,
    weighted_pose_fusion,
)
from .outputs import PoseFusionOutput
from .phase import PhaseEncoding
from .skeleton import CommonSkeleton, build_common_skeleton
from .velocity import BOUNDARY_STRATEGIES, compute_velocity


@dataclass(frozen=True)
class SpatialConfig:
    """Pose Branch settings.

    Attributes:
        layers: Number of transformer blocks over joints.
    """

    layers: int = 2


@dataclass(frozen=True)
class ShortMotionConfig:
    """Short-term motion branch settings.

    Attributes:
        enabled: Whether the branch is used (ablation switch).
        layers: Number of temporal transformer blocks.
        cycle_ratio: Fraction of one phase-normalised cycle visible to the
            local attention band (0.25 = a quarter cycle).
    """

    enabled: bool = True
    layers: int = 2
    cycle_ratio: float = 0.25


@dataclass(frozen=True)
class LongMotionConfig:
    """Long-term motion branch settings.

    Attributes:
        enabled: Whether the branch is used (ablation switch).
        layers: Number of temporal transformer blocks.
        num_cycles: Number of complete cycles the training window spans
            (0.5, 1, 2, ...); the DataModule builds windows of
            ``num_cycles * samples_per_cycle`` samples.  ``None`` selects the
            full sequence as context (one window per sequence).
    """

    enabled: bool = True
    layers: int = 2
    num_cycles: float | None = 2.0


@dataclass(frozen=True)
class PhaseEncodingConfig:
    """Phase encoding settings.

    Attributes:
        enabled: Adds ``sin(2 pi h phi), cos(2 pi h phi)`` channels.
        harmonics: Number of harmonics ``H``.
    """

    enabled: bool = True
    harmonics: int = 1


@dataclass(frozen=True)
class FiLMConfig:
    """FiLM motion-guidance settings.

    Attributes:
        enabled: When false the pose feature bypasses modulation.
        gamma_bound: Optional ``tanh`` bound on ``gamma``.
    """

    enabled: bool = True
    gamma_bound: float | None = None


@dataclass(frozen=True)
class CrossViewConfig:
    """Bidirectional cross-view attention settings.

    Attributes:
        enabled: When false the views are not exchanged.
        layers: Number of cross-attention blocks.
    """

    enabled: bool = True
    layers: int = 1


@dataclass(frozen=True)
class ReliabilityConfig:
    """Joint-wise reliability settings.

    Attributes:
        enabled: When false logits are zero (equal weights up to validity).
    """

    enabled: bool = True


@dataclass(frozen=True)
class ResidualConfig:
    """Residual refinement settings.

    Attributes:
        enabled: When false ``Delta_P`` is zero.
        max_delta: Bound radius in canonical units (``None`` = unbounded).
    """

    enabled: bool = True
    max_delta: float | None = 0.25


@dataclass(frozen=True)
class CycleAwareModelConfig:
    """Complete model configuration (mirrors ``configs/cycle_aware/model``).

    Attributes:
        skeleton: Common skeleton variant name.
        hidden_dim: Feature width ``D``.
        num_heads: Attention heads in every transformer.
        mlp_ratio: Feed-forward width relative to ``D``.
        dropout: Dropout probability inside attention and MLPs.
        samples_per_cycle: Samples per phase-normalised cycle ``S``.
        velocity_boundary: First-sample velocity strategy.
        spatial, short_motion, long_motion, phase_encoding, film,
        cross_view, reliability, residual: Sub-module settings.
    """

    skeleton: str = "mhr70_major"
    hidden_dim: int = 128
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    samples_per_cycle: int = 64
    velocity_boundary: str = "replicate"
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    short_motion: ShortMotionConfig = field(default_factory=ShortMotionConfig)
    long_motion: LongMotionConfig = field(default_factory=LongMotionConfig)
    phase_encoding: PhaseEncodingConfig = field(default_factory=PhaseEncodingConfig)
    film: FiLMConfig = field(default_factory=FiLMConfig)
    cross_view: CrossViewConfig = field(default_factory=CrossViewConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.num_heads <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if self.samples_per_cycle < 2:
            raise ValueError("samples_per_cycle must be at least 2")
        if self.velocity_boundary not in BOUNDARY_STRATEGIES:
            raise ValueError(f"velocity_boundary must be one of {BOUNDARY_STRATEGIES}")
        if self.long_motion.num_cycles is not None and self.long_motion.num_cycles <= 0:
            raise ValueError("long_motion.num_cycles must be positive or None")

    @property
    def window_length(self) -> int | None:
        """Training window length ``num_cycles * samples_per_cycle`` (``None`` = full sequence)."""
        if self.long_motion.num_cycles is None:
            return None
        return max(2, int(round(self.long_motion.num_cycles * self.samples_per_cycle)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "CycleAwareModelConfig") -> "CycleAwareModelConfig":
        """Build the config from a (possibly nested) mapping such as an OmegaConf node.

        Args:
            value: Mapping with the field names of this dataclass; nested
                mappings are converted to the nested dataclasses.  Unknown
                keys raise so that Hydra typos are caught early.

        Returns:
            The validated configuration.
        """
        if isinstance(value, cls):
            return value
        return _dataclass_from_mapping(cls, value)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serialisable dictionary."""
        return asdict(self)


def _dataclass_from_mapping(cls: type, value: Mapping[str, Any]) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)  # type: ignore[assignment]
    except ImportError:  # pragma: no cover - omegaconf is a hard dependency
        pass
    if not isinstance(value, Mapping):
        raise TypeError(f"{cls.__name__} expects a mapping, got {type(value).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(value) - set(known))
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
    kwargs: dict[str, Any] = {}
    for name, item in value.items():
        factory = known[name].default_factory
        default = factory() if factory is not MISSING else None
        if isinstance(item, Mapping) and default is not None and is_dataclass(default):
            # Nested section (e.g. ``film: {enabled: false}``) -> nested dataclass.
            kwargs[name] = _dataclass_from_mapping(type(default), item)
        else:
            kwargs[name] = item
    return cls(**kwargs)


class CycleAwareFusionModel(nn.Module):
    """Dual-view 3D pose fusion with cycle-aware motion guidance.

    Attributes:
        config: The :class:`CycleAwareModelConfig` used to build the model.
        skeleton: The :class:`CommonSkeleton` (``J = skeleton.num_joints``).
        spatial: Shared Pose Branch.
        short_motion: Shared short-term motion branch (or ``None``).
        long_motion: Shared long-term motion branch (or ``None``).
        phase_encoding: Phase encoder (or ``None``).
        motion_fusion: ``F_motion = MLP([F_short ; F_long])``.
        film: FiLM guidance (or ``None``).
        cross_view: Bidirectional cross-view attention (or ``None``).
        reliability: Joint-wise reliability head.
        residual: Residual refinement head.

    Example:
        >>> model = CycleAwareFusionModel(CycleAwareModelConfig(hidden_dim=32, samples_per_cycle=8))
        >>> out = model(pose_a, pose_b, valid_a, valid_b, delta_t, phase, phase_valid)
        >>> out.pose.shape  # [B, T, J, 3]
    """

    def __init__(self, config: CycleAwareModelConfig | Mapping[str, Any] | None = None, *, skeleton: CommonSkeleton | None = None) -> None:
        super().__init__()
        self.config = CycleAwareModelConfig.from_mapping(config) if config is not None else CycleAwareModelConfig()
        self.skeleton = skeleton or build_common_skeleton(self.config.skeleton)
        joints, dim = self.skeleton.num_joints, self.config.hidden_dim
        common = dict(heads=self.config.num_heads, mlp_ratio=self.config.mlp_ratio, dropout=self.config.dropout)
        self.spatial = SpatialTransformer(joints, dim, layers=self.config.spatial.layers, **common)
        self.short_motion = (
            ShortMotionTransformer(
                joints,
                dim,
                samples_per_cycle=self.config.samples_per_cycle,
                cycle_ratio=self.config.short_motion.cycle_ratio,
                layers=self.config.short_motion.layers,
                **common,
            )
            if self.config.short_motion.enabled
            else None
        )
        self.phase_encoding = PhaseEncoding(self.config.phase_encoding.harmonics) if self.config.phase_encoding.enabled else None
        phase_channels = self.phase_encoding.channels if self.phase_encoding is not None else 0
        self.long_motion = (
            LongMotionTransformer(joints, dim, phase_channels=phase_channels, layers=self.config.long_motion.layers, **common)
            if self.config.long_motion.enabled
            else None
        )
        self.motion_fusion = MotionFusion(dim, dropout=self.config.dropout)
        self.film = FiLMMotionGuidance(dim, gamma_bound=self.config.film.gamma_bound) if self.config.film.enabled else None
        self.cross_view = (
            BidirectionalCrossViewAttention(dim, layers=self.config.cross_view.layers, **common)
            if self.config.cross_view.enabled
            else None
        )
        self.reliability = JointReliabilityHead(dim, enabled=self.config.reliability.enabled, dropout=self.config.dropout)
        self.residual = ResidualRefinement(dim, max_delta=self.config.residual.max_delta, enabled=self.config.residual.enabled, dropout=self.config.dropout)

    @property
    def num_joints(self) -> int:
        """Number of joints ``J``."""
        return self.skeleton.num_joints

    def _validate(
        self,
        pose_a: torch.Tensor,
        pose_b: torch.Tensor,
        valid_a: torch.Tensor,
        valid_b: torch.Tensor,
        delta_t: torch.Tensor,
        frame_mask: torch.Tensor | None,
        phase: torch.Tensor | None,
        phase_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if pose_a.ndim != 4 or pose_a.shape[-1] != 3 or pose_a.shape[2] != self.num_joints:
            raise ValueError(f"pose_a must have shape [B, T, {self.num_joints}, 3]")
        if pose_b.shape != pose_a.shape:
            raise ValueError("pose_b must match pose_a")
        batch, frames = pose_a.shape[:2]
        if valid_a.shape != pose_a.shape[:-1] or valid_b.shape != pose_a.shape[:-1]:
            raise ValueError("valid masks must have shape [B, T, J]")
        if frame_mask is None:
            frame_mask = torch.ones(batch, frames, dtype=torch.bool, device=pose_a.device)
        elif frame_mask.shape != (batch, frames):
            raise ValueError("frame_mask must have shape [B, T]")
        delta = torch.as_tensor(delta_t, dtype=pose_a.dtype, device=pose_a.device)
        if delta.ndim == 0:
            delta = delta.expand(batch, frames)
        elif delta.shape != (batch, frames):
            raise ValueError("delta_t must be scalar or have shape [B, T]")
        if (phase is None) != (phase_valid is None):
            raise ValueError("phase and phase_valid must be given together")
        if phase is None:
            phase = torch.zeros(batch, frames, dtype=pose_a.dtype, device=pose_a.device)
            phase_valid = torch.zeros(batch, frames, dtype=torch.bool, device=pose_a.device)
        elif phase.shape != (batch, frames) or phase_valid.shape != (batch, frames):
            raise ValueError("phase and phase_valid must have shape [B, T]")
        frame_mask = frame_mask.bool()
        valid_a = valid_a.bool() & frame_mask[..., None] & torch.isfinite(pose_a).all(dim=-1)
        valid_b = valid_b.bool() & frame_mask[..., None] & torch.isfinite(pose_b).all(dim=-1)
        pose_a = torch.where(valid_a[..., None], pose_a, torch.zeros_like(pose_a))
        pose_b = torch.where(valid_b[..., None], pose_b, torch.zeros_like(pose_b))
        return pose_a, pose_b, valid_a, valid_b, delta, phase.to(pose_a.dtype), phase_valid.bool() & frame_mask

    def encode_view(
        self,
        pose: torch.Tensor,
        valid: torch.Tensor,
        delta_t: torch.Tensor,
        phase_channels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the shared per-view encoder.

        Args:
            pose: ``[B, T, J, 3]`` canonical keypoints (zero where invalid).
            valid: ``[B, T, J]`` bool validity including the frame mask.
            delta_t: ``[B, T]`` physical sample intervals.
            phase_channels: ``[B, T, 2H]`` phase encoding or ``None``.

        Returns:
            Tuple ``(F_pose, F_short, F_long, F_motion, H)``, each
            ``[B, T, J, D]``.
        """
        velocity, _ = compute_velocity(pose, delta_t, valid, boundary=self.config.velocity_boundary)
        pose_feature = self.spatial(pose, valid)
        zeros = torch.zeros_like(pose_feature)
        short = self.short_motion(pose, velocity, valid) if self.short_motion is not None else zeros
        long = self.long_motion(pose, velocity, valid, phase_channels) if self.long_motion is not None else zeros
        motion = self.motion_fusion(short, long)
        motion = torch.where(valid[..., None], motion, torch.zeros_like(motion))
        guided = self.film(pose_feature, motion) if self.film is not None else pose_feature
        guided = torch.where(valid[..., None], guided, torch.zeros_like(guided))
        return pose_feature, short, long, motion, guided

    def forward(
        self,
        pose_a: torch.Tensor,
        pose_b: torch.Tensor,
        valid_a: torch.Tensor,
        valid_b: torch.Tensor,
        delta_t: torch.Tensor,
        phase: torch.Tensor | None = None,
        phase_valid: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> PoseFusionOutput:
        """Fuse two views.

        Args:
            pose_a: ``P_A`` with shape ``[B, T, J, 3]`` (canonical frame).
            pose_b: ``P_B`` with shape ``[B, T, J, 3]``.
            valid_a: ``[B, T, J]`` bool validity of View A joints.
            valid_b: ``[B, T, J]`` bool validity of View B joints.
            delta_t: ``[B, T]`` seconds between consecutive samples (or a
                scalar).
            phase: Optional ``[B, T]`` cycle phase in ``[0, 1)``.
            phase_valid: Optional ``[B, T]`` bool validity of ``phase``.
            frame_mask: Optional ``[B, T]`` bool; false on padding frames.

        Returns:
            :class:`PoseFusionOutput` with ``pose = base_pose + delta_pose``.
        """
        pose_a, pose_b, valid_a, valid_b, delta_t, phase, phase_valid = self._validate(
            pose_a, pose_b, valid_a, valid_b, delta_t, frame_mask, phase, phase_valid
        )
        phase_channels = self.phase_encoding(phase, phase_valid) if self.phase_encoding is not None else None
        pose_feature_a, short_a, long_a, motion_a, guided_a = self.encode_view(pose_a, valid_a, delta_t, phase_channels)
        pose_feature_b, short_b, long_b, motion_b, guided_b = self.encode_view(pose_b, valid_b, delta_t, phase_channels)
        if self.cross_view is not None:
            cross_a, cross_b = self.cross_view(guided_a, guided_b, valid_a, valid_b)
        else:
            cross_a, cross_b = guided_a, guided_b
        logits, weight_a, weight_b = self.reliability(cross_a, cross_b, valid_a, valid_b)
        base_pose, valid = weighted_pose_fusion(pose_a, pose_b, weight_a, weight_b, valid_a, valid_b)
        delta_pose = self.residual(cross_a, cross_b, weight_a, weight_b, base_pose, valid)
        return PoseFusionOutput(
            pose=base_pose + delta_pose,
            base_pose=base_pose,
            delta_pose=delta_pose,
            weight_a=weight_a,
            weight_b=weight_b,
            reliability_logits=logits,
            valid=valid,
            pose_feature_a=pose_feature_a,
            pose_feature_b=pose_feature_b,
            short_motion_a=short_a,
            short_motion_b=short_b,
            long_motion_a=long_a,
            long_motion_b=long_b,
            motion_feature_a=motion_a,
            motion_feature_b=motion_b,
            guided_feature_a=guided_a,
            guided_feature_b=guided_b,
            cross_feature_a=cross_a,
            cross_feature_b=cross_b,
        )
