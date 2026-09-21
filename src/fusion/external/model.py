"""External learned baselines on the shared dual-view contract (see package docstring).

Common skeleton of every baseline (mirrors the proposed model so that only the
learned part differs):

    P_base = closed-form base rule on the original inputs with equal weights
             (``fusion.depth_alpha``: 0 = plain average, 0.8 = the v1.1 rule)
    features per joint and frame = [P_A ; P_B ; P_base ; valid_A ; valid_B]   (11 channels)
    Delta_P = max_delta * tanh(backbone(features))                            (bounded residual)
    P_hat   = P_base + Delta_P

``muc_weights`` is the exception: it predicts per-view per-joint weights and
uses them in the base rule instead of a residual (learned weighted average, no
residual), which is what MUC does instead of averaging.

Shapes: poses ``[B, T, J, 3]``, validity ``[B, T, J]``, depth axes ``[B, T, 3]``.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..model import FusionConfig
from ..modules.reliability import MASKED_LOGIT
from ..modules.weighted_fusion import depth_aware_pose_fusion
from ..outputs import PoseFusionOutput
from ..skeleton import CommonSkeleton, build_common_skeleton

EXTERNAL_BACKBONES = ("tcn", "smoothnet", "metapose_mlp", "muc_weights")
FEATURE_CHANNELS = 11  # P_A, P_B, P_base (3 each) + valid_A, valid_B


@dataclass(frozen=True)
class ExternalModelConfig:
    """Configuration of one external baseline (``src/configs/fusion/model/external_*.yaml``).

    Attributes:
        backbone: One of :data:`EXTERNAL_BACKBONES`.
        skeleton: Common skeleton variant.
        samples_per_cycle: Samples per phase-normalised cycle (data contract).
        hidden_dim: Width of the backbone.
        max_delta: Residual bound in canonical units (``None`` = unbounded).
        dropout: Dropout inside the backbone.
        fusion: Base rule (``depth_alpha`` 0 = plain average).
        tcn_dilations: Dilations of the residual TCN blocks.
        smoothnet_window: SmoothNet sliding window length (samples).
        smoothnet_res_hidden: Width of the SmoothNet residual blocks.
        smoothnet_blocks: Number of SmoothNet residual blocks.
        mlp_layers: Hidden layers of the per-frame MLP baselines.
    """

    backbone: str = "tcn"
    skeleton: str = "mhr70_major"
    samples_per_cycle: int = 64
    hidden_dim: int = 128
    max_delta: float | None = 0.25
    dropout: float = 0.0
    fusion: FusionConfig = field(default_factory=FusionConfig)
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    smoothnet_window: int = 32
    smoothnet_res_hidden: int = 256
    smoothnet_blocks: int = 3
    mlp_layers: int = 3

    def __post_init__(self) -> None:
        if self.backbone not in EXTERNAL_BACKBONES:
            raise ValueError(f"backbone must be one of {EXTERNAL_BACKBONES}")
        if self.hidden_dim <= 0 or self.samples_per_cycle < 2 or self.mlp_layers < 1:
            raise ValueError("hidden_dim, samples_per_cycle and mlp_layers must be positive")
        if self.max_delta is not None and self.max_delta <= 0:
            raise ValueError("max_delta must be positive or None")
        if self.smoothnet_window < 2 or self.smoothnet_blocks < 1:
            raise ValueError("smoothnet_window >= 2 and smoothnet_blocks >= 1")
        object.__setattr__(self, "tcn_dilations", tuple(int(d) for d in self.tcn_dilations))

    @property
    def architecture_version(self) -> str:
        return f"external/{self.backbone}"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "ExternalModelConfig") -> "ExternalModelConfig":
        """Build from a (nested) mapping; unknown keys raise."""
        if isinstance(value, cls):
            return value
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)  # type: ignore[assignment]
        except ImportError:  # pragma: no cover
            pass
        payload = dict(value)
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(payload) - set(known))
        if unknown:
            raise ValueError(f"unknown ExternalModelConfig fields: {unknown}")
        if isinstance(payload.get("fusion"), Mapping):
            payload["fusion"] = FusionConfig(**dict(payload["fusion"]))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------- backbones
class _ResidualTCNBlock(nn.Module):
    """Same-length non-causal dilated residual block (VideoPose3D / plain-TCN B2)."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        values = values * mask
        residual = values
        values = self.dropout(self.activation(self.conv1(values) * mask))
        values = self.conv2(values) * mask
        return self.activation(values + residual) * mask


class TCNBackbone(nn.Module):
    """Per-joint feature MLP, dilated residual TCN over time (per joint), per-joint output."""

    def __init__(self, hidden: int, dilations: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(FEATURE_CHANNELS, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(_ResidualTCNBlock(hidden, d, dropout) for d in dilations)
        self.decoder = nn.Linear(hidden, 3)

    def forward(self, features: Tensor, frame_mask: Tensor) -> Tensor:  # [B, T, J, 11] -> [B, T, J, 3]
        batch, frames, joints, _ = features.shape
        hidden = self.encoder(features).permute(0, 2, 3, 1).reshape(batch * joints, -1, frames)  # [B*J, D, T]
        mask = frame_mask.to(hidden.dtype)[:, None, :].repeat_interleave(joints, dim=0)  # [B*J, 1, T]
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = hidden.reshape(batch, joints, -1, frames).permute(0, 3, 1, 2)  # [B, T, J, D]
        return self.decoder(hidden)


class SmoothNetBackbone(nn.Module):
    """SmoothNet (Zeng et al., 2022): channel-independent temporal MLP over sliding windows.

    Each input channel (joint x feature) is refined by the same window-to-window
    MLP (encoder -> residual blocks -> decoder), applied to every window of
    ``window`` samples with stride 1; overlapping outputs are averaged, exactly
    like the reference inference.  The channel mixing that SmoothNet lacks is
    provided only by a per-joint linear layer from the refined feature channels
    to the residual, so the temporal modelling stays SmoothNet's.
    """

    def __init__(self, window: int, hidden: int, res_hidden: int, blocks: int, dropout: float) -> None:
        super().__init__()
        self.window = int(window)
        self.encoder = nn.Sequential(nn.Linear(self.window, hidden), nn.LeakyReLU(0.1))
        self.res_blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, res_hidden), nn.Dropout(dropout), nn.LeakyReLU(0.2), nn.Linear(res_hidden, hidden), nn.Dropout(dropout), nn.LeakyReLU(0.2))
            for _ in range(blocks)
        )
        self.decoder = nn.Linear(hidden, self.window)
        self.output = nn.Linear(FEATURE_CHANNELS, 3)

    def _refine(self, windows: Tensor) -> Tensor:  # [N, window] -> [N, window]
        hidden = self.encoder(windows)
        for block in self.res_blocks:
            hidden = hidden + block(hidden)
        return self.decoder(hidden)

    def forward(self, features: Tensor, frame_mask: Tensor) -> Tensor:  # [B, T, J, 11] -> [B, T, J, 3]
        batch, frames, joints, channels = features.shape
        window = min(self.window, frames)
        signal = (features * frame_mask[:, :, None, None].to(features.dtype)).permute(0, 2, 3, 1).reshape(batch, joints * channels, frames)
        if frames < self.window:  # pad by repeating the last sample, as the reference code does
            signal = torch.cat([signal, signal[..., -1:].expand(-1, -1, self.window - frames)], dim=-1)
            window = self.window
        windows = signal.unfold(-1, window, 1)  # [B, C, N, window]
        refined = self._refine(windows.reshape(-1, window)).reshape(windows.shape)
        # Overlap-average the sliding outputs back onto the timeline.
        total = signal.shape[-1]
        accum = signal.new_zeros(batch, joints * channels, total)
        count = signal.new_zeros(1, 1, total)
        for start in range(windows.shape[2]):
            accum[..., start : start + window] += refined[:, :, start]
            count[..., start : start + window] += 1
        refined_signal = (accum / count)[..., :frames].reshape(batch, joints, channels, frames).permute(0, 3, 1, 2)
        return self.output(refined_signal)


class FrameMLPBackbone(nn.Module):
    """MetaPose-style per-frame aggregation: one MLP over the concatenated views of the whole body."""

    def __init__(self, joints: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        dims = [joints * FEATURE_CHANNELS] + [hidden] * layers
        stack: list[nn.Module] = []
        for i in range(layers):
            stack += [nn.Linear(dims[i], dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*stack)
        self.head = nn.Linear(hidden, joints * 3)
        self.joints = joints

    def forward(self, features: Tensor, frame_mask: Tensor) -> Tensor:  # [B, T, J, 11] -> [B, T, J, 3]
        batch, frames = features.shape[:2]
        out = self.head(self.body(features.reshape(batch, frames, -1)))
        return out.reshape(batch, frames, self.joints, 3)


class WeightMLPBackbone(nn.Module):
    """MUC-style learned per-view per-joint weights from both views of the frame."""

    def __init__(self, joints: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        dims = [joints * 8] + [hidden] * layers  # P_A, P_B, valid_A, valid_B per joint
        stack: list[nn.Module] = []
        for i in range(layers):
            stack += [nn.Linear(dims[i], dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*stack)
        self.head = nn.Linear(hidden, joints * 2)
        self.joints = joints

    def forward(self, pose_a: Tensor, pose_b: Tensor, valid_a: Tensor, valid_b: Tensor) -> Tensor:  # -> logits [B, T, J, 2]
        batch, frames = pose_a.shape[:2]
        inputs = torch.cat((pose_a, pose_b, valid_a[..., None].to(pose_a.dtype), valid_b[..., None].to(pose_a.dtype)), dim=-1)
        return self.head(self.body(inputs.reshape(batch, frames, -1))).reshape(batch, frames, self.joints, 2)


class _ResidualScale(nn.Module):
    """Holds the residual bound so diagnostics can read ``model.residual.max_delta``."""

    def __init__(self, max_delta: float | None) -> None:
        super().__init__()
        self.max_delta = max_delta

    def forward(self, raw: Tensor) -> Tensor:
        return raw if self.max_delta is None else self.max_delta * torch.tanh(raw)


# ----------------------------------------------------------------------------- wrapper
class ExternalFusionModel(nn.Module):
    """External baseline with the proposed model's input/output contract.

    Attributes:
        config: :class:`ExternalModelConfig`.
        skeleton: Common skeleton.
        backbone: The architecture-specific network.
        residual: Residual bound (``None`` for ``muc_weights``, which has no residual).
        film: Always ``None`` (no FiLM; keeps the Lightning diagnostics uniform).
    """

    film = None

    def __init__(self, config: ExternalModelConfig | Mapping[str, Any] | None = None, *, skeleton: CommonSkeleton | None = None) -> None:
        super().__init__()
        self.config = ExternalModelConfig.from_mapping(config) if config is not None else ExternalModelConfig()
        self.skeleton = skeleton or build_common_skeleton(self.config.skeleton)
        cfg, joints = self.config, self.skeleton.num_joints
        if cfg.backbone == "tcn":
            self.backbone: nn.Module = TCNBackbone(cfg.hidden_dim, cfg.tcn_dilations, cfg.dropout)
        elif cfg.backbone == "smoothnet":
            self.backbone = SmoothNetBackbone(cfg.smoothnet_window, cfg.hidden_dim, cfg.smoothnet_res_hidden, cfg.smoothnet_blocks, cfg.dropout)
        elif cfg.backbone == "metapose_mlp":
            self.backbone = FrameMLPBackbone(joints, cfg.hidden_dim, cfg.mlp_layers, cfg.dropout)
        else:
            self.backbone = WeightMLPBackbone(joints, cfg.hidden_dim, cfg.mlp_layers, cfg.dropout)
        self.residual = _ResidualScale(cfg.max_delta)
        # Residual baselines start as the base rule (zero residual), like the proposed model.
        if cfg.backbone != "muc_weights":
            last = self.backbone.decoder if cfg.backbone == "tcn" else (self.backbone.output if cfg.backbone == "smoothnet" else self.backbone.head)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            nn.init.zeros_(self.backbone.head.weight)
            nn.init.zeros_(self.backbone.head.bias)

    @property
    def num_joints(self) -> int:
        return self.skeleton.num_joints

    def gradient_module_map(self) -> dict[str, nn.Module]:
        return {"backbone": self.backbone}

    def fuse_base(self, pose_a, pose_b, weight_a, weight_b, valid_a, valid_b, depth_a=None, depth_b=None):
        """The base rule of this configuration (same contract as the proposed model)."""
        depth_a, depth_b = self._validate_depth(depth_a, depth_b, pose_a)
        return depth_aware_pose_fusion(pose_a, pose_b, weight_a, weight_b, valid_a, valid_b, depth_a, depth_b, alpha=float(self.config.fusion.depth_alpha), min_precision=float(self.config.fusion.min_precision))

    def _validate_depth(self, depth_a, depth_b, pose_a):
        if depth_a is None or depth_b is None or float(self.config.fusion.depth_alpha) == 0.0:
            return None, None
        axes = []
        for depth in (depth_a, depth_b):
            depth = torch.as_tensor(depth, dtype=pose_a.dtype, device=pose_a.device)
            if depth.shape != pose_a.shape[:2] + (3,):
                raise ValueError("depth axes must have shape [B, T, 3]")
            norm = torch.linalg.vector_norm(depth, dim=-1, keepdim=True)
            usable = torch.isfinite(depth).all(dim=-1, keepdim=True) & (norm > 1e-6)
            axes.append(torch.where(usable, depth / norm.clamp_min(1e-6), torch.zeros_like(depth)))
        return axes[0], axes[1]

    def forward(
        self,
        pose_a: Tensor,
        pose_b: Tensor,
        valid_a: Tensor,
        valid_b: Tensor,
        delta_t: Tensor,
        phase: Tensor | None = None,
        phase_valid: Tensor | None = None,
        frame_mask: Tensor | None = None,
        depth_a: Tensor | None = None,
        depth_b: Tensor | None = None,
    ) -> PoseFusionOutput:
        if pose_a.ndim != 4 or pose_a.shape[-1] != 3 or pose_a.shape[2] != self.num_joints or pose_b.shape != pose_a.shape:
            raise ValueError(f"poses must have shape [B, T, {self.num_joints}, 3]")
        batch, frames = pose_a.shape[:2]
        if frame_mask is None:
            frame_mask = torch.ones(batch, frames, dtype=torch.bool, device=pose_a.device)
        frame_mask = frame_mask.bool()
        valid_a = valid_a.bool() & frame_mask[..., None] & torch.isfinite(pose_a).all(dim=-1)
        valid_b = valid_b.bool() & frame_mask[..., None] & torch.isfinite(pose_b).all(dim=-1)
        pose_a = torch.where(valid_a[..., None], pose_a, torch.zeros_like(pose_a))
        pose_b = torch.where(valid_b[..., None], pose_b, torch.zeros_like(pose_b))
        half = torch.full_like(pose_a[..., :1], 0.5)
        if self.config.backbone == "muc_weights":
            logits = self.backbone(pose_a, pose_b, valid_a, valid_b)
            any_valid = valid_a | valid_b
            logits = torch.where(any_valid[..., None], logits, torch.zeros_like(logits))
            masked = torch.stack((valid_a, valid_b), dim=-1) | ~any_valid[..., None]
            weights = torch.softmax(torch.where(masked, logits, torch.full_like(logits, MASKED_LOGIT)), dim=-1)
            weight_a, weight_b = weights[..., 0:1], weights[..., 1:2]
            base_pose, valid = self.fuse_base(pose_a, pose_b, weight_a, weight_b, valid_a, valid_b, depth_a, depth_b)
            delta_pose = torch.zeros_like(base_pose)
        else:
            weight_a, weight_b = half, half
            logits = torch.zeros(batch, frames, self.num_joints, 2, dtype=pose_a.dtype, device=pose_a.device)
            base_pose, valid = self.fuse_base(pose_a, pose_b, half, half, valid_a, valid_b, depth_a, depth_b)
            features = torch.cat((pose_a, pose_b, base_pose, valid_a[..., None].to(pose_a.dtype), valid_b[..., None].to(pose_a.dtype)), dim=-1)
            delta_pose = self.residual(self.backbone(features, frame_mask))
            delta_pose = torch.where(valid[..., None], delta_pose, torch.zeros_like(delta_pose))
        zeros = pose_a.new_zeros(batch, frames, self.num_joints, 1)
        return PoseFusionOutput(
            pose=base_pose + delta_pose,
            base_pose=base_pose,
            delta_pose=delta_pose,
            weight_a=weight_a,
            weight_b=weight_b,
            reliability_logits=logits,
            valid=valid,
            pose_feature_a=zeros,
            pose_feature_b=zeros,
            short_motion_a=zeros,
            short_motion_b=zeros,
            long_motion_a=zeros,
            long_motion_b=zeros,
            motion_feature_a=zeros,
            motion_feature_b=zeros,
            guided_feature_a=zeros,
            guided_feature_b=zeros,
            cross_feature_a=zeros,
            cross_feature_b=zeros,
        )
