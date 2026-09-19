"""Per-joint temporal transformer shared by the short- and long-term branches.

Both motion branches of the cycle-aware fusion model apply self-attention
along *time* independently for every joint.  They differ only in the temporal
receptive field (a local band versus the full cycle-scale window) and in the
extra phase channels the long-term branch receives.  This module holds the
shared implementation; :mod:`.short_motion` and :mod:`.long_motion` configure
it and document the research role of each branch.

Method:
    Input token for joint j at sample t:

        m[t, j] = W_in [ p[t, j] ; v[t, j] ; c[t] ] + PE(t)

    with position ``p`` (3), physical velocity ``v`` (3) and optional
    per-frame conditioning ``c`` (e.g. the phase encoding), plus a sinusoidal
    temporal positional encoding ``PE``.  A masked transformer then attends
    over the ``T`` samples of each joint trajectory.

Tensor transformation:
    [B, T, J, C_in]
        -> project         [B, T, J, D]
        -> fold B and J    [B * J, T, D]    (attention over time per joint)
        -> encoder         [B * J, T, D]
        -> unfold          [B, T, J, D]

    Folding the joint axis into the batch axis makes the temporal attention
    of one joint independent of the other joints; cross-joint interaction is
    the job of the Spatial Transformer and of the FiLM conditioning.
"""

from __future__ import annotations

import torch
from torch import nn

from .transformer import MaskedTransformerEncoder, sinusoidal_positions


def local_band_mask(length: int, half_window: int, *, device: torch.device) -> torch.Tensor:
    """Return an ``[L, L]`` bool mask allowing ``|q - k| <= half_window``.

    Args:
        length: Sequence length ``L``.
        half_window: Half-width of the temporal band in samples.
        device: Target device.

    Returns:
        Bool tensor, true where attention is allowed.
    """
    position = torch.arange(length, device=device)
    return (position[:, None] - position[None, :]).abs() <= int(half_window)


class TemporalMotionTransformer(nn.Module):
    """Per-joint temporal attention over position, velocity and conditioning.

    Attributes:
        input_projection: Linear map from ``3 + 3 + conditioning_dim`` to ``D``.
        joint_embedding: Learned joint-identity embedding so the shared
            temporal weights can specialise per joint.
        encoder: Masked transformer over the temporal axis.
        half_window: Half-width of the local attention band, or ``None`` for
            full attention.
    """

    def __init__(
        self,
        num_joints: int,
        hidden_dim: int,
        *,
        conditioning_dim: int = 0,
        heads: int = 4,
        layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        half_window: int | None = None,
    ) -> None:
        super().__init__()
        if num_joints < 1 or hidden_dim < 1 or conditioning_dim < 0:
            raise ValueError("num_joints and hidden_dim must be positive; conditioning_dim non-negative")
        if half_window is not None and half_window < 0:
            raise ValueError("half_window must be non-negative")
        self.num_joints = int(num_joints)
        self.hidden_dim = int(hidden_dim)
        self.conditioning_dim = int(conditioning_dim)
        self.half_window = None if half_window is None else int(half_window)
        self.input_projection = nn.Linear(6 + self.conditioning_dim, hidden_dim)
        self.joint_embedding = nn.Embedding(num_joints, hidden_dim)
        self.encoder = MaskedTransformerEncoder(hidden_dim, heads, layers, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        pose: torch.Tensor,
        velocity: torch.Tensor,
        valid: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode joint trajectories.

        Args:
            pose: ``[B, T, J, 3]`` canonical positions.
            velocity: ``[B, T, J, 3]`` physical velocities.
            valid: ``[B, T, J]`` bool token validity (joint valid and frame
                not padded).
            conditioning: Optional ``[B, T, C]`` per-frame channels
                (broadcast over joints), required when ``conditioning_dim > 0``.

        Returns:
            ``[B, T, J, D]`` motion features, zero at invalid tokens.
        """
        if pose.ndim != 4 or pose.shape[-1] != 3 or pose.shape[2] != self.num_joints:
            raise ValueError(f"pose must have shape [B, T, {self.num_joints}, 3]")
        if velocity.shape != pose.shape:
            raise ValueError("velocity must match pose shape")
        if valid.shape != pose.shape[:-1]:
            raise ValueError("valid must have shape [B, T, J]")
        batch, frames, joints, _ = pose.shape
        valid = valid.bool()
        features = [pose, velocity]
        if self.conditioning_dim:
            if conditioning is None or conditioning.shape != (batch, frames, self.conditioning_dim):
                raise ValueError(f"conditioning must have shape [B, T, {self.conditioning_dim}]")
            features.append(conditioning[:, :, None, :].expand(batch, frames, joints, self.conditioning_dim))
        elif conditioning is not None:
            raise ValueError("this transformer was built without conditioning channels")
        inputs = torch.cat(features, dim=-1)
        inputs = torch.where(valid[..., None], inputs, torch.zeros_like(inputs))
        embedded = self.input_projection(inputs) + self.joint_embedding(torch.arange(joints, device=pose.device))[None, None]
        embedded = embedded + sinusoidal_positions(frames, self.hidden_dim, device=pose.device, dtype=embedded.dtype)[None, :, None, :]
        # Fold the joint dimension into the batch dimension so that temporal
        # self-attention is applied independently to each anatomical joint:
        # [B, T, J, D] -> [B, J, T, D] -> [B * J, T, D].
        tokens = embedded.permute(0, 2, 1, 3).reshape(batch * joints, frames, self.hidden_dim)
        token_valid = valid.permute(0, 2, 1).reshape(batch * joints, frames)
        band = None if self.half_window is None else local_band_mask(frames, self.half_window, device=pose.device)
        encoded = self.encoder(tokens, token_valid, band)
        return encoded.reshape(batch, joints, frames, self.hidden_dim).permute(0, 2, 1, 3)
