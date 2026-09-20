"""Short-Term Motion Transformer.

This module implements the short-term temporal motion encoder used in the
cycle-aware dual-view 3D pose fusion framework.

Research Motivation:
    The short-term motion branch models local intra-cycle dynamics such as
    joint velocity, local trajectory changes, and short temporal transitions.
    Unlike the long-term branch, this module is not intended to model complete
    motion cycles or inter-cycle periodic recurrence.  Its receptive field is
    restricted to a fraction of one phase-normalised cycle so that the feature
    of sample ``t`` summarises only the movement immediately around ``t``,
    which is where single-view artefacts such as jitter, spikes and short
    freezes are visible.

Method:
    For each body joint, the input sequence is represented using 3D position
    and physical velocity features.  Temporal self-attention is applied along
    the time dimension while preserving joint identity, and it is restricted
    to a local band ``|t_query - t_key| <= half_window`` with

        window      = round(cycle_ratio * samples_per_cycle)
        half_window = max(1, window // 2)

    so that the branch sees roughly ``cycle_ratio`` of a normalised cycle
    (default 0.25).

Input:
    pose: Tensor of shape [B, T, J, 3]
    velocity: Tensor of shape [B, T, J, 3]
    valid: Tensor of shape [B, T, J]

Output:
    motion_features: Tensor of shape [B, T, J, D]

Architecture Context:
    3D keypoints
        -> short-term temporal transformer
        -> short motion feature
        -> motion fusion
        -> FiLM pose conditioning

Notes:
    - Parameters are shared between View A and View B.
    - Temporal window length is configured through Hydra
      (``model.short_motion.cycle_ratio`` with ``model.samples_per_cycle``).
    - The short branch should typically cover approximately 0.25 of a
      normalized motion cycle.
    - For sequences without cycle annotations ``samples_per_cycle`` acts as
      a plain frame count.
"""

from __future__ import annotations

import torch
from torch import nn

from .temporal_transformer import TemporalMotionTransformer


class ShortMotionTransformer(nn.Module):
    """Local temporal attention per joint producing ``F_short``.

    Attributes:
        window: Number of samples covered by the local attention band.
        half_window: Half-width of the band.
        transformer: The underlying per-joint temporal transformer.
    """

    def __init__(
        self,
        num_joints: int,
        hidden_dim: int,
        *,
        samples_per_cycle: int,
        cycle_ratio: float = 0.25,
        heads: int = 4,
        layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if samples_per_cycle < 2:
            raise ValueError("samples_per_cycle must be at least 2")
        if not 0.0 < cycle_ratio <= 1.0:
            raise ValueError("cycle_ratio must be in (0, 1]")
        self.window = max(2, int(round(cycle_ratio * samples_per_cycle)))
        self.half_window = max(1, self.window // 2)
        self.transformer = TemporalMotionTransformer(
            num_joints,
            hidden_dim,
            conditioning_dim=0,
            heads=heads,
            layers=layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            half_window=self.half_window,
        )

    def forward(self, pose: torch.Tensor, velocity: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Encode local motion.

        Args:
            pose: ``[B, T, J, 3]`` canonical positions.
            velocity: ``[B, T, J, 3]`` physical velocities.
            valid: ``[B, T, J]`` bool token validity.

        Returns:
            ``F_short`` with shape ``[B, T, J, D]``.
        """
        return self.transformer(pose, velocity, valid)
