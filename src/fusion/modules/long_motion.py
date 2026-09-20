"""Long-Term Motion Transformer.

This module implements the long-term temporal motion encoder used in the
cycle-aware dual-view 3D pose fusion framework.

Research Motivation:
    The long-term temporal transformer models complete or consecutive motion
    cycles so that the representation can encode periodic recurrence, motion
    phase, and approximate bilateral symmetry.  When one view is corrupted at
    sample ``t``, the same phase of the previous or next cycle is usually
    intact; only a branch whose receptive field spans several cycles can
    exploit that redundancy.  Phase information tells the branch *which*
    samples correspond to each other across cycles, and bilateral alternation
    (left/right limbs half a cycle apart in gait-like motion) becomes visible
    at this scale.

Method:
    For each joint, tokens are built from 3D position, physical velocity and
    the sinusoidal phase encoding ``[sin(2 pi h phi), cos(2 pi h phi)]``
    (``h = 1..H``).  Full temporal self-attention is applied over the whole
    window, whose length is ``num_cycles * samples_per_cycle`` samples so that
    the branch sees ``num_cycles`` complete phase-normalised cycles.

Input:
    pose: Tensor of shape [B, T, J, 3]
    velocity: Tensor of shape [B, T, J, 3]
    valid: Tensor of shape [B, T, J]
    phase_encoding: Tensor of shape [B, T, 2H] (zeros when phase is unknown)

Output:
    motion_features: Tensor of shape [B, T, J, D]

Architecture Context:
    3D keypoints (+ phase)
        -> long-term temporal transformer
        -> long motion feature
        -> motion fusion
        -> FiLM pose conditioning

Notes:
    - Parameters are shared between View A and View B.
    - The context length (0.5, 1, 2 or all available cycles) is chosen by the
      DataModule window (``data.window.num_cycles``); this module does not
      truncate the window, it attends over everything it is given.
    - When phase is unavailable the branch degrades to a plain long-context
      temporal transformer.
"""

from __future__ import annotations

import torch
from torch import nn

from .temporal_transformer import TemporalMotionTransformer


class LongMotionTransformer(nn.Module):
    """Cycle-scale temporal attention per joint producing ``F_long``.

    Attributes:
        transformer: The underlying per-joint temporal transformer with
            ``2H`` phase conditioning channels.
    """

    def __init__(
        self,
        num_joints: int,
        hidden_dim: int,
        *,
        phase_channels: int,
        heads: int = 4,
        layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.phase_channels = int(phase_channels)
        self.transformer = TemporalMotionTransformer(
            num_joints,
            hidden_dim,
            conditioning_dim=self.phase_channels,
            heads=heads,
            layers=layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            half_window=None,
        )

    def forward(
        self,
        pose: torch.Tensor,
        velocity: torch.Tensor,
        valid: torch.Tensor,
        phase_encoding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode cycle-scale motion.

        Args:
            pose: ``[B, T, J, 3]`` canonical positions.
            velocity: ``[B, T, J, 3]`` physical velocities.
            valid: ``[B, T, J]`` bool token validity.
            phase_encoding: ``[B, T, 2H]`` phase channels; required when the
                module was built with ``phase_channels > 0``.

        Returns:
            ``F_long`` with shape ``[B, T, J, D]``.
        """
        return self.transformer(pose, velocity, valid, phase_encoding if self.phase_channels else None)
