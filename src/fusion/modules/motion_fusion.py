"""Motion Fusion: combining the short- and long-term motion features.

Research Motivation:
    The short-term branch describes what a joint is doing *right now* (local
    velocity, trajectory bends); the long-term branch describes where the
    frame lies within the repeated cycle and how the same phase looked in
    neighbouring cycles.  Both are needed to judge and correct a single-view
    estimate, so they are merged into one motion descriptor ``F_motion`` that
    later conditions the pose representation through FiLM.

Method:
    F_motion = MLP([F_short ; F_long])

    The concatenation along the channel axis gives ``2D`` channels; a
    two-layer MLP with GELU maps them back to ``D``.  In Architecture v1.0 no
    additional attention is used here: both inputs are already aligned per
    joint and per sample (same ``[B, T, J, *]`` layout), so a channel-wise
    mixing is sufficient and keeps the fusion point simple to ablate (setting
    one branch to zero exposes the contribution of the other).

Shapes:
    F_short   [B, T, J, D]
    F_long    [B, T, J, D]
    F_motion  [B, T, J, D]
"""

from __future__ import annotations

import torch
from torch import nn


class MotionFusion(nn.Module):
    """Channel-wise MLP fusion of the two motion branches.

    Attributes:
        mlp: ``Linear(2D, D) -> GELU -> Linear(D, D)``.
    """

    def __init__(self, hidden_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, short_motion: torch.Tensor, long_motion: torch.Tensor) -> torch.Tensor:
        """Fuse the branches.

        Args:
            short_motion: ``F_short`` with shape ``[B, T, J, D]``.
            long_motion: ``F_long`` with shape ``[B, T, J, D]``.

        Returns:
            ``F_motion`` with shape ``[B, T, J, D]``.
        """
        if short_motion.shape != long_motion.shape or short_motion.shape[-1] != self.hidden_dim:
            raise ValueError("short and long motion features must both have shape [B, T, J, D]")
        return self.mlp(torch.cat((short_motion, long_motion), dim=-1))
