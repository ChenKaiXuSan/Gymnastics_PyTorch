"""FiLM Motion Guidance: conditioning the pose representation on motion.

Research Motivation:
    The Pose Branch knows the body configuration of one frame but nothing
    about how the body moves; the motion branches know the dynamics but were
    computed per joint.  Feature-wise Linear Modulation (FiLM, Perez et al.,
    2018) lets the motion descriptor rescale and shift every channel of the
    pose feature, so the subsequent cross-view comparison and reliability
    estimate are *motion-guided*: a joint whose position is implausible given
    its trajectory (a spike, a freeze, an implausible speed) is modulated
    differently from a joint that moves consistently.

Method:
    gamma = f_gamma(F_motion)
    beta  = f_beta(F_motion)

    H = (1 + gamma) * F_pose + beta

    ``F_motion`` is the conditioning signal and ``F_pose`` the representation
    being modulated.  The *residual* form ``(1 + gamma)`` (instead of a plain
    ``gamma``) together with zero-initialised ``f_gamma`` and ``f_beta`` means
    that at initialisation, and whenever the motion branch is uninformative,
    ``H == F_pose``: the pose representation is preserved and the motion
    guidance is learned as a perturbation of it.

Shapes:
    F_pose    [B, T, J, D]
    F_motion  [B, T, J, D]
    gamma     [B, T, J, D]
    beta      [B, T, J, D]
    H         [B, T, J, D]
"""

from __future__ import annotations

import torch
from torch import nn


class FiLMMotionGuidance(nn.Module):
    """Residual feature-wise linear modulation of ``F_pose`` by ``F_motion``.

    Attributes:
        gamma: Linear map ``f_gamma`` (zero-initialised).
        beta: Linear map ``f_beta`` (zero-initialised).
        gamma_bound: Optional bound applied as ``tanh`` scaling to ``gamma``
            for numerical stability (``None`` disables it).
    """

    def __init__(self, hidden_dim: int, *, gamma_bound: float | None = None) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)
        if gamma_bound is not None and gamma_bound <= 0:
            raise ValueError("gamma_bound must be positive or None")
        self.gamma_bound = None if gamma_bound is None else float(gamma_bound)
        # Zero initialisation makes the modulation an identity at the start of
        # training so the pose representation is preserved (see module docs).
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, pose_feature: torch.Tensor, motion_feature: torch.Tensor) -> torch.Tensor:
        """Modulate the pose feature.

        Args:
            pose_feature: ``F_pose`` with shape ``[B, T, J, D]``.
            motion_feature: ``F_motion`` with shape ``[B, T, J, D]``.

        Returns:
            ``H = (1 + gamma) * F_pose + beta`` with shape ``[B, T, J, D]``.
        """
        if pose_feature.shape != motion_feature.shape or pose_feature.shape[-1] != self.hidden_dim:
            raise ValueError("pose and motion features must both have shape [B, T, J, D]")
        gamma = self.gamma(motion_feature)
        if self.gamma_bound is not None:
            gamma = self.gamma_bound * torch.tanh(gamma / self.gamma_bound)
        beta = self.beta(motion_feature)
        return (1.0 + gamma) * pose_feature + beta
