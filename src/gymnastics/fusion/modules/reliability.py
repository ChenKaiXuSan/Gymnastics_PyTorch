"""Joint-wise, time-dependent view reliability.

Research Motivation:
    Monocular failures are local: a self-occluded wrist in the side view, a
    depth-ambiguous elbow in the frontal view, a truncated foot for a few
    frames.  A single scalar "trust the face view" would waste the
    complementary information of the two cameras.  The model therefore
    predicts a reliability for *every joint and every time step*, and the
    prediction is made from the cross-view contextualised features so it can
    reflect disagreement between the views rather than the quality of one view
    in isolation.

Method:
    A shared scoring MLP ``g`` produces one logit per view from the view's
    own contextualised feature and the other view's feature:

        R_A(t, j) = g([C_A ; C_B ; 1[valid_A] ; 1[valid_B]])
        R_B(t, j) = g([C_B ; C_A ; 1[valid_B] ; 1[valid_A]])

        [w_A, w_B] = softmax([R_A, R_B])

    so that ``w_A(t, j) + w_B(t, j) = 1``.  Using the same ``g`` with the
    arguments swapped makes the weights equivariant under exchanging the views.
    Invalid views receive a large negative logit before the softmax, which
    gives ``w = 1`` to the only valid view; if neither view is valid the
    weights are ``0.5 / 0.5`` (and the fused joint is flagged invalid).

Shapes:
    C_A, C_B            [B, T, J, D]
    valid_A, valid_B    [B, T, J]
    reliability logits  [B, T, J, 2]
    w_A, w_B            [B, T, J, 1]
"""

from __future__ import annotations

import torch
from torch import nn

MASKED_LOGIT = -1.0e4


class JointReliabilityHead(nn.Module):
    """Predict per-joint, per-time-step view weights that sum to one.

    Attributes:
        scorer: The shared scoring MLP ``g``.
        enabled: When false, the logits are zero (equal weights up to
            validity masking); used for the "no learned reliability" ablation.
    """

    def __init__(self, hidden_dim: int, *, enabled: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.enabled = bool(enabled)
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _score(self, own: torch.Tensor, other: torch.Tensor, own_valid: torch.Tensor, other_valid: torch.Tensor) -> torch.Tensor:
        flags = torch.stack((own_valid, other_valid), dim=-1).to(dtype=own.dtype)
        return self.scorer(torch.cat((own, other, flags), dim=-1))[..., 0]

    def forward(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
        valid_a: torch.Tensor,
        valid_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the reliability weights.

        Args:
            feature_a: ``C_A`` with shape ``[B, T, J, D]``.
            feature_b: ``C_B`` with shape ``[B, T, J, D]``.
            valid_a: ``[B, T, J]`` bool validity of View A.
            valid_b: ``[B, T, J]`` bool validity of View B.

        Returns:
            Tuple ``(logits, weight_a, weight_b)`` with shapes
            ``[B, T, J, 2]``, ``[B, T, J, 1]`` and ``[B, T, J, 1]``.
        """
        if feature_a.shape != feature_b.shape or feature_a.shape[-1] != self.hidden_dim:
            raise ValueError("view features must both have shape [B, T, J, D]")
        if valid_a.shape != feature_a.shape[:-1] or valid_b.shape != feature_a.shape[:-1]:
            raise ValueError("validity masks must have shape [B, T, J]")
        valid_a, valid_b = valid_a.bool(), valid_b.bool()
        if self.enabled:
            logit_a = self._score(feature_a, feature_b, valid_a, valid_b)
            logit_b = self._score(feature_b, feature_a, valid_b, valid_a)
        else:
            logit_a = torch.zeros(feature_a.shape[:-1], dtype=feature_a.dtype, device=feature_a.device)
            logit_b = torch.zeros_like(logit_a)
        logits = torch.stack((logit_a, logit_b), dim=-1)
        # Joints invalid in both views carry no evidence: zero their logits so
        # the weights are exactly 0.5 / 0.5 (the fused joint is flagged invalid).
        any_valid = valid_a | valid_b
        logits = torch.where(any_valid[..., None], logits, torch.zeros_like(logits))
        # Push invalid views to (numerically) zero weight.
        masked = torch.stack((valid_a, valid_b), dim=-1) | ~any_valid[..., None]
        masked_logits = torch.where(masked, logits, torch.full_like(logits, MASKED_LOGIT))
        weights = torch.softmax(masked_logits, dim=-1)
        return logits, weights[..., 0:1], weights[..., 1:2]
