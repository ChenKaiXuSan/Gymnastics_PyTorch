"""Residual Refinement: a bounded correction on top of the weighted base pose.

Research Motivation:
    A convex combination of the two measurements cannot fix errors that both
    views share (a common-mode depth bias, a joint both estimators place
    slightly off the limb) nor exploit temporal or anatomical priors the
    network has learned.  Instead of generating the whole pose from scratch,
    which would discard the strong measurement anchor of the weighted fusion,
    the model predicts only a *correction* ``Delta_P``.  Predicting a residual
    keeps the output close to the measurements by construction, makes the
    learned contribution inspectable (``Delta_P`` can be visualised and
    regularised directly) and lets the zero-residual model coincide exactly
    with the interpretable weighted fusion.

Method:
    Delta_P = bound( MLP([ w_A * C_A + w_B * C_B ; |C_A - C_B| ; P_base ]) )

    P_hat   = P_base + Delta_P

    The reliability-weighted mean and the absolute difference of the two
    contextualised features are symmetric under swapping the views.  The
    bound is ``max_delta * tanh(.)`` in canonical units when ``max_delta`` is
    set, which keeps the correction within a known radius of the measurement;
    with ``max_delta = None`` the residual is unbounded.  Joints that are
    invalid in both views receive a zero residual.

Shapes:
    C_A, C_B     [B, T, J, D]
    w_A, w_B     [B, T, J, 1]
    P_base       [B, T, J, 3]
    Delta_P      [B, T, J, 3]
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualRefinement(nn.Module):
    """Predict the bounded residual ``Delta_P``.

    Attributes:
        mlp: Regression head from ``2D + 3`` channels to XYZ.
        max_delta: Bound radius in canonical units, or ``None`` for unbounded.
        enabled: When false, the residual is identically zero (ablation).
    """

    def __init__(self, hidden_dim: int, *, max_delta: float | None = 0.25, enabled: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        if max_delta is not None and max_delta <= 0:
            raise ValueError("max_delta must be positive or None")
        self.hidden_dim = int(hidden_dim)
        self.max_delta = None if max_delta is None else float(max_delta)
        self.enabled = bool(enabled)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        # Start from the identity (zero residual) so early training equals
        # the weighted fusion.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
        weight_a: torch.Tensor,
        weight_b: torch.Tensor,
        base_pose: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Predict ``Delta_P``.

        Args:
            feature_a: ``C_A`` with shape ``[B, T, J, D]``.
            feature_b: ``C_B`` with shape ``[B, T, J, D]``.
            weight_a: ``w_A`` with shape ``[B, T, J, 1]``.
            weight_b: ``w_B`` with shape ``[B, T, J, 1]``.
            base_pose: ``P_base`` with shape ``[B, T, J, 3]``.
            valid: ``[B, T, J]`` bool validity of the fused joint.

        Returns:
            ``Delta_P`` with shape ``[B, T, J, 3]``.
        """
        if feature_a.shape != feature_b.shape or feature_a.shape[-1] != self.hidden_dim:
            raise ValueError("view features must both have shape [B, T, J, D]")
        if base_pose.shape != feature_a.shape[:-1] + (3,):
            raise ValueError("base_pose must have shape [B, T, J, 3]")
        if valid.shape != feature_a.shape[:-1]:
            raise ValueError("valid must have shape [B, T, J]")
        if not self.enabled:
            return torch.zeros_like(base_pose)
        pooled = weight_a * feature_a + weight_b * feature_b
        difference = (feature_a - feature_b).abs()
        raw = self.mlp(torch.cat((pooled, difference, base_pose), dim=-1))
        delta = raw if self.max_delta is None else self.max_delta * torch.tanh(raw / self.max_delta)
        return torch.where(valid.bool()[..., None], delta, torch.zeros_like(delta))
