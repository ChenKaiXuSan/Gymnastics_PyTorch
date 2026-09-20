"""Structured output of the cycle-aware dual-view pose fusion network.

The model returns every intermediate representation, not just the fused pose,
so that loss functions, ablation studies, visualisations and debugging tools
can inspect the pipeline without re-running or instrumenting the model.

All tensors follow the ``[B, T, J, C]`` convention:

    B  batch size
    T  temporal samples in the window
    J  joints of the common skeleton
    C  channels (3 for coordinates, D for features, 1 for weights)

Fields and their role in the method:

    pose               P_hat   = P_base + Delta_P               [B, T, J, 3]
    base_pose          P_base  = w_A * P_A + w_B * P_B          [B, T, J, 3]
    delta_pose         Delta_P (residual refinement)            [B, T, J, 3]
    weight_a/b         w_A, w_B with w_A + w_B = 1              [B, T, J, 1]
    reliability_logits R (pre-softmax)                          [B, T, J, 2]
    valid              fused validity (A valid or B valid)      [B, T, J]
    pose_feature_a/b   F_pose from the Spatial Transformer      [B, T, J, D]
    short_motion_a/b   F_short from the short branch            [B, T, J, D]
    long_motion_a/b    F_long from the long branch              [B, T, J, D]
    motion_feature_a/b F_motion = MLP([F_short ; F_long])       [B, T, J, D]
    guided_feature_a/b H = (1 + gamma) * F_pose + beta          [B, T, J, D]
    cross_feature_a/b  C after bidirectional cross-attention    [B, T, J, D]
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass
class PoseFusionOutput:
    """Outputs produced by the dual-view pose fusion network.

    Attributes:
        pose: Final fused pose ``P_hat`` with shape ``[B, T, J, 3]``.
        base_pose: Reliability-weighted average of the original inputs,
            ``P_base``, shape ``[B, T, J, 3]``.
        delta_pose: Residual correction ``Delta_P`` such that
            ``pose == base_pose + delta_pose``, shape ``[B, T, J, 3]``.
        weight_a: Reliability weight of View A, shape ``[B, T, J, 1]``.
        weight_b: Reliability weight of View B, shape ``[B, T, J, 1]``;
            ``weight_a + weight_b == 1`` wherever the joint is valid in at
            least one view.
        reliability_logits: Pre-softmax reliability scores ``R`` with shape
            ``[B, T, J, 2]`` (index 0 is View A).
        valid: Boolean mask ``[B, T, J]`` that is true where the fused pose is
            defined (the joint is valid in at least one view).
        pose_feature_a: Spatial Transformer feature of View A, ``[B, T, J, D]``.
        pose_feature_b: Same for View B.
        short_motion_a: Short-term motion feature of View A, ``[B, T, J, D]``.
        short_motion_b: Same for View B.
        long_motion_a: Long-term motion feature of View A, ``[B, T, J, D]``.
        long_motion_b: Same for View B.
        motion_feature_a: Fused motion feature of View A, ``[B, T, J, D]``.
        motion_feature_b: Same for View B.
        guided_feature_a: FiLM-modulated pose feature of View A, ``[B, T, J, D]``.
        guided_feature_b: Same for View B.
        cross_feature_a: View A feature after cross-view attention, ``[B, T, J, D]``.
        cross_feature_b: Same for View B.
    """

    pose: torch.Tensor
    base_pose: torch.Tensor
    delta_pose: torch.Tensor

    weight_a: torch.Tensor
    weight_b: torch.Tensor
    reliability_logits: torch.Tensor
    valid: torch.Tensor

    pose_feature_a: torch.Tensor
    pose_feature_b: torch.Tensor

    short_motion_a: torch.Tensor
    short_motion_b: torch.Tensor

    long_motion_a: torch.Tensor
    long_motion_b: torch.Tensor

    motion_feature_a: torch.Tensor
    motion_feature_b: torch.Tensor

    guided_feature_a: torch.Tensor
    guided_feature_b: torch.Tensor

    cross_feature_a: torch.Tensor
    cross_feature_b: torch.Tensor

    def detach(self) -> "PoseFusionOutput":
        """Return a copy with every tensor detached from the autograd graph."""
        return PoseFusionOutput(**{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def to(self, device: torch.device | str) -> "PoseFusionOutput":
        """Return a copy with every tensor moved to ``device``."""
        return PoseFusionOutput(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})
