"""Spatial Transformer: the Pose Branch of the cycle-aware fusion model.

This module encodes the *static* configuration of the body at each time step
by applying self-attention across joints.

Research Motivation:
    A joint's plausibility depends on the other joints of the same frame: a
    wrist far from its elbow, or a shoulder line inconsistent with the hips,
    is a per-frame anatomical inconsistency that a monocular estimator
    produces frequently in one view but rarely in both.  Attention over joints
    lets every joint feature incorporate the whole-body context of that
    frame, which is exactly the information the reliability head later needs
    to decide which view to trust for that joint.

    Temporal relationships are deliberately *excluded* from the Pose Branch.
    Motion (velocity, local trajectory, cycle recurrence) is modelled by the
    short-term and long-term motion branches and injected into the pose
    representation later through FiLM.  Keeping the branches separate makes
    the ablation "pose only" versus "pose + motion" meaningful and prevents
    the pose encoder from absorbing temporal cues in an uncontrolled way.

Method:
    Joint embedding for joint j at time t:

        e[t, j] = W_in p[t, j] + E_joint[j] + E_valid[v[t, j]]

    where ``p`` is the 3D position (canonical body frame), ``E_joint`` a
    learned identity embedding (attention has no notion of token order, so
    joint identity must be added explicitly) and ``E_valid`` a learned flag
    embedding.  A masked transformer then attends across the ``J`` joints of
    each frame.

Tensor transformation:
    pose   [B, T, J, 3]
        -> embed          [B, T, J, D]
        -> fold B and T   [B * T, J, D]     (attention over joints)
        -> encoder        [B * T, J, D]
        -> unfold         [B, T, J, D]  = F_pose

Output semantics:
    ``F_pose[b, t, j]`` is a frame-local, whole-body-aware descriptor of
    joint ``j``.  It is zero for invalid joints.

Notes:
    * Parameters are shared between View A and View B (the model calls the
      same instance for both views).
    * Padding frames are handled by the caller through the ``valid`` mask.
"""

from __future__ import annotations

import torch
from torch import nn

from .transformer import MaskedTransformerEncoder


class SpatialTransformer(nn.Module):
    """Per-frame attention over joints producing ``F_pose``.

    Attributes:
        input_projection: Linear map from XYZ to the hidden dimension.
        joint_embedding: Learned joint-identity embedding ``[J, D]``.
        valid_embedding: Learned embedding of the joint validity flag.
        encoder: Masked transformer over the joint axis.
    """

    def __init__(
        self,
        num_joints: int,
        hidden_dim: int,
        *,
        heads: int = 4,
        layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_joints < 1:
            raise ValueError("num_joints must be positive")
        self.num_joints = int(num_joints)
        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Linear(3, hidden_dim)
        self.joint_embedding = nn.Embedding(num_joints, hidden_dim)
        self.valid_embedding = nn.Embedding(2, hidden_dim)
        self.encoder = MaskedTransformerEncoder(hidden_dim, heads, layers, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Encode one view.

        Args:
            pose: ``[B, T, J, 3]`` canonical 3D keypoints.
            valid: ``[B, T, J]`` bool validity (already combined with the
                frame mask by the caller).

        Returns:
            ``F_pose`` with shape ``[B, T, J, D]``; zero at invalid joints.

        Raises:
            ValueError: If the shapes are inconsistent with the module.
        """
        if pose.ndim != 4 or pose.shape[-1] != 3 or pose.shape[2] != self.num_joints:
            raise ValueError(f"pose must have shape [B, T, {self.num_joints}, 3]")
        if valid.shape != pose.shape[:-1]:
            raise ValueError("valid must have shape [B, T, J]")
        batch, frames, joints, _ = pose.shape
        valid = valid.bool()
        safe_pose = torch.where(valid[..., None], pose, torch.zeros_like(pose))
        joint_ids = torch.arange(joints, device=pose.device)
        embedded = (
            self.input_projection(safe_pose)
            + self.joint_embedding(joint_ids)[None, None]
            + self.valid_embedding(valid.long())
        )
        # Fold batch and time so attention runs over the J joints of one frame.
        tokens = embedded.reshape(batch * frames, joints, self.hidden_dim)
        token_valid = valid.reshape(batch * frames, joints)
        encoded = self.encoder(tokens, token_valid)
        return encoded.reshape(batch, frames, joints, self.hidden_dim)
