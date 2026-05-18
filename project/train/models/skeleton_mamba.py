import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from mamba_ssm import Mamba2 as MambaBlock
except ImportError:  # pragma: no cover
    from mamba_ssm import Mamba as MambaBlock


class SkeletonNormalizer(nn.Module):
    """Normalize 3D keypoints by root-centering and optional scale normalization."""

    def __init__(
        self,
        root_idx: int = 0,
        scale_joints: Optional[Tuple[int, int]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.root_idx = root_idx
        self.scale_joints = scale_joints
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, T, J, C), got {tuple(x.shape)}")
        if x.size(-1) < 3:
            raise ValueError("Expected at least 3 coordinates per joint.")

        root = x[:, :, self.root_idx : self.root_idx + 1, :3]
        x = x[:, :, :, :3] - root

        if self.scale_joints is not None:
            joint_a, joint_b = self.scale_joints
            scale = torch.norm(
                x[:, :, joint_a] - x[:, :, joint_b], dim=-1, keepdim=True
            )
            scale = scale.mean(dim=1, keepdim=True)
            x = x / (scale.unsqueeze(-1) + self.eps)

        return x


class MotionFeatureBuilder(nn.Module):
    """Build position, velocity and acceleration features."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = x
        vel = torch.zeros_like(pos)
        vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
        acc = torch.zeros_like(pos)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
        return torch.cat([pos, vel, acc], dim=-1)


class SkeletonMambaClassifier(nn.Module):
    """Minimal and stable skeleton classification baseline for 3D keypoints.

    Input shape:
        (B, T, J, 3)

    Pipeline:
        normalize -> motion features -> flatten joints -> linear embed -> Mamba -> pooling -> classifier
    """

    def __init__(
        self,
        num_joints: int,
        num_classes: int,
        total_class_num: int,
        num_coarse_classes: int = 7,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        root_idx: int = 0,
        scale_joints: Optional[Tuple[int, int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.num_classes = num_classes
        self.num_coarse_classes = num_coarse_classes
        self.normalizer = SkeletonNormalizer(
            root_idx=root_idx, scale_joints=scale_joints
        )
        self.motion_builder = MotionFeatureBuilder()
        self.embed = nn.Linear(num_joints * 9, d_model)

        # shared temporal encoder
        self.pre_norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.post_norm = nn.LayerNorm(d_model)

        # task-specific branches
        self.posture_pre_norm = nn.LayerNorm(d_model)
        self.posture_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.posture_post_norm = nn.LayerNorm(d_model)

        self.relax_pre_norm = nn.LayerNorm(d_model)
        self.relax_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.relax_post_norm = nn.LayerNorm(d_model)

        self.twist_pre_norm = nn.LayerNorm(d_model)
        self.twist_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.twist_post_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.posture_classifier = nn.Linear(d_model, num_classes)
        self.relax_classifier = nn.Linear(d_model, num_classes)
        self.twist_classifier = nn.Linear(d_model, num_classes)
        self.fusion_proj = nn.Linear(d_model * 3, d_model)
        self.num_total_classifier = nn.Linear(d_model, total_class_num)

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, T, J, 3), got {tuple(x.shape)}")

        x = self.normalizer(x)

        # branch 1: fine classification with mamba temporal modeling
        feat = self.motion_builder(x)  # (B, T, J, 9)
        batch_size, time_steps, num_joints, channels = feat.shape
        feat = feat.reshape(batch_size, time_steps, num_joints * channels)
        feat = self.embed(feat)
        shared_feat = feat + self.mamba(self.pre_norm(feat))
        shared_feat = self.post_norm(shared_feat)

        posture_seq = shared_feat + self.posture_mamba(self.posture_pre_norm(shared_feat))
        posture_seq = self.posture_post_norm(posture_seq)
        posture_feat = self.dropout(posture_seq.mean(dim=1))

        relax_seq = shared_feat + self.relax_mamba(self.relax_pre_norm(shared_feat))
        relax_seq = self.relax_post_norm(relax_seq)
        relax_feat = self.dropout(relax_seq.mean(dim=1))

        twist_seq = shared_feat + self.twist_mamba(self.twist_pre_norm(shared_feat))
        twist_seq = self.twist_post_norm(twist_seq)
        twist_feat = self.dropout(twist_seq.mean(dim=1))

        posture_logits = self.posture_classifier(posture_feat)
        relax_logits = self.relax_classifier(relax_feat)
        twist_logits = self.twist_classifier(twist_feat)

        fusion_feat = torch.cat([posture_feat, relax_feat, twist_feat], dim=-1)
        fusion_feat = self.dropout(self.fusion_proj(fusion_feat))
        total_logits = self.num_total_classifier(fusion_feat)

        if return_dict:
            return {
                "posture_logits": posture_logits,
                "relax_logits": relax_logits,
                "twist_logits": twist_logits,
                "total_logits": total_logits,
            }

        return torch.cat([posture_logits, relax_logits, twist_logits], dim=-1)


def build_skeleton_mamba_model(
    num_joints: int,
    num_classes: int,
    total_class_num: int,
    d_model: int = 256,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    root_idx: int = 0,
    scale_joints: Optional[Tuple[int, int]] = None,
    dropout: float = 0.0,
) -> SkeletonMambaClassifier:
    return SkeletonMambaClassifier(
        num_joints=num_joints,
        num_classes=num_classes,
        total_class_num=total_class_num,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        root_idx=root_idx,
        scale_joints=scale_joints,
        dropout=dropout,
    )
