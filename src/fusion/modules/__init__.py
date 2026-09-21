"""Neural building blocks of the cycle-aware dual-view fusion model.

Each module corresponds to one box of the architecture diagram in the package
docstring of :mod:`fusion` and documents the
mathematical operation it implements, the research reason for its existence,
and the tensor shapes it consumes and produces.

    transformer            masked pre-norm transformer encoder (shared core)
    spatial_transformer    Pose Branch: attention over joints, per frame
    short_motion           local intra-cycle temporal attention, per joint
    long_motion            cycle-scale temporal attention with phase, per joint
    motion_fusion          F_motion = MLP([F_short ; F_long])
    film                   H = (1 + gamma(F_motion)) * F_pose + beta(F_motion)
    cross_view_attention   H_A <-> H_B bidirectional exchange
    reliability            [w_A, w_B] = softmax(R), per joint and time step
    weighted_fusion        P_base = w_A * P_A + w_B * P_B                       (v1.0)
                           P_base = (L_A + L_B)^-1 (L_A P_A + L_B P_B),
                           L_v = w_v (I - alpha d_v d_v^T)                      (v1.1)
    residual_refinement    P_hat = P_base + Delta_P
"""

from .cross_view_attention import BidirectionalCrossViewAttention
from .film import FiLMMotionGuidance
from .long_motion import LongMotionTransformer
from .motion_fusion import MotionFusion
from .reliability import JointReliabilityHead
from .residual_refinement import ResidualRefinement
from .short_motion import ShortMotionTransformer
from .spatial_transformer import SpatialTransformer
from .transformer import MaskedTransformerEncoder, TransformerBlock
from .weighted_fusion import depth_aware_pose_fusion, weighted_pose_fusion

__all__ = [
    "BidirectionalCrossViewAttention",
    "FiLMMotionGuidance",
    "JointReliabilityHead",
    "LongMotionTransformer",
    "MaskedTransformerEncoder",
    "MotionFusion",
    "ResidualRefinement",
    "ShortMotionTransformer",
    "SpatialTransformer",
    "TransformerBlock",
    "depth_aware_pose_fusion",
    "weighted_pose_fusion",
]
