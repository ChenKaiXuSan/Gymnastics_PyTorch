"""Self-supervised, rotation-aware multi-view fusion components."""

from gymnastics.keypoints.config import SkeletonSpec, load_skeleton_spec
from gymnastics.keypoints.data import load_cached_trial, load_person_trials, write_person_cache
from gymnastics.keypoints.schema import PosePairTrial, valid_from_points
from .inference import canonicalize_trial, run_inference

__all__ = [
    "PosePairTrial",
    "SkeletonSpec",
    "load_cached_trial",
    "load_person_trials",
    "load_skeleton_spec",
    "valid_from_points",
    "write_person_cache",
    "canonicalize_trial",
    "run_inference",
]
