"""Self-supervised, rotation-aware multi-view fusion components."""

from .config import SkeletonSpec, load_skeleton_spec
from .data import load_cached_trial, load_person_trials, write_person_cache
from .schema import PosePairTrial, valid_from_points

__all__ = [
    "PosePairTrial",
    "SkeletonSpec",
    "load_cached_trial",
    "load_person_trials",
    "load_skeleton_spec",
    "valid_from_points",
    "write_person_cache",
]
