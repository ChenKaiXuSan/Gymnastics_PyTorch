"""Shared 3D-keypoint representation used by stages 3-4, the baselines and the benchmarks.

* :mod:`.schema`      -- ``PosePairTrial``, the synchronized face/side MHR70 contract
* :mod:`.config`      -- ``SkeletonSpec`` and ``load_skeleton_spec``
* :mod:`.geometry`    -- pelvis-centred canonical body frame (``canonicalize_pose``)
* :mod:`.trunk`       -- trunk rotation features
* :mod:`.features`    -- pose / quality / disagreement features
* :mod:`.base_fusion` -- quality-weighted base fusion
* :mod:`.data`        -- split-cycle trial loading and the immutable person cache

No model code lives here. The modules were extracted from the archived
rotation-aware package on 2026-09-19; the fusion model, the deterministic
matrix, the classical baselines, both public benchmarks and the paper scripts
depend on them.
"""

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
