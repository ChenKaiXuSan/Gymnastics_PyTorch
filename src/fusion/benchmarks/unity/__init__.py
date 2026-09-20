"""Unity native-3D external benchmark.

``cli`` is the ``python -m fusion benchmark-unity`` entry point (parser and
evaluation-only stages); ``config`` holds the shared configuration helpers;
``stages_supervised``, ``stages_extrinsic`` and ``stages_camera_feature`` hold
the training-stage families that the CLI dispatches to.
"""

from .dataset import group_evaluation_sequences, load_unity_benchmark
from .mapping import EVALUATION_JOINT_NAMES, map_mhr70_to_unity
from .supervised_data import UNITY_SUPERVISED_FOLDS, UnityFold

__all__ = [
    "EVALUATION_JOINT_NAMES",
    "UNITY_SUPERVISED_FOLDS",
    "UnityFold",
    "group_evaluation_sequences",
    "load_unity_benchmark",
    "map_mhr70_to_unity",
]
