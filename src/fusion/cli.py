"""Entry point of stage 4: ``python -m fusion <command> [args]``.

``train`` is the proposed model (Hydra overrides); the other commands run the
comparison baselines, the public benchmarks, the analyses and the archived
paper model against the same data.
"""

from __future__ import annotations

from typing import Sequence

from common.cli import dispatch

COMMANDS = {
    "train": ("fusion.train", "main", True, "train / evaluate the cycle-aware fusion model (Hydra overrides)"),
    "deterministic": ("fusion.baselines.experiment_matrix", "main", False, "deterministic comparison matrix and classical baselines"),
    "benchmark-freeman": ("fusion.benchmarks.freeman.cli", "main", True, "FreeMan public benchmark (download, infer, fuse, evaluate, report)"),
    "benchmark-freeman-train": ("fusion.benchmarks.freeman.training_cli", "main", True, "archived model trained on FreeMan subject-disjoint folds"),
    "benchmark-unity": ("fusion.benchmarks.unity.cli", "main", True, "Unity native-3D benchmark"),
    "benchmark-sportspose": ("fusion.benchmarks.sportspose.cli", "main", True, "SportsPose benchmark preparation (inspect, select-views, infer)"),
    "analyze": ("fusion.analysis.main", "main", False, "metrics and analysis outputs of saved sequences"),
    "cohort-cycle": ("fusion.analysis.cohort_cycle.cli", "main", True, "out-of-fold cohort and repeated-cycle analysis"),
    "rotation-aware": ("fusion.archive.rotation_aware.cli", "main", True, "archived rotation-aware paper model (reproduction only)"),
}


def main(argv: Sequence[str] | None = None) -> int:
    return dispatch("fusion", "Stage 4: the cycle-aware fusion model, baselines, benchmarks and analyses", COMMANDS, argv)
