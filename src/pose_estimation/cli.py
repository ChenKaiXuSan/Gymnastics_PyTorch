"""Entry point of stage 1: ``python -m pose_estimation [run] [SAM3D options]``."""

from __future__ import annotations

from typing import Sequence

from common.cli import dispatch

COMMANDS = {
    "run": ("pose_estimation.main", "main", True, "run SAM3D-Body on the paired face/side videos"),
}


def main(argv: Sequence[str] | None = None) -> int:
    return dispatch("pose_estimation", "Stage 1: SAM3D-Body pose estimation", COMMANDS, argv, default="run")
