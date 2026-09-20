"""Entry point of stage 3: ``python -m pseudo_gt {calibrate|estimate-extrinsics|triangulate}``."""

from __future__ import annotations

from typing import Sequence

from common.cli import dispatch

COMMANDS = {
    "calibrate": ("pseudo_gt.calibration", "main", False, "chessboard intrinsics per camera"),
    "estimate-extrinsics": ("pseudo_gt.estimate_extrinsics", "main", False, "per-person face/side extrinsics from SAM3D 2D correspondences"),
    "triangulate": ("pseudo_gt.sam3d_from_split_cycle", "main", False, "triangulate the SAM3D 2D keypoints into the 3D pseudo-reference"),
}


def main(argv: Sequence[str] | None = None) -> int:
    return dispatch("pseudo_gt", "Stage 3: calibration, extrinsics and the triangulated pseudo-reference", COMMANDS, argv)
