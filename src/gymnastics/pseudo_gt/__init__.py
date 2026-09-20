"""Stage 3 - pseudo ground truth: calibration, extrinsics and triangulation.

Chessboard intrinsics (:mod:`.calibration`), per-person extrinsics estimated
from SAM3D 2D correspondences (:mod:`.estimate_extrinsics`) and triangulation
of the face/side 2D keypoints into the 3D pseudo-reference
(:mod:`.sam3d_from_split_cycle`). The result is read only by evaluation code;
no training stage may import it.
CLI: ``gymnastics calibrate``, ``gymnastics triangulate``,
``gymnastics triangulate estimate-extrinsics``.
"""
