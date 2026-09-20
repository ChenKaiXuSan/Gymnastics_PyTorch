"""Stage 1 - pose estimation: SAM3D-Body inference on the raw face/side videos.

Produces per-view 3D and 2D keypoints (MHR70) under
``<data_root>/sam3d_body_results/person/<id>/{face,side}/*.npz``.
CLI: ``python -m pose_estimation run``.
"""
