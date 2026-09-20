"""Multi-view gymnastics motion analysis.

Pipeline packages, in data-flow order:

1. :mod:`gymnastics.pose_estimation`  SAM3D-Body inference on the raw face and
   side videos; per-view 3D and 2D keypoints (MHR70, 70 joints).
2. :mod:`gymnastics.cycle_alignment`  face/side temporal offset, movement-cycle
   segmentation and turn-around middles, written as cycle records.
3. :mod:`gymnastics.pseudo_gt`        camera calibration, per-person extrinsics
   and triangulation of the SAM3D 2D keypoints into the 3D pseudo-reference
   used only for evaluation.
4. :mod:`gymnastics.fusion`           the proposed cycle-aware dual-view fusion
   network: data modules, model, self-supervised losses, Lightning training.

Supporting packages: :mod:`gymnastics.keypoints` (shared 3D-keypoint
representation used by stages 3-4, the baselines and the benchmarks),
:mod:`gymnastics.baselines` (deterministic comparison matrix and classical
baselines), :mod:`gymnastics.benchmarks` (FreeMan and Unity),
:mod:`gymnastics.analysis` (metrics, reports, cohort statistics),
:mod:`gymnastics.common` (paths, config helpers, MHR70 metadata) and
:mod:`gymnastics.archive` (the frozen rotation-aware model of the paper).
"""

__version__ = "0.1.0"
