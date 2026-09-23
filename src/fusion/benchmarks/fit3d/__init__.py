"""Fit3D benchmark: release layout, view selection and the SAM3D cache.

Fit3D (Fieraru et al., CVPR 2021) is the project's third dataset: eight
subjects with 3D ground truth perform 47 fitness exercises in front of four
synchronised calibrated cameras at 50 fps, and the release annotates every
repetition. It is the closest public analogue of the private recordings
(repeated exercise, trunk and arm motion) and the only one whose movement
cycles are ground truth rather than detector output.

    ``python -m fusion benchmark-fit3d inspect``       coverage of release, cache and repetitions
    ``python -m fusion benchmark-fit3d select-views``  face/side camera pair per sequence

Training reads the result through ``data=fit3d``
(:mod:`fusion.data.fit3d`); the cycle records are written by
``python -m cycle_alignment cycles fit3d``.
"""

from .schema import CAMERAS, JOINTS3D_25_TO_MHR70, NATIVE_FPS, TRAIN_SUBJECTS, Fit3DSequence, SelectedViews

__all__ = ["CAMERAS", "JOINTS3D_25_TO_MHR70", "NATIVE_FPS", "TRAIN_SUBJECTS", "Fit3DSequence", "SelectedViews"]
