"""Deterministic rows under the model protocol: missing views and joints."""

from __future__ import annotations

import numpy as np
import pytest

from common.skeletons.mhr70 import MHR70_INDEX, MHR70_NAMES
from fusion.baselines.model_protocol import DeterministicTrialTransform
from fusion.keypoints.schema import PosePairTrial


def _trial(frames: int = 30) -> PosePairTrial:
    rng = np.random.default_rng(0)
    # A smooth movement (the temporal filters need temporal coherence).
    rest = rng.normal(scale=0.3, size=(1, 70, 3)) + np.array([0.0, 0.0, 3.0])
    phase = np.linspace(0.0, 2.0 * np.pi, frames)[:, None, None]
    body = (rest + 0.05 * np.sin(phase + rng.uniform(0, 2 * np.pi, size=(1, 70, 3)))).astype(np.float32)
    for name, position in (("left-hip", (-0.1, 0.0, 3.0)), ("right-hip", (0.1, 0.0, 3.0)), ("left-shoulder", (-0.2, 0.5, 3.0)), ("right-shoulder", (0.2, 0.5, 3.0))):
        body[:, MHR70_INDEX[name]] = position
    face = body + rng.normal(scale=0.002, size=body.shape).astype(np.float32)
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    side = (body - body[:, :1]) @ rotation + body[:, :1] + rng.normal(scale=0.002, size=body.shape).astype(np.float32)
    valid_face = np.ones((frames, 70), dtype=bool)
    valid_side = np.ones((frames, 70), dtype=bool)
    # Frame 3: view A missing entirely (stored as zeros, as every trial stores it).
    valid_face[3] = False
    face[3] = 0.0
    # Frame 5: one wrist missing in view B only.
    wrist = MHR70_INDEX["right-wrist"]
    valid_side[5, wrist] = False
    side[5, wrist] = 0.0
    return PosePairTrial(
        face=face, side=side, valid_face=valid_face, valid_side=valid_side,
        timestamps=np.arange(frames, dtype=np.float64) / 30.0,
        face_map=np.arange(frames, dtype=np.int32), side_map=np.arange(frames, dtype=np.int32),
        joint_names=tuple(MHR70_NAMES), person_id="p", trial_id="t", fps=30.0, source_metadata={},
    )


@pytest.mark.parametrize("method", ("avg_body_current", "kalman_rts_body_fusion", "jitter_weighted_body_average", "avg_body_depthaware"))
def test_missing_views_take_the_other_view_and_zeros_are_never_fused(method):
    trial = _trial()
    out = DeterministicTrialTransform(method)(trial)  # used to raise: zero points marked valid
    wrist = MHR70_INDEX["right-wrist"]
    # Every joint valid in either view stays valid, so all methods score the same joint-frames.
    assert out.valid_face.all() and np.array_equal(out.valid_face, out.valid_side)
    # View A missing -> the frame is view B's points.
    np.testing.assert_allclose(out.face[3], trial.side[3], atol=1e-6)
    # Joint missing in view B only -> view A's point, never a fusion with the stored
    # zero (which would put the wrist ~1.5 m off). The Kalman smoother treats it as
    # an absent measurement and returns its smoothed state, so it is close, not equal.
    tolerance = 0.05 if method.startswith("kalman") else 1e-6
    np.testing.assert_allclose(out.face[5, wrist], trial.face[5, wrist], atol=tolerance)
    assert np.isfinite(out.face).all() and not np.any(np.all(out.face == 0.0, axis=-1))


def test_single_view_rows_keep_their_own_validity():
    trial = _trial()
    view_a = DeterministicTrialTransform("view_a")(trial)
    assert not view_a.valid_face[3].any() and view_a.valid_face[5].all()
    view_b = DeterministicTrialTransform("view_b")(trial)
    assert view_b.valid_face[3].all() and not view_b.valid_face[5, MHR70_INDEX["right-wrist"]]
