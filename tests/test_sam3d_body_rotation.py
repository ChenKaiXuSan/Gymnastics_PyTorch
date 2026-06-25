import numpy as np
from omegaconf import OmegaConf

from SAM3Dbody.main import maybe_rotate_frames


def test_maybe_rotate_frames_defaults_to_original_frames():
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    cfg = OmegaConf.create({"infer": {}})

    frames = maybe_rotate_frames([frame], cfg)

    assert frames[0] is frame


def test_maybe_rotate_frames_can_rotate_clockwise():
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    cfg = OmegaConf.create(
        {"infer": {"rotate_frames": True, "rotate_code": "ROTATE_90_CLOCKWISE"}}
    )

    frames = maybe_rotate_frames([frame], cfg)

    assert np.array_equal(frames[0], np.rot90(frame, k=3))
