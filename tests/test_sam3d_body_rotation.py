import numpy as np
from omegaconf import OmegaConf

from SAM3Dbody.main import maybe_rotate_frames, terminate_processes


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


class FakeProcess:
    def __init__(self, alive=True):
        self._alive = alive
        self.terminated = False
        self.killed = False
        self.join_calls = []

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self._alive = False

    def join(self, timeout=None):
        self.join_calls.append(timeout)


def test_terminate_processes_terminates_and_joins_alive_processes():
    proc = FakeProcess(alive=True)

    terminate_processes([proc], join_timeout=0.1)

    assert proc.terminated
    assert proc.killed
    assert proc.join_calls == [0.1, 0.1]


def test_terminate_processes_ignores_stopped_processes():
    proc = FakeProcess(alive=False)

    terminate_processes([proc], join_timeout=0.1)

    assert not proc.terminated
    assert not proc.killed
    assert proc.join_calls == []
