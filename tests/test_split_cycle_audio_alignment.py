import numpy as np

from split_cycle import main as split_main


def test_audio_envelope_offset_uses_side_minus_face_frame_convention():
    fps = 60.0
    hop_seconds = 1.0 / fps
    face_env = np.zeros(80, dtype=np.float32)
    side_env = np.zeros(80, dtype=np.float32)

    face_env[20] = 1.0
    side_env[25] = 1.0

    offset, confidence = split_main.estimate_offset_from_audio_envelopes(
        face_env, side_env, hop_seconds=hop_seconds, fps=fps
    )

    assert offset == 5
    assert confidence > 0.9


def test_choose_alignment_offset_averages_when_audio_agrees():
    offset, source = split_main.choose_alignment_offset(
        offset_kpt=8, offset_audio=10, audio_confidence=0.8, tolerance_frames=4
    )

    assert offset == 9
    assert source == "kpt_audio_avg"


def test_choose_alignment_offset_falls_back_to_keypoints_when_audio_disagrees():
    offset, source = split_main.choose_alignment_offset(
        offset_kpt=8, offset_audio=30, audio_confidence=0.8, tolerance_frames=4
    )

    assert offset == 8
    assert source == "kpt"
