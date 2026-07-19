from pathlib import Path

import numpy as np
import torch

from fuse.metadata.mhr70 import mhr_names
from fuse.rotation_aware.config import load_skeleton_spec
from fuse.rotation_aware.inference import (
    canonicalize_trial,
    overlap_taper,
    run_inference,
)
from fuse.rotation_aware.model import RotationAwareFusionModel
from fuse.rotation_aware.schema import PosePairTrial, valid_from_points


SPEC = load_skeleton_spec(Path("configs/fuse/skeleton_mhr70.yaml"))


def _trial(frames: int = 130) -> PosePairTrial:
    base = np.ones((frames, len(mhr_names), 3), dtype=np.float32)
    base[..., 0] += np.linspace(0.0, 1.0, frames)[:, None]
    base[:, 9, 0] = -1.0
    base[:, 10, 0] = 1.0
    base[:, 5, 1] = 2.0
    base[:, 6, 1] = 2.0
    base[:, 2, 1] = 3.0
    return PosePairTrial(
        face=base,
        side=base + np.array([3.0, 0.0, 0.0], dtype=np.float32),
        valid_face=valid_from_points(base),
        valid_side=valid_from_points(base + 1),
        timestamps=np.arange(frames, dtype=np.float64) / 50.0,
        face_map=np.arange(frames, dtype=np.int32),
        side_map=np.arange(frames, dtype=np.int32),
        joint_names=tuple(mhr_names),
        person_id="1",
        trial_id="cycle_000",
        fps=50.0,
    )


def test_canonicalizes_full_trial_and_restores_to_face_reference() -> None:
    trial = _trial()
    canonical = canonicalize_trial(trial, SPEC)

    assert canonical.face_transform.scale.shape == (1,)
    assert canonical.face_transform.scale.item() > 0
    np.testing.assert_allclose(
        canonical.restore_face(canonical.trial.face), trial.face, atol=1e-5
    )


def test_overlap_taper_is_deterministic_and_never_zero() -> None:
    weights = overlap_taper(128)
    assert weights.shape == (128,)
    assert np.all(weights > 0)
    np.testing.assert_allclose(weights, overlap_taper(128))


def test_inference_writes_compatible_finite_output_with_physical_omega(
    tmp_path: Path,
) -> None:
    torch.manual_seed(2)
    trial = _trial()
    model = RotationAwareFusionModel(SPEC, hidden_channels=8)
    result = run_inference(
        model,
        trial,
        SPEC,
        output_root=tmp_path,
        run_id="tiny",
        window_length=128,
        stride=64,
    )

    with np.load(result.sequence_path) as data:
        assert {
            "kpts_world",
            "kpts_body",
            "kpts_fused_canonical",
            "kpts_base_canonical",
            "theta_fused_rad",
            "omega_fused_rad_s",
            "quality_face",
            "quality_side",
            "frame_valid",
            "face_map",
            "side_map",
            "metadata",
        } <= set(data.files)
        assert data["kpts_world"].shape == (130, len(mhr_names), 3)
        assert data["omega_fused_rad_s"].shape == (130,)
        assert np.isfinite(data["kpts_world"]).all()
        assert np.isfinite(data["omega_fused_rad_s"]).all()
        assert data["face_map"].tolist() == trial.face_map.tolist()
        assert "face_reference_uncalibrated" in str(data["metadata"].item())


def test_inference_does_not_restore_invalid_canonical_joints_as_fake_world_points(
    tmp_path: Path,
) -> None:
    trial = _trial(8)
    face = np.array(trial.face, copy=True)
    side = np.array(trial.side, copy=True)
    valid_face = np.array(trial.valid_face, copy=True)
    valid_side = np.array(trial.valid_side, copy=True)
    face[:, 20] = 0
    side[:, 20] = 0
    valid_face[:, 20] = False
    valid_side[:, 20] = False
    invalid = PosePairTrial(
        face,
        side,
        valid_face,
        valid_side,
        trial.timestamps,
        trial.face_map,
        trial.side_map,
        trial.joint_names,
        "1",
        "cycle_001",
        trial.fps,
    )
    result = run_inference(
        RotationAwareFusionModel(SPEC, hidden_channels=8),
        invalid,
        SPEC,
        output_root=tmp_path,
        run_id="tiny",
    )

    with np.load(result.sequence_path) as data:
        assert "joint_valid" in data.files
        assert not data["joint_valid"][:, 20].any()
        assert not data["kpts_world"][:, 20].any()
        assert not data["kpts_body"][:, 20].any()
