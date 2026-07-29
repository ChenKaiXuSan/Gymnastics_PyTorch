from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gymnastics.common.skeletons.mhr70 import mhr_names
from gymnastics.fusion.rotation_aware.config import load_skeleton_spec
from gymnastics.fusion.rotation_aware.dataset import (
    PosePairWindowDataset,
    SplitManifest,
    WindowConfig,
)
from gymnastics.fusion.rotation_aware.real_camera_data import (
    CameraWindowDataset,
    load_real_camera_trials,
)
from gymnastics.fusion.rotation_aware.schema import PosePairTrial


SPEC = load_skeleton_spec(Path("configs/fusion/skeleton_mhr70.yaml"))


def _trial() -> PosePairTrial:
    points = (
        np.arange(4 * len(mhr_names) * 3, dtype=np.float32)
        .reshape(4, len(mhr_names), 3)
        + 1.0
    )
    valid = np.ones((4, len(mhr_names)), dtype=bool)
    return PosePairTrial(
        face=points,
        side=points + 5.0,
        valid_face=valid,
        valid_side=valid,
        timestamps=np.arange(4, dtype=np.float64) / 60.0,
        face_map=np.asarray((10, 12, 14, 16), dtype=np.int32),
        side_map=np.asarray((21, 23, 25, 27), dtype=np.int32),
        joint_names=tuple(mhr_names),
        person_id="1",
        trial_id="cycle_000",
        fps=60.0,
    )


def _write_calibration(path: Path) -> None:
    np.savez(
        path,
        camera_matrix=np.asarray(
            ((1000.0, 0.0, 540.0), (0.0, 1000.0, 960.0), (0.0, 0.0, 1.0))
        ),
        dist_coeffs=np.zeros(5),
        image_size=np.asarray((1080, 1920)),
    )


def _write_sam3d(path: Path, x_offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keypoints = np.stack(
        (
            np.linspace(250.0, 800.0, len(mhr_names)) + x_offset,
            np.linspace(350.0, 1450.0, len(mhr_names)),
        ),
        axis=-1,
    ).astype(np.float32)
    np.savez(path, output=np.asarray({"pred_keypoints_2d": keypoints}, dtype=object))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    sam3d = tmp_path / "sam3d"
    for frame in (10, 12, 14, 16):
        _write_sam3d(
            sam3d / "1" / "face" / f"{frame:06d}_sam3d_body.npz",
            0.0,
        )
    for frame in (21, 23, 25, 27):
        _write_sam3d(
            sam3d / "1" / "side" / f"{frame:06d}_sam3d_body.npz",
            20.0,
        )
    audit = tmp_path / "estimated_extrinsics.json"
    audit.write_text(
        json.dumps(
            {
                "persons": {
                    "1": {
                        "R": np.eye(3).tolist(),
                        "t": [3.0, 0.0, 0.0],
                        "rig_cluster": 1,
                        "method": "per_person",
                        "inlier_ratio": 0.25,
                        "num_frames": 120,
                        "holdout_reproj_px": 4.0,
                        "bone_cv_pct": 2.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    face_calibration = tmp_path / "face.npz"
    side_calibration = tmp_path / "side.npz"
    _write_calibration(face_calibration)
    _write_calibration(side_calibration)
    return sam3d, audit, face_calibration, side_calibration


def test_real_trial_joins_2d_by_frame_map_without_reference_3d(
    tmp_path: Path,
) -> None:
    sam3d, audit, face_calibration, side_calibration = _fixture(tmp_path)

    trials = load_real_camera_trials(
        raw_trials=[_trial()],
        skeleton=SPEC,
        sam3d_person_root=sam3d,
        camera_audit_path=audit,
        face_calibration_path=face_calibration,
        side_calibration_path=side_calibration,
        ablation="G4",
    )

    assert len(trials) == 1
    assert trials[0].camera_features is not None
    assert trials[0].camera_features.joint_features.shape == (4, 70, 8)
    assert trials[0].camera_fit is not None
    assert trials[0].camera_fit.person_id == "1"
    assert trials[0].canonical_trial.trial.face_map.tolist() == [10, 12, 14, 16]
    assert not hasattr(trials[0], "triangulated_3d")


def test_real_ablation_masks_and_wrong_camera_are_deterministic(
    tmp_path: Path,
) -> None:
    sam3d, audit, face_calibration, side_calibration = _fixture(tmp_path)

    def build(ablation: str):
        return load_real_camera_trials(
            raw_trials=[_trial()],
            skeleton=SPEC,
            sam3d_person_root=sam3d,
            camera_audit_path=audit,
            face_calibration_path=face_calibration,
            side_calibration_path=side_calibration,
            ablation=ablation,
        )[0]

    g0, g1, g2, g3, g5a, g5b = (
        build("G0"),
        build("G1"),
        build("G2"),
        build("G3"),
        build("G5"),
        build("G5"),
    )

    assert g0.camera_features is None
    assert g1.camera_features is not None
    assert g2.camera_features is not None
    assert g3.camera_features is not None
    assert not g1.camera_features.joint_features.any()
    assert not g2.camera_features.joint_features.any()
    assert g3.camera_features.joint_features.any()
    np.testing.assert_array_equal(
        g5a.camera_fit.fitted.rotation_face_to_side,
        g5b.camera_fit.fitted.rotation_face_to_side,
    )
    assert not np.allclose(
        g3.camera_fit.fitted.rotation_face_to_side,
        g5a.camera_fit.fitted.rotation_face_to_side,
    )


def test_camera_window_dataset_pads_the_matching_cycle_features(
    tmp_path: Path,
) -> None:
    sam3d, audit, face_calibration, side_calibration = _fixture(tmp_path)
    real = load_real_camera_trials(
        raw_trials=[_trial()],
        skeleton=SPEC,
        sam3d_person_root=sam3d,
        camera_audit_path=audit,
        face_calibration_path=face_calibration,
        side_calibration_path=side_calibration,
        ablation="G4",
    )
    base = PosePairWindowDataset(
        [real[0].canonical_trial.trial],
        skeleton=SPEC,
        manifest=SplitManifest(train=("1",), val=(), test=()),
        split="train",
        config=WindowConfig(length=8, train_stride=4, eval_stride=4),
    )

    sample = CameraWindowDataset(base, real)[0]

    assert sample["camera_global_features"].shape == (19,)
    assert sample["camera_joint_features"].shape == (8, 70, 8)
    assert sample["camera_valid"][:4].all()
    assert not sample["camera_valid"][4:].any()
    assert not sample["camera_joint_features"][4:].any()


def test_missing_mapped_sam3d_frame_is_rejected(tmp_path: Path) -> None:
    sam3d, audit, face_calibration, side_calibration = _fixture(tmp_path)
    missing = sam3d / "1" / "side" / "000027_sam3d_body.npz"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="000027_sam3d_body"):
        load_real_camera_trials(
            raw_trials=[_trial()],
            skeleton=SPEC,
            sam3d_person_root=sam3d,
            camera_audit_path=audit,
            face_calibration_path=face_calibration,
            side_calibration_path=side_calibration,
            ablation="G4",
        )
