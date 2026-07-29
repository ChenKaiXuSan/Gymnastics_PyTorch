from __future__ import annotations

from pathlib import Path

import numpy as np

from gymnastics.benchmarks.unity.camera_guided_data import (
    UnityCameraGuidedWindowDataset,
    build_camera_guided_sequences,
    camera_conditioning_config,
)
from gymnastics.benchmarks.unity.dataset import load_unity_benchmark
from gymnastics.benchmarks.unity.supervised_data import UNITY_SUPERVISED_FOLDS


UNITY_ROOT = Path("/home/data/xchen/gymnastics/unity_benchmark")
SAM3D_ROOT = Path("local/runs/unity_benchmark/sam3d")
SKELETON = Path("configs/fusion/skeleton_mhr70.yaml")


def _sequences(ablation: str):
    benchmark = load_unity_benchmark(UNITY_ROOT)
    return build_camera_guided_sequences(
        benchmark,
        SAM3D_ROOT,
        skeleton_path=SKELETON,
        fps=60.0,
        fold=UNITY_SUPERVISED_FOLDS["left_to_right"],
        ablation=ablation,
    )


def test_camera_guided_sequences_expose_no_unity_3d_and_fit_train_only() -> None:
    sequences = _sequences("G4")
    train = sequences["continuous_left_060_r00"]
    test = sequences["continuous_right_060_r00"]

    assert not hasattr(train, "gt_unity16_m")
    assert not hasattr(test, "gt_unity16_m")
    assert train.fitted_camera.fit_sample_ids.tolist() == train.sample_ids.tolist()
    assert not set(train.fitted_camera.fit_sample_ids) & set(test.sample_ids)
    assert train.camera_features is not None
    assert test.camera_features is not None
    assert train.camera_features.joint_features.shape == (97, 70, 8)


def test_g0_has_no_camera_tensors_and_g1_to_g5_have_documented_modes() -> None:
    assert camera_conditioning_config("G0") is None
    assert camera_conditioning_config("G1").mode == "additive"
    assert camera_conditioning_config("G2").mode == "additive"
    assert camera_conditioning_config("G3").mode == "additive"
    assert camera_conditioning_config("G4").mode == "film"
    assert camera_conditioning_config("G5").mode == "film"

    sequence = _sequences("G0")["continuous_left_060_r00"]
    dataset = UnityCameraGuidedWindowDataset(
        sequence,
        skeleton_path=SKELETON,
        length=128,
        stride=32,
    )
    sample = dataset[0]
    assert "camera_global_features" not in sample
    assert "camera_joint_features" not in sample
    assert "camera_valid" not in sample


def test_camera_guided_window_pads_features_without_changing_global_fit() -> None:
    sequence = _sequences("G4")["continuous_left_060_r00"]
    dataset = UnityCameraGuidedWindowDataset(
        sequence,
        skeleton_path=SKELETON,
        length=128,
        stride=32,
    )

    sample = dataset[0]

    assert sample["camera_global_features"].shape == (19,)
    assert sample["camera_joint_features"].shape == (128, 70, 8)
    assert sample["camera_valid"].shape == (128, 70)
    assert np.isfinite(sample["camera_global_features"].numpy()).all()
    assert not sample["camera_valid"][97:].any()
    assert not sample["camera_joint_features"][97:].any()
    assert sample["complete_cycle"]


def test_g1_masks_quality_and_joint_geometry_while_g2_adds_quality() -> None:
    g1 = _sequences("G1")["continuous_left_060_r00"].camera_features
    g2 = _sequences("G2")["continuous_left_060_r00"].camera_features
    g3 = _sequences("G3")["continuous_left_060_r00"].camera_features
    assert g1 is not None and g2 is not None and g3 is not None

    assert np.all(g1.global_features[-2:] == 0)
    assert np.any(g2.global_features[-2:] != 0)
    assert np.all(g1.joint_features == 0)
    assert np.all(g2.joint_features == 0)
    assert np.any(g3.joint_features != 0)
