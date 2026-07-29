from __future__ import annotations

import numpy as np
import pytest
import torch

from gymnastics.benchmarks.unity.extrinsic_models import (
    ExtrinsicGateModel,
    ExtrinsicResidualTCN,
    LearnableTriangulationModel,
    relative_camera_rotation,
)
from gymnastics.benchmarks.unity.schema import UnityCamera


def _camera(
    camera_id: str,
    camera_to_world: np.ndarray,
) -> UnityCamera:
    world_to_camera = np.linalg.inv(camera_to_world)
    return UnityCamera(
        camera_id=camera_id,
        image_size=(640, 480),
        camera_to_world=camera_to_world,
        world_to_camera=world_to_camera,
        clip_projection=np.eye(4),
    )


def test_relative_camera_rotation_maps_cam1_vectors_into_cam0() -> None:
    cam0_to_world = np.eye(4)
    cam1_to_world = np.eye(4)
    cam1_to_world[:3, :3] = np.asarray(
        (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    rotation = relative_camera_rotation(
        _camera("cam0", cam0_to_world),
        _camera("cam1", cam1_to_world),
    )
    cam1_x = np.asarray((1.0, 0.0, 0.0))
    np.testing.assert_allclose(
        rotation @ cam1_x,
        np.asarray((0.0, 1.0, 0.0)),
        atol=1e-7,
    )


def test_extrinsic_gate_initializes_to_equal_average_after_rotation() -> None:
    model = ExtrinsicGateModel(joint_count=2, pelvis_index=0, hidden_channels=8)
    face = torch.tensor([[[[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]]])
    side = torch.tensor([[[[4.0, 5.0, 6.0], [4.0, 6.0, 6.0]]]])
    valid = torch.ones((1, 1, 2), dtype=torch.bool)
    rotation = torch.eye(3)
    output = model(face, side, valid, valid, rotation)
    expected = torch.tensor([[[[1.0, 2.0, 3.0], [1.5, 2.5, 3.0]]]])
    torch.testing.assert_close(output.points, expected)
    torch.testing.assert_close(
        output.diagnostics["gate"],
        torch.full((1, 1, 2), 0.5),
    )


def test_extrinsic_gate_copies_the_only_valid_view() -> None:
    model = ExtrinsicGateModel(joint_count=2, pelvis_index=0, hidden_channels=8)
    face = torch.tensor([[[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]]])
    side = torch.tensor([[[[4.0, 5.0, 6.0], [4.0, 6.0, 6.0]]]])
    valid_face = torch.tensor([[[True, False]]])
    valid_side = torch.tensor([[[True, True]]])
    output = model(face, side, valid_face, valid_side, torch.eye(3))
    torch.testing.assert_close(output.points[0, 0, 0], face[0, 0, 0])
    torch.testing.assert_close(
        output.points[0, 0, 1],
        torch.tensor([1.0, 3.0, 3.0]),
    )
    assert output.valid.all()


def test_extrinsic_residual_is_bounded_by_configured_limit() -> None:
    model = ExtrinsicResidualTCN(
        joint_count=2,
        pelvis_index=0,
        hidden_channels=8,
        max_delta_m=0.05,
    )
    with torch.no_grad():
        model.output_head.bias.fill_(100.0)
    face = torch.zeros((1, 2, 2, 3))
    side = torch.zeros_like(face)
    valid = torch.ones((1, 2, 2), dtype=torch.bool)
    output = model(face, side, valid, valid, torch.eye(3))
    assert torch.all(output.diagnostics["delta"].abs() <= 0.0500001)
    torch.testing.assert_close(
        output.points,
        torch.full_like(output.points, 0.05),
        rtol=0,
        atol=1e-6,
    )


def test_learnable_triangulation_recovers_noise_free_world_point() -> None:
    model = LearnableTriangulationModel(hidden_channels=8)
    projection = torch.tensor(
        [
            [[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            [[100.0, 0.0, 0.0, -100.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        ]
    )
    pixels = torch.tensor([[[[[40.0, 20.0]], [[20.0, 20.0]]]]])
    valid = torch.ones((1, 1, 2, 1), dtype=torch.bool)
    output = model(
        pixels,
        valid,
        projection,
        image_size=torch.tensor([[200.0, 100.0], [200.0, 100.0]]),
    )
    torch.testing.assert_close(
        output.points,
        torch.tensor([[[[2.0, 1.0, 5.0]]]]),
        rtol=0,
        atol=1e-4,
    )
    assert output.valid.item()


def test_learnable_triangulation_rejects_a_single_valid_view() -> None:
    model = LearnableTriangulationModel(hidden_channels=8)
    projection = torch.eye(3, 4).repeat(2, 1, 1)
    pixels = torch.zeros((1, 1, 2, 1, 2))
    valid = torch.tensor([[[[True], [False]]]])
    output = model(
        pixels,
        valid,
        projection,
        image_size=torch.ones((2, 2)),
    )
    assert not output.valid.item()
    torch.testing.assert_close(output.points, torch.zeros_like(output.points))


def test_models_reject_non_rotation_camera_geometry() -> None:
    model = ExtrinsicGateModel(joint_count=2, pelvis_index=0, hidden_channels=8)
    points = torch.zeros((1, 1, 2, 3))
    valid = torch.ones((1, 1, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="rotation"):
        model(points, points, valid, valid, torch.ones((3, 3)))
