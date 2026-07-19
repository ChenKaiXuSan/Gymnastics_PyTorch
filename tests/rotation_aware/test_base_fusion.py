import torch

from fuse.rotation_aware.base_fusion import arithmetic_fusion, quality_weighted_fusion


def test_base_fusion_falls_back_to_only_valid_view():
    face = torch.tensor([[[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]]])
    side = face + 10.0
    face_valid = torch.tensor([[[True, True]]])
    side_valid = torch.zeros_like(face_valid)
    qf = torch.ones(1, 1)
    qs = torch.zeros(1, 1)

    out = quality_weighted_fusion(face, side, face_valid, side_valid, qf, qs)

    torch.testing.assert_close(out.points[face_valid], face[face_valid])
    assert torch.equal(out.valid, face_valid)


def test_arithmetic_fusion_is_mask_aware_and_handles_both_invalid():
    face = torch.tensor([[[[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]]])
    side = torch.tensor([[[[4.0, 8.0, 12.0], [16.0, 20.0, 24.0]]]])
    face_valid = torch.tensor([[[True, False]]])
    side_valid = torch.tensor([[[True, False]]])

    out = arithmetic_fusion(face, side, face_valid, side_valid)

    torch.testing.assert_close(out.points[0, 0, 0], torch.tensor([3.0, 6.0, 9.0]))
    torch.testing.assert_close(out.points[0, 0, 1], torch.zeros(3))
    assert out.valid.tolist() == [[[True, False]]]
    assert torch.isfinite(out.points).all()


def test_quality_fusion_is_view_swap_invariant_and_detaches_quality_weights():
    face = torch.tensor([[[[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]], requires_grad=True)
    side = torch.tensor([[[[3.0, 0.0, 0.0], [9.0, 0.0, 0.0]]]])
    face_valid = torch.tensor([[[True, True]]])
    side_valid = torch.tensor([[[True, True]]])
    qf = torch.tensor([[0.75]], requires_grad=True)
    qs = torch.tensor([[0.25]], requires_grad=True)

    forward = quality_weighted_fusion(face, side, face_valid, side_valid, qf, qs)
    reverse = quality_weighted_fusion(side, face, side_valid, face_valid, qs, qf)
    forward.points.sum().backward()

    torch.testing.assert_close(forward.points, reverse.points)
    assert qf.grad is None
    assert qs.grad is None


def test_base_fusions_reject_mismatched_shapes():
    points = torch.zeros(1, 1, 3, 3)
    valid = torch.ones(1, 1, 3, dtype=torch.bool)

    try:
        arithmetic_fusion(points, points[:, :, :-1], valid, valid[:, :, :-1])
    except ValueError as error:
        assert "equal shape" in str(error)
    else:
        raise AssertionError("expected shape validation")
