"""Behavioral tests for per-frame bidirectional cross-view attention."""

from __future__ import annotations

import pytest
import torch

from gymnastics.fusion.rotation_aware.cross_attention import (
    BidirectionalCrossViewAttention,
)


def test_attention_rejects_non_divisible_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        BidirectionalCrossViewAttention(hidden_channels=10, num_heads=4)


def test_attention_preserves_shape_and_swaps_directionally() -> None:
    torch.manual_seed(3)
    face = torch.randn(2, 3, 5, 8)
    side = torch.randn(2, 3, 5, 8)
    valid_face = torch.ones(2, 3, 5, dtype=torch.bool)
    valid_side = torch.ones(2, 3, 5, dtype=torch.bool)
    block = BidirectionalCrossViewAttention(8, num_heads=2).eval()

    face_out, side_out = block(face, side, valid_face, valid_side)
    swapped_side, swapped_face = block(side, face, valid_side, valid_face)

    assert face_out.shape == face.shape
    assert side_out.shape == side.shape
    torch.testing.assert_close(face_out, swapped_face, atol=1e-6, rtol=0)
    torch.testing.assert_close(side_out, swapped_side, atol=1e-6, rtol=0)


def test_attention_handles_empty_source_frames_without_nan() -> None:
    torch.manual_seed(5)
    face = torch.randn(1, 2, 3, 8, requires_grad=True)
    side = torch.randn(1, 2, 3, 8, requires_grad=True)
    valid_face = torch.tensor([[[True, True, False], [True, False, False]]])
    valid_side = torch.tensor([[[True, False, True], [False, False, False]]])
    block = BidirectionalCrossViewAttention(8, num_heads=2)

    face_out, side_out = block(face, side, valid_face, valid_side)

    assert torch.isfinite(face_out).all()
    assert torch.isfinite(side_out).all()
    assert torch.equal(
        face_out[~valid_face], torch.zeros_like(face_out[~valid_face])
    )
    assert torch.equal(
        side_out[~valid_side], torch.zeros_like(side_out[~valid_side])
    )
    (face_out.square().sum() + side_out.square().sum()).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in block.parameters()
    )


@pytest.mark.parametrize(
    ("face_shape", "side_shape", "mask_shape", "message"),
    [
        ((1, 2, 3, 8), (1, 2, 4, 8), (1, 2, 3), "equal shape"),
        ((1, 2, 3), (1, 2, 3), (1, 2), r"\[B, T, J, C\]"),
        ((1, 2, 3, 8), (1, 2, 3, 8), (1, 2, 2), r"\[B, T, J\]"),
    ],
)
def test_attention_rejects_malformed_shapes(
    face_shape: tuple[int, ...],
    side_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
    message: str,
) -> None:
    face = torch.randn(face_shape)
    side = torch.randn(side_shape)
    valid_face = torch.ones(mask_shape, dtype=torch.bool)
    valid_side = torch.ones(mask_shape, dtype=torch.bool)

    with pytest.raises(ValueError, match=message):
        BidirectionalCrossViewAttention(8, num_heads=2)(
            face, side, valid_face, valid_side
        )
