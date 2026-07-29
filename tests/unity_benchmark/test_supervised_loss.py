from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gymnastics.benchmarks.unity.mapping import map_mhr70_to_unity
from gymnastics.benchmarks.unity.supervised_loss import (
    UnitySupervisedLossConfig,
    apply_torch_sim3,
    compute_unity_supervised_loss,
    masked_window_sim3,
    torch_map_mhr70_to_unity16,
)


def test_torch_mapping_matches_numpy_and_propagates_derived_gradients() -> None:
    points = torch.arange(
        2 * 70 * 3, dtype=torch.float32
    ).reshape(2, 70, 3)
    points.requires_grad_(True)
    valid = torch.ones((2, 70), dtype=torch.bool)

    mapped, mapped_valid = torch_map_mhr70_to_unity16(points, valid)
    expected = map_mhr70_to_unity(
        points.detach().numpy(), valid.numpy()
    )

    np.testing.assert_allclose(
        mapped.detach().numpy(), expected.points, rtol=0, atol=0
    )
    assert torch.equal(
        mapped_valid, torch.from_numpy(np.array(expected.valid, copy=True))
    )
    mapped[:, 0].sum().backward()
    assert torch.all(points.grad[:, 9] == 0.5)
    assert torch.all(points.grad[:, 10] == 0.5)


def test_masked_window_sim3_recovers_one_transform_and_has_gradients() -> None:
    torch.manual_seed(7)
    target = torch.randn(2, 12, 16, 3)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    prediction = (
        1.7 * torch.einsum("btjc,cd->btjd", target, rotation)
        + torch.tensor([2.0, -1.0, 0.4])
    )
    prediction.requires_grad_(True)
    valid = torch.ones((2, 12, 16), dtype=torch.bool)

    transform = masked_window_sim3(prediction, target, valid)
    aligned = apply_torch_sim3(prediction, transform)
    loss = torch.linalg.vector_norm(aligned - target, dim=-1).mean()

    assert loss.item() < 1e-4
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_masked_window_sim3_does_not_fit_each_frame_independently() -> None:
    torch.manual_seed(11)
    target = torch.randn(1, 2, 16, 3)
    prediction = target.clone()
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    prediction[:, 1] = torch.einsum(
        "bjc,cd->bjd", prediction[:, 1], rotation
    )
    valid = torch.ones((1, 2, 16), dtype=torch.bool)

    transform = masked_window_sim3(prediction, target, valid)
    aligned = apply_torch_sim3(prediction, transform)
    residual = torch.linalg.vector_norm(
        aligned[:, 1] - target[:, 1], dim=-1
    ).mean()

    assert residual.item() > 1e-2


@pytest.mark.parametrize(
    ("prediction", "target", "valid"),
    [
        (
            torch.randn(1, 1, 16, 3),
            torch.randn(1, 1, 16, 3),
            torch.tensor(
                [[[True, True] + [False] * 14]], dtype=torch.bool
            ).reshape(1, 1, 16),
        ),
        (
            torch.ones(1, 2, 16, 3),
            torch.ones(1, 2, 16, 3),
            torch.ones(1, 2, 16, dtype=torch.bool),
        ),
    ],
)
def test_masked_window_sim3_rejects_degenerate_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="degenerate Sim3"):
        masked_window_sim3(prediction, target, valid)


def _supervised_loss_fixture():
    torch.manual_seed(17)
    fused = torch.randn(1, 8, 70, 3, requires_grad=True)
    output = SimpleNamespace(
        fused_kpts=fused,
        valid=torch.ones((1, 8, 70), dtype=torch.bool),
    )
    target, target_valid = torch_map_mhr70_to_unity16(
        fused.detach(), output.valid
    )
    target_valid[:, 0, 0] = False
    padding = torch.ones((1, 8), dtype=torch.bool)
    padding[:, -1] = False
    batch = {
        "gt_unity16_m": target.clone(),
        "gt_valid": target_valid.clone(),
        "padding_mask": padding,
    }
    return output, batch


def test_supervised_loss_uses_only_valid_non_padded_unity16_points() -> None:
    output, batch = _supervised_loss_fixture()
    config = UnitySupervisedLossConfig(
        unity_3d_weight=1.0,
        self_supervised_weight=0.1,
        smooth_l1_beta_m=0.02,
    )
    self_loss = output.fused_kpts.sum() * 0.0

    original = compute_unity_supervised_loss(
        output, batch, config, self_supervised=self_loss
    )
    batch["gt_unity16_m"][~batch["gt_valid"]] = 1e6
    batch["gt_unity16_m"][~batch["padding_mask"]] = -1e6
    changed = compute_unity_supervised_loss(
        output, batch, config, self_supervised=self_loss
    )

    torch.testing.assert_close(original.unity_3d, changed.unity_3d)


def test_supervised_loss_forms_exact_weighted_total_and_finite_gradients() -> None:
    output, batch = _supervised_loss_fixture()
    config = UnitySupervisedLossConfig(
        unity_3d_weight=1.0,
        self_supervised_weight=0.1,
        smooth_l1_beta_m=0.02,
    )
    self_loss = output.fused_kpts.square().mean()

    losses = compute_unity_supervised_loss(
        output, batch, config, self_supervised=self_loss
    )

    torch.testing.assert_close(
        losses.total, losses.unity_3d + 0.1 * losses.self_supervised
    )
    losses.total.backward()
    assert output.fused_kpts.grad is not None
    assert torch.isfinite(output.fused_kpts.grad).all()


def test_supervised_loss_rejects_nan_in_valid_target() -> None:
    output, batch = _supervised_loss_fixture()
    batch["gt_unity16_m"][0, 1, 1, 0] = torch.nan

    with pytest.raises(FloatingPointError, match="non-finite"):
        compute_unity_supervised_loss(
            output,
            batch,
            UnitySupervisedLossConfig(),
            self_supervised=output.fused_kpts.sum() * 0.0,
        )


def test_supervised_loss_rejects_non_finite_total() -> None:
    output, batch = _supervised_loss_fixture()

    with pytest.raises(FloatingPointError, match="non-finite"):
        compute_unity_supervised_loss(
            output,
            batch,
            UnitySupervisedLossConfig(),
            self_supervised=torch.tensor(float("inf")),
        )
