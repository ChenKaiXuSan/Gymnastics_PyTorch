"""Version-2 objectives: cross-cycle target, reliability CE, feature periodicity / mirror symmetry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gymnastics.fusion.cycle_aware.cycle_target import CycleTargetConfig, cross_cycle_target
from gymnastics.fusion.cycle_aware.data.windows import CycleWindowDataset, WindowConfig
from gymnastics.fusion.cycle_aware.losses import (
    LossConfig,
    compute_losses,
    cycle_loss,
    feature_periodicity_loss,
    feature_symmetry_loss,
    reliability_loss,
)
from gymnastics.fusion.cycle_aware.model import CycleAwareFusionModel
from gymnastics.fusion.cycle_aware.train import compose_config, run
from tests.cycle_aware.conftest import make_sample

S = 8


def _cycles(n_cycles: int, joints: int = 4, noise: float = 0.0, seed: int = 0):
    """n identical cycles of S samples (plus optional noise) for both views."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(S, joints, 3)).astype(np.float32)
    frames = n_cycles * S
    view_a = np.concatenate([base + rng.normal(scale=noise, size=base.shape) for _ in range(n_cycles)]).astype(np.float32)
    view_b = np.concatenate([base + rng.normal(scale=noise, size=base.shape) for _ in range(n_cycles)]).astype(np.float32)
    valid = np.ones((frames, joints), dtype=bool)
    bounds = [(c * S, (c + 1) * S) for c in range(n_cycles)]
    return base, view_a, view_b, valid, bounds


def test_cross_cycle_target_is_leave_one_out_median():
    base, view_a, view_b, valid, bounds = _cycles(4)
    # Corrupt cycle 2 heavily in both views: the target of cycle 2 must not see it.
    view_a[2 * S : 3 * S] += 5.0
    view_b[2 * S : 3 * S] += 5.0
    target, conf = cross_cycle_target(view_a, view_b, valid, valid, bounds, samples_per_cycle=S, config=CycleTargetConfig(neighbors=None, tau=0.05))
    np.testing.assert_allclose(target[2 * S : 3 * S], base, atol=1e-6)  # median of the three clean cycles
    assert conf[2 * S : 3 * S].min() > 0.99
    # Other cycles see the corrupted one as one outlier among 6 candidates: the
    # median still equals the clean cycle and confidence stays high.
    np.testing.assert_allclose(target[:S], base, atol=1e-6)
    assert conf[:S].min() > 0.99
    # neighbors=1 for cycle 0 uses only cycle 1 (2 candidates) -> below min_candidates -> no target.
    _, conf_near = cross_cycle_target(view_a, view_b, valid, valid, bounds, samples_per_cycle=S, config=CycleTargetConfig(neighbors=1, min_candidates=4))
    assert conf_near[:S].max() == 0.0 and conf_near[S : 2 * S].min() > 0.0


def test_cross_cycle_confidence_falls_with_dispersion_and_validity():
    base, view_a, view_b, valid, bounds = _cycles(3, noise=0.0)
    view_a[S : 2 * S, 0] += 1.0  # joint 0 disagrees strongly in two of the four candidates
    view_b[S : 2 * S, 0] += 1.0
    valid_b = valid.copy()
    valid_b[:, 1] = False  # joint 1 only ever observed by view A -> 2 candidates per cycle
    target, conf = cross_cycle_target(view_a, view_b, valid, valid_b, bounds, samples_per_cycle=S, config=CycleTargetConfig(neighbors=None, tau=0.05, min_candidates=3))
    assert conf[2 * S :, 0].max() < conf[2 * S :, 2].min()  # dispersion lowers confidence
    assert conf[:, 1].max() == 0.0  # too few candidates
    with pytest.raises(ValueError):
        cross_cycle_target(view_a, view_b, valid, valid, [(0, S + 1)], samples_per_cycle=S, config=CycleTargetConfig())
    single, single_conf = cross_cycle_target(view_a[:S], view_b[:S], valid[:S], valid[:S], [(0, S)], samples_per_cycle=S, config=CycleTargetConfig())
    assert single_conf.max() == 0.0
    trimmed, _ = cross_cycle_target(view_a, view_b, valid, valid, bounds, samples_per_cycle=S, config=CycleTargetConfig(neighbors=None, aggregation="trimmed_mean"))
    assert np.isfinite(trimmed).all()


def test_window_dataset_ships_cycle_target(skeleton):
    sample = make_sample(skeleton, frames=64, period=16, mids=True)
    dataset = CycleWindowDataset([sample], skeleton=skeleton, window=WindowConfig(num_cycles=2, samples_per_cycle=S), split="train", cycle_target=CycleTargetConfig(neighbors=2, tau=0.05))
    item = dataset[0]
    assert item["cycle_target"].shape == (16, skeleton.num_joints, 3) and item["cycle_confidence"].shape == (16, skeleton.num_joints)
    assert (item["cycle_confidence"] > 0).any()
    disabled = CycleWindowDataset([sample], skeleton=skeleton, window=WindowConfig(num_cycles=2, samples_per_cycle=S), split="train")
    assert disabled[0]["cycle_confidence"].max() == 0.0


def test_cycle_loss_weights_by_confidence():
    pred = torch.zeros(1, 4, 2, 3)
    target = torch.ones(1, 4, 2, 3)
    valid = torch.ones(1, 4, 2, dtype=torch.bool)
    conf = torch.zeros(1, 4, 2)
    assert cycle_loss(pred, valid, target, conf, kind="l2", beta=0.05).item() == 0.0
    conf[0, :, 0] = 1.0
    assert cycle_loss(pred, valid, target, conf, kind="l2", beta=0.05).item() == pytest.approx(3.0)
    conf[0, :, 1] = 0.5
    pred[0, :, 1] = 1.0  # joint 1 exact
    assert cycle_loss(pred, valid, target, conf, kind="l2", beta=0.05).item() == pytest.approx(3.0 * 1.0 / 1.5)


def test_reliability_loss_labels_the_undamaged_view():
    logits = torch.zeros(1, 2, 3, 2)
    logits[0, :, 0, 0] = 5.0  # joint 0: strongly trusts A
    logits[0, :, 1, 1] = 5.0  # joint 1: strongly trusts B
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    mask_a = torch.zeros(1, 2, 3, dtype=torch.bool)
    mask_b = torch.zeros(1, 2, 3, dtype=torch.bool)
    assert reliability_loss(logits, mask_a, mask_b, valid, valid).item() == 0.0  # no labels
    mask_b[0, :, 0] = True  # B corrupted at joint 0 -> label A: logits already right
    assert reliability_loss(logits, mask_a, mask_b, valid, valid).item() < 0.01
    mask_b[0, :, 1] = True  # B corrupted at joint 1 but logits trust B -> large loss
    assert reliability_loss(logits, mask_a, mask_b, valid, valid).item() > 1.0
    mask_a[0, :, 1] = True  # both corrupted -> joint 1 drops out of the loss
    assert reliability_loss(logits, mask_a, mask_b, valid, valid).item() < 0.01


def test_feature_periodicity_and_mirror_symmetry():
    B, T, J, D = 1, 2 * S, 4, 6
    feature = torch.randn(B, S, J, D)
    valid = torch.ones(B, T, J, dtype=torch.bool)
    cycle_index = (torch.arange(T) // S)[None]
    periodic = torch.cat((feature, feature), dim=1)  # cycle 1 == cycle 0
    assert feature_periodicity_loss(periodic, valid, cycle_index, S).item() == pytest.approx(0.0, abs=1e-6)
    assert feature_periodicity_loss(torch.cat((feature, -feature), dim=1), valid, cycle_index, S).item() == pytest.approx(2.0, abs=1e-5)
    # Mirror symmetry: second half = first half with joints (0,1) swapped and (2,3) swapped.
    pairs = ((0, 1), (2, 3))
    half = torch.randn(B, S // 2, J, D)
    mirrored = half[:, :, [1, 0, 3, 2]]
    sym = torch.cat((half, mirrored), dim=1)  # one cycle of S samples
    half_index = ((torch.arange(S) % S) >= S // 2).long()[None]
    assert feature_symmetry_loss(sym, valid[:, :S], cycle_index[:, :S], half_index, S, pairs).item() == pytest.approx(0.0, abs=1e-6)
    unmirrored = torch.cat((half, half), dim=1)
    assert feature_symmetry_loss(unmirrored, valid[:, :S], cycle_index[:, :S], half_index, S, pairs).item() > 0.1
    assert feature_symmetry_loss(sym, valid[:, :S], cycle_index[:, :S], torch.full_like(half_index, -1), S, pairs).item() == 0.0


def test_compute_losses_v2_end_to_end(tiny_config, tiny_batch, skeleton):
    torch.manual_seed(0)
    model = CycleAwareFusionModel(tiny_config)
    batch = dict(tiny_batch)
    B, T, J = batch["pose_a"].shape[:3]
    batch["cycle_index"] = (torch.arange(T) // S)[None].repeat(B, 1)
    batch["half_index"] = ((torch.arange(T) % S) >= S // 2).long()[None].repeat(B, 1)
    batch["cycle_target"] = batch["pose_a"].clone()
    batch["cycle_confidence"] = torch.ones(B, T, J)
    batch["corruption_mask_a"] = torch.zeros(B, T, J, dtype=torch.bool)
    batch["corruption_mask_b"] = torch.zeros(B, T, J, dtype=torch.bool)
    batch["corruption_mask_a"][:, :, 2] = True
    output = model(**tiny_batch)
    losses = compute_losses(output, batch, skeleton=skeleton, config=LossConfig(), samples_per_cycle=S)
    for name, value in losses.as_dict().items():
        assert value.ndim == 0 and torch.isfinite(value), name
    assert losses.cycle.item() > 0 and losses.reliability.item() > 0 and losses.feature_periodicity.item() > 0 and losses.feature_symmetry.item() > 0
    assert losses.recovery.item() >= 0 and LossConfig().weights["recovery"] == 0.0
    losses.total.backward()
    assert all(p.grad is not None for p in model.reliability.parameters())
    with pytest.raises(ValueError):
        LossConfig(residual_norm="linf")


def test_hydra_v2_default_and_v1_preset(tmp_path: Path):
    cfg = compose_config(["experiment=smoke", f"output_root={tmp_path}", "run_name=v2"])
    assert cfg.loss.cycle_weight == 1.0 and cfg.loss.recovery_weight == 0.0 and cfg.data.cycle_target.enabled
    result = run(cfg)
    assert "test/cycle" in result["test_metrics"] and "test/cycle_target_error" in result["test_metrics"]
    assert any(k.startswith("train/reliability") for k in result["fit_metrics"])
    v1 = compose_config(["experiment=smoke", "experiment=v1", f"output_root={tmp_path}", "run_name=v1"]) if False else compose_config(["experiment=v1", f"output_root={tmp_path}", "run_name=v1", "trainer=debug", "samples_per_cycle=8", "model.hidden_dim=16", "model.num_heads=2", "data.options.subjects=4", "data.options.frames=48", "data.options.period=12"])
    assert v1.loss.recovery_weight == 1.0 and v1.loss.cycle_weight == 0.0 and v1.loss.residual_norm == "l2"
    run(v1)
