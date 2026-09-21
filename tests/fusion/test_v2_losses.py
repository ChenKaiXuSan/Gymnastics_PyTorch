"""Version-2 objectives: cross-cycle target, reliability CE, feature periodicity / mirror symmetry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fusion.cycle_target import ConfidenceConfig, CycleTargetConfig, ViewConsensusConfig, cross_cycle_target, view_consensus
from fusion.data.windows import CycleWindowDataset, WindowConfig
from fusion.diagnostics import feature_variance, film_statistics, phase_similarity, reliability_statistics, residual_statistics
from fusion.losses import (
    DeadZoneConfig,
    LossConfig,
    compute_losses,
    contrastive_periodicity_loss,
    cycle_loss,
    feature_periodicity_loss,
    feature_symmetry_loss,
    natural_variation_dead_zone,
    phase_pair_masks,
    reliability_loss,
)
from fusion.model import CycleAwareFusionModel
from fusion.train import compose_config, run
from tests.fusion.conftest import make_sample

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


def _cfg(**kwargs) -> CycleTargetConfig:
    return CycleTargetConfig.from_mapping(kwargs)


def test_cross_cycle_target_is_leave_one_out_median():
    base, view_a, view_b, valid, bounds = _cycles(4)
    # Corrupt cycle 2 heavily in both views: the target of cycle 2 must not see it.
    view_a[2 * S : 3 * S] += 5.0
    view_b[2 * S : 3 * S] += 5.0
    ref = cross_cycle_target(view_a, view_b, valid, valid, bounds, samples_per_cycle=S, config=_cfg(neighbors=None))
    np.testing.assert_allclose(ref.target[2 * S : 3 * S], base, atol=1e-6)  # median of the three clean cycles
    assert ref.confidence[2 * S : 3 * S].min() > 0.99 and (ref.candidate_count[2 * S : 3 * S] == 3).all()
    # Other cycles see the corrupted one as one outlier among 3 candidates: the
    # median still equals the clean cycle; the MAD stays 0 (median of 0, 0, big).
    np.testing.assert_allclose(ref.target[:S], base, atol=1e-6)
    assert ref.confidence[:S].min() > 0.99 and np.isfinite(ref.dispersion[:S]).all()
    # neighbors=1 for cycle 0 uses only cycle 1 (1 candidate) -> below min_candidates=2 -> no target.
    near = cross_cycle_target(view_a, view_b, valid, valid, bounds, samples_per_cycle=S, config=_cfg(neighbors=1))
    assert near.confidence[:S].max() == 0.0 and near.confidence[S : 2 * S].min() > 0.0
    assert np.isinf(near.dispersion[:S]).all()


def test_view_consensus_rules():
    a = np.zeros((2, 3, 3), dtype=np.float32)
    b = np.zeros((2, 3, 3), dtype=np.float32)
    b[:, 0, 0] = 0.1  # small disagreement -> mean
    b[:, 1, 0] = 1.0  # large disagreement -> no consensus
    valid_a = np.ones((2, 3), dtype=bool)
    valid_b = np.ones((2, 3), dtype=bool)
    valid_b[:, 2] = False  # joint 2 only in A
    pose, valid, disagreement = view_consensus(a, b, valid_a, valid_b, ViewConsensusConfig(disagreement_threshold=0.15))
    assert valid[:, 0].all() and not valid[:, 1].any() and valid[:, 2].all()
    np.testing.assert_allclose(pose[:, 0, 0], 0.05)
    np.testing.assert_allclose(pose[:, 2], a[:, 2])
    assert np.isnan(disagreement[:, 2]).all() and disagreement[0, 1] == pytest.approx(1.0)
    none, none_valid, _ = view_consensus(a, b, np.zeros_like(valid_a), np.zeros_like(valid_b), ViewConsensusConfig())
    assert not none_valid.any() and not none.any()


def test_cross_cycle_confidence_falls_with_dispersion_and_validity():
    base, view_a, view_b, valid, bounds = _cycles(4, noise=0.0)
    for c in (1, 2):  # joint 0 disagrees in two of the three other cycles of cycle 3
        view_a[c * S : (c + 1) * S, 0] += 1.0 * c
        view_b[c * S : (c + 1) * S, 0] += 1.0 * c
    valid_b = valid.copy()
    valid_b[:, 1] = False  # joint 1 only observed by view A -> consensus from A alone, still one candidate per cycle
    ref = cross_cycle_target(view_a, view_b, valid, valid_b, bounds, samples_per_cycle=S, config=_cfg(neighbors=None))
    assert ref.confidence[3 * S :, 0].max() < ref.confidence[3 * S :, 2].min()  # dispersion lowers confidence
    assert ref.confidence[:, 1].min() > 0.99  # single-view consensus still counts
    strict = cross_cycle_target(view_a, view_b, valid, valid_b, bounds, samples_per_cycle=S, config=_cfg(neighbors=None, confidence={"min_candidates": 4}))
    assert strict.confidence.max() == 0.0  # only 3 other cycles
    with pytest.raises(ValueError):
        cross_cycle_target(view_a, view_b, valid, valid, [(0, S + 1)], samples_per_cycle=S, config=_cfg())
    single = cross_cycle_target(view_a[:S], view_b[:S], valid[:S], valid[:S], [(0, S)], samples_per_cycle=S, config=_cfg())
    assert single.confidence.max() == 0.0
    trimmed = cross_cycle_target(view_a, view_b, valid, valid, bounds, samples_per_cycle=S, config=_cfg(neighbors=None, aggregation="trimmed_mean"))
    assert np.isfinite(trimmed.target).all()
    # Disagreeing views inside a cycle remove that cycle's candidate.
    view_b[:, 3] = view_a[:, 3] + 5.0
    gone = cross_cycle_target(view_a, view_b, valid, valid, bounds, samples_per_cycle=S, config=_cfg(neighbors=None))
    assert gone.candidate_count[:, 3].max() == 0 and gone.confidence[:, 3].max() == 0.0
    with pytest.raises(ValueError):
        ConfidenceConfig(compatibility=True)
    legacy = CycleTargetConfig.from_mapping({"tau": 0.1, "min_candidates": 3})
    assert legacy.confidence.tau == 0.1 and legacy.confidence.min_candidates == 3


def test_window_dataset_ships_cycle_target(skeleton):
    sample = make_sample(skeleton, frames=64, period=16, mids=True)
    dataset = CycleWindowDataset([sample], skeleton=skeleton, window=WindowConfig(num_cycles=2, samples_per_cycle=S), split="train", cycle_target=_cfg(neighbors=2))
    item = dataset[0]
    assert item["cycle_target"].shape == (16, skeleton.num_joints, 3) and item["cycle_confidence"].shape == (16, skeleton.num_joints)
    assert (item["cycle_confidence"] > 0).any() and item["cycle_dispersion"].shape == (16, skeleton.num_joints)
    assert torch.isfinite(item["cycle_dispersion"][item["cycle_confidence"] > 0]).all()
    disabled = CycleWindowDataset([sample], skeleton=skeleton, window=WindowConfig(num_cycles=2, samples_per_cycle=S), split="train")
    assert disabled[0]["cycle_confidence"].max() == 0.0


def test_cycle_loss_weights_by_confidence():
    pred = torch.zeros(1, 4, 2, 3)
    target = torch.ones(1, 4, 2, 3)  # distance sqrt(3)
    valid = torch.ones(1, 4, 2, dtype=torch.bool)
    conf = torch.zeros(1, 4, 2)
    zero_zone = torch.zeros(1, 4, 2)
    assert cycle_loss(pred, valid, target, conf, zero_zone, kind="l2", beta=0.05).item() == 0.0
    conf[0, :, 0] = 1.0
    assert cycle_loss(pred, valid, target, conf, zero_zone, kind="l2", beta=0.05).item() == pytest.approx(3.0)
    conf[0, :, 1] = 0.5
    pred[0, :, 1] = 1.0  # joint 1 exact
    assert cycle_loss(pred, valid, target, conf, zero_zone, kind="l2", beta=0.05).item() == pytest.approx(3.0 * 1.0 / 1.5)


def test_dead_zone_only_penalises_excess_deviation():
    pred = torch.zeros(1, 1, 3, 3)
    target = torch.zeros(1, 1, 3, 3)
    target[0, 0, 0, 0] = 0.02  # d = 0.02 < delta
    target[0, 0, 1, 0] = 0.10  # d = 0.10 > delta
    valid = torch.ones(1, 1, 3, dtype=torch.bool)
    conf = torch.ones(1, 1, 3)
    dispersion = torch.tensor([[[0.05, 0.05, float("inf")]]])
    delta = natural_variation_dead_zone(dispersion, DeadZoneConfig(scale=1.0, minimum=0.0))
    torch.testing.assert_close(delta, torch.tensor([[[0.05, 0.05, 0.0]]]))  # undefined dispersion -> minimum
    loss = cycle_loss(pred, valid, target, conf, delta, kind="l1", beta=0.05)
    assert loss.item() == pytest.approx((0.0 + 0.05 + 0.0) / 3.0)  # only the excess 0.10 - 0.05
    no_zone = cycle_loss(pred, valid, target, conf, torch.zeros_like(delta), kind="l1", beta=0.05)
    assert no_zone.item() == pytest.approx((0.02 + 0.10) / 3.0)
    clipped = natural_variation_dead_zone(dispersion, DeadZoneConfig(scale=2.0, minimum=0.01, maximum=0.08))
    torch.testing.assert_close(clipped, torch.tensor([[[0.08, 0.08, 0.01]]]))
    assert natural_variation_dead_zone(dispersion, DeadZoneConfig(enabled=False)).max() == 0.0
    with pytest.raises(ValueError):
        DeadZoneConfig(minimum=0.1, maximum=0.05)


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


def test_phase_pair_masks_and_contrastive_periodicity():
    T = 2 * S
    phase = (torch.arange(T) % S / S)[None].float()
    phase_valid = torch.ones(1, T, dtype=torch.bool)
    cycle_index = (torch.arange(T) // S)[None]
    positive, negative = phase_pair_masks(phase, phase_valid, cycle_index, 0.25)
    assert positive[0, 0].nonzero().flatten().tolist() == [S]  # same phase in the next cycle
    assert not positive[0, S].any()  # last cycle has no next cycle
    # margin 0.25 with S = 8: phases differing by 3/8 or 4/8 are negatives (2/8 is not).
    assert negative[0, 0].nonzero().flatten().tolist() == [S + 3, S + 4, S + 5]
    feature = torch.randn(1, S, 3, 6)
    valid = torch.ones(1, T, 3, dtype=torch.bool)
    periodic = torch.cat((feature, feature), dim=1)
    loss = contrastive_periodicity_loss(periodic, valid, phase, phase_valid, cycle_index, temperature=0.1, negative_phase_margin=0.25)
    assert torch.isfinite(loss) and loss.item() < 0.5
    constant = torch.ones(1, T, 3, 6)
    collapsed = contrastive_periodicity_loss(constant, valid, phase, phase_valid, cycle_index, temperature=0.1, negative_phase_margin=0.25)
    assert collapsed.item() == pytest.approx(np.log(1 + 3), abs=1e-4)  # log(1 + |negatives|): the trivial solution is not optimal
    assert loss.item() < collapsed.item()
    leaf = periodic.clone().requires_grad_(True)
    contrastive_periodicity_loss(leaf, valid, phase, phase_valid, cycle_index, temperature=0.1, negative_phase_margin=0.25).backward()
    assert torch.isfinite(leaf.grad).all()
    wide = contrastive_periodicity_loss(periodic, valid, phase, phase_valid, cycle_index, temperature=0.1, negative_phase_margin=0.5)
    assert wide.item() == 0.0  # no phase is farther than 0.5 -> no negatives -> skipped
    invalid = torch.zeros(1, T, 3, dtype=torch.bool)
    assert contrastive_periodicity_loss(periodic, invalid, phase, phase_valid, cycle_index, temperature=0.1, negative_phase_margin=0.25).item() == 0.0


def test_diagnostics_statistics():
    torch.manual_seed(0)
    B, T, J, D = 2, 2 * S, 4, 6
    feature = torch.randn(B, T, J, D)
    valid = torch.ones(B, T, J, dtype=torch.bool)
    var = feature_variance(feature, valid)
    assert set(var) == {"var_total", "var_time", "var_joint", "var_batch", "var_channel"}
    assert var["var_total"].item() == pytest.approx(1.0, abs=0.2) and var["var_time"].item() > 0.5 and var["var_joint"].item() > 0.5
    constant = torch.ones(B, T, J, D)
    var_c = feature_variance(constant, valid)
    assert var_c["var_time"].item() == 0.0 and var_c["var_joint"].item() == 0.0 and var_c["var_total"].item() == 0.0
    phase = (torch.arange(T) % S / S)[None].float().repeat(B, 1)
    phase_valid = torch.ones(B, T, dtype=torch.bool)
    cycle_index = (torch.arange(T) // S)[None].repeat(B, 1)
    periodic = torch.cat((feature[:, :S], feature[:, :S]), dim=1)
    sims = phase_similarity(periodic, valid, phase, phase_valid, cycle_index, negative_phase_margin=0.25, generator=torch.Generator().manual_seed(0))
    assert sims["sim_same_phase"].item() == pytest.approx(1.0, abs=1e-5) and sims["sim_diff_phase"].item() < 0.7
    assert sims["collapse_gap"].item() > 0.3 and abs(sims["sim_random_joint"].item()) < 0.7
    sims_c = phase_similarity(constant, valid, phase, phase_valid, cycle_index, negative_phase_margin=0.25)
    assert sims_c["sim_same_phase"].item() == pytest.approx(1.0) and sims_c["sim_diff_phase"].item() == pytest.approx(1.0) and sims_c["collapse_gap"].item() == pytest.approx(0.0)
    from fusion.modules.film import FiLMMotionGuidance

    film = FiLMMotionGuidance(D)
    stats = film_statistics(film, feature, valid)
    assert stats["gamma_abs_mean"].item() == 0.0 and stats["beta_abs_max"].item() == 0.0  # zero init
    with torch.no_grad():
        film.gamma.weight.fill_(0.5)
    assert film_statistics(film, feature, valid)["gamma_abs_mean"].item() > 0.0
    delta = torch.zeros(B, T, J, 3)
    delta[0, 0, 0] = torch.tensor([0.25, 0.1, 0.0])
    res = residual_statistics(delta, valid, 0.25, saturation_ratio=0.95)
    assert res["delta_abs_max"].item() == pytest.approx(0.25) and res["delta_saturation"].item() == pytest.approx(1 / (B * T * J * 3))
    half = torch.full((B, T, J, 1), 0.5)
    rel = reliability_statistics(half, half, valid, threshold=0.9)
    assert rel["entropy"].item() == pytest.approx(np.log(2), abs=1e-6) and rel["frac_w_a_gt"].item() == 0.0
    one = torch.ones(B, T, J, 1)
    rel1 = reliability_statistics(one, torch.zeros_like(one), valid, threshold=0.9)
    assert rel1["entropy"].item() == pytest.approx(0.0, abs=1e-6) and rel1["frac_w_a_gt"].item() == 1.0 and rel1["w_b_mean"].item() == 0.0


def test_compute_losses_v2_end_to_end(tiny_config, tiny_batch, skeleton):
    torch.manual_seed(0)
    model = CycleAwareFusionModel(tiny_config)
    batch = dict(tiny_batch)
    B, T, J = batch["pose_a"].shape[:3]
    batch["cycle_index"] = (torch.arange(T) // S)[None].repeat(B, 1)
    batch["half_index"] = ((torch.arange(T) % S) >= S // 2).long()[None].repeat(B, 1)
    batch["cycle_target"] = batch["pose_a"].clone() + 0.3
    batch["cycle_confidence"] = torch.ones(B, T, J)
    batch["cycle_dispersion"] = torch.full((B, T, J), 0.01)
    batch["corruption_mask_a"] = torch.zeros(B, T, J, dtype=torch.bool)
    batch["corruption_mask_b"] = torch.zeros(B, T, J, dtype=torch.bool)
    batch["corruption_mask_a"][:, :, 2] = True
    output = model(**tiny_batch)
    losses = compute_losses(output, batch, skeleton=skeleton, config=LossConfig(), samples_per_cycle=S)
    logged = losses.as_dict()
    for name, value in logged.items():
        assert value.ndim == 0 and torch.isfinite(value), name
    assert losses.cycle.item() > 0 and losses.reliability.item() > 0 and losses.periodicity.item() > 0 and losses.symmetry.item() > 0
    assert logged["reliability_weighted"].item() == pytest.approx(0.02 * losses.reliability.item())
    assert losses.recovery.item() == 0.0 and LossConfig().weights["recovery"] == 0.0 and LossConfig().reliability.weight == 0.02
    losses.total.backward()
    assert all(p.grad is not None for p in model.reliability.parameters())
    # Contrastive periodicity through the config path.
    contrastive = compute_losses(model(**tiny_batch), batch, skeleton=skeleton, config=LossConfig.from_mapping({"periodicity": {"type": "contrastive"}}), samples_per_cycle=S)
    assert torch.isfinite(contrastive.periodicity) and contrastive.periodicity.item() > 0
    # External reference takes priority over the cross-cycle target.
    batch["reference"] = batch["pose_a"].clone()
    batch["reference_valid"] = torch.ones(B, T, J, dtype=torch.bool)
    with_reference = compute_losses(model(**tiny_batch), batch, skeleton=skeleton, config=LossConfig(), samples_per_cycle=S)
    assert with_reference.cycle.item() < losses.cycle.item()
    with pytest.raises(ValueError):
        LossConfig.from_mapping({"residual": {"norm": "linf"}})


def test_hydra_v2_default_and_v1_preset(tmp_path: Path):
    # The v2 objectives (on the v1.0 base) are reproduced through experiment=v2.
    cfg = compose_config(["experiment=[smoke,v2]", f"output_root={tmp_path}", "run_name=v2", "diagnostics.gradient_norm.enabled=true", "diagnostics.gradient_norm.interval=1"])
    assert cfg.loss.cycle.weight == 1.0 and cfg.loss.reliability.weight == 0.02 and cfg.loss.recovery.weight == 0.0 and cfg.data.cycle_target.enabled and cfg.loss.periodicity.type == "contrastive"
    assert cfg.model.fusion.depth_alpha == 0.0 and cfg.data.cycle_target.consensus.method == "mean"
    result = run(cfg)
    metrics = result["test_metrics"]
    assert "test/cycle_raw" in metrics and "test/cycle_weighted" in metrics and "test/total" in metrics and "test/cycle_target_error" in metrics
    for name in ("motion_var_time", "motion_var_joint", "motion_sim_same_phase", "motion_sim_diff_phase", "motion_sim_random_joint", "film_gamma_abs_mean_a", "residual_delta_saturation", "reliability_entropy", "reliability_w_a_mean"):
        assert f"test/diag/{name}" in metrics, name
    assert any(k.startswith("train/reliability_raw") for k in result["fit_metrics"])
    assert any(k.startswith("train/diag/grad_norm/reliability_head") for k in result["fit_metrics"])
    v1 = compose_config(["experiment=v1", f"output_root={tmp_path}", "run_name=v1", "trainer=debug", "samples_per_cycle=8", "model.hidden_dim=16", "model.num_heads=2", "data.options.subjects=4", "data.options.frames=48", "data.options.period=12"])
    assert v1.loss.recovery.weight == 1.0 and v1.loss.cycle.weight == 0.0 and v1.loss.residual.norm == "l2"
    run(v1)
    contrastive = compose_config(["experiment=smoke", "loss.periodicity.type=cosine", f"output_root={tmp_path}", "run_name=v2c"])
    assert run(contrastive)["test_metrics"]["test/periodicity_raw"] >= 0
