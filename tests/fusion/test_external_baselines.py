"""External learned baselines (fusion.external) on the shared contract."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from fusion.external import EXTERNAL_BACKBONES, ExternalFusionModel, ExternalModelConfig
from fusion.lightning_module import CycleAwareFusionModule, build_fusion_model
from fusion.model import CycleAwareFusionModel
from fusion.modules import depth_aware_pose_fusion
from fusion.train import compose_config, run

E_X = torch.tensor([1.0, 0.0, 0.0])
E_Z = torch.tensor([0.0, 0.0, 1.0])


def _depth(batch):
    B, T = batch["pose_a"].shape[:2]
    return E_Z.expand(B, T, 3).clone(), E_X.expand(B, T, 3).clone()


def _config(backbone: str, **overrides) -> ExternalModelConfig:
    base = {"backbone": backbone, "hidden_dim": 16, "samples_per_cycle": 8, "smoothnet_window": 4, "smoothnet_res_hidden": 8, "smoothnet_blocks": 1, "mlp_layers": 2, "tcn_dilations": (1, 2)}
    return ExternalModelConfig.from_mapping({**base, **overrides})


@pytest.mark.parametrize("backbone", EXTERNAL_BACKBONES)
def test_external_forward_contract_and_initial_identity(backbone, tiny_batch):
    torch.manual_seed(0)
    model = ExternalFusionModel(_config(backbone))
    depth_a, depth_b = _depth(tiny_batch)
    out = model(**tiny_batch, depth_a=depth_a, depth_b=depth_b)
    B, T, J = tiny_batch["pose_a"].shape[:3]
    assert out.pose.shape == (B, T, J, 3) and out.valid.shape == (B, T, J) and out.reliability_logits.shape == (B, T, J, 2)
    torch.testing.assert_close(out.pose, out.base_pose + out.delta_pose)
    torch.testing.assert_close(out.weight_a + out.weight_b, torch.ones(B, T, J, 1))
    assert torch.isfinite(out.pose).all() and not out.valid[1, 12:].any()
    # Zero-initialised heads: every baseline starts exactly at the base rule with equal weights.
    half = torch.full((B, T, J, 1), 0.5)
    pose_a = torch.where(tiny_batch["valid_a"][..., None], tiny_batch["pose_a"], 0.0)
    pose_b = torch.where(tiny_batch["valid_b"][..., None], tiny_batch["pose_b"], 0.0)
    frame = tiny_batch["frame_mask"][..., None]
    alpha = model.config.fusion.depth_alpha
    rule, _ = depth_aware_pose_fusion(pose_a, pose_b, half, half, tiny_batch["valid_a"] & frame, tiny_batch["valid_b"] & frame, depth_a, depth_b, alpha=alpha)
    torch.testing.assert_close(out.pose, torch.where(out.valid[..., None], rule, torch.zeros_like(rule)), atol=1e-6, rtol=1e-5)
    # Gradients reach the backbone.
    (out.pose.square().mean() + out.reliability_logits.square().mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())


def test_muc_weights_are_masked_and_learned_per_view(tiny_batch):
    torch.manual_seed(1)
    model = ExternalFusionModel(_config("muc_weights", max_delta=None, fusion={"depth_alpha": 0.0}))
    with torch.no_grad():
        model.backbone.head.weight.normal_(std=0.5)
    out = model(**tiny_batch)
    assert torch.equal(out.delta_pose, torch.zeros_like(out.delta_pose))
    torch.testing.assert_close(out.weight_b[0, :, 3], torch.ones(tiny_batch["pose_a"].shape[1], 1))  # valid in B only
    assert not torch.allclose(out.weight_a[out.valid & tiny_batch["valid_a"] & tiny_batch["valid_b"]], torch.full((1,), 0.5))
    assert model.config.fusion.depth_alpha == 0.0 and model.residual.max_delta is None


def test_external_config_validation():
    with pytest.raises(ValueError):
        ExternalModelConfig(backbone="lstm")
    with pytest.raises(ValueError):
        ExternalModelConfig.from_mapping({"backbone": "tcn", "unknown": 1})
    cfg = ExternalModelConfig.from_mapping({"backbone": "tcn", "fusion": {"depth_alpha": 0.0}})
    assert cfg.fusion.depth_alpha == 0.0 and cfg.architecture_version == "external/tcn"


def test_build_fusion_model_dispatch_and_checkpoint_round_trip(tiny_batch, tmp_path):
    arch, model = build_fusion_model({"hidden_dim": 16, "num_heads": 2, "samples_per_cycle": 8})
    assert arch == "cycle_aware" and isinstance(model, CycleAwareFusionModel)
    arch, model = build_fusion_model({"architecture": "external", "backbone": "tcn", "hidden_dim": 16, "tcn_dilations": [1]})
    assert arch == "external" and isinstance(model, ExternalFusionModel)
    with pytest.raises(ValueError):
        build_fusion_model({"architecture": "gan"})
    module = CycleAwareFusionModule(model_config={"architecture": "external", "backbone": "metapose_mlp", "hidden_dim": 16, "mlp_layers": 1, "samples_per_cycle": 8})
    assert module.hparams["model_config"]["architecture"] == "external"
    assert set(module.gradient_module_map()) == {"backbone"}
    with torch.no_grad():
        module.model.backbone.head.weight.normal_(std=0.1)
    path = tmp_path / "ext.ckpt"
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams), "pytorch-lightning_version": "2.0"}, path)
    reloaded = CycleAwareFusionModule.load_from_checkpoint(str(path), map_location="cpu")
    assert isinstance(reloaded.model, ExternalFusionModel) and reloaded.model.config.backbone == "metapose_mlp"
    with torch.no_grad():
        torch.testing.assert_close(reloaded(tiny_batch).pose, module(tiny_batch).pose)


@pytest.mark.parametrize("backbone", ("tcn", "smoothnet"))
def test_external_hydra_smoke_run(backbone, tmp_path):
    small = {"tcn": ["model.tcn_dilations=[1,2]"], "smoothnet": ["model.smoothnet_window=4", "model.smoothnet_res_hidden=8", "model.smoothnet_blocks=1"]}[backbone]
    # experiment=smoke sets cycle-aware-only model keys, so the small synthetic setup is spelled out.
    tiny = ["trainer=debug", "samples_per_cycle=8", "num_cycles=2", "data.batch_size=4", "data.options.subjects=4", "data.options.frames=48", "data.options.period=12", "optimizer.warmup_steps=2"]
    cfg = compose_config([f"model=external_{backbone}", "model.hidden_dim=16", *small, *tiny, "loss.reliability.weight=0", f"output_root={tmp_path}", f"run_name=ext_{backbone}"])
    assert cfg.model.architecture == "external" and cfg.data.cycle_target.consensus.depth_alpha == 0.8
    metrics = run(cfg)["test_metrics"]
    assert "test/pa_mpjpe" in metrics and "test/pa_mpjpe_rule" in metrics and metrics["test/reliability_raw"] == 0.0
