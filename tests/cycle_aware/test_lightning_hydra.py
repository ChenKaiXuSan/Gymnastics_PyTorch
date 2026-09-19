"""Integration tests: Hydra composition, Lightning fast_dev_run and a short fit."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from gymnastics.fusion.cycle_aware.data.synthetic import SyntheticDataModule
from gymnastics.fusion.cycle_aware.lightning_module import CycleAwareFusionModule, OptimizerConfig
from gymnastics.fusion.cycle_aware.train import build_module, compose_config, run

TINY_MODEL = {"hidden_dim": 16, "num_heads": 2, "samples_per_cycle": 8, "spatial": {"layers": 1}, "short_motion": {"layers": 1}, "long_motion": {"layers": 1}}
TINY_DATA = {"name": "synthetic", "batch_size": 4, "window": {"num_cycles": 2, "samples_per_cycle": 8}, "options": {"subjects": 4, "sequences_per_subject": 1, "frames": 48, "period": 12}}


def test_lightning_module_steps_run_and_log():
    torch.manual_seed(0)
    module = CycleAwareFusionModule(TINY_MODEL, {"periodicity_weight": 0.1}, {"warmup_steps": 1})
    datamodule = SyntheticDataModule(TINY_DATA)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    val_batch = next(iter(datamodule.val_dataloader()))
    module.validation_step(val_batch, 0)
    prediction = module.predict_step(val_batch, 0)
    assert prediction["pose"].shape == val_batch["pose_a"].shape
    assert OptimizerConfig.from_mapping({"betas": [0.9, 0.99]}).betas == (0.9, 0.99)


def test_fast_dev_run_with_trainer(tmp_path: Path):
    module = CycleAwareFusionModule(TINY_MODEL)
    datamodule = SyntheticDataModule(TINY_DATA)
    trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False, default_root_dir=str(tmp_path))
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule, verbose=False)
    metrics = trainer.logged_metrics
    assert "test/total" in metrics and "test/pa_mpjpe" in metrics


def test_hydra_compose_and_ablation_switching():
    cfg = compose_config(["experiment=smoke"])
    assert cfg.data.name == "synthetic" and cfg.model.samples_per_cycle == 8 and cfg.trainer.max_epochs == 2
    assert cfg.data.window.samples_per_cycle == cfg.model.samples_per_cycle
    module = build_module(cfg)
    assert module.model.config.hidden_dim == 16 and module.model.short_motion is not None
    for experiment, attribute in (("no_film", "film"), ("no_cross_view", "cross_view"), ("no_short_motion", "short_motion"), ("no_long_motion", "long_motion")):
        cfg = compose_config([f"experiment={experiment}", "samples_per_cycle=8", "model.hidden_dim=16", "model.num_heads=2"])
        assert getattr(build_module(cfg).model, attribute) is None, experiment
        assert experiment in cfg.run_name
    cfg = compose_config(["data=freeman", "data.options.subjects=[1,2]", "corruption=none", "model.residual.max_delta=null"])
    assert cfg.data.name == "freeman" and cfg.data.options.subjects == [1, 2]
    assert not cfg.data.corruption.enabled and cfg.model.residual.max_delta is None
    with pytest.raises(Exception):
        compose_config(["model.hidden_dims=4"])


def test_full_hydra_smoke_run(tmp_path: Path):
    cfg = compose_config(["experiment=smoke", f"output_root={tmp_path}", "seed=1"])
    result = run(cfg)
    run_dir = Path(result["run_dir"])
    assert (run_dir / "config.yaml").is_file() and (run_dir / "result.json").is_file()
    assert result["data"]["subjects"] == {"train": 2, "val": 1, "test": 1}
    assert any(key.startswith("train/total") for key in result["fit_metrics"])
    assert "test/total" in result["test_metrics"] and "test/pa_mpjpe" in result["test_metrics"]
    assert (run_dir / "checkpoints" / "last.ckpt").is_file()
    restored = CycleAwareFusionModule.load_from_checkpoint(run_dir / "checkpoints" / "last.ckpt", map_location="cpu")
    assert restored.model.config.hidden_dim == 16
