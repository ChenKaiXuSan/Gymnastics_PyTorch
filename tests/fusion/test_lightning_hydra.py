"""Integration tests: Hydra composition, Lightning fast_dev_run and a short fit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from fusion.data.synthetic import SyntheticDataModule
from fusion.lightning_module import CycleAwareFusionModule, OptimizerConfig
from fusion.train import build_module, compose_config, run

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
    assert "test/total" in metrics and "test/pa_mpjpe" in metrics and "test/ta_mpjpe" in metrics  # synthetic: canonical reference


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
    assert any(path.name.startswith("epoch") and path.suffix == ".ckpt" for path in (run_dir / "checkpoints").iterdir())
    restored = CycleAwareFusionModule.load_from_checkpoint(run_dir / "checkpoints" / "last.ckpt", map_location="cpu")
    assert restored.model.config.hidden_dim == 16


def test_fold_files_and_sweep(tmp_path: Path):
    from fusion.data.folds import make_subject_folds, read_fold_file, write_fold_files
    from fusion.train import run_folds

    subjects = [f"subject_{i:02d}" for i in range(6)]
    folds = make_subject_folds(subjects, k=3, seed=0, stratify=lambda s: "a" if int(s[-2:]) % 2 else "b")
    assert len(folds) == 3
    for fold in folds:
        assert not (set(fold["train"]) & set(fold["test"])) and not (set(fold["val"]) & set(fold["test"]))
        assert set(fold["train"]) | set(fold["val"]) | set(fold["test"]) == set(subjects)
    assert sorted(folds[0]["test"] + folds[1]["test"] + folds[2]["test"]) == subjects  # every subject tested once
    assert all(fold["train"] for fold in folds)
    folds_dir = tmp_path / "folds"
    paths = write_fold_files(folds, folds_dir, dataset="synthetic")
    assert read_fold_file(paths[0])["test"] == folds[0]["test"]

    cfg = compose_config(["experiment=smoke", f"output_root={tmp_path}", "run_name=sweep", f"folds_dir={folds_dir}", "data.options.subjects=6", "fold_ids=[1,3]"])
    sweep = run_folds(cfg)
    assert sweep["folds"] == ["fold_01", "fold_03"]
    assert "test/pa_mpjpe" in sweep["summary"] and sweep["summary"]["test/pa_mpjpe"]["folds"] == 2
    assert (tmp_path / "sweep" / "summary.csv").is_file() and (tmp_path / "sweep" / "fold_01" / "result.json").is_file()
    fold_result = json.loads((tmp_path / "sweep" / "fold_01" / "result.json").read_text())
    assert fold_result["data"]["subjects"] == {"train": len(folds[0]["train"]), "val": len(folds[0]["val"]), "test": len(folds[0]["test"])}


def test_summarize_sweep_collects_fold_results(tmp_path: Path):
    from fusion.summarize import summarize_sweep

    for fold, value in (("fold_01", 0.1), ("fold_02", 0.3)):
        (tmp_path / fold).mkdir()
        (tmp_path / fold / "result.json").write_text(json.dumps({"test_metrics": {"test/pa_mpjpe": value, "test/total": 1.0}}))
    (tmp_path / "fold_03").mkdir()  # job not finished
    payload = summarize_sweep(tmp_path)
    assert payload["missing"] == ["fold_03"] and list(payload["folds"]) == ["fold_01", "fold_02"]
    assert payload["summary"]["test/pa_mpjpe"]["mean"] == pytest.approx(0.2)
    assert (tmp_path / "summary.csv").read_text().splitlines()[0] == "fold,test/pa_mpjpe,test/total"


def test_weighted_folds_balance_volume():
    from fusion.data.folds import make_subject_folds

    subjects = [f"{i:02d}" for i in range(1, 21)]
    weights = {s: float(i * i) for i, s in enumerate(subjects, start=1)}  # 1 .. 400
    folds = make_subject_folds(subjects, k=5, seed=0, weights=weights)
    volumes = [sum(weights[s] for s in fold["test"]) for fold in folds]
    assert max(volumes) - min(volumes) < 0.15 * max(volumes)
    assert sorted(s for fold in folds for s in fold["test"]) == subjects
