"""Hydra entry point for cycle-aware fusion training and evaluation.

Usage (repository root):

    conda run -n gymnastic python -m gymnastics.fusion.cycle_aware.train data=synthetic
    conda run -n gymnastic gymnastics fuse cycle-aware data=gymnastics trainer.max_epochs=50
    conda run -n gymnastic gymnastics fuse cycle-aware data=freeman experiment=no_film

Configuration is composed from ``configs/cycle_aware`` (``config.yaml`` and
its groups ``model``, ``data``, ``loss``, ``corruption``, ``trainer`` and
``experiment``); every argument is a Hydra override.  Outputs (checkpoints,
CSV logs, the resolved configuration) are written below
``<output_root>/<run_name>``.

Programmatic use:
    ``compose_config(overrides)`` returns the resolved ``DictConfig``;
    ``run(cfg)`` executes fit (and test) and returns the trainer's logged
    metrics, which is what the integration tests call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from gymnastics.common.paths import CONFIG_ROOT, PROJECT_ROOT

from .data import build_datamodule
from .lightning_module import CycleAwareFusionModule
from .model import CycleAwareModelConfig

CONFIG_DIR = CONFIG_ROOT / "cycle_aware"


def compose_config(overrides: Sequence[str] = (), *, config_name: str = "config") -> DictConfig:
    """Compose the Hydra configuration with dot-list overrides.

    Args:
        overrides: Hydra override strings such as ``"data=freeman"`` or
            ``"model.film.enabled=false"``.
        config_name: Root configuration file name.

    Returns:
        The composed configuration (not yet resolved).
    """
    unsupported = [value for value in overrides if value.startswith("-")]
    if unsupported:
        raise ValueError(f"overrides use key=value syntax; unsupported: {unsupported}")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def build_module(cfg: DictConfig) -> CycleAwareFusionModule:
    """Instantiate the LightningModule from the ``model``, ``loss`` and ``optimizer`` nodes."""
    model = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model, dict)
    model.pop("name", None)
    return CycleAwareFusionModule(
        CycleAwareModelConfig.from_mapping(model),
        OmegaConf.to_container(cfg.loss, resolve=True),
        OmegaConf.to_container(cfg.optimizer, resolve=True),
    )


def build_trainer(cfg: DictConfig, run_dir: Path) -> pl.Trainer:
    """Instantiate the Lightning trainer with CSV logging and checkpointing."""
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_cfg, dict)
    callbacks: list[pl.Callback] = []
    if not trainer_cfg.get("fast_dev_run", False):
        callbacks.append(ModelCheckpoint(dirpath=str(run_dir / "checkpoints"), filename="{epoch:03d}-{val/total:.4f}", monitor="val/total", mode="min", save_last=True, save_top_k=1))
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    logger = CSVLogger(save_dir=str(run_dir), name="logs")
    return pl.Trainer(
        default_root_dir=str(run_dir),
        max_epochs=int(trainer_cfg.get("max_epochs", 1)),
        accelerator=str(trainer_cfg.get("accelerator", "auto")),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", 32),
        fast_dev_run=trainer_cfg.get("fast_dev_run", False),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 10)),
        limit_train_batches=trainer_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
        limit_test_batches=trainer_cfg.get("limit_test_batches", 1.0),
        enable_progress_bar=bool(trainer_cfg.get("enable_progress_bar", True)),
        deterministic=bool(trainer_cfg.get("deterministic", False)),
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=int(trainer_cfg.get("num_sanity_val_steps", 1)),
    )


def run(cfg: DictConfig) -> dict[str, Any]:
    """Fit (and optionally test) according to ``cfg``.

    Returns:
        Dictionary with the run directory, the data summary, the final
        training metrics and the test metrics (if run).
    """
    pl.seed_everything(int(cfg.seed), workers=True)
    run_dir = Path(str(cfg.output_root))
    run_dir = (run_dir if run_dir.is_absolute() else PROJECT_ROOT / run_dir) / str(cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    datamodule = build_datamodule(OmegaConf.to_container(cfg.data, resolve=True))  # type: ignore[arg-type]
    module = build_module(cfg)
    trainer = build_trainer(cfg, run_dir)
    trainer.fit(module, datamodule=datamodule)
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "data": datamodule.summary(),
        "fit_metrics": {k: float(v) for k, v in trainer.logged_metrics.items()},
    }
    if bool(cfg.get("test_after_fit", True)) and not bool(cfg.trainer.get("fast_dev_run", False)):
        test_metrics = trainer.test(module, datamodule=datamodule, verbose=False)
        result["test_metrics"] = {k: float(v) for row in test_metrics for k, v in row.items()}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry: every argument is a Hydra override."""
    import sys

    overrides = list(argv) if argv is not None else sys.argv[1:]
    cfg = compose_config(overrides)
    if bool(cfg.get("print_config", False)):
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return 0
    result = run(cfg)
    print(json.dumps({"run_dir": result["run_dir"], "data": result["data"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
