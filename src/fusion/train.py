"""Hydra entry point for cycle-aware fusion training and evaluation.

Usage (repository root):

    python -m fusion.train data=synthetic
    python -m fusion train data=gymnastics trainer.max_epochs=50
    python -m fusion train data=freeman experiment=no_film

Cross-validation (one run per fold file, then a summary):

    python -m fusion train data=gymnastics \
        folds_dir=src/configs/fusion/folds/gymnastics run_name=gym_v1_5fold

Transfer / evaluation of an existing checkpoint (no training):

    python -m fusion train data=gymnastics test_only=true \
        checkpoint=local/runs/cycle_aware/freeman_all40_v1_reference_supervised_5fold_seed0/fold_01/checkpoints/last.ckpt

``checkpoint`` without ``test_only`` initialises the model from those weights
and then trains (fine-tuning).

Configuration is composed from ``src/configs/fusion`` (``config.yaml`` and
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
from typing import Any, Mapping, Sequence

import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from common.paths import CONFIG_ROOT, PROJECT_ROOT

from .data import build_datamodule
from .lightning_module import CycleAwareFusionModule

CONFIG_DIR = CONFIG_ROOT / "fusion"


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
    """Instantiate the LightningModule from the ``model``, ``loss``, ``optimizer`` and ``evaluation`` nodes.

    With ``cfg.checkpoint`` set, the model weights are loaded from that
    checkpoint (architecture taken from the checkpoint's hyper-parameters) and
    the loss / optimizer / evaluation settings come from ``cfg``.
    """
    loss = OmegaConf.to_container(cfg.loss, resolve=True)
    optimizer = OmegaConf.to_container(cfg.optimizer, resolve=True)
    evaluation = OmegaConf.to_container(cfg.get("evaluation") or {}, resolve=True)
    diagnostics = OmegaConf.to_container(cfg.get("diagnostics") or {}, resolve=True)
    checkpoint = cfg.get("checkpoint")
    if checkpoint:
        path = Path(str(checkpoint))
        path = path if path.is_absolute() else PROJECT_ROOT / path
        return CycleAwareFusionModule.load_from_checkpoint(str(path), map_location="cpu", loss_config=loss, optimizer_config=optimizer, evaluation_config=evaluation, diagnostics_config=diagnostics)
    model = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model, dict)
    model.pop("name", None)
    # The mapping may carry ``architecture`` (cycle_aware | external); the
    # Lightning module dispatches on it and validates the remaining fields.
    return CycleAwareFusionModule(model, loss, optimizer, evaluation, diagnostics)


def build_trainer(cfg: DictConfig, run_dir: Path) -> pl.Trainer:
    """Instantiate the Lightning trainer with CSV logging and checkpointing."""
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_cfg, dict)
    callbacks: list[pl.Callback] = []
    if not trainer_cfg.get("fast_dev_run", False):
        # The metric name contains a slash; without auto_insert_metric_name=False
        # Lightning would turn it into a sub-directory.
        callbacks.append(ModelCheckpoint(dirpath=str(run_dir / "checkpoints"), filename="epoch{epoch:03d}-val_total{val/total:.4f}", auto_insert_metric_name=False, monitor="val/total", mode="min", save_last=True, save_top_k=1))
        # ``final.ckpt`` holds the weights this run REPORTS: ``trainer.test`` runs on
        # the module in memory, i.e. the last epoch, and ``last.ckpt`` is not those
        # weights -- Lightning only refreshes it when a monitored save happens
        # (ModelCheckpoint.on_validation_end), so with save_top_k=1 it is a copy of
        # the best-val checkpoint. That is a different model whenever validation
        # stops improving early, which is what reference-supervised runs do
        # (their validation replays the corruption, so ``val/total`` selects for
        # robustness, not accuracy). Without this callback a checkpoint cannot
        # reproduce the run's own test numbers.
        callbacks.append(ModelCheckpoint(dirpath=str(run_dir / "checkpoints"), filename="final", auto_insert_metric_name=False, monitor=None, every_n_epochs=1, save_top_k=1, enable_version_counter=False))
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    logger = CSVLogger(save_dir=str(run_dir), name="logs")
    return pl.Trainer(
        default_root_dir=str(run_dir),
        max_epochs=int(trainer_cfg.get("max_epochs", 1)),
        max_steps=int(trainer_cfg.get("max_steps", -1) or -1),
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
    import torch

    num_threads = cfg.trainer.get("num_threads")
    if num_threads:
        # On large shared CPU boxes the default (one thread per core) plus
        # DataLoader workers oversubscribes the machine; cap it explicitly.
        torch.set_num_threads(int(num_threads))
    matmul_precision = cfg.trainer.get("matmul_precision")
    if matmul_precision and torch.cuda.is_available():
        # "high" lets fp32 matmuls use TF32 tensor cores (H100/A100); the
        # model's fp32 paths (Procrustes metrics, losses) keep fp32 accumulation.
        torch.set_float32_matmul_precision(str(matmul_precision))
    run_dir = Path(str(cfg.output_root))
    run_dir = (run_dir if run_dir.is_absolute() else PROJECT_ROOT / run_dir) / str(cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    datamodule = build_datamodule(OmegaConf.to_container(cfg.data, resolve=True))  # type: ignore[arg-type]
    module = build_module(cfg)
    trainer = build_trainer(cfg, run_dir)
    test_only = bool(cfg.get("test_only", False))
    if test_only and not cfg.get("checkpoint"):
        raise ValueError("test_only=true requires checkpoint=<path>")
    result: dict[str, Any] = {"run_dir": str(run_dir), "checkpoint": str(cfg.get("checkpoint") or ""), "test_only": test_only}
    if not test_only:
        trainer.fit(module, datamodule=datamodule)
        result["fit_metrics"] = {k: float(v) for k, v in trainer.logged_metrics.items()}
    result["data"] = datamodule.summary()
    if (test_only or bool(cfg.get("test_after_fit", True))) and not bool(cfg.trainer.get("fast_dev_run", False)):
        test_metrics = trainer.test(module, datamodule=datamodule, verbose=False)
        result["test_metrics"] = {k: float(v) for row in test_metrics for k, v in row.items()}
        result["data"] = datamodule.summary()
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def _summarise(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation of every test metric over the folds."""
    from .summarize import summarise

    return summarise(dict(results))  # type: ignore[arg-type]


def run_folds(cfg: DictConfig) -> dict[str, Any]:
    """Run ``run`` once per fold file below ``cfg.folds_dir`` and summarise.

    Each fold uses ``data.fold_json = <folds_dir>/fold_NN.json`` and writes to
    ``<output_root>/<run_name>/fold_NN``; the sweep directory receives
    ``summary.json`` (per-fold test metrics plus mean and sd) and
    ``summary.csv`` (one row per fold).
    """
    import csv

    folds_dir = Path(str(cfg.folds_dir))
    folds_dir = folds_dir if folds_dir.is_absolute() else PROJECT_ROOT / folds_dir
    fold_files = sorted(folds_dir.glob("fold_*.json"))
    wanted = cfg.get("fold_ids")
    if wanted:
        wanted_names = {f"fold_{int(i):02d}" for i in wanted}
        fold_files = [f for f in fold_files if f.stem in wanted_names]
    if not fold_files:
        raise FileNotFoundError(f"no fold files selected in {folds_dir}")
    sweep_name = str(cfg.run_name)
    sweep_dir = Path(str(cfg.output_root))
    sweep_dir = (sweep_dir if sweep_dir.is_absolute() else PROJECT_ROOT / sweep_dir) / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for fold_file in fold_files:
        fold_cfg = OmegaConf.merge(cfg, OmegaConf.create({"folds_dir": None, "fold_ids": None, "run_name": f"{sweep_name}/{fold_file.stem}", "data": {"fold_json": str(fold_file)}}))
        print(f"[sweep] {fold_file.stem}: {fold_file}")
        results[fold_file.stem] = run(fold_cfg)  # type: ignore[arg-type]
        payload = {"folds_dir": str(folds_dir), "folds": results, "summary": _summarise(results)}
        (sweep_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    names = sorted({name for fold in results.values() for name in fold.get("test_metrics", {})})
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", *names])
        for fold_name, fold in results.items():
            writer.writerow([fold_name, *[fold.get("test_metrics", {}).get(name, "") for name in names]])
        summary = _summarise(results)
        writer.writerow(["mean", *[summary[name]["mean"] for name in names]])
        writer.writerow(["sd", *[summary[name]["sd"] for name in names]])
    return {"sweep_dir": str(sweep_dir), "folds": list(results), "summary": _summarise(results)}


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry: every argument is a Hydra override."""
    import sys

    overrides = list(argv) if argv is not None else sys.argv[1:]
    cfg = compose_config(overrides)
    if bool(cfg.get("print_config", False)):
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return 0
    if cfg.get("folds_dir"):
        sweep = run_folds(cfg)
        print(json.dumps({"sweep_dir": sweep["sweep_dir"], "folds": sweep["folds"], "summary": sweep["summary"]}, indent=2))
        return 0
    result = run(cfg)
    print(json.dumps({"run_dir": result["run_dir"], "data": result["data"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
