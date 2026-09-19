"""PyTorch Lightning wrapper around :class:`CycleAwareFusionModel`.

The LightningModule owns the *training logic* only: it calls the pure
PyTorch model, evaluates the self-supervised objectives, logs them, and
configures the optimiser.  It contains no architecture and no dataset code,
so the same module trains on every DataModule of :mod:`.data`.

Steps:
    training_step    corrupted views -> model -> losses (see :mod:`.losses`)
    validation_step  same objectives on replayed corruption, plus diagnostic
                     metrics (recovery error on corrupted joints, weight
                     statistics, reference error when available)
    test_step        diagnostics on clean inputs, reference error when
                     available
    predict_step     returns the fused pose and reliability weights

Logged quantities (prefix ``train/``, ``val/``, ``test/``):
    total, recovery, periodicity, symmetry, residual      loss terms
    weight_a_mean, weight_entropy                          reliability statistics
    corrupted_error                                        |P_hat - P*| on corrupted joints
    pa_mpjpe, ta_mpjpe                                     Procrustes / translation aligned
                                                           error versus the reference

Optimisation:
    AdamW with linear warm-up followed by cosine decay to
    ``min_learning_rate_ratio`` of the base rate over the scheduled steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import pytorch_lightning as pl
import torch

from .losses import LossBreakdown, LossConfig, compute_losses, masked_mean, pseudo_target
from .metrics import per_joint_error
from .model import CycleAwareFusionModel, CycleAwareModelConfig
from .outputs import PoseFusionOutput
from .skeleton import CommonSkeleton


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimiser and schedule settings (``configs/cycle_aware/trainer``).

    Attributes:
        learning_rate: Base AdamW learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warm-up steps.
        min_learning_rate_ratio: Final cosine value relative to the base rate.
        betas: AdamW betas.
    """

    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_steps: int = 100
    min_learning_rate_ratio: float = 0.05
    betas: tuple[float, float] = (0.9, 0.98)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "OptimizerConfig" | None) -> "OptimizerConfig":
        """Build from a mapping; ``None`` gives defaults."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)  # type: ignore[assignment]
        except ImportError:  # pragma: no cover
            pass
        payload = dict(value)
        if "betas" in payload:
            payload["betas"] = tuple(float(b) for b in payload["betas"])
        return cls(**payload)


class CycleAwareFusionModule(pl.LightningModule):
    """Lightning training wrapper.

    Attributes:
        model: The pure :class:`CycleAwareFusionModel`.
        skeleton: Common skeleton (from the model).
        loss_config: :class:`LossConfig`.
        optimizer_config: :class:`OptimizerConfig`.
    """

    def __init__(
        self,
        model_config: CycleAwareModelConfig | Mapping[str, Any] | None = None,
        loss_config: LossConfig | Mapping[str, Any] | None = None,
        optimizer_config: OptimizerConfig | Mapping[str, Any] | None = None,
        *,
        skeleton: CommonSkeleton | None = None,
    ) -> None:
        super().__init__()
        self.model = CycleAwareFusionModel(model_config, skeleton=skeleton)
        self.skeleton = self.model.skeleton
        self.loss_config = LossConfig.from_mapping(loss_config)
        self.optimizer_config = OptimizerConfig.from_mapping(optimizer_config)
        # Saved under the constructor argument names so that
        # ``load_from_checkpoint`` can rebuild the exact same architecture.
        self.save_hyperparameters(
            {
                "model_config": self.model.config.to_dict(),
                "loss_config": self.loss_config.to_dict(),
                "optimizer_config": asdict(self.optimizer_config),
            }
        )

    # ----- forward helpers ----------------------------------------------------
    def forward(self, batch: Mapping[str, Any]) -> PoseFusionOutput:  # type: ignore[override]
        """Run the model on a :class:`FusionBatch`."""
        return self.model(
            batch["pose_a"],
            batch["pose_b"],
            batch["valid_a"],
            batch["valid_b"],
            batch["delta_t"],
            batch.get("phase"),
            batch.get("phase_valid"),
            batch.get("frame_mask"),
        )

    def _losses(self, output: PoseFusionOutput, batch: Mapping[str, Any]) -> LossBreakdown:
        return compute_losses(output, batch, skeleton=self.skeleton, config=self.loss_config, samples_per_cycle=self.model.config.samples_per_cycle)

    def _diagnostics(self, output: PoseFusionOutput, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        frame_mask = batch["frame_mask"][..., None]
        valid = output.valid & frame_mask
        metrics: dict[str, torch.Tensor] = {}
        weight_a = output.weight_a[..., 0]
        metrics["weight_a_mean"] = masked_mean(weight_a, valid)
        entropy = -(output.weight_a * torch.log(output.weight_a.clamp_min(1e-8)) + output.weight_b * torch.log(output.weight_b.clamp_min(1e-8)))[..., 0]
        metrics["weight_entropy"] = masked_mean(entropy, valid)
        if "clean_a" in batch:
            target, target_valid = pseudo_target(batch["clean_a"], batch["clean_b"], batch["clean_valid_a"], batch["clean_valid_b"], consensus_distance=self.loss_config.consensus_distance)
            corrupted = (batch["corruption_mask_a"] | batch["corruption_mask_b"]) & target_valid & valid
            error = torch.linalg.vector_norm(output.pose - target, dim=-1)
            metrics["corrupted_error"] = masked_mean(error, corrupted)
            base_error = torch.linalg.vector_norm(output.base_pose - target, dim=-1)
            metrics["corrupted_error_base"] = masked_mean(base_error, corrupted)
        reference_valid = batch.get("reference_valid")
        if reference_valid is not None and bool(reference_valid.any()):
            usable = reference_valid & valid
            for name, align in (("pa_mpjpe", "procrustes"), ("ta_mpjpe", "translation")):
                error, mask = per_joint_error(output.pose, batch["reference"], usable, align=align)
                metrics[name] = masked_mean(error, mask)
                base_error, _ = per_joint_error(output.base_pose, batch["reference"], usable, align=align)
                metrics[f"{name}_base"] = masked_mean(base_error, mask)
        return metrics

    def _log_all(self, prefix: str, losses: LossBreakdown | None, metrics: Mapping[str, torch.Tensor], batch_size: int) -> None:
        if losses is not None:
            for name, value in losses.as_dict().items():
                self.log(f"{prefix}/{name}", value, on_step=prefix == "train", on_epoch=True, prog_bar=name == "total", batch_size=batch_size)
        for name, value in metrics.items():
            self.log(f"{prefix}/{name}", value, on_step=False, on_epoch=True, batch_size=batch_size)

    # ----- Lightning hooks ----------------------------------------------------
    def on_train_epoch_start(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "set_epoch"):
            datamodule.set_epoch(self.current_epoch)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        output = self(batch)
        losses = self._losses(output, batch)
        self._log_all("train", losses, self._diagnostics(output, batch), batch["pose_a"].shape[0])
        return losses.total

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:  # type: ignore[override]
        output = self(batch)
        losses = self._losses(output, batch)
        self._log_all("val", losses, self._diagnostics(output, batch), batch["pose_a"].shape[0])

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:  # type: ignore[override]
        output = self(batch)
        losses = self._losses(output, batch)
        self._log_all("test", losses, self._diagnostics(output, batch), batch["pose_a"].shape[0])

    def predict_step(self, batch: Mapping[str, Any], batch_idx: int) -> dict[str, Any]:  # type: ignore[override]
        output = self(batch)
        return {
            "pose": output.pose,
            "base_pose": output.base_pose,
            "delta_pose": output.delta_pose,
            "weight_a": output.weight_a,
            "valid": output.valid,
            "window_id": list(batch.get("window_id", [])),
            "window_start": batch.get("window_start"),
        }

    def configure_optimizers(self):  # type: ignore[override]
        config = self.optimizer_config
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=config.betas)
        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        warmup = max(0, int(config.warmup_steps))
        floor = float(config.min_learning_rate_ratio)

        def schedule(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            if total_steps <= warmup:
                return 1.0
            progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
            return floor + (1.0 - floor) * 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
