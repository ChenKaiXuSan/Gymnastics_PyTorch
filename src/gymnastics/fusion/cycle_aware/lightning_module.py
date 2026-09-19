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
    total, recovery, periodicity, symmetry, half_symmetry, residual   loss terms
    weight_a_mean, weight_entropy                          reliability statistics
    corrupted_error(_base)                                 |P_hat - P*| on corrupted joints
    pa_mpjpe(_base|_face|_side)                            Procrustes-aligned error versus the
                                                           reference for the model, the weighted
                                                           base and each input view
    pa_mpjpe_corrupted(_base)                              the same restricted to corrupted joints
                                                           (test_with_corruption)
    svf_fraction, both_fail_fraction                       share of frames where exactly one /
                                                           both views exceed the failure threshold
    pa_mpjpe_svf(_base|_face|_side|_oracle)                errors on single-view-failure frames
    rom_retention, peak_omega_retention (_base)            measurement preservation relative to
                                                           the two inputs (and _vs_reference)
    ta_mpjpe                                               translation-aligned error, only
                                                           when the reference shares the
                                                           canonical frame (synthetic data)

``EvaluationConfig`` (``configs/cycle_aware/evaluation``) sets the failure
threshold (in reference units, metres for the real references) and which of
the optional diagnostics run.

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
from .measurement import retention_ratios
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


@dataclass(frozen=True)
class EvaluationConfig:
    """Optional diagnostics (``configs/cycle_aware/evaluation``).

    Attributes:
        failure_threshold: Per-frame mean PA error (reference units, metres
            for the real references) above which a view counts as failed.
        single_view_failure: Log errors on frames where exactly one view failed.
        per_view_error: Log the PA error of each input view.
        measurement: Log ROM / peak-velocity retention ratios.
    """

    failure_threshold: float = 0.15
    single_view_failure: bool = True
    per_view_error: bool = True
    measurement: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "EvaluationConfig" | None) -> "EvaluationConfig":
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
        return cls(**dict(value))


class CycleAwareFusionModule(pl.LightningModule):
    """Lightning training wrapper.

    Attributes:
        model: The pure :class:`CycleAwareFusionModel`.
        skeleton: Common skeleton (from the model).
        loss_config: :class:`LossConfig`.
        optimizer_config: :class:`OptimizerConfig`.
        evaluation_config: :class:`EvaluationConfig`.
    """

    def __init__(
        self,
        model_config: CycleAwareModelConfig | Mapping[str, Any] | None = None,
        loss_config: LossConfig | Mapping[str, Any] | None = None,
        optimizer_config: OptimizerConfig | Mapping[str, Any] | None = None,
        evaluation_config: EvaluationConfig | Mapping[str, Any] | None = None,
        *,
        skeleton: CommonSkeleton | None = None,
    ) -> None:
        super().__init__()
        self.model = CycleAwareFusionModel(model_config, skeleton=skeleton)
        self.skeleton = self.model.skeleton
        self.loss_config = LossConfig.from_mapping(loss_config)
        self.optimizer_config = OptimizerConfig.from_mapping(optimizer_config)
        self.evaluation_config = EvaluationConfig.from_mapping(evaluation_config)
        # Saved under the constructor argument names so that
        # ``load_from_checkpoint`` can rebuild the exact same architecture.
        self.save_hyperparameters(
            {
                "model_config": self.model.config.to_dict(),
                "loss_config": self.loss_config.to_dict(),
                "optimizer_config": asdict(self.optimizer_config),
                "evaluation_config": asdict(self.evaluation_config),
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
        corrupted = None
        if "clean_a" in batch:
            target, target_valid = pseudo_target(batch["clean_a"], batch["clean_b"], batch["clean_valid_a"], batch["clean_valid_b"], consensus_distance=self.loss_config.consensus_distance)
            corrupted = (batch["corruption_mask_a"] | batch["corruption_mask_b"]) & target_valid & valid
            error = torch.linalg.vector_norm(output.pose - target, dim=-1)
            metrics["corrupted_error"] = masked_mean(error, corrupted)
            base_error = torch.linalg.vector_norm(output.base_pose - target, dim=-1)
            metrics["corrupted_error_base"] = masked_mean(base_error, corrupted)
        reference_valid = batch.get("reference_valid")
        if reference_valid is not None and bool(reference_valid.any()):
            reference = batch["reference"]
            usable = reference_valid & valid
            alignments = [("pa_mpjpe", "procrustes")]
            canonical = batch.get("reference_canonical")
            # A translation-only comparison is meaningful only when the reference
            # lives in the same canonical body frame as the prediction.
            if canonical is not None and bool(canonical.all()):
                alignments.append(("ta_mpjpe", "translation"))
            errors: dict[str, torch.Tensor] = {}
            masks: dict[str, torch.Tensor] = {}
            for name, align in alignments:
                errors[name], masks[name] = per_joint_error(output.pose, reference, usable, align=align)
                metrics[name] = masked_mean(errors[name], masks[name])
                base_error, _ = per_joint_error(output.base_pose, reference, usable, align=align)
                metrics[f"{name}_base"] = masked_mean(base_error, masks[name])
                if name == "pa_mpjpe":
                    errors["base"] = base_error
            cfg = self.evaluation_config
            if cfg.per_view_error or cfg.single_view_failure:
                face_error, face_mask = per_joint_error(batch["pose_a"], reference, reference_valid & batch["valid_a"] & frame_mask, align="procrustes")
                side_error, side_mask = per_joint_error(batch["pose_b"], reference, reference_valid & batch["valid_b"] & frame_mask, align="procrustes")
                if cfg.per_view_error:
                    metrics["pa_mpjpe_face"] = masked_mean(face_error, face_mask)
                    metrics["pa_mpjpe_side"] = masked_mean(side_error, side_mask)
            if corrupted is not None:
                # Reference error restricted to the joints the corruption touched.
                metrics["pa_mpjpe_corrupted"] = masked_mean(errors["pa_mpjpe"], masks["pa_mpjpe"] & corrupted)
                metrics["pa_mpjpe_corrupted_base"] = masked_mean(errors["base"], masks["pa_mpjpe"] & corrupted)
            if cfg.single_view_failure:
                # Frame-level mean error of each view; a view "fails" above the threshold.
                def frame_error(error: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    count = mask.sum(dim=-1)
                    return torch.where(mask, error, torch.zeros_like(error)).sum(dim=-1) / count.clamp_min(1), count > 0
                fa, fa_ok = frame_error(face_error, face_mask)
                fb, fb_ok = frame_error(side_error, side_mask)
                measured = fa_ok & fb_ok & batch["frame_mask"]
                fail_a, fail_b = fa > cfg.failure_threshold, fb > cfg.failure_threshold
                single = measured & (fail_a ^ fail_b)
                both = measured & fail_a & fail_b
                denominator = measured.sum().clamp_min(1).to(fa.dtype)
                metrics["svf_fraction"] = single.sum().to(fa.dtype) / denominator
                metrics["both_fail_fraction"] = both.sum().to(fa.dtype) / denominator
                if bool(single.any()):
                    joint_single = single[..., None] & masks["pa_mpjpe"]
                    metrics["pa_mpjpe_svf"] = masked_mean(errors["pa_mpjpe"], joint_single)
                    metrics["pa_mpjpe_svf_base"] = masked_mean(errors["base"], joint_single)
                    metrics["pa_mpjpe_svf_face"] = masked_mean(face_error, single[..., None] & face_mask)
                    metrics["pa_mpjpe_svf_side"] = masked_mean(side_error, single[..., None] & side_mask)
                    # Oracle: the better view on every single-failure frame.
                    oracle = torch.where(fail_a[..., None], side_error, face_error)
                    oracle_mask = torch.where(fail_a[..., None], side_mask, face_mask)
                    metrics["pa_mpjpe_svf_oracle"] = masked_mean(oracle, single[..., None] & oracle_mask)
            if cfg.measurement and "cycle_index" in batch:
                cycle_index = batch["cycle_index"]
                delta_t = batch["delta_t"]
                for prefix, pose, pose_valid in (("", output.pose, valid), ("base_", output.base_pose, valid)):
                    ratios = retention_ratios(pose, pose_valid, batch["pose_a"], batch["valid_a"] & frame_mask, batch["pose_b"], batch["valid_b"] & frame_mask, cycle_index, delta_t, self.skeleton, reference=reference, reference_valid=reference_valid & frame_mask)
                    for name, value in ratios.items():
                        if torch.isfinite(value):
                            metrics[f"{name}_base" if prefix else name] = value
        elif self.evaluation_config.measurement and "cycle_index" in batch:
            for prefix, pose in (("", output.pose), ("_base", output.base_pose)):
                ratios = retention_ratios(pose, valid, batch["pose_a"], batch["valid_a"] & frame_mask, batch["pose_b"], batch["valid_b"] & frame_mask, batch["cycle_index"], batch["delta_t"], self.skeleton)
                for name, value in ratios.items():
                    if torch.isfinite(value):
                        metrics[f"{name}{prefix}"] = value
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
