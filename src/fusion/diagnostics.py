"""Training diagnostics for the cycle-aware fusion model (``MotionDiagnostics``).

Research Motivation:
    The version-2 objectives act on intermediate representations (motion
    features, reliability logits, residuals) whose failure modes are silent:
    a feature-level periodicity loss can be driven to zero by a *constant*
    motion feature, a zero-initialised FiLM can stay at zero so the motion
    branch never influences the pose feature, the reliability head can sit
    at 0.5 / 0.5 forever or collapse to one view, and the bounded residual
    can saturate at its limit.  None of these show up in the loss curves.
    This module computes the statistics that expose them; they are logged
    under ``diag/<name>`` by the Lightning module and switched per group in
    ``configs/fusion/diagnostics``.

Metrics (all scalars; ``F`` = motion feature ``[B, T, J, D]``, ``m`` = validity ``[B, T, J]``):

    Feature variance decomposition (:func:`feature_variance`)
        var_total   = Var over all valid (b, t, j) of F                 (mean over channels)
        var_time    = mean_{b,j} Var_t F(b, t, j)                       -> 0: phases indistinguishable
        var_joint   = mean_{b,t} Var_j F(b, t, j)                       -> 0: joints indistinguishable
        var_batch   = Var_b of the per-window mean feature
        var_channel = mean_{b,t,j} Var_d F(b, t, j, d)

    Phase similarity (:func:`phase_similarity`)
        sim_same_phase   = mean cos(F(cycle i, phi, j), F(cycle i+1, phi, j))
        sim_diff_phase   = mean cos(F(cycle i, phi, j), F(cycle i+1, phi', j)),  |phi - phi'|_circ > margin
        sim_random_joint = mean cos(F(t, j), F(t, j')),  j' a random other joint
        collapse_gap     = sim_same_phase - sim_diff_phase
        A healthy representation has sim_same_phase > sim_diff_phase; all
        three near 1 with var_time -> 0 is a representation collapse.

    FiLM (:func:`film_statistics`)   gamma = f_gamma(F_motion), beta = f_beta(F_motion)
        gamma_abs_mean, gamma_std, gamma_abs_max, beta_abs_mean, beta_std, beta_abs_max
        (near zero after training = the motion branch does not modulate the pose feature)

    Residual (:func:`residual_statistics`)
        delta_abs_mean, delta_abs_median, delta_abs_max (per-axis |Delta_P|),
        delta_saturation = fraction of valid components with |Delta_P| >= saturation_ratio * max_delta

    Reliability (:func:`reliability_statistics`)
        entropy = -w_A log w_A - w_B log w_B (max ln 2 = 0.693 at 0.5 / 0.5),
        w_a_mean, w_b_mean, frac_w_a_gt (w_A > threshold), frac_w_b_gt

    Gradient norms (:func:`gradient_norms`)
        L2 norm of the gradients of each named sub-module (pose encoder,
        short / long motion encoders, FiLM, cross-view attention,
        reliability head, residual head); computed only every
        ``gradient_norm.interval`` optimisation steps when enabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GradientNormConfig:
    """Optional gradient-norm diagnostics (``diagnostics.gradient_norm``).

    Attributes:
        enabled: Compute per-module gradient norms.
        interval: Every ``interval``-th optimisation step.
    """

    enabled: bool = False
    interval: int = 100

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("gradient_norm.interval must be positive")


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Which diagnostic groups to log (``configs/fusion/diagnostics``).

    Attributes:
        motion_feature: Feature variance decomposition.
        phase_similarity: Same / different-phase and random-joint similarities.
        phase_negative_margin: Circular phase margin (cycles) for "different phase".
        film: FiLM gamma / beta statistics.
        residual: Residual magnitude and saturation.
        residual_saturation_ratio: |Delta_P| >= ratio * max_delta counts as saturated.
        reliability: Weight entropy and hard-selection fractions.
        reliability_threshold: Weight above which a view counts as "selected".
        gradient_norm: :class:`GradientNormConfig`.
    """

    motion_feature: bool = True
    phase_similarity: bool = True
    phase_negative_margin: float = 0.25
    film: bool = True
    residual: bool = True
    residual_saturation_ratio: float = 0.95
    reliability: bool = True
    reliability_threshold: float = 0.9
    gradient_norm: GradientNormConfig = field(default_factory=GradientNormConfig)

    def __post_init__(self) -> None:
        if not 0.0 < self.phase_negative_margin <= 0.5:
            raise ValueError("phase_negative_margin must be in (0, 0.5]")
        if not 0.0 < self.residual_saturation_ratio <= 1.0 or not 0.5 < self.reliability_threshold < 1.0:
            raise ValueError("residual_saturation_ratio in (0, 1], reliability_threshold in (0.5, 1)")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "DiagnosticsConfig" | None) -> "DiagnosticsConfig":
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
        gradient = GradientNormConfig(**dict(payload.pop("gradient_norm", {}) or {}))
        return cls(gradient_norm=gradient, **payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _masked_variance(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Variance of ``values`` along ``dim`` over masked entries (0 where < 2 entries)."""
    weight = mask.to(values.dtype).unsqueeze(-1)
    count = weight.sum(dim=dim, keepdim=True)
    mean = (values * weight).sum(dim=dim, keepdim=True) / count.clamp_min(1.0)
    var = (((values - mean) ** 2) * weight).sum(dim=dim, keepdim=True) / (count - 1).clamp_min(1.0)
    return torch.where(count >= 2, var, torch.zeros_like(var)).squeeze(dim)


def feature_variance(feature: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Variance decomposition of a ``[B, T, J, D]`` feature over valid entries.

    Args:
        feature: ``[B, T, J, D]``.
        valid: ``[B, T, J]`` bool.

    Returns:
        ``var_total``, ``var_time``, ``var_joint``, ``var_batch``, ``var_channel``
        (all averaged over the remaining axes).
    """
    valid = valid.bool()
    flat = feature[valid]  # [N, D]
    total = flat.var(dim=0, unbiased=False).mean() if flat.shape[0] > 1 else feature.new_zeros(())
    var_time = _masked_variance(feature, valid, dim=1).mean(dim=-1)  # [B, J]
    var_joint = _masked_variance(feature, valid, dim=2).mean(dim=-1)  # [B, T]
    window_mean = (feature * valid[..., None]).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)[:, None]  # [B, D]
    var_batch = window_mean.var(dim=0, unbiased=False).mean() if feature.shape[0] > 1 else feature.new_zeros(())
    var_channel = feature.var(dim=-1, unbiased=False)[valid].mean() if flat.shape[0] > 0 else feature.new_zeros(())
    return {
        "var_total": total,
        "var_time": var_time[valid.any(dim=1)].mean() if valid.any() else feature.new_zeros(()),
        "var_joint": var_joint[valid.any(dim=2)].mean() if valid.any() else feature.new_zeros(()),
        "var_batch": var_batch,
        "var_channel": var_channel,
    }


def phase_similarity(
    feature: Tensor,
    valid: Tensor,
    phase: Tensor,
    phase_valid: Tensor,
    cycle_index: Tensor,
    *,
    negative_phase_margin: float,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Same-phase / different-phase / random-joint cosine similarities.

    Args:
        feature: ``[B, T, J, D]`` motion feature.
        valid: ``[B, T, J]`` bool.
        phase: ``[B, T]`` phase.
        phase_valid: ``[B, T]`` bool.
        cycle_index: ``[B, T]`` cycle index.
        negative_phase_margin: Circular margin (cycles) defining "different phase".
        generator: Optional generator for the random joint pairing.

    Returns:
        ``sim_same_phase``, ``sim_diff_phase``, ``sim_random_joint``,
        ``collapse_gap`` (same − diff); NaN where no pair exists.
    """
    from .losses import phase_pair_masks

    unit = torch.nn.functional.normalize(feature, dim=-1)
    positive, negative = phase_pair_masks(phase, phase_valid, cycle_index, negative_phase_margin)
    joint_valid = valid.bool().permute(0, 2, 1)  # [B, J, T]
    pair_valid = joint_valid[:, :, :, None] & joint_valid[:, :, None, :]
    similarity = torch.einsum("btjd,bsjd->bjts", unit, unit)
    pos = positive[:, None] & pair_valid
    neg = negative[:, None] & pair_valid
    nan = feature.new_tensor(float("nan"))
    same = similarity[pos].mean() if bool(pos.any()) else nan
    diff = similarity[neg].mean() if bool(neg.any()) else nan
    joints = feature.shape[2]
    if joints > 1:
        offset = torch.randint(1, joints, (joints,), generator=generator, device="cpu").to(feature.device)
        partner = (torch.arange(joints, device=feature.device) + offset) % joints
        random_sim = (unit * unit[:, :, partner]).sum(dim=-1)
        random_mask = valid.bool() & valid.bool()[:, :, partner]
        random = random_sim[random_mask].mean() if bool(random_mask.any()) else nan
    else:
        random = nan
    return {"sim_same_phase": same, "sim_diff_phase": diff, "sim_random_joint": random, "collapse_gap": same - diff}


def film_statistics(film: nn.Module, motion_feature: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """``gamma`` / ``beta`` statistics of a :class:`~fusion.modules.film.FiLMMotionGuidance`.

    The modulation is recomputed from the motion feature with the module's
    own linear maps (no architectural change).
    """
    with torch.no_grad():
        gamma = film.gamma(motion_feature)
        if getattr(film, "gamma_bound", None) is not None:
            gamma = film.gamma_bound * torch.tanh(gamma / film.gamma_bound)
        beta = film.beta(motion_feature)
    mask = valid.bool()
    g, b = gamma[mask], beta[mask]
    if g.numel() == 0:
        nan = motion_feature.new_tensor(float("nan"))
        return {k: nan for k in ("gamma_abs_mean", "gamma_std", "gamma_abs_max", "beta_abs_mean", "beta_std", "beta_abs_max")}
    return {
        "gamma_abs_mean": g.abs().mean(), "gamma_std": g.std(), "gamma_abs_max": g.abs().max(),
        "beta_abs_mean": b.abs().mean(), "beta_std": b.std(), "beta_abs_max": b.abs().max(),
    }


def residual_statistics(delta: Tensor, valid: Tensor, max_delta: float | None, *, saturation_ratio: float) -> dict[str, Tensor]:
    """Magnitude and saturation of the residual ``Delta_P`` (per-axis values)."""
    values = delta[valid.bool()].abs().flatten()
    if values.numel() == 0:
        nan = delta.new_tensor(float("nan"))
        return {"delta_abs_mean": nan, "delta_abs_median": nan, "delta_abs_max": nan, "delta_saturation": nan}
    out = {"delta_abs_mean": values.mean(), "delta_abs_median": values.median(), "delta_abs_max": values.max()}
    out["delta_saturation"] = (values >= saturation_ratio * max_delta).float().mean() if max_delta is not None else delta.new_zeros(())
    return out


def reliability_statistics(weight_a: Tensor, weight_b: Tensor, valid: Tensor, *, threshold: float) -> dict[str, Tensor]:
    """Entropy and selection statistics of the reliability weights ``[B, T, J, 1]``."""
    mask = valid.bool()
    wa, wb = weight_a[..., 0][mask], weight_b[..., 0][mask]
    if wa.numel() == 0:
        nan = weight_a.new_tensor(float("nan"))
        return {k: nan for k in ("entropy", "w_a_mean", "w_b_mean", "frac_w_a_gt", "frac_w_b_gt")}
    entropy = -(wa * torch.log(wa.clamp_min(1e-8)) + wb * torch.log(wb.clamp_min(1e-8)))
    return {
        "entropy": entropy.mean(), "w_a_mean": wa.mean(), "w_b_mean": wb.mean(),
        "frac_w_a_gt": (wa > threshold).float().mean(), "frac_w_b_gt": (wb > threshold).float().mean(),
    }


def gradient_norms(named_modules: Mapping[str, nn.Module]) -> dict[str, Tensor]:
    """L2 norm of the accumulated gradients of each named module (after backward)."""
    out: dict[str, Tensor] = {}
    for name, module in named_modules.items():
        grads = [p.grad.detach().flatten() for p in module.parameters() if p.grad is not None]
        out[f"grad_norm/{name}"] = torch.cat(grads).norm() if grads else torch.zeros(())
    return out
