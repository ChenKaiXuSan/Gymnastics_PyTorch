"""External fusion baselines shared by the private, FreeMan and Unity evaluations.

Every method takes the two synchronized MHR70 sequences (face reference view
first), fuses them in the pelvis-centred body frame exactly as
``current_body_average`` does, and restores the result at the face pelvis. The
classical rows need only numpy/scipy; the SmoothNet row applies the public
Human3.6M checkpoint of Zeng et al. (ECCV 2022) zero-shot as a temporal
refiner on top of the body-frame average.

Missing observations are handled uniformly: a joint whose coordinates are not
finite is treated as an absent measurement (Kalman), receives zero reliability
weight (jitter weighting), or is filled by nearest valid neighbour before
filtering (Butterworth, SmoothNet); the mask is re-applied afterwards.

Noise parameters are estimated from the sequence itself (robust second-
difference statistics), never from any reference, so the methods remain
label-free on every dataset.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

import numpy as np

from gymnastics.baselines.experiment_matrix import (
    BASELINE_METHODS,
    CLASSICAL_METHODS,
    EXTERNAL_REFINER_METHODS,
    build_body_frame,
    kpts_body_to_world,
    kpts_world_to_body,
)

__all__ = [
    "BASELINE_METHODS",
    "CLASSICAL_METHODS",
    "EXTERNAL_REFINER_METHODS",
    "fuse_baseline",
]

BUTTERWORTH_CUTOFF_HZ = 6.0
BUTTERWORTH_ORDER = 4
JITTER_WINDOW = 9
SMOOTHNET_WINDOW = 32
SMOOTHNET_CHECKPOINT_ENV = "GYMNASTICS_SMOOTHNET_CHECKPOINT"
DEFAULT_SMOOTHNET_CHECKPOINT = Path(
    "local/weights/smoothnet/h36m_fcn_3D_checkpoint_32.pth.tar"
)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def _body_views(face: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return kpts_world_to_body(face), kpts_world_to_body(side)


def _restore_world(fused_body: np.ndarray, face: np.ndarray) -> np.ndarray:
    pelvis, rotation = build_body_frame(face)
    return kpts_body_to_world(fused_body.astype(np.float32), pelvis, rotation)


def _finite_mask(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1)


def _fill_nearest(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid frames per joint with the nearest valid frame (edge hold)."""
    filled = np.array(points, dtype=np.float64, copy=True)
    frames = np.arange(points.shape[0])
    for joint in range(points.shape[1]):
        ok = valid[:, joint]
        if ok.all():
            continue
        if not ok.any():
            filled[:, joint] = 0.0
            continue
        good = frames[ok]
        nearest = good[np.abs(frames[:, None] - good[None, :]).argmin(axis=1)]
        filled[:, joint] = filled[nearest, joint]
    return filled


def _second_difference(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Second temporal difference and its validity mask."""
    if points.shape[0] < 3:
        empty = np.zeros((0, *points.shape[1:]), dtype=np.float64)
        return empty, np.zeros((0, points.shape[1]), dtype=bool)
    d2 = points[2:] - 2.0 * points[1:-1] + points[:-2]
    ok = valid[2:] & valid[1:-1] & valid[:-2]
    return d2, ok


def _robust_variance(values: np.ndarray) -> float:
    """Squared scaled MAD; falls back to a tiny floor for degenerate input."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-8
    mad = np.median(np.abs(values - np.median(values)))
    return float(max((1.4826 * mad) ** 2, 1e-8))


def _measurement_noise(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-joint measurement variance from second-difference jitter [J]."""
    d2, ok = _second_difference(points, valid)
    joints = points.shape[1]
    noise = np.full(joints, 1e-8, dtype=np.float64)
    if d2.shape[0] == 0:
        return noise
    for joint in range(joints):
        samples = d2[ok[:, joint], joint].reshape(-1)
        # White measurement noise of variance r has second-difference variance 6r.
        noise[joint] = _robust_variance(samples) / 6.0
    return noise


def _process_noise(points: np.ndarray, valid: np.ndarray, fps: float) -> float:
    """Acceleration variance of the low-passed signal, in (units/s^2)^2."""
    filled = _fill_nearest(points, valid)
    kernel = np.ones(11, dtype=np.float64) / 11.0
    if filled.shape[0] >= 11:
        flat = filled.reshape(filled.shape[0], -1)
        smooth = np.stack(
            [np.convolve(flat[:, c], kernel, mode="same") for c in range(flat.shape[1])],
            axis=1,
        ).reshape(filled.shape)
    else:
        smooth = filled
    d2, ok = _second_difference(smooth, valid)
    if d2.shape[0] == 0:
        return 1e-6
    accel = d2[ok] * float(fps) ** 2
    return float(max(np.mean(accel**2), 1e-8))


# --------------------------------------------------------------------------- #
# Kalman fusion (constant-velocity, two measurements per frame)
# --------------------------------------------------------------------------- #


def _kalman_channels(
    measurements: tuple[np.ndarray, ...],
    masks: tuple[np.ndarray, ...],
    noises: tuple[np.ndarray, ...],
    *,
    fps: float,
    sigma_a2: float,
    smoother: bool,
) -> np.ndarray:
    """Run one constant-velocity Kalman filter per channel.

    ``measurements[v]`` is ``[T, C]``, ``masks[v]`` is ``[T, C]`` and
    ``noises[v]`` is ``[C]``. Returns the filtered (or RTS-smoothed) positions
    ``[T, C]``.
    """
    frames, channels = measurements[0].shape
    dt = 1.0 / float(fps)
    transition = np.array([[1.0, dt], [0.0, 1.0]])
    process = sigma_a2 * np.array(
        [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]]
    )
    state = np.zeros((channels, 2))
    first = np.zeros(channels)
    counts = np.zeros(channels)
    for z, m in zip(measurements, masks):
        first += np.where(m[0], z[0], 0.0)
        counts += m[0]
    state[:, 0] = np.divide(first, counts, out=np.zeros(channels), where=counts > 0)
    base_noise = np.mean(np.stack(noises), axis=0)
    cov = np.zeros((channels, 2, 2))
    cov[:, 0, 0] = base_noise
    cov[:, 1, 1] = base_noise / dt**2
    filtered = np.zeros((frames, channels, 2))
    filtered_cov = np.zeros((frames, channels, 2, 2))
    predicted = np.zeros((frames, channels, 2))
    predicted_cov = np.zeros((frames, channels, 2, 2))
    for t in range(frames):
        if t > 0:
            state = state @ transition.T
            cov = transition @ cov @ transition.T + process
        predicted[t] = state
        predicted_cov[t] = cov
        for z, m, r in zip(measurements, masks, noises):
            observed = m[t]
            if not observed.any():
                continue
            innovation = z[t] - state[:, 0]
            s = cov[:, 0, 0] + r
            gain = cov[:, :, 0] / s[:, None]
            gain = np.where(observed[:, None], gain, 0.0)
            state = state + gain * innovation[:, None]
            identity = np.eye(2)[None]
            update = identity - gain[:, :, None] @ np.array([[1.0, 0.0]])[None]
            cov = update @ cov
        filtered[t] = state
        filtered_cov[t] = cov
    if not smoother:
        return filtered[:, :, 0]
    smoothed = filtered.copy()
    smoothed_cov = filtered_cov.copy()
    for t in range(frames - 2, -1, -1):
        pred_cov = predicted_cov[t + 1]
        gain = filtered_cov[t] @ transition.T @ np.linalg.pinv(pred_cov)
        smoothed[t] = filtered[t] + np.einsum(
            "cij,cj->ci", gain, smoothed[t + 1] - predicted[t + 1]
        )
        smoothed_cov[t] = filtered_cov[t] + gain @ (
            smoothed_cov[t + 1] - pred_cov
        ) @ np.transpose(gain, (0, 2, 1))
    return smoothed[:, :, 0]


def kalman_body_fusion(
    face: np.ndarray,
    side: np.ndarray,
    *,
    fps: float,
    smoother: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    face_body, side_body = _body_views(face, side)
    valid_face = _finite_mask(face_body)
    valid_side = _finite_mask(side_body)
    joints = face.shape[1]
    noise_face = _measurement_noise(face_body, valid_face)
    noise_side = _measurement_noise(side_body, valid_side)
    both = valid_face & valid_side
    average = np.where(
        both[..., None],
        0.5 * (np.nan_to_num(face_body) + np.nan_to_num(side_body)),
        np.where(valid_face[..., None], np.nan_to_num(face_body), np.nan_to_num(side_body)),
    )
    sigma_a2 = _process_noise(average, valid_face | valid_side, fps)
    frames = face.shape[0]
    fused_flat = _kalman_channels(
        (
            np.nan_to_num(face_body).reshape(frames, -1),
            np.nan_to_num(side_body).reshape(frames, -1),
        ),
        (
            np.repeat(valid_face, 3, axis=1),
            np.repeat(valid_side, 3, axis=1),
        ),
        (np.repeat(noise_face, 3), np.repeat(noise_side, 3)),
        fps=fps,
        sigma_a2=sigma_a2,
        smoother=smoother,
    )
    fused_body = fused_flat.reshape(frames, joints, 3)
    fused_body = np.where((valid_face | valid_side)[..., None], fused_body, np.nan)
    extra = {
        "fusion_frame": "pelvis_centred_body_frame",
        "kalman_model": "constant_velocity_per_joint_axis",
        "kalman_smoother": "rts" if smoother else "none",
        "measurement_noise_source": "robust_second_difference_per_view_joint",
        "process_noise_sigma_a2": float(sigma_a2),
        "mean_measurement_noise_face": float(np.mean(noise_face)),
        "mean_measurement_noise_side": float(np.mean(noise_side)),
    }
    return _restore_world(fused_body, face), extra


# --------------------------------------------------------------------------- #
# Reliability (inverse local jitter) weighted average
# --------------------------------------------------------------------------- #


def _local_jitter(points: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Moving RMS of second differences per joint, ``[T, J]``."""
    frames, joints = valid.shape
    d2, ok = _second_difference(points, valid)
    energy = np.zeros((frames, joints))
    count = np.zeros((frames, joints))
    if d2.shape[0] > 0:
        mag = np.where(ok, np.linalg.norm(d2, axis=-1), 0.0)
        energy[1:-1] = mag**2
        count[1:-1] = ok
    kernel = np.ones(window)
    pad = window // 2
    out = np.zeros((frames, joints))
    for joint in range(joints):
        e = np.convolve(energy[:, joint], kernel, mode="full")[pad : pad + frames]
        c = np.convolve(count[:, joint], kernel, mode="full")[pad : pad + frames]
        out[:, joint] = np.sqrt(np.divide(e, c, out=np.zeros_like(e), where=c > 0))
    return out


def jitter_weighted_body_average(
    face: np.ndarray,
    side: np.ndarray,
    *,
    window: int = JITTER_WINDOW,
    eps: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    face_body, side_body = _body_views(face, side)
    valid_face = _finite_mask(face_body)
    valid_side = _finite_mask(side_body)
    jitter_face = _local_jitter(face_body, valid_face, window)
    jitter_side = _local_jitter(side_body, valid_side, window)
    if eps is None:
        pooled = np.concatenate([jitter_face[valid_face], jitter_side[valid_side]])
        pooled = pooled[pooled > 0]
        eps = float(np.median(pooled)) * 0.1 if pooled.size else 1e-6
        eps = max(eps, 1e-8)
    w_face = np.where(valid_face, 1.0 / (jitter_face + eps), 0.0)
    w_side = np.where(valid_side, 1.0 / (jitter_side + eps), 0.0)
    total = w_face + w_side
    w_face = np.divide(w_face, total, out=np.zeros_like(w_face), where=total > 0)
    w_side = np.divide(w_side, total, out=np.zeros_like(w_side), where=total > 0)
    fused_body = (
        w_face[..., None] * np.nan_to_num(face_body)
        + w_side[..., None] * np.nan_to_num(side_body)
    )
    fused_body = np.where((total > 0)[..., None], fused_body, np.nan)
    extra = {
        "fusion_frame": "pelvis_centred_body_frame",
        "weight_source": "inverse_local_second_difference_rms",
        "jitter_window": int(window),
        "jitter_eps": float(eps),
        "mean_face_weight": float(np.mean(w_face[total > 0])) if (total > 0).any() else 0.5,
    }
    return _restore_world(fused_body, face), extra


# --------------------------------------------------------------------------- #
# Butterworth low-pass on the body-frame average
# --------------------------------------------------------------------------- #


def butterworth_body_average(
    face: np.ndarray,
    side: np.ndarray,
    *,
    fps: float,
    cutoff_hz: float = BUTTERWORTH_CUTOFF_HZ,
    order: int = BUTTERWORTH_ORDER,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.signal import butter, filtfilt

    face_body, side_body = _body_views(face, side)
    valid = _finite_mask(face_body) & _finite_mask(side_body)
    average = 0.5 * (np.nan_to_num(face_body) + np.nan_to_num(side_body))
    filled = _fill_nearest(average, valid)
    nyquist = 0.5 * float(fps)
    normalized = min(cutoff_hz / nyquist, 0.99)
    b, a = butter(order, normalized, btype="low")
    padlen = 3 * (max(len(a), len(b)) - 1)
    frames = filled.shape[0]
    flat = filled.reshape(frames, -1)
    if frames > padlen:
        smoothed = filtfilt(b, a, flat, axis=0)
    elif frames > 3:
        smoothed = filtfilt(b, a, flat, axis=0, padlen=frames - 2)
    else:
        smoothed = flat
    fused_body = np.where(valid[..., None], smoothed.reshape(filled.shape), np.nan)
    extra = {
        "fusion_frame": "pelvis_centred_body_frame",
        "filter": "butterworth_zero_phase",
        "cutoff_hz": float(cutoff_hz),
        "order": int(order),
        "fps": float(fps),
    }
    return _restore_world(fused_body, face), extra


# --------------------------------------------------------------------------- #
# SmoothNet zero-shot refinement of the body-frame average
# --------------------------------------------------------------------------- #

_SMOOTHNET_CACHE: dict[str, Any] = {}


def resolve_smoothnet_checkpoint(path: str | Path | None = None) -> Path:
    candidate = Path(path or os.environ.get(SMOOTHNET_CHECKPOINT_ENV, DEFAULT_SMOOTHNET_CHECKPOINT))
    if not candidate.is_absolute():
        from gymnastics.common.paths import PROJECT_ROOT

        candidate = PROJECT_ROOT / candidate
    if not candidate.is_file():
        raise FileNotFoundError(
            "SmoothNet checkpoint not found; download the public Human3.6M "
            f"3D checkpoint (window 32) to {candidate} or set "
            f"{SMOOTHNET_CHECKPOINT_ENV}"
        )
    return candidate


def build_smoothnet(
    window_size: int,
    *,
    hidden_size: int,
    res_hidden_size: int,
    num_blocks: int,
    dropout: float = 0.0,
) -> Any:
    """SmoothNet (Zeng et al., ECCV 2022): temporal-only MLP refiner.

    Ported from github.com/cure-lab/SmoothNet (lib/models/smoothnet.py), which
    is released for non-commercial scientific research. Input/output are
    ``[N, C, T]`` with ``T == window_size``.
    """
    import torch
    from torch import nn

    class ResBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = nn.Linear(hidden_size, res_hidden_size)
            self.linear2 = nn.Linear(res_hidden_size, hidden_size)
            self.lrelu = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = x
            x = self.lrelu(self.dropout(self.linear1(x)))
            x = self.lrelu(self.dropout(self.linear2(x)))
            return x + identity

    class SmoothNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.window_size = window_size
            self.encoder = nn.Sequential(
                nn.Linear(window_size, hidden_size), nn.LeakyReLU(0.1)
            )
            self.res_blocks = nn.Sequential(*[ResBlock() for _ in range(num_blocks)])
            self.decoder = nn.Linear(hidden_size, window_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.res_blocks(self.encoder(x)))

    return SmoothNet()


def load_smoothnet(checkpoint: str | Path | None = None) -> tuple[Any, dict[str, Any]]:
    import torch

    path = resolve_smoothnet_checkpoint(checkpoint)
    key = str(path)
    if key in _SMOOTHNET_CACHE:
        return _SMOOTHNET_CACHE[key]
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw) if isinstance(raw, Mapping) else raw
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    window = int(state["encoder.0.weight"].shape[1])
    hidden = int(state["encoder.0.weight"].shape[0])
    res_hidden = int(state["res_blocks.0.linear1.weight"].shape[0])
    blocks = len({k.split(".")[1] for k in state if k.startswith("res_blocks.")})
    model = build_smoothnet(
        window, hidden_size=hidden, res_hidden_size=res_hidden, num_blocks=blocks
    )
    model.load_state_dict(state)
    model.eval()
    info = {
        "checkpoint_path": key,
        "window_size": window,
        "hidden_size": hidden,
        "res_hidden_size": res_hidden,
        "num_blocks": blocks,
    }
    _SMOOTHNET_CACHE[key] = (model, info)
    return model, info


def smoothnet_refine(
    sequence: np.ndarray,
    model: Any,
    window: int,
    *,
    scale: float,
) -> np.ndarray:
    """Sliding-window (step 1) refinement with overlap averaging, ``[T, C]``."""
    import torch

    frames, channels = sequence.shape
    if frames < window:
        pad_after = window - frames
        padded = np.concatenate([sequence, np.repeat(sequence[-1:], pad_after, axis=0)])
    else:
        padded = sequence
    total = padded.shape[0]
    starts = np.arange(0, total - window + 1)
    windows = np.stack([padded[s : s + window] for s in starts])  # [N, T, C]
    inputs = torch.from_numpy(
        (windows * scale).transpose(0, 2, 1).astype(np.float32)
    )  # [N, C, T]
    outputs = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], 512):
            outputs.append(model(inputs[start : start + 512]).numpy())
    refined = np.concatenate(outputs).transpose(0, 2, 1) / scale  # [N, T, C]
    accum = np.zeros((total, channels))
    count = np.zeros((total, 1))
    for i, s in enumerate(starts):
        accum[s : s + window] += refined[i]
        count[s : s + window] += 1.0
    return (accum / count)[:frames]


def smoothnet_body_average(
    face: np.ndarray,
    side: np.ndarray,
    *,
    checkpoint: str | Path | None = None,
    input_scale: float = 1000.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    model, info = load_smoothnet(checkpoint)
    face_body, side_body = _body_views(face, side)
    valid = _finite_mask(face_body) & _finite_mask(side_body)
    average = 0.5 * (np.nan_to_num(face_body) + np.nan_to_num(side_body))
    filled = _fill_nearest(average, valid)
    frames = filled.shape[0]
    refined = smoothnet_refine(
        filled.reshape(frames, -1), model, int(info["window_size"]), scale=input_scale
    ).reshape(filled.shape)
    fused_body = np.where(valid[..., None], refined, np.nan)
    extra = {
        "fusion_frame": "pelvis_centred_body_frame",
        "refiner": "smoothnet_zero_shot",
        "refiner_input": "body_frame_average",
        "refiner_input_scale": float(input_scale),
        "refiner_training_data": "Human3.6M FCN 3D (public checkpoint)",
        **info,
    }
    return _restore_world(fused_body, face), extra


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


def fuse_baseline(
    method: str,
    face: np.ndarray,
    side: np.ndarray,
    *,
    fps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one registered baseline on world-frame ``[T,70,3]`` views."""
    face = np.asarray(face, dtype=np.float32)
    side = np.asarray(side, dtype=np.float32)
    if face.shape != side.shape or face.ndim != 3 or face.shape[-1] != 3:
        raise ValueError("face and side must have equal shape [T, J, 3]")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if method == "kalman_body_fusion":
        return kalman_body_fusion(face, side, fps=fps, smoother=False)
    if method == "kalman_rts_body_fusion":
        return kalman_body_fusion(face, side, fps=fps, smoother=True)
    if method == "jitter_weighted_body_average":
        return jitter_weighted_body_average(face, side)
    if method == "butterworth_body_average":
        return butterworth_body_average(face, side, fps=fps)
    if method == "smoothnet_body_average":
        return smoothnet_body_average(face, side)
    raise ValueError(f"unknown baseline method: {method}")
