"""Read-only visualizations for saved rotation-aware fusion sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import SkeletonSpec


@dataclass(frozen=True)
class VisualizationOutputs:
    theta_omega_path: Path
    quality_path: Path
    animation_path: Path | None = None


def visualize_saved_sequence(
    sequence_path: str | Path,
    output_dir: str | Path,
    *,
    skeleton: SkeletonSpec | None = None,
    animation: bool = False,
) -> VisualizationOutputs:
    """Write curves (and optionally a simple skeleton animation) without changing NPZ data."""
    with np.load(sequence_path, allow_pickle=False) as data:
        theta, omega = (
            np.array(data["theta_fused_rad"], copy=True),
            np.array(data["omega_fused_rad_s"], copy=True),
        )
        face, side = (
            np.array(data["quality_face"], copy=True),
            np.array(data["quality_side"], copy=True),
        )
        points = np.array(data["kpts_world"], copy=True)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    axes[0].plot(theta)
    axes[0].set_ylabel("theta (rad)")
    axes[1].plot(omega)
    axes[1].set_ylabel("omega (rad/s)")
    axes[1].set_xlabel("frame")
    figure.tight_layout()
    theta_path = target / "theta_omega.png"
    figure.savefig(theta_path)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 3))
    axis.plot(face, label="face")
    axis.plot(side, label="side")
    axis.legend()
    axis.set_xlabel("frame")
    axis.set_ylabel("quality")
    figure.tight_layout()
    quality_path = target / "quality.png"
    figure.savefig(quality_path)
    plt.close(figure)
    animation_path = None
    if animation:
        from matplotlib.animation import FuncAnimation

        with np.load(sequence_path, allow_pickle=False) as data:
            required = (
                "kpts_face_canonical",
                "kpts_side_canonical",
                "kpts_base_canonical",
                "kpts_fused_canonical",
            )
            missing = [name for name in required if name not in data.files]
            if missing:
                raise ValueError(
                    f"four-skeleton animation requires saved arrays: {missing}"
                )
            sequences = [np.array(data[name], copy=True) for name in required]
        if skeleton is None:
            raise ValueError("four-skeleton animation requires a SkeletonSpec")
        figure, axes = plt.subplots(2, 2, figsize=(8, 8))
        lines = []
        low, high = (
            float(np.nanmin(np.concatenate(sequences, axis=1)[..., :2])),
            float(np.nanmax(np.concatenate(sequences, axis=1)[..., :2])),
        )
        if low == high:
            low, high = low - 1.0, high + 1.0
        for axis, name in zip(axes.flat, ("face", "side", "base", "fused")):
            lines.append([axis.plot([], [], "-")[0] for _ in skeleton.bones])
            axis.set_title(name)
            axis.set_aspect("equal")
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)

        def draw(index: int):
            drawn = []
            for segments, values in zip(lines, sequences):
                for line, (left, right) in zip(segments, skeleton.bones):
                    pair = values[index, [left, right]]
                    line.set_data(pair[:, 0], pair[:, 1]) if np.isfinite(
                        pair
                    ).all() and np.any(pair != 0) else line.set_data([], [])
                    drawn.append(line)
            return tuple(drawn)

        animation_path = target / "four_skeletons.gif"
        FuncAnimation(figure, draw, frames=len(points), blit=True).save(
            animation_path, writer="pillow", fps=10
        )
        plt.close(figure)
    return VisualizationOutputs(theta_path, quality_path, animation_path)
