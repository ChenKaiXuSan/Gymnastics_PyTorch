"""Synthetic periodic dual-view data for smoke tests and CI.

Source:
    Generated in memory, no files.  Each "subject" has a random rest pose and
    per-joint sinusoidal amplitudes; each "sequence" is several cycles of that
    motion with subject-specific period, observed by two views that add
    independent Gaussian noise and, in View B, a permanently missing joint.

Original skeleton / coordinate system:
    Directly the common skeleton in canonical units (no canonicalisation is
    applied because the data are generated in the canonical frame).

Views and synchronisation:
    Both views share the same timestamps (perfectly synchronised).  View A is
    "observed" by a camera looking along the body z axis and View B by a
    camera looking along the body x axis; both samples carry the matching
    canonical transforms so ``depth_a`` / ``depth_b`` are defined, and the
    noise of each view can be inflated along its own optical axis
    (``depth_noise_ratio``) to mimic monocular depth uncertainty.

Ground truth:
    The clean generating motion is attached as ``reference`` in the same
    frame, so translation-aligned MPJPE is meaningful here.

Cycle information:
    Exact, from the generating period.

Options (``data.options``):
    subjects, sequences_per_subject, frames, period, fps, noise,
    depth_noise_ratio (noise multiplier along each view's optical axis, 1 = isotropic)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..sample import CanonicalTransformRecord, DualViewSample
from .base import DualViewDataModule, SplitSpec

# Camera optical axes of the two synthetic views, as rows 2 of the canonical
# rotations: View A looks along body z, View B along body x (both proper rotations).
VIEW_A_ROTATION = np.eye(3, dtype=np.float32)
VIEW_B_ROTATION = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)


def synthetic_transform(rotation: np.ndarray, frames: int) -> CanonicalTransformRecord:
    """Constant canonical transform (identity origin/scale) with the given rotation."""
    return CanonicalTransformRecord(
        rotation=np.broadcast_to(rotation, (frames, 3, 3)).copy(),
        origin=np.zeros((frames, 3), dtype=np.float32),
        scale=1.0,
        valid=np.ones(frames, dtype=bool),
    )


class SyntheticDataModule(DualViewDataModule):
    """Generates periodic dual-view samples (see module docstring)."""

    def load_samples(self) -> Sequence[DualViewSample]:
        options = dict(self.config.options)
        subjects = int(options.get("subjects", 6))
        sequences = int(options.get("sequences_per_subject", 2))
        frames = int(options.get("frames", 96))
        period = int(options.get("period", 24))
        fps = float(options.get("fps", 30.0))
        noise = float(options.get("noise", 0.01))
        depth_ratio = float(options.get("depth_noise_ratio", 1.0))
        # Per-axis noise scale of each view in the body frame (depth axis inflated).
        noise_a = noise * np.array([1.0, 1.0, depth_ratio], dtype=np.float32)
        noise_b = noise * np.array([depth_ratio, 1.0, 1.0], dtype=np.float32)
        rng = np.random.default_rng(self.config.seed)
        joints = self.skeleton.num_joints
        samples: list[DualViewSample] = []
        for subject in range(subjects):
            rest = rng.normal(scale=0.5, size=(joints, 3)).astype(np.float32)
            amplitude = rng.normal(scale=0.2, size=(joints, 3)).astype(np.float32)
            for sequence in range(sequences):
                phase_offset = rng.uniform(0, 2 * np.pi)
                t = np.arange(frames, dtype=np.float32)
                clean = rest[None] + amplitude[None] * np.sin(2 * np.pi * t / period + phase_offset)[:, None, None]
                view_a = clean + (rng.normal(size=clean.shape) * noise_a).astype(np.float32)
                view_b = clean + (rng.normal(size=clean.shape) * noise_b).astype(np.float32)
                valid_a = np.ones((frames, joints), dtype=bool)
                valid_b = np.ones((frames, joints), dtype=bool)
                valid_b[:, joints - 1] = False
                bounds = tuple((s, s + period) for s in range(0, frames - period + 1, period))
                samples.append(
                    DualViewSample(
                        dataset="synthetic",
                        subject_id=f"subject_{subject:02d}",
                        sequence_id=f"sequence_{sequence:02d}",
                        view_a=view_a.astype(np.float32),
                        view_b=view_b.astype(np.float32),
                        valid_a=valid_a,
                        valid_b=valid_b,
                        timestamps=np.arange(frames, dtype=np.float64) / fps,
                        joint_names=self.skeleton.joint_names,
                        cycle_bounds=bounds,
                        reference=clean.astype(np.float32),
                        reference_valid=np.ones((frames, joints), dtype=bool),
                        reference_canonical=True,
                        transform_a=synthetic_transform(VIEW_A_ROTATION, frames),
                        transform_b=synthetic_transform(VIEW_B_ROTATION, frames),
                        metadata={"fps": fps, "period": period, "depth_noise_ratio": depth_ratio},
                    )
                )
        return samples

    def default_split(self, samples: Sequence[DualViewSample]) -> SplitSpec:
        subjects = sorted({sample.subject_id for sample in samples})
        n_val = max(1, len(subjects) // 4) if len(subjects) > 2 else 1
        n_test = max(1, len(subjects) // 4) if len(subjects) > 2 else 0
        return SplitSpec(
            train=tuple(subjects[: len(subjects) - n_val - n_test]),
            val=tuple(subjects[len(subjects) - n_val - n_test : len(subjects) - n_test]),
            test=tuple(subjects[len(subjects) - n_test :]),
        )
