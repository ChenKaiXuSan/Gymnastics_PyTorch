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
    Both views share the same timestamps (perfectly synchronised).

Ground truth:
    The clean generating motion is attached as ``reference`` in the same
    frame, so translation-aligned MPJPE is meaningful here.

Cycle information:
    Exact, from the generating period.

Options (``data.options``):
    subjects, sequences_per_subject, frames, period, fps, noise
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..sample import DualViewSample
from .base import DualViewDataModule, SplitSpec


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
                view_a = clean + rng.normal(scale=noise, size=clean.shape).astype(np.float32)
                view_b = clean + rng.normal(scale=noise, size=clean.shape).astype(np.float32)
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
                        metadata={"fps": fps, "period": period},
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
