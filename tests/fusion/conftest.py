"""Shared synthetic fixtures for the cycle-aware fusion tests.

The fixtures build small periodic dual-view sequences in the canonical body
frame so every test runs in well under a second on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fusion.model import CycleAwareModelConfig
from fusion.sample import DualViewSample
from fusion.skeleton import CommonSkeleton, build_common_skeleton


def periodic_pose(skeleton: CommonSkeleton, frames: int, period: int, *, seed: int = 0, noise: float = 0.0) -> np.ndarray:
    """A periodic ``[T, J, 3]`` motion whose joints oscillate with ``period``."""
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.5, size=(skeleton.num_joints, 3)).astype(np.float32)
    amplitude = rng.normal(scale=0.2, size=(skeleton.num_joints, 3)).astype(np.float32)
    t = np.arange(frames, dtype=np.float32)
    phase = 2.0 * np.pi * t / period
    pose = base[None] + amplitude[None] * np.sin(phase)[:, None, None]
    if noise:
        pose = pose + rng.normal(scale=noise, size=pose.shape).astype(np.float32)
    return pose.astype(np.float32)


def make_sample(
    skeleton: CommonSkeleton,
    *,
    frames: int = 48,
    period: int = 16,
    fps: float = 30.0,
    subject: str = "s1",
    sequence: str = "seq0",
    seed: int = 0,
    with_reference: bool = False,
    cycles: bool = True,
    mids: bool = False,
) -> DualViewSample:
    """Build a synthetic :class:`DualViewSample` with complete cycles."""
    clean = periodic_pose(skeleton, frames, period, seed=seed)
    rng = np.random.default_rng(seed + 100)
    view_a = clean + rng.normal(scale=0.01, size=clean.shape).astype(np.float32)
    view_b = clean + rng.normal(scale=0.01, size=clean.shape).astype(np.float32)
    valid_a = np.ones(clean.shape[:2], dtype=bool)
    valid_b = np.ones(clean.shape[:2], dtype=bool)
    valid_b[:, -1] = False  # one joint always missing in view B
    bounds = tuple((s, s + period) for s in range(0, frames - period + 1, period)) if cycles else ()
    # The generating sinusoid peaks a quarter period after the start.
    cycle_mids = tuple(s + period // 4 for s, _ in bounds) if (cycles and mids) else ()
    return DualViewSample(
        dataset="synthetic",
        subject_id=subject,
        sequence_id=sequence,
        view_a=view_a,
        view_b=view_b,
        valid_a=valid_a,
        valid_b=valid_b,
        timestamps=np.arange(frames, dtype=np.float64) / fps,
        joint_names=skeleton.joint_names,
        cycle_bounds=bounds,
        cycle_mids=cycle_mids,
        reference=clean if with_reference else None,
        reference_valid=np.ones(clean.shape[:2], dtype=bool) if with_reference else None,
        reference_canonical=False,
        metadata={"fps": fps},
    )


@pytest.fixture(scope="session")
def skeleton() -> CommonSkeleton:
    return build_common_skeleton("mhr70_major")


@pytest.fixture(scope="session")
def tiny_config() -> CycleAwareModelConfig:
    return CycleAwareModelConfig.from_mapping(
        {
            "skeleton": "mhr70_major",
            "hidden_dim": 16,
            "num_heads": 2,
            "samples_per_cycle": 8,
            "spatial": {"layers": 1},
            "short_motion": {"layers": 1},
            "long_motion": {"layers": 1, "num_cycles": 2},
            "cross_view": {"layers": 1},
        }
    )


@pytest.fixture
def tiny_batch(skeleton: CommonSkeleton) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    batch, frames, joints = 2, 16, skeleton.num_joints
    pose_a = torch.randn(batch, frames, joints, 3)
    pose_b = pose_a + 0.05 * torch.randn(batch, frames, joints, 3)
    valid_a = torch.ones(batch, frames, joints, dtype=torch.bool)
    valid_b = torch.ones(batch, frames, joints, dtype=torch.bool)
    valid_a[0, :, 3] = False
    valid_b[1, 5:9, 7] = False
    valid_a[1, 5:9, 7] = False
    frame_mask = torch.ones(batch, frames, dtype=torch.bool)
    frame_mask[1, 12:] = False
    phase = (torch.arange(frames, dtype=torch.float32) % 8 / 8.0)[None].repeat(batch, 1)
    phase_valid = torch.ones(batch, frames, dtype=torch.bool)
    return {
        "pose_a": pose_a,
        "pose_b": pose_b,
        "valid_a": valid_a,
        "valid_b": valid_b,
        "delta_t": torch.full((batch, frames), 1.0 / 30.0),
        "phase": phase,
        "phase_valid": phase_valid,
        "frame_mask": frame_mask,
    }
