"""Person-level self-supervised, baseline, and isolated triangulation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import SkeletonSpec
from .trunk import extract_trunk_features


@dataclass(frozen=True)
class MethodSequence:
    method: str
    kpts_world: np.ndarray
    timestamps: np.ndarray
    frame_valid: np.ndarray | None = None
    joint_valid: np.ndarray | None = None
    trial_id: str = "cycle_000"
    face_map: np.ndarray | None = None
    side_map: np.ndarray | None = None
    reference_kpts: np.ndarray | None = None
    swap_error: float | None = None
    corruption_recovery: float | None = None


@dataclass(frozen=True)
class EvaluationReport:
    person_metrics: list[dict[str, Any]]
    joint_metrics: list[dict[str, Any]]


def _valid(sequence: MethodSequence) -> np.ndarray:
    points = np.asarray(sequence.kpts_world)
    valid = np.isfinite(points).all(axis=-1) & np.any(points != 0, axis=-1)
    if sequence.frame_valid is not None:
        valid &= np.asarray(sequence.frame_valid, dtype=bool)[:, None]
    if sequence.joint_valid is not None:
        valid &= np.asarray(sequence.joint_valid, dtype=bool)
    return valid


def _derivative(values: np.ndarray, timestamps: np.ndarray, order: int) -> np.ndarray:
    result, times = (
        np.asarray(values, dtype=np.float64),
        np.asarray(timestamps, dtype=np.float64),
    )
    for _ in range(order):
        if len(result) < 2:
            return np.zeros_like(result)
        result = np.diff(result, axis=0) / np.maximum(
            np.diff(times).reshape((-1,) + (1,) * (result.ndim - 1)), 1e-8
        )
        times = 0.5 * (times[1:] + times[:-1])
    return result


def _trunk(
    points: np.ndarray,
    valid: np.ndarray,
    timestamps: np.ndarray,
    skeleton: SkeletonSpec,
) -> tuple[np.ndarray, np.ndarray]:
    median_dt = float(np.median(np.diff(timestamps)))
    dt = np.r_[1.0 / max(median_dt, 1e-8), np.diff(timestamps)].astype(np.float32)
    result = extract_trunk_features(
        torch.from_numpy(points.astype(np.float32)).unsqueeze(0),
        torch.from_numpy(valid).unsqueeze(0),
        skeleton,
        torch.from_numpy(dt[None]),
    )
    return result.angle.squeeze(0).numpy(), result.omega.squeeze(0).numpy()


def external_metrics_from_reference(
    candidate: np.ndarray,
    reference: np.ndarray,
    candidate_valid: np.ndarray | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Optional pseudo-GT metrics; this is intentionally isolated from training."""
    candidate, reference = (
        np.asarray(candidate, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )
    if candidate.shape != reference.shape:
        raise ValueError("candidate and external reference shapes must match")
    root_a, root_b = (
        0.5 * (candidate[:, 9] + candidate[:, 10]),
        0.5 * (reference[:, 9] + reference[:, 10]),
    )
    errors = np.linalg.norm(
        (candidate - root_a[:, None]) - (reference - root_b[:, None]), axis=-1
    )
    valid = np.isfinite(candidate).all(axis=-1) & np.isfinite(reference).all(axis=-1)
    if candidate_valid is not None:
        valid &= np.asarray(candidate_valid, dtype=bool)
    values = errors[valid]
    summary = {
        name: (float(fn(values)) if len(values) else float("nan"))
        for name, fn in (
            ("mpjpe", np.mean),
            ("median", np.median),
            ("p95", lambda item: np.percentile(item, 95)),
        )
    }
    joints = [
        {
            "joint": joint,
            **{
                name: (
                    float(fn(errors[:, joint][valid[:, joint]]))
                    if valid[:, joint].any()
                    else float("nan")
                )
                for name, fn in (
                    ("mpjpe", np.mean),
                    ("median", np.median),
                    ("p95", lambda item: np.percentile(item, 95)),
                )
            },
        }
        for joint in range(candidate.shape[1])
    ]
    return summary, joints


def _self_metrics(sequence: MethodSequence, skeleton: SkeletonSpec) -> dict[str, float]:
    points, timestamps, valid = (
        np.asarray(sequence.kpts_world, dtype=np.float32),
        np.asarray(sequence.timestamps, dtype=np.float64),
        _valid(sequence),
    )
    if points.shape[0] != len(timestamps):
        raise ValueError(f"timestamps length does not match {sequence.trial_id}")
    cvs, rigidity = [], []
    for left, right in skeleton.bones:
        lengths = np.linalg.norm(points[:, left] - points[:, right], axis=-1)
        values = lengths[valid[:, left] & valid[:, right]]
        if len(values):
            baseline = max(abs(float(np.median(values))), 1e-8)
            cvs.append(float(np.std(values) / baseline))
            rigidity.append(float(np.mean(np.abs(values - baseline) / baseline)))
    theta, omega = _trunk(points, valid, timestamps, skeleton)
    rom, peak = (
        float(np.ptp(theta)),
        float(np.max(np.abs(omega))) if len(omega) else 0.0,
    )
    rom_retention = peak_retention = float("nan")
    if sequence.reference_kpts is not None:
        reference = np.asarray(sequence.reference_kpts, dtype=np.float32)
        if reference.shape != points.shape:
            raise ValueError(f"reference_kpts shape does not match {sequence.trial_id}")
        ref_theta, ref_omega = _trunk(
            reference,
            np.isfinite(reference).all(axis=-1) & np.any(reference != 0, axis=-1),
            timestamps,
            skeleton,
        )
        ref_rom, ref_peak = (
            float(np.ptp(ref_theta)),
            float(np.max(np.abs(ref_omega))) if len(ref_omega) else 0.0,
        )
        rom_retention, peak_retention = (
            rom / ref_rom if ref_rom > 1e-8 else float("nan"),
            peak / ref_peak if ref_peak > 1e-8 else float("nan"),
        )
    jerk, angular_jerk = (
        _derivative(points, timestamps, 3),
        _derivative(theta, timestamps, 3),
    )
    return {
        "bone_cv": float(np.mean(cvs)) if cvs else float("nan"),
        "rigidity": float(np.mean(rigidity)) if rigidity else float("nan"),
        "joint_jerk": float(np.mean(np.abs(jerk))) if jerk.size else float("nan"),
        "trunk_angular_jerk": float(np.mean(np.abs(angular_jerk)))
        if angular_jerk.size
        else float("nan"),
        "rom_retention": rom_retention,
        "peak_angular_velocity_retention": peak_retention,
        "swap_error": float(sequence.swap_error)
        if sequence.swap_error is not None
        else float("nan"),
        "fixed_corruption_recovery": float(sequence.corruption_recovery)
        if sequence.corruption_recovery is not None
        else float("nan"),
        "theta_rom": rom,
        "peak_omega": peak,
    }


def evaluate_person_trials(
    person_id: str,
    sequences: Sequence[MethodSequence],
    skeleton: SkeletonSpec,
    references: Mapping[str, np.ndarray] | None = None,
) -> EvaluationReport:
    grouped: dict[str, list[MethodSequence]] = {}
    for sequence in sequences:
        grouped.setdefault(sequence.method, []).append(sequence)
    rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    for method, cycles in grouped.items():
        metrics = [_self_metrics(cycle, skeleton) for cycle in cycles]
        row: dict[str, Any] = {
            "person_id": str(person_id),
            "method": method,
            "cycles": len(cycles),
        }
        for key in metrics[0]:
            row[key] = (
                float(np.nanmean([item[key] for item in metrics]))
                if np.isfinite([item[key] for item in metrics]).any()
                else float("nan")
            )
        if references is not None:
            external = []
            for cycle in cycles:
                if cycle.trial_id not in references:
                    raise ValueError(f"missing external reference for {cycle.trial_id}")
                reference = np.asarray(references[cycle.trial_id])
                if reference.shape != cycle.kpts_world.shape:
                    raise ValueError(
                        f"external reference length/shape mismatch for {cycle.trial_id}"
                    )
                external.append(
                    external_metrics_from_reference(
                        cycle.kpts_world, reference, _valid(cycle)
                    )
                )
            for key in external[0][0]:
                row[key] = float(np.nanmean([item[0][key] for item in external]))
            for joint in range(cycles[0].kpts_world.shape[1]):
                joint_rows.append(
                    {
                        "person_id": str(person_id),
                        "method": method,
                        "joint": joint,
                        **{
                            key: float(
                                np.nanmean([item[1][joint][key] for item in external])
                            )
                            for key in ("mpjpe", "median", "p95")
                        },
                    }
                )
        rows.append(row)
    return EvaluationReport(rows, joint_rows)


def load_triangulated_references(
    triangulated_root: str | Path, person_id: str, sequences: Sequence[MethodSequence]
) -> dict[str, np.ndarray]:
    """Isolated adapter for active triangulation outputs, matched by face/side maps."""
    from analysis.compare_fused_triangulated import load_triangulated_sequence

    by_trial = {sequence.trial_id: sequence for sequence in sequences}
    references: dict[str, np.ndarray] = {}
    for root in sorted(
        (Path(triangulated_root) / f"person_{person_id}").glob("cycle_*")
    ):
        trial_id = root.name
        if trial_id not in by_trial:
            continue
        sequence = by_trial[trial_id]
        if sequence.face_map is None or sequence.side_map is None:
            raise ValueError(
                f"frame maps are required for triangulation matching: {trial_id}"
            )
        joints, pairs = load_triangulated_sequence(root)
        index = {
            (int(face), int(side)): i
            for i, (face, side) in enumerate(zip(sequence.face_map, sequence.side_map))
        }
        matched = [
            (index[pair], tri) for tri, pair in enumerate(pairs) if pair in index
        ]
        if not matched:
            raise ValueError(f"no triangulation frame pairs match {trial_id}")
        if len(matched) != len(sequence.kpts_world):
            raise ValueError(f"triangulation does not cover every frame of {trial_id}")
        reference = np.empty_like(sequence.kpts_world)
        for fused, tri in matched:
            reference[fused] = joints[tri]
        references[trial_id] = reference
    return references


def discover_method_sequences(
    inference_root: str | Path, old_fuse_root: str | Path, person_id: str
) -> tuple[list[MethodSequence], dict[str, str]]:
    """Discover available new/old compact outputs without touching old method files."""
    found: list[MethodSequence] = []
    status: dict[str, str] = {
        "face_only": "absent",
        "side_only": "absent",
        "canonical_arithmetic": "absent",
        "quality_mean": "absent",
        "rotation_aware_self_supervised": "absent",
        **{f"A{index}": "absent" for index in range(7)},
    }
    for path in sorted(
        (Path(inference_root) / f"person_{person_id}").glob("*/fused_sequence.npz")
    ):
        with np.load(path, allow_pickle=False) as data:
            frames = np.array(data["frame_valid"])
            joints = np.array(data["joint_valid"]) if "joint_valid" in data else None
            maps = np.array(data["face_map"])
            side_maps = np.array(data["side_map"])
            for method, field in (
                ("face_only", "kpts_face_world"),
                ("side_only", "kpts_side_world"),
                ("canonical_arithmetic", "kpts_arithmetic_world"),
                ("quality_mean", "kpts_base_world"),
                ("rotation_aware_self_supervised", "kpts_world"),
            ):
                if field not in data.files:
                    continue
                found.append(
                    MethodSequence(
                        method,
                        np.array(data[field]),
                        np.arange(len(data[field]), dtype=np.float64),
                        frames,
                        joints,
                        path.parent.name,
                        maps,
                        side_maps,
                    )
                )
                status[method] = "available"
    for method_root in sorted(
        Path(old_fuse_root).glob("*/person_" + str(person_id) + "/fused_sequence.npz")
    ):
        method = method_root.parents[1].name
        with np.load(method_root, allow_pickle=True) as data:
            found.append(
                MethodSequence(
                    method,
                    np.array(data["kpts_world"]),
                    np.arange(len(data["kpts_world"]), dtype=np.float64),
                    trial_id="full_sequence",
                    face_map=np.array(data["face_map"]),
                    side_map=np.array(data["side_map"]),
                )
            )
        status[method] = "available"
    return found, status
