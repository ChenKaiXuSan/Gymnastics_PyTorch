"""Person-level self-supervised, baseline, and isolated triangulation evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import SkeletonSpec
from .trunk import extract_trunk_features

ABLATION_REGISTRY = {
    "A0": "face_only",
    "A1": "side_only",
    "A2": "canonical_arithmetic",
    "A3": "quality_mean",
    "A4": "learned_spatial",
    "A5": "learned_rotation_temporal",
    "A6": "rotation_aware_self_supervised",
    "A7": "learned_rom_peak",
    "A8": "learned_twist_rom_peak",
    "A9": "learned_twist_rom_peak_rate",
}


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
    diagnostic_status: Mapping[str, str] = field(default_factory=dict)


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


def _derivative(
    values: np.ndarray,
    timestamps: np.ndarray,
    order: int,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    result, times = (
        np.asarray(values, dtype=np.float64),
        np.asarray(timestamps, dtype=np.float64),
    )
    mask = (
        np.asarray(valid, dtype=bool)
        if valid is not None
        else np.ones(
            result.shape[:-1] if result.ndim == 3 else result.shape, dtype=bool
        )
    )
    for _ in range(order):
        if len(result) < 2:
            return np.zeros_like(result), np.zeros_like(mask, dtype=bool)
        result = np.diff(result, axis=0) / np.maximum(
            np.diff(times).reshape((-1,) + (1,) * (result.ndim - 1)), 1e-8
        )
        mask = mask[1:] & mask[:-1]
        result = np.where(mask[..., None] if result.ndim == 3 else mask, result, 0.0)
        times = 0.5 * (times[1:] + times[:-1])
    return result, mask


def _circular_rom(theta: np.ndarray, valid: np.ndarray) -> float:
    values, mask = np.asarray(theta, dtype=np.float64), np.asarray(valid, dtype=bool)
    runs: list[np.ndarray] = []
    start: int | None = None
    for index, is_valid in enumerate(mask):
        if is_valid and start is None:
            start = index
        elif not is_valid and start is not None:
            runs.append(values[start:index])
            start = None
    if start is not None:
        runs.append(values[start:])
    return float(max((np.ptp(np.unwrap(run)) for run in runs), default=0.0))


def _trunk(
    points: np.ndarray,
    valid: np.ndarray,
    timestamps: np.ndarray,
    skeleton: SkeletonSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(timestamps) < 2:
        dt: np.ndarray = np.ones(len(timestamps), dtype=np.float32)
    else:
        differences = np.diff(timestamps)
        if not np.isfinite(differences).all() or np.any(differences <= 0):
            raise ValueError("timestamps must be strictly increasing for trunk metrics")
        median_dt = float(np.median(differences))
        dt = np.r_[median_dt, differences].astype(np.float32)
    result = extract_trunk_features(
        torch.from_numpy(points.astype(np.float32)).unsqueeze(0),
        torch.from_numpy(valid).unsqueeze(0),
        skeleton,
        torch.from_numpy(dt[None]),
    )
    return (
        result.angle.squeeze(0).numpy(),
        result.omega.squeeze(0).numpy(),
        result.angle_valid.squeeze(0).numpy(),
        result.omega_valid.squeeze(0).numpy(),
    )


EXTERNAL_ALIGNMENT_MODES = ("root", "similarity", "procrustes")
DEFAULT_EXTERNAL_ALIGNMENT = "similarity"


def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares (Umeyama) similarity mapping ``src`` onto ``dst``."""
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    variance = float((src_c**2).sum())
    if variance < 1e-12:
        return 1.0, np.eye(3), dst_mean - src_mean
    u, singular, vt = np.linalg.svd(src_c.T @ dst_c)
    sign = float(np.sign(np.linalg.det(u @ vt)))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    scale = float(singular[0] + singular[1] + sign * singular[2]) / variance
    return scale, rotation, dst_mean - scale * (src_mean @ rotation)


def _align_external(
    candidate: np.ndarray,
    reference: np.ndarray,
    usable: np.ndarray,
    alignment: str,
) -> np.ndarray:
    """Remove the static world-frame and scale mismatch before measuring error.

    The fused keypoints live in the SAM3D world frame while the triangulated
    reference lives in the calibrated camera frame; the two differ by a fixed
    rotation and a per-person scale. ``root`` leaves that difference inside the
    error and makes the metric nearly blind to the fusion itself.
    """
    if alignment == "similarity":
        if usable.sum() < 3:
            return candidate
        scale, rotation, translation = _fit_similarity(
            candidate[usable], reference[usable]
        )
        return scale * (candidate @ rotation) + translation
    aligned = np.array(candidate, copy=True)
    for frame in range(candidate.shape[0]):
        frame_usable = usable[frame]
        if frame_usable.sum() < 3:
            continue
        scale, rotation, translation = _fit_similarity(
            candidate[frame][frame_usable], reference[frame][frame_usable]
        )
        aligned[frame] = scale * (candidate[frame] @ rotation) + translation
    return aligned


def _external_errors(
    candidate: np.ndarray,
    reference: np.ndarray,
    skeleton: SkeletonSpec,
    candidate_valid: np.ndarray | None = None,
    alignment: str = DEFAULT_EXTERNAL_ALIGNMENT,
) -> tuple[np.ndarray, np.ndarray]:
    if alignment not in EXTERNAL_ALIGNMENT_MODES:
        raise ValueError(
            f"alignment must be one of {EXTERNAL_ALIGNMENT_MODES}: {alignment}"
        )
    candidate, reference = (
        np.asarray(candidate, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )
    if candidate.shape != reference.shape:
        raise ValueError("candidate and external reference shapes must match")
    if alignment != "root":
        usable = np.isfinite(candidate).all(axis=-1) & np.isfinite(reference).all(axis=-1)
        if candidate_valid is not None:
            usable = usable & np.asarray(candidate_valid, dtype=bool)
        candidate = _align_external(candidate, reference, usable, alignment)
    hip_indices: list[int] = []
    for role_name in ("left_hip", "right_hip"):
        role = skeleton.role(role_name)
        if role.kind != "joint" or len(role.joints) != 1:
            raise ValueError(f"Skeleton role {role_name} must resolve exactly one joint")
        hip_indices.append(skeleton.joint_index(role.joints[0]))
    left_hip, right_hip = hip_indices
    root_a, root_b = (
        0.5 * (candidate[:, left_hip] + candidate[:, right_hip]),
        0.5 * (reference[:, left_hip] + reference[:, right_hip]),
    )
    errors = np.linalg.norm(
        (candidate - root_a[:, None]) - (reference - root_b[:, None]), axis=-1
    )
    valid = np.isfinite(candidate).all(axis=-1) & np.isfinite(reference).all(axis=-1)
    reference_roots = np.any(reference[:, left_hip] != 0, axis=-1) & np.any(
        reference[:, right_hip] != 0, axis=-1
    )
    candidate_roots = np.any(candidate[:, left_hip] != 0, axis=-1) & np.any(
        candidate[:, right_hip] != 0, axis=-1
    )
    valid &= (
        valid[:, left_hip]
        & valid[:, right_hip]
        & reference_roots
        & candidate_roots
    )[:, None]
    if candidate_valid is not None:
        candidate_valid = np.asarray(candidate_valid, dtype=bool)
        valid &= candidate_valid
        valid &= (
            candidate_valid[:, left_hip] & candidate_valid[:, right_hip]
        )[:, None]
    return errors, valid


def external_metrics_from_reference(
    candidate: np.ndarray,
    reference: np.ndarray,
    skeleton: SkeletonSpec,
    candidate_valid: np.ndarray | None = None,
    alignment: str = DEFAULT_EXTERNAL_ALIGNMENT,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Optional pseudo-GT metrics; this is intentionally isolated from training."""
    errors, valid = _external_errors(
        candidate, reference, skeleton, candidate_valid, alignment=alignment
    )
    values = errors[valid]
    summary = {
        name: (float(fn(values)) if len(values) else float("nan"))
        for name, fn in (
            ("mpjpe", np.mean),
            ("median", np.median),
            ("p95", lambda item: np.percentile(item, 95)),
        )
    }
    summary["matched_frames"] = float(valid.any(axis=1).sum())
    summary["matched_points"] = float(valid.sum())
    joints = [
        {
            "joint": joint,
            "valid_points": float(valid[:, joint].sum()),
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
        for joint in range(errors.shape[1])
    ]
    return summary, joints


def _pooled_self_metrics(
    sequences: Sequence[MethodSequence], skeleton: SkeletonSpec
) -> dict[str, float]:
    bone_samples: list[list[np.ndarray]] = [[] for _ in skeleton.bones]
    joint_jerk_samples: list[np.ndarray] = []
    trunk_jerk_samples: list[np.ndarray] = []
    cycle_roms: list[float] = []
    cycle_peaks: list[float] = []
    matched_roms: list[float] = []
    matched_peaks: list[float] = []
    reference_roms: list[float] = []
    reference_peaks: list[float] = []
    valid_frames = 0.0
    valid_points = 0.0
    diagnostic_weights: list[float] = []
    swap_errors: list[float] = []
    corruption_recoveries: list[float] = []

    for sequence in sequences:
        points = np.asarray(sequence.kpts_world, dtype=np.float32)
        timestamps = np.asarray(sequence.timestamps, dtype=np.float64)
        valid = _valid(sequence)
        if points.shape[0] != len(timestamps):
            raise ValueError(f"timestamps length does not match {sequence.trial_id}")
        valid_frames += float(valid.any(axis=1).sum())
        cycle_valid_points = float(valid.sum())
        valid_points += cycle_valid_points
        diagnostic_weights.append(cycle_valid_points)
        swap_errors.append(
            float(sequence.swap_error)
            if sequence.swap_error is not None
            else float("nan")
        )
        corruption_recoveries.append(
            float(sequence.corruption_recovery)
            if sequence.corruption_recovery is not None
            else float("nan")
        )

        for bone_index, (left, right) in enumerate(skeleton.bones):
            lengths = np.linalg.norm(points[:, left] - points[:, right], axis=-1)
            usable = valid[:, left] & valid[:, right] & np.isfinite(lengths)
            if usable.any():
                bone_samples[bone_index].append(lengths[usable])

        theta, omega, theta_valid, omega_valid = _trunk(
            points, valid, timestamps, skeleton
        )
        rom = _circular_rom(theta, theta_valid)
        peak = float(np.max(np.abs(omega[omega_valid]))) if omega_valid.any() else 0.0
        cycle_roms.append(rom)
        cycle_peaks.append(peak)
        jerk, jerk_valid = _derivative(points, timestamps, 3, valid)
        if jerk_valid.any():
            joint_jerk_samples.append(np.abs(jerk[jerk_valid]).reshape(-1))
        angular_jerk, angular_jerk_valid = _derivative(
            theta, timestamps, 3, theta_valid
        )
        if angular_jerk_valid.any():
            trunk_jerk_samples.append(np.abs(angular_jerk[angular_jerk_valid]))

        if sequence.reference_kpts is not None:
            reference = np.asarray(sequence.reference_kpts, dtype=np.float32)
            if reference.shape != points.shape:
                raise ValueError(f"reference_kpts shape does not match {sequence.trial_id}")
            ref_theta, ref_omega, ref_theta_valid, ref_omega_valid = _trunk(
                reference,
                np.isfinite(reference).all(axis=-1)
                & np.any(reference != 0, axis=-1),
                timestamps,
                skeleton,
            )
            matched_roms.append(rom)
            matched_peaks.append(peak)
            reference_roms.append(_circular_rom(ref_theta, ref_theta_valid))
            reference_peaks.append(
                float(np.max(np.abs(ref_omega[ref_omega_valid])))
                if ref_omega_valid.any()
                else 0.0
            )

    cvs: list[float] = []
    rigidity: list[float] = []
    for samples in bone_samples:
        if not samples:
            continue
        values = np.concatenate(samples)
        baseline = max(abs(float(np.median(values))), 1e-8)
        cvs.append(float(np.std(values) / baseline))
        rigidity.append(float(np.mean(np.abs(values - baseline) / baseline)))

    def pooled_mean(samples: list[np.ndarray]) -> float:
        return float(np.mean(np.concatenate(samples))) if samples else float("nan")

    def weighted_diagnostic(values: list[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        weights = np.asarray(diagnostic_weights, dtype=np.float64)
        usable = np.isfinite(array) & (weights > 0)
        return (
            float(np.average(array[usable], weights=weights[usable]))
            if usable.any()
            else float("nan")
        )

    rom = max(cycle_roms, default=0.0)
    peak = max(cycle_peaks, default=0.0)
    matched_rom = max(matched_roms, default=0.0)
    matched_peak = max(matched_peaks, default=0.0)
    reference_rom = max(reference_roms, default=0.0)
    reference_peak = max(reference_peaks, default=0.0)
    return {
        "valid_frames": valid_frames,
        "valid_points": valid_points,
        "bone_cv": float(np.mean(cvs)) if cvs else float("nan"),
        "rigidity": float(np.mean(rigidity)) if rigidity else float("nan"),
        "joint_jerk": pooled_mean(joint_jerk_samples),
        "trunk_angular_jerk": pooled_mean(trunk_jerk_samples),
        "rom_retention": (
            matched_rom / reference_rom if reference_rom > 1e-8 else float("nan")
        ),
        "peak_angular_velocity_retention": (
            matched_peak / reference_peak
            if reference_peak > 1e-8
            else float("nan")
        ),
        "swap_error": weighted_diagnostic(swap_errors),
        "fixed_corruption_recovery": weighted_diagnostic(corruption_recoveries),
        "theta_rom": rom,
        "peak_omega": peak,
    }


def evaluate_person_trials(
    person_id: str,
    sequences: Sequence[MethodSequence],
    skeleton: SkeletonSpec,
    references: Mapping[str, np.ndarray] | None = None,
    alignment: str = DEFAULT_EXTERNAL_ALIGNMENT,
) -> EvaluationReport:
    grouped: dict[str, list[MethodSequence]] = {}
    for sequence in sequences:
        grouped.setdefault(sequence.method, []).append(sequence)
    rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    for method, cycles in grouped.items():
        metrics = _pooled_self_metrics(cycles, skeleton)
        row: dict[str, Any] = {
            "person_id": str(person_id),
            "method": method,
            "cycles": len(cycles),
            **metrics,
        }
        for diagnostic in ("swap_error", "fixed_corruption_recovery"):
            statuses = {
                cycle.diagnostic_status.get(
                    diagnostic, "unavailable_missing_diagnostic"
                )
                for cycle in cycles
            }
            row[f"{diagnostic}_availability"] = (
                statuses.pop() if len(statuses) == 1 else "mixed"
            )
        legacy = any(
            cycle.diagnostic_status.get("swap_error") == "unsupported_legacy_output"
            for cycle in cycles
        )
        for metric in ("rom_retention", "peak_angular_velocity_retention"):
            row[f"{metric}_availability"] = (
                "unsupported_legacy_output"
                if legacy
                else (
                    "measured"
                    if np.isfinite(row[metric])
                    else "unavailable_missing_reference"
                )
            )
        if references is not None:
            external_errors: list[np.ndarray] = []
            external_masks: list[np.ndarray] = []
            missing_trials = 0
            for cycle in cycles:
                if cycle.trial_id not in references:
                    missing_trials += 1
                    continue
                reference = np.asarray(references[cycle.trial_id])
                if reference.shape != cycle.kpts_world.shape:
                    raise ValueError(
                        f"external reference length/shape mismatch for {cycle.trial_id}"
                    )
                errors, mask = _external_errors(
                    cycle.kpts_world,
                    reference,
                    skeleton,
                    _valid(cycle),
                    alignment=alignment,
                )
                external_errors.append(errors)
                external_masks.append(mask)
            row["external_matched_trials"] = len(external_errors)
            row["external_missing_trials"] = missing_trials
            if external_errors:
                errors = np.concatenate(external_errors, axis=0)
                mask = np.concatenate(external_masks, axis=0)
                values = errors[mask]
                for key, function in (
                    ("mpjpe", np.mean),
                    ("median", np.median),
                    ("p95", lambda item: np.percentile(item, 95)),
                ):
                    row[key] = float(function(values)) if len(values) else float("nan")
                row["external_matched_frames"] = float(mask.any(axis=1).sum())
                row["external_valid_points"] = float(mask.sum())
                for joint in range(cycles[0].kpts_world.shape[1]):
                    joint_values = errors[:, joint][mask[:, joint]]
                    joint_rows.append(
                        {
                            "person_id": str(person_id),
                            "method": method,
                            "joint": joint,
                            "valid_points": float(len(joint_values)),
                            **{
                                key: (
                                    float(function(joint_values))
                                    if len(joint_values)
                                    else float("nan")
                                )
                                for key, function in (
                                    ("mpjpe", np.mean),
                                    ("median", np.median),
                                    ("p95", lambda item: np.percentile(item, 95)),
                                )
                            },
                        }
                    )
            else:
                row.update(
                    {
                        "mpjpe": float("nan"),
                        "median": float("nan"),
                        "p95": float("nan"),
                        "external_matched_frames": 0.0,
                        "external_valid_points": 0.0,
                    }
                )
        rows.append(row)
    return EvaluationReport(rows, joint_rows)


def load_triangulated_references(
    triangulated_root: str | Path, person_id: str, sequences: Sequence[MethodSequence]
) -> dict[str, np.ndarray]:
    """Isolated adapter for active triangulation outputs, matched by face/side maps."""
    from gymnastics.analysis.compare_fused_triangulated import load_triangulated_sequence

    by_trial: dict[str, MethodSequence] = {}
    for sequence in sequences:
        if sequence.method in {"A6", "rotation_aware_self_supervised"}:
            by_trial[sequence.trial_id] = sequence
        else:
            by_trial.setdefault(sequence.trial_id, sequence)
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
            continue
        reference = np.full_like(sequence.kpts_world, np.nan)
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
        **{name: "absent" for name in ABLATION_REGISTRY},
    }
    for path in sorted(
        (Path(inference_root) / f"person_{person_id}").glob("*/fused_sequence.npz")
    ):
        with np.load(path, allow_pickle=False) as data:
            metadata = (
                json.loads(str(data["metadata"].item()))
                if "metadata" in data.files
                else {}
            )
            if not isinstance(metadata, dict):
                metadata = {}
            diagnostics = (
                json.loads(str(data["diagnostics"].item()))
                if "diagnostics" in data.files
                else {}
            )
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            learned_ablation = str(metadata.get("ablation", "A6"))
            if learned_ablation not in {"A4", "A5", "A6", "A7", "A8", "A9"}:
                learned_ablation = "A6"
            frames = np.array(data["frame_valid"])
            joints = np.array(data["joint_valid"]) if "joint_valid" in data else None
            maps = np.array(data["face_map"])
            side_maps = np.array(data["side_map"])
            timestamps = (
                np.array(data["timestamps"], dtype=np.float64)
                if "timestamps" in data.files
                else np.arange(len(data["kpts_world"]), dtype=np.float64)
                / float(data["fps"].item() if "fps" in data.files else 60.0)
            )
            reference = (
                np.array(data["reference_kpts_world"])
                if "reference_kpts_world" in data.files
                else (
                    np.array(data["kpts_base_world"])
                    if "kpts_base_world" in data.files
                    else None
                )
            )
            swap = (
                float(data["swap_error"].item()) if "swap_error" in data.files else None
            )
            recovery = (
                float(data["fixed_corruption_recovery"].item())
                if "fixed_corruption_recovery" in data.files
                else None
            )
            for method, field in (
                ("face_only", "kpts_face_world"),
                ("side_only", "kpts_side_world"),
                ("canonical_arithmetic", "kpts_arithmetic_world"),
                ("quality_mean", "kpts_base_world"),
                ("rotation_aware_self_supervised", "kpts_world"),
            ):
                if field not in data.files:
                    continue
                reported_method = (
                    learned_ablation
                    if method == "rotation_aware_self_supervised"
                    else method
                )
                diagnostic = diagnostics.get(
                    reported_method, diagnostics.get(method, {})
                )
                if not isinstance(diagnostic, dict):
                    diagnostic = {}
                swap = diagnostic.get("swap_error")
                recovery = diagnostic.get("fixed_corruption_recovery")
                finite_swap = isinstance(swap, (int, float)) and np.isfinite(
                    float(swap)
                )
                status_map = {
                    "swap_error": (
                        "measured"
                        if finite_swap
                        else "unavailable_no_common_valid_points"
                    ),
                    "fixed_corruption_recovery": str(
                        diagnostic.get(
                            "fixed_corruption_recovery_status",
                            "unsupported",
                        )
                    ),
                }
                found.append(
                    MethodSequence(
                        reported_method,
                        np.array(data[field]),
                        timestamps,
                        frames,
                        joints,
                        path.parent.name,
                        maps,
                        side_maps,
                        reference,
                        float(swap) if swap is not None else None,
                        float(recovery) if recovery is not None else None,
                        status_map,
                    )
                )
                status[reported_method] = "available"
                if (
                    method == "rotation_aware_self_supervised"
                    and learned_ablation == "A6"
                ):
                    status[method] = "available"
                for ablation, mapped_method in ABLATION_REGISTRY.items():
                    if mapped_method == method:
                        if ablation in {"A0", "A1", "A2", "A3"}:
                            found.append(
                                MethodSequence(
                                    ablation,
                                    np.array(data[field]),
                                    timestamps,
                                    frames,
                                    joints,
                                    path.parent.name,
                                    maps,
                                    side_maps,
                                    reference,
                                    swap,
                                    recovery,
                                    status_map,
                                )
                            )
                            status[ablation] = "available"
    for method_root in sorted(
        Path(old_fuse_root).glob("*/person_" + str(person_id) + "/fused_sequence.npz")
    ):
        method = method_root.parents[1].name
        with np.load(method_root, allow_pickle=True) as data:
            points = np.array(data["kpts_world"])
            face_map, side_map = np.array(data["face_map"]), np.array(data["side_map"])
            fps = float(data["fps"].item()) if "fps" in data.files else 60.0
            timestamps = np.arange(len(points), dtype=np.float64) / fps
            pair_index = {
                (int(face), int(side)): index
                for index, (face, side) in enumerate(zip(face_map, side_map))
            }
            templates = [
                sequence
                for sequence in found
                if sequence.method
                in {"A4", "A5", "A6", "A7", "A8", "A9", "rotation_aware_self_supervised"}
                and sequence.face_map is not None
                and sequence.side_map is not None
            ]
            sliced = 0
            for template in templates:
                if template.face_map is None or template.side_map is None:
                    continue
                template_face_map = np.asarray(template.face_map)
                template_side_map = np.asarray(template.side_map)
                indices = [
                    pair_index.get((int(face), int(side)))
                    for face, side in zip(template_face_map, template_side_map)
                ]
                if any(index is None for index in indices):
                    continue
                selection = np.asarray(indices, dtype=np.int64)
                found.append(
                    MethodSequence(
                        method,
                        points[selection],
                        timestamps[selection],
                        trial_id=template.trial_id,
                        face_map=template_face_map,
                        side_map=template_side_map,
                        diagnostic_status={
                            "swap_error": "unsupported_legacy_output",
                            "fixed_corruption_recovery": "unsupported_legacy_output",
                        },
                    )
                )
                sliced += 1
            if not sliced:
                found.append(
                    MethodSequence(
                        method,
                        points,
                        timestamps,
                        trial_id="full_sequence",
                        face_map=face_map,
                        side_map=side_map,
                        diagnostic_status={
                            "swap_error": "unsupported_legacy_output",
                            "fixed_corruption_recovery": "unsupported_legacy_output",
                        },
                    )
                )
        status[method] = "available"
    return found, status
