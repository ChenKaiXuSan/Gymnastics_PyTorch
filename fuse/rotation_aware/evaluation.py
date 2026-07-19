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


def _external_errors(
    candidate: np.ndarray,
    reference: np.ndarray,
    candidate_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    reference_roots = np.any(reference[:, 9] != 0, axis=-1) & np.any(
        reference[:, 10] != 0, axis=-1
    )
    candidate_roots = np.any(candidate[:, 9] != 0, axis=-1) & np.any(
        candidate[:, 10] != 0, axis=-1
    )
    valid &= (valid[:, 9] & valid[:, 10] & reference_roots & candidate_roots)[:, None]
    if candidate_valid is not None:
        candidate_valid = np.asarray(candidate_valid, dtype=bool)
        valid &= candidate_valid
        valid &= (candidate_valid[:, 9] & candidate_valid[:, 10])[:, None]
    return errors, valid


def external_metrics_from_reference(
    candidate: np.ndarray,
    reference: np.ndarray,
    candidate_valid: np.ndarray | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Optional pseudo-GT metrics; this is intentionally isolated from training."""
    errors, valid = _external_errors(candidate, reference, candidate_valid)
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
    theta, omega, theta_valid, omega_valid = _trunk(points, valid, timestamps, skeleton)
    rom, peak = (
        _circular_rom(theta, theta_valid),
        float(np.max(np.abs(omega[omega_valid]))) if omega_valid.any() else 0.0,
    )
    rom_retention = peak_retention = float("nan")
    if sequence.reference_kpts is not None:
        reference = np.asarray(sequence.reference_kpts, dtype=np.float32)
        if reference.shape != points.shape:
            raise ValueError(f"reference_kpts shape does not match {sequence.trial_id}")
        ref_theta, ref_omega, ref_theta_valid, ref_omega_valid = _trunk(
            reference,
            np.isfinite(reference).all(axis=-1) & np.any(reference != 0, axis=-1),
            timestamps,
            skeleton,
        )
        ref_rom, ref_peak = (
            _circular_rom(ref_theta, ref_theta_valid),
            float(np.max(np.abs(ref_omega[ref_omega_valid])))
            if ref_omega_valid.any()
            else 0.0,
        )
        rom_retention, peak_retention = (
            rom / ref_rom if ref_rom > 1e-8 else float("nan"),
            peak / ref_peak if ref_peak > 1e-8 else float("nan"),
        )
    jerk, jerk_valid = _derivative(points, timestamps, 3, valid)
    angular_jerk, angular_jerk_valid = _derivative(theta, timestamps, 3, theta_valid)
    return {
        "valid_frames": float(valid.any(axis=1).sum()),
        "valid_points": float(valid.sum()),
        "bone_cv": float(np.mean(cvs)) if cvs else float("nan"),
        "rigidity": float(np.mean(rigidity)) if rigidity else float("nan"),
        "joint_jerk": float(np.mean(np.abs(jerk[jerk_valid])))
        if jerk_valid.any()
        else float("nan"),
        "trunk_angular_jerk": float(np.mean(np.abs(angular_jerk[angular_jerk_valid])))
        if angular_jerk_valid.any()
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
        weights = np.asarray(
            [item["valid_points"] for item in metrics], dtype=np.float64
        )
        for key in metrics[0]:
            values = np.asarray([item[key] for item in metrics], dtype=np.float64)
            usable = np.isfinite(values) & (weights > 0)
            if key in {"valid_frames", "valid_points"}:
                row[key] = float(values.sum())
            elif usable.any():
                row[key] = float(np.average(values[usable], weights=weights[usable]))
            else:
                row[key] = float("nan")
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
                    cycle.kpts_world, reference, _valid(cycle)
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
    from analysis.compare_fused_triangulated import load_triangulated_sequence

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
            if learned_ablation not in {"A4", "A5", "A6"}:
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
                in {"A4", "A5", "A6", "rotation_aware_self_supervised"}
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
