"""Prespecified motion feature contracts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Mapping

import numpy as np
import pandas as pd

from gymnastics.common.skeletons import MHR70_INDEX

from .cohorts import sha256_file
from .joints import MAJOR_JOINT_INDICES
from .preprocess import (
    align_rotation_direction,
    normalized_cycle_positions,
    phase_normalize,
)
from .qc import evaluate_cycle_qc, interpolate_short_gaps


LEFT_HIP = MHR70_INDEX["left-hip"]
RIGHT_HIP = MHR70_INDEX["right-hip"]
LEFT_SHOULDER = MHR70_INDEX["left-shoulder"]
RIGHT_SHOULDER = MHR70_INDEX["right-shoulder"]
RIGHT_WRIST = MHR70_INDEX["right-wrist"]
LEFT_WRIST = MHR70_INDEX["left-wrist"]
CORE_JOINTS = tuple(
    sorted(
        {
            LEFT_HIP,
            RIGHT_HIP,
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            RIGHT_WRIST,
            LEFT_WRIST,
        }
    )
)
CORE_OUTCOMES = (
    "trunk_axial_rotation_rom",
    "angular_speed_p95",
    "peak_rotation_phase",
    "trunk_tilt_p95",
    "wrist_lead_p95",
    "cycle_duration",
    "log_dimensionless_angular_jerk",
    "whole_body_repeatability",
)


@dataclass(frozen=True)
class CycleFeatures:
    """Tidy feature record for one repeated movement cycle."""

    person_id: str
    cohort: str
    outer_fold: int
    cycle_id: str
    cycle_index: int
    normalized_cycle_position: float
    pose_source: str
    eligible: bool
    exclusion_reasons: tuple[str, ...]
    values: Mapping[str, float | None]


def axial_rotation_from_pose(kpts_body: np.ndarray) -> np.ndarray:
    """Derive pelvis-to-shoulder axial rotation in the canonical x-z plane."""
    points = np.asarray(kpts_body, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (70, 3):
        raise ValueError("axial rotation requires pose shaped (frames, 70, 3)")
    shoulder = (
        points[:, RIGHT_SHOULDER] - points[:, LEFT_SHOULDER]
    )[:, [0, 2]]
    pelvis = (points[:, RIGHT_HIP] - points[:, LEFT_HIP])[:, [0, 2]]
    shoulder_norm = np.linalg.norm(shoulder, axis=1)
    pelvis_norm = np.linalg.norm(pelvis, axis=1)
    if (
        np.any(~np.isfinite(shoulder))
        or np.any(~np.isfinite(pelvis))
        or np.any((shoulder_norm <= 1e-12) | (pelvis_norm <= 1e-12))
    ):
        raise ValueError("shoulder or pelvis horizontal axis is degenerate")
    shoulder_angle = np.arctan2(shoulder[:, 1], shoulder[:, 0])
    pelvis_angle = np.arctan2(pelvis[:, 1], pelvis[:, 0])
    return np.unwrap(pelvis_angle - shoulder_angle)


def match_pose_frames(
    source_pose: np.ndarray,
    source_face_map: np.ndarray,
    target_face_map: np.ndarray,
) -> np.ndarray:
    """Select full-sequence pose frames matching one cycle's face IDs."""
    pose = np.asarray(source_pose)
    source = np.asarray(source_face_map)
    target = np.asarray(target_face_map)
    if pose.ndim != 3 or pose.shape[1:] != (70, 3):
        raise ValueError("source pose must have shape (frames, 70, 3)")
    if source.ndim != 1 or len(source) != len(pose) or target.ndim != 1:
        raise ValueError("face maps must be 1D and match source pose length")
    unique, counts = np.unique(source, return_counts=True)
    if np.any(counts > 1):
        raise ValueError("source face map contains duplicate frame IDs")
    lookup = {
        int(frame_id): int(index)
        for index, frame_id in enumerate(source.tolist())
    }
    missing = sorted(
        {int(frame_id) for frame_id in target.tolist()} - set(lookup)
    )
    if missing:
        raise ValueError(
            f"target face map contains missing frame IDs: {missing[:5]}"
        )
    return pose[
        np.asarray([lookup[int(frame_id)] for frame_id in target], dtype=int)
    ]


def angular_jerk(
    theta: np.ndarray,
    timestamps: np.ndarray,
    *,
    epsilon: float = 1e-12,
    minimum_rom: float = 1e-6,
) -> float:
    """Return log dimensionless angular jerk for one complete cycle."""
    angle = np.unwrap(np.asarray(theta, dtype=np.float64))
    time = np.asarray(timestamps, dtype=np.float64)
    if angle.ndim != 1 or time.shape != angle.shape or len(angle) < 7:
        raise ValueError("angular jerk requires matching 1D trajectories")
    if not np.all(np.isfinite(angle)) or not np.all(np.diff(time) > 0):
        raise ValueError("angular jerk inputs must be finite and increasing")
    amplitude = float(np.quantile(angle, 0.95) - np.quantile(angle, 0.05))
    if amplitude <= minimum_rom:
        raise ValueError("negligible angular ROM")
    duration = float(time[-1] - time[0])
    velocity = np.gradient(angle, time)
    acceleration = np.gradient(velocity, time)
    jerk = np.gradient(acceleration, time)
    integral = float(np.trapezoid(jerk**2, time))
    dimensionless = np.sqrt(duration**5 * integral / amplitude**2)
    return float(np.log(dimensionless + epsilon))


def compute_core_scalars(
    theta: np.ndarray,
    omega: np.ndarray,
    timestamps: np.ndarray,
    kpts_body: np.ndarray,
    *,
    direction_sign: int,
) -> dict[str, float | None]:
    """Compute the seven per-cycle core scalars before repeatability."""
    angle = np.unwrap(np.asarray(theta, dtype=np.float64))
    angular_velocity = np.asarray(omega, dtype=np.float64)
    time = np.asarray(timestamps, dtype=np.float64)
    points = np.asarray(kpts_body, dtype=np.float64)
    if (
        angle.ndim != 1
        or angular_velocity.shape != angle.shape
        or time.shape != angle.shape
        or points.shape != (len(angle), 70, 3)
    ):
        raise ValueError("core feature arrays have incompatible shapes")
    if direction_sign not in {-1, 1}:
        raise ValueError("direction sign must be -1 or 1")

    tilt = trunk_tilt_trajectory(points)
    wrist_angle = wrist_lead_trajectory(points, direction_sign)

    try:
        jerk_value: float | None = angular_jerk(angle, time)
    except ValueError:
        jerk_value = None
    return {
        "trunk_axial_rotation_rom": float(
            np.quantile(angle, 0.95) - np.quantile(angle, 0.05)
        ),
        "angular_speed_p95": float(
            np.quantile(np.abs(angular_velocity), 0.95)
        ),
        "peak_rotation_phase": peak_rotation_phase(angle),
        "trunk_tilt_p95": float(np.quantile(np.abs(tilt), 0.95)),
        "wrist_lead_p95": float(
            np.quantile(np.abs(wrist_angle), 0.95)
        ),
        "cycle_duration": float(time[-1] - time[0]),
        "log_dimensionless_angular_jerk": jerk_value,
    }


def leave_one_cycle_out_repeatability(
    trajectories: np.ndarray,
) -> np.ndarray:
    """RMS error to the median of all other phase-normalized cycles."""
    values = np.asarray(trajectories, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] < 2:
        raise ValueError("repeatability requires at least two pose cycles")
    errors = np.empty(values.shape[0], dtype=np.float64)
    for index in range(values.shape[0]):
        template = np.median(
            np.delete(values, index, axis=0),
            axis=0,
        )
        errors[index] = np.sqrt(np.mean((values[index] - template) ** 2))
    return errors


def peak_rotation_phase(theta: np.ndarray) -> float:
    """Return normalized phase of maximum absolute median-centred rotation."""
    angle = np.asarray(theta, dtype=np.float64)
    if angle.ndim != 1 or len(angle) < 2 or not np.all(np.isfinite(angle)):
        raise ValueError("peak phase requires a finite 1D trajectory")
    peak_index = int(np.argmax(np.abs(angle - np.median(angle))))
    return peak_index / float(len(angle) - 1)


def trunk_tilt_trajectory(kpts_body: np.ndarray) -> np.ndarray:
    """Unsigned trunk deviation from the body-frame vertical axis."""
    points = np.asarray(kpts_body, dtype=np.float64)
    hip_center = 0.5 * (
        points[:, LEFT_HIP] + points[:, RIGHT_HIP]
    )
    shoulder_center = 0.5 * (
        points[:, LEFT_SHOULDER] + points[:, RIGHT_SHOULDER]
    )
    trunk = shoulder_center - hip_center
    norm = np.linalg.norm(trunk, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError("trunk vector is degenerate")
    return np.arccos(np.clip(trunk[:, 1] / norm, -1.0, 1.0))


def wrist_lead_trajectory(
    kpts_body: np.ndarray,
    direction_sign: int,
) -> np.ndarray:
    """Lagging-wrist wrapping angle relative to the pelvis lateral axis.

    The existing analysis convention maps positive/CCW rotation to the right
    lagging wrist and negative/CW rotation to the left lagging wrist. Indices
    come from the canonical MHR70 names (right wrist 41, left wrist 62).
    """
    points = np.asarray(kpts_body, dtype=np.float64)
    wrist_index = RIGHT_WRIST if direction_sign > 0 else LEFT_WRIST
    hip_center = 0.5 * (
        points[:, LEFT_HIP] + points[:, RIGHT_HIP]
    )
    wrist = points[:, wrist_index] - hip_center
    hip_axis = points[:, RIGHT_HIP] - points[:, LEFT_HIP]
    wrist_horizontal = wrist[:, [0, 2]]
    hip_horizontal = hip_axis[:, [0, 2]]
    wrist_norm = np.linalg.norm(wrist_horizontal, axis=1)
    hip_norm = np.linalg.norm(hip_horizontal, axis=1)
    if np.any((wrist_norm <= 1e-12) | (hip_norm <= 1e-12)):
        raise ValueError("wrist or hip horizontal vector is degenerate")
    cosine = np.sum(wrist_horizontal * hip_horizontal, axis=1) / (
        wrist_norm * hip_norm
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def extract_publication_features(
    publication_root: str | Path,
    output_root: str | Path,
    *,
    people: set[str] | None = None,
    phase_points: int = 101,
    minimum_person_cycles: int = 4,
    pose_source: str = "fused",
    deterministic_root: str | Path | None = None,
) -> dict[str, int]:
    """Extract tidy cycle/person features from an audited OOF publication."""
    if pose_source not in {"fused", "face", "side", "deterministic"}:
        raise ValueError(f"unsupported pose source: {pose_source}")
    if pose_source == "deterministic" and deterministic_root is None:
        raise ValueError("deterministic pose source requires deterministic_root")
    publication = Path(publication_root)
    output = Path(output_root)
    provenance_path = publication / "oof_provenance.csv"
    if not provenance_path.is_file():
        raise ValueError(f"OOF provenance is missing: {provenance_path}")
    if output.exists():
        raise FileExistsError(f"feature output already exists: {output}")
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"feature staging already exists: {staging}")

    with provenance_path.open(encoding="utf-8", newline="") as handle:
        provenance_rows = list(csv.DictReader(handle))
    if people is not None:
        provenance_rows = [
            row for row in provenance_rows if row["person_id"] in people
        ]
    provenance_rows.sort(
        key=lambda row: (int(row["person_id"]), _cycle_number(row["cycle_id"]))
    )
    if not provenance_rows:
        raise ValueError("OOF publication contains no selected cycles")

    by_person: dict[str, list[dict[str, str]]] = {}
    for row in provenance_rows:
        by_person.setdefault(row["person_id"], []).append(row)

    cycle_records: list[CycleFeatures] = []
    qc_rows: list[dict[str, object]] = []
    phase_records: list[dict[str, object]] = []
    repeatability_inputs: dict[str, list[tuple[int, np.ndarray]]] = {}
    for person_id, person_rows in by_person.items():
        deterministic_pose: np.ndarray | None = None
        deterministic_face_map: np.ndarray | None = None
        if pose_source == "deterministic":
            deterministic_path = (
                Path(str(deterministic_root))
                / f"person_{person_id}"
                / "fused_sequence.npz"
            )
            if not deterministic_path.is_file():
                raise ValueError(
                    "deterministic pose sequence is missing: "
                    f"{deterministic_path}"
                )
            with np.load(
                deterministic_path,
                allow_pickle=False,
            ) as deterministic_archive:
                required = ("kpts_body", "face_map")
                missing = [
                    name
                    for name in required
                    if name not in deterministic_archive
                ]
                if missing:
                    raise ValueError(
                        "deterministic arrays missing from "
                        f"{deterministic_path}: {missing}"
                    )
                deterministic_pose = np.asarray(
                    deterministic_archive["kpts_body"],
                    dtype=np.float64,
                )
                deterministic_face_map = np.asarray(
                    deterministic_archive["face_map"]
                )
        positions = normalized_cycle_positions(len(person_rows))
        for cycle_offset, (row, position) in enumerate(
            zip(person_rows, positions, strict=True),
            start=1,
        ):
            sequence_path = publication / row["prediction_path"]
            with np.load(sequence_path, allow_pickle=False) as archive:
                required = [
                    "timestamps",
                    "frame_valid",
                    "joint_valid",
                ]
                if pose_source == "fused":
                    required.extend(
                        (
                            "kpts_body",
                            "theta_fused_rad",
                            "omega_fused_rad_s",
                        )
                    )
                elif pose_source in {"face", "side"}:
                    required.append(f"kpts_{pose_source}_canonical")
                else:
                    required.append("face_map")
                missing = [name for name in required if name not in archive]
                if missing:
                    raise ValueError(
                        f"feature arrays missing from {sequence_path}: {missing}"
                    )
                if pose_source == "fused":
                    kpts = np.asarray(
                        archive["kpts_body"],
                        dtype=np.float64,
                    )
                    theta: np.ndarray | None = np.asarray(
                        archive["theta_fused_rad"],
                        dtype=np.float64,
                    )
                    omega: np.ndarray | None = np.asarray(
                        archive["omega_fused_rad_s"],
                        dtype=np.float64,
                    )
                elif pose_source in {"face", "side"}:
                    kpts = np.asarray(
                        archive[f"kpts_{pose_source}_canonical"],
                        dtype=np.float64,
                    )
                    theta = None
                    omega = None
                else:
                    if (
                        deterministic_pose is None
                        or deterministic_face_map is None
                    ):
                        raise AssertionError(
                            "deterministic source was not loaded"
                        )
                    kpts = match_pose_frames(
                        deterministic_pose,
                        deterministic_face_map,
                        np.asarray(archive["face_map"]),
                    ).astype(np.float64, copy=False)
                    theta = None
                    omega = None
                timestamps = np.asarray(
                    archive["timestamps"],
                    dtype=np.float64,
                )
                frame_valid = np.asarray(
                    archive["frame_valid"],
                    dtype=bool,
                )
                joint_valid = np.asarray(
                    archive["joint_valid"],
                    dtype=bool,
                )

            qc = evaluate_cycle_qc(
                frame_valid,
                joint_valid,
                timestamps,
            )
            reasons = list(qc.exclusion_reasons)
            values: dict[str, float | None] = {
                outcome: None for outcome in CORE_OUTCOMES
            }
            eligible = qc.globally_eligible
            if eligible and not qc.joints_eligible(CORE_JOINTS):
                eligible = False
                reasons.append("core_joint_valid_fraction")
            if eligible:
                try:
                    kpts_clean = kpts.copy()
                    for joint_index in MAJOR_JOINT_INDICES:
                        point_valid = (
                            frame_valid
                            & joint_valid[:, joint_index]
                            & np.all(
                                np.isfinite(kpts[:, joint_index]),
                                axis=1,
                            )
                        )
                        kpts_clean[:, joint_index] = interpolate_short_gaps(
                            kpts[:, joint_index],
                            point_valid,
                        )
                    if theta is None or omega is None:
                        theta_clean = axial_rotation_from_pose(kpts_clean)
                        omega_clean = np.gradient(theta_clean, timestamps)
                    else:
                        signal_valid = (
                            frame_valid
                            & np.isfinite(theta)
                            & np.isfinite(omega)
                        )
                        theta_clean = interpolate_short_gaps(
                            theta,
                            signal_valid,
                        )
                        omega_clean = interpolate_short_gaps(
                            omega,
                            signal_valid,
                        )
                    aligned_theta, direction_sign = align_rotation_direction(
                        np.unwrap(theta_clean)
                    )
                    scalar_values = compute_core_scalars(
                        aligned_theta,
                        omega_clean * direction_sign,
                        timestamps,
                        kpts_clean,
                        direction_sign=direction_sign,
                    )
                    theta_phase = phase_normalize(
                        aligned_theta,
                        timestamps,
                        points=phase_points,
                    )
                    omega_phase = phase_normalize(
                        omega_clean * direction_sign,
                        timestamps,
                        points=phase_points,
                    )
                    tilt_phase = phase_normalize(
                        trunk_tilt_trajectory(kpts_clean),
                        timestamps,
                        points=phase_points,
                    )
                    wrist_phase = phase_normalize(
                        wrist_lead_trajectory(
                            kpts_clean,
                            direction_sign,
                        ),
                        timestamps,
                        points=phase_points,
                    )
                    pose_phase = phase_normalize(
                        kpts_clean[:, MAJOR_JOINT_INDICES],
                        timestamps,
                        points=phase_points,
                    )
                    scalar_values["peak_rotation_phase"] = (
                        peak_rotation_phase(theta_phase)
                    )
                    values.update(scalar_values)
                    phase_index = len(phase_records)
                    phase_records.append(
                        {
                            "person_id": person_id,
                            "cohort": row["cohort"],
                            "outer_fold": int(row["outer_fold"]),
                            "cycle_id": row["cycle_id"],
                            "pose_source": pose_source,
                            "theta": theta_phase,
                            "omega": omega_phase,
                            "tilt": tilt_phase,
                            "wrist": wrist_phase,
                        }
                    )
                    repeatability_inputs.setdefault(person_id, []).append(
                        (phase_index, pose_phase)
                    )
                except ValueError as error:
                    eligible = False
                    reasons.append(
                        "feature_error:" + re.sub(r"\s+", "_", str(error))
                    )

            cycle_records.append(
                CycleFeatures(
                    person_id=person_id,
                    cohort=row["cohort"],
                    outer_fold=int(row["outer_fold"]),
                    cycle_id=row["cycle_id"],
                    cycle_index=cycle_offset,
                    normalized_cycle_position=float(position),
                    pose_source=pose_source,
                    eligible=eligible,
                    exclusion_reasons=tuple(reasons),
                    values=values,
                )
            )
            qc_rows.append(
                {
                    "person_id": person_id,
                    "cohort": row["cohort"],
                    "outer_fold": int(row["outer_fold"]),
                    "cycle_id": row["cycle_id"],
                    "pose_source": pose_source,
                    "globally_eligible": qc.globally_eligible,
                    "feature_eligible": eligible,
                    "valid_frame_fraction": float(frame_valid.mean()),
                    "minimum_joint_valid_fraction": float(
                        qc.joint_valid_fraction[
                            np.asarray(MAJOR_JOINT_INDICES)
                        ].min()
                    ),
                    "exclusion_reasons": ";".join(reasons),
                }
            )

    record_by_key = {
        (record.person_id, record.cycle_id): index
        for index, record in enumerate(cycle_records)
    }
    for person_id, entries in repeatability_inputs.items():
        if len(entries) < 2:
            continue
        trajectories = np.stack([entry[1] for entry in entries])
        errors = leave_one_cycle_out_repeatability(trajectories)
        for (phase_index, _), error in zip(entries, errors, strict=True):
            phase_record = phase_records[phase_index]
            record_index = record_by_key[
                (person_id, str(phase_record["cycle_id"]))
            ]
            record = cycle_records[record_index]
            updated = dict(record.values)
            updated["whole_body_repeatability"] = float(error)
            cycle_records[record_index] = CycleFeatures(
                **{
                    **record.__dict__,
                    "values": updated,
                }
            )

    cycle_rows = [_flatten_cycle(record) for record in cycle_records]
    person_rows = _person_summaries(
        cycle_rows,
        minimum_cycles=minimum_person_cycles,
    )
    staging.mkdir(parents=True)
    try:
        pd.DataFrame(cycle_rows).to_csv(
            staging / "cycle_features.csv",
            index=False,
            float_format="%.10g",
        )
        pd.DataFrame(person_rows).to_csv(
            staging / "person_features.csv",
            index=False,
            float_format="%.10g",
        )
        pd.DataFrame(qc_rows).to_csv(
            staging / "qc_exclusions.csv",
            index=False,
            float_format="%.10g",
        )
        _write_phase_curves(phase_records, staging / "phase_curves.npz")
        summary = {
            "schema_version": 1,
            "pose_source": pose_source,
            "people": len(by_person),
            "cycles": len(cycle_records),
            "eligible_cycles": sum(record.eligible for record in cycle_records),
            "excluded_cycles": sum(
                not record.eligible for record in cycle_records
            ),
        }
        (staging / "qc_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "pose_source": pose_source,
            "source_provenance_sha256": sha256_file(provenance_path),
            "outputs": {
                path.name: sha256_file(path)
                for path in sorted(staging.iterdir())
                if path.is_file()
            },
        }
        (staging / "feature_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "people": len(by_person),
        "cycles": len(cycle_records),
        "eligible_cycles": sum(record.eligible for record in cycle_records),
    }


def _flatten_cycle(record: CycleFeatures) -> dict[str, object]:
    return {
        "person_id": record.person_id,
        "cohort": record.cohort,
        "outer_fold": record.outer_fold,
        "cycle_id": record.cycle_id,
        "cycle_index": record.cycle_index,
        "normalized_cycle_position": record.normalized_cycle_position,
        "pose_source": record.pose_source,
        "eligible": record.eligible,
        "exclusion_reasons": ";".join(record.exclusion_reasons),
        **record.values,
    }


def _person_summaries(
    cycle_rows: list[dict[str, object]],
    *,
    minimum_cycles: int,
) -> list[dict[str, object]]:
    table = pd.DataFrame(cycle_rows)
    rows: list[dict[str, object]] = []
    for person_id, group in table.groupby("person_id", sort=False):
        result: dict[str, object] = {
            "person_id": person_id,
            "cohort": group["cohort"].iloc[0],
            "outer_fold": int(group["outer_fold"].iloc[0]),
            "pose_source": group["pose_source"].iloc[0],
            "total_cycles": len(group),
            "eligible_cycles": int(group["eligible"].sum()),
        }
        for outcome in CORE_OUTCOMES:
            usable = group.loc[
                group["eligible"] & group[outcome].notna(),
                ["normalized_cycle_position", outcome],
            ]
            result[f"{outcome}_n"] = len(usable)
            if len(usable) < minimum_cycles:
                result[f"{outcome}_median"] = np.nan
                result[f"{outcome}_mad"] = np.nan
                result[f"{outcome}_slope"] = np.nan
                continue
            outcome_values = usable[outcome].to_numpy(dtype=float)
            median = float(np.median(outcome_values))
            result[f"{outcome}_median"] = median
            result[f"{outcome}_mad"] = float(
                np.median(np.abs(outcome_values - median))
            )
            result[f"{outcome}_slope"] = float(
                np.polyfit(
                    usable["normalized_cycle_position"].to_numpy(dtype=float),
                    outcome_values,
                    1,
                )[0]
            )
        rows.append(result)
    return rows


def _write_phase_curves(
    records: list[dict[str, object]],
    path: Path,
) -> None:
    if not records:
        np.savez_compressed(
            path,
            person_id=np.array([], dtype="U1"),
            theta=np.empty((0, 101)),
        )
        return
    np.savez_compressed(
        path,
        person_id=np.asarray([record["person_id"] for record in records]),
        cohort=np.asarray([record["cohort"] for record in records]),
        outer_fold=np.asarray(
            [record["outer_fold"] for record in records],
            dtype=np.int64,
        ),
        cycle_id=np.asarray([record["cycle_id"] for record in records]),
        theta=np.stack([record["theta"] for record in records]),
        omega=np.stack([record["omega"] for record in records]),
        tilt=np.stack([record["tilt"] for record in records]),
        wrist=np.stack([record["wrist"] for record in records]),
    )


def _cycle_number(cycle_id: str) -> int:
    match = re.search(r"(\d+)$", cycle_id)
    if match is None:
        raise ValueError(f"cycle ID has no numeric suffix: {cycle_id}")
    return int(match.group(1))
