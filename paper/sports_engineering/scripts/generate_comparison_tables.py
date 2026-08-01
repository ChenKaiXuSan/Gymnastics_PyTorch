#!/usr/bin/env python3
"""Generate source-checked extrinsic and per-joint paper tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from gymnastics.analysis.cohort_cycle.joints import MAJOR_JOINT_INDICES
from gymnastics.common.skeletons import MHR70_NAMES
from gymnastics.fusion.deterministic.experiment_matrix import (
    NO_EXTRINSIC_METHODS,
    build_pair_index,
    load_triangulated_sequence,
)
from gymnastics.fusion.rotation_aware.config import SkeletonSpec, load_skeleton_spec
from gymnastics.fusion.rotation_aware.evaluation import _external_errors


LEARNED_JOINT_METHODS = ("A0", "A1", "A2", "A6")
EXTRINSIC_JOINT_METHODS = (
    "extrinsic_r_average",
    "extrinsic_r_quality_average",
)
EXPECTED_TEST_PEOPLE = 14
EXPECTED_JOINTS = 70
PERSON_BASELINE_METHOD = "avg_body_current"
WORLD_BASELINE_METHOD = "avg_world_face_ref"
JOINT_EVALUATION_PROTOCOL = "similarity_plus_hip_centering"
DISPLAY_NAMES = {
    "A0": "Face",
    "A1": "Side",
    "A2": "Body-frame mean",
    "A6": "A6",
    "extrinsic_r_average": "Extrinsic-R",
    "extrinsic_r_quality_average": "Extrinsic-R quality",
    PERSON_BASELINE_METHOD: "Body-frame average",
}
DETERMINISTIC_DISPLAY_NAMES = {
    "sim3_face_stable_joint_weight": "Pseudo-reference-fitted joint weights (leaky)",
    "avg_body_current": "Body-frame average",
    "sim3_face_stable": "Similarity alignment, stable joints",
    "sim3_face_stable_bodypart_weight": "Similarity alignment, body-part weights",
    "sim3_face_stable_smooth_transform": "Similarity alignment, side smoothing",
    "sim3_face_stable_smooth_kpt": "Similarity alignment, output smoothing",
    "sim3_face_all": "Similarity alignment, all joints",
    "avg_world_face_ref": "World-coordinate average",
    "root_face_stable": "Root alignment and average",
}


def load_all_people(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    people = tuple(
        str(person_id)
        for partition in ("train", "val", "test")
        for person_id in payload.get(partition, ())
    )
    if len(people) != 137 or len(set(people)) != 137:
        raise ValueError("split must contain exactly 137 unique people")
    return people


def load_test_people(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    people = tuple(str(person_id) for person_id in payload.get("test", ()))
    if len(people) != EXPECTED_TEST_PEOPLE or len(set(people)) != EXPECTED_TEST_PEOPLE:
        raise ValueError("split must contain exactly 14 test people")
    return people


def load_joint_metrics(path: Path, methods: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"person_id": str})
    required = {"person_id", "method", "joint", "mpjpe"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame.loc[frame["method"].isin(methods)].copy()
    found_methods = set(frame["method"])
    if found_methods != set(methods):
        raise ValueError(
            f"{path} must contain methods {sorted(methods)}, found {sorted(found_methods)}"
        )
    frame["person_id"] = frame["person_id"].astype(str)
    frame["joint"] = pd.to_numeric(frame["joint"], errors="raise").astype(int)
    frame["mpjpe"] = pd.to_numeric(frame["mpjpe"], errors="raise")
    if not np.isfinite(frame["mpjpe"].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite MPJPE")
    key = ["person_id", "method", "joint"]
    if frame.duplicated(key).any():
        duplicate = frame.loc[frame.duplicated(key, keep=False), key].iloc[0].tolist()
        raise ValueError(f"duplicate person-method-joint row: {duplicate}")
    return frame


def _validate_joint_coverage(
    frame: pd.DataFrame,
    methods: tuple[str, ...],
    people: tuple[str, ...],
) -> pd.DataFrame:
    selected = frame.loc[frame["person_id"].isin(people)].copy()
    if set(selected["person_id"]) != set(people):
        missing = sorted(set(people) - set(selected["person_id"]))
        raise ValueError(f"joint metrics are missing test people: {missing}")
    expected_joints = set(range(EXPECTED_JOINTS))
    for person_id in people:
        for method in methods:
            joints = set(
                selected.loc[
                    (selected["person_id"] == person_id)
                    & (selected["method"] == method),
                    "joint",
                ].astype(int)
            )
            if joints != expected_joints:
                raise ValueError(
                    f"{person_id}/{method} must contain all 70 canonical joints"
                )
    return selected


def build_joint_summary(
    learned: pd.DataFrame,
    extrinsic: pd.DataFrame,
    test_people: tuple[str, ...],
) -> pd.DataFrame:
    if "evaluation_protocol" not in learned or "evaluation_protocol" not in extrinsic:
        raise ValueError("joint sources must declare the same evaluation protocol")
    protocols = set(learned["evaluation_protocol"]) | set(
        extrinsic["evaluation_protocol"]
    )
    if protocols != {JOINT_EVALUATION_PROTOCOL}:
        raise ValueError("joint sources must use the same evaluation protocol")
    learned = _validate_joint_coverage(learned, LEARNED_JOINT_METHODS, test_people)
    extrinsic = _validate_joint_coverage(
        extrinsic, EXTRINSIC_JOINT_METHODS, test_people
    )
    combined = pd.concat([learned, extrinsic], ignore_index=True)
    person_joint = (
        combined.groupby(["person_id", "method", "joint"], as_index=False)["mpjpe"]
        .mean()
    )
    summary = person_joint.groupby(["joint", "method"])["mpjpe"].mean().unstack()
    summary = summary.loc[range(EXPECTED_JOINTS)].reset_index()
    summary.insert(1, "joint_name", list(MHR70_NAMES))
    ordered = [
        "joint",
        "joint_name",
        *LEARNED_JOINT_METHODS,
        *EXTRINSIC_JOINT_METHODS,
    ]
    summary = summary.loc[:, ordered]
    value_columns = [*LEARNED_JOINT_METHODS, *EXTRINSIC_JOINT_METHODS]
    summary.loc[:, value_columns] = summary.loc[:, value_columns] * 1000.0
    return summary


def evaluate_matched_metrics(
    person_id: str,
    method: str,
    matched_cycles: Sequence[tuple[np.ndarray, np.ndarray]],
    skeleton: SkeletonSpec,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    pooled_errors: list[np.ndarray] = []
    errors_by_joint: list[list[np.ndarray]] = [
        [] for _ in range(len(skeleton.joint_names))
    ]
    matched_frames = 0
    for candidate, reference in matched_cycles:
        errors, valid = _external_errors(
            np.asarray(candidate, dtype=np.float64),
            np.asarray(reference, dtype=np.float64),
            skeleton,
            alignment="similarity",
        )
        matched_frames += int(valid.any(axis=1).sum())
        pooled_errors.append(errors[valid])
        for joint in range(errors.shape[1]):
            errors_by_joint[joint].append(errors[:, joint][valid[:, joint]])
    pooled = np.concatenate(pooled_errors) if pooled_errors else np.asarray([], dtype=float)
    if not len(pooled):
        raise ValueError(f"{person_id}/{method} has no matched values")
    person_row: dict[str, float | int | str] = {
        "person_id": str(person_id),
        "method": method,
        "matched_frames": matched_frames,
        "valid_points": int(len(pooled)),
        "mpjpe": float(pooled.mean()),
        "median": float(np.median(pooled)),
        "p95": float(np.percentile(pooled, 95)),
        "evaluation_protocol": JOINT_EVALUATION_PROTOCOL,
    }
    rows = []
    for joint, chunks in enumerate(errors_by_joint):
        values = np.concatenate(chunks) if chunks else np.asarray([], dtype=float)
        if not len(values):
            raise ValueError(f"{person_id}/{method}/joint_{joint} has no matched values")
        rows.append(
            {
                "person_id": str(person_id),
                "method": method,
                "joint": joint,
                "valid_points": int(len(values)),
                "mpjpe": float(values.mean()),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "evaluation_protocol": JOINT_EVALUATION_PROTOCOL,
            }
        )
    return person_row, pd.DataFrame(rows)


def evaluate_matched_joint_metrics(
    person_id: str,
    method: str,
    matched_cycles: Sequence[tuple[np.ndarray, np.ndarray]],
    skeleton: SkeletonSpec,
) -> pd.DataFrame:
    _, joint_rows = evaluate_matched_metrics(
        person_id, method, matched_cycles, skeleton
    )
    return joint_rows


def reevaluate_compact_metrics(
    method_roots: Mapping[str, Path],
    people: tuple[str, ...],
    triangulated_root: Path,
    skeleton: SkeletonSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    person_rows: list[dict[str, float | int | str]] = []
    joint_frames: list[pd.DataFrame] = []
    for method, method_root in method_roots.items():
        for person_id in people:
            compact_path = method_root / f"person_{person_id}" / "fused_sequence.npz"
            if not compact_path.exists():
                raise FileNotFoundError(compact_path)
            with np.load(compact_path, allow_pickle=False) as data:
                candidate = np.asarray(data["kpts_world"], dtype=np.float64)
                face_map = np.asarray(data["face_map"], dtype=int)
                side_map = np.asarray(data["side_map"], dtype=int)
            pair_index = build_pair_index(face_map, side_map)
            matched_cycles: list[tuple[np.ndarray, np.ndarray]] = []
            person_root = triangulated_root / f"person_{person_id}"
            for cycle_root in sorted(person_root.glob("cycle_*")):
                reference, pairs = load_triangulated_sequence(cycle_root)
                candidate_frames = []
                reference_frames = []
                for reference_frame, pair in zip(reference, pairs):
                    candidate_index = pair_index.get(pair)
                    if candidate_index is None:
                        continue
                    candidate_frames.append(candidate[candidate_index])
                    reference_frames.append(reference_frame)
                if candidate_frames:
                    matched_cycles.append(
                        (np.stack(candidate_frames), np.stack(reference_frames))
                    )
            if not matched_cycles:
                raise ValueError(f"{person_id}/{method} has no matched triangulated cycles")
            person_row, joint_rows = evaluate_matched_metrics(
                person_id, method, matched_cycles, skeleton
            )
            person_rows.append(person_row)
            joint_frames.append(joint_rows)
    return pd.DataFrame(person_rows), pd.concat(joint_frames, ignore_index=True)


def reevaluate_compact_joint_metrics(
    method_roots: Mapping[str, Path],
    people: tuple[str, ...],
    triangulated_root: Path,
    skeleton: SkeletonSpec,
) -> pd.DataFrame:
    _, joint_rows = reevaluate_compact_metrics(
        method_roots, people, triangulated_root, skeleton
    )
    return joint_rows


def load_person_metrics(path: Path, methods: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"person_id": str})
    required = {"person_id", "method", "mpjpe"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame.loc[frame["method"].isin(methods), list(required)].copy()
    if set(frame["method"]) != set(methods):
        raise ValueError(f"{path} does not contain all requested methods")
    frame["person_id"] = frame["person_id"].astype(str)
    frame["mpjpe"] = pd.to_numeric(frame["mpjpe"], errors="raise")
    if not np.isfinite(frame["mpjpe"].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite MPJPE")
    if frame.duplicated(["person_id", "method"]).any():
        raise ValueError(f"{path} contains duplicate person-method rows")
    return frame


def load_cached_person_metrics(
    path: Path,
    methods: Sequence[str],
    people: Sequence[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"person_id": str})
    required = {"person_id", "method", "mpjpe", "evaluation_protocol"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame.loc[
        frame["method"].isin(methods) & frame["person_id"].isin(people)
    ].copy()
    if set(frame["evaluation_protocol"]) != {JOINT_EVALUATION_PROTOCOL}:
        raise ValueError("cached person metrics use the wrong evaluation protocol")
    if frame.duplicated(["person_id", "method"]).any():
        raise ValueError("cached person metrics contain duplicate person-method rows")
    expected = {(str(person), method) for method in methods for person in people}
    actual = set(zip(frame["person_id"].astype(str), frame["method"].astype(str)))
    if actual != expected:
        raise ValueError("cached person metrics lack complete method-person coverage")
    frame["mpjpe"] = pd.to_numeric(frame["mpjpe"], errors="raise")
    if not np.isfinite(frame["mpjpe"].to_numpy(dtype=float)).all():
        raise ValueError("cached person metrics contain non-finite MPJPE")
    return frame


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, float(p_values[index]) * (count - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _bootstrap_mean_difference(
    differences: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0, len(differences), size=(repetitions, len(differences))
    )
    sampled_means = differences[sample_indices].mean(axis=1)
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return float(low), float(high)


def build_extrinsic_summary(
    deterministic: pd.DataFrame,
    extrinsic: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    if (
        "evaluation_protocol" not in deterministic
        or "evaluation_protocol" not in extrinsic
    ):
        raise ValueError("person sources must declare the same evaluation protocol")
    protocols = set(deterministic["evaluation_protocol"]) | set(
        extrinsic["evaluation_protocol"]
    )
    if protocols != {JOINT_EVALUATION_PROTOCOL}:
        raise ValueError("person sources must use the same evaluation protocol")
    deterministic = deterministic.loc[
        deterministic["method"] == PERSON_BASELINE_METHOD,
        ["person_id", "method", "mpjpe"],
    ].copy()
    extrinsic = extrinsic.loc[
        extrinsic["method"].isin(EXTRINSIC_JOINT_METHODS),
        ["person_id", "method", "mpjpe"],
    ].copy()
    for frame in (deterministic, extrinsic):
        frame["person_id"] = frame["person_id"].astype(str)
        frame["mpjpe"] = pd.to_numeric(frame["mpjpe"], errors="raise")
        if not np.isfinite(frame["mpjpe"].to_numpy(dtype=float)).all():
            raise ValueError("person metrics contain non-finite MPJPE")
        if frame.duplicated(["person_id", "method"]).any():
            raise ValueError("person metrics contain duplicate person-method rows")
    people = set(deterministic["person_id"])
    if not people:
        raise ValueError("body-frame baseline is empty")
    for method in EXTRINSIC_JOINT_METHODS:
        method_people = set(extrinsic.loc[extrinsic["method"] == method, "person_id"])
        if method_people != people:
            raise ValueError(f"{method} people do not match body-frame baseline")

    combined = pd.concat([deterministic, extrinsic], ignore_index=True)
    wide = combined.pivot(index="person_id", columns="method", values="mpjpe")
    baseline = wide[PERSON_BASELINE_METHOD].to_numpy(dtype=float)
    raw_p_values: list[float] = []
    differences_by_method: dict[str, np.ndarray] = {}
    for method in EXTRINSIC_JOINT_METHODS:
        differences = wide[method].to_numpy(dtype=float) - baseline
        differences_by_method[method] = differences
        if np.allclose(differences, 0.0):
            raw_p_values.append(1.0)
        else:
            raw_p_values.append(float(wilcoxon(differences, alternative="two-sided").pvalue))
    adjusted = dict(zip(EXTRINSIC_JOINT_METHODS, _holm_adjust(raw_p_values)))

    rows: list[dict[str, float | int | str]] = [
        {
            "method": PERSON_BASELINE_METHOD,
            "n": len(people),
            "mean_mm": float(baseline.mean() * 1000.0),
            "std_mm": float(baseline.std(ddof=1) * 1000.0),
            "delta_mm": 0.0,
            "ci_low_mm": np.nan,
            "ci_high_mm": np.nan,
            "p_holm": np.nan,
            "improved_people": 0,
            "evaluation_protocol": JOINT_EVALUATION_PROTOCOL,
        }
    ]
    for method_index, method in enumerate(EXTRINSIC_JOINT_METHODS):
        values = wide[method].to_numpy(dtype=float)
        differences = differences_by_method[method]
        low, high = _bootstrap_mean_difference(
            differences,
            repetitions=bootstrap_repetitions,
            seed=20260731 + method_index,
        )
        rows.append(
            {
                "method": method,
                "n": len(people),
                "mean_mm": float(values.mean() * 1000.0),
                "std_mm": float(values.std(ddof=1) * 1000.0),
                "delta_mm": float(differences.mean() * 1000.0),
                "ci_low_mm": low * 1000.0,
                "ci_high_mm": high * 1000.0,
                "p_holm": adjusted[method],
                "improved_people": int((differences < 0.0).sum()),
                "evaluation_protocol": JOINT_EVALUATION_PROTOCOL,
            }
        )
    return pd.DataFrame(rows)


def build_extrinsic_summaries(
    person_metrics: pd.DataFrame,
    test_people: Sequence[str],
    *,
    bootstrap_repetitions: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_people = tuple(str(person_id) for person_id in test_people)
    if not test_people or len(set(test_people)) != len(test_people):
        raise ValueError("test people must be non-empty and unique")
    relevant_methods = (PERSON_BASELINE_METHOD, *EXTRINSIC_JOINT_METHODS)
    relevant = person_metrics.loc[
        person_metrics["method"].isin(relevant_methods)
    ].copy()
    all_people = set(
        relevant.loc[
            relevant["method"] == PERSON_BASELINE_METHOD, "person_id"
        ].astype(str)
    )
    if not set(test_people).issubset(all_people):
        raise ValueError("test people must be present in all-participant metrics")
    heldout = relevant.loc[
        relevant["person_id"].astype(str).isin(test_people)
    ].copy()

    def summarize(frame: pd.DataFrame) -> pd.DataFrame:
        return build_extrinsic_summary(
            frame.loc[frame["method"] == PERSON_BASELINE_METHOD],
            frame.loc[frame["method"].isin(EXTRINSIC_JOINT_METHODS)],
            bootstrap_repetitions=bootstrap_repetitions,
        )

    return summarize(heldout), summarize(relevant)


def build_calibration_association(
    person_metrics: pd.DataFrame,
    extrinsics: Mapping[str, object],
) -> dict[str, float | int | str]:
    selected = person_metrics.loc[
        person_metrics["method"] == "extrinsic_r_average"
    ].copy()
    if set(selected.get("evaluation_protocol", ())) != {JOINT_EVALUATION_PROTOCOL}:
        raise ValueError("calibration association requires the unified evaluation protocol")
    persons = extrinsics.get("persons")
    if not isinstance(persons, Mapping):
        raise ValueError("extrinsics payload must contain a persons mapping")
    reprojection: list[float] = []
    errors: list[float] = []
    for row in selected.itertuples(index=False):
        metadata = persons.get(str(row.person_id))
        if not isinstance(metadata, Mapping) or "holdout_reproj_px" not in metadata:
            raise ValueError(f"missing holdout reprojection error for person {row.person_id}")
        reprojection.append(float(metadata["holdout_reproj_px"]))
        errors.append(float(row.mpjpe))
    rho, p_value = spearmanr(reprojection, errors)
    return {
        "n": len(errors),
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "evaluation_protocol": JOINT_EVALUATION_PROTOCOL,
    }


def build_coordinate_summary(person_metrics: pd.DataFrame) -> pd.DataFrame:
    methods = (WORLD_BASELINE_METHOD, PERSON_BASELINE_METHOD)
    selected = person_metrics.loc[person_metrics["method"].isin(methods)].copy()
    if set(selected.get("evaluation_protocol", ())) != {JOINT_EVALUATION_PROTOCOL}:
        raise ValueError("coordinate summary requires the unified evaluation protocol")
    people_by_method = {
        method: set(selected.loc[selected["method"] == method, "person_id"].astype(str))
        for method in methods
    }
    if not people_by_method[WORLD_BASELINE_METHOD] or (
        people_by_method[WORLD_BASELINE_METHOD]
        != people_by_method[PERSON_BASELINE_METHOD]
    ):
        raise ValueError("coordinate methods must cover the same people")
    if selected.duplicated(["person_id", "method"]).any():
        raise ValueError("coordinate metrics contain duplicate person-method rows")
    wide = selected.pivot(index="person_id", columns="method", values="mpjpe")
    world_mean = float(wide[WORLD_BASELINE_METHOD].mean())
    rows: list[dict[str, float | int | str]] = []
    for method in methods:
        values = wide[method].to_numpy(dtype=float)
        mean = float(values.mean())
        rows.append(
            {
                "method": method,
                "n": len(values),
                "mean_mm": mean * 1000.0,
                "std_mm": float(values.std(ddof=1) * 1000.0),
                "reduction_vs_world_pct": (
                    100.0 * (world_mean - mean) / world_mean
                    if method == PERSON_BASELINE_METHOD
                    else 0.0
                ),
                "evaluation_protocol": JOINT_EVALUATION_PROTOCOL,
            }
        )
    return pd.DataFrame(rows)


def build_deterministic_summary(
    person_metrics: pd.DataFrame,
    *,
    methods: Sequence[str] = NO_EXTRINSIC_METHODS,
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    selected = person_metrics.loc[person_metrics["method"].isin(methods)].copy()
    if set(selected.get("evaluation_protocol", ())) != {JOINT_EVALUATION_PROTOCOL}:
        raise ValueError("deterministic summary requires the unified evaluation protocol")
    expected_people: set[str] | None = None
    rows: list[dict[str, float | int | str]] = []
    for method_index, method in enumerate(methods):
        method_rows = selected.loc[selected["method"] == method].copy()
        people = set(method_rows["person_id"].astype(str))
        if expected_people is None:
            expected_people = people
        if not people or people != expected_people:
            raise ValueError("deterministic methods must cover the same people")
        if method_rows.duplicated(["person_id", "method"]).any():
            raise ValueError("deterministic metrics contain duplicate person-method rows")
        values = method_rows["mpjpe"].to_numpy(dtype=float)
        low, high = _bootstrap_mean_difference(
            values,
            repetitions=bootstrap_repetitions,
            seed=20260801 + method_index,
        )
        rows.append(
            {
                "method": method,
                "n": len(values),
                "mean_mm": float(values.mean() * 1000.0),
                "std_mm": float(values.std(ddof=1) * 1000.0),
                "ci_low_mm": low * 1000.0,
                "ci_high_mm": high * 1000.0,
                "evaluation_protocol": JOINT_EVALUATION_PROTOCOL,
            }
        )
    return pd.DataFrame(rows)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    rendered = value
    for source, replacement in replacements.items():
        rendered = rendered.replace(source, replacement)
    return rendered


def _format_accuracy_row(
    label: str,
    values: list[float],
) -> str:
    minimum = min(values)
    cells = []
    for value in values:
        rendered = f"{value:.1f}"
        if np.isclose(value, minimum, rtol=0.0, atol=5e-10):
            rendered = rf"\textbf{{{rendered}}}"
        cells.append(rendered)
    return f"{_latex_escape(label)} & " + " & ".join(cells) + r" \\ % joint-row"


def render_main_joint_table(summary: pd.DataFrame) -> str:
    methods = ["A0", "A1", "A2", "A6", "extrinsic_r_average"]
    selected = summary.set_index("joint").loc[list(MAJOR_JOINT_INDICES)]
    rows = [
        _format_accuracy_row(
            str(row["joint_name"]), [float(row[method]) for method in methods]
        )
        for _, row in selected.iterrows()
    ]
    macro = [float(selected[method].mean()) for method in methods]
    mean_row = _format_accuracy_row("20-joint macro mean", macro).replace(
        "% joint-row", "% macro-row"
    )
    return """\\begin{table*}[t]
\\caption{Per-joint agreement on the 14 held-out participants. Values are mean
participant-level MPJPE in mm after sequence-level similarity alignment to the
same-video pseudo-reference and framewise hip centring. Lower is better; bold indicates the lowest
descriptive value in each row. Extrinsic-R is a camera-assisted comparator.}
\\label{tab:joint-accuracy-main}
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{3.5pt}
\\begin{tabular}{lrrrrr}
\\toprule
Joint & Face & Side & Body-frame mean & A6 & Extrinsic-R \\\\
\\midrule
""" + "\n".join(rows) + """
\\midrule
""" + mean_row + """
\\bottomrule
\\end{tabular}
\\end{table*}
"""


def render_all_joint_table(summary: pd.DataFrame) -> str:
    methods = [
        "A0",
        "A1",
        "A2",
        "A6",
        "extrinsic_r_average",
        "extrinsic_r_quality_average",
    ]
    rows = [
        _format_accuracy_row(
            str(row["joint_name"]), [float(row[method]) for method in methods]
        )
        for _, row in summary.sort_values("joint").iterrows()
    ]
    return """\\begin{longtable}{lrrrrrr}
\\caption{Complete MHR70 per-joint agreement on the same 14 held-out
participants. Values are participant-level mean MPJPE in mm; bold is the
lowest descriptive value in each row. Each cycle uses one similarity alignment
followed by framewise hip centring.}\\label{tab:joint-accuracy-all70}\\\\
\\toprule
Joint & Face & Side & Body mean & A6 & Extrinsic-R & Extrinsic-R quality \\\\
\\midrule
\\endfirsthead
\\multicolumn{7}{c}{\\tablename\\ \\thetable{} -- continued}\\\\
\\toprule
Joint & Face & Side & Body mean & A6 & Extrinsic-R & Extrinsic-R quality \\\\
\\midrule
\\endhead
\\midrule
\\multicolumn{7}{r}{Continued on next page}\\\\
\\endfoot
\\bottomrule
\\endlastfoot
""" + "\n".join(rows) + """
\\end{longtable}
"""


def render_extrinsic_table(summary: pd.DataFrame, *, scope: str) -> str:
    n_values = set(pd.to_numeric(summary["n"], errors="raise").astype(int))
    if len(n_values) != 1:
        raise ValueError("extrinsic table rows must use one participant count")
    participant_count = n_values.pop()
    if scope == "heldout":
        if participant_count != EXPECTED_TEST_PEOPLE:
            raise ValueError("heldout extrinsic table must contain 14 participants")
        caption_lead = (
            "Camera-assisted deterministic comparison on the same 14 held-out "
            "participants used in Table~1."
        )
        label = "tab:extrinsic-comparison"
    elif scope == "all_participants":
        if participant_count != 137:
            raise ValueError("all-participant extrinsic table must contain 137 participants")
        caption_lead = (
            "Secondary camera-assisted deterministic comparison over all 137 "
            "participants."
        )
        label = "tab:extrinsic-comparison-all137"
    else:
        raise ValueError(f"unknown extrinsic table scope: {scope}")
    rows = []
    for row in summary.itertuples(index=False):
        if row.method == PERSON_BASELINE_METHOD:
            comparison = "--"
            p_value = "--"
            improved = "--"
        else:
            comparison = (
                f"{row.delta_mm:+.3f} "
                f"[{row.ci_low_mm:+.3f}, {row.ci_high_mm:+.3f}]"
            )
            p_value = f"{row.p_holm:.3g}"
            improved = f"{int(row.improved_people)}/{participant_count}"
        rows.append(
            f"{DISPLAY_NAMES[row.method]} & {row.mean_mm:.3f} $\\pm$ "
            f"{row.std_mm:.3f} & {comparison} & {p_value} & {improved} \\\\"
        )
    return f"""\\begin{{table*}}[t]
\\caption{{{caption_lead}
The difference is method minus the calibration-free body-frame average, with a
participant-bootstrap 95\\% confidence interval. $p$ values are Holm-adjusted
paired Wilcoxon tests. Each cycle uses one similarity alignment to the
same-video pseudo-reference followed by framewise hip centring.}}
\\label{{{label}}}
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Method & MPJPE (mm) & Difference [95\\% CI] & $p_{{\\rm Holm}}$ & Improved people \\\\
\\midrule
""" + "\n".join(rows) + f"""
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


def render_deterministic_table(summary: pd.DataFrame) -> str:
    rows = [
        (
            f"{DETERMINISTIC_DISPLAY_NAMES.get(row.method, row.method)} & "
            f"{row.mean_mm:.3f} & {row.std_mm:.3f} & "
            f"[{row.ci_low_mm:.3f}, {row.ci_high_mm:.3f}] "
            r"\\ % deterministic-row"
        )
        for row in summary.itertuples(index=False)
    ]
    return """\\begin{table}[h]
\\caption{Agreement with the private triangulated pseudo-reference over 137
participants. Values are mm after one similarity alignment per cycle followed
by framewise hip centring. The pseudo-reference-fitted row is a leakage
diagnostic and is not an eligible label-free method.}
\\label{tab:deterministic}
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{3pt}
\\begin{tabular}{p{0.42\\linewidth}rrr}
\\toprule
Method & Mean & SD & 95\\% CI\\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""


def _default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[3]
    evaluation = root / (
        "local/runs/fuse_rotation_aware/evaluation/"
        "all137_a4_e100_seed0+all137_a5_e100_seed0+all137_a6_e100_seed0+"
        "all137_a7_e100_seed0+all137_a8_e100_seed0+all137_a9_e100_seed0"
    )
    return {
        "root": root,
        "split": root / "configs/fusion/folds/paper_137_a6_split.json",
        "learned_joint": evaluation / "metrics_by_joint.csv",
        "deterministic_root": root / "local/runs/fuse_experiments/avg_body_current",
        "extrinsic_root": root / "local/runs/fuse_extrinsic_baselines",
        "extrinsics": root / "local/runs/analysis/extrinsics/estimated_extrinsics.json",
        "triangulated_root": Path(
            "/home/data/xchen/gymnastics/sam3d_triangulated/person"
        ),
        "skeleton": root / "configs/fusion/skeleton_mhr70.yaml",
        "output": root / "paper/sports_engineering/generated",
    }


def parse_args() -> argparse.Namespace:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=defaults["split"])
    parser.add_argument("--learned-joint", type=Path, default=defaults["learned_joint"])
    parser.add_argument("--deterministic-root", type=Path, default=defaults["deterministic_root"])
    parser.add_argument("--extrinsic-root", type=Path, default=defaults["extrinsic_root"])
    parser.add_argument("--extrinsics", type=Path, default=defaults["extrinsics"])
    parser.add_argument("--triangulated-root", type=Path, default=defaults["triangulated_root"])
    parser.add_argument("--skeleton", type=Path, default=defaults["skeleton"])
    parser.add_argument("--output", type=Path, default=defaults["output"])
    parser.add_argument(
        "--reuse-person-cache",
        action="store_true",
        help="reuse a complete protocol-checked 137-person CSV while recomputing test14 joint rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_people = load_all_people(args.split)
    test_people = load_test_people(args.split)
    learned_joint = load_joint_metrics(args.learned_joint, LEARNED_JOINT_METHODS)
    learned_joint["evaluation_protocol"] = JOINT_EVALUATION_PROTOCOL
    skeleton = load_skeleton_spec(args.skeleton)
    method_roots = {
        **{
            method: args.deterministic_root.parent / method
            for method in NO_EXTRINSIC_METHODS
        },
        **{
            method: args.extrinsic_root / method
            for method in EXTRINSIC_JOINT_METHODS
        },
    }
    person_cache = args.output / "pseudo_reference_person_metrics_matched_137.csv"
    if args.reuse_person_cache and person_cache.exists():
        unified_person = load_cached_person_metrics(
            person_cache, tuple(method_roots), all_people
        )
        extrinsic_joint = reevaluate_compact_joint_metrics(
            {
                method: method_roots[method]
                for method in EXTRINSIC_JOINT_METHODS
            },
            test_people,
            args.triangulated_root,
            skeleton,
        )
        person_source = "protocol_checked_cache"
    else:
        unified_person, unified_joint = reevaluate_compact_metrics(
            method_roots,
            all_people,
            args.triangulated_root,
            skeleton,
        )
        extrinsic_joint = unified_joint.loc[
            unified_joint["method"].isin(EXTRINSIC_JOINT_METHODS)
            & unified_joint["person_id"].isin(test_people)
        ].copy()
        person_source = "fresh_compact_reevaluation"
    joint_summary = build_joint_summary(learned_joint, extrinsic_joint, test_people)
    extrinsic_person = unified_person.loc[
        unified_person["method"].isin(EXTRINSIC_JOINT_METHODS)
    ].copy()
    extrinsic_test_summary, extrinsic_all_summary = build_extrinsic_summaries(
        unified_person,
        test_people,
    )
    coordinate_summary = build_coordinate_summary(unified_person)
    deterministic_summary = build_deterministic_summary(unified_person)
    calibration_association = build_calibration_association(
        extrinsic_person,
        json.loads(args.extrinsics.read_text(encoding="utf-8")),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    unified_person.to_csv(
        args.output / "pseudo_reference_person_metrics_matched_137.csv", index=False
    )
    unified_person.loc[
        unified_person["method"].isin(
            (PERSON_BASELINE_METHOD, *EXTRINSIC_JOINT_METHODS)
        )
    ].to_csv(
        args.output / "extrinsic_person_metrics_matched_137.csv", index=False
    )
    extrinsic_joint.to_csv(
        args.output / "extrinsic_joint_metrics_test14.csv", index=False
    )
    joint_summary.to_csv(args.output / "joint_accuracy_test14.csv", index=False)
    extrinsic_test_summary.to_csv(
        args.output / "extrinsic_comparison_test14.csv", index=False
    )
    extrinsic_all_summary.to_csv(
        args.output / "extrinsic_comparison_137.csv", index=False
    )
    coordinate_summary.to_csv(
        args.output / "coordinate_comparison_137.csv", index=False
    )
    deterministic_summary.to_csv(
        args.output / "deterministic_comparison_137.csv", index=False
    )
    pd.DataFrame([calibration_association]).to_csv(
        args.output / "extrinsic_calibration_association_137.csv", index=False
    )
    (args.output / "joint_accuracy_main.tex").write_text(
        render_main_joint_table(joint_summary), encoding="utf-8"
    )
    (args.output / "joint_accuracy_all70.tex").write_text(
        render_all_joint_table(joint_summary), encoding="utf-8"
    )
    (args.output / "extrinsic_comparison.tex").write_text(
        render_extrinsic_table(extrinsic_test_summary, scope="heldout"),
        encoding="utf-8",
    )
    (args.output / "extrinsic_comparison_all137.tex").write_text(
        render_extrinsic_table(extrinsic_all_summary, scope="all_participants"),
        encoding="utf-8",
    )
    (args.output / "deterministic_comparison_all.tex").write_text(
        render_deterministic_table(deterministic_summary), encoding="utf-8"
    )
    print(
        f"generated comparison_people={unified_person['person_id'].nunique()} "
        f"test_people={len(test_people)} joints={len(joint_summary)} "
        f"person_source={person_source} output={args.output}"
    )


if __name__ == "__main__":
    main()
