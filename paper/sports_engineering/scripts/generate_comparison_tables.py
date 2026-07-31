#!/usr/bin/env python3
"""Generate source-checked extrinsic and per-joint paper tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from gymnastics.analysis.cohort_cycle.joints import MAJOR_JOINT_INDICES
from gymnastics.common.skeletons import MHR70_NAMES
from gymnastics.fusion.deterministic.experiment_matrix import (
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


def evaluate_matched_joint_metrics(
    person_id: str,
    method: str,
    matched_cycles: Sequence[tuple[np.ndarray, np.ndarray]],
    skeleton: SkeletonSpec,
) -> pd.DataFrame:
    errors_by_joint: list[list[np.ndarray]] = [
        [] for _ in range(len(skeleton.joint_names))
    ]
    for candidate, reference in matched_cycles:
        errors, valid = _external_errors(
            np.asarray(candidate, dtype=np.float64),
            np.asarray(reference, dtype=np.float64),
            skeleton,
            alignment="similarity",
        )
        for joint in range(errors.shape[1]):
            errors_by_joint[joint].append(errors[:, joint][valid[:, joint]])
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
    return pd.DataFrame(rows)


def reevaluate_compact_joint_metrics(
    method_roots: Mapping[str, Path],
    people: tuple[str, ...],
    triangulated_root: Path,
    skeleton: SkeletonSpec,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
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
            frames.append(
                evaluate_matched_joint_metrics(
                    person_id, method, matched_cycles, skeleton
                )
            )
    return pd.concat(frames, ignore_index=True)


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
same-video pseudo-reference. Lower is better; bold indicates the lowest
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
lowest descriptive value in each row.}\\label{tab:joint-accuracy-all70}\\\\
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


def render_extrinsic_table(summary: pd.DataFrame) -> str:
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
            improved = f"{int(row.improved_people)}/137"
        rows.append(
            f"{DISPLAY_NAMES[row.method]} & {row.mean_mm:.3f} $\\pm$ "
            f"{row.std_mm:.3f} & {comparison} & {p_value} & {improved} \\\\"
        )
    return """\\begin{table*}[t]
\\caption{Camera-assisted deterministic comparison over all 137 participants.
The difference is method minus the calibration-free body-frame average, with a
participant-bootstrap 95\\% confidence interval. $p$ values are Holm-adjusted
paired Wilcoxon tests. Evaluation uses a same-video pseudo-reference.}
\\label{tab:extrinsic-comparison}
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{3pt}
\\begin{tabular}{lcccc}
\\toprule
Method & MPJPE (mm) & Difference [95\\% CI] & $p_{\\rm Holm}$ & Improved people \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table*}
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
        "deterministic_person": root / "local/runs/fuse_experiments/metrics_by_person.csv",
        "extrinsic_root": root / "local/runs/fuse_extrinsic_baselines",
        "extrinsic_person": root / "local/runs/fuse_extrinsic_baselines/metrics_by_person.csv",
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
    parser.add_argument("--deterministic-person", type=Path, default=defaults["deterministic_person"])
    parser.add_argument("--extrinsic-root", type=Path, default=defaults["extrinsic_root"])
    parser.add_argument("--extrinsic-person", type=Path, default=defaults["extrinsic_person"])
    parser.add_argument("--triangulated-root", type=Path, default=defaults["triangulated_root"])
    parser.add_argument("--skeleton", type=Path, default=defaults["skeleton"])
    parser.add_argument("--output", type=Path, default=defaults["output"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_people = load_test_people(args.split)
    learned_joint = load_joint_metrics(args.learned_joint, LEARNED_JOINT_METHODS)
    learned_joint["evaluation_protocol"] = JOINT_EVALUATION_PROTOCOL
    skeleton = load_skeleton_spec(args.skeleton)
    extrinsic_joint = reevaluate_compact_joint_metrics(
        {
            method: args.extrinsic_root / method
            for method in EXTRINSIC_JOINT_METHODS
        },
        test_people,
        args.triangulated_root,
        skeleton,
    )
    joint_summary = build_joint_summary(learned_joint, extrinsic_joint, test_people)
    deterministic_person = load_person_metrics(
        args.deterministic_person, (PERSON_BASELINE_METHOD,)
    )
    extrinsic_person = load_person_metrics(
        args.extrinsic_person, EXTRINSIC_JOINT_METHODS
    )
    extrinsic_summary = build_extrinsic_summary(
        deterministic_person, extrinsic_person
    )

    args.output.mkdir(parents=True, exist_ok=True)
    extrinsic_joint.to_csv(
        args.output / "extrinsic_joint_metrics_test14.csv", index=False
    )
    joint_summary.to_csv(args.output / "joint_accuracy_test14.csv", index=False)
    extrinsic_summary.to_csv(args.output / "extrinsic_comparison_137.csv", index=False)
    (args.output / "joint_accuracy_main.tex").write_text(
        render_main_joint_table(joint_summary), encoding="utf-8"
    )
    (args.output / "joint_accuracy_all70.tex").write_text(
        render_all_joint_table(joint_summary), encoding="utf-8"
    )
    (args.output / "extrinsic_comparison.tex").write_text(
        render_extrinsic_table(extrinsic_summary), encoding="utf-8"
    )
    print(
        f"generated extrinsic_people={extrinsic_person['person_id'].nunique()} "
        f"test_people={len(test_people)} joints={len(joint_summary)} output={args.output}"
    )


if __name__ == "__main__":
    main()
