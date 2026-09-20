"""Build the cross-dataset main comparison matrix (methods x datasets).

Rows are fusion methods grouped into blocks; columns are the three evaluation
datasets:

* private gymnastics data, 14 held-out participants, person-level MPJPE after
  one similarity alignment per cycle plus framewise hip centring against the
  triangulated pseudo-reference (the Table 1 protocol);
* FreeMan, ten subjects / 552 sessions, subject-balanced PA-MPJPE and
  session-level Sim3 MPJPE against the FreeMan markerless reference;
* Unity, 199 rendered samples, sequence-level similarity-aligned MPJPE and
  trunk-angle MAE against native 3D.

Each cell is read from the artefacts already produced by the corresponding
pipeline; missing artefacts render as "--" so the table can be regenerated
incrementally while experiments are still running.

Usage (repository root):
    PYTHONPATH=src python paper/sports_engineering/scripts/generate_main_matrix.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_comparison_tables import (  # noqa: E402
    load_test_people,
    reevaluate_compact_metrics,
)
from fusion.keypoints.config import load_skeleton_spec  # noqa: E402

PRIVATE_SPLIT = ROOT / "src/configs/shared/folds/paper_137_a6_split.json"
_PRIVATE_EVALUATION_ROOT = ROOT / "local/runs/fuse_rotation_aware/evaluation"
_PRIVATE_LEARNED_CANDIDATES = (
    # Preferred: the evaluation that also contains the B1 plain-TCN run.
    _PRIVATE_EVALUATION_ROOT
    / "all137_a4_e100_seed0+all137_a5_e100_seed0+all137_a6_e100_seed0+"
    "all137_a7_e100_seed0+all137_a8_e100_seed0+all137_a9_e100_seed0+all137_b1_e100_seed0+all137_b2_e100_seed0"
    / "metrics_by_person.csv",
    _PRIVATE_EVALUATION_ROOT
    / "all137_a4_e100_seed0+all137_a5_e100_seed0+all137_a6_e100_seed0+"
    "all137_a7_e100_seed0+all137_a8_e100_seed0+all137_a9_e100_seed0+all137_b1_e100_seed0"
    / "metrics_by_person.csv",
    _PRIVATE_EVALUATION_ROOT
    / "all137_a4_e100_seed0+all137_a5_e100_seed0+all137_a6_e100_seed0+"
    "all137_a7_e100_seed0+all137_a8_e100_seed0+all137_a9_e100_seed0"
    / "metrics_by_person.csv",
)
PRIVATE_LEARNED_METRICS = next(
    (path for path in _PRIVATE_LEARNED_CANDIDATES if path.exists()),
    _PRIVATE_LEARNED_CANDIDATES[-1],
)
PRIVATE_EXTRINSIC_ROOT = ROOT / "local/runs/fuse_extrinsic_baselines"
PRIVATE_EXTERNAL_ROOT = ROOT / "local/runs/fuse_external_baselines"
PRIVATE_TRIANGULATED_ROOT = Path(
    "/work/1/HP260146/chenkaixu/gymnastics/sam3d_triangulated/person"
)
SKELETON = ROOT / "src/configs/shared/skeleton_mhr70.yaml"
FREEMAN_ROOTS = (
    ROOT / "local/runs/freeman_benchmark_cluster",
    ROOT / "local/runs/freeman_external_baselines",
    ROOT / "local/runs/freeman_trained_fusion",
)
FREEMAN_SUBJECTS = (1, 7, 9, 12, 15, 19, 22, 29, 35, 36)
UNITY_SUMMARY = ROOT / "local/runs/unity_benchmark/evaluation/metrics_summary.csv"
UNITY_SUPERVISED = (
    ROOT / "local/runs/unity_benchmark/supervised_finetune/evaluation/by_ablation.csv"
)
OUTPUT = ROOT / "paper/sports_engineering/generated"


@dataclass(frozen=True)
class Row:
    block: str
    label: str
    private: str | None
    freeman: str | None
    unity: str | None
    note: str = ""


ROWS: tuple[Row, ...] = (
    Row("Single view", "Face view", "A0", "view_a", "cam0"),
    Row("Single view", "Side view", "A1", "view_b", "cam1"),
    Row("Naive averaging", "World-coordinate average", "avg_world_face_ref", "avg_world_face_ref", "avg_world_face_ref"),
    Row("Naive averaging", "Root alignment and average", "root_face_stable", "root_face_stable", "root_face_stable"),
    Row("Coordinate-normalized averaging", "Body-frame average", "avg_body_current", "avg_body_current", "avg_body_current"),
    Row("Coordinate-normalized averaging", "Similarity alignment, stable joints", "sim3_face_stable", "sim3_face_stable", "sim3_face_stable"),
    Row("Coordinate-normalized averaging", "Similarity alignment, output smoothing", "sim3_face_stable_smooth_kpt", "sim3_face_stable_smooth_kpt", "sim3_face_stable_smooth_kpt"),
    Row("Classical fusion and filtering", "Kalman filter fusion", "kalman_body_fusion", "kalman_body_fusion", "kalman_body_fusion"),
    Row("Classical fusion and filtering", "Kalman RTS smoother fusion", "kalman_rts_body_fusion", "kalman_rts_body_fusion", "kalman_rts_body_fusion"),
    Row("Classical fusion and filtering", "Reliability-weighted average", "jitter_weighted_body_average", "jitter_weighted_body_average", "jitter_weighted_body_average"),
    Row("Classical fusion and filtering", "Butterworth 6 Hz on body-frame average", "butterworth_body_average", "butterworth_body_average", "butterworth_body_average"),
    Row("External learned refiner", "SmoothNet (H3.6M weights), zero-shot", "smoothnet_body_average", "smoothnet_body_average", "smoothnet_body_average"),
    Row("External learned refiner", "Plain TCN, same self-supervision, unbounded (B1)", "B1", "rotation_aware:all137_b1_e100_seed0", "B1"),
    Row("External learned refiner", "Plain TCN, same self-supervision, 5 cm bound (B2)", "B2", "rotation_aware:all137_b2_e100_seed0", "B2"),
    Row("Self-supervised fusion (ours)", "A4 spatial objectives", "A4", "rotation_aware:all137_a4_e100_seed0", "A4"),
    Row("Self-supervised fusion (ours)", "A5 rotation/temporal", "A5", "rotation_aware:all137_a5_e100_seed0", "A5"),
    Row("Self-supervised fusion (ours)", "A6 complete model", "A6", "rotation_aware:all137_a6_e100_seed0", "A6"),
    Row("Self-supervised fusion (ours)", "A6 trained in-domain", None, "rotation_aware_trained:a6_e12_s0", "A6:unity_supervised", "FreeMan: subject-disjoint self-supervised; Unity: direction-held-out, Unity 3D supervised"),
    Row("Camera-assisted / calibrated", "Estimated camera rotation (Extrinsic-R)", "extrinsic_r_average", None, None),
    Row("Camera-assisted / calibrated", "Calibrated 2D triangulation", None, None, "triangulation_sam3d2d", "private pseudo-reference and FreeMan reference are themselves triangulations"),
)


def _fmt(value: float | None, digits: int = 2) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


# --------------------------------------------------------------------------- #
# private
# --------------------------------------------------------------------------- #


PRIVATE_UNITS: pd.DataFrame | None = None
FREEMAN_UNITS: pd.DataFrame | None = None


def private_column(reevaluate_roots: dict[str, Path]) -> dict[str, tuple[float, float]]:
    """Person-level mean and SD (mm) over the 14 held-out participants."""
    global PRIVATE_UNITS
    test_people = load_test_people(PRIVATE_SPLIT)
    frames: list[pd.DataFrame] = []
    if PRIVATE_LEARNED_METRICS.exists():
        table = pd.read_csv(PRIVATE_LEARNED_METRICS, dtype={"person_id": str})
        table = table[table["person_id"].isin(test_people)]
        counts = table.groupby("method")["person_id"].nunique()
        table = table[table["method"].isin(counts[counts == len(test_people)].index)]
        frames.append(table[["person_id", "method", "mpjpe"]])
    available = {m: r for m, r in reevaluate_roots.items() if (r / f"person_{test_people[0]}").exists()}
    if available:
        skeleton = load_skeleton_spec(SKELETON)
        person, _ = reevaluate_compact_metrics(
            available, test_people, PRIVATE_TRIANGULATED_ROOT, skeleton
        )
        frames.append(person[["person_id", "method", "mpjpe"]])
    if not frames:
        return {}
    units = pd.concat(frames, ignore_index=True).drop_duplicates(["person_id", "method"], keep="last")
    units["value_mm"] = units["mpjpe"].astype(float) * 1000.0
    PRIVATE_UNITS = units.rename(columns={"person_id": "unit"})[["unit", "method", "value_mm"]]
    values: dict[str, tuple[float, float]] = {}
    for method, group in units.groupby("method"):
        mm = group["value_mm"].to_numpy(dtype=float)
        values[str(method)] = (float(mm.mean()), float(mm.std(ddof=1)))
    return values


# --------------------------------------------------------------------------- #
# FreeMan
# --------------------------------------------------------------------------- #


def freeman_column() -> dict[str, tuple[float, float]]:
    """Subject-balanced (PA-MPJPE, Sim3 MPJPE) in mm over the ten subjects."""
    records: list[dict[str, object]] = []
    for root in FREEMAN_ROOTS:
        metrics_root = root / "evaluation" / "session_metrics"
        for subject in FREEMAN_SUBJECTS:
            path = metrics_root / f"subject_{subject:02d}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload["rows"]:
                if row.get("classification", "VALID") != "VALID" and not str(
                    row["method"]
                ).startswith("sim3_face_stable_joint_weight"):
                    continue
                records.append(
                    {
                        "subject_id": int(row["subject_id"]),
                        "session_id": row["session_id"],
                        "method": row["method"],
                        "pa": float(row["pa_mpjpe_mm"]),
                        "sim3": float(row["sim3_mpjpe_mm"]),
                    }
                )
    if not records:
        return {}
    table = pd.DataFrame(records).drop_duplicates(["subject_id", "session_id", "method"])
    counts = table.groupby("method")["session_id"].nunique()
    complete = counts[counts == counts.max()].index
    table = table[table["method"].isin(complete)]
    by_subject = table.groupby(["method", "subject_id"], as_index=False)[["pa", "sim3"]].mean()
    global FREEMAN_UNITS
    FREEMAN_UNITS = by_subject.rename(columns={"subject_id": "unit", "pa": "value_mm"})[
        ["unit", "method", "value_mm"]
    ]
    by_method = by_subject.groupby("method")[["pa", "sim3"]].mean()
    return {
        str(method): (float(row["pa"]), float(row["sim3"]))
        for method, row in by_method.iterrows()
    }


# --------------------------------------------------------------------------- #
# Unity
# --------------------------------------------------------------------------- #


def unity_column() -> dict[str, tuple[float, float]]:
    values: dict[str, tuple[float, float]] = {}
    if UNITY_SUMMARY.exists():
        table = pd.read_csv(UNITY_SUMMARY)
        for _, row in table.iterrows():
            values[str(row["method"])] = (float(row["mpjpe_mm"]), float(row["angle_mae_deg"]))
    if UNITY_SUPERVISED.exists():
        table = pd.read_csv(UNITY_SUPERVISED)
        for _, row in table.iterrows():
            values[f"{row['ablation']}:unity_supervised"] = (
                float(row["macro_mpjpe_mm"]),
                float(row["macro_angle_mae_deg"]),
            )
    return values


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def paired_statistics(
    units: pd.DataFrame | None,
    *,
    dataset: str,
    reference: str,
    key_column: str,
    seed: int = 20260918,
    samples: int = 10000,
) -> pd.DataFrame:
    """Unit-level paired differences (method minus reference) for every table row."""
    from scipy.stats import wilcoxon

    if units is None or reference not in set(units["method"]):
        return pd.DataFrame()
    wide = units.pivot(index="unit", columns="method", values="value_mm")
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    p_values: list[float] = []
    for row in ROWS:
        key = getattr(row, key_column)
        if key is None or key == reference or key not in wide.columns:
            continue
        paired = wide[[key, reference]].dropna()
        diff = (paired[key] - paired[reference]).to_numpy(dtype=float)
        if len(diff) < 2:
            continue
        resampled = rng.choice(diff, size=(samples, len(diff)), replace=True).mean(axis=1)
        p_value = 1.0 if np.allclose(diff, 0.0) else float(wilcoxon(diff).pvalue)
        p_values.append(p_value)
        records.append(
            {
                "dataset": dataset,
                "block": row.block,
                "method": row.label,
                "key": key,
                "reference": reference,
                "units": int(len(diff)),
                "difference_mm": float(diff.mean()),
                "ci_low_mm": float(np.quantile(resampled, 0.025)),
                "ci_high_mm": float(np.quantile(resampled, 0.975)),
                "improved_units": int((diff < 0).sum()),
                "p_wilcoxon": p_value,
            }
        )
    table = pd.DataFrame(records)
    if not table.empty:
        table["p_holm"] = _holm(p_values)
    return table


def build_table(
    private: dict[str, tuple[float, float]],
    freeman: dict[str, tuple[float, float]],
    unity: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    rows = []
    for row in ROWS:
        p = private.get(row.private) if row.private else None
        f = freeman.get(row.freeman) if row.freeman else None
        u = unity.get(row.unity) if row.unity else None
        rows.append(
            {
                "block": row.block,
                "method": row.label,
                "private_key": row.private,
                "freeman_key": row.freeman,
                "unity_key": row.unity,
                "private_mpjpe_mm": p[0] if p else np.nan,
                "private_sd_mm": p[1] if p else np.nan,
                "freeman_pa_mpjpe_mm": f[0] if f else np.nan,
                "freeman_sim3_mpjpe_mm": f[1] if f else np.nan,
                "unity_mpjpe_mm": u[0] if u else np.nan,
                "unity_angle_mae_deg": u[1] if u else np.nan,
                "note": row.note,
            }
        )
    return pd.DataFrame(rows)


def render_latex(table: pd.DataFrame) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\caption{Cross-dataset comparison. Private: person-level mean $\\pm$ SD "
        "MPJPE (mm) on 14 held-out participants after one similarity alignment per "
        "cycle and framewise hip centring to the triangulated pseudo-reference. "
        "FreeMan: subject-balanced per-frame Procrustes (PA) and session-level Sim3 "
        "MPJPE (mm) over ten subjects / 552 sessions against the markerless "
        "multi-view reference. Unity: sequence-level similarity-aligned MPJPE (mm) "
        "and trunk-angle MAE ($^\\circ$) against native 3D over 199 samples. "
        "All learned rows except ``trained in-domain'' use private-data checkpoints "
        "zero-shot. ``--'' marks a cell that is not defined for that dataset.}",
        "\\label{tab:main-matrix}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        " & & Private & \\multicolumn{2}{c}{FreeMan} & \\multicolumn{2}{c}{Unity}\\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Block & Method & MPJPE & PA & Sim3 & MPJPE & Angle\\\\",
        "\\midrule",
    ]
    previous_block = None
    for _, row in table.iterrows():
        if previous_block is not None and row["block"] != previous_block:
            lines.append("\\addlinespace")
        block = row["block"] if row["block"] != previous_block else ""
        previous_block = row["block"]
        private = (
            f"{_fmt(row['private_mpjpe_mm'])} $\\pm$ {_fmt(row['private_sd_mm'])}"
            if np.isfinite(row["private_mpjpe_mm"])
            else "--"
        )
        lines.append(
            f"{block} & {row['method']} & {private} & "
            f"{_fmt(row['freeman_pa_mpjpe_mm'])} & {_fmt(row['freeman_sim3_mpjpe_mm'])} & "
            f"{_fmt(row['unity_mpjpe_mm'])} & {_fmt(row['unity_angle_mae_deg'])}\\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--skip-private-reevaluation",
        action="store_true",
        help="use only the cached learned-metrics CSV for the private column",
    )
    args = parser.parse_args()
    reevaluate_roots: dict[str, Path] = {}
    if not args.skip_private_reevaluation:
        for method in ("extrinsic_r_average", "extrinsic_r_quality_average"):
            reevaluate_roots[method] = PRIVATE_EXTRINSIC_ROOT / method
        from fusion.baselines.methods import BASELINE_METHODS

        for method in BASELINE_METHODS:
            reevaluate_roots[method] = PRIVATE_EXTERNAL_ROOT / method
    table = build_table(private_column(reevaluate_roots), freeman_column(), unity_column())
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "main_matrix.csv", index=False)
    paired = pd.concat(
        [
            paired_statistics(PRIVATE_UNITS, dataset="private_test14", reference="A6", key_column="private"),
            paired_statistics(
                FREEMAN_UNITS,
                dataset="freeman_pa",
                reference="rotation_aware:all137_a6_e100_seed0",
                key_column="freeman",
            ),
        ],
        ignore_index=True,
    )
    paired.to_csv(args.output / "main_matrix_paired_vs_a6.csv", index=False)
    if not paired.empty:
        print()
        print(
            paired[["dataset", "method", "units", "difference_mm", "ci_low_mm", "ci_high_mm", "improved_units", "p_holm"]]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )
    (args.output / "main_matrix.tex").write_text(render_latex(table), encoding="utf-8")
    display = table[
        ["block", "method", "private_mpjpe_mm", "freeman_pa_mpjpe_mm", "freeman_sim3_mpjpe_mm", "unity_mpjpe_mm", "unity_angle_mae_deg"]
    ]
    print(display.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"wrote {args.output / 'main_matrix.csv'} and main_matrix.tex")


if __name__ == "__main__":
    main()
