#!/usr/bin/env python3
"""Fail fast on evidence and formatting mistakes in the manuscript source."""

from __future__ import annotations

import csv
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COHORT_LABELS = {
    "trunk_axial_rotation_rom": "Axial rotation ROM",
    "angular_speed_p95": "Angular speed (P95)",
    "peak_rotation_phase": "Peak rotation phase",
    "trunk_tilt_p95": "Trunk tilt (P95)",
    "wrist_lead_p95": "Wrist wrapping (P95)",
    "cycle_duration": "Cycle duration",
    "log_dimensionless_angular_jerk": "Log dimensionless jerk",
    "whole_body_repeatability": "Whole-body repeatability",
}


def tex_sources() -> str:
    paths = [ROOT / "manuscript.tex"]
    paths.extend(sorted((ROOT / "sections").glob("*.tex")))
    paths.extend(sorted((ROOT / "tables").glob("*.tex")))
    paths.extend(sorted((ROOT / "figures").glob("*.tex")))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths if path.exists())


def verify_cohort_tables() -> None:
    core_path = ROOT / "artifacts" / "cohort_core_mixed_models.csv"
    variability_path = ROOT / "artifacts" / "cohort_variability_results.csv"
    sensitivity_path = (
        ROOT / "artifacts" / "cohort_sensitivity_mixed_models.csv"
    )
    table_path = ROOT / "tables" / "cohort_cycle_results.tex"
    sensitivity_table_path = ROOT / "tables" / "cohort_cycle_sensitivity.tex"
    required = (
        core_path,
        variability_path,
        sensitivity_path,
        table_path,
        sensitivity_table_path,
        ROOT / "figures" / "cohort_cycle_analysis.pdf",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"cohort analysis artifacts are missing: {missing}")

    core_rows = list(csv.DictReader(core_path.open(encoding="utf-8")))
    variability_rows = {
        row["outcome"]: row
        for row in csv.DictReader(variability_path.open(encoding="utf-8"))
    }
    if [row["outcome"] for row in core_rows] != list(COHORT_LABELS):
        raise SystemExit("cohort core CSV does not contain the eight prespecified outcomes")
    table = table_path.read_text(encoding="utf-8")
    for row in core_rows:
        outcome = row["outcome"]
        variability = variability_rows[outcome]
        expected = (
            f"{float(row['cohort_effect']):.4f} "
            f"[{float(row['cohort_ci_low']):.4f}, "
            f"{float(row['cohort_ci_high']):.4f}] & "
            f"{float(row['cohort_p_holm']):.4f}"
        )
        table_line = next(
            (line for line in table.splitlines() if line.startswith(f"{COHORT_LABELS[outcome]} &")),
            "",
        )
        if expected not in table_line:
            raise SystemExit(f"cohort table does not match core CSV for {outcome}")
        variability_fragment = (
            f"{float(variability['median_difference']):.4f} "
            f"({float(variability['p_holm']):.4f})"
        )
        if variability_fragment not in table_line:
            raise SystemExit(f"cohort table does not match variability CSV for {outcome}")

    sensitivity_rows = list(csv.DictReader(sensitivity_path.open(encoding="utf-8")))
    sensitivity = {
        (row["outcome"], row["source"]): row
        for row in sensitivity_rows
    }
    source_order = (
        "oof_a6",
        "face",
        "side",
        "deterministic",
    )
    sensitivity_table = sensitivity_table_path.read_text(encoding="utf-8")
    for outcome in ("angular_speed_p95", "log_dimensionless_angular_jerk"):
        label = COHORT_LABELS[outcome]
        if f"{label} &" not in sensitivity_table:
            raise SystemExit(f"cohort sensitivity table lacks {outcome}")
        table_line = sensitivity_table.split(f"{label} &", 1)[1].split(
            r"\\",
            1,
        )[0]
        for source in source_order:
            row = sensitivity[(outcome, source)]
            fragment = (
                f"{float(row['cohort_effect']):.4f} "
                f"[{float(row['cohort_ci_low']):.4f}, "
                f"{float(row['cohort_ci_high']):.4f}] "
                f"({float(row['cohort_p_holm_within_source']):.4f})"
            )
            if fragment not in table_line:
                raise SystemExit(
                    "cohort sensitivity table does not match CSV for "
                    f"{outcome}/{source}"
                )


def main() -> None:
    manuscript = (ROOT / "manuscript.tex").read_text(encoding="utf-8")
    sources = tex_sources()
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", manuscript, flags=re.S)
    if abstract_match is None:
        raise SystemExit("missing abstract")
    abstract_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract_match.group(1))
    if len(abstract_words) > 250:
        raise SystemExit(f"abstract has {len(abstract_words)} words; maximum is 250")

    highlights = [line.strip() for line in (ROOT / "highlights.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 3 <= len(highlights) <= 5:
        raise SystemExit(f"expected 3-5 highlights, found {len(highlights)}")
    too_long = [(line, len(line)) for line in highlights if len(line) > 85]
    if too_long:
        raise SystemExit(f"highlights exceed 85 characters: {too_long}")

    for marker in ("TODO", "TBD", "FIXME", "XXX"):
        if marker in sources:
            raise SystemExit(f"uncontrolled placeholder found: {marker}")

    forbidden = (
        "triangulated ground truth",
        "triangulated 3D ground truth",
        "independent 3D ground truth",
        "motion-capture ground truth",
    )
    lowered = sources.lower()
    for phrase in forbidden:
        if phrase in lowered:
            raise SystemExit(f"forbidden evidence claim found: {phrase}")

    if "Kaixu Chen" not in manuscript or "chenkaixusan@gmail.com" not in manuscript:
        raise SystemExit("author metadata is incomplete")

    bib_text = (ROOT / "references.bib").read_text(encoding="utf-8") if (ROOT / "references.bib").exists() else ""
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    cited = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", sources):
        cited.update(key.strip() for key in group.split(","))
    missing = sorted(cited - bib_keys)
    if missing:
        raise SystemExit(f"citation keys missing from references.bib: {missing}")

    summary_path = ROOT / "artifacts/deterministic_summary.csv"
    table_path = ROOT / "tables/deterministic_baselines.tex"
    if summary_path.exists() and table_path.exists():
        rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
        table = table_path.read_text(encoding="utf-8")
        if len(rows) != 9 or any(f"{float(row['mean']):.4f}" not in table for row in rows):
            raise SystemExit("deterministic table does not match generated summary")

    verify_cohort_tables()

    learned_table = ROOT / "tables" / "learned_results.tex"
    unity_table = ROOT / "tables" / "unity_benchmark.tex"
    if not learned_table.is_file() or not unity_table.is_file():
        raise SystemExit("held-out learned or Unity table is missing")
    learned_text = learned_table.read_text(encoding="utf-8")
    unity_text = unity_table.read_text(encoding="utf-8")
    for fragment in ("held-out test set ($N=14$)", "60.78", "A3-relative"):
        if fragment not in learned_text:
            raise SystemExit(f"learned table lacks audited fragment: {fragment}")
    for fragment in ("Unity native 3D", "178.506", "30.259"):
        if fragment not in unity_text:
            raise SystemExit(f"Unity table lacks audited fragment: {fragment}")

    required_cohort_language = (
        "person-disjoint",
        "928 cycles",
        "80 elderly",
        "57 student",
        "cannot be interpreted as ageing effects",
        "mid-repetition reference",
        "pose-source-sensitive",
    )
    for phrase in required_cohort_language:
        if phrase.lower() not in lowered:
            raise SystemExit(f"required cohort evidence boundary is missing: {phrase}")

    # The experimental matrix is complete (single-seed learned study A4--A9,
    # per-family robustness, and the temporal-offset sweep are all reported), so a
    # pending-result marker is no longer required; the count is still reported.
    pending = len(re.findall(r"\\resultpending\{", sources))

    print(f"manuscript check passed: abstract_words={len(abstract_words)} highlights={len(highlights)} citations={len(cited)} pending_results={pending}")


if __name__ == "__main__":
    main()
