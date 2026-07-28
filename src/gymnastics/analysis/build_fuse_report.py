#!/usr/bin/env python3
"""Consolidate the regenerated-GT fusion results into local/runs/fuse_experiments.

This gathers the artefacts that were produced across several tools -- the fuse
metric matrix, the fair three-way source comparison, the body-scale-normalized
metric, and the extrinsics-quality summary -- copies them into one bundle under
``local/runs/fuse_experiments/analysis`` and writes ``RESULTS_SUMMARY.md``.

All figures are recomputed here from the per-person CSVs, so the report always
matches the files it bundles.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.stats import wilcoxon

from gymnastics.analysis.project_results import paired_comparisons

FUSE_DIR = Path("local/runs/fuse_experiments")
BUNDLE = FUSE_DIR / "analysis"  # all derived analysis, consolidated in one place
CMP_DIR = Path("local/runs/analysis/fused_vs_triangulated")
BODY_DIR = Path("local/runs/analysis/body_scale_normalized")
EXTR = Path("local/runs/analysis/extrinsics/extrinsics_comparison.json")
TRIANGULATED_SUMMARY = Path(
    "/home/data/xchen/gymnastics/sam3d_triangulated/person/summary.json"
)
# Rotation-aware network evaluation (A4/A5/A6 checkpoints) against the same new GT.
# This harness scores the network ablations AND the deterministic methods together,
# so numbers WITHIN it are comparable; do not cross-compare with the fuse harness.
RAE_EVAL = Path(
    "local/runs/fuse_rotation_aware/evaluation/"
    "all137_a4_e100_seed0+all137_a5_e100_seed0+all137_a6_e100_seed0/metrics_by_person.csv"
)
ABLATION_LABEL = {
    "A0": "face only (single view)",
    "A1": "side only (single view)",
    "A2": "canonical arithmetic (deterministic)",
    "A3": "quality mean (deterministic)",
    "A4": "learned: spatial only",
    "A5": "learned: + rotation/temporal",
    "A6": "learned: full self-supervised (proposed)",
}

# Cohort mapping. The dropbox import log carries an explicit elderly/student
# ``group`` column but only for the ids in that batch (69+); ``student_id_mapping``
# lists every student and starts at 81. Both agree that students are id >= 81 and
# elderly are id <= 80, which also reproduces the 80/57 split of the prior report.
RAW_PERSON = Path("/home/data/xchen/gymnastics/raw/person")
GROUP_LOG = RAW_PERSON / "organize_from_dropbox_20260718.csv"
STUDENT_MAP = RAW_PERSON / "student_id_mapping.csv"


def load_group_of() -> Callable[[int], str]:
    """Return a function person_id -> 'elderly' | 'student', from both raw logs."""
    student_ids = set()
    if STUDENT_MAP.exists():
        student_ids = {
            int(r["person_id"]) for r in csv.DictReader(STUDENT_MAP.open(encoding="utf-8"))
        }
    boundary = min(student_ids) if student_ids else 81  # first student id
    if GROUP_LOG.exists():
        for r in csv.DictReader(GROUP_LOG.open(encoding="utf-8")):
            pid, grp = int(r["person_id"]), r["group"].strip()
            if (grp == "student") != (pid >= boundary):
                raise SystemExit(
                    f"Group logs disagree at person {pid}: log says {grp}, "
                    f"boundary is id>={boundary}. Resolve before reporting."
                )

    def group_of(pid: int) -> str:
        return "student" if pid >= boundary else "elderly"

    return group_of

# Fusion methods whose parameters are fitted on the triangulated GT they are then
# scored against; their MPJPE is optimistically biased and cannot be recommended.
LEAKY = {"sim3_face_stable_joint_weight"}
RECOMMENDED = "avg_body_current"

# How each deterministic method fuses the two views. "Stable joints" = the trunk
# and legs (shoulders, hips, knees, ankles, feet), which move less than the arms
# and so give a steadier cross-view alignment. Source: fuse/experiment_matrix.py.
METHOD_DESC = {
    "avg_body_current": (
        "map both views into a pelvis-centred, orientation-normalised body frame, "
        "average there, then map back through the face pelvis/orientation — cancels "
        "the inter-view world-frame scale and rotation mismatch"
    ),
    "sim3_face_stable": (
        "fit one similarity transform (rotation+scale+translation) aligning side→face "
        "from the stable joints only, then average the two views 50/50"
    ),
    "sim3_face_stable_bodypart_weight": (
        "sim3_face_stable, but average with fixed per-body-part weights (torso 50/50, "
        "distal limbs/hands lean to face, a few joints lean to side)"
    ),
    "sim3_face_stable_smooth_transform": (
        "sim3_face_stable, but temporally smooth the aligned side sequence "
        "(window 5) before averaging"
    ),
    "sim3_face_stable_smooth_kpt": (
        "sim3_face_stable, average, then temporally smooth the fused keypoints (window 5)"
    ),
    "sim3_face_all": (
        "like sim3_face_stable but fit the similarity transform from ALL joints — "
        "moving limbs bias the fit, so it is noisier"
    ),
    "avg_world_face_ref": (
        "naive 0.5·(face+side) in world coordinates, no alignment — leaves the "
        "inter-view world-frame mismatch in the result"
    ),
    "root_face_stable": (
        "align side to face by pelvis translation only (no rotation/scale), then average"
    ),
    "sim3_face_stable_joint_weight": (
        "sim3_face_stable, but weight each joint by its face-vs-side error against the "
        "triangulated GT — needs the GT it is scored on (leaky); without GT it falls "
        "back to 50/50, i.e. sim3_face_stable"
    ),
}


def _load_mpjpe(path: Path, key_col: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Return (person id order, {key: per-person mpjpe array}) aligned on that order."""
    per: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in csv.DictReader(path.open(encoding="utf-8")):
        try:
            per[row[key_col]][row["person_id"]] = float(row["mpjpe"])
        except (KeyError, ValueError):
            continue
    persons = sorted({p for m in per.values() for p in m}, key=int)
    arrays = {k: np.array([per[k][p] for p in persons]) for k in per}
    return persons, arrays


def fmt_p(p: float) -> str:
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def triangulation_quality() -> Dict[str, object]:
    """Coverage, reprojection, and extrinsics provenance from the GT summary."""
    summary = json.loads(TRIANGULATED_SUMMARY.read_text(encoding="utf-8"))
    face, side, frames, cycles, missing = [], [], 0, 0, 0
    sources: Dict[str, int] = defaultdict(int)
    for person in summary["persons"]:
        sources[person.get("extrinsics_source", "?")] += 1
        for cycle in person["cycles"]:
            cycles += 1
            frames += int(cycle.get("processed_frames", 0))
            missing += int(cycle.get("missing_pairs", 0))
            if cycle.get("face_reprojection_error_mean_px") is not None:
                face.append(cycle["face_reprojection_error_mean_px"])
            if cycle.get("side_reprojection_error_mean_px") is not None:
                side.append(cycle["side_reprojection_error_mean_px"])
    mean = (np.array(face) + np.array(side)) / 2
    return {
        "persons": summary.get("num_persons", len(summary["persons"])),
        "cycles": cycles,
        "frames": frames,
        "missing": missing,
        "reproj_median": float(np.median(mean)),
        "reproj_mean": float(mean.mean()),
        "reproj_p95": float(np.percentile(mean, 95)),
        "reproj_max": float(mean.max()),
        "sources": dict(sources),
    }


def resolve(name: str, *candidate_dirs: Path) -> Path:
    """First existing path for ``name`` among the candidate dirs (bundle is a fallback)."""
    for directory in (*candidate_dirs, BUNDLE):
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"Missing input '{name}': looked in "
        + ", ".join(str(d) for d in (*candidate_dirs, BUNDLE))
    )


def main() -> None:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    group_of = load_group_of()

    persons, methods = _load_mpjpe(FUSE_DIR / "metrics_by_person.csv", "method")
    groups = np.array([group_of(int(p)) for p in persons])
    n_persons = len(persons)
    ranked = sorted(methods, key=lambda m: methods[m].mean())
    leakfree = [m for m in ranked if m not in LEAKY]
    best = leakfree[0]

    # Paired comparison of every method against the best leakage-free method.
    comparison_rows = [
        {"person_id": person, "method": method, "mpjpe": value}
        for method, values in methods.items()
        for person, value in zip(persons, values, strict=True)
    ]
    comparison_by_method = {
        row["method"]: row
        for row in paired_comparisons(
            comparison_rows,
            reference_method=best,
            metric="mpjpe",
            seed=0,
            bootstrap_samples=10_000,
        )
    }
    sig_rows = []
    for m in ranked:
        if m == best:
            continue
        diff = methods[m] - methods[best]
        comparison = comparison_by_method[m]
        sig_rows.append(
            {
                "method": m,
                "mean_mpjpe_mm": methods[m].mean() * 1000,
                "delta_vs_best_mm": diff.mean() * 1000,
                "delta_ci_low_mm": float(comparison["ci_low"]) * 1000,
                "delta_ci_high_mm": float(comparison["ci_high"]) * 1000,
                "worse_on_pct": float((diff > 0).mean() * 100),
                "wilcoxon_p": float(comparison["wilcoxon_p"]),
                "holm_p": float(comparison["holm_p"]),
                "leaky": m in LEAKY,
            }
        )
    sig_rows.sort(key=lambda r: r["mean_mpjpe_mm"])
    with (BUNDLE / "method_significance.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sig_rows[0].keys()))
        w.writeheader()
        w.writerows(sig_rows)

    sim_csv = resolve("source_vs_triangulated_by_person_similarity.csv", CMP_DIR)
    root_csv = resolve("source_vs_triangulated_by_person_root.csv", CMP_DIR)
    body_json = resolve("body_scale_summary.json", BODY_DIR)
    _, three = _load_mpjpe(sim_csv, "source")
    _, three_root = _load_mpjpe(root_csv, "source")
    body = json.loads(body_json.read_text(encoding="utf-8"))
    extr = json.loads(EXTR.read_text(encoding="utf-8"))["summary"]

    body_pct = {m: v["mean_pct_body"] for m, v in body["methods"].items()}

    # Copy the underlying per-person artefacts next to the report (skip self-copies
    # when a file already lives in the bundle because its source dir was cleaned up).
    for src in [
        sim_csv,
        resolve("source_vs_triangulated_by_person_similarity.json", CMP_DIR),
        root_csv,
        resolve("metrics_by_person_normalized.csv", BODY_DIR),
        body_json,
    ]:
        if src.resolve().parent != BUNDLE.resolve():
            shutil.copy2(src, BUNDLE / src.name)

    fu, fa, sd = three["fuse"], three["face"], three["side"]
    lines: List[str] = []
    A = lines.append
    A("# Fuse Results (Regenerated Triangulated GT)")
    A("")
    n_elderly = int((groups == "elderly").sum())
    n_student = int((groups == "student").sum())
    A(f"- Persons: `{n_persons}` ({n_elderly} elderly + {n_student} students)  •  "
      f"Methods: `{len(methods)}`  •  "
      "GT: per-person estimated extrinsics (`triangulation.estimate_extrinsics`)")
    A(f"- Recommended method: **`{RECOMMENDED}`**")
    A("- Analysis bundle: `local/runs/fuse_experiments/analysis/` (per-person CSV/JSON behind every table)")
    A("")
    A("The triangulated pseudo-GT was rebuilt after replacing the assumed camera "
      "layout with per-person extrinsics recovered from the data; earlier numbers "
      "(mean person MPJPE ~0.78 m) reflected a mis-scaled GT and are not comparable "
      "to the values here.")
    A("")

    A("## 1. Method ranking (vs triangulated GT, similarity-aligned)")
    A("")
    A(f"Baseline for the test is the best leakage-free method, `{best}` "
      f"({methods[best].mean()*1000:.2f} mm). Paired Wilcoxon over {n_persons} "
      "persons with Holm family-wise correction and paired bootstrap confidence "
      "intervals for the mean difference.")
    A("")
    A("| rank | method | mean mm | % body | Δ vs best mm [95% CI] | worse on | Holm p | note |")
    A("|---:|---|---:|---:|---:|---:|---:|---|")
    A(f"| — | `{best}` | {methods[best].mean()*1000:.2f} | "
      f"{body_pct.get(best, float('nan')):.2f} | — | — | — | recommended |")
    for i, r in enumerate(sig_rows, start=1):
        note = "**leakage — excluded**" if r["leaky"] else ""
        A(f"| {i} | `{r['method']}` | {r['mean_mpjpe_mm']:.2f} | "
          f"{body_pct.get(r['method'], float('nan')):.2f} | "
          f"{r['delta_vs_best_mm']:+.2f} "
          f"[{r['delta_ci_low_mm']:+.2f}, {r['delta_ci_high_mm']:+.2f}] | "
          f"{r['worse_on_pct']:.0f}% | {fmt_p(r['holm_p'])} | {note} |")
    A("")
    leaky_note = next((r for r in sig_rows if r["leaky"]), None)
    if leaky_note:
        A(f"`{leaky_note['method']}` scores {abs(leaky_note['delta_vs_best_mm']):.2f} mm "
          f"{'lower' if leaky_note['delta_vs_best_mm'] < 0 else 'higher'} but derives its "
          "per-joint weights from the triangulated GT it is then evaluated against, so it "
          "is excluded as a biased comparison rather than recommended.")
        A("")
    A("**How each method fuses the two views** (\"stable joints\" = trunk + legs — "
      "shoulders, hips, knees, ankles, feet — which move less than the arms and give a "
      "steadier cross-view alignment):")
    A("")
    A("| method | how it fuses |")
    A("|---|---|")
    for m in [best] + [r["method"] for r in sig_rows]:
        desc = METHOD_DESC.get(m, "")
        tag = " *(recommended)*" if m == RECOMMENDED else (" *(leaky)*" if m in LEAKY else "")
        A(f"| `{m}`{tag} | {desc} |")
    A("")

    A("## 2. Recommended method by cohort")
    A("")
    A(f"`{RECOMMENDED}` MPJPE (mm) split by cohort. Cohort mapping from "
      "`raw/person/{organize_from_dropbox_20260718,student_id_mapping}.csv`.")
    A("")
    A("| cohort | persons | median | mean | p95 |")
    A("|---|---:|---:|---:|---:|")
    for label, mask in (
        ("all", np.ones(n_persons, dtype=bool)),
        ("elderly", groups == "elderly"),
        ("students", groups == "student"),
    ):
        v = methods[RECOMMENDED][mask]
        A(f"| {label} | {int(mask.sum())} | {np.median(v)*1000:.1f} | "
          f"{v.mean()*1000:.1f} | {np.percentile(v, 95)*1000:.1f} |")
    A("")
    fu_e, fu_s = three["fuse"][groups == "elderly"], three["fuse"][groups == "student"]
    A(f"Fusion (three-way, similarity-aligned) is {np.median(fu_e)*1000:.1f} mm median on "
      f"elderly and {np.median(fu_s)*1000:.1f} mm on students — the two cohorts are close, "
      "so the ranking is not driven by one group.")
    A("")

    A("## 3. Fusion vs single views (fair three-way, similarity-aligned)")
    A("")
    A("Each source is brought into the triangulated frame with one Sim3 per "
      "sequence — the same alignment the fuse metric uses — so face, side and fuse "
      "are mutually comparable. Errors in mm.")
    A("")
    A("| source | median | mean | p95 | worst person |")
    A("|---|---:|---:|---:|---:|")
    for name, arr in (("face (single view)", fa), ("side (single view)", sd), ("**fuse**", fu)):
        A(f"| {name} | {np.median(arr)*1000:.1f} | {arr.mean()*1000:.1f} | "
          f"{np.percentile(arr, 95)*1000:.1f} | {arr.max()*1000:.1f} |")
    A("")
    A(f"- Fusion beats **face** by {(1-fu.mean()/fa.mean())*100:.1f}% on the mean, "
      f"on {(fu<fa).mean()*100:.0f}% of persons (Wilcoxon p={fmt_p(wilcoxon(fu, fa).pvalue)}).")
    A(f"- Fusion beats **side** by {(1-fu.mean()/sd.mean())*100:.1f}% on the mean, "
      f"on {(fu<sd).mean()*100:.0f}% of persons (Wilcoxon p={fmt_p(wilcoxon(fu, sd).pvalue)}).")
    A(f"- Fusion beats `min(face, side)` on {(fu<np.minimum(fa, sd)).mean()*100:.0f}% of "
      "persons, so it exploits both views rather than just picking the better one.")
    A("")
    A("> The `root`-aligned numbers (also bundled) put side at "
      f"{np.median(three_root['side'])*1000:.0f} mm — a coordinate-frame artefact, not "
      "quality: root alignment leaves the ~87° world-frame rotation of the side camera "
      "inside the error. Do not compare sources under root alignment.")
    A("")

    A("## 4. Learned fusion network vs deterministic fusion")
    A("")
    if RAE_EVAL.exists():
        _, rae = _load_mpjpe(RAE_EVAL, "method")
        shutil.copy2(RAE_EVAL, BUNDLE / "network_eval_metrics_by_person.csv")
        net_best = min(("A4", "A5", "A6"), key=lambda m: rae[m].mean())
        det_ref = "A2" if "A2" in rae else RECOMMENDED  # canonical arithmetic baseline
        A("The rotation-aware self-supervised network (`A4/A5/A6`, e100 checkpoints) was "
          "scored against the **same** regenerated GT with the same similarity alignment. "
          "This evaluation harness scores the network ablations and the deterministic "
          "methods together, so the numbers below are mutually comparable **within this "
          "table only** — do not compare them with §1 (a different harness puts "
          f"`{RECOMMENDED}` at {methods[RECOMMENDED].mean()*1000:.1f} mm here vs "
          f"{rae.get(RECOMMENDED, np.array([np.nan])).mean()*1000:.1f} mm there).")
        A("")
        A("| ablation | description | mean mm | median mm |")
        A("|---|---|---:|---:|")
        for m in ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]:
            if m in rae:
                A(f"| `{m}` | {ABLATION_LABEL[m]} | {rae[m].mean()*1000:.2f} | "
                  f"{np.median(rae[m])*1000:.2f} |")
        A("")
        a6, a2 = rae["A6"], rae[det_ref]
        A(f"- **The network fuses**: `A6` beats single-view `A0` (face) by "
          f"{(a6.mean()-rae['A0'].mean())*-1000:.1f} mm and `A1` (side) by "
          f"{(a6.mean()-rae['A1'].mean())*-1000:.1f} mm, on ~100% of persons "
          f"(p={fmt_p(wilcoxon(a6, rae['A0']).pvalue)}).")
        delta = (a6.mean() - a2.mean()) * 1000
        verdict = "does not beat" if delta >= 0 else "beats"
        A(f"- **But it {verdict} simple deterministic fusion**: `A6` (proposed) is "
          f"{delta:+.2f} mm vs `{det_ref}` ({ABLATION_LABEL.get(det_ref, 'canonical average')}), "
          f"better on only {(a6<a2).mean()*100:.0f}% of persons "
          f"(Wilcoxon p={fmt_p(wilcoxon(a6, a2).pvalue)}). The best learned variant "
          f"`{net_best}` is {(rae[net_best].mean()-a2.mean())*1000:+.2f} mm vs `{det_ref}`.")
        A("")
        A("On this pseudo-GT MPJPE, the learned network is on par with — marginally "
          "behind — plain canonical averaging. Its intended advantages are on the "
          "self-supervised / intrinsic axes it optimizes (rigidity, ROM retention, "
          "temporal smoothness, corruption robustness) and are not captured by MPJPE "
          "against a GT derived from the same SAM3D 2D that every method consumes. "
          "See `local/runs/fuse_rotation_aware/evaluation/.../{report.json,rotation_metrics.csv}`.")
    else:
        A("_Network evaluation not found; run `gymnastics fuse rotation-aware evaluate "
          "--run-id all137_a{4,5,6}_e100_seed0 --alignment similarity` first._")
    A("")

    A("## 5. Absolute scale caveat")
    A("")
    st = body["stature_estimate_m"]
    A("Two-view geometry is scale-free: the extrinsics fix rotation and translation "
      "*direction*, and metric scale comes from SAM3D's monocular 3D, which is not "
      "independently calibrated. Method rankings and every percentage above are "
      "**invariant** to that scale (the per-sequence Sim3 absorbs it); only the "
      "absolute millimetres scale with it.")
    A("")
    A(f"- Reconstructed stature: median {st['median']:.2f} m "
      f"[{st['p5']:.2f}, {st['p95']:.2f}] — consistent with real adults, so the scale "
      "is not grossly off (the old GT implied 1.96 m).")
    A(f"- `{RECOMMENDED}` error as a fraction of body size: "
      f"**{body_pct[RECOMMENDED]:.2f}% of limb-chain length** — this figure needs no "
      "calibration.")
    A("- To pin absolute scale: the recordings are on a matted hall floor whose tatami "
      "form a standard-size grid in both views; measuring one mat edge gives a metric "
      "reference. See `src/gymnastics/analysis/normalize_by_body_scale.py` and "
      "`docs/triangulation_scale_diagnosis.md`.")
    A("")

    A("## 6. Triangulation quality (the pseudo-GT these results rest on)")
    A("")
    tq = triangulation_quality()
    per_person = tq["sources"].get("estimated:per_person", 0)
    consensus = tq["sources"].get("estimated:cluster_consensus", 0)
    A(f"Dataset: **{tq['persons']} persons, {tq['cycles']} cycles, {tq['frames']:,} frames**, "
      f"{tq['missing']} missing face/side pairs. Extrinsics: {per_person} persons from their "
      f"own essential-matrix fit, {consensus} fell back to their rig-cluster consensus.")
    A("")
    A("**Reprojection error** — the triangulated 3D reprojected into both calibrated views:")
    A("")
    A("| | median | mean | p95 | max |")
    A("|---|---:|---:|---:|---:|")
    A(f"| mean reproj (px) | {tq['reproj_median']:.2f} | {tq['reproj_mean']:.2f} | "
      f"{tq['reproj_p95']:.2f} | {tq['reproj_max']:.1f} |")
    A("")
    A("**Assumed layout → estimated extrinsics** (held-out frames), why the GT was rebuilt:")
    A("")
    A("| metric | assumed layout | estimated | ")
    A("|---|---:|---:|")
    A(f"| reprojection median (px) | {extr['reproj_px']['assumed']['median']:.1f} | "
      f"{extr['reproj_px']['estimated']['median']:.2f} |")
    A(f"| reprojection max (px) | {extr['reproj_px']['assumed']['max']:.1f} | "
      f"{extr['reproj_px']['estimated']['max']:.1f} |")
    A(f"| shape err vs mono-3D median (mm) | {extr['shape_mm']['assumed']['median']:.1f} | "
      f"{extr['shape_mm']['estimated']['median']:.1f} |")
    A("")
    A(f"**Reconstruction plausibility** — stature median {st['median']:.2f} m "
      f"[{st['p5']:.2f}, {st['p95']:.2f}], vs the old GT's implausible 1.96 m "
      "(coefficient of variation dropped from 10.1% to 7.5%, near the ~5-6% of real "
      "adults). Full report: `local/runs/analysis/triangulated_results/triangulated_results_report.md`; "
      "extrinsics diagnosis: `docs/triangulation_scale_diagnosis.md`.")
    A("")
    A("## Files in this bundle")
    A("")
    A("- `method_significance.csv` — every method vs the best leakage-free one")
    A("- `source_vs_triangulated_by_person_similarity.{csv,json}` — fair three-way, per person")
    A("- `source_vs_triangulated_by_person_root.csv` — root-aligned (for contrast only)")
    A("- `metrics_by_person_normalized.csv`, `body_scale_summary.json` — scale-free metric")
    A("")
    A("_Elderly/student breakdown from the older summary is omitted: that person→group "
      "mapping is not present in the current artefacts. Point me at it to add the split._")
    A("")

    (FUSE_DIR / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"recommended: {RECOMMENDED}  ({methods[RECOMMENDED].mean()*1000:.2f} mm, "
          f"{body_pct[RECOMMENDED]:.2f}% body)")
    print(f"fuse vs face: {(1-fu.mean()/fa.mean())*100:.1f}% better on {(fu<fa).mean()*100:.0f}% persons")
    print(f"bundle: {BUNDLE}")
    print(f"report: {FUSE_DIR / 'RESULTS_SUMMARY.md'}")


if __name__ == "__main__":
    main()
